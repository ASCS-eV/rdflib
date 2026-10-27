"""
A type mapping of ``@id`` or ``@vocab`` applies to strings only, and ``@none``
applies no type at all (Value Expansion, JSON-LD 1.1 Processing Algorithms and
API, section 5.3.2). A number or boolean under such a term is a value object
without ``@type``, so it keeps the datatype its JSON type implies.

The parser copied the keyword into ``@type`` instead. Under ``@vocab`` the
keyword expanded to the empty string, so ``2`` became a literal with the empty
IRI as datatype: equal to neither ``2`` nor ``"2"``, and written by N-Triples
as the plain string ``"2"``.
"""

import json

import pytest

from rdflib import RDF, XSD, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection

NS = Namespace("http://example.org/ns/")
KEYWORD_TYPES = ["@id", "@vocab", "@none"]
NATIVE_VALUES = [
    pytest.param(2, Literal(2), id="integer"),
    pytest.param(2.5, Literal(2.5), id="double"),
    pytest.param(True, Literal(True), id="true"),
    pytest.param(False, Literal(False), id="false"),
]


def _parse(context: dict, value) -> Graph:
    # With a vocabulary, the keyword copied into @type expanded to an IRI
    # under @none as well; without one it expanded to None and hid the bug.
    doc = {"@context": {"@vocab": str(NS), **context}, "@id": str(NS.s), "p": value}
    g = Graph()
    g.parse(data=json.dumps(doc), format="json-ld")
    return g


def _term(**mapping):
    return {"@id": str(NS.p), **mapping}


@pytest.mark.parametrize("keyword", KEYWORD_TYPES)
@pytest.mark.parametrize(("value", "literal"), NATIVE_VALUES)
def test_a_native_value_under_a_keyword_typed_term_keeps_its_datatype(
    keyword, value, literal
):
    g = _parse({"p": _term(**{"@type": keyword})}, value)

    (parsed,) = g.objects(NS.s, NS.p)
    assert parsed == literal
    assert parsed.datatype == literal.datatype


@pytest.mark.parametrize("keyword", KEYWORD_TYPES)
@pytest.mark.parametrize(("value", "literal"), NATIVE_VALUES)
def test_a_native_list_member_under_a_keyword_typed_term_keeps_its_datatype(
    keyword, value, literal
):
    g = _parse({"p": _term(**{"@type": keyword, "@container": "@list"})}, [value])

    (head,) = g.objects(NS.s, NS.p)
    (parsed,) = Collection(g, head)
    assert parsed == literal
    assert parsed.datatype == literal.datatype


@pytest.mark.parametrize("keyword", KEYWORD_TYPES)
def test_a_native_value_under_a_keyword_typed_term_takes_no_language(keyword):
    """Section 5.3.2 adds a language to strings only."""
    g = _parse({"@language": "en", "p": _term(**{"@type": keyword})}, 2)

    (parsed,) = g.objects(NS.s, NS.p)
    assert parsed == Literal(2)


def test_a_string_under_a_none_typed_term_takes_the_default_language():
    """``@none`` leaves the value untyped, so the language applies to it."""
    g = _parse({"@language": "en", "p": _term(**{"@type": "@none"})}, "x")

    (parsed,) = g.objects(NS.s, NS.p)
    assert parsed == Literal("x", lang="en")


@pytest.mark.parametrize("term_language", ["de", None], ids=["de", "null"])
@pytest.mark.parametrize("default_language", [None, "fr"], ids=["no-default", "fr"])
def test_a_typed_term_has_no_language_mapping(term_language, default_language):
    """Create Term Definition (section 4.2.2, step 22) sets a language mapping
    only for a term definition without ``@type``, so the default language
    applies, if there is one."""
    context = {"p": _term(**{"@type": "@none", "@language": term_language})}
    if default_language:
        context["@language"] = default_language
    g = _parse(context, "x")

    (parsed,) = g.objects(NS.s, NS.p)
    assert parsed == Literal("x", lang=default_language)


@pytest.mark.parametrize("term_language", ["de", None], ids=["de", "null"])
@pytest.mark.parametrize("default_language", [None, "fr"], ids=["no-default", "fr"])
def test_a_null_typed_term_keeps_its_language_mapping(term_language, default_language):
    """A null ``@type`` removes the type mapping, so the term is untyped and
    its own language mapping applies."""
    context = {"p": _term(**{"@type": None, "@language": term_language})}
    if default_language:
        context["@language"] = default_language
    g = _parse(context, "x")

    (parsed,) = g.objects(NS.s, NS.p)
    assert parsed == Literal("x", lang=term_language)


@pytest.mark.parametrize(
    ("keyword", "iri"),
    [("@id", URIRef("http://base.example/x")), ("@vocab", NS.x)],
)
def test_a_string_under_an_iri_typed_term_is_still_an_iri(keyword, iri):
    doc = {
        "@context": {"@vocab": str(NS), "p": _term(**{"@type": keyword})},
        "@id": str(NS.s),
        "p": "x",
    }
    g = Graph()
    g.parse(data=json.dumps(doc), format="json-ld", base="http://base.example/")

    assert list(g.objects(NS.s, NS.p)) == [iri]


def test_a_native_value_under_a_datatype_typed_term_is_still_typed():
    g = _parse({"p": _term(**{"@type": str(XSD.decimal)})}, 2)

    assert list(g.objects(NS.s, NS.p)) == [Literal("2", datatype=XSD.decimal)]


@pytest.mark.parametrize("keyword", KEYWORD_TYPES)
def test_a_null_under_a_keyword_typed_term_is_dropped(keyword):
    g = _parse({"p": _term(**{"@type": keyword})}, None)

    assert len(g) == 0


def test_a_null_under_a_json_typed_term_is_still_a_json_literal():
    g = _parse({"@version": 1.1, "p": _term(**{"@type": "@json"})}, None)

    assert list(g.objects(NS.s, NS.p)) == [Literal("null", datatype=RDF.JSON)]
