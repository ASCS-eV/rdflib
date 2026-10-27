"""
A term with ``"@container": "@list"`` and ``"@type": "@id"`` or ``"@vocab"``
fits a list of node references only.

IRI Compaction (JSON-LD 1.1 Processing Algorithms and API, section 6.2)
selects a term for a list by the most specific type that fits every member.
The serializer used such a term for any list, so literal members were written
as bare strings under an IRI-coercing term, and parsing turned them into
IRIs: ``"x"`` became ``<http://base.example/x>``, and ``2`` became
``<http://base.example/2>``. A list with a literal member now goes under a
list term without a type, under the plain term for the property, or under the
property IRI as a list object.
"""

import json

import pytest

from rdflib import RDF, BNode, Graph, Literal, Namespace
from rdflib.collection import Collection
from rdflib.compare import isomorphic

NS = Namespace("http://example.org/ns/")
BASE = "http://base.example/"
ID_LIST_CONTEXT = {
    "items": {"@id": str(NS.items), "@container": "@list", "@type": "@id"}
}
VOCAB_LIST_CONTEXT = {
    "@vocab": str(NS),
    "items": {"@id": str(NS.items), "@container": "@list", "@type": "@vocab"},
}
COERCING_CONTEXTS = [ID_LIST_CONTEXT, VOCAB_LIST_CONTEXT]
COERCING_IDS = ["id-list-term", "vocab-list-term"]
LITERALS = [Literal("x"), Literal(2)]
NODES = [NS.a, NS.b]


def _list_graph(members):
    g = Graph()
    head = BNode()
    Collection(g, head, members)
    g.add((NS.s, NS.items, head))
    return g


def _round_trip(g: Graph, context):
    data = g.serialize(format="json-ld", context=context)
    g2 = Graph()
    g2.parse(data=data, format="json-ld", base=BASE)
    return json.loads(data), g2


@pytest.mark.parametrize("context", COERCING_CONTEXTS, ids=COERCING_IDS)
@pytest.mark.parametrize(
    "members",
    [LITERALS, [NS.a, Literal("x")]],
    ids=["literals", "node-and-literal"],
)
def test_a_list_with_a_literal_member_round_trips(context, members):
    g = _list_graph(members)

    _doc, g2 = _round_trip(g, context)

    assert len(g2) == len(g)
    assert isomorphic(g, g2)


@pytest.mark.parametrize("context", COERCING_CONTEXTS, ids=COERCING_IDS)
def test_a_list_with_a_literal_member_is_a_list_object_under_the_iri(context):
    g = _list_graph(LITERALS)

    doc, _g2 = _round_trip(g, context)

    assert "items" not in doc
    assert doc[str(NS.items)] == {"@list": ["x", 2]}


def test_a_list_with_a_literal_member_uses_an_untyped_list_term():
    context = {**ID_LIST_CONTEXT, "list": {"@id": str(NS.items), "@container": "@list"}}
    g = _list_graph(LITERALS)

    doc, g2 = _round_trip(g, context)

    assert "items" not in doc
    assert doc["list"] == ["x", 2]
    assert isomorphic(g, g2)


def test_a_list_with_a_literal_member_uses_the_plain_term():
    context = {**ID_LIST_CONTEXT, "item": {"@id": str(NS.items)}}
    g = _list_graph(LITERALS)

    doc, g2 = _round_trip(g, context)

    assert "items" not in doc
    assert doc["item"] == {"@list": ["x", 2]}
    assert isomorphic(g, g2)


@pytest.mark.parametrize("context", COERCING_CONTEXTS, ids=COERCING_IDS)
def test_a_nested_list_with_a_literal_member_round_trips(context):
    """A nested list is a list object; under a coercing term its members
    would take the coercion."""
    g = Graph()
    inner, outer = BNode(), BNode()
    Collection(g, inner, LITERALS)
    Collection(g, outer, [inner])
    g.add((NS.s, NS.items, outer))

    doc, g2 = _round_trip(g, context)

    assert doc[str(NS.items)] == {"@list": [{"@list": ["x", 2]}]}
    assert isomorphic(g, g2)


@pytest.mark.parametrize(
    "plain_term",
    [{"@type": "@id"}, {"@type": "@vocab"}],
    ids=["id-typed", "vocab-typed"],
)
def test_a_list_with_a_literal_member_skips_a_coercing_plain_term(plain_term):
    """With no term that fits, the list goes under the IRI, not under a
    coercing term without ``@list``."""
    context = {
        "@vocab": str(NS),
        **ID_LIST_CONTEXT,
        "ref": {"@id": str(NS.items), **plain_term},
    }
    g = _list_graph(LITERALS)

    doc, g2 = _round_trip(g, context)

    assert "ref" not in doc
    assert doc[str(NS.items)] == {"@list": ["x", 2]}
    assert isomorphic(g, g2)


def test_a_list_under_a_vocab_typed_plain_term_is_a_reference():
    """Like under an ``@id``-typed term: the head is referenced and its chain
    written explicitly, instead of a list object whose members would take the
    coercion."""
    context = {"@vocab": str(NS), "ref": {"@id": str(NS.items), "@type": "@vocab"}}
    g = _list_graph(LITERALS)

    doc, g2 = _round_trip(g, context)

    (node,) = (n for n in doc["@graph"] if n["@id"] == str(NS.s))
    assert node["ref"].startswith("_:")
    assert isomorphic(g, g2)


@pytest.mark.parametrize("context", COERCING_CONTEXTS, ids=COERCING_IDS)
def test_a_list_of_node_references_still_uses_the_coercing_term(context):
    g = _list_graph(NODES)

    doc, g2 = _round_trip(g, context)

    assert len(doc["items"]) == 2
    assert all(isinstance(member, str) for member in doc["items"])
    assert isomorphic(g, g2)


@pytest.mark.parametrize("context", COERCING_CONTEXTS, ids=COERCING_IDS)
def test_blank_node_members_under_a_coercing_term_are_references(context):
    """A described member gets its own node object; a bare one gets none."""
    described, bare = BNode(), BNode()
    g = _list_graph([described, bare])
    g.add((described, NS.p, Literal("x")))

    doc, g2 = _round_trip(g, context)

    (node,) = (n for n in doc["@graph"] if n["@id"] == str(NS.s))
    assert [member.startswith("_:") for member in node["items"]] == [True, True]
    assert sorted(n["@id"] for n in doc["@graph"]) == sorted(
        [str(NS.s), described.n3()]
    )
    assert isomorphic(g, g2)


@pytest.mark.parametrize("context", COERCING_CONTEXTS, ids=COERCING_IDS)
def test_an_empty_list_still_round_trips(context):
    g = Graph()
    g.add((NS.s, NS.items, RDF.nil))

    _doc, g2 = _round_trip(g, context)

    assert isomorphic(g, g2)
