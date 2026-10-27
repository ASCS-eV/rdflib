"""
The JSON-LD serializer joins a further value for a property to the value it
has already written. Whether one is present is a question about the key, not
about the truth of the value: with an active context ``0``, ``0.0``, ``""``
and ``false`` are written as bare JSON values, and testing them for truth made
the next value for the same property replace them, so one triple was lost.
"""

import json

import pytest

from rdflib import Dataset, Graph, Literal, Namespace
from rdflib.compare import isomorphic

NS = Namespace("http://example.org/ns/")
VOCAB_CONTEXT = {"@vocab": str(NS)}
LANGUAGE_MAP_CONTEXT = {"p": {"@id": str(NS.p), "@container": "@language"}}
LANGUAGE_SET_CONTEXT = {"p": {"@id": str(NS.p), "@container": ["@language", "@set"]}}

FALSY_VALUES = [Literal(0), Literal(0.0), Literal(""), Literal(False)]
FALSY_IDS = ["0", "0.0", "empty-string", "false"]


def _graph(values):
    g = Graph()
    for value in values:
        g.add((NS.s, NS.p, value))
    return g


def _round_trip(g: Graph, context):
    data = g.serialize(format="json-ld", context=context)
    g2 = Graph()
    g2.parse(data=data, format="json-ld")
    return json.loads(data), g2


@pytest.mark.parametrize("falsy", FALSY_VALUES, ids=FALSY_IDS)
@pytest.mark.parametrize(
    "falsy_first", [True, False], ids=["falsy-first", "falsy-second"]
)
def test_a_falsy_value_beside_another_value_round_trips(falsy, falsy_first):
    """The falsy value was lost when the store yielded it first."""
    other = Literal("a")
    g = _graph([falsy, other] if falsy_first else [other, falsy])

    doc, g2 = _round_trip(g, VOCAB_CONTEXT)

    assert {str(v) for v in doc["p"]} == {str(falsy.toPython()), "a"}
    assert len(g2) == len(g)
    assert isomorphic(g, g2)


@pytest.mark.parametrize(
    "falsy_first", [True, False], ids=["falsy-first", "falsy-second"]
)
def test_an_empty_string_in_a_language_map_round_trips(falsy_first):
    values = [Literal("", lang="en"), Literal("a", lang="en")]
    g = _graph(values if falsy_first else values[::-1])

    doc, g2 = _round_trip(g, LANGUAGE_MAP_CONTEXT)

    assert sorted(doc["p"]["en"]) == ["", "a"]
    assert len(g2) == len(g)
    assert isomorphic(g, g2)


def test_a_language_set_starts_without_a_null():
    """The same test turned a missing entry into ``[None]`` before appending."""
    g = _graph([Literal("a", lang="en")])

    doc, g2 = _round_trip(g, LANGUAGE_SET_CONTEXT)

    assert doc["p"] == {"en": ["a"]}
    assert isomorphic(g, g2)


def test_a_graph_named_by_the_base_iri_keeps_its_name():
    """The graph name shrinks to ``""`` against the base, and was tested for truth."""
    ds = Dataset()
    ds.graph(NS.g).add((NS.s, NS.p, Literal("named")))
    ds.default_graph.add((NS.d, NS.p, Literal("default")))

    data = ds.serialize(format="json-ld", base=str(NS.g), context={"@base": str(NS.g)})
    ds2 = Dataset()
    ds2.parse(data=data, format="json-ld")

    objects = json.loads(data)["@graph"]
    assert any(obj.get("@id") == "" and "@graph" in obj for obj in objects)
    assert (NS.s, NS.p, Literal("named")) in ds2.graph(NS.g)
    assert (NS.d, NS.p, Literal("default")) in ds2.default_graph


def test_several_falsy_values_beside_another_value_round_trip():
    g = _graph([Literal(0), Literal(""), Literal("a")])

    doc, g2 = _round_trip(g, VOCAB_CONTEXT)

    assert sorted(doc["p"], key=str) == ["", 0, "a"]
    assert isomorphic(g, g2)


def test_a_reference_equal_to_the_base_beside_another_round_trips():
    """A coerced IRI that shrinks to ``""`` is a value like any other."""
    context = {"ref": {"@id": str(NS.p), "@type": "@id"}}
    g = _graph([NS.s, NS.o])

    data = g.serialize(format="json-ld", context=context, base=str(NS.s))
    g2 = Graph()
    g2.parse(data=data, format="json-ld", base=str(NS.s))

    assert sorted(json.loads(data)["ref"]) == ["", "/ns/o"]
    assert isomorphic(g, g2)


@pytest.mark.parametrize("falsy", FALSY_VALUES, ids=FALSY_IDS)
def test_a_falsy_value_without_a_context_round_trips(falsy):
    """Without a context every value is an object; this must stay so."""
    g = _graph([falsy, Literal("a")])

    _doc, g2 = _round_trip(g, None)

    assert isomorphic(g, g2)
