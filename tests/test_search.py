import random

import networkx as nx
import pytest

from trace_backend.schema import Triple
from trace_backend.search import clause_agreement, search_paths


def t(s, p, o):
    return Triple(subject=s, predicate=p, object=o)


def signature(path):
    return tuple((s.source, s.predicate, s.target, s.triple_id) for s in path.steps)


def test_hint_rank_does_not_prune_routes_and_prefix_is_weighted():
    triples = [t("a", "wrong", "b"), t("a", "p", "c"), t("c", "q", "b")]
    result = search_paths(triples, "a", "b", [["p", "q"]])
    assert len(result.paths) == 2
    assert result.paths[0].nodes == ["a", "c", "b"]
    assert result.paths[0].agreement == 2
    assert clause_agreement(["p", "q"], [["x"], ["p", "q"]]) == 1
    assert clause_agreement(["p", "x", "q"], [["p", "q"]]) == 1
    assert search_paths(triples, "a", "b", []).paths[0].hops == 1


def test_reverse_edges_and_parallel_predicates_keep_original_identity():
    triples = [t("b", "p", "a"), t("b", "q", "a"), t("a", "self", "a")]
    result = search_paths(triples, "a", "b", [], max_hops=3)
    assert len(result.paths) == 2
    assert all(p.steps[0].reversed for p in result.paths)
    assert {p.steps[0].triple_id for p in result.paths} == {t.id for t in triples[:2]}
    assert all(p.hops == 1 for p in result.paths)


def test_reciprocal_equal_predicate_triples_are_one_paper_route():
    forward, backward = t("a", "p", "b"), t("b", "p", "a")
    result = search_paths([forward, backward], "a", "b", [])
    assert len(result.paths) == 1
    assert result.paths[0].steps[0].triple_id == forward.id
    assert not result.paths[0].steps[0].reversed


@pytest.mark.parametrize("seed", range(10))
def test_repeated_predicates_against_independent_algorithm_one_reference(seed):
    rng = random.Random(seed)
    triples = [
        t(str(rng.randrange(5)), rng.choice(["p", "q"]), str(rng.randrange(5))) for _ in range(15)
    ]
    adjacency = {}
    for triple in triples:
        adjacency.setdefault(triple.subject, set()).add((triple.predicate, triple.object))
        adjacency.setdefault(triple.object, set()).add((triple.predicate, triple.subject))
    expected = set()

    def visit(nodes, predicates):
        if nodes[-1] == "4":
            expected.add((tuple(nodes), tuple(predicates)))
        elif len(predicates) < 4:
            for predicate, neighbor in adjacency.get(nodes[-1], set()):
                if neighbor not in nodes:
                    visit(nodes + [neighbor], predicates + [predicate])

    visit(["0"], [])
    result = search_paths(triples, "0", "4", [], max_hops=4)
    assert result.complete
    actual = {(tuple(p.nodes), tuple(s.predicate for s in p.steps)) for p in result.paths}
    assert actual == expected
    assert len(result.paths) == len(expected)


@pytest.mark.parametrize("seed", range(25))
def test_enumeration_against_independent_multigraph_search(seed):
    rng = random.Random(seed)
    triples = [
        t(str(a), f"p{i}", str(b))
        for i, (a, b) in enumerate((rng.randrange(6), rng.randrange(6)) for _ in range(12))
    ]
    graph = nx.MultiGraph()
    for edge in triples:
        graph.add_edge(edge.subject, edge.object, key=edge.id, predicate=edge.predicate)
    graph.add_nodes_from(["0", "5"])
    expected = set()
    for path in nx.all_simple_edge_paths(graph, "0", "5", cutoff=4):
        expected.add(tuple((a, graph[a][b][key]["predicate"], b, key) for a, b, key in path))
    actual = search_paths(triples, "0", "5", [["p0", "p1"]], max_hops=4)
    assert actual.complete
    assert {signature(p) for p in actual.paths} == expected


def test_limits_and_missing_targets():
    triples = [t("a", "p", "b"), t("b", "p", "c")]
    result = search_paths(triples, "a", "c", [], max_states=1)
    assert not result.complete
    assert result.reason == "state_limit"
    assert not result.paths
    assert search_paths(triples, "a", "c", [], max_hops=1).complete
    assert not search_paths(triples, "a", "c", [], max_hops=1).paths
    assert search_paths(triples, "a", "absent", []).complete
    assert search_paths(triples, "a", "a", []).paths[0].hops == 0
