import json
from contextlib import contextmanager

import pytest

from trace_backend.errors import TraceError
from trace_backend.graph import GraphStore
from trace_backend.schema import Example, Triple


@pytest.fixture
def store(tmp_path):
    graph = GraphStore(tmp_path / "graph.sqlite")
    graph.initialize()
    graph.add_triples(
        [
            Triple(subject="a", predicate="p", object="b"),
            Triple(subject="c", predicate="p", object="a"),
            Triple(subject="b", predicate="p", object="d"),
            Triple(subject="x", predicate="p", object="z"),
        ]
    )
    return graph


def test_exact_anchor_union_only(store):
    assert {t.key for t in store.incident(["a", "z"])} == {
        ("a", "p", "b"),
        ("c", "p", "a"),
        ("x", "p", "z"),
    }
    assert len(store.neighborhood("a", 2, 100)) == 3
    with pytest.raises(TraceError, match="exceeds"):
        store.incident(["a"], 1)


def test_full_graph_candidates_never_silently_truncate(store):
    triples = store.all_triples(4)
    assert len(triples) == 4
    assert [t.key for t in triples] == sorted(t.key for t in triples)
    with pytest.raises(TraceError) as error:
        store.all_triples(3)
    assert error.value.code == "candidate_limit"


def test_idempotent_import_and_label_ambiguity(store):
    store.add_triples(store.incident(["a"]))
    assert store.stats()["triples"] == 4
    with store.connection(write=True) as c:
        c.execute("UPDATE entities SET label='Name' WHERE id IN ('a','b')")
    with pytest.raises(TraceError, match="Multiple"):
        store.resolve("Name")
    assert store.resolve("a") == "a"
    assert store.resolve("missing") is None


def test_split_leakage_rolls_back(store):
    example = Example(
        id="q", question="Q?", topic_entities=["a"], answers=["b"], triples=[], split="train"
    )
    store.add_examples([example])
    with pytest.raises(ValueError, match="different splits"):
        store.add_examples([example.model_copy(update={"split": "test"})])
    assert store.question("q").split == "train"


def test_missing_graph_never_creates_an_empty_database(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(TraceError, match="Build"):
        GraphStore(path).stats()
    assert not path.exists()


def test_fingerprint_ignores_import_order_and_sqlite_layout(tmp_path):
    triples = [
        Triple(subject="a", predicate="p", object="b"),
        Triple(subject="c", predicate="r", object="a"),
    ]
    examples = [
        Example(
            id=name,
            question=name + "?",
            topic_entities=["a"],
            answers=["b"],
            triples=triples,
            split="test",
        )
        for name in ("first", "second")
    ]
    fingerprints = []
    for index, items in enumerate((examples, list(reversed(examples)))):
        graph = GraphStore(tmp_path / f"graph-{index}.sqlite")
        graph.initialize()
        graph.add_triples(triples if index == 0 else reversed(triples))
        graph.add_examples(items)
        if index == 1:
            with graph.connection(write=True) as c:
                for ex in examples:
                    # Equivalent JSON with different whitespace and key order.
                    payload = dict(reversed(list(ex.model_dump().items())))
                    c.execute(
                        "UPDATE questions SET payload=? WHERE id=?",
                        (json.dumps(payload, indent=2), ex.id),
                    )
        with graph.connection(write=True) as writer:
            writer.execute("UPDATE entities SET label='Entity A' WHERE id='a'")
            writer.commit()
            # Keep the writer open so the digest includes committed WAL contents.
            before_checkpoint = graph.fingerprint()
            writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            assert graph.fingerprint() == before_checkpoint
        fingerprints.append(before_checkpoint)
    assert fingerprints[0] == fingerprints[1]
    assert len(fingerprints[0]) == 64


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE triples SET predicate='new' WHERE subject='a'",
        "UPDATE entities SET label='New label' WHERE id='a'",
        "UPDATE entities SET urls='[\"https://example.org/a\"]' WHERE id='a'",
        "UPDATE questions SET payload=json_set(payload, '$.question', 'Changed?')",
        "UPDATE predicates SET label='New predicate label' WHERE id='p'",
        "UPDATE predicates SET label_source='llm' WHERE id='p'",
        "UPDATE metadata SET value='different_scope' WHERE key='scope'",
    ],
)
def test_fingerprint_tracks_all_relevant_content(store, statement):
    store.add_examples(
        [
            Example(
                id="q",
                question="Q?",
                topic_entities=["a"],
                answers=["b"],
                triples=[],
                split="test",
            )
        ]
    )
    store.save_predicate_label("p", "Predicate", "tokens")
    original = store.fingerprint()
    with store.connection(write=True) as c:
        c.execute(statement)
    assert store.fingerprint() != original


def test_fingerprint_uses_a_single_read_snapshot(store, monkeypatch):
    original = store.fingerprint()
    connection = store.connection
    wrote = False

    @contextmanager
    def connection_with_concurrent_write(write=False):
        nonlocal wrote
        with connection(write) as c:

            def after_first_table(statement):
                nonlocal wrote
                if "FROM triples ORDER BY" in statement and not wrote:
                    wrote = True
                    with connection(write=True) as writer:
                        writer.execute("UPDATE metadata SET value='new_scope' WHERE key='scope'")

            c.set_trace_callback(after_first_table)
            yield c

    monkeypatch.setattr(store, "connection", connection_with_concurrent_write)
    assert store.fingerprint() == original
    assert wrote
    assert store.fingerprint() != original
