import pytest

from trace_backend.cli import humanize
from trace_backend.graph import GraphStore
from trace_backend.schema import Triple


class Humanizer:
    def __init__(self):
        self.batches = []

    async def humanize(self, predicates):
        self.batches.append(predicates)
        return {p: f"humanized {p}" for p in predicates}


@pytest.fixture
def store(tmp_path):
    graph = GraphStore(tmp_path / "graph.sqlite")
    graph.initialize()
    graph.add_triples(
        [
            Triple(subject="a", predicate="people.person.place_of_birth", object="b"),
            Triple(subject="b", predicate="location.location.containedby", object="c"),
        ]
    )
    return graph


async def test_live_humanization_upgrades_token_labels_and_then_reuses_cache(store):
    provider = Humanizer()
    token_result = await humanize(store, provider, 1, token_only=True)
    assert token_result == {
        "new_labels": 2,
        "upgraded_labels": 0,
        "processed_labels": 2,
        "label_source": "tokens",
    }
    assert provider.batches == []
    assert set(store.predicate_label_sources().values()) == {"tokens"}

    live_result = await humanize(store, provider, 1, token_only=False)
    assert live_result == {
        "new_labels": 0,
        "upgraded_labels": 2,
        "processed_labels": 2,
        "label_source": "llm",
    }
    assert len(provider.batches) == 2
    assert all(label.startswith("humanized ") for label in store.predicate_labels().values())
    assert set(store.predicate_label_sources().values()) == {"llm"}

    cached_result = await humanize(store, provider, 1, token_only=False)
    assert cached_result["processed_labels"] == 0
    assert cached_result["new_labels"] == cached_result["upgraded_labels"] == 0
    assert len(provider.batches) == 2


async def test_token_only_pass_preserves_live_labels_and_fills_only_missing_labels(store):
    provider = Humanizer()
    predicate = store.predicates()[0]
    store.save_predicate_label(predicate, "Existing LLM label", "llm")

    result = await humanize(store, provider, 32, token_only=True)
    assert result["new_labels"] == result["processed_labels"] == 1
    assert result["upgraded_labels"] == 0
    assert provider.batches == []
    assert store.predicate_labels()[predicate] == "Existing LLM label"
    assert store.predicate_label_sources()[predicate] == "llm"

    result = await humanize(store, provider, 32, token_only=False)
    assert result["new_labels"] == 0
    assert result["upgraded_labels"] == result["processed_labels"] == 1
    assert provider.batches == [[p for p in store.predicates() if p != predicate]]
    assert store.predicate_labels()[predicate] == "Existing LLM label"


async def test_live_humanization_counts_new_and_upgraded_labels_separately(store):
    provider = Humanizer()
    store.save_predicate_label(store.predicates()[0], "token fallback", "tokens")

    result = await humanize(store, provider, 32, token_only=False)
    assert result["new_labels"] == 1
    assert result["upgraded_labels"] == 1
    assert result["processed_labels"] == 2
    assert set(store.predicate_label_sources().values()) == {"llm"}
