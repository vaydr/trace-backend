import itertools
import threading

import pytest
import torch
from torch import nn

from trace_backend.ml.encoder import TextEncoder
from trace_backend.ml.features import make_features
from trace_backend.ml.model import RetrieverModel
from trace_backend.ml.training import evaluate_model
from trace_backend.ranking import RANKING_RULE
from trace_backend.retrieval import SubgraphRetriever
from trace_backend.schema import Triple


@pytest.fixture
def encoder():
    return TextEncoder({"kind": "hash", "dimension": 8}, "cpu")


def serving_retriever(model, encoder):
    # Exercise the serving method without downloading or training a checkpoint.
    retriever = object.__new__(SubgraphRetriever)
    retriever.model = model.eval()
    retriever.encoder = encoder
    retriever.device = torch.device("cpu")
    retriever.lock = threading.Lock()
    return retriever


def evaluate_candidates(tmp_path, model, encoder, candidates, gold, k):
    sample = {
        "id": "ranking-probe",
        "features": make_features("Which tail?", ["topic"], candidates, encoder),
        "targets": torch.tensor([float(t.id == gold.id) for t in candidates]),
        "answers": [gold.object],
        "triple_ids": [t.id for t in candidates],
        "subjects": [t.subject for t in candidates],
        "objects": [t.object for t in candidates],
    }
    torch.save(sample, tmp_path / "sample.pt")
    return evaluate_model(
        model, tmp_path, [{"file": "sample.pt"}], torch.device("cpu"), k, batch_size=1
    )


@pytest.mark.parametrize("k", [1, 2])
def test_symmetric_opaque_tails_have_same_evaluation_and_serving_order(tmp_path, encoder, k):
    candidates = [Triple(subject="topic", predicate="p", object=tail) for tail in ("m.a", "m.b")]
    expected = sorted(candidates, key=lambda triple: triple.id)
    # This fixture also distinguishes fact-ID ordering from lexical tuple ordering.
    assert expected != sorted(candidates, key=lambda triple: triple.key)
    torch.manual_seed(42)
    model = RetrieverModel(8).eval()
    retriever = serving_retriever(model, encoder)
    for permutation in itertools.permutations(candidates):
        ordered = list(permutation)
        features = make_features("Which tail?", ["topic"], ordered, encoder)
        assert features["num_non_text_entities"] == 2
        with torch.inference_mode():
            logits = model(features, batch_size=1)
        # Equal structural positions and a shared opaque-entity vector yield
        # genuine model ties, without forcing any model parameters to zero.
        assert logits[0].item() == logits[1].item()
        evaluation = evaluate_candidates(tmp_path, model, encoder, ordered, expected[0], k)
        served = retriever.retrieve("Which tail?", ["topic"], ordered, k)
        expected_ids = [triple.id for triple in expected[:k]]
        assert evaluation["predictions"][0]["triple_ids"] == expected_ids
        assert [triple.id for triple in served] == expected_ids
        assert evaluation["predictions"][0]["scores"] == [triple.score for triple in served]
        assert evaluation["triple_recall"] == 1.0
        assert evaluation["answer_recall"] == 1.0
        assert evaluation["answer_hit"] == 1.0
        assert evaluation["ranking"] == RANKING_RULE


class TailLogits(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def forward(self, features, batch_size=4096):
        return torch.tensor(
            [self.values[features["entities"][i]] for i in features["tails"].tolist()],
            dtype=torch.float32,
        )


@pytest.mark.parametrize("lower,higher", [(100.0, 101.0), (-101.0, -100.0)])
@pytest.mark.parametrize("k", [1, 2])
def test_saturated_sigmoid_does_not_change_raw_logit_ranking(tmp_path, encoder, lower, higher, k):
    candidates = sorted(
        [Triple(subject="topic", predicate="p", object=tail) for tail in ("m.a", "m.b")],
        key=lambda triple: triple.id,
    )
    # The larger fact ID has the larger logit, so treating rounded sigmoid
    # scores as a tie would select the wrong fact.
    expected = list(reversed(candidates))
    values = {candidates[0].object: lower, candidates[1].object: higher}
    model = TailLogits(values)
    retriever = serving_retriever(model, encoder)
    rounded = torch.sigmoid(torch.tensor([lower, higher], dtype=torch.float32))
    assert rounded[0].item() == rounded[1].item()
    for permutation in itertools.permutations(candidates):
        ordered = list(permutation)
        evaluation = evaluate_candidates(tmp_path, model, encoder, ordered, expected[0], k)
        served = retriever.retrieve("Which tail?", ["topic"], ordered, k)
        expected_ids = [triple.id for triple in expected[:k]]
        assert evaluation["predictions"][0]["triple_ids"] == expected_ids
        assert [triple.id for triple in served] == expected_ids
        assert evaluation["predictions"][0]["scores"] == [triple.score for triple in served]
        assert evaluation["triple_recall"] == 1.0
        assert evaluation["answer_recall"] == 1.0
        assert evaluation["answer_hit"] == 1.0
