import copy
import json

import pytest
import torch
import yaml

from trace_backend.data import write_examples
from trace_backend.ml.encoder import CachedEncoder, TextEncoder
from trace_backend.ml.features import make_features, weak_labels
from trace_backend.ml.model import RetrieverModel, mean_propagate
from trace_backend.ml.training import load_checkpoint, load_config, load_manifest, prepare, train
from trace_backend.schema import Example, Triple


@pytest.fixture
def config():
    return {
        "seed": 42,
        "device": "cpu",
        "encoder": {"kind": "hash", "dimension": 8, "batch_size": 2},
        "model": {"topic_pe": True, "forward_rounds": 2, "reverse_rounds": 2},
        "training": {
            "epochs": 2,
            "patience": 20,
            "learning_rate": 0.01,
            "eval_k": 1,
            "score_batch_size": 1,
            "gradient_clip": 1.0,
        },
    }


@pytest.fixture
def data(tmp_path):
    for split in ("train", "validation", "test"):
        triples = [
            Triple(subject="A", predicate="p", object="B"),
            Triple(subject="B", predicate="q", object="C"),
            Triple(subject="A", predicate="r", object="m.cvt"),
        ]
        write_examples(
            tmp_path / "data" / f"{split}.jsonl",
            [
                Example(
                    id=split,
                    question="Where? " + split,
                    topic_entities=["A"],
                    answers=["C"],
                    triples=triples,
                    split=split,
                )
            ],
        )
    return tmp_path / "data"


def test_mean_direction_and_duplicate_edges():
    values = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    heads, tails = torch.tensor([0, 2, 0]), torch.tensor([1, 1, 2])
    result = mean_propagate(values, heads, tails)
    torch.testing.assert_close(result, torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
    reverse = mean_propagate(values, tails, heads)
    torch.testing.assert_close(reverse[0], torch.tensor([0.5, 0.5]))


def test_all_shortest_weak_supervision_preserves_parallel_predicates():
    triples = [
        Triple(subject=s, predicate=p, object=o)
        for s, p, o in [
            ("a", "p", "b"),
            ("a", "q", "b"),
            ("b", "p", "d"),
            ("a", "p", "c"),
            ("c", "p", "d"),
            ("c", "irrelevant", "x"),
        ]
    ]
    assert weak_labels(triples, ["a"], ["d"]).tolist() == [1, 1, 1, 1, 1, 0]
    assert weak_labels(triples, ["d"], ["a"]).tolist() == [1, 1, 1, 1, 1, 0]
    assert not weak_labels(triples, ["missing"], ["d"]).any()


def test_features_and_chunked_gradients(tmp_path, config):
    encoder = CachedEncoder(TextEncoder(config["encoder"], "cpu"), tmp_path / "cache.db")
    triples = [
        Triple(subject="A", predicate="p", object="m.cvt"),
        Triple(subject="m.cvt", predicate="q", object="B"),
    ]
    f = make_features("Q?", ["A"], triples, encoder)
    assert f["num_non_text_entities"] == 1
    torch.manual_seed(42)
    a, b = RetrieverModel(8), RetrieverModel(8)
    b.load_state_dict(a.state_dict())
    a(f, 1).sum().backward()
    b(f, 100).sum().backward()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad)
    torch.testing.assert_close(encoder.encode(["A"]), encoder.encode(["A"]))


@pytest.mark.parametrize("explicit_seed", [True, False])
def test_training_checkpoint_reload_and_exact_resume(tmp_path, data, config, explicit_seed):
    if not explicit_seed:
        config.pop("seed")
    prepared = tmp_path / "prepared"
    manifest = prepare(data, prepared, config, tmp_path / "cache.db")
    assert manifest["splits"]["train"][0]["supervised"]
    report = train(prepared, tmp_path / "partial", config)
    assert report["epochs_run"] == 2
    resumed = copy.deepcopy(config)
    resumed["training"]["epochs"] = 4
    train(prepared, tmp_path / "resumed", resumed, tmp_path / "partial" / "last.pt")
    train(prepared, tmp_path / "full", resumed)
    model, checkpoint = load_checkpoint(tmp_path / "resumed" / "last.pt", "cpu")
    complete, _ = load_checkpoint(tmp_path / "full" / "last.pt", "cpu")
    assert checkpoint["epoch"] == 3
    assert checkpoint["config"]["seed"] == 42
    for left, right in zip(model.parameters(), complete.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert (tmp_path / "resumed" / "best.pt").is_file()
    history = [
        json.loads(x) for x in (tmp_path / "full" / "history.jsonl").read_text().splitlines()
    ]
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in history)


@pytest.mark.parametrize("interrupted_epoch", [0, 1])
@pytest.mark.parametrize("new_output", [False, True])
def test_resume_recovers_interrupted_best_save_without_another_epoch(
    tmp_path, data, config, monkeypatch, interrupted_epoch, new_output
):
    import trace_backend.ml.training as training

    prepared, output = tmp_path / "prepared", tmp_path / "model"
    config["training"]["epochs"] = interrupted_epoch + 1
    prepare(data, prepared, config, tmp_path / "cache.db")
    metrics = iter([0.25, 0.75])
    monkeypatch.setattr(training, "evaluate_model", lambda *args: {"triple_recall": next(metrics)})
    real_save = training.atomic_torch_save

    def fail_best_save(checkpoint, path):
        if path.name == "best.pt" and checkpoint["epoch"] == interrupted_epoch:
            raise OSError("interrupted best checkpoint save")
        real_save(checkpoint, path)

    monkeypatch.setattr(training, "atomic_torch_save", fail_best_save)
    with pytest.raises(OSError, match="interrupted best"):
        train(prepared, output, config)
    last = torch.load(output / "last.pt", weights_only=True)
    assert last["epoch"] == interrupted_epoch
    assert last["stale_epochs"] == 0
    source_best = (output / "best.pt").read_bytes() if interrupted_epoch else None
    monkeypatch.setattr(training, "atomic_torch_save", real_save)
    destination = tmp_path / "resumed" if new_output else output
    report = train(prepared, destination, config, output / "last.pt")
    assert report["epochs_run"] == 0
    recovered = torch.load(destination / "best.pt", weights_only=True)
    for key in ("run_id", "epoch", "best_metric", "stale_epochs", "data_fingerprint", "ranking"):
        assert recovered[key] == last[key]
    for key, tensor in last["model_state_dict"].items():
        torch.testing.assert_close(recovered["model_state_dict"][key], tensor, rtol=0, atol=0)
    if new_output:
        if source_best is None:
            assert not (output / "best.pt").exists()
        else:
            assert (output / "best.pt").read_bytes() == source_best


def test_resume_preserves_early_stop_and_allows_increased_patience(
    tmp_path, data, config, monkeypatch
):
    import trace_backend.ml.training as training

    prepared, output = tmp_path / "prepared", tmp_path / "model"
    prepare(data, prepared, config, tmp_path / "cache.db")
    config["training"].update(epochs=10, patience=1)
    monkeypatch.setattr(training, "evaluate_model", lambda *args: {"triple_recall": 0.5})
    assert train(prepared, output, config)["epochs_run"] == 2
    paths = [output / name for name in ("last.pt", "best.pt", "history.jsonl")]
    before = {path: path.read_bytes() for path in paths}
    assert train(prepared, output, config, output / "last.pt")["epochs_run"] == 0
    assert {path: path.read_bytes() for path in paths} == before
    config["training"]["patience"] = 2
    assert train(prepared, output, config, output / "last.pt")["epochs_run"] == 1
    last = torch.load(output / "last.pt", weights_only=True)
    assert last["epoch"] == 2
    assert last["stale_epochs"] == 2


def test_resume_cannot_reconstruct_missing_best_from_a_non_best_epoch(
    tmp_path, data, config, monkeypatch
):
    import trace_backend.ml.training as training

    prepared, output = tmp_path / "prepared", tmp_path / "model"
    prepare(data, prepared, config, tmp_path / "cache.db")
    monkeypatch.setattr(training, "evaluate_model", lambda *args: {"triple_recall": 0.5})
    train(prepared, output, config)
    (output / "best.pt").unlink()
    with pytest.raises(ValueError, match="requires the sibling best"):
        train(prepared, tmp_path / "resumed", config, output / "last.pt")
    assert not (tmp_path / "resumed").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "unrelated-run"),
        ("data_fingerprint", "different-data"),
        ("ranking", "different-ranking"),
        ("stale_epochs", 1),
        ("epoch", 2),
        ("best_metric", 0.75),
        ("best_metric", float("nan")),
    ],
)
def test_resume_does_not_repair_an_incompatible_best(
    tmp_path, data, config, monkeypatch, field, value
):
    import trace_backend.ml.training as training

    prepared, output = tmp_path / "prepared", tmp_path / "model"
    config["training"]["epochs"] = 1
    prepare(data, prepared, config, tmp_path / "cache.db")
    monkeypatch.setattr(training, "evaluate_model", lambda *args: {"triple_recall": 0.5})
    train(prepared, output, config)
    best = torch.load(output / "best.pt", weights_only=True)
    best[field] = value
    torch.save(best, output / "best.pt")
    before = (output / "best.pt").read_bytes()
    with pytest.raises(ValueError, match="best.pt does not belong"):
        train(prepared, tmp_path / "resumed", config, output / "last.pt")
    assert not (tmp_path / "resumed").exists()
    assert (output / "best.pt").read_bytes() == before


def test_no_supervision_fails_instead_of_saving_random_weights(tmp_path, data, config):
    prepared = tmp_path / "prepared"
    prepare(data, prepared, config, tmp_path / "cache.db")
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest["splits"]["train"][0]["supervised"] = False
    import hashlib

    without = {k: v for k, v in manifest.items() if k != "fingerprint"}
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(without, sort_keys=True).encode()
    ).hexdigest()
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="supervision"):
        train(prepared, tmp_path / "model", config)


def test_prepare_rerun_is_immutable_and_failure_preserves_old_manifest(
    tmp_path, data, config, monkeypatch
):
    import trace_backend.ml.training as training

    prepared = tmp_path / "prepared"
    original = prepare(data, prepared, config, tmp_path / "cache.db")
    original_bytes = (prepared / "manifest.json").read_bytes()
    rerun = prepare(data, prepared, config, tmp_path / "cache.db")
    assert rerun["fingerprint"] == original["fingerprint"]
    train_path = data / "train.jsonl"
    examples = [json.loads(line) for line in train_path.read_text().splitlines()]
    examples[0]["question"] = "Changed question"
    train_path.write_text("\n".join(json.dumps(ex) for ex in examples) + "\n")
    real_make_features = training.make_features

    def fail_on_validation(question, topics, triples, encoder):
        if question.endswith("validation"):
            raise ValueError("interrupted preparation")
        return real_make_features(question, topics, triples, encoder)

    monkeypatch.setattr(training, "make_features", fail_on_validation)
    with pytest.raises(ValueError, match="interrupted"):
        prepare(data, prepared, config, tmp_path / "cache.db")
    assert (prepared / "manifest.json").read_bytes() == original_bytes
    assert load_manifest(prepared)["fingerprint"] == original["fingerprint"]


def test_prepared_tensor_tampering_fails_before_training(tmp_path, data, config):
    prepared = tmp_path / "prepared"
    manifest = prepare(data, prepared, config, tmp_path / "cache.db")
    sample_path = prepared / manifest["splits"]["train"][0]["file"]
    sample = torch.load(sample_path, weights_only=True)
    sample["features"]["q_emb"].fill_(123)
    torch.save(sample, sample_path)
    with pytest.raises(ValueError, match="tensor checksum"):
        train(prepared, tmp_path / "model", config)
    assert not (tmp_path / "model").exists()


def test_training_refuses_different_requested_encoder(tmp_path, data, config):
    prepared = tmp_path / "prepared"
    prepare(data, prepared, config, tmp_path / "cache.db")
    incompatible = copy.deepcopy(config)
    incompatible["encoder"]["normalize"] = False
    with pytest.raises(ValueError, match="Requested encoder differs"):
        train(prepared, tmp_path / "model", incompatible)
    assert not (tmp_path / "model").exists()


def test_resume_refuses_existing_other_run_or_unrelated_best(tmp_path, data, config):
    prepared = tmp_path / "prepared"
    prepare(data, prepared, config, tmp_path / "cache.db")
    train(prepared, tmp_path / "first", config)
    train(prepared, tmp_path / "other", config)
    resume = tmp_path / "first" / "last.pt"
    with pytest.raises(ValueError, match="Output already contains"):
        train(prepared, tmp_path / "other", config, resume)
    # Even identically configured experiments must not exchange their best model.
    (tmp_path / "first" / "best.pt").write_bytes((tmp_path / "other" / "best.pt").read_bytes())
    with pytest.raises(ValueError, match="best.pt does not belong"):
        train(prepared, tmp_path / "new", config, resume)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("ranking", [None, "previous-ranking-rule"])
def test_resume_rejects_missing_or_different_ranking_but_weights_load(
    tmp_path, data, config, ranking
):
    from trace_backend.ranking import RANKING_RULE

    prepared = tmp_path / "prepared"
    prepare(data, prepared, config, tmp_path / "cache.db")
    train(prepared, tmp_path / "first", config)
    checkpoint = tmp_path / "first" / "last.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["ranking"] == RANKING_RULE
    if ranking is None:
        state.pop("ranking")
    else:
        state["ranking"] = ranking
    torch.save(state, checkpoint)
    # Historical weights remain usable for inference; their old selection
    # metrics must not silently become the basis for a resumed experiment.
    model, loaded = load_checkpoint(checkpoint, "cpu")
    assert isinstance(model, RetrieverModel)
    assert loaded.get("ranking") == ranking
    with pytest.raises(ValueError, match="ranking"):
        train(prepared, tmp_path / "resumed", config, checkpoint)
    assert not (tmp_path / "resumed").exists()


def test_resume_rejects_sibling_best_with_different_ranking(tmp_path, data, config):
    prepared = tmp_path / "prepared"
    prepare(data, prepared, config, tmp_path / "cache.db")
    train(prepared, tmp_path / "first", config)
    best = tmp_path / "first" / "best.pt"
    state = torch.load(best, map_location="cpu", weights_only=True)
    state["ranking"] = "previous-ranking-rule"
    torch.save(state, best)
    with pytest.raises(ValueError, match="best.pt does not belong"):
        train(prepared, tmp_path / "resumed", config, tmp_path / "first" / "last.pt")
    assert not (tmp_path / "resumed").exists()


@pytest.mark.parametrize("clipping", [None, "omitted"])
def test_unclipped_default_matches_bce_adam_step(tmp_path, data, config, clipping):
    config = copy.deepcopy(config)
    config["training"]["epochs"] = 1
    if clipping == "omitted":
        config["training"].pop("gradient_clip")
    else:
        config["training"]["gradient_clip"] = clipping
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    config = load_config(config_path)
    prepared = tmp_path / "prepared"
    manifest = prepare(data, prepared, config, tmp_path / "cache.db")
    sample = torch.load(prepared / manifest["splits"]["train"][0]["file"], weights_only=True)
    torch.manual_seed(config["seed"])
    reference = RetrieverModel(8, **config["model"])
    optimizer = torch.optim.Adam(reference.parameters(), lr=config["training"]["learning_rate"])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        reference(sample["features"]), sample["targets"]
    )
    loss.backward()
    optimizer.step()
    train(prepared, tmp_path / "model", config)
    actual, _ = load_checkpoint(tmp_path / "model" / "last.pt", "cpu")
    for expected, observed in zip(reference.parameters(), actual.parameters(), strict=True):
        torch.testing.assert_close(expected, observed)
