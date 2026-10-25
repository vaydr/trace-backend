import hashlib
import json

import pytest

from trace_backend.cli import parser, run
from trace_backend.data import write_examples
from trace_backend.graph import GraphStore
from trace_backend.schema import Example, Triple


@pytest.mark.parametrize("configuration_source", ["environment", "dotenv"])
@pytest.mark.parametrize("explicit_database", [False, True])
def test_humanize_uses_configured_graph_unless_db_is_explicit(
    tmp_path, monkeypatch, capsys, configuration_source, explicit_database
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRACE_GRAPH_DB", raising=False)
    stores = {
        "default": GraphStore(tmp_path / "data" / "graph.sqlite"),
        "configured": GraphStore(tmp_path / "configured.sqlite"),
        "explicit": GraphStore(tmp_path / "explicit.sqlite"),
    }
    for name, store in stores.items():
        store.initialize()
        store.add_triples([Triple(subject="A", predicate=f"person.{name}_label", object="B")])

    if configuration_source == "environment":
        monkeypatch.setenv("TRACE_GRAPH_DB", str(stores["configured"].path))
    else:
        (tmp_path / ".env").write_text(f"TRACE_GRAPH_DB={stores['configured'].path}\n")

    arguments = ["humanize", "--token-only"]
    selected = "explicit" if explicit_database else "configured"
    if explicit_database:
        arguments.extend(["--db", str(stores["explicit"].path)])
    run(parser().parse_args(arguments))

    assert json.loads(capsys.readouterr().out)["processed_labels"] == 1
    for name, store in stores.items():
        expected = {f"person.{name}_label": f"{name} label"} if name == selected else {}
        assert store.predicate_labels() == expected


@pytest.mark.parametrize("replacement_phase", [None, "load", "evaluation"])
def test_evaluate_report_binds_metrics_to_checkpoint_data_and_configuration(
    tmp_path, monkeypatch, capsys, replacement_phase
):
    import torch

    from trace_backend.ml import training
    from trace_backend.ml.training import prepare, train

    config = {
        "seed": 42,
        "device": "cpu",
        "encoder": {"kind": "hash", "dimension": 4, "batch_size": 2},
        "model": {"topic_pe": True, "forward_rounds": 1, "reverse_rounds": 1},
        "training": {
            "epochs": 1,
            "patience": 1,
            "learning_rate": 0.001,
            "eval_k": 1,
            "score_batch_size": 2,
        },
    }
    data = tmp_path / "data"
    prepared = tmp_path / "prepared"
    model_directory = tmp_path / "model"
    for split in ("train", "validation", "test"):
        write_examples(
            data / f"{split}.jsonl",
            [
                Example(
                    id=f"cli-{split}",
                    question=f"Where is {split} subject?",
                    topic_entities=[f"{split} subject"],
                    answers=[f"{split} answer"],
                    triples=[
                        Triple(
                            subject=f"{split} subject",
                            predicate="location",
                            object=f"{split} answer",
                        ),
                        Triple(
                            subject=f"{split} subject", predicate="likes", object=f"{split} other"
                        ),
                    ],
                    split=split,
                )
            ],
        )
    manifest = prepare(data, prepared, config, tmp_path / "embeddings.sqlite")
    train(prepared, model_directory, config)
    checkpoint_path = model_directory / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if replacement_phase:
        function = "load_checkpoint" if replacement_phase == "load" else "evaluate_model"
        original = getattr(training, function)

        def replace_checkpoint(*args, **kwargs):
            result = original(*args, **kwargs)
            torch.save({**checkpoint, "epoch": 99}, checkpoint_path)
            return result

        monkeypatch.setattr(training, function, replace_checkpoint)
    output = tmp_path / "evaluation.json"
    capsys.readouterr()

    arguments = parser().parse_args(
        [
            "evaluate",
            "--prepared",
            str(prepared),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "validation",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--k",
            "1",
        ]
    )
    if replacement_phase == "load":
        with pytest.raises(ValueError, match="Checkpoint changed while loading"):
            run(arguments)
        assert not output.exists()
        return
    run(arguments)

    result = json.loads(output.read_text())
    assert result["examples"] == result["supervised_examples"] == result["k"] == 1
    assert {"triple_recall", "answer_recall", "answer_hit", "ranking"} <= result.keys()
    assert [prediction["id"] for prediction in result["predictions"]] == ["cli-validation"]
    assert result["provenance"] == {
        "checkpoint_sha256": checkpoint_sha256,
        "prepared_data_fingerprint": manifest["fingerprint"],
        "split": "validation",
        "encoder": checkpoint["encoder"],
        "configuration": checkpoint["config"],
        "device": "cpu",
        "checkpoint_epoch": 0,
        "checkpoint_run_id": checkpoint["run_id"],
    }
    assert json.loads(capsys.readouterr().out)["provenance"] == result["provenance"]
