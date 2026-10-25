from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ..data import file_sha256, read_examples
from ..ranking import RANKING_RULE, rank_triple_indices
from .encoder import CachedEncoder, TextEncoder, select_device
from .features import make_features, move_features, weak_labels
from .model import RetrieverModel


def atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("wb") as file:
            torch.save(value, file)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    for section in ("encoder", "model", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing {section} configuration")
    train = config["training"]
    for key in ("epochs", "patience", "eval_k", "score_batch_size"):
        if not isinstance(train.get(key), int) or train[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if train.get("learning_rate", 0) <= 0:
        raise ValueError("learning_rate must be positive")
    if train.get("gradient_clip") is not None and train["gradient_clip"] <= 0:
        raise ValueError("gradient_clip must be null or positive")
    return config


def _encoder_signature(config: dict) -> dict:
    """Embedding semantics, independent of batch size."""
    result = {k: v for k, v in config.items() if k != "batch_size"}
    result.setdefault("kind", "gte")
    result.setdefault("normalize", True)
    result.setdefault("max_length", 8192)
    if result["kind"] == "hash":
        result.setdefault("dimension", 32)
    return result


def _save_prepared_sample(record: dict, directory: Path, sample_id: str) -> tuple[str, str]:
    """Store tensors by content hash to preserve existing manifests."""
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        try:
            torch.save(record, file)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        digest = file_sha256(temporary)
        name = hashlib.sha256(sample_id.encode()).hexdigest() + "." + digest + ".pt"
        temporary.replace(directory / name)
        return name, digest
    finally:
        temporary.unlink(missing_ok=True)


def prepare(data_dir: Path, output: Path, config: dict, cache: Path) -> dict:
    encoder = CachedEncoder(TextEncoder(config["encoder"], config.get("device", "auto")), cache)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 2,
        "encoder_requested": copy.deepcopy(config["encoder"]),
        "encoder": encoder.config,
        "encoder_identity": encoder.identity,
        "dimension": encoder.dimension,
        "splits": {},
        "sources": {},
    }
    all_ids = set()
    for split in ("train", "validation", "test"):
        source = data_dir / f"{split}.jsonl"
        records = []
        for ex in read_examples(source):
            if ex.split != split:
                raise ValueError(f"Split mismatch for {ex.id}")
            if ex.id in all_ids:
                raise ValueError(f"Split leakage: {ex.id} occurs in multiple splits")
            all_ids.add(ex.id)
            features = make_features(ex.question, ex.topic_entities, ex.triples, encoder)
            labels = weak_labels(ex.triples, ex.topic_entities, ex.answers)
            record = {
                "id": ex.id,
                "features": features,
                "targets": labels,
                "answers": ex.answers,
                "triple_ids": [t.id for t in ex.triples],
                "subjects": [t.subject for t in ex.triples],
                "objects": [t.object for t in ex.triples],
            }
            name, digest = _save_prepared_sample(record, output / split, ex.id)
            records.append(
                {
                    "id": ex.id,
                    "file": f"{split}/{name}",
                    "sha256": digest,
                    "supervised": bool(labels.any()),
                    "triples": len(ex.triples),
                }
            )
        if not records:
            raise ValueError(f"Empty split: {split}")
        manifest["splits"][split] = records
        manifest["sources"][split] = file_sha256(source)
    payload = json.dumps(manifest, sort_keys=True)
    manifest["fingerprint"] = hashlib.sha256(payload.encode()).hexdigest()
    with tempfile.NamedTemporaryFile(mode="w", dir=output, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(json.dumps(manifest, indent=2) + "\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(output / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def load_manifest(prepared: Path) -> dict:
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest.get("format_version") != 2:
        raise ValueError("Unsupported prepared-data version; run trace prepare again")
    claimed = manifest["fingerprint"]
    check = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if hashlib.sha256(json.dumps(check, sort_keys=True).encode()).hexdigest() != claimed:
        raise ValueError("Prepared manifest fingerprint mismatch")
    for split in ("train", "validation", "test"):
        for record in manifest["splits"][split]:
            path = prepared / record["file"]
            if not path.is_file() or file_sha256(path) != record.get("sha256"):
                raise ValueError(
                    f"Prepared tensor checksum mismatch: {record['id']}; run prepare again"
                )
    return manifest


@torch.inference_mode()
def evaluate_model(
    model: RetrieverModel, prepared: Path, records: list[dict], device, k: int, batch_size: int
) -> dict:
    model.eval()
    triple_recall, answer_recall, answer_hit = [], [], []
    predictions = []
    for record in records:
        sample = torch.load(prepared / record["file"], map_location="cpu", weights_only=True)
        features = move_features(sample["features"], device)
        logits = model(features, batch_size).cpu()
        chosen = rank_triple_indices(logits.tolist(), sample["triple_ids"], k)
        positives = sample["targets"].sum().item()
        if positives:
            triple_recall.append(sample["targets"][chosen].sum().item() / positives)
        nodes = {sample["subjects"][i] for i in chosen} | {sample["objects"][i] for i in chosen}
        gold = set(sample["answers"])
        if gold:
            answer_recall.append(len(nodes & gold) / len(gold))
            answer_hit.append(float(bool(nodes & gold)))
        predictions.append(
            {
                "id": sample["id"],
                "triple_ids": [sample["triple_ids"][i] for i in chosen],
                "scores": torch.sigmoid(logits[chosen]).tolist(),
            }
        )
    return {
        "examples": len(records),
        "supervised_examples": len(triple_recall),
        "triple_recall": float(np.mean(triple_recall)) if triple_recall else None,
        "answer_recall": float(np.mean(answer_recall)) if answer_recall else None,
        "answer_hit": float(np.mean(answer_hit)) if answer_hit else None,
        "k": k,
        "ranking": RANKING_RULE,
        "predictions": predictions,
    }


def train(prepared: Path, output: Path, config: dict, resume: Path | None = None) -> dict:
    contains_run = any((output / name).exists() for name in ("best.pt", "last.pt", "history.jsonl"))
    if contains_run and (resume is None or output.resolve() != resume.parent.resolve()):
        raise ValueError(
            "Output already contains a training run; resume its own checkpoint or choose a new output directory"
        )
    manifest = load_manifest(prepared)
    if _encoder_signature(config["encoder"]) != _encoder_signature(manifest["encoder_requested"]):
        raise ValueError(
            "Requested encoder differs from prepared embeddings; run prepare with the same config"
        )
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = select_device(config.get("device", "auto"))
    model = RetrieverModel(manifest["dimension"], **config["model"]).to(device)
    training = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    records = [r for r in manifest["splits"]["train"] if r["supervised"] and r["triples"]]
    val = manifest["splits"]["validation"]
    if not records or not any(r["supervised"] for r in val):
        raise ValueError(
            "Training and validation require at least one nonempty shortest-path supervision example"
        )
    if {r["id"] for r in records} & {r["id"] for r in val}:
        raise ValueError("Training/validation IDs overlap")
    config = copy.deepcopy(config)
    config["seed"] = seed
    config["encoder"] = manifest["encoder"]
    start, best, stale = 0, -math.inf, 0
    run_id = uuid.uuid4().hex
    sibling_best = None
    recovered_best = None
    if resume:
        previous = torch.load(resume, map_location="cpu", weights_only=True)
        if previous.get("format_version") != 1 or not previous.get("run_id"):
            raise ValueError("Resume checkpoint lacks run provenance; start a new training run")
        if previous.get("ranking") != RANKING_RULE:
            raise ValueError("Resume checkpoint uses different ranking; start a new training run")
        if previous["data_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Resume checkpoint uses different data/embeddings")
        old_config = previous["config"]
        if old_config["model"] != config["model"] or old_config["encoder"] != config["encoder"]:
            raise ValueError("Resume model/encoder differs from checkpoint")
        if (
            old_config.get("seed", 42) != seed
            or old_config["training"]["learning_rate"] != training["learning_rate"]
            or old_config["training"].get("gradient_clip") != training.get("gradient_clip")
            or old_config["training"]["eval_k"] != training["eval_k"]
        ):
            raise ValueError(
                "Resume seed, learning rate, gradient clipping, or evaluation K differs from checkpoint"
            )
        sibling_best = resume.parent / "best.pt"
        if not sibling_best.exists():
            if previous["stale_epochs"] != 0:
                raise ValueError("Resume requires the sibling best.pt checkpoint")
            recovered_best = previous
        else:
            saved_best = torch.load(sibling_best, map_location="cpu", weights_only=True)
            saved_epoch, saved_metric = saved_best.get("epoch"), saved_best.get("best_metric")
            if (
                saved_best.get("format_version") != previous["format_version"]
                or saved_best.get("run_id") != previous["run_id"]
                or saved_best.get("ranking") != RANKING_RULE
                or saved_best.get("data_fingerprint") != previous["data_fingerprint"]
                or saved_best.get("stale_epochs") != 0
                or not isinstance(saved_epoch, int)
                or not 0 <= saved_epoch <= previous["epoch"]
                or not isinstance(saved_metric, (int, float))
                or not math.isfinite(saved_metric)
            ):
                raise ValueError("Resume best.pt does not belong to the checkpoint's training run")
            if saved_metric != previous["best_metric"]:
                if (
                    previous["stale_epochs"] != 0
                    or saved_epoch >= previous["epoch"]
                    or saved_metric >= previous["best_metric"]
                ):
                    raise ValueError(
                        "Resume best.pt does not belong to the checkpoint's training run"
                    )
                recovered_best = previous
        run_id = previous["run_id"]
        model.load_state_dict(previous["model_state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        torch.set_rng_state(previous["rng_state"])
        start, best, stale = (
            previous["epoch"] + 1,
            previous["best_metric"],
            previous["stale_epochs"],
        )
    output.mkdir(parents=True, exist_ok=True)
    if recovered_best is not None:
        # last.pt commits first. An improving checkpoint contains the exact
        # best state, so it can repair an interrupted subsequent best.pt save.
        atomic_torch_save(recovered_best, output / "best.pt")
    elif resume and not (output / "best.pt").exists():
        # Retain the best checkpoint from epochs preceding the resume point.
        shutil.copyfile(sibling_best, output / "best.pt")
    history = []
    for epoch in range(start, training["epochs"]):
        if stale >= training["patience"]:
            break
        model.train()
        order = list(records)
        random.Random(seed + epoch).shuffle(order)
        total_loss = 0.0
        for record in order:
            sample = torch.load(prepared / record["file"], map_location="cpu", weights_only=True)
            features = move_features(sample["features"], device)
            targets = sample["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            nodes = model.node_features(features)
            n = len(targets)
            for begin in range(0, n, training["score_batch_size"]):
                end = min(begin + training["score_batch_size"], n)
                logits = model.score_chunk(features, nodes, begin, end)
                loss = (
                    F.binary_cross_entropy_with_logits(logits, targets[begin:end], reduction="sum")
                    / n
                )
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss during training")
                loss.backward(retain_graph=end < n)
                total_loss += loss.item()
            if training.get("gradient_clip") is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training["gradient_clip"], error_if_nonfinite=True
                )
            elif any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ValueError("Non-finite gradient during training")
            optimizer.step()
        metrics = evaluate_model(
            model, prepared, val, device, training["eval_k"], training["score_batch_size"]
        )
        score = metrics["triple_recall"]
        improved = score > best
        best = max(best, score)
        stale = 0 if improved else stale + 1
        checkpoint = {
            "format_version": 1,
            "run_id": run_id,
            "ranking": RANKING_RULE,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "dimension": manifest["dimension"],
            "encoder": manifest["encoder"],
            "data_fingerprint": manifest["fingerprint"],
            "epoch": epoch,
            "best_metric": best,
            "stale_epochs": stale,
            "rng_state": torch.get_rng_state(),
            "training_examples": len(records),
            "skipped_train_examples": len(manifest["splits"]["train"]) - len(records),
        }
        atomic_torch_save(checkpoint, output / "last.pt")
        if improved:
            atomic_torch_save(checkpoint, output / "best.pt")
        log = {
            "epoch": epoch,
            "loss": total_loss / len(records),
            "validation": {k: v for k, v in metrics.items() if k != "predictions"},
            "improved": improved,
        }
        history.append(log)
        with (output / "history.jsonl").open("a") as f:
            f.write(json.dumps(log) + "\n")
        print(json.dumps(log), flush=True)
    report = {
        "epochs_run": len(history),
        "best_validation_triple_recall": best,
        "checkpoint": str(output / "best.pt"),
        "device": str(device),
        "training_examples": len(records),
        "data_fingerprint": manifest["fingerprint"],
        "encoder": manifest["encoder"],
        "ranking": RANKING_RULE,
    }
    (output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_checkpoint(path: Path, device: str = "auto"):
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}; run trace train first")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint. Use this repository's training workflow")
    model = RetrieverModel(checkpoint["dimension"], **checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(select_device(device)).eval(), checkpoint
