from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS was requested but is unavailable")
    return device


class TextEncoder:
    def __init__(self, config: dict, device: str = "auto"):
        self.config = dict(config)
        self.kind = config.get("kind", "gte")
        self.device = select_device(device)
        self.batch_size = int(config.get("batch_size", 8))
        self.max_length = int(config.get("max_length", 8192))
        self.normalize = config.get("normalize", True)
        if self.batch_size < 1 or self.max_length < 1:
            raise ValueError("Encoder batch_size and max_length must be positive")
        if self.kind == "hash":
            self.dimension = int(config.get("dimension", 32))
            self.config.update(name="test-only-hash", dimension=self.dimension, revision="1")
        elif self.kind == "sentence_transformer":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                config["name"], revision=config.get("revision"), device=str(self.device)
            )
            self.model.max_seq_length = self.max_length
            dimension_method = getattr(
                self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension
            )
            self.dimension = dimension_method()
            self.config["revision"] = self.model[0].auto_model.config._commit_hash or config.get(
                "revision"
            )
        elif self.kind == "gte":
            from transformers import AutoModel, AutoTokenizer

            # Eager attention avoids mandatory CUDA/xformers/FlashAttention.
            self.tokenizer = AutoTokenizer.from_pretrained(
                config["name"], revision=config.get("revision")
            )
            self.model = (
                AutoModel.from_pretrained(
                    config["name"],
                    revision=config.get("revision"),
                    code_revision=config.get("code_revision"),
                    trust_remote_code=True,
                    unpad_inputs=False,
                    use_memory_efficient_attention=False,
                )
                .to(self.device)
                .eval()
            )
            self.dimension = self.model.config.hidden_size
            self.config["revision"] = self.model.config._commit_hash or config.get("revision")
        else:
            raise ValueError(f"Unknown encoder kind: {self.kind}")
        self.config["dimension"] = self.dimension
        self.identity = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.dimension), dtype=torch.float32)
        if self.kind == "hash":
            # Synthetic embeddings for tests only.
            values = []
            for text in texts:
                seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
                values.append(
                    np.random.default_rng(seed).normal(size=self.dimension).astype(np.float32)
                )
            out = torch.from_numpy(np.stack(values))
            return F.normalize(out, dim=1) if self.normalize else out
        if self.kind == "sentence_transformer":
            return (
                self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                .detach()
                .float()
                .cpu()
            )
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start : start + self.batch_size],
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            emb = self.model(**batch).last_hidden_state[:, 0].float()
            chunks.append((F.normalize(emb, dim=1) if self.normalize else emb).cpu())
        return torch.cat(chunks)


class CachedEncoder:
    def __init__(self, encoder: TextEncoder, path: Path):
        self.encoder = encoder
        self.dimension = encoder.dimension
        self.config = encoder.config
        self.identity = encoder.identity
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute(
                "CREATE TABLE IF NOT EXISTS embeddings(key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )

    def encode(self, texts: list[str]) -> torch.Tensor:
        keys = [hashlib.sha256((self.identity + "\0" + t).encode()).hexdigest() for t in texts]
        found = {}
        with sqlite3.connect(self.path, timeout=60) as c:
            for key in dict.fromkeys(keys):
                row = c.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
                if row:
                    value = np.frombuffer(row[0], dtype=np.float32).copy()
                    if value.size != self.dimension:
                        raise ValueError("Embedding cache dimension mismatch")
                    found[key] = torch.from_numpy(value)
            missing = {key: text for key, text in zip(keys, texts, strict=True) if key not in found}
            items = list(missing.items())
            for start in range(0, len(items), self.encoder.batch_size):
                batch = items[start : start + self.encoder.batch_size]
                for (key, _), vector in zip(
                    batch, self.encoder.encode([t for _, t in batch]), strict=True
                ):
                    found[key] = vector
                    c.execute(
                        "INSERT OR IGNORE INTO embeddings VALUES (?,?)",
                        (key, vector.numpy().astype(np.float32).tobytes()),
                    )
        return (
            torch.stack([found[key] for key in keys]) if keys else torch.zeros((0, self.dimension))
        )
