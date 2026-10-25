from __future__ import annotations

import threading
from pathlib import Path

from .data import file_sha256
from .ranking import RANKING_RULE, rank_triple_indices
from .schema import ScoredTriple, Triple


class SubgraphRetriever:
    def __init__(
        self,
        checkpoint: Path,
        embedding_cache: Path,
        device: str = "auto",
        allow_test_encoder: bool = False,
    ):
        from .ml.encoder import CachedEncoder, TextEncoder
        from .ml.training import load_checkpoint

        self.model, self.checkpoint = load_checkpoint(checkpoint, device)
        if self.checkpoint["encoder"].get("kind") == "hash" and not allow_test_encoder:
            raise ValueError("A synthetic test checkpoint cannot serve production queries")
        self.encoder = CachedEncoder(
            TextEncoder(self.checkpoint["encoder"], device), embedding_cache
        )
        if self.encoder.dimension != self.checkpoint["dimension"]:
            raise ValueError("Encoder dimension differs from checkpoint")
        self.device = next(self.model.parameters()).device
        self.lock = threading.Lock()
        self.metadata = {
            "name": "SubgraphRAG",
            "epoch": self.checkpoint["epoch"],
            "encoder": self.checkpoint["encoder"],
            "data_fingerprint": self.checkpoint["data_fingerprint"],
            "checkpoint_sha256": file_sha256(checkpoint),
            "ranking": RANKING_RULE,
        }

    def retrieve(
        self, question: str, topics: list[str], candidates: list[Triple], k: int
    ) -> list[ScoredTriple]:
        import torch

        from .ml.features import make_features, move_features

        if not candidates:
            return []
        with self.lock, torch.inference_mode():
            features = move_features(
                make_features(question, topics, candidates, self.encoder), self.device
            )
            logits = self.model(features).cpu()
            chosen = rank_triple_indices(logits.tolist(), [t.id for t in candidates], k)
            scores = torch.sigmoid(logits[chosen]).tolist()
        return [
            ScoredTriple(**candidates[i].model_dump(), score=score)
            for i, score in zip(chosen, scores, strict=True)
        ]
