"""Shared retrieval ordering for checkpoint selection, evaluation, and serving."""

from collections.abc import Sequence

RANKING_RULE = "raw_logit_desc_triple_id_asc_v1"


def rank_triple_indices(logits: Sequence[float], triple_ids: Sequence[str], k: int) -> list[int]:
    if len(logits) != len(triple_ids):
        raise ValueError("One logit is required per triple")
    if k < 1:
        raise ValueError("k must be positive")
    # Rank before sigmoid saturation; break ties by stable fact ID.
    return sorted(range(len(triple_ids)), key=lambda i: (-logits[i], triple_ids[i]))[:k]
