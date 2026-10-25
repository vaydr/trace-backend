import re
from collections import defaultdict, deque

import torch
import torch.nn.functional as F

from ..schema import Triple


def is_opaque(entity: str) -> bool:
    return bool(re.fullmatch(r"(?:https?://rdf\.freebase\.com/ns/)?[mg]\.[A-Za-z0-9_]+", entity))


def make_features(question: str, topics: list[str], triples: list[Triple], encoder) -> dict:
    entities = sorted({v for t in triples for v in (t.subject, t.object)})
    text = [e for e in entities if not is_opaque(e)]
    opaque = [e for e in entities if is_opaque(e)]
    entities = text + opaque
    relations = sorted({t.predicate for t in triples})
    eids = {v: i for i, v in enumerate(entities)}
    rids = {v: i for i, v in enumerate(relations)}
    topic_mask = torch.tensor([int(e in topics) for e in entities], dtype=torch.long)
    return {
        "q_emb": encoder.encode([question]),
        "entity_embs": encoder.encode(text),
        "relation_embs": encoder.encode(relations),
        "num_non_text_entities": len(opaque),
        "topic_one_hot": F.one_hot(topic_mask, num_classes=2).float(),
        "heads": torch.tensor([eids[t.subject] for t in triples], dtype=torch.long),
        "relations": torch.tensor([rids[t.predicate] for t in triples], dtype=torch.long),
        "tails": torch.tensor([eids[t.object] for t in triples], dtype=torch.long),
        "entities": entities,
    }


def move_features(features: dict, device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }


def _distances(adjacency: dict, start: str) -> dict[str, int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency[vertex]:
            if neighbor not in distances:
                distances[neighbor] = distances[vertex] + 1
                queue.append(neighbor)
    return distances


def weak_labels(triples: list[Triple], topics: list[str], answers: list[str]) -> torch.Tensor:
    """Union of directed shortest paths in either orientation, as upstream.

    Distance equalities avoid enumerating exponentially many shortest paths.
    Parallel predicates between the same endpoints all receive supervision.
    """
    forward, backward = defaultdict(set), defaultdict(set)
    for t in triples:
        forward[t.subject].add(t.object)
        backward[t.object].add(t.subject)
    labels = torch.zeros(len(triples))
    caches = {}

    def distances(direction, vertex):
        key = direction, vertex
        if key not in caches:
            caches[key] = _distances(forward if direction == "forward" else backward, vertex)
        return caches[key]

    for topic in topics:
        for answer in answers:
            directions = [(topic, answer), (answer, topic)]
            available = [(s, t, distances("forward", s).get(t)) for s, t in directions]
            best = min((d for _, _, d in available if d is not None), default=None)
            if best is None or best == 0:
                continue
            for start, end, length in available:
                if length != best:
                    continue
                ds, dt = distances("forward", start), distances("backward", end)
                for i, triple in enumerate(triples):
                    if (
                        ds.get(triple.subject, float("inf"))
                        + 1
                        + dt.get(triple.object, float("inf"))
                        == best
                    ):
                        labels[i] = 1
    return labels
