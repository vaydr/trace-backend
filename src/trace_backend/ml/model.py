"""SubgraphRAG MLP with directional distance encoding.

Architecture follows Graph-COM/SubgraphRAG at 48eb05e56d11293fa780849512310462c45b956b.
See THIRD_PARTY_NOTICES.md and licenses/SubgraphRAG.txt.
"""

import torch
from torch import nn


def mean_propagate(
    values: torch.Tensor, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    result = torch.zeros_like(values)
    result.index_add_(0, target, values[source])
    counts = torch.zeros((values.shape[0], 1), device=values.device)
    counts.index_add_(0, target, torch.ones((len(target), 1), device=values.device))
    return result / counts.clamp_min(1)


class RetrieverModel(nn.Module):
    def __init__(
        self,
        dimension: int,
        topic_pe: bool = True,
        forward_rounds: int = 2,
        reverse_rounds: int = 2,
    ):
        super().__init__()
        self.dimension = dimension
        self.topic_pe = topic_pe
        self.forward_rounds = forward_rounds
        self.reverse_rounds = reverse_rounds
        self.non_text_entity_emb = nn.Embedding(1, dimension)
        structural_dim = 2 * int(topic_pe) + 2 * (forward_rounds + reverse_rounds)
        self.pred = nn.Sequential(
            nn.Linear(4 * dimension + 2 * structural_dim, dimension),
            nn.ReLU(),
            nn.Linear(dimension, 1),
        )

    def node_features(self, features: dict) -> torch.Tensor:
        entities = torch.cat(
            [
                features["entity_embs"],
                self.non_text_entity_emb.weight.expand(features["num_non_text_entities"], -1),
            ]
        )
        pieces = [entities]
        topic = features["topic_one_hot"]
        if self.topic_pe:
            pieces.append(topic)
        for source, target, rounds in [
            (features["heads"], features["tails"], self.forward_rounds),
            (features["tails"], features["heads"], self.reverse_rounds),
        ]:
            values = topic
            for _ in range(rounds):
                values = mean_propagate(values, source, target)
                pieces.append(values)
        return torch.cat(pieces, dim=1)

    def score_chunk(
        self, features: dict, nodes: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        h = features["heads"][start:end]
        r = features["relations"][start:end]
        t = features["tails"][start:end]
        return self.pred(
            torch.cat(
                [
                    features["q_emb"].expand(len(h), -1),
                    nodes[h],
                    features["relation_embs"][r],
                    nodes[t],
                ],
                dim=1,
            )
        ).flatten()

    def forward(self, features: dict, batch_size: int = 4096) -> torch.Tensor:
        nodes = self.node_features(features)
        n = len(features["heads"])
        if not n:
            return nodes.new_empty(0)
        return torch.cat(
            [
                self.score_chunk(features, nodes, i, min(i + batch_size, n))
                for i in range(0, n, batch_size)
            ]
        )
