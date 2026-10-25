"""Independent architecture check against the released SubgraphRAG equations.

The reference uses dense adjacency multiplication rather than the production
scatter implementation. It follows retrieve/src/model/retriever.py at upstream
48eb05e56d11293fa780849512310462c45b956b, without a PyTorch Geometric dependency.
"""

import pytest
import torch
import torch.nn.functional as F

from trace_backend.ml.model import RetrieverModel


def _dense_reference(features, parameters, topic_pe, forward_rounds, reverse_rounds):
    text = features["entity_embs"]
    nontext = parameters["non_text_entity_emb.weight"].repeat(features["num_non_text_entities"], 1)
    entities = torch.cat((text, nontext), dim=0)
    topic = features["topic_one_hot"]
    n = len(entities)

    # A[target, source] counts every directed edge, including repeated endpoints.
    # Mean aggregation divides by incoming edge count, not unique neighbor count.
    counts = [[0] * n for _ in range(n)]
    for head, tail in zip(features["heads"], features["tails"], strict=True):
        counts[int(tail)][int(head)] += 1
    adjacency = torch.tensor(counts, dtype=text.dtype)
    pieces = [entities]
    if topic_pe:
        pieces.append(topic)
    for matrix, rounds in ((adjacency, forward_rounds), (adjacency.T, reverse_rounds)):
        transition = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1)
        propagated = topic
        for _ in range(rounds):
            propagated = transition @ propagated
            pieces.append(propagated)
    nodes = torch.cat(pieces, dim=1)

    # Literal upstream concatenation: question, head entity + PE/DDE, relation,
    # tail entity + PE/DDE. Assemble rows independently of score_chunk's indexing.
    rows = []
    for head, relation, tail in zip(
        features["heads"], features["relations"], features["tails"], strict=True
    ):
        rows.append(
            torch.cat(
                (
                    features["q_emb"][0],
                    nodes[int(head)],
                    features["relation_embs"][int(relation)],
                    nodes[int(tail)],
                )
            )
        )
    triples = torch.stack(rows)
    hidden = F.relu(F.linear(triples, parameters["pred.0.weight"], parameters["pred.0.bias"]))
    logits = F.linear(hidden, parameters["pred.2.weight"], parameters["pred.2.bias"]).flatten()
    return nodes, logits


@pytest.mark.parametrize(
    ("topic_pe", "forward_rounds", "reverse_rounds"),
    [(True, 2, 2), (False, 1, 3), (True, 3, 1), (False, 0, 2)],
)
def test_dense_upstream_architecture_logits_and_gradients(topic_pe, forward_rounds, reverse_rounds):
    generator = torch.Generator().manual_seed(916)
    dimension = 5
    features = {
        "q_emb": torch.randn((1, dimension), generator=generator, dtype=torch.float64),
        "entity_embs": torch.randn((4, dimension), generator=generator, dtype=torch.float64),
        "relation_embs": torch.randn((3, dimension), generator=generator, dtype=torch.float64),
        "num_non_text_entities": 2,
        # Node 3 is isolated; nodes 4 and 5 share the learned nontext/CVT vector.
        # Two topics ensure both channels and nontext topic handling are exercised.
        "topic_one_hot": F.one_hot(torch.tensor([1, 0, 0, 0, 0, 1]), 2).double(),
        "heads": torch.tensor([0, 0, 2, 1, 4, 5, 2, 4]),
        "relations": torch.tensor([0, 1, 2, 0, 1, 2, 1, 2]),
        "tails": torch.tensor([1, 1, 1, 4, 5, 2, 0, 4]),
    }
    differentiable_inputs = ("q_emb", "entity_embs", "relation_embs")
    for key in differentiable_inputs:
        features[key].requires_grad_()
    reference_features = {
        key: value.detach().clone().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in features.items()
    }

    model = RetrieverModel(dimension, topic_pe, forward_rounds, reverse_rounds).double()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(
                0.2 * torch.randn(parameter.shape, generator=generator, dtype=torch.float64)
            )
            if name == "pred.0.bias":
                parameter.add_(0.5)
    reference_parameters = {
        name: parameter.detach().clone().requires_grad_()
        for name, parameter in model.named_parameters()
    }

    reference_nodes, expected = _dense_reference(
        reference_features, reference_parameters, topic_pe, forward_rounds, reverse_rounds
    )
    torch.testing.assert_close(
        model.node_features(features), reference_nodes, rtol=1e-10, atol=1e-12
    )
    # Deliberately split the eight triples into uneven chunks.
    actual = model(features, batch_size=3)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

    targets = torch.tensor([1, 0, 1, 0, 1, 1, 0, 0], dtype=torch.float64)
    F.binary_cross_entropy_with_logits(actual, targets).backward()
    F.binary_cross_entropy_with_logits(expected, targets).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0, name
        torch.testing.assert_close(
            parameter.grad, reference_parameters[name].grad, rtol=1e-9, atol=1e-12
        )
    for key in differentiable_inputs:
        assert features[key].grad is not None
        torch.testing.assert_close(
            features[key].grad, reference_features[key].grad, rtol=1e-9, atol=1e-12
        )
