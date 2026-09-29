"""Semantic checks for the optional reviewer-study controls."""
from types import SimpleNamespace

import pytest
import torch

from lara_align.decode import GreedyCandidateDecoder, SingleLogRepairDecoder
from lara_align.exact import Pm4PyExactAligner
from lara_align.features import pm4py_to_features
from lara_align.model import LARANeuralModel
from lara_align.synthetic import make_sequence_net, trace_from_labels
from lara_align.training import LARALoss, targets_from_alignment
from lara_align.types import CostModel
from lara_align.verify import verify_alignment


def test_single_log_repair_can_reject_a_greedy_enabling_path():
    net, initial, final = make_sequence_net(["A", "B", "C", "D"])
    trace = trace_from_labels(["A", "D", "B", "C", "C", "D"])
    features = pm4py_to_features(net, initial, final, trace)
    scores = SimpleNamespace(
        sync_logits=torch.zeros((6, 4)), model_move_logits=torch.zeros(4),
        log_move_logits=torch.tensor([-3., 4., -3., -3., -3., -3.]),
    )
    greedy = GreedyCandidateDecoder().decode(net, initial, final, trace, features, scores)
    repaired = SingleLogRepairDecoder(max_alternatives=1).decode(
        net, initial, final, trace, features, scores)
    exact = Pm4PyExactAligner().align_trace(net, initial, final, trace)
    assert greedy.cost == 6 and repaired.cost == exact.cost == 2
    assert repaired.metadata["repair_alternatives"] == 1
    assert repaired.metadata["base_cost"] == 6
    assert verify_alignment(repaired, net, initial, final, trace).legal


@pytest.mark.parametrize("labels", [[], ["A", "B"], ["B", "A"], ["X", "X", "X"]])
def test_repair_retains_a_legal_incumbent_under_custom_costs(labels):
    net, initial, final = make_sequence_net(["A", "B"], silent_prefix=True)
    trace = trace_from_labels(labels)
    costs = CostModel(log=3, model=2, silent_model=0)
    features = pm4py_to_features(net, initial, final, trace, cost_model=costs)
    greedy = GreedyCandidateDecoder().decode(net, initial, final, trace, features,
                                             cost_model=costs)
    repaired = SingleLogRepairDecoder().decode(net, initial, final, trace, features,
                                               cost_model=costs)
    verification = verify_alignment(repaired, net, initial, final, trace, costs)
    assert verification.legal and repaired.cost == verification.cost
    assert repaired.cost <= greedy.cost


def test_log_loss_ablation_removes_log_head_gradients_but_keeps_active_heads():
    net, initial, final = make_sequence_net(["A", "B"])
    trace = trace_from_labels(["A", "X", "B"])
    exact = Pm4PyExactAligner().align_trace(net, initial, final, trace)
    features = pm4py_to_features(net, initial, final, trace)
    targets = targets_from_alignment(exact.alignment, features, exact.cost)
    torch.manual_seed(9)
    model = LARANeuralModel(hidden_dim=16, num_heads=2, graph_layers=1,
                           trace_layers=1, num_regions=2, dropout=0.)
    output = model(features)
    default = LARALoss()(output, features, targets)
    explicit = LARALoss(log_loss_weight=1)(output, features, targets)
    ablated = LARALoss(log_loss_weight=0)(output, features, targets)
    assert torch.equal(default["total"], explicit["total"])
    assert torch.allclose(default["total"] - ablated["total"], default["log_move"])
    ablated["total"].backward()
    assert torch.count_nonzero(model.recomposer.log_head.weight.grad) == 0
    assert torch.count_nonzero(model.recomposer.model_head.weight.grad) > 0
    assert torch.isfinite(model.encoder.label_embedding.weight.grad).all()


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_log_loss_weight_rejects_invalid_values(weight):
    with pytest.raises(ValueError):
        LARALoss(log_loss_weight=weight)
