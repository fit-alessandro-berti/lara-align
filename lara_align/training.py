from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from lara_align.features import PetriTraceFeatures
from lara_align.model import LARAForwardOutput
from lara_align.types import Alignment, MoveKind


@dataclass
class AlignmentTargets:
    sync_transition_targets: torch.Tensor
    log_move_targets: torch.Tensor
    model_move_targets: torch.Tensor
    optimal_cost: torch.Tensor | None = None


class LARALoss(nn.Module):
    """Composite loss for move imitation, cost shaping, and router regularization."""

    def __init__(
        self,
        move_weight: float = 1.0,
        cost_weight: float = 0.03,
        router_boundary_weight: float = 0.01,
        router_balance_weight: float = 0.01,
        router_entropy_weight: float = 0.001,
        cost_beta: float = 1.0,
        bce_label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.move_weight = move_weight
        self.cost_weight = cost_weight
        self.router_boundary_weight = router_boundary_weight
        self.router_balance_weight = router_balance_weight
        self.router_entropy_weight = router_entropy_weight
        self.cost_beta = cost_beta
        self.bce_label_smoothing = bce_label_smoothing

    def forward(
        self,
        output: LARAForwardOutput,
        features: PetriTraceFeatures,
        targets: AlignmentTargets,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        losses["sync_move"] = _sync_cross_entropy(
            output.sync_logits, targets.sync_transition_targets
        )
        losses["log_move"] = _binary_cross_entropy_with_optional_empty_logits(
            output.log_move_logits,
            _smooth_binary_targets(
                targets.log_move_targets.to(output.log_move_logits.device).float(),
                self.bce_label_smoothing,
            ),
        )
        losses["model_move"] = _binary_cross_entropy_with_optional_empty_logits(
            output.model_move_logits,
            _smooth_binary_targets(
                targets.model_move_targets.to(output.model_move_logits.device).float(),
                self.bce_label_smoothing,
            ),
        )
        move_loss = losses["sync_move"] + losses["log_move"] + losses["model_move"]

        if targets.optimal_cost is not None:
            predicted_cost = output.local_upper_bounds.sum()
            target_cost = targets.optimal_cost.to(predicted_cost.device).float()
            losses["cost"] = F.smooth_l1_loss(
                predicted_cost,
                target_cost,
                beta=self.cost_beta,
            )
        else:
            losses["cost"] = torch.zeros((), device=output.model_move_logits.device)

        router = router_regularization(
            features,
            output.transition_region_probs,
            output.event_region_probs,
        )
        losses.update({f"router_{key}": value for key, value in router.items()})
        losses["total"] = (
            self.move_weight * move_loss
            + self.cost_weight * losses["cost"]
            + self.router_boundary_weight * router["boundary"]
            + self.router_balance_weight * router["balance"]
            + self.router_entropy_weight * router["entropy"]
        )
        return losses


def targets_from_alignment(
    alignment: Alignment,
    features: PetriTraceFeatures,
    optimal_cost: int | None = None,
) -> AlignmentTargets:
    transition_indices = {
        transition_name: index
        for index, transition_name in enumerate(features.transition_names)
    }
    sync_targets = torch.full((features.trace_length,), -1, dtype=torch.long)
    log_targets = torch.zeros((features.trace_length,), dtype=torch.float32)
    model_targets = torch.zeros((features.num_transitions,), dtype=torch.float32)

    event_index = 0
    for move in alignment.moves:
        if move.kind == MoveKind.SYNCHRONOUS:
            if move.transition_name in transition_indices and event_index < features.trace_length:
                sync_targets[event_index] = transition_indices[move.transition_name]
            event_index += 1
        elif move.kind == MoveKind.LOG:
            if event_index < features.trace_length:
                log_targets[event_index] = 1.0
            event_index += 1
        elif move.transition_name in transition_indices:
            model_targets[transition_indices[move.transition_name]] = 1.0

    return AlignmentTargets(
        sync_transition_targets=sync_targets,
        log_move_targets=log_targets,
        model_move_targets=model_targets,
        optimal_cost=(
            torch.tensor(float(optimal_cost), dtype=torch.float32)
            if optimal_cost is not None
            else None
        ),
    )


def router_regularization(
    features: PetriTraceFeatures,
    transition_region_probs: torch.Tensor,
    event_region_probs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    boundary = _boundary_cut_penalty(features, transition_region_probs)
    if event_region_probs.numel() == 0:
        loads = transition_region_probs.mean(dim=0)
    else:
        loads = torch.cat([transition_region_probs, event_region_probs], dim=0).mean(dim=0)
    balance = ((loads - loads.mean()) ** 2).mean()
    entropy = _entropy(transition_region_probs)
    if event_region_probs.numel() > 0:
        entropy = 0.5 * (entropy + _entropy(event_region_probs))
    return {"boundary": boundary, "balance": balance, "entropy": entropy}


def _sync_cross_entropy(sync_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.to(sync_logits.device)
    mask = targets >= 0
    if not torch.any(mask):
        return torch.zeros((), device=sync_logits.device)
    return F.cross_entropy(sync_logits[mask], targets[mask])


def _boundary_cut_penalty(
    features: PetriTraceFeatures,
    transition_region_probs: torch.Tensor,
) -> torch.Tensor:
    if features.edge_index.numel() == 0 or transition_region_probs.shape[0] == 0:
        return torch.zeros((), device=transition_region_probs.device)

    place_to_transitions: dict[int, list[int]] = {}
    transition_offset = features.num_places
    sources, targets = features.edge_index
    for source, target in zip(sources.tolist(), targets.tolist()):
        if source < transition_offset <= target:
            place_to_transitions.setdefault(source, []).append(target - transition_offset)
        elif target < transition_offset <= source:
            place_to_transitions.setdefault(target, []).append(source - transition_offset)

    penalties: list[torch.Tensor] = []
    for incident_transitions in place_to_transitions.values():
        unique = sorted(set(incident_transitions))
        for left_index, left in enumerate(unique):
            for right in unique[left_index + 1 :]:
                same_region_prob = (
                    transition_region_probs[left] * transition_region_probs[right]
                ).sum()
                penalties.append(1.0 - same_region_prob)

    if not penalties:
        return torch.zeros((), device=transition_region_probs.device)
    return torch.stack(penalties).mean()


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()


def _smooth_binary_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return targets
    if smoothing >= 0.5:
        raise ValueError("bce_label_smoothing must be in [0, 0.5)")
    return targets * (1.0 - 2.0 * smoothing) + smoothing


def _binary_cross_entropy_with_optional_empty_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets)
