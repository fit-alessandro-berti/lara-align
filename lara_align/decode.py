from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Sequence

import torch
from pm4py.objects.petri_net.obj import Marking, PetriNet

from lara_align.features import PetriTraceFeatures
from lara_align.model import LARAForwardOutput
from lara_align.types import Alignment, AlignmentMove, CostModel
from lara_align.verify import (
    enabled_transitions,
    fire_transition,
    marking_equals,
    normalize_marking,
    trace_labels,
    verify_alignment,
)


@dataclass
class GreedyCandidateDecoder:
    """Constrained decoder that turns neural move scores into a legal candidate when possible."""

    max_prefix_model_depth: int = 8
    max_final_model_depth: int = 32

    def decode(
        self,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        trace: Sequence[object],
        features: PetriTraceFeatures,
        output: LARAForwardOutput | None = None,
        cost_model: CostModel | None = None,
        activity_key: str = "concept:name",
    ) -> Alignment:
        costs = cost_model or CostModel()
        labels = trace_labels(trace, activity_key=activity_key)
        transitions = sorted(net.transitions, key=lambda transition: str(transition.name))
        transition_indices = {str(name): index for index, name in enumerate(features.transition_names)}
        sync_logits = _detach_scores(output.sync_logits) if output is not None else None
        model_logits = _detach_scores(output.model_move_logits) if output is not None else None

        marking = normalize_marking(initial_marking)
        moves: list[AlignmentMove] = []

        for event_index, label in enumerate(labels):
            transition = self._best_enabled_sync_transition(
                transitions,
                transition_indices,
                marking,
                label,
                event_index,
                sync_logits,
            )
            if transition is None:
                prefix_path = self._find_path_to_enable_label(
                    net,
                    marking,
                    label,
                    transition_indices,
                    model_logits,
                    max_depth=self.max_prefix_model_depth,
                )
                if prefix_path:
                    for path_transition in prefix_path:
                        moves.append(_model_move(path_transition))
                        marking = fire_transition(path_transition, marking)
                    transition = self._best_enabled_sync_transition(
                        transitions,
                        transition_indices,
                        marking,
                        label,
                        event_index,
                        sync_logits,
                    )

            if transition is None:
                moves.append(AlignmentMove(log_label=label, transition_name=None))
                continue

            moves.append(
                AlignmentMove(
                    log_label=label,
                    transition_name=str(transition.name),
                    transition_label=None if transition.label is None else str(transition.label),
                )
            )
            marking = fire_transition(transition, marking)

        final_path = self._find_path_to_final(
            net,
            marking,
            final_marking,
            transition_indices,
            model_logits,
            max_depth=self.max_final_model_depth,
        )
        for transition in final_path:
            moves.append(_model_move(transition))
            marking = fire_transition(transition, marking)

        alignment = Alignment(moves=moves, source="neural_greedy")
        verification = verify_alignment(
            alignment,
            net,
            initial_marking,
            final_marking,
            trace,
            costs,
            activity_key=activity_key,
        )
        alignment.cost = verification.cost
        alignment.metadata["legal"] = verification.legal
        if verification.reason:
            alignment.metadata["verification_reason"] = verification.reason
        return alignment

    def _best_enabled_sync_transition(
        self,
        transitions: list[PetriNet.Transition],
        transition_indices: dict[str, int],
        marking: Counter,
        label: str,
        event_index: int,
        sync_logits: torch.Tensor | None,
    ) -> PetriNet.Transition | None:
        candidates = [
            transition
            for transition in transitions
            if transition.label == label and _is_enabled_cached(transition, marking)
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda transition: self._sync_score(
                transition, transition_indices, event_index, sync_logits
            ),
        )

    def _find_path_to_enable_label(
        self,
        net: PetriNet,
        start_marking: Counter,
        label: str,
        transition_indices: dict[str, int],
        model_logits: torch.Tensor | None,
        max_depth: int,
    ) -> list[PetriNet.Transition]:
        def is_goal(marking: Counter) -> bool:
            return any(
                transition.label == label
                for transition in enabled_transitions(net, marking)
            )

        return self._bounded_marking_search(
            net,
            start_marking,
            is_goal,
            transition_indices,
            model_logits,
            max_depth,
        )

    def _find_path_to_final(
        self,
        net: PetriNet,
        start_marking: Counter,
        final_marking: Marking,
        transition_indices: dict[str, int],
        model_logits: torch.Tensor | None,
        max_depth: int,
    ) -> list[PetriNet.Transition]:
        def is_goal(marking: Counter) -> bool:
            return marking_equals(marking, final_marking)

        return self._bounded_marking_search(
            net,
            start_marking,
            is_goal,
            transition_indices,
            model_logits,
            max_depth,
        )

    def _bounded_marking_search(
        self,
        net: PetriNet,
        start_marking: Counter,
        is_goal,
        transition_indices: dict[str, int],
        model_logits: torch.Tensor | None,
        max_depth: int,
    ) -> list[PetriNet.Transition]:
        if is_goal(start_marking):
            return []
        queue: deque[tuple[Counter, list[PetriNet.Transition]]] = deque(
            [(normalize_marking(start_marking), [])]
        )
        visited = {_marking_key(start_marking)}

        while queue:
            marking, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for transition in self._ordered_enabled(
                net, marking, transition_indices, model_logits
            ):
                try:
                    next_marking = fire_transition(transition, marking)
                except ValueError:
                    continue
                key = _marking_key(next_marking)
                if key in visited:
                    continue
                next_path = [*path, transition]
                if is_goal(next_marking):
                    return next_path
                visited.add(key)
                queue.append((next_marking, next_path))
        return []

    def _ordered_enabled(
        self,
        net: PetriNet,
        marking: Counter,
        transition_indices: dict[str, int],
        model_logits: torch.Tensor | None,
    ) -> list[PetriNet.Transition]:
        transitions = enabled_transitions(net, marking)
        return sorted(
            transitions,
            key=lambda transition: (
                1 if transition.label is None else 0,
                self._model_score(transition, transition_indices, model_logits),
                str(transition.name),
            ),
            reverse=True,
        )

    def _sync_score(
        self,
        transition: PetriNet.Transition,
        transition_indices: dict[str, int],
        event_index: int,
        sync_logits: torch.Tensor | None,
    ) -> float:
        if sync_logits is None:
            return 0.0
        transition_index = transition_indices.get(str(transition.name))
        if transition_index is None or event_index >= sync_logits.shape[0]:
            return 0.0
        return float(sync_logits[event_index, transition_index])

    def _model_score(
        self,
        transition: PetriNet.Transition,
        transition_indices: dict[str, int],
        model_logits: torch.Tensor | None,
    ) -> float:
        if model_logits is None:
            return 0.0
        transition_index = transition_indices.get(str(transition.name))
        if transition_index is None:
            return 0.0
        return float(model_logits[transition_index])


def _model_move(transition: PetriNet.Transition) -> AlignmentMove:
    return AlignmentMove(
        log_label=None,
        transition_name=str(transition.name),
        transition_label=None if transition.label is None else str(transition.label),
    )


def _detach_scores(scores: torch.Tensor) -> torch.Tensor:
    return scores.detach().cpu()


def _marking_key(marking: Counter | Marking) -> tuple[tuple[str, int], ...]:
    normalized = normalize_marking(marking)
    return tuple(sorted((str(place.name), int(tokens)) for place, tokens in normalized.items()))


def _is_enabled_cached(transition: PetriNet.Transition, marking: Counter) -> bool:
    for arc in transition.in_arcs:
        if marking[arc.source] < int(arc.weight):
            return False
    return True
