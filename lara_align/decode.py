from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from pm4py.objects.petri_net.obj import Marking, PetriNet

from lara_align.features import PetriTraceFeatures
from lara_align.model import LARAForwardOutput
from lara_align.types import Alignment, AlignmentMove, CostModel
from lara_align.verify import trace_labels, verify_alignment

_MarkingDict = dict


class _NetRuntime:
    """Precomputed net structure and move scores for fast marking search.

    The generic helpers in `lara_align.verify` normalize the marking into a
    fresh Counter on every enabledness check, which dominates decode time.
    This runtime indexes presets/postsets once per decode, keeps markings as
    plain dicts without zero entries, and extracts neural scores into Python
    floats up front.
    """

    def __init__(
        self,
        net: PetriNet,
        feature_transition_names: Sequence[str],
        sync_logits: torch.Tensor | None,
        model_logits: torch.Tensor | None,
    ) -> None:
        self.transitions = sorted(net.transitions, key=lambda t: str(t.name))
        feature_index = {str(name): i for i, name in enumerate(feature_transition_names)}

        model_scores = model_logits.tolist() if model_logits is not None else None
        self.sync_scores: list[list[float]] | None = (
            sync_logits.tolist() if sync_logits is not None else None
        )
        self.feature_index = feature_index

        self.preset: dict[PetriNet.Transition, list[tuple[object, int]]] = {}
        self.postset: dict[PetriNet.Transition, list[tuple[object, int]]] = {}
        self.consumers: dict[object, list[PetriNet.Transition]] = {}
        self.always_enabled: list[PetriNet.Transition] = []
        self.by_label: dict[str, list[PetriNet.Transition]] = {}
        self.model_score: dict[PetriNet.Transition, float] = {}
        self.search_order_key: dict[PetriNet.Transition, tuple] = {}

        for transition in self.transitions:
            preset = [(arc.source, int(arc.weight)) for arc in transition.in_arcs]
            postset = [(arc.target, int(arc.weight)) for arc in transition.out_arcs]
            self.preset[transition] = preset
            self.postset[transition] = postset
            if preset:
                for place, _ in preset:
                    self.consumers.setdefault(place, []).append(transition)
            else:
                self.always_enabled.append(transition)

            if transition.label is not None:
                self.by_label.setdefault(str(transition.label), []).append(transition)

            index = feature_index.get(str(transition.name))
            score = (
                model_scores[index]
                if model_scores is not None and index is not None
                else 0.0
            )
            self.model_score[transition] = score
            self.search_order_key[transition] = (
                1 if transition.label is None else 0,
                score,
                str(transition.name),
            )

    def sync_score(self, transition: PetriNet.Transition, event_index: int) -> float:
        if self.sync_scores is None or event_index >= len(self.sync_scores):
            return 0.0
        index = self.feature_index.get(str(transition.name))
        if index is None:
            return 0.0
        return self.sync_scores[event_index][index]

    def is_enabled(self, transition: PetriNet.Transition, marking: _MarkingDict) -> bool:
        for place, weight in self.preset[transition]:
            if marking.get(place, 0) < weight:
                return False
        return True

    def fire(self, transition: PetriNet.Transition, marking: _MarkingDict) -> _MarkingDict:
        next_marking = dict(marking)
        for place, weight in self.preset[transition]:
            remaining = next_marking.get(place, 0) - weight
            if remaining < 0:
                raise ValueError(f"Transition {transition.name!s} is not enabled")
            if remaining:
                next_marking[place] = remaining
            else:
                next_marking.pop(place, None)
        for place, weight in self.postset[transition]:
            next_marking[place] = next_marking.get(place, 0) + weight
        return next_marking

    def enabled(self, marking: _MarkingDict) -> list[PetriNet.Transition]:
        result: list[PetriNet.Transition] = []
        seen: set[int] = set()
        for place in marking:
            for transition in self.consumers.get(place, ()):
                key = id(transition)
                if key in seen:
                    continue
                seen.add(key)
                if self.is_enabled(transition, marking):
                    result.append(transition)
        result.extend(self.always_enabled)
        return result

    def search_ordered_enabled(self, marking: _MarkingDict) -> list[PetriNet.Transition]:
        return sorted(
            self.enabled(marking),
            key=self.search_order_key.__getitem__,
            reverse=True,
        )

    def best_enabled_sync_transition(
        self,
        label: str,
        event_index: int,
        marking: _MarkingDict,
    ) -> PetriNet.Transition | None:
        best: PetriNet.Transition | None = None
        best_score = float("-inf")
        for transition in self.by_label.get(label, ()):
            if not self.is_enabled(transition, marking):
                continue
            score = self.sync_score(transition, event_index)
            if score > best_score:
                best = transition
                best_score = score
        return best


def _marking_dict(marking: Marking | dict) -> _MarkingDict:
    return {place: int(tokens) for place, tokens in marking.items() if int(tokens) != 0}


def _marking_cache_key(marking: _MarkingDict) -> frozenset:
    return frozenset(marking.items())


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
        runtime = _NetRuntime(
            net,
            features.transition_names,
            output.sync_logits.detach().cpu() if output is not None else None,
            output.model_move_logits.detach().cpu() if output is not None else None,
        )

        marking = _marking_dict(initial_marking)
        final = _marking_dict(final_marking)
        moves: list[AlignmentMove] = []

        for event_index, label in enumerate(labels):
            transition = runtime.best_enabled_sync_transition(label, event_index, marking)
            if transition is None:
                prefix_path = self._find_path(
                    runtime,
                    marking,
                    lambda candidate: runtime.best_enabled_sync_transition(
                        label, event_index, candidate
                    )
                    is not None,
                    max_depth=self.max_prefix_model_depth,
                )
                if prefix_path:
                    for path_transition in prefix_path:
                        moves.append(_model_move(path_transition))
                        marking = runtime.fire(path_transition, marking)
                    transition = runtime.best_enabled_sync_transition(
                        label, event_index, marking
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
            marking = runtime.fire(transition, marking)

        final_path = self._find_path(
            runtime,
            marking,
            lambda candidate: candidate == final,
            max_depth=self.max_final_model_depth,
        )
        for transition in final_path:
            moves.append(_model_move(transition))
            marking = runtime.fire(transition, marking)

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
        alignment.metadata["verification"] = verification
        if verification.reason:
            alignment.metadata["verification_reason"] = verification.reason
        return alignment

    def _find_path(
        self,
        runtime: _NetRuntime,
        start_marking: _MarkingDict,
        is_goal: Callable[[_MarkingDict], bool],
        max_depth: int,
    ) -> list[PetriNet.Transition]:
        if is_goal(start_marking):
            return []
        queue: deque[tuple[_MarkingDict, list[PetriNet.Transition]]] = deque(
            [(start_marking, [])]
        )
        visited = {_marking_cache_key(start_marking)}

        while queue:
            marking, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for transition in runtime.search_ordered_enabled(marking):
                next_marking = runtime.fire(transition, marking)
                key = _marking_cache_key(next_marking)
                if key in visited:
                    continue
                next_path = [*path, transition]
                if is_goal(next_marking):
                    return next_path
                visited.add(key)
                queue.append((next_marking, next_path))
        return []


def _model_move(transition: PetriNet.Transition) -> AlignmentMove:
    return AlignmentMove(
        log_label=None,
        transition_name=str(transition.name),
        transition_label=None if transition.label is None else str(transition.label),
    )
