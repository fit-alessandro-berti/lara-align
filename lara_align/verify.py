from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Iterable, Sequence

from pm4py.objects.petri_net.obj import Marking, PetriNet

from lara_align.types import Alignment, AlignmentMove, CostModel, MoveKind, VerificationResult


def trace_labels(trace: Sequence[object], activity_key: str = "concept:name") -> list[str]:
    labels: list[str] = []
    for event in trace:
        if isinstance(event, Mapping):
            labels.append(str(event[activity_key]))
        else:
            labels.append(str(getattr(event, activity_key)))
    return labels


def transition_by_name(net: PetriNet) -> dict[str, PetriNet.Transition]:
    return {str(transition.name): transition for transition in net.transitions}


def normalize_marking(marking: Marking | Counter | dict) -> Counter:
    return Counter({place: int(tokens) for place, tokens in marking.items() if int(tokens) != 0})


def marking_equals(left: Marking | Counter | dict, right: Marking | Counter | dict) -> bool:
    return normalize_marking(left) == normalize_marking(right)


def is_enabled(transition: PetriNet.Transition, marking: Marking | Counter | dict) -> bool:
    current = normalize_marking(marking)
    for arc in transition.in_arcs:
        if current[arc.source] < int(arc.weight):
            return False
    return True


def enabled_transitions(net: PetriNet, marking: Marking | Counter | dict) -> list[PetriNet.Transition]:
    return sorted(
        [transition for transition in net.transitions if is_enabled(transition, marking)],
        key=lambda transition: str(transition.name),
    )


def fire_transition(
    transition: PetriNet.Transition, marking: Marking | Counter | dict
) -> Counter:
    if not is_enabled(transition, marking):
        raise ValueError(f"Transition {transition.name!s} is not enabled")

    next_marking = normalize_marking(marking)
    for arc in transition.in_arcs:
        next_marking[arc.source] -= int(arc.weight)
        if next_marking[arc.source] == 0:
            del next_marking[arc.source]
    for arc in transition.out_arcs:
        next_marking[arc.target] += int(arc.weight)
    return next_marking


def fire_sequence(
    transitions: Iterable[PetriNet.Transition],
    initial_marking: Marking | Counter | dict,
) -> Counter:
    marking = normalize_marking(initial_marking)
    for transition in transitions:
        marking = fire_transition(transition, marking)
    return marking


def move_cost(move: AlignmentMove, cost_model: CostModel, trace_index: int) -> int:
    if move.kind == MoveKind.LOG:
        return cost_model.log_cost(trace_index)
    if move.kind == MoveKind.MODEL:
        if move.transition_name is None:
            raise ValueError("Model move must carry a transition name")
        return cost_model.model_cost(move.transition_name, move.transition_label)
    if move.transition_name is None:
        raise ValueError("Synchronous move must carry a transition name")
    return cost_model.sync_cost(move.transition_name)


def verify_alignment(
    alignment: Alignment,
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    trace: Sequence[object],
    cost_model: CostModel | None = None,
    activity_key: str = "concept:name",
) -> VerificationResult:
    """Simulate an alignment and verify trace/model projections exactly."""

    costs = cost_model or CostModel()
    labels = trace_labels(trace, activity_key=activity_key)
    transitions = transition_by_name(net)
    marking = normalize_marking(initial_marking)
    trace_index = 0
    total_cost = 0

    for move_index, move in enumerate(alignment.moves):
        if move.log_label is not None:
            if trace_index >= len(labels):
                return VerificationResult(
                    legal=False,
                    cost=total_cost,
                    consumed_events=trace_index,
                    final_marking_reached=False,
                    failure_index=move_index,
                    reason="alignment consumes more log events than the trace contains",
                )
            if move.log_label != labels[trace_index]:
                return VerificationResult(
                    legal=False,
                    cost=total_cost,
                    consumed_events=trace_index,
                    final_marking_reached=False,
                    failure_index=move_index,
                    reason=(
                        f"log projection mismatch: expected {labels[trace_index]!r}, "
                        f"got {move.log_label!r}"
                    ),
                )

        if move.transition_name is not None:
            transition = transitions.get(move.transition_name)
            if transition is None:
                return VerificationResult(
                    legal=False,
                    cost=total_cost,
                    consumed_events=trace_index,
                    final_marking_reached=False,
                    failure_index=move_index,
                    reason=f"unknown transition {move.transition_name!r}",
                )
            if move.transition_label != transition.label:
                return VerificationResult(
                    legal=False,
                    cost=total_cost,
                    consumed_events=trace_index,
                    final_marking_reached=False,
                    failure_index=move_index,
                    reason=(
                        f"transition label mismatch for {move.transition_name!r}: "
                        f"expected {transition.label!r}, got {move.transition_label!r}"
                    ),
                )
            if move.kind == MoveKind.SYNCHRONOUS and move.log_label != transition.label:
                return VerificationResult(
                    legal=False,
                    cost=total_cost,
                    consumed_events=trace_index,
                    final_marking_reached=False,
                    failure_index=move_index,
                    reason=(
                        f"synchronous move label mismatch: event {move.log_label!r}, "
                        f"transition {transition.label!r}"
                    ),
                )
            if not is_enabled(transition, marking):
                return VerificationResult(
                    legal=False,
                    cost=total_cost,
                    consumed_events=trace_index,
                    final_marking_reached=False,
                    failure_index=move_index,
                    reason=f"transition {move.transition_name!r} is not enabled",
                )
            marking = fire_transition(transition, marking)

        try:
            total_cost += move_cost(move, costs, trace_index)
        except (IndexError, ValueError) as exc:
            return VerificationResult(
                legal=False,
                cost=total_cost,
                consumed_events=trace_index,
                final_marking_reached=False,
                failure_index=move_index,
                reason=str(exc),
            )

        if move.log_label is not None:
            trace_index += 1

    final_reached = marking_equals(marking, final_marking)
    if trace_index != len(labels):
        return VerificationResult(
            legal=False,
            cost=total_cost,
            consumed_events=trace_index,
            final_marking_reached=final_reached,
            failure_index=len(alignment.moves),
            reason=f"alignment consumed {trace_index} of {len(labels)} trace events",
        )
    if not final_reached:
        return VerificationResult(
            legal=False,
            cost=total_cost,
            consumed_events=trace_index,
            final_marking_reached=False,
            failure_index=len(alignment.moves),
            reason="model projection did not reach the final marking",
        )

    return VerificationResult(
        legal=True,
        cost=total_cost,
        consumed_events=trace_index,
        final_marking_reached=True,
    )
