from __future__ import annotations

from dataclasses import dataclass

from lara_align.types import Alignment, CostModel, MoveKind
from lara_align.verify import (
    enabled_transitions,
    fire_transition,
    is_enabled,
    move_cost,
    normalize_marking,
    transition_by_name,
)


@dataclass
class ReplaySnapshot:
    move_index: int
    trace_index_before: int
    trace_index_after: int
    marking_before: dict[str, int]
    enabled_transitions: list[str]
    selected_move: dict
    fired_transition: str | None
    marking_after: dict[str, int]
    move_cost: int | None
    cumulative_cost: int
    verification_state: str
    error: str | None = None


def _named_marking(marking) -> dict[str, int]:
    return {
        str(place.name): int(tokens)
        for place, tokens in normalize_marking(marking).items()
    }


def replay_alignment(
    alignment: Alignment,
    net,
    initial_marking,
    final_marking,
    labels: list[str],
    cost_model: CostModel | None = None,
) -> list[ReplaySnapshot]:
    costs = cost_model or CostModel()
    transitions = transition_by_name(net)
    marking = normalize_marking(initial_marking)
    trace_index = 0
    cumulative = 0
    snapshots: list[ReplaySnapshot] = []
    failed = False

    for move_index, move in enumerate(alignment.moves):
        before = normalize_marking(marking)
        enabled = [str(t.name) for t in enabled_transitions(net, before)]
        state = "valid"
        error = None
        fired = None
        step_cost = None
        after_trace_index = trace_index + (1 if move.log_label is not None else 0)

        if failed:
            state = "not_replayed_after_failure"
        elif move.log_label is not None and (
            trace_index >= len(labels) or labels[trace_index] != move.log_label
        ):
            state = "invalid"
            error = "The move does not match the next observed event."
            failed = True
        elif move.transition_name is not None:
            transition = transitions.get(move.transition_name)
            if transition is None:
                state = "invalid"
                error = f"Unknown transition {move.transition_name!r}."
                failed = True
            elif not is_enabled(transition, before):
                state = "invalid"
                missing = [
                    str(arc.source.name)
                    for arc in transition.in_arcs
                    if before.get(arc.source, 0) < int(arc.weight)
                ]
                error = "Transition is not enabled; missing tokens in: " + ", ".join(missing)
                failed = True
            else:
                marking = fire_transition(transition, before)
                fired = move.transition_name

        if not failed:
            try:
                step_cost = move_cost(move, costs, trace_index)
                cumulative += step_cost
            except (ValueError, IndexError) as exc:
                state = "invalid"
                error = str(exc)
                failed = True
        if move.log_label is not None and not failed:
            trace_index = after_trace_index

        snapshots.append(
            ReplaySnapshot(
                move_index=move_index,
                trace_index_before=trace_index - (1 if move.log_label is not None and not failed else 0),
                trace_index_after=trace_index,
                marking_before=_named_marking(before),
                enabled_transitions=enabled,
                selected_move={
                    "log_label": move.log_label,
                    "transition_id": move.transition_name,
                    "transition_label": move.transition_label,
                    "kind": move.kind.value,
                },
                fired_transition=fired,
                marking_after=_named_marking(marking),
                move_cost=step_cost,
                cumulative_cost=cumulative,
                verification_state=state,
                error=error,
            )
        )
    return snapshots


def alignment_move_rows(
    alignment: Alignment,
    net,
    initial_marking,
    final_marking,
    labels: list[str],
    cost_model: CostModel | None = None,
) -> list[dict]:
    snapshots = replay_alignment(
        alignment, net, initial_marking, final_marking, labels, cost_model
    )
    rows = []
    for snapshot in snapshots:
        move = snapshot.selected_move
        transition_label = move["transition_label"]
        move_type = move["kind"]
        if move_type == MoveKind.MODEL.value and transition_label is None:
            move_type = "silent"
        rows.append(
            {
                "Move index": snapshot.move_index,
                "Trace position": snapshot.trace_index_before,
                "Log label": move["log_label"] or ">>",
                "Transition ID": move["transition_id"] or ">>",
                "Transition label": (
                    "τ" if move["transition_id"] and transition_label is None else transition_label or ">>"
                ),
                "Move type": move_type,
                "Cost": snapshot.move_cost,
                "Cumulative cost": snapshot.cumulative_cost,
                "Marking valid": snapshot.verification_state == "valid",
                "Verification error": snapshot.error,
            }
        )
    return rows


def anchored_alignment(alignment: Alignment | None, trace_length: int) -> list[dict]:
    slots = [{"before": [], "consume": None} for _ in range(trace_length)]
    trailing: list = []
    if alignment is None:
        return [*slots, {"before": trailing, "consume": None}]
    trace_position = 0
    for move in alignment.moves:
        if move.log_label is None:
            target = slots[trace_position]["before"] if trace_position < trace_length else trailing
            target.append(move)
        elif trace_position < trace_length:
            slots[trace_position]["consume"] = move
            trace_position += 1
    return [*slots, {"before": trailing, "consume": None}]


def alignment_differences(
    candidate: Alignment | None, exact: Alignment | None, trace_length: int
) -> list[int]:
    left = anchored_alignment(candidate, trace_length)
    right = anchored_alignment(exact, trace_length)

    def signature(slot):
        return (
            tuple((move.transition_name, move.transition_label) for move in slot["before"]),
            None
            if slot["consume"] is None
            else (
                slot["consume"].log_label,
                slot["consume"].transition_name,
                slot["consume"].transition_label,
            ),
        )

    return [index for index, (a, b) in enumerate(zip(left, right)) if signature(a) != signature(b)]
