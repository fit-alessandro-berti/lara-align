from __future__ import annotations

from lara_align.types import Alignment


def synchronous_score_rows(candidate, exact_alignment: Alignment | None, net, initial_marking):
    if candidate is None:
        return []
    exact_choices = []
    if exact_alignment is not None:
        exact_choices = [
            move.transition_name
            for move in exact_alignment.moves
            if move.log_label is not None
        ]
    rows = []
    for event in candidate.neural_diagnostics.get("sync_scores", []):
        event_index = event["event_index"]
        for transition in event["transitions"]:
            rows.append(
                {
                    "Event": event_index,
                    "Observed label": event["observed_label"],
                    "Transition ID": transition["transition_id"],
                    "Transition label": transition["transition_label"],
                    "Decoder score": transition["score"],
                    "Rank": transition["rank"],
                    "Selected": transition["selected"],
                    "Exact selected": (
                        event_index < len(exact_choices)
                        and exact_choices[event_index] == transition["transition_id"]
                    ),
                }
            )
    return rows


def log_score_rows(candidate, exact_alignment: Alignment | None):
    if candidate is None:
        return []
    candidate_log_positions = {
        index
        for index, move in enumerate(
            [m for m in candidate.alignment.moves if m.log_label is not None]
        )
        if move.transition_name is None
    } if candidate.alignment else set()
    exact_log_positions = {
        index
        for index, move in enumerate(
            [m for m in exact_alignment.moves if m.log_label is not None]
        )
        if move.transition_name is None
    } if exact_alignment else set()
    return [
        {
            "Trace position": index,
            "Observed label": candidate.labels[index],
            "Predicted deviation score": score,
            "Candidate log move": index in candidate_log_positions,
            "Exact log move": index in exact_log_positions,
        }
        for index, score in enumerate(
            candidate.neural_diagnostics.get("log_move_scores", [])
        )
    ]
