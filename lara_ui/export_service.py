from __future__ import annotations

import csv
import io
import json
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Iterable

from lara_align.types import Alignment, AlignmentMove
from lara_ui.alignment_runner import result_row
from lara_ui.ui_types import TraceAlignmentResult


def _csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def variant_csv(results: Iterable[TraceAlignmentResult]) -> bytes:
    rows = []
    for result in results:
        base = result_row(result)
        candidate = result.candidate
        exact = result.exact_result
        base.update(
            {
                "Activity sequence": "|".join(result.variant.labels),
                "Candidate validity": None if candidate is None else candidate.legal,
                "Certification status": result.status,
                "Final alignment source": result.final_alignment_source,
                "Failure reason": result.error
                or (
                    candidate.verification.reason
                    if candidate and candidate.verification
                    else None
                ),
                "Exact visited states": (
                    exact.exact_diagnostics.get("visited_states") if exact else None
                ),
                "Exact queued states": (
                    exact.exact_diagnostics.get("queued_states") if exact else None
                ),
                "Exact traversed arcs": (
                    exact.exact_diagnostics.get("traversed_arcs") if exact else None
                ),
            }
        )
        rows.append(base)
    return _csv_bytes(rows)


def case_csv(results: Iterable[TraceAlignmentResult]) -> bytes:
    rows = []
    for result in results:
        candidate = result.candidate
        exact = result.exact_result
        candidate_cost = None if candidate is None else candidate.cost
        exact_cost = None if exact is None else exact.exact_cost
        for case_id in result.variant.case_ids:
            rows.append(
                {
                    "Case ID": case_id,
                    "Variant ID": result.variant.variant_id,
                    "Alignment status": result.status,
                    "Candidate cost": candidate_cost,
                    "Exact cost": exact_cost,
                    "Cost gap": (
                        candidate_cost - exact_cost
                        if candidate_cost is not None and exact_cost is not None
                        else None
                    ),
                    "Certified": bool(exact and exact.certified_optimal),
                    "Final alignment source": result.final_alignment_source,
                }
            )
    return _csv_bytes(rows)


def move_csv(rows: list[dict]) -> bytes:
    return _csv_bytes(rows)


def move_text(rows: list[dict]) -> str:
    return "\n".join(
        f"{row['Move index']:>3}  {row['Log label']} | "
        f"{row['Transition ID']} / {row['Transition label']}  "
        f"[{row['Move type']}, cost={row['Cost']}]"
        for row in rows
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, AlignmentMove):
        return {
            "log_label": value.log_label,
            "transition_id": value.transition_name,
            "transition_label": value.transition_label,
            "move_type": value.kind.value,
            "silent": value.is_silent_model_move,
        }
    if isinstance(value, Alignment):
        return {
            "moves": [_jsonable(move) for move in value.moves],
            "cost": value.cost,
            "source": value.source,
            "metadata": _jsonable(value.metadata),
        }
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def full_json(
    results: Iterable[TraceAlignmentResult],
    run_configuration: dict,
    checkpoint_metadata: dict,
    log_summary: dict,
    net_summary: dict,
    cost_model: Any,
) -> bytes:
    results = list(results)
    payload = {
        "schema": "lara-align-workbench/v1",
        "run_configuration": run_configuration,
        "checkpoint_metadata": checkpoint_metadata,
        "log_summary": log_summary,
        "petri_net_summary": net_summary,
        "cost_model": cost_model,
        "variant_to_cases": {
            result.variant.variant_id: {
                "activity_sequence": result.variant.labels,
                "case_ids": result.variant.case_ids,
            }
            for result in results
        },
        "results": results,
    }
    return json.dumps(_jsonable(payload), indent=2, sort_keys=True).encode("utf-8")
