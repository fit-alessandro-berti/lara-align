from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Iterable

from lara_align.certifier import CertifyingAlignmentSystem
from lara_align.types import Alignment, MoveKind
from lara_ui.ui_types import TraceAlignmentResult, TraceVariant


TERMINAL_STATES = {
    "candidate_ready",
    "illegal_candidate",
    "candidate_failure",
    "certified",
    "repaired",
    "exact_timeout",
    "exact_failure",
    "illegal_candidate_exact_available",
    "exact_only",
    "cancelled",
}


class WorkbenchRunner:
    def __init__(
        self,
        system: CertifyingAlignmentSystem,
        net,
        initial_marking,
        final_marking,
        timeout_seconds: float | None = None,
    ) -> None:
        self.system = system
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.timeout_seconds = timeout_seconds

    def queued(self, variants: Iterable[TraceVariant]) -> dict[str, TraceAlignmentResult]:
        return {
            variant.variant_id: TraceAlignmentResult(variant=variant)
            for variant in variants
        }

    def propose(self, result: TraceAlignmentResult) -> TraceAlignmentResult:
        result.state = "candidate_running"
        result.started_at = result.started_at or perf_counter()
        result.timeline.append({"state": result.state, "at": perf_counter()})
        candidate = self.system.propose_trace(
            self.net,
            self.initial_marking,
            self.final_marking,
            result.variant.trace,
            trace_key=result.variant.variant_id,
        )
        result.candidate = candidate
        result.candidate_completed_at = perf_counter()
        if candidate.error:
            result.state = "candidate_failure"
            result.error = candidate.error
        elif candidate.legal:
            result.state = "candidate_ready"
        else:
            result.state = "illegal_candidate"
            result.error = (
                candidate.verification.reason if candidate.verification else None
            )
        result.timeline.append({"state": result.state, "at": perf_counter()})
        return result

    def certify(self, result: TraceAlignmentResult) -> TraceAlignmentResult:
        result.state = "certifying"
        result.timeline.append({"state": result.state, "at": perf_counter()})
        certification = self.system.certify_trace(
            result.candidate,
            self.net,
            self.initial_marking,
            self.final_marking,
            result.variant.trace,
            timeout_seconds=self.timeout_seconds,
        )
        result.certification = certification
        result.state = certification.status
        result.completed_at = perf_counter()
        result.error = certification.error
        result.timeline.append({"state": result.state, "at": result.completed_at})
        return result

    def exact_only(self, result: TraceAlignmentResult) -> TraceAlignmentResult:
        result.state = "certifying"
        result.started_at = result.started_at or perf_counter()
        result.timeline.append({"state": result.state, "at": perf_counter()})
        exact = self.system.certify_trace(
            None,
            self.net,
            self.initial_marking,
            self.final_marking,
            result.variant.trace,
            timeout_seconds=self.timeout_seconds,
        )
        result.exact_only = exact
        result.state = exact.status
        result.completed_at = perf_counter()
        result.error = exact.error
        result.timeline.append({"state": result.state, "at": result.completed_at})
        return result


def move_counts(alignment: Alignment | None) -> Counter:
    counts = Counter(sync=0, log=0, model=0, silent=0)
    if alignment is None:
        return counts
    for move in alignment.moves:
        if move.kind == MoveKind.SYNCHRONOUS:
            counts["sync"] += 1
        elif move.kind == MoveKind.LOG:
            counts["log"] += 1
        elif move.is_silent_model_move:
            counts["silent"] += 1
        else:
            counts["model"] += 1
    return counts


def result_row(result: TraceAlignmentResult) -> dict:
    candidate = result.candidate
    exact = result.exact_result
    counts = move_counts(result.candidate_alignment)
    candidate_cost = None if candidate is None else candidate.cost
    exact_cost = None if exact is None else exact.exact_cost
    return {
        "Variant": result.variant.variant_id,
        "Cases": result.variant.frequency,
        "Coverage": result.variant.coverage,
        "Length": result.variant.length,
        "Status": result.status,
        "Legal": None if candidate is None else candidate.legal,
        "Candidate cost": candidate_cost,
        "Exact cost": exact_cost,
        "Gap": (
            candidate_cost - exact_cost
            if candidate_cost is not None and exact_cost is not None
            else None
        ),
        "Certified": bool(exact and exact.certified_optimal),
        "Sync moves": counts["sync"],
        "Log moves": counts["log"],
        "Model moves": counts["model"],
        "Silent moves": counts["silent"],
        "Candidate time (s)": None if candidate is None else candidate.total_seconds,
        "Exact time (s)": None if exact is None else exact.exact_seconds,
        "Sequence": " → ".join(result.variant.labels),
    }


def result_rows(results: Iterable[TraceAlignmentResult]) -> list[dict]:
    return [result_row(result) for result in results]


def certification_subset(
    results: Iterable[TraceAlignmentResult],
    policy: str,
    limit: int | None = None,
    cost_threshold: int = 1,
    manual_ids: set[str] | None = None,
    duplicate_labels: set[str] | None = None,
) -> list[TraceAlignmentResult]:
    items = [result for result in results if result.candidate is not None]
    if policy == "All legal candidates":
        selected = [result for result in items if result.candidate and result.candidate.legal]
    elif policy == "Illegal candidates":
        selected = [result for result in items if result.candidate and not result.candidate.legal]
    elif policy == "Candidates with deviations":
        selected = [
            result
            for result in items
            if result.candidate and (result.candidate.cost or 0) > 0
        ]
    elif policy == "Cost at or above threshold":
        selected = [
            result
            for result in items
            if result.candidate
            and result.candidate.cost is not None
            and result.candidate.cost >= cost_threshold
        ]
    elif policy == "Most frequent variants":
        selected = sorted(items, key=lambda result: -result.variant.frequency)
    elif policy == "Longest variants":
        selected = sorted(items, key=lambda result: -result.variant.length)
    elif policy == "Duplicate-label variants":
        duplicate_labels = duplicate_labels or set()
        selected = [
            result
            for result in items
            if set(result.variant.labels) & duplicate_labels
        ]
    elif policy == "Manual selection":
        manual_ids = manual_ids or set()
        selected = [result for result in items if result.variant.variant_id in manual_ids]
    else:
        selected = items
    return selected if limit is None else selected[:limit]


def aggregate_metrics(results: Iterable[TraceAlignmentResult]) -> dict:
    items = list(results)
    candidates = [result for result in items if result.candidate is not None]
    exact = [result for result in items if result.exact_result is not None]
    total_cases = sum(result.variant.frequency for result in items) or 1
    candidate_cases = sum(result.variant.frequency for result in candidates)

    def average(values):
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None

    def weighted_average(pairs):
        pairs = [(value, weight) for value, weight in pairs if value is not None]
        denominator = sum(weight for _, weight in pairs)
        return (
            sum(value * weight for value, weight in pairs) / denominator
            if denominator
            else None
        )

    legal = [result for result in candidates if result.candidate and result.candidate.legal]
    certified = [result for result in exact if result.status == "certified"]
    repaired = [result for result in exact if result.status == "repaired"]
    timed_out = [result for result in exact if result.status == "exact_timeout"]
    candidate_costs = [result.candidate.cost for result in candidates if result.candidate]
    exact_costs = [result.exact_result.exact_cost for result in exact if result.exact_result]
    gaps = [
        result.candidate.cost - result.exact_result.exact_cost
        for result in exact
        if result.candidate
        and result.candidate.cost is not None
        and result.exact_result
        and result.exact_result.exact_cost is not None
    ]
    return {
        "variants_queued": len(items),
        "candidates_completed": len(candidates),
        "variants_certified": len(certified),
        "cases_covered": candidate_cases / total_cases,
        "candidate_legal_rate": len(legal) / len(candidates) if candidates else None,
        "certification_rate": len(certified) / len(exact) if exact else None,
        "repair_rate": len(repaired) / len(exact) if exact else None,
        "exact_timeout_rate": len(timed_out) / len(exact) if exact else None,
        "mean_candidate_cost": average(candidate_costs),
        "mean_exact_cost": average(exact_costs),
        "mean_cost_gap": average(gaps),
        "mean_candidate_runtime": average(
            [result.candidate.total_seconds for result in candidates if result.candidate]
        ),
        "mean_exact_runtime": average(
            [result.exact_result.exact_seconds for result in exact if result.exact_result]
        ),
        "case_weighted_candidate_cost": weighted_average(
            [
                (result.candidate.cost, result.variant.frequency)
                for result in candidates
                if result.candidate
            ]
        ),
        "case_weighted_exact_cost": weighted_average(
            [
                (result.exact_result.exact_cost, result.variant.frequency)
                for result in exact
                if result.exact_result
            ]
        ),
        "case_weighted_legal_rate": (
            sum(result.variant.frequency for result in legal) / candidate_cases
            if candidate_cases
            else None
        ),
    }
