from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pm4py.objects.log.obj import Event, Trace

from lara_align.types import CandidateResult, CertificationResult


@dataclass
class TraceVariant:
    variant_id: str
    labels: list[str]
    case_ids: list[str]
    first_index: int
    frequency: int
    coverage: float
    selected: bool = True

    @property
    def length(self) -> int:
        return len(self.labels)

    @property
    def trace(self) -> Trace:
        """Materialize a PM4Py trace while keeping the variant immutable."""

        return Trace(
            [Event({"concept:name": label}) for label in self.labels],
            attributes={"concept:name": self.variant_id},
        )


@dataclass
class TraceAlignmentResult:
    variant: TraceVariant
    state: str = "queued"
    candidate: CandidateResult | None = None
    certification: CertificationResult | None = None
    exact_only: CertificationResult | None = None
    started_at: float | None = None
    candidate_completed_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    timeline: list[dict[str, Any]] = field(default_factory=list)

    @property
    def exact_result(self) -> CertificationResult | None:
        return self.certification or self.exact_only

    @property
    def candidate_alignment(self):
        return None if self.candidate is None else self.candidate.alignment

    @property
    def exact_alignment(self):
        exact = self.exact_result
        return None if exact is None else exact.exact_alignment

    @property
    def status(self) -> str:
        if self.state == "cancelled":
            return "cancelled"
        exact = self.exact_result
        if exact is not None:
            return exact.status
        if self.candidate is None:
            return self.state
        if self.candidate.error:
            return "candidate_failure"
        if not self.candidate.legal:
            return "illegal_candidate"
        return "candidate_ready"

    @property
    def final_alignment_source(self) -> str:
        exact = self.exact_result
        if exact is not None:
            return exact.final_alignment_source
        return "candidate" if self.candidate and self.candidate.legal else "none"
