from __future__ import annotations

from enum import Enum
from time import perf_counter
from typing import Sequence

import torch
from pm4py.objects.petri_net.obj import Marking, PetriNet

from lara_align.decode import GreedyCandidateDecoder
from lara_align.exact import Pm4PyExactAligner
from lara_align.features import pm4py_to_features
from lara_align.model import LARANeuralModel
from lara_align.types import (
    Alignment,
    CandidateResult,
    CertificationResult,
    CostModel,
    LARAResult,
)
from lara_align.verify import trace_labels, verify_alignment


class LARAMode(str, Enum):
    FAST = "fast"
    CERTIFIED = "certified"
    ANYTIME = "anytime"


class CertifyingAlignmentSystem:
    """Neural-guided, certifying alignment system.

    The neural model proposes a candidate. The verifier decides legality. The
    exact pm4py backend is the only component allowed to certify optimality.
    """

    def __init__(
        self,
        model: LARANeuralModel | None = None,
        decoder: GreedyCandidateDecoder | None = None,
        exact_aligner: Pm4PyExactAligner | None = None,
        cost_model: CostModel | None = None,
        device: torch.device | str = "cpu",
        use_guidance: bool = True,
    ) -> None:
        self.model = model or LARANeuralModel()
        self.decoder = decoder or GreedyCandidateDecoder()
        self.exact_aligner = exact_aligner or Pm4PyExactAligner()
        self.cost_model = cost_model or CostModel()
        self.device = torch.device(device)
        self.model.to(self.device)
        # With use_guidance=False the decoder runs on structural heuristics
        # only (no neural forward pass); used for guidance ablations.
        self.use_guidance = use_guidance

    def align(
        self,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        trace: Sequence[object],
        mode: LARAMode | str = LARAMode.CERTIFIED,
        activity_key: str = "concept:name",
        timeout_seconds: float | None = None,
    ) -> LARAResult:
        selected_mode = LARAMode(mode)
        candidate_result = self.propose_trace(
            net,
            initial_marking,
            final_marking,
            trace,
            trace_key="trace",
            activity_key=activity_key,
        )
        candidate = candidate_result.alignment
        candidate_verification = candidate_result.verification

        if candidate is None or candidate_verification is None:
            return LARAResult(
                alignment=None,
                legal=False,
                cost=None,
                lower_bound=None,
                upper_bound=None,
                certified_optimal=False,
                mode=selected_mode.value,
                candidate_alignment=candidate,
                diagnostics={"candidate_failure": candidate_result.error},
            )

        if selected_mode == LARAMode.FAST:
            upper_bound = candidate_verification.cost if candidate_verification.legal else None
            return LARAResult(
                alignment=candidate,
                legal=candidate_verification.legal,
                cost=upper_bound,
                lower_bound=None,
                upper_bound=upper_bound,
                certified_optimal=False,
                mode=selected_mode.value,
                verifier=candidate_verification,
                candidate_alignment=candidate,
                diagnostics={"candidate_source": candidate.source},
            )

        certification = self.certify_trace(
            candidate_result,
            net,
            initial_marking,
            final_marking,
            trace,
            activity_key=activity_key,
            timeout_seconds=timeout_seconds,
        )

        if certification.exact_alignment is None or certification.exact_cost is None:
            return self._uncertified_result(
                selected_mode,
                candidate,
                candidate_verification,
                diagnostics={
                    **certification.exact_diagnostics,
                    "exact_status": certification.status,
                },
            )

        if certification.status == "certified":
            candidate.metadata["certified_by"] = "pm4py_state_equation_a_star"
            return LARAResult(
                alignment=candidate,
                legal=True,
                cost=candidate_verification.cost,
                lower_bound=certification.exact_cost,
                upper_bound=candidate_verification.cost,
                certified_optimal=True,
                mode=selected_mode.value,
                verifier=candidate_verification,
                candidate_alignment=candidate,
                exact_alignment=certification.exact_alignment,
                diagnostics={
                    "candidate_source": candidate.source,
                    "exact_backend": certification.exact_diagnostics,
                },
            )

        exact_verification = verify_alignment(
            certification.exact_alignment,
            net,
            initial_marking,
            final_marking,
            trace,
            self.cost_model,
            activity_key=activity_key,
        )
        certification.exact_alignment.cost = exact_verification.cost
        return LARAResult(
            alignment=certification.exact_alignment,
            legal=exact_verification.legal,
            cost=(
                exact_verification.cost
                if exact_verification.legal
                else certification.exact_cost
            ),
            lower_bound=certification.exact_cost,
            upper_bound=(
                candidate_verification.cost
                if candidate_verification.legal
                else exact_verification.cost
            ),
            certified_optimal=(
                certification.certified_optimal and exact_verification.legal
            ),
            mode=selected_mode.value,
            verifier=exact_verification,
            candidate_alignment=candidate,
            exact_alignment=certification.exact_alignment,
            diagnostics={
                "candidate_legal": candidate_verification.legal,
                "candidate_cost": (
                    candidate_verification.cost if candidate_verification.legal else None
                ),
                "candidate_failure": candidate_verification.reason,
                "exact_backend": certification.exact_diagnostics,
            },
        )

    def propose_trace(
        self,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        trace: Sequence[object],
        trace_key: str = "trace",
        activity_key: str = "concept:name",
        retain_full_diagnostics: bool = False,
    ) -> CandidateResult:
        """Generate and independently verify a candidate without exact search."""

        labels = trace_labels(trace, activity_key=activity_key)
        feature_seconds = inference_seconds = decoding_seconds = 0.0
        try:
            started = perf_counter()
            features = pm4py_to_features(
                net,
                initial_marking,
                final_marking,
                trace,
                cost_model=self.cost_model,
                activity_key=activity_key,
            ).to(self.device)
            feature_seconds = perf_counter() - started

            output = None
            if self.use_guidance:
                self.model.eval()
                started = perf_counter()
                with torch.inference_mode():
                    output = self.model(features)
                inference_seconds = perf_counter() - started

            started = perf_counter()
            alignment = self.decoder.decode(
                net,
                initial_marking,
                final_marking,
                trace,
                features,
                output,
                self.cost_model,
                activity_key=activity_key,
            )
            decoding_seconds = perf_counter() - started

            started = perf_counter()
            verification = verify_alignment(
                alignment,
                net,
                initial_marking,
                final_marking,
                trace,
                self.cost_model,
                activity_key=activity_key,
            )
            verification_seconds = perf_counter() - started
            alignment.cost = verification.cost

            diagnostics = self._neural_diagnostics(
                features,
                output,
                alignment,
                retain_full=retain_full_diagnostics,
            )
            diagnostics["decoder"] = {
                "guided": self.use_guidance,
                "max_prefix_model_depth": self.decoder.max_prefix_model_depth,
                "max_final_model_depth": self.decoder.max_final_model_depth,
            }
            return CandidateResult(
                trace_key=trace_key,
                labels=labels,
                alignment=alignment,
                verification=verification,
                cost=verification.cost if verification.legal else None,
                feature_seconds=feature_seconds,
                inference_seconds=inference_seconds,
                decoding_seconds=decoding_seconds,
                verification_seconds=verification_seconds,
                neural_diagnostics=diagnostics,
            )
        except Exception as exc:
            return CandidateResult(
                trace_key=trace_key,
                labels=labels,
                alignment=None,
                verification=None,
                cost=None,
                feature_seconds=feature_seconds,
                inference_seconds=inference_seconds,
                decoding_seconds=decoding_seconds,
                error=f"{type(exc).__name__}: {exc}",
            )

    def certify_trace(
        self,
        candidate: CandidateResult | None,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        trace: Sequence[object],
        activity_key: str = "concept:name",
        timeout_seconds: float | None = None,
    ) -> CertificationResult:
        """Run exact alignment and classify certification or repair outcome."""

        started = perf_counter()
        try:
            exact = self.exact_aligner.align_trace(
                net,
                initial_marking,
                final_marking,
                trace,
                self.cost_model,
                activity_key=activity_key,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            elapsed = perf_counter() - started
            timed_out = isinstance(exc, TimeoutError)
            return CertificationResult(
                candidate=candidate,
                exact_alignment=None,
                exact_cost=None,
                exact_seconds=elapsed,
                status="exact_timeout" if timed_out else "exact_failure",
                certified_optimal=False,
                final_alignment_source=(
                    "candidate" if candidate is not None and candidate.legal else "none"
                ),
                timed_out=timed_out,
                error=f"{type(exc).__name__}: {exc}",
                exact_diagnostics={"backend": "pm4py_state_equation_a_star"},
            )

        elapsed = perf_counter() - started
        diagnostics = dict(exact.diagnostics or {})
        if exact.alignment is None or exact.cost is None:
            timed_out = exact.timed_out
            return CertificationResult(
                candidate=candidate,
                exact_alignment=exact.alignment,
                exact_cost=exact.cost,
                exact_seconds=elapsed,
                status="exact_timeout" if timed_out else "exact_failure",
                certified_optimal=False,
                final_alignment_source=(
                    "candidate" if candidate is not None and candidate.legal else "none"
                ),
                timed_out=timed_out,
                error=exact.error or diagnostics.get("reason"),
                exact_diagnostics=diagnostics,
            )

        if candidate is None:
            return CertificationResult(
                candidate=None,
                exact_alignment=exact.alignment,
                exact_cost=exact.cost,
                exact_seconds=elapsed,
                status="exact_only",
                certified_optimal=exact.optimal,
                final_alignment_source="exact_only",
                exact_diagnostics=diagnostics,
            )

        gap = candidate.cost - exact.cost if candidate.cost is not None else None
        if candidate.legal and candidate.cost == exact.cost:
            status = "certified"
            source = "candidate"
            certified = exact.optimal
        elif candidate.legal:
            status = "repaired"
            source = "exact_repair"
            certified = exact.optimal
        else:
            status = "illegal_candidate_exact_available"
            source = "exact_repair"
            certified = exact.optimal
        return CertificationResult(
            candidate=candidate,
            exact_alignment=exact.alignment,
            exact_cost=exact.cost,
            exact_seconds=elapsed,
            status=status,
            certified_optimal=certified,
            final_alignment_source=source,
            cost_gap=gap,
            exact_diagnostics=diagnostics,
        )

    def _neural_candidate(
        self,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        trace: Sequence[object],
        activity_key: str,
    ) -> Alignment:
        features = pm4py_to_features(
            net,
            initial_marking,
            final_marking,
            trace,
            cost_model=self.cost_model,
            activity_key=activity_key,
        ).to(self.device)
        if self.use_guidance:
            self.model.eval()
            with torch.inference_mode():
                output = self.model(features)
        else:
            output = None
        return self.decoder.decode(
            net,
            initial_marking,
            final_marking,
            trace,
            features,
            output,
            self.cost_model,
            activity_key=activity_key,
        )

    @staticmethod
    def _neural_diagnostics(
        features,
        output,
        alignment: Alignment,
        retain_full: bool,
    ) -> dict:
        """Convert decoder-relevant tensors to bounded, JSON-friendly values."""

        if output is None:
            return {"guided": False}
        transition_names = list(features.transition_names)
        transition_labels = list(features.transition_labels)
        selected_by_event = [
            move.transition_name
            for move in alignment.moves
            if move.log_label is not None
        ]
        sync = output.sync_logits.detach().cpu()
        sync_rows = []
        for event_index, event_label in enumerate(features.event_labels):
            compatible = []
            for transition_index, transition_label in enumerate(transition_labels):
                if transition_label != event_label:
                    continue
                compatible.append(
                    {
                        "transition_id": transition_names[transition_index],
                        "transition_label": transition_label,
                        "score": float(sync[event_index, transition_index]),
                    }
                )
            compatible.sort(key=lambda item: item["score"], reverse=True)
            for rank, item in enumerate(compatible, start=1):
                item["rank"] = rank
                item["selected"] = (
                    event_index < len(selected_by_event)
                    and item["transition_id"] == selected_by_event[event_index]
                )
            sync_rows.append(
                {
                    "event_index": event_index,
                    "observed_label": event_label,
                    "transitions": compatible if retain_full else compatible[:10],
                }
            )

        model_scores = [
            {
                "transition_id": name,
                "transition_label": label,
                "score": float(score),
            }
            for name, label, score in zip(
                transition_names,
                transition_labels,
                output.model_move_logits.detach().cpu().tolist(),
            )
        ]
        model_scores.sort(key=lambda item: item["score"], reverse=True)
        diagnostics = {
            "guided": True,
            "sync_scores": sync_rows,
            "model_move_scores": model_scores if retain_full else model_scores[:20],
            "log_move_scores": output.log_move_logits.detach().cpu().tolist(),
            "event_region_probs": output.event_region_probs.detach().cpu().tolist(),
            "transition_region_probs": (
                output.transition_region_probs.detach().cpu().tolist()
                if retain_full
                else []
            ),
            "transition_names": transition_names if retain_full else [],
            "experimental_warning": (
                "Internal local bounds, uncertainty, and sketch heads are not "
                "calibrated explanations."
            ),
        }
        return diagnostics

    def _uncertified_result(
        self,
        mode: LARAMode,
        candidate: Alignment,
        candidate_verification,
        diagnostics: dict,
    ) -> LARAResult:
        if candidate_verification.legal:
            return LARAResult(
                alignment=candidate,
                legal=True,
                cost=candidate_verification.cost,
                lower_bound=0,
                upper_bound=candidate_verification.cost,
                certified_optimal=False,
                mode=mode.value,
                verifier=candidate_verification,
                candidate_alignment=candidate,
                diagnostics={
                    "candidate_source": candidate.source,
                    "lower_bound_source": "trivial",
                    **diagnostics,
                },
            )
        return LARAResult(
            alignment=candidate,
            legal=False,
            cost=None,
            lower_bound=0,
            upper_bound=None,
            certified_optimal=False,
            mode=mode.value,
            verifier=candidate_verification,
            candidate_alignment=candidate,
            diagnostics={
                "candidate_source": candidate.source,
                "candidate_failure": candidate_verification.reason,
                **diagnostics,
            },
        )
