from __future__ import annotations

from enum import Enum
from typing import Sequence

import torch
from pm4py.objects.petri_net.obj import Marking, PetriNet

from lara_align.decode import GreedyCandidateDecoder
from lara_align.exact import Pm4PyExactAligner
from lara_align.features import pm4py_to_features
from lara_align.model import LARANeuralModel
from lara_align.types import Alignment, CostModel, LARAResult
from lara_align.verify import verify_alignment


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
    ) -> None:
        self.model = model or LARANeuralModel()
        self.decoder = decoder or GreedyCandidateDecoder()
        self.exact_aligner = exact_aligner or Pm4PyExactAligner()
        self.cost_model = cost_model or CostModel()
        self.device = torch.device(device)
        self.model.to(self.device)

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
        candidate = self._neural_candidate(
            net,
            initial_marking,
            final_marking,
            trace,
            activity_key=activity_key,
        )
        candidate_verification = verify_alignment(
            candidate,
            net,
            initial_marking,
            final_marking,
            trace,
            self.cost_model,
            activity_key=activity_key,
        )
        if candidate_verification.legal:
            candidate.cost = candidate_verification.cost

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
                diagnostics={"candidate_source": candidate.source},
            )

        exact = self.exact_aligner.align_trace(
            net,
            initial_marking,
            final_marking,
            trace,
            self.cost_model,
            activity_key=activity_key,
            timeout_seconds=timeout_seconds,
        )

        if exact.alignment is None or exact.cost is None:
            return self._uncertified_result(
                selected_mode,
                candidate,
                candidate_verification,
                diagnostics=exact.diagnostics or {},
            )

        if candidate_verification.legal and candidate_verification.cost == exact.cost:
            candidate.metadata["certified_by"] = "pm4py_state_equation_a_star"
            return LARAResult(
                alignment=candidate,
                legal=True,
                cost=candidate_verification.cost,
                lower_bound=exact.cost,
                upper_bound=candidate_verification.cost,
                certified_optimal=True,
                mode=selected_mode.value,
                verifier=candidate_verification,
                exact_alignment=exact.alignment,
                diagnostics={
                    "candidate_source": candidate.source,
                    "exact_backend": exact.diagnostics or {},
                },
            )

        exact_verification = verify_alignment(
            exact.alignment,
            net,
            initial_marking,
            final_marking,
            trace,
            self.cost_model,
            activity_key=activity_key,
        )
        exact.alignment.cost = exact_verification.cost
        return LARAResult(
            alignment=exact.alignment,
            legal=exact_verification.legal,
            cost=exact_verification.cost if exact_verification.legal else exact.cost,
            lower_bound=exact.cost,
            upper_bound=(
                candidate_verification.cost
                if candidate_verification.legal
                else exact_verification.cost
            ),
            certified_optimal=exact.optimal and exact_verification.legal,
            mode=selected_mode.value,
            verifier=exact_verification,
            exact_alignment=exact.alignment,
            diagnostics={
                "candidate_legal": candidate_verification.legal,
                "candidate_cost": (
                    candidate_verification.cost if candidate_verification.legal else None
                ),
                "candidate_failure": candidate_verification.reason,
                "exact_backend": exact.diagnostics or {},
            },
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
        self.model.eval()
        with torch.no_grad():
            output = self.model(features)
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
            diagnostics={
                "candidate_source": candidate.source,
                "candidate_failure": candidate_verification.reason,
                **diagnostics,
            },
        )
