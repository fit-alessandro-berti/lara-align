from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.petri_net.obj import Marking, PetriNet

from lara_align.types import SKIP, Alignment, AlignmentMove, CostModel
from lara_align.verify import trace_labels, verify_alignment


@dataclass
class ExactAlignmentResult:
    alignment: Alignment | None
    cost: int | None
    optimal: bool
    raw: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None


class Pm4PyExactAligner:
    """Transition-aware exact alignment backend using pm4py."""

    def align_trace(
        self,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        trace: Sequence[object],
        cost_model: CostModel | None = None,
        activity_key: str = "concept:name",
        timeout_seconds: float | None = None,
    ) -> ExactAlignmentResult:
        costs = cost_model or CostModel()
        parameters = self._parameters(net, trace, costs, activity_key, timeout_seconds)
        raw = alignments.apply_trace(
            trace,
            net,
            initial_marking,
            final_marking,
            parameters=parameters,
            variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR,
        )
        if raw is None:
            return ExactAlignmentResult(
                alignment=None,
                cost=None,
                optimal=False,
                raw=None,
                diagnostics={"reason": "pm4py returned no alignment"},
            )

        parsed = parse_pm4py_alignment(raw, net)
        verified = verify_alignment(
            parsed,
            net,
            initial_marking,
            final_marking,
            trace,
            costs,
            activity_key=activity_key,
        )
        if not verified.legal:
            return ExactAlignmentResult(
                alignment=parsed,
                cost=raw.get("cost"),
                optimal=False,
                raw=raw,
                diagnostics={"reason": verified.reason},
            )

        parsed.cost = verified.cost
        parsed.metadata.update(
            {
                "pm4py_cost": raw.get("cost"),
                "visited_states": raw.get("visited_states"),
                "queued_states": raw.get("queued_states"),
                "traversed_arcs": raw.get("traversed_arcs"),
                "lp_solved": raw.get("lp_solved"),
            }
        )
        return ExactAlignmentResult(
            alignment=parsed,
            cost=verified.cost,
            optimal=True,
            raw=raw,
            diagnostics=dict(parsed.metadata),
        )

    def _parameters(
        self,
        net: PetriNet,
        trace: Sequence[object],
        cost_model: CostModel,
        activity_key: str,
        timeout_seconds: float | None,
    ) -> dict[Any, Any]:
        trace_costs = list(cost_model.trace_costs) if cost_model.trace_costs else None
        if trace_costs is None:
            trace_costs = [cost_model.log for _ in trace_labels(trace, activity_key)]

        params: dict[Any, Any] = {
            alignments.Parameters.ACTIVITY_KEY: activity_key,
            alignments.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True,
            alignments.Parameters.PARAM_TRACE_COST_FUNCTION: trace_costs,
            alignments.Parameters.PARAM_MODEL_COST_FUNCTION: {
                transition: cost_model.model_cost(
                    str(transition.name),
                    None if transition.label is None else str(transition.label),
                )
                for transition in net.transitions
            },
            alignments.Parameters.PARAM_SYNC_COST_FUNCTION: {
                transition: cost_model.sync_cost(str(transition.name))
                for transition in net.transitions
            },
            alignments.Parameters.ENABLE_BEST_WORST_COST: False,
        }
        if timeout_seconds is not None:
            params[alignments.Parameters.PARAM_MAX_ALIGN_TIME_TRACE] = timeout_seconds
        return params


def parse_pm4py_alignment(raw: dict[str, Any], net: PetriNet) -> Alignment:
    transitions = {str(transition.name): transition for transition in net.transitions}
    moves: list[AlignmentMove] = []

    for item in raw.get("alignment", []):
        log_label: str | None
        transition_name: str | None
        transition_label: str | None

        if _is_transition_aware_item(item):
            (log_transition_name, model_transition_name), (left_label, right_label) = item
            log_label = None if left_label == SKIP else str(left_label)
            transition_name = None if model_transition_name == SKIP else str(model_transition_name)
            if transition_name is None:
                transition_label = None
            else:
                transition = transitions[transition_name]
                transition_label = None if transition.label is None else str(transition.label)
                if right_label != transition.label:
                    transition_label = None if right_label is None else str(right_label)
            _ = log_transition_name
        else:
            left_label, right_label = item
            log_label = None if left_label == SKIP else str(left_label)
            if right_label == SKIP:
                transition_name = None
                transition_label = None
            else:
                transition_label = None if right_label is None else str(right_label)
                transition_name = _find_transition_name(transitions, transition_label)

        moves.append(
            AlignmentMove(
                log_label=log_label,
                transition_name=transition_name,
                transition_label=transition_label,
            )
        )

    return Alignment(moves=moves, cost=raw.get("cost"), source="pm4py_exact")


def _is_transition_aware_item(item: object) -> bool:
    return (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], tuple)
        and len(item[0]) == 2
        and isinstance(item[1], tuple)
        and len(item[1]) == 2
    )


def _find_transition_name(
    transitions: dict[str, PetriNet.Transition], transition_label: str | None
) -> str | None:
    candidates = [
        name
        for name, transition in transitions.items()
        if (None if transition.label is None else str(transition.label)) == transition_label
    ]
    if not candidates:
        return None
    return sorted(candidates)[0]
