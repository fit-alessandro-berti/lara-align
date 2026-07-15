from __future__ import annotations

from pm4py.objects.log.obj import Event, Trace

from lara_align.certifier import CertifyingAlignmentSystem
from lara_align.exact import ExactAlignmentResult
from lara_align.model import LARANeuralModel
from lara_align.synthetic import make_sequence_net, trace_from_labels
from lara_align.types import Alignment, AlignmentMove, CostModel
from lara_ui.alignment_runner import WorkbenchRunner
from lara_ui.export_service import case_csv, full_json, variant_csv
from lara_ui.input_service import ParsedLog
from lara_ui.replay_service import alignment_differences, anchored_alignment
from lara_ui.ui_types import TraceAlignmentResult, TraceVariant
from lara_ui.variant_service import group_trace_variants


def _small_model() -> LARANeuralModel:
    return LARANeuralModel(
        hidden_dim=32,
        num_heads=4,
        graph_layers=1,
        trace_layers=1,
        num_regions=2,
        sketches_per_region=2,
        dropout=0.0,
    )


class DeliberatelyExpensiveDecoder:
    max_prefix_model_depth = 0
    max_final_model_depth = 0

    def decode(self, net, initial_marking, final_marking, trace, features, output, cost_model, activity_key):
        # A legal but needlessly expensive alignment: skip both observed events,
        # then execute the model as model-only moves.
        return Alignment(
            [
                AlignmentMove("A", None),
                AlignmentMove("B", None),
                AlignmentMove(None, "t0_A", "A"),
                AlignmentMove(None, "t1_B", "B"),
            ],
            source="test_expensive_candidate",
        )


class TimeoutAligner:
    def align_trace(self, *args, **kwargs):
        return ExactAlignmentResult(
            alignment=None,
            cost=None,
            optimal=False,
            diagnostics={"reason": "exact alignment timed out"},
            timed_out=True,
        )


def test_staged_candidate_is_available_before_certification_and_preserved_after_repair():
    net, im, fm = make_sequence_net(["A", "B"])
    trace = trace_from_labels(["A", "B"])
    system = CertifyingAlignmentSystem(
        model=_small_model(),
        decoder=DeliberatelyExpensiveDecoder(),
    )

    candidate = system.propose_trace(net, im, fm, trace, trace_key="V001")

    assert candidate.legal
    assert candidate.cost == 4
    assert candidate.alignment.source == "test_expensive_candidate"
    assert candidate.feature_seconds >= 0
    assert candidate.inference_seconds >= 0
    assert candidate.decoding_seconds >= 0
    assert candidate.verification_seconds >= 0

    certified = system.certify_trace(candidate, net, im, fm, trace)

    assert certified.status == "repaired"
    assert certified.exact_cost == 0
    assert certified.cost_gap == 4
    assert certified.candidate is candidate
    assert certified.candidate_alignment is candidate.alignment
    assert certified.exact_alignment is not candidate.alignment
    assert certified.final_alignment_source == "exact_repair"

    legacy = system.align(net, im, fm, trace, mode="certified")
    assert legacy.certified_optimal
    assert legacy.cost == 0
    assert legacy.candidate_alignment is not None
    assert legacy.candidate_alignment.source == "test_expensive_candidate"
    assert legacy.exact_alignment is legacy.alignment


def test_exact_timeout_is_not_reported_as_failure_or_infeasibility():
    net, im, fm = make_sequence_net(["A"])
    trace = trace_from_labels(["A"])
    system = CertifyingAlignmentSystem(
        model=_small_model(),
        exact_aligner=TimeoutAligner(),
    )
    candidate = system.propose_trace(net, im, fm, trace, trace_key="V001")

    result = system.certify_trace(candidate, net, im, fm, trace, timeout_seconds=0.1)

    assert result.status == "exact_timeout"
    assert result.timed_out
    assert result.exact_cost is None
    assert result.final_alignment_source == "candidate"
    assert result.error == "exact alignment timed out"


def test_variant_grouping_retains_complete_case_mapping():
    traces = [
        Trace([Event({"concept:name": "A"}), Event({"concept:name": "B"})]),
        Trace([Event({"concept:name": "A"}), Event({"concept:name": "B"})]),
        Trace([Event({"concept:name": "A"}), Event({"concept:name": "X"})]),
    ]
    parsed = ParsedLog(
        traces=traces,
        case_ids=["case-1", "case-2", "case-3"],
        activity_key="concept:name",
        case_id_key="concept:name",
        source_name="test.xes",
        empty_trace_count=0,
    )

    variants = group_trace_variants(parsed)

    assert [variant.variant_id for variant in variants] == ["V001", "V002"]
    assert variants[0].labels == ["A", "B"]
    assert variants[0].case_ids == ["case-1", "case-2"]
    assert variants[0].frequency == 2
    assert variants[0].coverage == 2 / 3


def test_alignment_comparison_is_anchored_to_events_not_raw_move_indices():
    candidate = Alignment(
        [
            AlignmentMove(None, "tau-a", None),
            AlignmentMove("A", "t-a", "A"),
            AlignmentMove(None, "tau-b", None),
            AlignmentMove("B", "t-b", "B"),
        ]
    )
    exact = Alignment(
        [
            AlignmentMove("A", "t-a", "A"),
            AlignmentMove("B", "t-b-alt", "B"),
            AlignmentMove(None, "tau-end", None),
        ]
    )

    slots = anchored_alignment(candidate, 2)
    differences = alignment_differences(candidate, exact, 2)

    assert slots[0]["before"][0].transition_name == "tau-a"
    assert slots[0]["consume"].transition_name == "t-a"
    assert slots[1]["before"][0].transition_name == "tau-b"
    assert slots[1]["consume"].transition_name == "t-b"
    assert differences == [0, 1, 2]


def test_variant_and_case_exports_preserve_provenance_and_transition_identity():
    net, im, fm = make_sequence_net(["A"])
    variant = TraceVariant("V001", ["A"], ["c1", "c2"], 0, 2, 1.0)
    runner = WorkbenchRunner(
        CertifyingAlignmentSystem(model=_small_model()), net, im, fm
    )
    result = TraceAlignmentResult(variant)
    runner.propose(result)
    runner.certify(result)

    variant_text = variant_csv([result]).decode()
    case_text = case_csv([result]).decode()
    json_text = full_json(
        [result], {}, {}, {}, {}, CostModel()
    ).decode()

    assert "Candidate validity" in variant_text
    assert "c1,V001" in case_text
    assert "c2,V001" in case_text
    assert '"transition_id": "t0_A"' in json_text
    assert '"final_alignment_source": "candidate"' in json_text
