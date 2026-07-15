from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from pm4py.objects.log.obj import Event, Trace

from lara_align.certifier import CertifyingAlignmentSystem
from lara_align.exact import ExactAlignmentResult
from lara_align.model import LARANeuralModel
from lara_align.synthetic import make_sequence_net, trace_from_labels
from lara_align.types import Alignment, AlignmentMove, CostModel
from lara_ui.alignment_runner import WorkbenchRunner
from lara_ui.export_service import case_csv, full_json, variant_csv
from lara_ui.input_service import (
    ParsedLog,
    discover_petri_net_inductive,
    parse_pnml_bytes,
    parse_xes_bytes,
)
from lara_ui.replay_service import alignment_differences, anchored_alignment
from lara_ui.ui_types import TraceAlignmentResult, TraceVariant
from lara_ui.variant_service import (
    MAX_LIVE_VARIANTS,
    default_selected_variant_ids,
    group_trace_variants,
    limit_live_variants,
    order_variants,
)


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


def test_setup_order_and_frequency_based_default_selection_are_capped_at_500():
    variants = [
        TraceVariant(
            variant_id=f"V{index:03d}",
            labels=[str(index)],
            case_ids=[],
            first_index=index,
            frequency=700 - index,
            coverage=0.0,
        )
        for index in range(600)
    ]

    ordered = order_variants(reversed(variants), "Most frequent variants first")
    selected = default_selected_variant_ids(ordered)

    assert [variant.frequency for variant in ordered] == sorted(
        (variant.frequency for variant in variants), reverse=True
    )
    assert len(selected) == MAX_LIVE_VARIANTS
    assert all(
        next(variant for variant in variants if variant.variant_id == variant_id).frequency
        >= 10
        for variant_id in selected
    )
    assert len(limit_live_variants(ordered, 10_000)) == MAX_LIVE_VARIANTS

    boundary_variants = [
        TraceVariant(f"B{frequency}", [str(frequency)], [], frequency, frequency, 0.0)
        for frequency in [9, 10, 11]
    ]
    assert default_selected_variant_ids(boundary_variants) == ["B11", "B10"]


@pytest.mark.parametrize(
    ("filename", "expected_cases"),
    [
        ("running-example.xes", 6),
        ("receipt.xes", 1434),
        ("roadtraffic100traces.xes", 100),
    ],
)
def test_every_bundled_xes_can_be_loaded_and_automatically_discovered(
    filename, expected_cases
):
    data = (Path("files") / filename).read_bytes()

    parsed_log = parse_xes_bytes(data, filename)
    parsed_net = discover_petri_net_inductive(parsed_log, noise_threshold=0.0)

    assert len(parsed_log.traces) == expected_cases
    assert parsed_net.source_kind == "discovered_from_event_log"
    assert parsed_net.source_name == f"Discovered from {filename}"
    assert parsed_net.discovery_algorithm == "Inductive Miner (IM)"
    assert parsed_net.discovery_noise_threshold == 0.0
    assert parsed_net.net.places
    assert parsed_net.net.transitions
    assert sum(parsed_net.initial_marking.values()) > 0
    assert sum(parsed_net.final_marking.values()) > 0


def test_positive_noise_threshold_selects_inductive_miner_infrequent():
    data = Path("files/running-example.xes").read_bytes()
    parsed_log = parse_xes_bytes(data, "user-upload.xes")

    parsed_net = discover_petri_net_inductive(
        parsed_log,
        noise_threshold=0.2,
        disable_fallthroughs=True,
    )

    assert parsed_net.discovery_algorithm == "Inductive Miner - infrequent (IMf)"
    assert parsed_net.discovery_noise_threshold == 0.2
    assert parsed_net.discovery_disable_fallthroughs is True


def test_compressed_xes_upload_is_supported():
    compressed = gzip.compress(Path("files/running-example.xes").read_bytes())

    parsed_log = parse_xes_bytes(compressed, "user-upload.xes.gz")

    assert len(parsed_log.traces) == 6
    assert parsed_log.source_name == "user-upload.xes.gz"


def test_discovery_rejects_invalid_noise_thresholds():
    parsed_log = parse_xes_bytes(
        Path("files/running-example.xes").read_bytes(), "upload.xes"
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        discover_petri_net_inductive(parsed_log, noise_threshold=1.1)


def test_uploaded_pnml_is_identified_as_an_independent_reference_model():
    parsed_net = parse_pnml_bytes(
        Path("files/running-example.pnml").read_bytes(),
        "user-model.pnml",
        source_kind="uploaded_pnml",
    )

    assert parsed_net.source_kind == "uploaded_pnml"
    assert parsed_net.source_name == "user-model.pnml"
    assert parsed_net.discovery_algorithm is None
    assert parsed_net.initial_marking
    assert parsed_net.final_marking


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
