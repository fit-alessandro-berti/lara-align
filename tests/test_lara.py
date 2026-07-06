from pm4py.objects.log.obj import Event, Trace

from lara_align import Alignment, AlignmentMove, CertifyingAlignmentSystem, LARAMode
from lara_align.data import (
    AlignmentSample,
    DatasetMetadata,
    load_metadata,
    load_split,
    save_metadata,
    save_split,
)
from lara_align.decode import GreedyCandidateDecoder
from lara_align.exact import Pm4PyExactAligner
from lara_align.features import pm4py_to_features
from lara_align.model import LARANeuralModel
from lara_align.synthetic import (
    make_duplicate_label_choice_net,
    make_sequence_net,
    trace_from_labels,
)
from lara_align.training import LARALoss, targets_from_alignment
from lara_align.verify import verify_alignment
from scripts.test_model import EvaluationRecord, format_human_report


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


def test_verifier_accepts_legal_sequence_alignment():
    net, im, fm = make_sequence_net(["A", "B"])
    trace = trace_from_labels(["A", "B"])
    alignment = Alignment(
        [
            AlignmentMove("A", "t0_A", "A"),
            AlignmentMove("B", "t1_B", "B"),
        ]
    )

    result = verify_alignment(alignment, net, im, fm, trace)

    assert result.legal
    assert result.cost == 0


def test_exact_backend_repairs_inserted_log_event():
    net, im, fm = make_sequence_net(["A", "B"])
    trace = trace_from_labels(["A", "X", "B"])

    result = Pm4PyExactAligner().align_trace(net, im, fm, trace)

    assert result.optimal
    assert result.cost == 1
    assert result.alignment is not None
    assert result.alignment.to_pm4py_label_alignment() == [
        ("A", "A"),
        ("X", ">>"),
        ("B", "B"),
    ]


def test_exact_backend_preserves_duplicate_transition_identity():
    net, im, fm = make_duplicate_label_choice_net()
    trace = trace_from_labels(["A", "C"])

    result = Pm4PyExactAligner().align_trace(net, im, fm, trace)

    assert result.optimal
    assert result.alignment is not None
    sync_transition_names = [
        move.transition_name for move in result.alignment.moves if move.log_label is not None
    ]
    assert sync_transition_names == ["t_right_A", "t_right_C"]


def test_neural_model_forward_shapes():
    net, im, fm = make_sequence_net(["A", "B", "C"], silent_prefix=True)
    trace = trace_from_labels(["A", "B", "C"])
    features = pm4py_to_features(net, im, fm, trace)
    model = _small_model()

    output = model(features)

    assert output.sync_logits.shape == (3, features.num_transitions)
    assert output.transition_region_probs.shape == (features.num_transitions, 2)
    assert output.event_region_probs.shape == (3, 2)
    assert output.local_sketch_logits.shape == (2, 2)


def test_certifying_system_returns_legal_certified_alignment():
    net, im, fm = make_sequence_net(["A", "B"])
    trace = Trace(
        [
            Event({"concept:name": "A"}),
            Event({"concept:name": "X"}),
            Event({"concept:name": "B"}),
        ]
    )
    lara = CertifyingAlignmentSystem(model=_small_model())

    result = lara.align(net, im, fm, trace, mode=LARAMode.CERTIFIED)

    assert result.legal
    assert result.certified_optimal
    assert result.cost == 1
    assert result.lower_bound == 1
    assert result.upper_bound in {1, None} or result.upper_bound >= 1


def test_training_loss_smoke():
    net, im, fm = make_sequence_net(["A", "B"])
    trace = trace_from_labels(["A", "B"])
    exact = Pm4PyExactAligner().align_trace(net, im, fm, trace)
    features = pm4py_to_features(net, im, fm, trace)
    model = _small_model()
    output = model(features)
    targets = targets_from_alignment(exact.alignment, features, optimal_cost=exact.cost)

    losses = LARALoss()(output, features, targets)

    assert losses["total"].isfinite()
    assert losses["total"].ndim == 0


def test_fast_mode_uses_neural_decoder_and_verifier():
    net, im, fm = make_sequence_net(["A", "B"])
    trace = trace_from_labels(["A", "B"])
    lara = CertifyingAlignmentSystem(
        model=_small_model(),
        decoder=GreedyCandidateDecoder(max_prefix_model_depth=2, max_final_model_depth=2),
    )

    result = lara.align(net, im, fm, trace, mode=LARAMode.FAST)

    assert result.legal
    assert result.cost == 0
    assert not result.certified_optimal


def test_dataset_split_roundtrip(tmp_path):
    net, im, fm = make_sequence_net(["A", "B"])
    trace = trace_from_labels(["A", "B"])
    exact = Pm4PyExactAligner().align_trace(net, im, fm, trace)
    sample = AlignmentSample(
        sample_id="train-000000",
        split="train",
        net=net,
        initial_marking=im,
        final_marking=fm,
        trace=trace,
        optimal_alignment=exact.alignment,
        optimal_cost=exact.cost,
        metadata={"family": "sequence"},
    )

    save_split(tmp_path, "train", [sample])
    save_metadata(
        tmp_path,
        DatasetMetadata(
            version=1,
            split_counts={"train": 1, "val": 0, "test": 0},
            seed=7,
            generator="test",
        ),
    )

    loaded = load_split(tmp_path, "train")
    metadata = load_metadata(tmp_path)

    assert metadata.seed == 7
    assert loaded[0].sample_id == "train-000000"
    assert loaded[0].optimal_cost == 0
    assert verify_alignment(
        loaded[0].optimal_alignment,
        loaded[0].net,
        loaded[0].initial_marking,
        loaded[0].final_marking,
        loaded[0].trace,
    ).legal


def test_human_evaluation_report_contains_alignment_examples():
    optimal = Alignment(
        [
            AlignmentMove("A", "t0_A", "A"),
            AlignmentMove("B", "t1_B", "B"),
        ],
        cost=0,
        source="pm4py_exact",
    )
    predicted = Alignment(
        [
            AlignmentMove("A", "t0_A", "A"),
            AlignmentMove("B", "t1_B", "B"),
        ],
        cost=0,
        source="neural_greedy",
    )
    record = EvaluationRecord(
        sample_id="test-000000",
        family="sequence",
        trace=["A", "B"],
        optimal_alignment=optimal,
        predicted_alignment=predicted,
        optimal_cost=0,
        predicted_cost=0,
        legal=True,
        failure_reason=None,
        exact_alignment_match=True,
        label_alignment_match=True,
        move_count_gap=0,
    )
    metrics = {
        "split": "test",
        "samples": 1,
        "checkpoint": "runs/lara/best.pt",
        "checkpoint_epoch": 1,
        "elapsed_seconds": 0.1,
        "legal_rate": 1.0,
        "optimal_cost_rate": 1.0,
        "exact_alignment_match_rate": 1.0,
        "label_alignment_match_rate": 1.0,
        "certified_optimal_rate": None,
        "legal_gap_count": 1,
        "gap_mean": 0.0,
        "gap_median": 0.0,
        "gap_min": 0,
        "gap_max": 0,
        "gap_p90": 0.0,
        "gap_p95": 0.0,
        "gap_std": 0.0,
        "gap_relative_mean": 0.0,
        "loss_total": 0.0,
        "by_family": {
            "sequence": {
                "samples": 1,
                "legal_rate": 1.0,
                "optimal_cost_rate": 1.0,
                "gap_mean": 0.0,
            }
        },
    }

    class Args:
        num_examples = 1
        example_selection = "first"
        max_moves = 10

    report = format_human_report(metrics, [record], Args())

    assert "Replay And Optimality" in report
    assert "pm4py optimal:" in report
    assert "LARA reconstructed:" in report
    assert "replayable alignments:" in report
    assert "equal optimum cost:" in report
