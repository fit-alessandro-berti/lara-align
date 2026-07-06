import csv

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
    generate_block_structured_example,
    make_duplicate_label_choice_net,
    make_sequence_net,
    trace_from_labels,
)
from lara_align.training import LARALoss, targets_from_alignment
from lara_align.verify import verify_alignment
from scripts.test_model import EvaluationRecord, format_human_report
from scripts.train_model import _metrics_csv_row, _write_metrics_csv_row


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


def test_duplicate_transitions_get_distinct_embeddings():
    """Reverse typed edges must break the preset symmetry of duplicate labels.

    Before bidirectional message passing, `t_left_A` and `t_right_A` received
    provably identical embeddings, so the sync cross-entropy on duplicate
    choices was frozen at log(2) regardless of training.
    """

    net, im, fm = make_duplicate_label_choice_net()
    trace = trace_from_labels(["A", "C"])
    features = pm4py_to_features(net, im, fm, trace)
    # Two graph layers are the minimum for the disambiguating suffix context
    # to reach the duplicate transitions (suffix -> shared place -> duplicate).
    model = LARANeuralModel(
        hidden_dim=32,
        num_heads=4,
        graph_layers=2,
        trace_layers=1,
        num_regions=2,
        sketches_per_region=2,
        dropout=0.0,
    )
    model.eval()

    output = model(features)

    left = features.transition_names.index("t_left_A")
    right = features.transition_names.index("t_right_A")
    assert not output.transition_embeddings[left].equal(
        output.transition_embeddings[right]
    )
    assert output.sync_logits[0, left] != output.sync_logits[0, right]


def test_features_include_reverse_edge_types():
    net, im, fm = make_sequence_net(["A", "B"])
    features = pm4py_to_features(net, im, fm, trace_from_labels(["A", "B"]))

    edge_types = set(features.edge_type.tolist())
    assert edge_types == {0, 1, 2, 3}
    # One forward and one reverse edge per arc.
    assert features.edge_index.shape[1] == 2 * 2 * len(net.transitions)


def test_block_structured_example_is_exactly_alignable():
    from random import Random

    example = generate_block_structured_example(rng=Random(7), num_leaves=10)

    assert example.metadata["family"] == "block_structured"
    assert len(example.fitting_trace) > 0

    fitting = Pm4PyExactAligner().align_trace(
        example.net,
        example.initial_marking,
        example.final_marking,
        example.fitting_trace,
    )
    assert fitting.optimal
    assert fitting.cost == 0

    deviated = Pm4PyExactAligner().align_trace(
        example.net,
        example.initial_marking,
        example.final_marking,
        example.trace,
    )
    assert deviated.optimal
    assert verify_alignment(
        deviated.alignment,
        example.net,
        example.initial_marking,
        example.final_marking,
        example.trace,
    ).legal


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


def test_training_loss_handles_empty_trace():
    net, im, fm = make_sequence_net(["A"])
    trace = trace_from_labels([])
    alignment = Alignment([AlignmentMove(None, "t0_A", "A")], cost=1)
    features = pm4py_to_features(net, im, fm, trace)
    model = _small_model()
    output = model(features)
    targets = targets_from_alignment(alignment, features, optimal_cost=1)

    losses = LARALoss()(output, features, targets)

    assert losses["log_move"].isfinite()
    assert losses["total"].isfinite()


def test_training_metrics_csv_writes_epoch_rows(tmp_path):
    path = tmp_path / "nested" / "metrics.csv"
    first_row = _metrics_csv_row(
        epoch=1,
        train_metrics={"total": 1.0, "move": 0.8, "samples": 2.0},
        val_metrics={"total": 1.5, "move": 1.2, "samples": 1.0},
        elapsed_seconds=3.25,
        lr=5e-4,
        best_val=1.5,
        improved=True,
        epochs_without_improvement=0,
    )
    second_row = _metrics_csv_row(
        epoch=2,
        train_metrics={"total": 0.9, "move": 0.7, "samples": 2.0},
        val_metrics={"total": 1.6, "move": 1.3, "samples": 1.0},
        elapsed_seconds=3.5,
        lr=5e-4,
        best_val=1.5,
        improved=False,
        epochs_without_improvement=1,
    )

    _write_metrics_csv_row(path, first_row, include_header=True)
    _write_metrics_csv_row(path, second_row, include_header=False)

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert rows[0]["improved"] == "1"
    assert rows[1]["improved"] == "0"
    assert rows[0]["train_total"] == "1.0"
    assert rows[1]["val_move"] == "1.3"


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
