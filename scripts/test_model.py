from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean, median, pstdev
import sys
from time import perf_counter
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lara_align.certifier import CertifyingAlignmentSystem, LARAMode  # noqa: E402
from lara_align.checkpoint import load_checkpoint  # noqa: E402
from lara_align.data import AlignmentSample, load_split  # noqa: E402
from lara_align.features import pm4py_to_features  # noqa: E402
from lara_align.training import LARALoss, targets_from_alignment  # noqa: E402
from lara_align.types import Alignment, AlignmentMove  # noqa: E402
from lara_align.verify import trace_labels  # noqa: E402


@dataclass
class EvaluationRecord:
    sample_id: str
    family: str
    trace: list[str]
    optimal_alignment: Alignment
    predicted_alignment: Alignment | None
    optimal_cost: int
    predicted_cost: int | None
    legal: bool
    failure_reason: str | None
    exact_alignment_match: bool
    label_alignment_match: bool
    move_count_gap: int | None
    certified_optimal: bool | None = None
    behavior_id: str = ""
    representation_kind: str = "unknown"
    trace_id: str = ""
    fast_seconds: float | None = None
    exact_visited_states: int | None = None
    exact_queued_states: int | None = None
    exact_seconds: float | None = None

    @property
    def cost_gap(self) -> int | None:
        if self.predicted_cost is None:
            return None
        return self.predicted_cost - self.optimal_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LARA model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/lara_synthetic"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/lara/best.pt"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Print a human-readable report or machine-readable JSON.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of example alignments to print in human mode.",
    )
    parser.add_argument(
        "--example-selection",
        choices=["worst", "first"],
        default="worst",
        help="Choose examples with largest cost gaps/invalidity or the first N samples.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=40,
        help="Maximum moves shown per alignment in human mode.",
    )
    parser.add_argument(
        "--run-certified",
        action="store_true",
        help="Also run certified mode through pm4py for each sample.",
    )
    parser.add_argument(
        "--no-guidance",
        action="store_true",
        help=(
            "Ablation: decode without neural move scores (structural "
            "heuristics only) to isolate the learned model's contribution."
        ),
    )
    parser.add_argument(
        "--representation-kind",
        action="append",
        default=None,
        help="Evaluate only this representation kind; repeat for multiple kinds.",
    )
    parser.add_argument(
        "--motif",
        action="append",
        default=None,
        help="Evaluate only this behavior motif; repeat for multiple motifs.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    samples = load_split(args.data_dir, args.split)
    if args.representation_kind:
        selected = set(args.representation_kind)
        samples = [
            sample
            for sample in samples
            if sample.metadata.get("representation_kind") in selected
        ]
    if args.motif:
        selected = set(args.motif)
        samples = [sample for sample in samples if sample.metadata.get("motif") in selected]
    if args.max_samples is not None:
        samples = samples[: max(0, args.max_samples)]
    if not samples:
        raise SystemExit("no samples matched the requested evaluation filters")
    criterion = LARALoss()

    start = perf_counter()
    metrics, records = evaluate_model(
        model,
        samples,
        criterion,
        device,
        run_certified=args.run_certified,
        use_guidance=not args.no_guidance,
    )
    metrics["elapsed_seconds"] = perf_counter() - start
    metrics["split"] = args.split
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["checkpoint_epoch"] = checkpoint.get("epoch")
    metrics["guidance"] = not args.no_guidance

    payload = {
        "metrics": metrics,
        "examples": [_record_to_json(record) for record in _select_examples(records, args)],
    }

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_human_report(metrics, records, args))

    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(payload, indent=2, sort_keys=True))


def evaluate_model(
    model,
    samples: list[AlignmentSample],
    criterion: LARALoss,
    device: torch.device,
    run_certified: bool = False,
    use_guidance: bool = True,
) -> tuple[dict[str, Any], list[EvaluationRecord]]:
    model.eval()
    system = CertifyingAlignmentSystem(
        model=model, device=device, use_guidance=use_guidance
    )

    loss_totals: dict[str, float] = {}
    certified_optimal = 0
    records: list[EvaluationRecord] = []

    with torch.no_grad():
        for sample in samples:
            losses = _sample_losses(model, criterion, sample, device)
            for key, value in losses.items():
                loss_totals[key] = loss_totals.get(key, 0.0) + float(value.detach().cpu())

            fast_start = perf_counter()
            fast = system.align(
                sample.net,
                sample.initial_marking,
                sample.final_marking,
                sample.trace,
                mode=LARAMode.FAST,
            )
            fast_seconds = perf_counter() - fast_start

            certified_value: bool | None = None
            if run_certified:
                certified = system.align(
                    sample.net,
                    sample.initial_marking,
                    sample.final_marking,
                    sample.trace,
                    mode=LARAMode.CERTIFIED,
                )
                certified_value = certified.certified_optimal
                if certified.certified_optimal:
                    certified_optimal += 1

            records.append(_make_record(sample, fast, certified_value, fast_seconds))

    count = max(1, len(samples))
    averaged_losses = {f"loss_{key}": value / count for key, value in loss_totals.items()}
    metrics = {
        "samples": len(samples),
        **averaged_losses,
        **_alignment_metrics(records),
        "certified_optimal_rate": certified_optimal / count if run_certified else None,
    }
    return metrics, records


def format_human_report(
    metrics: dict[str, Any],
    records: list[EvaluationRecord],
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append("LARA Alignment Evaluation")
    lines.append("=" * 25)
    lines.append(f"split:              {metrics['split']}")
    lines.append(f"samples:            {metrics['samples']}")
    lines.append(f"checkpoint:         {metrics['checkpoint']}")
    lines.append(f"checkpoint epoch:   {metrics['checkpoint_epoch']}")
    lines.append(
        f"neural guidance:    {'on' if metrics.get('guidance', True) else 'off (ablation)'}"
    )
    lines.append(f"elapsed:            {metrics['elapsed_seconds']:.2f}s")
    lines.append("")
    lines.append("Replay And Optimality")
    lines.append("-" * 22)
    lines.append(f"replayable alignments:       {_pct(metrics['legal_rate'])}")
    lines.append(f"equal optimum cost:          {_pct(metrics['optimal_cost_rate'])}")
    lines.append(f"exact transition sequence:   {_pct(metrics['exact_alignment_match_rate'])}")
    lines.append(f"same label-level alignment:  {_pct(metrics['label_alignment_match_rate'])}")
    if metrics["certified_optimal_rate"] is not None:
        lines.append(f"certified optimal:           {_pct(metrics['certified_optimal_rate'])}")
    lines.append("")
    lines.append("Cost Gap: reconstructed cost - pm4py optimum")
    lines.append("-" * 45)
    lines.append(f"valid gaps counted: {metrics['legal_gap_count']}")
    lines.append(f"mean / median:     {_fmt(metrics['gap_mean'])} / {_fmt(metrics['gap_median'])}")
    lines.append(f"min / max:         {_fmt(metrics['gap_min'])} / {_fmt(metrics['gap_max'])}")
    lines.append(f"p90 / p95:         {_fmt(metrics['gap_p90'])} / {_fmt(metrics['gap_p95'])}")
    lines.append(f"std dev:           {_fmt(metrics['gap_std'])}")
    lines.append(f"mean relative gap: {_fmt(metrics['gap_relative_mean'])}")
    lines.append("")
    paired = metrics.get("paired_equivalence", {})
    if isinstance(paired, dict) and int(paired.get("pair_count", 0)) > 0:
        lines.append("Paired Equivalent Representations")
        lines.append("-" * 33)
        lines.append(f"paired traces:                 {paired.get('pair_count', 0)}")
        lines.append(
            f"exact optimal-cost consistency: {_pct(paired.get('exact_cost_consistency_rate'))}"
        )
        lines.append(
            f"legal on every representation:  {_pct(paired.get('all_variants_legal_rate'))}"
        )
        lines.append(
            f"predicted-cost consistency:     {_pct(paired.get('predicted_cost_consistency_rate'))}"
        )
        lines.append(
            f"optimality agreement:           {_pct(paired.get('optimality_agreement_rate'))}"
        )
        lines.append(
            f"mean fast runtime ratio:        {_fmt(paired.get('fast_runtime_ratio_mean'))}"
        )
        lines.append(
            f"mean exact visited-state ratio: {_fmt(paired.get('exact_visited_state_ratio_mean'))}"
        )
        lines.append(
            f"mean exact runtime ratio:       {_fmt(paired.get('exact_runtime_ratio_mean'))}"
        )
        lines.append("")
    lines.append("Losses")
    lines.append("-" * 6)
    for key in sorted(k for k in metrics if k.startswith("loss_")):
        lines.append(f"{key:24s} {_fmt(metrics[key])}")
    lines.append("")
    lines.append("By Family")
    lines.append("-" * 9)
    for family, family_metrics in sorted(metrics["by_family"].items()):
        lines.append(
            f"{family:24s} n={family_metrics['samples']:3d} "
            f"replay={_pct(family_metrics['legal_rate'])} "
            f"opt_cost={_pct(family_metrics['optimal_cost_rate'])} "
            f"mean_gap={_fmt(family_metrics['gap_mean'])}"
        )
    lines.append("")
    lines.append("By Representation")
    lines.append("-" * 17)
    for representation, values in sorted(metrics.get("by_representation", {}).items()):
        lines.append(
            f"{representation:24s} n={values['samples']:3d} "
            f"replay={_pct(values['legal_rate'])} "
            f"opt_cost={_pct(values['optimal_cost_rate'])} "
            f"mean_gap={_fmt(values['gap_mean'])}"
        )

    examples = _select_examples(records, args)
    if examples:
        lines.append("")
        lines.append(f"Example Alignments ({args.example_selection}, {len(examples)} shown)")
        lines.append("-" * 48)
        for index, record in enumerate(examples, start=1):
            lines.extend(_format_example(index, record, args.max_moves))
    return "\n".join(lines)


def _make_record(
    sample: AlignmentSample,
    fast_result,
    certified_value: bool | None,
    fast_seconds: float | None = None,
) -> EvaluationRecord:
    predicted = fast_result.alignment
    exact_match = predicted is not None and _alignment_identity(
        predicted
    ) == _alignment_identity(sample.optimal_alignment)
    label_match = predicted is not None and (
        predicted.to_pm4py_label_alignment()
        == sample.optimal_alignment.to_pm4py_label_alignment()
    )
    move_count_gap = (
        len(predicted.moves) - len(sample.optimal_alignment.moves)
        if predicted is not None
        else None
    )
    exact_diagnostics = sample.metadata.get("exact_diagnostics", {})
    exact_diagnostics = exact_diagnostics if isinstance(exact_diagnostics, dict) else {}
    return EvaluationRecord(
        sample_id=sample.sample_id,
        family=str(sample.metadata.get("family", "unknown")),
        trace=trace_labels(sample.trace),
        optimal_alignment=sample.optimal_alignment,
        predicted_alignment=predicted,
        optimal_cost=sample.optimal_cost,
        predicted_cost=fast_result.cost,
        legal=fast_result.legal,
        failure_reason=fast_result.verifier.reason if fast_result.verifier else None,
        exact_alignment_match=exact_match,
        label_alignment_match=label_match,
        move_count_gap=move_count_gap,
        certified_optimal=certified_value,
        behavior_id=str(sample.metadata.get("behavior_id", sample.sample_id)),
        representation_kind=str(sample.metadata.get("representation_kind", "unknown")),
        trace_id=str(sample.metadata.get("trace_id", sample.sample_id)),
        fast_seconds=fast_seconds,
        exact_visited_states=_optional_int(exact_diagnostics.get("visited_states")),
        exact_queued_states=_optional_int(exact_diagnostics.get("queued_states")),
        exact_seconds=_optional_float(exact_diagnostics.get("elapsed_seconds")),
    )


def _alignment_metrics(records: list[EvaluationRecord]) -> dict[str, Any]:
    sample_count = max(1, len(records))
    legal_records = [record for record in records if record.legal and record.cost_gap is not None]
    gaps = [int(record.cost_gap) for record in legal_records if record.cost_gap is not None]
    relative_gaps = [
        record.cost_gap / max(1, record.optimal_cost)
        for record in legal_records
        if record.cost_gap is not None
    ]
    optimal_cost_count = sum(1 for record in legal_records if record.cost_gap == 0)
    exact_match_count = sum(1 for record in records if record.exact_alignment_match)
    label_match_count = sum(1 for record in records if record.label_alignment_match)

    return {
        "legal_count": len(legal_records),
        "illegal_count": len(records) - len(legal_records),
        "legal_rate": len(legal_records) / sample_count,
        "optimal_cost_count": optimal_cost_count,
        "optimal_cost_rate": optimal_cost_count / sample_count,
        "exact_alignment_match_count": exact_match_count,
        "exact_alignment_match_rate": exact_match_count / sample_count,
        "label_alignment_match_count": label_match_count,
        "label_alignment_match_rate": label_match_count / sample_count,
        "legal_gap_count": len(gaps),
        "gap_mean": _safe_mean(gaps),
        "gap_median": median(gaps) if gaps else None,
        "gap_min": min(gaps) if gaps else None,
        "gap_max": max(gaps) if gaps else None,
        "gap_std": pstdev(gaps) if len(gaps) > 1 else 0.0 if gaps else None,
        "gap_p50": _percentile(gaps, 0.50),
        "gap_p90": _percentile(gaps, 0.90),
        "gap_p95": _percentile(gaps, 0.95),
        "gap_relative_mean": _safe_mean(relative_gaps),
        "move_count_gap_mean": _safe_mean(
            [record.move_count_gap for record in records if record.move_count_gap is not None]
        ),
        "by_family": _family_metrics(records),
        "by_representation": _group_metrics(
            records, lambda record: record.representation_kind
        ),
        "paired_equivalence": _paired_equivalence_metrics(records),
        "failure_reasons": _failure_reasons(records),
    }


def _paired_equivalence_metrics(records: list[EvaluationRecord]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[EvaluationRecord]] = {}
    for record in records:
        grouped.setdefault((record.behavior_id, record.trace_id), []).append(record)
    groups = [
        values
        for values in grouped.values()
        if len({record.representation_kind for record in values}) > 1
    ]
    exact_consistent = 0
    all_legal = 0
    predicted_consistent = 0
    optimality_agreement = 0
    fast_ratios: list[float] = []
    visited_ratios: list[float] = []
    queued_ratios: list[float] = []
    exact_runtime_ratios: list[float] = []
    by_pair: dict[str, list[list[EvaluationRecord]]] = {}
    for values in groups:
        exact_consistent += int(len({record.optimal_cost for record in values}) == 1)
        all_legal += int(all(record.legal for record in values))
        predicted = [record.predicted_cost for record in values]
        predicted_consistent += int(
            all(value is not None for value in predicted) and len(set(predicted)) == 1
        )
        statuses = [record.legal and record.cost_gap == 0 for record in values]
        optimality_agreement += int(len(set(statuses)) == 1)
        fast_ratio = _max_min_ratio(
            [record.fast_seconds for record in values if record.fast_seconds is not None]
        )
        visited_ratio = _max_min_ratio(
            [
                record.exact_visited_states
                for record in values
                if record.exact_visited_states is not None
            ]
        )
        queued_ratio = _max_min_ratio(
            [
                record.exact_queued_states
                for record in values
                if record.exact_queued_states is not None
            ]
        )
        exact_runtime_ratio = _max_min_ratio(
            [record.exact_seconds for record in values if record.exact_seconds is not None]
        )
        if fast_ratio is not None:
            fast_ratios.append(fast_ratio)
        if visited_ratio is not None:
            visited_ratios.append(visited_ratio)
        if queued_ratio is not None:
            queued_ratios.append(queued_ratio)
        if exact_runtime_ratio is not None:
            exact_runtime_ratios.append(exact_runtime_ratio)
        pair = "__vs__".join(sorted({record.representation_kind for record in values}))
        by_pair.setdefault(pair, []).append(values)
    count = len(groups)
    return {
        "pair_count": count,
        "exact_cost_consistency_rate": exact_consistent / count if count else 0.0,
        "all_variants_legal_rate": all_legal / count if count else 0.0,
        "predicted_cost_consistency_rate": predicted_consistent / count if count else 0.0,
        "optimality_agreement_rate": optimality_agreement / count if count else 0.0,
        "fast_runtime_ratio_mean": _safe_mean(fast_ratios),
        "exact_visited_state_ratio_mean": _safe_mean(visited_ratios),
        "exact_queued_state_ratio_mean": _safe_mean(queued_ratios),
        "exact_runtime_ratio_mean": _safe_mean(exact_runtime_ratios),
        "by_representation_pair": {
            pair: {
                "count": len(values),
                "all_variants_legal_rate": sum(
                    all(record.legal for record in group) for group in values
                )
                / len(values),
                "predicted_cost_consistency_rate": sum(
                    all(record.predicted_cost is not None for record in group)
                    and len({record.predicted_cost for record in group}) == 1
                    for group in values
                )
                / len(values),
                "duplicate_transition_selection_accuracy": _safe_mean(
                    [
                        float(record.exact_alignment_match)
                        for group in values
                        for record in group
                        if record.representation_kind == "duplicate_prefix"
                    ]
                ),
                "invisible_routing_label_accuracy": _safe_mean(
                    [
                        float(record.label_alignment_match)
                        for group in values
                        for record in group
                        if record.representation_kind == "silent_routing"
                    ]
                ),
            }
            for pair, values in sorted(by_pair.items())
        },
    }


def _family_metrics(records: list[EvaluationRecord]) -> dict[str, dict[str, Any]]:
    return _group_metrics(records, lambda record: record.family)


def _group_metrics(records: list[EvaluationRecord], key_fn) -> dict[str, dict[str, Any]]:
    families = sorted({key_fn(record) for record in records})
    result: dict[str, dict[str, Any]] = {}
    for family in families:
        family_records = [record for record in records if key_fn(record) == family]
        count = max(1, len(family_records))
        legal = [record for record in family_records if record.legal and record.cost_gap is not None]
        gaps = [record.cost_gap for record in legal if record.cost_gap is not None]
        result[family] = {
            "samples": len(family_records),
            "legal_rate": len(legal) / count,
            "optimal_cost_rate": sum(1 for record in legal if record.cost_gap == 0) / count,
            "gap_mean": _safe_mean(gaps),
            "gap_max": max(gaps) if gaps else None,
        }
    return result


def _failure_reasons(records: list[EvaluationRecord]) -> dict[str, int]:
    reasons: dict[str, int] = {}
    for record in records:
        if record.legal:
            continue
        reason = record.failure_reason or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    return reasons


def _select_examples(records: list[EvaluationRecord], args: argparse.Namespace) -> list[EvaluationRecord]:
    if args.num_examples <= 0:
        return []
    if args.example_selection == "first":
        return records[: args.num_examples]
    return sorted(records, key=_example_priority, reverse=True)[: args.num_examples]


def _example_priority(record: EvaluationRecord) -> tuple[int, int, int]:
    invalid = 1 if not record.legal else 0
    gap = record.cost_gap if record.cost_gap is not None else 10**9
    move_gap = abs(record.move_count_gap or 0)
    return invalid, int(gap), move_gap


def _format_example(index: int, record: EvaluationRecord, max_moves: int) -> list[str]:
    gap = record.cost_gap
    lines = [
        "",
        f"[{index}] {record.sample_id} family={record.family}",
        f"trace: {' '.join(record.trace)}",
        (
            f"legal={record.legal} optimal_cost={record.optimal_cost} "
            f"reconstructed_cost={record.predicted_cost} gap={gap}"
        ),
        (
            f"exact_transition_match={record.exact_alignment_match} "
            f"label_alignment_match={record.label_alignment_match} "
            f"move_count_gap={record.move_count_gap}"
        ),
    ]
    if record.failure_reason:
        lines.append(f"failure_reason={record.failure_reason}")
    lines.append("pm4py optimal:")
    lines.extend(_format_alignment(record.optimal_alignment, max_moves))
    lines.append("LARA reconstructed:")
    if record.predicted_alignment is None:
        lines.append("  <no alignment>")
    else:
        lines.extend(_format_alignment(record.predicted_alignment, max_moves))
    return lines


def _format_alignment(alignment: Alignment, max_moves: int) -> list[str]:
    rows = ["  idx  log        model_transition        model_label  kind"]
    for index, move in enumerate(alignment.moves[:max_moves]):
        rows.append(
            f"  {index:>3}  {_move_log(move):<9}  "
            f"{_move_transition(move):<22}  {_move_model_label(move):<11}  {move.kind.value}"
        )
    if len(alignment.moves) > max_moves:
        rows.append(f"  ... {len(alignment.moves) - max_moves} more moves")
    return rows


def _alignment_identity(alignment: Alignment) -> list[tuple[str | None, str | None, str | None]]:
    return [
        (move.log_label, move.transition_name, move.transition_label)
        for move in alignment.moves
    ]


def _record_to_json(record: EvaluationRecord) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "family": record.family,
        "trace": record.trace,
        "optimal_cost": record.optimal_cost,
        "predicted_cost": record.predicted_cost,
        "cost_gap": record.cost_gap,
        "legal": record.legal,
        "failure_reason": record.failure_reason,
        "exact_alignment_match": record.exact_alignment_match,
        "label_alignment_match": record.label_alignment_match,
        "move_count_gap": record.move_count_gap,
        "certified_optimal": record.certified_optimal,
        "behavior_id": record.behavior_id,
        "representation_kind": record.representation_kind,
        "trace_id": record.trace_id,
        "fast_seconds": record.fast_seconds,
        "exact_visited_states": record.exact_visited_states,
        "exact_queued_states": record.exact_queued_states,
        "exact_seconds": record.exact_seconds,
        "optimal_alignment": [
            _move_to_json(move) for move in record.optimal_alignment.moves
        ],
        "predicted_alignment": (
            [_move_to_json(move) for move in record.predicted_alignment.moves]
            if record.predicted_alignment is not None
            else None
        ),
    }


def _move_to_json(move: AlignmentMove) -> dict[str, str | None]:
    return {
        "log_label": move.log_label,
        "transition_name": move.transition_name,
        "transition_label": move.transition_label,
        "kind": move.kind.value,
    }


def _sample_losses(
    model,
    criterion: LARALoss,
    sample: AlignmentSample,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    features = pm4py_to_features(
        sample.net,
        sample.initial_marking,
        sample.final_marking,
        sample.trace,
    ).to(device)
    targets = targets_from_alignment(
        sample.optimal_alignment,
        features,
        optimal_cost=sample.optimal_cost,
        mask_transition_identity=not bool(
            sample.metadata.get("transition_identity_identifiable", True)
        ),
    )
    output = model(features)
    return criterion(output, features, targets)


def _move_log(move: AlignmentMove) -> str:
    return move.log_label if move.log_label is not None else ">>"


def _move_transition(move: AlignmentMove) -> str:
    return move.transition_name if move.transition_name is not None else ">>"


def _move_model_label(move: AlignmentMove) -> str:
    if move.transition_name is None:
        return ">>"
    return move.transition_label if move.transition_label is not None else "tau"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _safe_mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def _max_min_ratio(values: list[float | int]) -> float | None:
    clean = [float(value) for value in values]
    if len(clean) < 2:
        return None
    return max(clean) / max(min(clean), 1e-12)


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


if __name__ == "__main__":
    main()
