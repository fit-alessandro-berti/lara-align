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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    samples = load_split(args.data_dir, args.split)
    criterion = LARALoss()

    start = perf_counter()
    metrics, records = evaluate_model(
        model,
        samples,
        criterion,
        device,
        run_certified=args.run_certified,
    )
    metrics["elapsed_seconds"] = perf_counter() - start
    metrics["split"] = args.split
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["checkpoint_epoch"] = checkpoint.get("epoch")

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
) -> tuple[dict[str, Any], list[EvaluationRecord]]:
    model.eval()
    system = CertifyingAlignmentSystem(model=model, device=device)

    loss_totals: dict[str, float] = {}
    certified_optimal = 0
    records: list[EvaluationRecord] = []

    with torch.no_grad():
        for sample in samples:
            losses = _sample_losses(model, criterion, sample, device)
            for key, value in losses.items():
                loss_totals[key] = loss_totals.get(key, 0.0) + float(value.detach().cpu())

            fast = system.align(
                sample.net,
                sample.initial_marking,
                sample.final_marking,
                sample.trace,
                mode=LARAMode.FAST,
            )

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

            records.append(_make_record(sample, fast, certified_value))

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

    examples = _select_examples(records, args)
    if examples:
        lines.append("")
        lines.append(f"Example Alignments ({args.example_selection}, {len(examples)} shown)")
        lines.append("-" * 48)
        for index, record in enumerate(examples, start=1):
            lines.extend(_format_example(index, record, args.max_moves))
    return "\n".join(lines)


def _make_record(sample: AlignmentSample, fast_result, certified_value: bool | None) -> EvaluationRecord:
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
        "failure_reasons": _failure_reasons(records),
    }


def _family_metrics(records: list[EvaluationRecord]) -> dict[str, dict[str, Any]]:
    families = sorted({record.family for record in records})
    result: dict[str, dict[str, Any]] = {}
    for family in families:
        family_records = [record for record in records if record.family == family]
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
