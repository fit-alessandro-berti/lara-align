from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pm4py.algo.conformance.alignments.edit_distance import algorithm as edit_alignments  # noqa: E402
from pm4py.algo.conformance.alignments.petri_net import algorithm as pn_alignments  # noqa: E402
from pm4py.objects.log.obj import EventLog  # noqa: E402
from pm4py.objects.petri_net.obj import PetriNet  # noqa: E402

from lara_align.data import AlignmentSample, load_split  # noqa: E402
from lara_align.exact import parse_pm4py_alignment  # noqa: E402
from lara_align.types import Alignment, CostModel  # noqa: E402
from lara_align.verify import trace_labels, verify_alignment  # noqa: E402


PETRI_VARIANTS = {
    "discounted_a_star": pn_alignments.Variants.VERSION_DISCOUNTED_A_STAR,
    "tandem_repeats": pn_alignments.Variants.APPROX_TANDEM_REPEATS,
    "sliding_window": pn_alignments.Variants.APPROX_SLIDING_WINDOW,
    "fixed_horizon": pn_alignments.Variants.APPROX_FIXED_HORIZON,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PM4Py approximate alignment variants on the LARA "
            "synthetic split and rescore every returned alignment with "
            "LARA's replay verifier."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/lara_synthetic"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-trace PM4Py timeout for approximate variants.",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=100000,
        help="Expansion budget for PM4Py approximate variants that support it.",
    )
    parser.add_argument(
        "--variants",
        default="discounted_a_star,tandem_repeats,sliding_window,fixed_horizon,edit_distance_subset",
        help="Comma-separated benchmark variants.",
    )
    parser.add_argument(
        "--sliding-window-size",
        type=int,
        default=20,
        help="Window size for APPROX_SLIDING_WINDOW.",
    )
    parser.add_argument(
        "--sliding-max-candidates",
        type=int,
        default=5,
        help="Retained candidates per window for APPROX_SLIDING_WINDOW.",
    )
    parser.add_argument(
        "--fixed-horizon",
        type=int,
        default=4,
        help="Prefix horizon for APPROX_FIXED_HORIZON.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1,
        help=(
            "Representative count per shared-net trace group for "
            "edit_distance APPROX_SUBSET."
        ),
    )
    parser.add_argument(
        "--lara-guided",
        type=Path,
        default=Path("runs/lara/test_guided.json"),
        help="Existing LARA guided baseline JSON for the same split.",
    )
    parser.add_argument(
        "--lara-unguided",
        type=Path,
        default=Path("runs/lara/test_unguided.json"),
        help="Existing LARA unguided baseline JSON for the same split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/lara/pm4py_approx_comparison.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_split(args.data_dir, args.split)
    if args.max_samples is not None:
        samples = samples[: max(0, args.max_samples)]
    if not samples:
        raise SystemExit("no samples to benchmark")

    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in selected if name not in PETRI_VARIANTS and name != "edit_distance_subset"]
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}")

    print("PM4Py Approximate Alignment Benchmark")
    print("=" * 38)
    print(f"split:          {args.split}")
    print(f"samples:        {len(samples)}")
    print(f"timeout:        {args.timeout}s")
    print(f"variants:       {', '.join(selected)}")
    print()

    records: list[dict[str, Any]] = []
    for variant_name in selected:
        start = perf_counter()
        if variant_name == "edit_distance_subset":
            variant_records = _run_edit_distance_subset(samples, args)
        else:
            variant_records = _run_petri_variant(variant_name, samples, args)
        elapsed = perf_counter() - start
        records.extend(variant_records)
        summary = _aggregate_variant(variant_name, variant_records)
        print(
            f"{variant_name:22s} "
            f"legal={_pct(summary['legal_rate'])} "
            f"optimal={_pct(summary['optimal_cost_rate'])} "
            f"mean_gap={_fmt(summary['gap_mean'])} "
            f"p95={_fmt(summary['gap_p95'])} "
            f"median_time={1000 * summary['seconds_median']:.2f}ms "
            f"elapsed={elapsed:.2f}s"
        )

    variants = {
        name: _aggregate_variant(name, [record for record in records if record["variant"] == name])
        for name in selected
    }
    payload = {
        "config": {
            "data_dir": str(args.data_dir),
            "split": args.split,
            "samples": len(samples),
            "timeout": args.timeout,
            "max_expansions": args.max_expansions,
            "variants": selected,
            "sliding_window_size": args.sliding_window_size,
            "sliding_max_candidates": args.sliding_max_candidates,
            "fixed_horizon": args.fixed_horizon,
            "subset_size": args.subset_size,
        },
        "lara_baselines": _load_lara_baselines(args),
        "variants": variants,
        "by_family": _aggregate_by(records, "family"),
        "by_representation": _aggregate_by(records, "representation_kind"),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nJSON results written to {args.output}")


def _run_petri_variant(
    variant_name: str, samples: list[AlignmentSample], args: argparse.Namespace
) -> list[dict[str, Any]]:
    variant = PETRI_VARIANTS[variant_name]
    records = []
    for index, sample in enumerate(samples, 1):
        if index == 1 or index % 100 == 0 or index == len(samples):
            print(f"  {variant_name}: {index}/{len(samples)}")
        parameters = _petri_parameters(sample, args, variant_name)
        start = perf_counter()
        raw = None
        error = None
        try:
            raw = pn_alignments.apply_trace(
                sample.trace,
                sample.net,
                sample.initial_marking,
                sample.final_marking,
                parameters=parameters,
                variant=variant,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            error = f"{type(exc).__name__}: {exc}"
        seconds = perf_counter() - start
        records.append(_record_from_raw(variant_name, sample, raw, seconds, error))
    return records


def _run_edit_distance_subset(
    samples: list[AlignmentSample], args: argparse.Namespace
) -> list[dict[str, Any]]:
    groups: dict[str, list[AlignmentSample]] = defaultdict(list)
    for sample in samples:
        groups[str(sample.metadata.get("variant_id", sample.sample_id))].append(sample)

    records = []
    sorted_groups = sorted(groups.values(), key=lambda group: group[0].sample_id)
    for index, group in enumerate(sorted_groups, 1):
        if index == 1 or index % 50 == 0 or index == len(sorted_groups):
            print(f"  edit_distance_subset: group {index}/{len(sorted_groups)}")
        first = group[0]
        log = EventLog([sample.trace for sample in group])
        parameters = {
            "activity_key": "concept:name",
            "subset_size": args.subset_size,
            "selection_method": "frequency",
            "max_align_time_trace": args.timeout,
            "max_expansions": args.max_expansions,
            "ret_tuple_as_trans_desc": True,
        }
        start = perf_counter()
        raw_results = None
        error = None
        try:
            raw_results = edit_alignments.apply(
                log,
                first.net,
                variant=edit_alignments.Variants.APPROX_SUBSET,
                parameters=parameters,
                initial_marking=first.initial_marking,
                final_marking=first.final_marking,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            error = f"{type(exc).__name__}: {exc}"
        group_seconds = perf_counter() - start
        per_trace_seconds = group_seconds / max(1, len(group))
        if raw_results is None:
            raw_results = [None] * len(group)
        for sample, raw in zip(group, raw_results):
            records.append(
                _record_from_raw(
                    "edit_distance_subset",
                    sample,
                    raw,
                    per_trace_seconds,
                    error,
                    group_seconds=group_seconds,
                    group_size=len(group),
                )
            )
    return records


def _petri_parameters(
    sample: AlignmentSample, args: argparse.Namespace, variant_name: str
) -> dict[Any, Any]:
    trace_costs = [1 for _ in trace_labels(sample.trace)]
    model_costs = {
        transition: (0 if transition.label is None else 1)
        for transition in sample.net.transitions
    }
    sync_costs = {transition: 0 for transition in sample.net.transitions}
    parameters: dict[Any, Any] = {
        pn_alignments.Parameters.ACTIVITY_KEY: "concept:name",
        pn_alignments.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True,
        pn_alignments.Parameters.PARAM_TRACE_COST_FUNCTION: trace_costs,
        pn_alignments.Parameters.PARAM_MODEL_COST_FUNCTION: model_costs,
        pn_alignments.Parameters.PARAM_SYNC_COST_FUNCTION: sync_costs,
        pn_alignments.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: args.timeout,
        pn_alignments.Parameters.ENABLE_BEST_WORST_COST: False,
        "max_expansions": args.max_expansions,
    }
    if variant_name == "sliding_window":
        parameters.update(
            {
                "window_size": args.sliding_window_size,
                "max_candidates": args.sliding_max_candidates,
            }
        )
    if variant_name == "fixed_horizon":
        parameters.update({"horizon": args.fixed_horizon})
    return parameters


def _record_from_raw(
    variant_name: str,
    sample: AlignmentSample,
    raw: dict[str, Any] | None,
    seconds: float,
    error: str | None,
    **extra: Any,
) -> dict[str, Any]:
    alignment = None
    parse_error = None
    verification = None
    if raw is not None:
        try:
            alignment = parse_pm4py_alignment(raw, sample.net)
        except Exception as exc:  # pragma: no cover - diagnostic path
            parse_error = f"{type(exc).__name__}: {exc}"
    if alignment is not None:
        verification = verify_alignment(
            alignment,
            sample.net,
            sample.initial_marking,
            sample.final_marking,
            sample.trace,
            CostModel(),
        )

    legal = bool(verification.legal) if verification is not None else False
    verified_cost = int(verification.cost) if verification is not None and legal else None
    gap = verified_cost - int(sample.optimal_cost) if verified_cost is not None else None
    exact_alignment_match = (
        _transition_signature(alignment) == _transition_signature(sample.optimal_alignment)
        if alignment is not None and legal
        else False
    )
    label_alignment_match = (
        alignment.to_pm4py_label_alignment()
        == sample.optimal_alignment.to_pm4py_label_alignment()
        if alignment is not None and legal
        else False
    )
    record = {
        "variant": variant_name,
        "sample_id": sample.sample_id,
        "family": str(sample.metadata.get("family", "unknown")),
        "motif": str(sample.metadata.get("motif", "unknown")),
        "behavior_id": str(sample.metadata.get("behavior_id", "")),
        "variant_id": str(sample.metadata.get("variant_id", "")),
        "trace_id": str(sample.metadata.get("trace_id", "")),
        "representation_kind": str(sample.metadata.get("representation_kind", "unknown")),
        "trace_len": len(sample.trace),
        "num_places": sample.metadata.get("structural_statistics", {}).get("num_places"),
        "num_transitions": sample.metadata.get("structural_statistics", {}).get("num_transitions"),
        "num_invisible": sample.metadata.get("structural_statistics", {}).get("num_invisible"),
        "duplicate_label_count": sample.metadata.get("structural_statistics", {}).get("duplicate_label_count"),
        "optimal_cost": int(sample.optimal_cost),
        "returned": raw is not None,
        "pm4py_is_valid": raw.get("is_valid") if raw is not None else None,
        "legal": legal,
        "verified_cost": verified_cost,
        "cost_gap": gap,
        "optimal_cost_match": gap == 0 if gap is not None else False,
        "exact_alignment_match": exact_alignment_match,
        "label_alignment_match": label_alignment_match,
        "seconds": seconds,
        "raw_cost": raw.get("cost") if raw is not None else None,
        "raw_standard_cost": raw.get("standard_cost") if raw is not None else None,
        "visited_states": raw.get("visited_states") if raw is not None else None,
        "queued_states": raw.get("queued_states") if raw is not None else None,
        "traversed_arcs": raw.get("traversed_arcs") if raw is not None else None,
        "fallback_used": raw.get("fallback_used") if raw is not None else None,
        "fallback_reason": raw.get("fallback_reason") if raw is not None else None,
        "approximation_method": raw.get("approximation_method") if raw is not None else None,
        "error": error,
        "parse_error": parse_error,
        "verification_reason": verification.reason if verification is not None else None,
        **extra,
    }
    return _json_safe(record)


def _transition_signature(alignment: Alignment | None) -> list[tuple[str | None, str | None, str | None]]:
    if alignment is None:
        return []
    return [
        (move.log_label, move.transition_name, move.transition_label)
        for move in alignment.moves
    ]


def _aggregate_variant(variant_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "variant": variant_name,
            "samples": 0,
            "returned_rate": 0.0,
            "legal_rate": 0.0,
            "optimal_cost_rate": 0.0,
            "seconds_median": 0.0,
        }
    legal = [record for record in records if record["legal"]]
    gaps = [record["cost_gap"] for record in legal if record["cost_gap"] is not None]
    seconds = [record["seconds"] for record in records]
    returned = [record for record in records if record["returned"]]
    valid_flags = [
        record for record in records if record.get("pm4py_is_valid") is True
    ]
    return _json_safe(
        {
            "variant": variant_name,
            "samples": len(records),
            "returned": len(returned),
            "returned_rate": len(returned) / len(records),
            "pm4py_valid_rate": len(valid_flags) / len(records),
            "legal": len(legal),
            "legal_rate": len(legal) / len(records),
            "optimal_cost": sum(1 for record in legal if record["cost_gap"] == 0),
            "optimal_cost_rate": sum(1 for record in legal if record["cost_gap"] == 0)
            / len(records),
            "exact_alignment_match_rate": sum(
                1 for record in legal if record["exact_alignment_match"]
            )
            / len(records),
            "label_alignment_match_rate": sum(
                1 for record in legal if record["label_alignment_match"]
            )
            / len(records),
            "gap_count": len(gaps),
            "gap_mean": mean(gaps) if gaps else None,
            "gap_median": median(gaps) if gaps else None,
            "gap_p90": _quantile(gaps, 0.90),
            "gap_p95": _quantile(gaps, 0.95),
            "gap_max": max(gaps) if gaps else None,
            "seconds_mean": mean(seconds),
            "seconds_median": median(seconds),
            "seconds_p90": _quantile(seconds, 0.90),
            "seconds_max": max(seconds),
            "fallback_rate": sum(1 for record in records if record.get("fallback_used"))
            / len(records),
            "error_count": sum(1 for record in records if record.get("error")),
            "parse_error_count": sum(1 for record in records if record.get("parse_error")),
        }
    )


def _aggregate_by(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["variant"], str(record.get(key, "unknown")))].append(record)
    return {
        f"{variant}::{value}": _aggregate_variant(variant, group)
        for (variant, value), group in sorted(groups.items())
    }


def _load_lara_baselines(args: argparse.Namespace) -> dict[str, Any]:
    baselines = {}
    for name, path in {
        "lara_guided_fast": args.lara_guided,
        "lara_unguided_fast": args.lara_unguided,
    }.items():
        if not path.exists():
            baselines[name] = {"available": False, "path": str(path)}
            continue
        payload = json.loads(path.read_text())
        metrics = payload.get("metrics", payload)
        baselines[name] = {
            "available": True,
            "path": str(path),
            "samples": metrics.get("samples"),
            "legal_rate": metrics.get("legal_rate"),
            "optimal_cost_rate": metrics.get("optimal_cost_rate"),
            "gap_mean": metrics.get("gap_mean"),
            "gap_p95": metrics.get("gap_p95"),
            "gap_max": metrics.get("gap_max"),
            "seconds_total": metrics.get("elapsed_seconds"),
            "exact_alignment_match_rate": metrics.get("exact_alignment_match_rate"),
            "label_alignment_match_rate": metrics.get("label_alignment_match_rate"),
        }
    return baselines


def _quantile(values: Iterable[float | int], q: float) -> float | int | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    main()
