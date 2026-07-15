from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from statistics import mean, median
import sys
from time import perf_counter
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pm4py.algo.conformance.alignments.petri_net import algorithm as pn_alignments  # noqa: E402

from lara_align.certifier import CertifyingAlignmentSystem, LARAMode  # noqa: E402
from lara_align.checkpoint import load_checkpoint  # noqa: E402
from lara_align.decode import GreedyCandidateDecoder  # noqa: E402
from lara_align.exact import Pm4PyExactAligner, parse_pm4py_alignment  # noqa: E402
from lara_align.synthetic import generate_block_structured_example  # noqa: E402
from lara_align.types import CostModel  # noqa: E402
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
            "Compare LARA fast mode and PM4Py approximate Petri-net aligners "
            "on generated block-structured stress nets."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/lara/best.pt"))
    parser.add_argument("--sizes", default="5,10,20,40")
    parser.add_argument("--deviation-rates", default="0.15,0.35")
    parser.add_argument("--samples-per-config", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exact-timeout", type=float, default=30.0)
    parser.add_argument("--approx-timeout", type=float, default=5.0)
    parser.add_argument("--max-expansions", type=int, default=100000)
    parser.add_argument("--alphabet-fraction", type=float, default=0.7)
    parser.add_argument("--loop-redo-prob", type=float, default=0.3)
    parser.add_argument("--max-loop-redos", type=int, default=2)
    parser.add_argument("--max-prefix-depth", type=int, default=8)
    parser.add_argument("--max-final-depth", type=int, default=64)
    parser.add_argument("--sliding-window-size", type=int, default=20)
    parser.add_argument("--sliding-max-candidates", type=int, default=5)
    parser.add_argument("--fixed-horizon", type=int, default=4)
    parser.add_argument(
        "--variants",
        default="discounted_a_star,tandem_repeats,sliding_window,fixed_horizon",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/lara/pm4py_approx_scaling_comparison.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    deviation_rates = [
        float(value) for value in args.deviation_rates.split(",") if value.strip()
    ]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = [variant for variant in variants if variant not in PETRI_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}")

    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    decoder = GreedyCandidateDecoder(
        max_prefix_model_depth=args.max_prefix_depth,
        max_final_model_depth=args.max_final_depth,
    )
    lara = CertifyingAlignmentSystem(model=model, decoder=decoder, device=device)
    exact = Pm4PyExactAligner()

    print("PM4Py Approximation Scaling Benchmark")
    print("=" * 38)
    print(f"checkpoint:      {args.checkpoint} (epoch {checkpoint.get('epoch')})")
    print(f"sizes:           {sizes}")
    print(f"deviation rates: {deviation_rates}")
    print(f"samples/config:  {args.samples_per_config}")
    print(f"variants:        lara_fast, {', '.join(variants)}")
    print()

    records: list[dict[str, Any]] = []
    for size in sizes:
        for deviation_rate in deviation_rates:
            config_records = _run_config(size, deviation_rate, args, variants, lara, exact)
            records.extend(config_records)
            for variant_name in ["lara_fast", *variants]:
                summary = _aggregate(
                    [
                        record
                        for record in config_records
                        if record["variant"] == variant_name
                    ]
                )
                print(
                    f"size={size:2d} dev={deviation_rate:.2f} "
                    f"{variant_name:18s} "
                    f"legal={_pct(summary['legal_rate'])} "
                    f"optimal={_pct(summary['optimal_rate'])} "
                    f"mean_gap={_fmt(summary['gap_mean'])} "
                    f"median={1000 * summary['seconds_median']:.2f}ms"
                )
            print()

    rows = _rows(records)
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": {
            "sizes": sizes,
            "deviation_rates": deviation_rates,
            "samples_per_config": args.samples_per_config,
            "seed": args.seed,
            "exact_timeout": args.exact_timeout,
            "approx_timeout": args.approx_timeout,
            "max_expansions": args.max_expansions,
            "alphabet_fraction": args.alphabet_fraction,
            "loop_redo_prob": args.loop_redo_prob,
            "max_loop_redos": args.max_loop_redos,
            "max_prefix_depth": args.max_prefix_depth,
            "max_final_depth": args.max_final_depth,
            "sliding_window_size": args.sliding_window_size,
            "sliding_max_candidates": args.sliding_max_candidates,
            "fixed_horizon": args.fixed_horizon,
            "variants": variants,
        },
        "rows": rows,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"JSON results written to {args.output}")


def _run_config(
    size: int,
    deviation_rate: float,
    args: argparse.Namespace,
    variants: list[str],
    lara: CertifyingAlignmentSystem,
    exact: Pm4PyExactAligner,
) -> list[dict[str, Any]]:
    rng = Random((args.seed, size, round(deviation_rate * 1000)).__hash__())
    alphabet_size = max(2, int(round(size * args.alphabet_fraction)))
    records: list[dict[str, Any]] = []

    for index in range(args.samples_per_config):
        example = generate_block_structured_example(
            rng=rng,
            num_leaves=size,
            alphabet_size=alphabet_size,
            deviation_rate=deviation_rate,
            loop_redo_probability=args.loop_redo_prob,
            max_loop_redos=args.max_loop_redos,
        )
        exact_start = perf_counter()
        exact_result = exact.align_trace(
            example.net,
            example.initial_marking,
            example.final_marking,
            example.trace,
            timeout_seconds=args.exact_timeout,
        )
        exact_seconds = perf_counter() - exact_start
        exact_cost = exact_result.cost
        exact_solved = exact_cost is not None

        with torch.no_grad():
            start = perf_counter()
            lara_result = lara.align(
                example.net,
                example.initial_marking,
                example.final_marking,
                example.trace,
                mode=LARAMode.FAST,
            )
            lara_seconds = perf_counter() - start
        records.append(
            _base_record(
                "lara_fast",
                size,
                deviation_rate,
                index,
                example,
                lara_seconds,
                bool(lara_result.legal),
                lara_result.cost,
                exact_cost,
                exact_solved,
                exact_seconds,
                None,
            )
        )

        for variant_name in variants:
            start = perf_counter()
            raw = None
            error = None
            try:
                raw = pn_alignments.apply_trace(
                    example.trace,
                    example.net,
                    example.initial_marking,
                    example.final_marking,
                    parameters=_parameters(example, args, variant_name),
                    variant=PETRI_VARIANTS[variant_name],
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                error = f"{type(exc).__name__}: {exc}"
            seconds = perf_counter() - start
            legal = False
            cost = None
            reason = None
            if raw is not None:
                try:
                    alignment = parse_pm4py_alignment(raw, example.net)
                    verification = verify_alignment(
                        alignment,
                        example.net,
                        example.initial_marking,
                        example.final_marking,
                        example.trace,
                        CostModel(),
                    )
                    legal = verification.legal
                    cost = verification.cost if verification.legal else None
                    reason = verification.reason
                except Exception as exc:  # pragma: no cover - diagnostic path
                    reason = f"{type(exc).__name__}: {exc}"
            records.append(
                _base_record(
                    variant_name,
                    size,
                    deviation_rate,
                    index,
                    example,
                    seconds,
                    legal,
                    cost,
                    exact_cost,
                    exact_solved,
                    exact_seconds,
                    error or reason,
                    raw,
                )
            )
    return records


def _parameters(example, args: argparse.Namespace, variant_name: str) -> dict[Any, Any]:
    parameters: dict[Any, Any] = {
        pn_alignments.Parameters.ACTIVITY_KEY: "concept:name",
        pn_alignments.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True,
        pn_alignments.Parameters.PARAM_TRACE_COST_FUNCTION: [
            1 for _ in trace_labels(example.trace)
        ],
        pn_alignments.Parameters.PARAM_MODEL_COST_FUNCTION: {
            transition: (0 if transition.label is None else 1)
            for transition in example.net.transitions
        },
        pn_alignments.Parameters.PARAM_SYNC_COST_FUNCTION: {
            transition: 0 for transition in example.net.transitions
        },
        pn_alignments.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: args.approx_timeout,
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


def _base_record(
    variant: str,
    size: int,
    deviation_rate: float,
    index: int,
    example,
    seconds: float,
    legal: bool,
    cost: int | None,
    exact_cost: int | None,
    exact_solved: bool,
    exact_seconds: float,
    error: str | None,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gap = cost - exact_cost if legal and cost is not None and exact_cost is not None else None
    return _json_safe(
        {
            "variant": variant,
            "size": size,
            "deviation_rate": deviation_rate,
            "index": index,
            "trace_len": len(example.trace),
            "num_transitions": example.metadata["num_transitions"],
            "num_places": example.metadata["num_places"],
            "num_invisible": example.metadata["num_invisible"],
            "num_duplicate_labels": example.metadata["num_duplicate_labels"],
            "operator_counts": example.metadata["operator_counts"],
            "seconds": seconds,
            "legal": legal,
            "cost": cost,
            "exact_cost": exact_cost,
            "exact_solved": exact_solved,
            "exact_seconds": exact_seconds,
            "cost_gap": gap,
            "optimal": gap == 0 if gap is not None else False,
            "error": error,
            "pm4py_is_valid": raw.get("is_valid") if raw is not None else None,
            "fallback_used": raw.get("fallback_used") if raw is not None else None,
            "fallback_reason": raw.get("fallback_reason") if raw is not None else None,
            "visited_states": raw.get("visited_states") if raw is not None else None,
            "queued_states": raw.get("queued_states") if raw is not None else None,
            "traversed_arcs": raw.get("traversed_arcs") if raw is not None else None,
        }
    )


def _rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (record["size"], record["deviation_rate"], record["variant"])
            for record in records
        }
    )
    rows = []
    for size, deviation_rate, variant in keys:
        group = [
            record
            for record in records
            if record["size"] == size
            and record["deviation_rate"] == deviation_rate
            and record["variant"] == variant
        ]
        row = _aggregate(group)
        row.update({"size": size, "deviation_rate": deviation_rate, "variant": variant})
        rows.append(row)
    return rows


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "n": 0,
            "legal_rate": 0.0,
            "optimal_rate": 0.0,
            "gap_mean": None,
            "seconds_median": 0.0,
        }
    legal = [record for record in records if record["legal"]]
    comparable = [record for record in records if record["cost_gap"] is not None]
    gaps = [record["cost_gap"] for record in comparable]
    seconds = [record["seconds"] for record in records]
    return _json_safe(
        {
            "n": len(records),
            "legal_rate": len(legal) / len(records),
            "exact_solved_rate": sum(1 for record in records if record["exact_solved"])
            / len(records),
            "comparable": len(comparable),
            "optimal_rate": sum(1 for record in comparable if record["cost_gap"] == 0)
            / max(1, len(comparable)),
            "gap_mean": mean(gaps) if gaps else None,
            "gap_median": median(gaps) if gaps else None,
            "gap_p90": _quantile(gaps, 0.90),
            "gap_p95": _quantile(gaps, 0.95),
            "gap_max": max(gaps) if gaps else None,
            "seconds_mean": mean(seconds),
            "seconds_median": median(seconds),
            "seconds_p90": _quantile(seconds, 0.90),
            "seconds_max": max(seconds),
            "mean_trace_len": mean(record["trace_len"] for record in records),
            "mean_transitions": mean(record["num_transitions"] for record in records),
            "mean_duplicate_labels": mean(
                record["num_duplicate_labels"] for record in records
            ),
            "fallback_rate": sum(1 for record in records if record.get("fallback_used"))
            / len(records),
            "error_count": sum(1 for record in records if record.get("error")),
        }
    )


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
