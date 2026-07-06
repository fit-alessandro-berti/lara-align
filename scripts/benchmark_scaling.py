from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from statistics import mean, median
import sys
from time import perf_counter
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lara_align.certifier import CertifyingAlignmentSystem, LARAMode  # noqa: E402
from lara_align.checkpoint import load_checkpoint  # noqa: E402
from lara_align.decode import GreedyCandidateDecoder  # noqa: E402
from lara_align.exact import Pm4PyExactAligner  # noqa: E402
from lara_align.synthetic import generate_block_structured_example  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stress-test LARA against pm4py exact alignment on progressively "
            "larger block-structured Petri nets (concurrency, choices, loops, "
            "duplicate labels, invisible transitions) to locate the "
            "speed/quality threshold."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/lara/best.pt"))
    parser.add_argument(
        "--sizes",
        default="5,10,20,40",
        help="Comma-separated numbers of visible activities (tree leaves) per net.",
    )
    parser.add_argument(
        "--deviation-rates",
        default="0.15,0.35",
        help="Comma-separated deviation rates to cross with each size.",
    )
    parser.add_argument("--samples-per-config", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--exact-timeout",
        type=float,
        default=30.0,
        help="Per-trace timeout in seconds for the pm4py exact aligner.",
    )
    parser.add_argument(
        "--alphabet-fraction",
        type=float,
        default=0.7,
        help=(
            "Alphabet size as a fraction of the leaf count; below 1.0 forces "
            "duplicate labels."
        ),
    )
    parser.add_argument("--loop-redo-prob", type=float, default=0.3)
    parser.add_argument("--max-loop-redos", type=int, default=2)
    parser.add_argument(
        "--max-prefix-depth",
        type=int,
        default=8,
        help="Decoder bounded-search depth for enabling the next label.",
    )
    parser.add_argument(
        "--max-final-depth",
        type=int,
        default=64,
        help="Decoder bounded-search depth for reaching the final marking.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for machine-readable JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    deviation_rates = [
        float(value) for value in args.deviation_rates.split(",") if value.strip()
    ]

    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    decoder = GreedyCandidateDecoder(
        max_prefix_model_depth=args.max_prefix_depth,
        max_final_model_depth=args.max_final_depth,
    )
    system = CertifyingAlignmentSystem(model=model, decoder=decoder, device=device)
    exact = Pm4PyExactAligner()

    print("LARA Scaling Benchmark")
    print("=" * 22)
    print(f"checkpoint:        {args.checkpoint} (epoch {checkpoint.get('epoch')})")
    print(f"sizes:             {sizes}")
    print(f"deviation rates:   {deviation_rates}")
    print(f"samples/config:    {args.samples_per_config}")
    print(f"exact timeout:     {args.exact_timeout}s")
    print()

    rows: list[dict[str, Any]] = []
    for size in sizes:
        for deviation_rate in deviation_rates:
            records = _run_config(
                size,
                deviation_rate,
                args,
                system,
                exact,
            )
            row = _aggregate(size, deviation_rate, records)
            rows.append(row)
            print(_format_row(row))

    print()
    print(_format_table(rows))
    print()
    print(_threshold_summary(rows, deviation_rates))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "config": {
                "sizes": sizes,
                "deviation_rates": deviation_rates,
                "samples_per_config": args.samples_per_config,
                "exact_timeout": args.exact_timeout,
                "alphabet_fraction": args.alphabet_fraction,
                "loop_redo_prob": args.loop_redo_prob,
                "max_loop_redos": args.max_loop_redos,
                "max_prefix_depth": args.max_prefix_depth,
                "max_final_depth": args.max_final_depth,
                "seed": args.seed,
            },
            "rows": rows,
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nJSON results written to {args.output}")


def _run_config(
    size: int,
    deviation_rate: float,
    args: argparse.Namespace,
    system: CertifyingAlignmentSystem,
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

        start = perf_counter()
        with torch.no_grad():
            fast = system.align(
                example.net,
                example.initial_marking,
                example.final_marking,
                example.trace,
                mode=LARAMode.FAST,
            )
        fast_seconds = perf_counter() - start

        start = perf_counter()
        exact_result = exact.align_trace(
            example.net,
            example.initial_marking,
            example.final_marking,
            example.trace,
            timeout_seconds=args.exact_timeout,
        )
        exact_seconds = perf_counter() - start
        exact_solved = exact_result.alignment is not None and exact_result.cost is not None

        gap = None
        if fast.legal and fast.cost is not None and exact_solved:
            gap = int(fast.cost) - int(exact_result.cost)

        records.append(
            {
                "index": index,
                "trace_len": len(example.trace),
                "num_transitions": example.metadata["num_transitions"],
                "num_places": example.metadata["num_places"],
                "num_invisible": example.metadata["num_invisible"],
                "num_duplicate_labels": example.metadata["num_duplicate_labels"],
                "operator_counts": example.metadata["operator_counts"],
                "fast_seconds": fast_seconds,
                "fast_legal": bool(fast.legal),
                "fast_cost": fast.cost,
                "exact_seconds": exact_seconds,
                "exact_solved": exact_solved,
                "exact_cost": exact_result.cost,
                "exact_visited_states": (exact_result.diagnostics or {}).get(
                    "visited_states"
                ),
                "cost_gap": gap,
            }
        )
    return records


def _aggregate(size: int, deviation_rate: float, records: list[dict[str, Any]]) -> dict[str, Any]:
    fast_times = [record["fast_seconds"] for record in records]
    exact_times = [record["exact_seconds"] for record in records]
    solved = [record for record in records if record["exact_solved"]]
    legal = [record for record in records if record["fast_legal"]]
    comparable = [record for record in records if record["cost_gap"] is not None]
    gaps = [record["cost_gap"] for record in comparable]
    relative_gaps = [
        record["cost_gap"] / max(1, record["exact_cost"]) for record in comparable
    ]

    return {
        "size": size,
        "deviation_rate": deviation_rate,
        "n": len(records),
        "mean_trace_len": mean(record["trace_len"] for record in records),
        "mean_transitions": mean(record["num_transitions"] for record in records),
        "mean_places": mean(record["num_places"] for record in records),
        "mean_duplicate_labels": mean(
            record["num_duplicate_labels"] for record in records
        ),
        "fast_legal_rate": len(legal) / max(1, len(records)),
        "fast_time_mean": mean(fast_times),
        "fast_time_median": median(fast_times),
        "fast_time_max": max(fast_times),
        "exact_solved_rate": len(solved) / max(1, len(records)),
        "exact_timeouts": len(records) - len(solved),
        "exact_time_mean": mean(exact_times),
        "exact_time_median": median(exact_times),
        "exact_time_max": max(exact_times),
        "exact_visited_states_mean": _safe_mean(
            [record["exact_visited_states"] for record in solved]
        ),
        "speedup_median": median(exact_times) / max(1e-9, median(fast_times)),
        "comparable": len(comparable),
        "optimal_rate": (
            sum(1 for gap in gaps if gap == 0) / len(comparable) if comparable else None
        ),
        "gap_mean": mean(gaps) if gaps else None,
        "gap_max": max(gaps) if gaps else None,
        "gap_relative_mean": mean(relative_gaps) if relative_gaps else None,
        "records": records,
    }


def _format_row(row: dict[str, Any]) -> str:
    return (
        f"size={row['size']:>3} dev={row['deviation_rate']:.2f} "
        f"|T|={row['mean_transitions']:.0f} trace={row['mean_trace_len']:.0f} "
        f"legal={100 * row['fast_legal_rate']:.0f}% "
        f"optimal={_pct(row['optimal_rate'])} "
        f"gap={_fmt(row['gap_mean'])} "
        f"fast={1000 * row['fast_time_median']:.1f}ms "
        f"exact={1000 * row['exact_time_median']:.1f}ms "
        f"speedup={row['speedup_median']:.2f}x "
        f"timeouts={row['exact_timeouts']}"
    )


def _format_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'size':>4} {'dev':>5} {'|T|':>5} {'trace':>6} {'legal':>6} {'optimal':>8} "
        f"{'mean gap':>9} {'rel gap':>8} {'fast med':>9} {'exact med':>10} "
        f"{'speedup':>8} {'t/o':>4}"
    )
    lines = ["Summary", "-" * len(header), header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['size']:>4} {row['deviation_rate']:>5.2f} "
            f"{row['mean_transitions']:>5.0f} {row['mean_trace_len']:>6.1f} "
            f"{100 * row['fast_legal_rate']:>5.0f}% {_pct(row['optimal_rate']):>8} "
            f"{_fmt(row['gap_mean']):>9} {_fmt(row['gap_relative_mean']):>8} "
            f"{1000 * row['fast_time_median']:>7.1f}ms "
            f"{1000 * row['exact_time_median']:>8.1f}ms "
            f"{row['speedup_median']:>7.2f}x {row['exact_timeouts']:>4}"
        )
    return "\n".join(lines)


def _threshold_summary(rows: list[dict[str, Any]], deviation_rates: list[float]) -> str:
    lines = ["Speed/Quality Threshold"]
    lines.append("-" * 23)
    for deviation_rate in deviation_rates:
        matching = sorted(
            (row for row in rows if row["deviation_rate"] == deviation_rate),
            key=lambda row: row["size"],
        )
        crossover = next(
            (row for row in matching if row["speedup_median"] > 1.0), None
        )
        if crossover is None:
            lines.append(
                f"dev={deviation_rate:.2f}: exact A* stays faster at every tested "
                f"size (max speedup "
                f"{max(row['speedup_median'] for row in matching):.2f}x)"
            )
        else:
            lines.append(
                f"dev={deviation_rate:.2f}: LARA fast overtakes exact at size "
                f"{crossover['size']} "
                f"(speedup {crossover['speedup_median']:.2f}x, "
                f"legal {100 * crossover['fast_legal_rate']:.0f}%, "
                f"optimal {_pct(crossover['optimal_rate'])}, "
                f"mean gap {_fmt(crossover['gap_mean'])})"
            )
    return "\n".join(lines)


def _safe_mean(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.0f}%"


if __name__ == "__main__":
    main()
