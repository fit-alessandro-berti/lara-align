from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from typing import Any

os.environ.setdefault("PM4PY_SHOW_PROGRESS_BAR", "False")
sys._pm4py_welcome_shown = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from pm4py.objects.log.importer.xes import importer as xes_importer  # noqa: E402

from lara_align.certifier import CertifyingAlignmentSystem, LARAMode  # noqa: E402
from lara_align.checkpoint import load_checkpoint  # noqa: E402
from lara_align.decode import GreedyCandidateDecoder  # noqa: E402
from lara_align.exact import Pm4PyExactAligner  # noqa: E402
from lara_align.verify import trace_labels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot validation of LARA on a real-life XES event log. "
            "Discovers a Petri net with the inductive miner, then compares "
            "guided fast mode, unguided fast mode, and pm4py exact alignment "
            "per trace variant."
        )
    )
    parser.add_argument("event_log", type=Path, help="Path to the XES event log.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/lara/best.pt"))
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=0.0,
        help=(
            "Inductive-miner noise threshold. 0.0 guarantees a perfectly "
            "fitting model; higher values filter behavior so that some "
            "variants deviate."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exact-timeout", type=float, default=30.0)
    parser.add_argument("--activity-key", default="concept:name")
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Optionally cap the number of (most frequent) variants evaluated.",
    )
    parser.add_argument("--max-prefix-depth", type=int, default=8)
    parser.add_argument("--max-final-depth", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    decoder = GreedyCandidateDecoder(
        max_prefix_model_depth=args.max_prefix_depth,
        max_final_model_depth=args.max_final_depth,
    )
    guided = CertifyingAlignmentSystem(model=model, decoder=decoder, device=device)
    unguided = CertifyingAlignmentSystem(
        model=model, decoder=decoder, device=device, use_guidance=False
    )
    exact = Pm4PyExactAligner()

    log = xes_importer.apply(
        str(args.event_log), parameters={"show_progress_bar": False}
    )
    variants = _variants(log, args.activity_key)
    if args.max_variants is not None:
        variants = variants[: args.max_variants]

    import pm4py

    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(
        log, noise_threshold=args.noise_threshold
    )
    num_visible = sum(1 for t in net.transitions if t.label is not None)
    num_invisible = len(net.transitions) - num_visible
    labels_in_log = {label for _, labels, _ in variants for label in labels}
    labels_in_net = {
        str(t.label) for t in net.transitions if t.label is not None
    }

    records: list[dict[str, Any]] = []
    for variant_index, (representative, labels, count) in enumerate(variants):
        record: dict[str, Any] = {
            "variant": variant_index,
            "trace_count": count,
            "trace_len": len(labels),
        }

        start = perf_counter()
        exact_result = exact.align_trace(
            net,
            initial_marking,
            final_marking,
            representative,
            activity_key=args.activity_key,
            timeout_seconds=args.exact_timeout,
        )
        record["exact_seconds"] = perf_counter() - start
        record["exact_solved"] = (
            exact_result.alignment is not None and exact_result.cost is not None
        )
        record["exact_cost"] = exact_result.cost

        for key, system in (("guided", guided), ("unguided", unguided)):
            start = perf_counter()
            result = system.align(
                net,
                initial_marking,
                final_marking,
                representative,
                mode=LARAMode.FAST,
                activity_key=args.activity_key,
            )
            record[f"{key}_seconds"] = perf_counter() - start
            record[f"{key}_legal"] = bool(result.legal)
            record[f"{key}_cost"] = result.cost
            record[f"{key}_gap"] = (
                int(result.cost) - int(exact_result.cost)
                if result.legal and result.cost is not None and record["exact_solved"]
                else None
            )
        records.append(record)

    summary = {
        "log": str(args.event_log),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "noise_threshold": args.noise_threshold,
        "traces": len(log),
        "variants": len(variants),
        "traces_covered": sum(count for _, _, count in variants),
        "net_places": len(net.places),
        "net_visible_transitions": num_visible,
        "net_invisible_transitions": num_invisible,
        "log_labels": len(labels_in_log),
        "net_labels": len(labels_in_net),
        "mean_trace_len": mean(record["trace_len"] for record in records),
        "max_trace_len": max(record["trace_len"] for record in records),
        "exact": _method_summary(records, "exact"),
        "guided": _method_summary(records, "guided"),
        "unguided": _method_summary(records, "unguided"),
    }

    print(_format_report(summary))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "records": records}, indent=2, sort_keys=True)
        )
        print(f"\nJSON results written to {args.output}")


def _variants(log, activity_key: str) -> list[tuple[object, list[str], int]]:
    """Group traces into variants, most frequent first."""

    grouped: dict[tuple[str, ...], list[object]] = {}
    for trace in log:
        labels = tuple(trace_labels(trace, activity_key=activity_key))
        if not labels:
            continue
        grouped.setdefault(labels, []).append(trace)
    ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    return [(traces[0], list(labels), len(traces)) for labels, traces in ordered]


def _method_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    times = [record[f"{key}_seconds"] for record in records]
    result: dict[str, Any] = {
        "time_mean": mean(times),
        "time_median": median(times),
        "time_max": max(times),
    }
    if key == "exact":
        result["solved_variants"] = sum(1 for r in records if r["exact_solved"])
        result["timeouts"] = sum(1 for r in records if not r["exact_solved"])
        costs = [r["exact_cost"] for r in records if r["exact_solved"]]
        result["mean_cost"] = mean(costs) if costs else None
        return result

    legal_variants = sum(1 for r in records if r[f"{key}_legal"])
    comparable = [r for r in records if r[f"{key}_gap"] is not None]
    gaps = [r[f"{key}_gap"] for r in comparable]
    total_traces = sum(r["trace_count"] for r in records)
    result.update(
        {
            "legal_variants": legal_variants,
            "legal_variant_rate": legal_variants / max(1, len(records)),
            "legal_trace_rate": sum(
                r["trace_count"] for r in records if r[f"{key}_legal"]
            )
            / max(1, total_traces),
            "optimal_variant_rate": (
                sum(1 for g in gaps if g == 0) / len(comparable) if comparable else None
            ),
            "optimal_trace_rate": (
                sum(r["trace_count"] for r in comparable if r[f"{key}_gap"] == 0)
                / max(
                    1,
                    sum(r["trace_count"] for r in comparable),
                )
                if comparable
                else None
            ),
            "gap_mean": mean(gaps) if gaps else None,
            "gap_median": median(gaps) if gaps else None,
            "gap_max": max(gaps) if gaps else None,
            "gap_relative_mean": (
                mean(g / max(1, r["exact_cost"]) for g, r in zip(gaps, comparable))
                if gaps
                else None
            ),
        }
    )
    return result


def _format_report(summary: dict[str, Any]) -> str:
    lines = [
        "LARA Real-Log Validation",
        "=" * 24,
        f"log:                {summary['log']}",
        f"checkpoint:         {summary['checkpoint']} (epoch {summary['checkpoint_epoch']})",
        f"discovery:          inductive miner, noise={summary['noise_threshold']}",
        (
            f"net:                {summary['net_places']} places, "
            f"{summary['net_visible_transitions']} visible + "
            f"{summary['net_invisible_transitions']} invisible transitions"
        ),
        (
            f"log:                {summary['traces']} traces, "
            f"{summary['variants']} variants, "
            f"{summary['log_labels']} activity labels, "
            f"trace len mean/max {summary['mean_trace_len']:.1f}/{summary['max_trace_len']}"
        ),
        "",
        (
            f"{'method':<10} {'legal(var)':>10} {'legal(tr)':>10} {'opt(var)':>9} "
            f"{'opt(tr)':>8} {'gap mean':>9} {'gap max':>8} {'med ms':>7} {'max ms':>8}"
        ),
        "-" * 88,
    ]
    exact_summary = summary["exact"]
    lines.append(
        f"{'exact':<10} {'-':>10} {'-':>10} {'-':>9} {'-':>8} "
        f"{'-':>9} {'-':>8} "
        f"{1000 * exact_summary['time_median']:>7.1f} "
        f"{1000 * exact_summary['time_max']:>8.1f}"
    )
    for key in ("guided", "unguided"):
        method = summary[key]
        lines.append(
            f"{key:<10} "
            f"{_pct(method['legal_variant_rate']):>10} "
            f"{_pct(method['legal_trace_rate']):>10} "
            f"{_pct(method['optimal_variant_rate']):>9} "
            f"{_pct(method['optimal_trace_rate']):>8} "
            f"{_fmt(method['gap_mean']):>9} "
            f"{_fmt(method['gap_max']):>8} "
            f"{1000 * method['time_median']:>7.1f} "
            f"{1000 * method['time_max']:>8.1f}"
        )
    if exact_summary["timeouts"]:
        lines.append(f"\nexact timeouts: {exact_summary['timeouts']}")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


if __name__ == "__main__":
    main()
