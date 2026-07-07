from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lara_align.checkpoint import load_checkpoint  # noqa: E402
from lara_align.data import load_split  # noqa: E402
from lara_align.features import pm4py_to_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the diagnostic cost estimate (summed regional bounds) against optimal costs."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/lara_synthetic"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/lara_bidir/best.pt"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    samples = load_split(args.data_dir, args.split)

    records = []
    with torch.no_grad():
        for sample in samples:
            features = pm4py_to_features(
                sample.net,
                sample.initial_marking,
                sample.final_marking,
                sample.trace,
            ).to(device)
            output = model(features)
            lower = float(output.local_lower_bounds.sum())
            upper = float(output.local_upper_bounds.sum())
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "family": sample.metadata.get("family"),
                    "optimal_cost": sample.optimal_cost,
                    "estimate_lower": lower,
                    "estimate_upper": upper,
                }
            )

    deltas = [r["optimal_cost"] for r in records]
    uppers = [r["estimate_upper"] for r in records]
    lowers = [r["estimate_lower"] for r in records]
    n = len(records)
    abs_err = [abs(u - d) for u, d in zip(uppers, deltas)]
    signed_err = [u - d for u, d in zip(uppers, deltas)]
    covered = [lo <= d <= up for lo, up, d in zip(lowers, uppers, deltas)]
    widths = [up - lo for lo, up in zip(lowers, uppers)]

    fitting = [u for u, d in zip(uppers, deltas) if d == 0]
    deviating = [u for u, d in zip(uppers, deltas) if d > 0]
    # threshold classification: estimate > 0.5 predicts "trace deviates"
    correct = sum(1 for u, d in zip(uppers, deltas) if (u > 0.5) == (d > 0))

    metrics = {
        "n": n,
        "mae_upper_vs_optimal": sum(abs_err) / n,
        "mean_signed_error": sum(signed_err) / n,
        "pearson_r": pearson(uppers, deltas),
        "coverage_rate": sum(covered) / n,
        "mean_interval_width": sum(widths) / n,
        "mean_estimate_fitting": sum(fitting) / len(fitting) if fitting else None,
        "mean_estimate_deviating": sum(deviating) / len(deviating) if deviating else None,
        "deviation_detection_accuracy_at_0.5": correct / n,
    }

    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps({"metrics": metrics, "records": records}, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
