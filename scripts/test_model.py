from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lara_align.certifier import CertifyingAlignmentSystem, LARAMode  # noqa: E402
from lara_align.checkpoint import load_checkpoint  # noqa: E402
from lara_align.data import AlignmentSample, load_split  # noqa: E402
from lara_align.features import pm4py_to_features  # noqa: E402
from lara_align.training import LARALoss, targets_from_alignment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LARA model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/lara_synthetic"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/lara/best.pt"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metrics-output", type=Path, default=None)
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
    metrics = evaluate_model(
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

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def evaluate_model(
    model,
    samples: list[AlignmentSample],
    criterion: LARALoss,
    device: torch.device,
    run_certified: bool = False,
) -> dict[str, float | int | str | None]:
    model.eval()
    system = CertifyingAlignmentSystem(model=model, device=device)

    loss_totals: dict[str, float] = {}
    legal = 0
    optimal_cost = 0
    total_gap = 0.0
    legal_gap_count = 0
    certified_optimal = 0

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
            if fast.legal and fast.cost is not None:
                legal += 1
                gap = fast.cost - sample.optimal_cost
                total_gap += gap
                legal_gap_count += 1
                if gap == 0:
                    optimal_cost += 1

            if run_certified:
                certified = system.align(
                    sample.net,
                    sample.initial_marking,
                    sample.final_marking,
                    sample.trace,
                    mode=LARAMode.CERTIFIED,
                )
                if certified.certified_optimal:
                    certified_optimal += 1

    count = max(1, len(samples))
    averaged_losses = {f"loss_{key}": value / count for key, value in loss_totals.items()}
    return {
        "samples": len(samples),
        **averaged_losses,
        "fast_legal_rate": legal / count,
        "fast_optimal_cost_rate": optimal_cost / count,
        "fast_mean_gap_on_legal": (
            total_gap / legal_gap_count if legal_gap_count else None
        ),
        "certified_optimal_rate": (
            certified_optimal / count if run_certified else None
        ),
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


if __name__ == "__main__":
    main()
