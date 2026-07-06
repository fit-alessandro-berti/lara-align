from __future__ import annotations

import argparse
from pathlib import Path
from random import Random
import sys
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lara_align.checkpoint import build_model, save_checkpoint  # noqa: E402
from lara_align.data import AlignmentSample, load_split  # noqa: E402
from lara_align.features import pm4py_to_features  # noqa: E402
from lara_align.training import LARALoss, targets_from_alignment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LARA neural model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/lara_synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lara"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--trace-layers", type=int, default=1)
    parser.add_argument("--num-regions", type=int, default=4)
    parser.add_argument("--sketches-per-region", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--move-weight", type=float, default=1.0)
    parser.add_argument("--cost-weight", type=float, default=0.03)
    parser.add_argument("--cost-beta", type=float, default=1.0)
    parser.add_argument("--router-boundary-weight", type=float, default=0.01)
    parser.add_argument("--router-balance-weight", type=float, default=0.01)
    parser.add_argument("--router-entropy-weight", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = Random(args.seed)
    device = torch.device(args.device)

    train_samples = load_split(args.data_dir, "train")
    val_samples = load_split(args.data_dir, "val")
    model_config = _model_config(args)
    model = build_model(model_config).to(device)
    criterion = LARALoss(
        move_weight=args.move_weight,
        cost_weight=args.cost_weight,
        router_boundary_weight=args.router_boundary_weight,
        router_balance_weight=args.router_balance_weight,
        router_entropy_weight=args.router_entropy_weight,
        cost_beta=args.cost_beta,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    epochs_without_improvement = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        train_metrics = _run_epoch(
            model,
            train_samples,
            criterion,
            device,
            optimizer=optimizer,
            rng=rng,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size,
        )
        val_metrics = _run_epoch(
            model,
            val_samples,
            criterion,
            device,
            optimizer=None,
            rng=rng,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size,
        )
        elapsed = perf_counter() - start
        metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_seconds": elapsed,
            "args": _jsonable_args(args),
        }
        save_checkpoint(last_path, model, model_config, epoch, metrics)

        improved = val_metrics["total"] < best_val - args.min_delta
        if improved:
            best_val = val_metrics["total"]
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, model_config, epoch, metrics)
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:03d} "
            f"train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"val_move={val_metrics['move']:.4f} "
            f"val_cost={val_metrics['cost']:.4f} "
            f"val_log={val_metrics['log_move']:.4f} "
            f"val_model={val_metrics['model_move']:.4f} "
            f"best_val={best_val:.4f} "
            f"time={elapsed:.1f}s"
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early stopping after {args.patience} epochs without improvement")
            break

    print(f"best checkpoint: {best_path}")
    print(f"last checkpoint: {last_path}")


def _run_epoch(
    model,
    samples: list[AlignmentSample],
    criterion: LARALoss,
    device: torch.device,
    optimizer,
    rng: Random,
    max_grad_norm: float,
    batch_size: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    ordered = list(samples)
    if is_train:
        rng.shuffle(ordered)

    totals: dict[str, float] = {}
    step_size = max(1, batch_size)
    for start in range(0, len(ordered), step_size):
        batch = ordered[start : start + step_size]
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        for sample in batch:
            with torch.set_grad_enabled(is_train):
                losses = _sample_losses(model, criterion, sample, device)
                if is_train:
                    (losses["total"] / len(batch)).backward()

            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

        if is_train:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    count = max(1, len(ordered))
    averaged = {key: value / count for key, value in totals.items()}
    averaged["move"] = (
        averaged.get("sync_move", 0.0)
        + averaged.get("log_move", 0.0)
        + averaged.get("model_move", 0.0)
    )
    averaged["samples"] = float(len(ordered))
    return averaged


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


def _model_config(args: argparse.Namespace) -> dict:
    return {
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "graph_layers": args.graph_layers,
        "trace_layers": args.trace_layers,
        "num_regions": args.num_regions,
        "sketches_per_region": args.sketches_per_region,
        "dropout": args.dropout,
    }


def _jsonable_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values


if __name__ == "__main__":
    main()
