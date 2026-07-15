from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from random import Random
import sys
from time import perf_counter
from typing import Any

import torch

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is unavailable.
    _tqdm = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lara_align.checkpoint import build_model, save_checkpoint  # noqa: E402
from lara_align.data import AlignmentSample, load_split  # noqa: E402
from lara_align.features import (  # noqa: E402
    DEFAULT_HASH_VOCAB_SIZE,
    PetriTraceFeatures,
    pm4py_to_features,
)
from lara_align.training import LARALoss, targets_from_alignment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LARA neural model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/lara_synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lara"))
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help=(
            "Write per-epoch training metrics to this CSV path. "
            "Defaults to OUTPUT_DIR/metrics.csv."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--trace-layers", type=int, default=1)
    parser.add_argument("--num-regions", type=int, default=6)
    parser.add_argument("--sketches-per-region", type=int, default=4)
    parser.add_argument("--num-edge-types", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--move-weight", type=float, default=1.0)
    parser.add_argument("--cost-weight", type=float, default=0.03)
    parser.add_argument("--cost-beta", type=float, default=1.0)
    parser.add_argument("--bce-label-smoothing", type=float, default=0.03)
    parser.add_argument("--sync-label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--label-remap-probability",
        type=float,
        default=0.5,
        help=(
            "Probability of consistently renaming activity IDs within each training "
            "sample; preserves equality while discouraging synthetic-label memorization."
        ),
    )
    parser.add_argument("--router-boundary-weight", type=float, default=0.01)
    parser.add_argument("--router-balance-weight", type=float, default=0.01)
    parser.add_argument("--router-entropy-weight", type=float, default=0.001)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--representation-kind",
        action="append",
        default=None,
        help="Train/validate only on this representation kind; repeat as needed.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.label_remap_probability <= 1.0:
        raise SystemExit("label-remap-probability must be in [0, 1]")
    torch.manual_seed(args.seed)
    rng = Random(args.seed)
    device = torch.device(args.device)

    progress_enabled = not args.no_progress
    _progress_write(f"loading data from {args.data_dir}")
    train_samples, val_samples = _load_training_splits(args.data_dir, progress_enabled)
    if args.representation_kind:
        selected = set(args.representation_kind)
        train_samples = [
            sample
            for sample in train_samples
            if sample.metadata.get("representation_kind") in selected
        ]
        val_samples = [
            sample
            for sample in val_samples
            if sample.metadata.get("representation_kind") in selected
        ]
    if args.max_train_samples is not None:
        train_samples = train_samples[: max(0, args.max_train_samples)]
    if args.max_val_samples is not None:
        val_samples = val_samples[: max(0, args.max_val_samples)]
    if not train_samples or not val_samples:
        raise SystemExit("training filters produced an empty train or validation split")
    _progress_write(
        f"loaded data: train={len(train_samples)} samples, val={len(val_samples)} samples"
    )
    model_config = _model_config(args)
    model = build_model(model_config).to(device)
    criterion = LARALoss(
        move_weight=args.move_weight,
        cost_weight=args.cost_weight,
        router_boundary_weight=args.router_boundary_weight,
        router_balance_weight=args.router_balance_weight,
        router_entropy_weight=args.router_entropy_weight,
        cost_beta=args.cost_beta,
        bce_label_smoothing=args.bce_label_smoothing,
        sync_label_smoothing=args.sync_label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        threshold=args.min_delta,
        min_lr=args.min_lr,
    )

    best_val = float("inf")
    epochs_without_improvement = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"
    metrics_csv_path = _metrics_csv_path(args)
    metrics_csv_initialized = False

    _progress_write(f"starting training for up to {args.epochs} epochs")
    with _progress(
        range(1, args.epochs + 1),
        enabled=progress_enabled,
        desc="epochs",
        unit="epoch",
        dynamic_ncols=True,
    ) as epoch_progress:
        for epoch in epoch_progress:
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
                split_name="train",
                progress_enabled=progress_enabled,
                label_remap_probability=args.label_remap_probability,
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
                split_name="val",
                progress_enabled=progress_enabled,
                label_remap_probability=0.0,
            )
            elapsed = perf_counter() - start
            generalization_gap = _generalization_gap(train_metrics, val_metrics)
            metrics = {
                "train": train_metrics,
                "val": val_metrics,
                "generalization_gap": generalization_gap,
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
            scheduler.step(val_metrics["total"])
            lr = optimizer.param_groups[0]["lr"]
            _write_metrics_csv_row(
                metrics_csv_path,
                _metrics_csv_row(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    elapsed_seconds=elapsed,
                    lr=lr,
                    best_val=best_val,
                    improved=improved,
                    epochs_without_improvement=epochs_without_improvement,
                ),
                include_header=not metrics_csv_initialized,
            )
            metrics_csv_initialized = True
            epoch_progress.set_postfix(
                train=f"{train_metrics['total']:.4f}",
                val=f"{val_metrics['total']:.4f}",
                lr=f"{lr:.2e}",
                best=f"{best_val:.4f}",
            )

            _progress_write(
                f"epoch {epoch:03d} "
                f"train_total={train_metrics['total']:.4f} "
                f"val_total={val_metrics['total']:.4f} "
                f"val_move={val_metrics['move']:.4f} "
                f"val_cost={val_metrics['cost']:.4f} "
                f"val_log={val_metrics['log_move']:.4f} "
                f"val_model={val_metrics['model_move']:.4f} "
                f"gap={generalization_gap['total']:+.4f} "
                f"lr={lr:.2e} "
                f"best_val={best_val:.4f} "
                f"time={elapsed:.1f}s"
            )

            if args.patience > 0 and epochs_without_improvement >= args.patience:
                _progress_write(
                    f"early stopping after {args.patience} epochs without improvement"
                )
                break

    _progress_write(f"best checkpoint: {best_path}")
    _progress_write(f"last checkpoint: {last_path}")
    _progress_write(f"metrics csv: {metrics_csv_path}")


class _NoOpProgress:
    def __init__(self, iterable=None, **_: Any) -> None:
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def update(self, _: int = 1) -> None:
        pass

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        pass


class _ProgressFile:
    def __init__(self, handle, progress) -> None:
        self.handle = handle
        self.progress = progress

    def read(self, size: int = -1) -> bytes:
        chunk = self.handle.read(size)
        self.progress.update(len(chunk))
        return chunk

    def readline(self, size: int = -1) -> bytes:
        line = self.handle.readline(size)
        self.progress.update(len(line))
        return line


def _progress(iterable=None, *, enabled: bool, **kwargs):
    if enabled and _tqdm is not None:
        return _tqdm(iterable, **kwargs)
    return _NoOpProgress(iterable, **kwargs)


def _progress_write(message: str) -> None:
    if _tqdm is not None:
        _tqdm.write(message)
    else:
        print(message)


def _load_training_splits(
    data_dir: Path,
    progress_enabled: bool,
) -> tuple[list[AlignmentSample], list[AlignmentSample]]:
    return (
        _load_split_with_progress(data_dir, "train", progress_enabled),
        _load_split_with_progress(data_dir, "val", progress_enabled),
    )


def _load_split_with_progress(
    data_dir: Path,
    split: str,
    progress_enabled: bool,
) -> list[AlignmentSample]:
    if not progress_enabled or _tqdm is None:
        return load_split(data_dir, split)

    split_path = Path(data_dir) / f"{split}.pkl"
    total_bytes = split_path.stat().st_size
    with split_path.open("rb") as handle:
        with _progress(
            enabled=progress_enabled,
            total=total_bytes,
            desc=f"loading {split}",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress:
            samples = pickle.load(_ProgressFile(handle, progress))
    return list(samples)


def _run_epoch(
    model,
    samples: list[AlignmentSample],
    criterion: LARALoss,
    device: torch.device,
    optimizer,
    rng: Random,
    max_grad_norm: float,
    batch_size: int,
    split_name: str,
    progress_enabled: bool,
    label_remap_probability: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    ordered = list(samples)
    if is_train:
        rng.shuffle(ordered)

    totals: dict[str, float] = {}
    seen = 0
    step_size = max(1, batch_size)
    batch_starts = range(0, len(ordered), step_size)
    with _progress(
        batch_starts,
        enabled=progress_enabled,
        desc=split_name,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ) as batch_progress:
        for start in batch_progress:
            batch = ordered[start : start + step_size]
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            for sample in batch:
                with torch.set_grad_enabled(is_train):
                    remap_labels = (
                        is_train
                        and label_remap_probability > 0
                        and rng.random() < label_remap_probability
                    )
                    losses = _sample_losses(
                        model,
                        criterion,
                        sample,
                        device,
                        remap_labels=remap_labels,
                        rng=rng,
                    )
                    if is_train:
                        (losses["total"] / len(batch)).backward()

                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

            if is_train:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            seen += len(batch)
            batch_progress.set_postfix(
                total=f"{totals.get('total', 0.0) / max(1, seen):.4f}"
            )

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
    *,
    remap_labels: bool = False,
    rng: Random | None = None,
) -> dict[str, torch.Tensor]:
    features = pm4py_to_features(
        sample.net,
        sample.initial_marking,
        sample.final_marking,
        sample.trace,
    ).to(device)
    if remap_labels:
        if rng is None:
            raise ValueError("rng is required when remap_labels is enabled")
        features = _remap_activity_ids(features, rng)
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


def _remap_activity_ids(
    features: PetriTraceFeatures,
    rng: Random,
    *,
    hash_vocab_size: int = DEFAULT_HASH_VOCAB_SIZE,
) -> PetriTraceFeatures:
    """Apply a per-sample bijection to visible-label IDs without changing equality."""

    if hash_vocab_size <= 1:
        raise ValueError("hash_vocab_size must be greater than 1")
    all_ids = torch.cat(
        [features.transition_label_ids, features.event_label_ids], dim=0
    )
    visible_ids = sorted(
        int(value) for value in torch.unique(all_ids).tolist() if int(value) != 0
    )
    if len(visible_ids) > hash_vocab_size - 1:
        raise ValueError("sample has more visible labels than the hash vocabulary")
    replacements = rng.sample(range(1, hash_vocab_size), len(visible_ids))

    original_transition_ids = features.transition_label_ids
    original_event_ids = features.event_label_ids
    remapped_transition_ids = original_transition_ids.clone()
    remapped_event_ids = original_event_ids.clone()
    for source, target in zip(visible_ids, replacements):
        remapped_transition_ids[original_transition_ids == source] = target
        remapped_event_ids[original_event_ids == source] = target
    features.transition_label_ids = remapped_transition_ids
    features.event_label_ids = remapped_event_ids
    return features


def _model_config(args: argparse.Namespace) -> dict:
    return {
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "graph_layers": args.graph_layers,
        "trace_layers": args.trace_layers,
        "num_regions": args.num_regions,
        "sketches_per_region": args.sketches_per_region,
        "dropout": args.dropout,
        "num_edge_types": args.num_edge_types,
    }


def _metrics_csv_path(args: argparse.Namespace) -> Path:
    if args.metrics_csv is not None:
        return args.metrics_csv
    return args.output_dir / "metrics.csv"


def _metrics_csv_row(
    *,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    elapsed_seconds: float,
    lr: float,
    best_val: float,
    improved: bool,
    epochs_without_improvement: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "elapsed_seconds": elapsed_seconds,
        "lr": lr,
        "best_val": best_val,
        "improved": int(improved),
        "epochs_without_improvement": epochs_without_improvement,
    }
    row.update(_prefixed_metrics("train", train_metrics))
    row.update(_prefixed_metrics("val", val_metrics))
    row.update(
        {
            f"gap_{key}": value
            for key, value in _generalization_gap(train_metrics, val_metrics).items()
        }
    )
    return row


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": metrics[key] for key in sorted(metrics)}


def _generalization_gap(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> dict[str, float]:
    return {
        key: val_metrics[key] - train_metrics[key]
        for key in sorted(train_metrics.keys() & val_metrics.keys())
        if key != "samples"
    }


def _write_metrics_csv_row(
    path: Path,
    row: dict[str, Any],
    *,
    include_header: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if include_header else "a"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if include_header:
            writer.writeheader()
        writer.writerow(row)


def _jsonable_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values


if __name__ == "__main__":
    main()
