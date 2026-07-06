from __future__ import annotations

import argparse
from pathlib import Path
from random import Random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lara_align.data import (  # noqa: E402
    SPLITS,
    AlignmentSample,
    DatasetMetadata,
    save_metadata,
    save_split,
)
from lara_align.exact import Pm4PyExactAligner  # noqa: E402
from lara_align.synthetic import (  # noqa: E402
    generate_sequence_example,
    inject_deviations,
    make_duplicate_label_choice_net,
    trace_from_labels,
)
from lara_align.verify import verify_alignment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize exact-labeled LARA train/validation/test data."
    )
    parser.add_argument("--output", type=Path, default=Path("data/lara_synthetic"))
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--deviation-rate", type=float, default=0.25)
    parser.add_argument("--duplicate-fraction", type=float, default=0.2)
    parser.add_argument("--exact-timeout", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _prepare_output_dir(args.output, args.overwrite)

    rng = Random(args.seed)
    exact = Pm4PyExactAligner()
    split_sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }

    for split in SPLITS:
        samples = _generate_split(split, split_sizes[split], args, rng, exact)
        save_split(args.output, split, samples)
        print(f"wrote {len(samples):4d} {split} samples to {args.output / f'{split}.pkl'}")

    save_metadata(
        args.output,
        DatasetMetadata(
            version=1,
            split_counts=split_sizes,
            seed=args.seed,
            generator="mixed_sequence_duplicate_choice",
            parameters={
                "min_len": args.min_len,
                "max_len": args.max_len,
                "deviation_rate": args.deviation_rate,
                "duplicate_fraction": args.duplicate_fraction,
                "exact_timeout": args.exact_timeout,
            },
        ),
    )
    print(f"metadata written to {args.output / 'metadata.pkl'}")


def _prepare_output_dir(output: Path, overwrite: bool) -> None:
    existing = [output / f"{split}.pkl" for split in SPLITS]
    if any(path.exists() for path in existing) and not overwrite:
        raise SystemExit(
            f"{output} already contains split files; pass --overwrite to replace them"
        )
    output.mkdir(parents=True, exist_ok=True)


def _generate_split(
    split: str,
    target_size: int,
    args: argparse.Namespace,
    rng: Random,
    exact: Pm4PyExactAligner,
) -> list[AlignmentSample]:
    samples: list[AlignmentSample] = []
    attempts = 0
    max_attempts = max(100, target_size * 20)

    while len(samples) < target_size and attempts < max_attempts:
        attempts += 1
        example = _generate_example(args, rng)
        result = exact.align_trace(
            example.net,
            example.initial_marking,
            example.final_marking,
            example.trace,
            timeout_seconds=args.exact_timeout,
        )
        if result.alignment is None or result.cost is None or not result.optimal:
            continue
        verifier = verify_alignment(
            result.alignment,
            example.net,
            example.initial_marking,
            example.final_marking,
            example.trace,
        )
        if not verifier.legal:
            continue

        sample_index = len(samples)
        samples.append(
            AlignmentSample(
                sample_id=f"{split}-{sample_index:06d}",
                split=split,
                net=example.net,
                initial_marking=example.initial_marking,
                final_marking=example.final_marking,
                trace=example.trace,
                optimal_alignment=result.alignment,
                optimal_cost=verifier.cost,
                metadata={
                    **example.metadata,
                    "exact_diagnostics": result.diagnostics or {},
                    "attempt": attempts,
                },
            )
        )

    if len(samples) != target_size:
        raise SystemExit(
            f"generated {len(samples)} of {target_size} requested {split} samples "
            f"after {attempts} attempts"
        )
    return samples


def _generate_example(args: argparse.Namespace, rng: Random):
    if rng.random() < args.duplicate_fraction:
        suffix = rng.choice(["B", "C"])
        net, initial_marking, final_marking = make_duplicate_label_choice_net()
        fitting_labels = ["A", suffix]
        deviated_labels = inject_deviations(
            fitting_labels,
            rng,
            args.deviation_rate,
            ("A", "B", "C", "X"),
        )
        return _AnonymousExample(
            net=net,
            initial_marking=initial_marking,
            final_marking=final_marking,
            trace=trace_from_labels(deviated_labels),
            metadata={
                "family": "duplicate_label_choice",
                "labels": fitting_labels,
                "deviated_labels": deviated_labels,
            },
        )
    return generate_sequence_example(
        rng=rng,
        min_len=args.min_len,
        max_len=args.max_len,
        deviation_rate=args.deviation_rate,
    )


class _AnonymousExample:
    def __init__(self, net, initial_marking, final_marking, trace, metadata):
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.trace = trace
        self.metadata = metadata


if __name__ == "__main__":
    main()
