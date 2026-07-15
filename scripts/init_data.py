from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lara_align.data import (  # noqa: E402
    SPLITS,
    AlignmentSample,
    DatasetMetadata,
    save_metadata,
    save_split,
)
from lara_align.exact import Pm4PyExactAligner  # noqa: E402
from lara_align.families import (  # noqa: E402
    BehaviorFamilyConfig,
    generate_behavior_family,
    transition_identity_identifiable,
)
from lara_align.synthetic import trace_from_labels  # noqa: E402
from lara_align.verify import verify_alignment  # noqa: E402


PRESETS = (
    "smoke",
    "balanced_train",
    "iid_behavior",
    "equivalence_train",
    "equivalence_test",
    "equivalence_seen",
    "equivalence_unseen",
    "nonblock_ood",
    "scale_ood",
    "noise_ood",
    "sampling_ood",
    "loops_bounded",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize exact-labeled, behavior-family LARA data."
    )
    parser.add_argument("--output", type=Path, default=Path("data/lara_synthetic"))
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--train-families", type=int, default=None)
    parser.add_argument("--val-families", type=int, default=None)
    parser.add_argument("--test-families", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generator-config", type=Path, default=None)
    parser.add_argument("--preset", choices=PRESETS, default=None)
    parser.add_argument("--motif-weights", default=None, help="Comma-separated KIND=WEIGHT values.")
    parser.add_argument("--variants-per-behavior", type=int, default=None)
    parser.add_argument("--traces-per-behavior", type=int, default=None)
    parser.add_argument("--clean-fraction", type=float, default=None)
    parser.add_argument("--edit-count-weights", default=None, help="For example 1=0.5,2=0.3,3=0.2.")
    parser.add_argument("--min-visible-occurrences", type=int, default=None)
    parser.add_argument("--max-visible-occurrences", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--exact-timeout", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")

    # Compatibility aliases from the previous initializer. They remain optional
    # and translate into the closest behavior-family controls when supplied.
    parser.add_argument("--min-len", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-len", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--deviation-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--duplicate-fraction", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _prepare_output_dir(args.output, args.overwrite)
    config = _config_from_args(args)
    exact = Pm4PyExactAligner()
    rows_per_family = (
        config.representations.variants_per_behavior * config.logs.traces_per_behavior
    )
    split_sizes = {
        "train": (
            _positive(args.train_families, "train-families") * rows_per_family
            if args.train_families is not None
            else _positive(args.train_size, "train-size")
        ),
        "val": (
            _positive(args.val_families, "val-families") * rows_per_family
            if args.val_families is not None
            else _positive(args.val_size, "val-size")
        ),
        "test": (
            _positive(args.test_families, "test-families") * rows_per_family
            if args.test_families is not None
            else _positive(args.test_size, "test-size")
        ),
    }

    rejection_totals: Counter[str] = Counter()
    split_family_counts: dict[str, int] = {}
    for split in SPLITS:
        samples, rejections, family_count = _generate_family_split(
            split,
            split_sizes[split],
            config,
            exact,
            exact_timeout=args.exact_timeout,
            progress_every=args.progress_every,
        )
        rejection_totals.update(rejections)
        split_family_counts[split] = family_count
        save_split(args.output, split, samples)
        print(
            f"wrote {len(samples):4d} {split} samples from {family_count:4d} "
            f"behavior families to {args.output / f'{split}.pkl'}"
        )

    save_metadata(
        args.output,
        DatasetMetadata(
            version=2,
            split_counts=split_sizes,
            seed=config.seed,
            generator="behavioral_equivalence_families",
            parameters={
                "config": config.to_dict(),
                "exact_timeout": args.exact_timeout,
                "progress_every": args.progress_every,
                "split_unit": "behavior_id_before_variant_expansion",
                "family_counts": split_family_counts,
                "rejected_family_reasons": dict(sorted(rejection_totals.items())),
                "cost_consistency": "asserted_for_exact_equivalence_families",
            },
        ),
    )
    print(f"metadata written to {args.output / 'metadata.pkl'}")


def _generate_family_split(
    split: str,
    target_size: int,
    config: BehaviorFamilyConfig,
    exact: Pm4PyExactAligner,
    *,
    exact_timeout: float | None,
    progress_every: int,
) -> tuple[list[AlignmentSample], Counter[str], int]:
    samples: list[AlignmentSample] = []
    rejections: Counter[str] = Counter()
    family_index = 0
    accepted_families = 0
    alignment_cache: dict[tuple[object, ...], tuple[object, float]] = {}
    max_attempts = max(100, target_size * 20)

    while len(samples) < target_size and family_index < max_attempts:
        current_index = family_index
        family_index += 1
        try:
            family = generate_behavior_family(config, current_index, split)
        except Exception as exc:
            rejections[f"generation:{type(exc).__name__}"] += 1
            continue

        family_rows: list[tuple[object, object, object, object, float, bool]] = []
        family_failed = False
        for observed in family.noisy_traces:
            costs: list[int] = []
            trace = trace_from_labels(observed.labels)
            for variant in family.model_variants:
                cache_key = (
                    _net_signature(
                        variant.net,
                        variant.initial_marking,
                        variant.final_marking,
                    ),
                    tuple(observed.labels),
                    "standard_cost_v1",
                )
                cached = alignment_cache.get(cache_key)
                if cached is None:
                    exact_start = perf_counter()
                    try:
                        result = exact.align_trace(
                            variant.net,
                            variant.initial_marking,
                            variant.final_marking,
                            trace,
                            timeout_seconds=exact_timeout,
                        )
                    except Exception as exc:
                        rejections[f"exact_exception:{type(exc).__name__}"] += 1
                        family_failed = True
                        break
                    exact_seconds = perf_counter() - exact_start
                    alignment_cache[cache_key] = (result, exact_seconds)
                    cache_hit = False
                else:
                    result, exact_seconds = cached
                    cache_hit = True
                if result.alignment is None or result.cost is None or not result.optimal:
                    rejections["exact_alignment_failed_or_timed_out"] += 1
                    family_failed = True
                    break
                verifier = verify_alignment(
                    result.alignment,
                    variant.net,
                    variant.initial_marking,
                    variant.final_marking,
                    trace,
                )
                if not verifier.legal:
                    rejections["exact_alignment_failed_verification"] += 1
                    family_failed = True
                    break
                costs.append(verifier.cost)
                family_rows.append(
                    (observed, variant, result, verifier, exact_seconds, cache_hit)
                )
            if family_failed:
                break
            if family.equivalence_certificate.status == "exact" and len(set(costs)) != 1:
                rejections["equivalent_variant_cost_mismatch"] += 1
                family_failed = True
                break
        if family_failed:
            continue

        accepted_families += 1
        for observed, variant, result, verifier, exact_seconds, cache_hit in family_rows:
            if len(samples) >= target_size:
                break
            identifiable = transition_identity_identifiable(variant, observed.labels)
            sample_index = len(samples)
            samples.append(
                AlignmentSample(
                    sample_id=f"{split}-{sample_index:06d}",
                    split=split,
                    net=variant.net,
                    initial_marking=variant.initial_marking,
                    final_marking=variant.final_marking,
                    trace=trace_from_labels(observed.labels),
                    optimal_alignment=result.alignment,
                    optimal_cost=verifier.cost,
                    metadata={
                        **family.metadata,
                        "family": str(family.metadata["motif"]),
                        "behavior_id": family.behavior_id,
                        "variant_id": variant.variant_id,
                        "representation_kind": variant.representation_kind,
                        "equivalence_level": variant.equivalence_level,
                        "equivalence_certificate": family.equivalence_certificate.to_dict(),
                        "canonical_spec": family.canonical_spec.to_dict(),
                        "transformation_sequence": list(variant.transformation_sequence),
                        "structural_statistics": variant.structural_statistics,
                        "trace_id": observed.trace_id,
                        "clean_trace": list(observed.clean_trace),
                        "observed_trace": list(observed.labels),
                        "trace_edits": [edit.to_dict() for edit in observed.edits],
                        "edit_count": len(observed.edits),
                        "transition_identity_identifiable": identifiable,
                        "transition_identity_loss_masked": not identifiable,
                        "move_signature": _move_signature(
                            result.alignment, variant.representation_kind, observed.edits
                        ),
                        "optimal_cost_bucket": min(int(verifier.cost), 4),
                        "model_size_bucket": _size_bucket(
                            int(variant.structural_statistics["num_transitions"])
                        ),
                        "duplicate_label_count_bucket": min(
                            int(variant.structural_statistics["duplicate_label_count"]), 4
                        ),
                        "invisible_transition_count_bucket": min(
                            int(variant.structural_statistics["num_invisible"]), 4
                        ),
                        "non_free_choice": bool(
                            variant.structural_statistics["free_choice_violation_count"]
                        ),
                        "exact_diagnostics": {
                            **(result.diagnostics or {}),
                            "elapsed_seconds": exact_seconds,
                            "cache_hit": cache_hit,
                        },
                    },
                )
            )
            if progress_every > 0 and len(samples) % progress_every == 0:
                print(f"generated {len(samples):5d}/{target_size} {split} exact-labeled samples")

    if len(samples) != target_size:
        raise SystemExit(
            f"generated {len(samples)} of {target_size} requested {split} samples "
            f"after {family_index} family attempts; rejections={dict(rejections)}"
        )
    return samples, rejections, accepted_families


def _config_from_args(args: argparse.Namespace) -> BehaviorFamilyConfig:
    if args.generator_config is not None:
        config = BehaviorFamilyConfig.load(args.generator_config)
    elif args.preset is not None:
        config = BehaviorFamilyConfig.preset(args.preset)
    else:
        config = BehaviorFamilyConfig()
    if args.seed is not None:
        config = replace(config, seed=args.seed)

    structure = config.structure
    structure = replace(
        structure,
        min_visible_occurrences=(
            args.min_visible_occurrences
            if args.min_visible_occurrences is not None
            else args.min_len
            if args.min_len is not None
            else structure.min_visible_occurrences
        ),
        max_visible_occurrences=(
            args.max_visible_occurrences
            if args.max_visible_occurrences is not None
            else args.max_len
            if args.max_len is not None
            else structure.max_visible_occurrences
        ),
        max_depth=args.max_depth if args.max_depth is not None else structure.max_depth,
    )
    representations = replace(
        config.representations,
        variants_per_behavior=(
            max(1, args.variants_per_behavior)
            if args.variants_per_behavior is not None
            else config.representations.variants_per_behavior
        ),
    )
    logs = replace(
        config.logs,
        traces_per_behavior=(
            max(1, args.traces_per_behavior)
            if args.traces_per_behavior is not None
            else config.logs.traces_per_behavior
        ),
    )
    clean_fraction = config.noise.clean_fraction
    if args.clean_fraction is not None:
        clean_fraction = args.clean_fraction
    elif args.deviation_rate is not None:
        clean_fraction = max(0.0, min(1.0, 1.0 - args.deviation_rate))
    noise = replace(
        config.noise,
        clean_fraction=clean_fraction,
        edit_count_weights=(
            {int(key): value for key, value in _parse_weights(args.edit_count_weights).items()}
            if args.edit_count_weights is not None
            else config.noise.edit_count_weights
        ),
    )
    motif_weights = (
        _parse_weights(args.motif_weights)
        if args.motif_weights is not None
        else config.motif_weights
    )
    if args.duplicate_fraction is not None:
        duplicate = max(0.0, min(1.0, args.duplicate_fraction))
        remaining = 1.0 - duplicate
        motif_weights = {
            "duplicate_vs_silent": duplicate,
            "ordinary_tree": remaining / 3,
            "concurrent_vs_interleaved": remaining / 3,
            "m_nonfreechoice": remaining / 3,
        }
    return replace(
        config,
        structure=structure,
        representations=representations,
        logs=logs,
        noise=noise,
        motif_weights=motif_weights,
    )


def _prepare_output_dir(output: Path, overwrite: bool) -> None:
    existing = [output / f"{split}.pkl" for split in SPLITS]
    if any(path.exists() for path in existing) and not overwrite:
        raise SystemExit(
            f"{output} already contains split files; pass --overwrite to replace them"
        )
    output.mkdir(parents=True, exist_ok=True)


def _parse_weights(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        try:
            name, weight = item.split("=", 1)
            result[name.strip()] = float(weight)
        except ValueError as exc:
            raise ValueError("weights must use NAME=WEIGHT comma-separated syntax") from exc
    if not result or sum(max(0.0, value) for value in result.values()) <= 0:
        raise ValueError("at least one weight must be positive")
    return result


def _positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _move_signature(alignment, representation_kind: str, edits) -> str:
    from lara_align.types import MoveKind

    if representation_kind == "duplicate_prefix":
        return "duplicate_label_resolution"
    if representation_kind in {"silent_routing", "tau_refinement"} and any(
        move.kind == MoveKind.MODEL and move.transition_label is None
        for move in alignment.moves
    ):
        return "silent_model_routing"
    has_log = any(move.kind == MoveKind.LOG for move in alignment.moves)
    has_visible_model = any(
        move.kind == MoveKind.MODEL and move.transition_label is not None
        for move in alignment.moves
    )
    has_silent = any(
        move.kind == MoveKind.MODEL and move.transition_label is None
        for move in alignment.moves
    )
    if not edits and not has_log and not has_visible_model:
        return "clean" if not has_silent else "silent_model_routing"
    if has_log and (has_visible_model or has_silent):
        return "mixed_deviation"
    if has_log:
        return "log_only_deviation"
    if has_visible_model:
        return "visible_model_move"
    if has_silent:
        return "silent_model_routing"
    return "clean"


def _size_bucket(num_transitions: int) -> str:
    if num_transitions <= 5:
        return "small"
    if num_transitions <= 15:
        return "medium"
    return "large"


def _net_signature(net, initial_marking, final_marking) -> tuple[object, ...]:
    places = tuple(sorted(str(place.name) for place in net.places))
    transitions = tuple(
        sorted(
            (str(transition.name), None if transition.label is None else str(transition.label))
            for transition in net.transitions
        )
    )
    arcs = tuple(
        sorted(
            (str(arc.source.name), str(arc.target.name), int(arc.weight))
            for arc in net.arcs
        )
    )
    initial = tuple(sorted((str(place.name), int(tokens)) for place, tokens in initial_marking.items()))
    final = tuple(sorted((str(place.name), int(tokens)) for place, tokens in final_marking.items()))
    return places, transitions, arcs, initial, final


if __name__ == "__main__":
    main()
