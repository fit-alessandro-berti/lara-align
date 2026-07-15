from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from hashlib import blake2b
import json
from pathlib import Path
from random import Random
from typing import Any, Sequence

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from lara_align.synthetic import (
    _random_block_tree,
    make_block_structured_net,
)


EQUIVALENCE_SEMANTICS = "visible_complete_trace_language"
MOTIF_KINDS = (
    "ordinary_tree",
    "duplicate_vs_silent",
    "concurrent_vs_interleaved",
    "m_nonfreechoice",
)


@dataclass(frozen=True)
class BehaviorSpec:
    op: str
    label: str | None = None
    children: tuple["BehaviorSpec", ...] = ()

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"op": self.op}
        if self.label is not None:
            value["label"] = self.label
        if self.children:
            value["children"] = [child.to_dict() for child in self.children]
        return value


@dataclass(frozen=True)
class StructureConfig:
    min_visible_occurrences: int = 4
    max_visible_occurrences: int = 12
    max_depth: int = 6
    operator_weights: dict[str, float] = field(
        default_factory=lambda: {"seq": 0.4, "xor": 0.3, "and": 0.3}
    )
    alphabet_ratio: float = 0.7
    motif_context_size: int = 2


@dataclass(frozen=True)
class RepresentationConfig:
    variants_per_behavior: int = 2
    exact_equivalence_only_for_training: bool = True


@dataclass(frozen=True)
class LogConfig:
    traces_per_behavior: int = 2
    clean_pool_size: int = 16
    max_trace_length: int = 128


@dataclass(frozen=True)
class NoiseConfig:
    clean_fraction: float = 0.2
    edit_count_weights: dict[int, float] = field(
        default_factory=lambda: {1: 0.5, 2: 0.3, 3: 0.2}
    )
    operation_weights: dict[str, float] = field(
        default_factory=lambda: {
            "delete": 0.15,
            "insert": 0.15,
            "substitute": 0.15,
            "swap": 0.15,
            "repeat": 0.15,
            "prefix_truncate": 0.1,
            "suffix_truncate": 0.1,
            "outside_insert": 0.05,
        }
    )


@dataclass(frozen=True)
class ValidationConfig:
    exact_language_max_states: int = 5000
    exact_language_max_traces: int = 10000
    bounded_visible_length: int = 20
    reject_on_mismatch: bool = True


@dataclass(frozen=True)
class BehaviorFamilyConfig:
    seed: int = 13
    structure: StructureConfig = field(default_factory=StructureConfig)
    motif_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ordinary_tree": 0.25,
            "duplicate_vs_silent": 0.25,
            "concurrent_vs_interleaved": 0.25,
            "m_nonfreechoice": 0.25,
        }
    )
    representations: RepresentationConfig = field(default_factory=RepresentationConfig)
    logs: LogConfig = field(default_factory=LogConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["motifs"] = value.pop("motif_weights")
        value["noise"]["edit_count_weights"] = {
            str(key): weight for key, weight in self.noise.edit_count_weights.items()
        }
        return value

    @staticmethod
    def from_dict(data: dict[str, object]) -> "BehaviorFamilyConfig":
        structure = _mapping(data.get("structure"))
        representations = _mapping(data.get("representations"))
        logs = _mapping(data.get("logs"))
        noise = _mapping(data.get("noise"))
        validation = _mapping(data.get("validation"))
        edits = _mapping(noise.get("edit_count_weights"))
        return BehaviorFamilyConfig(
            seed=int(data.get("seed", 13)),
            structure=StructureConfig(
                min_visible_occurrences=int(structure.get("min_visible_occurrences", 4)),
                max_visible_occurrences=int(structure.get("max_visible_occurrences", 12)),
                max_depth=int(structure.get("max_depth", 6)),
                operator_weights={
                    str(key): float(value)
                    for key, value in _mapping(structure.get("operator_weights")).items()
                }
                or {"seq": 0.4, "xor": 0.3, "and": 0.3},
                alphabet_ratio=float(structure.get("alphabet_ratio", 0.7)),
                motif_context_size=int(structure.get("motif_context_size", 2)),
            ),
            motif_weights={
                str(key): float(value)
                for key, value in _mapping(data.get("motifs", data.get("motif_weights"))).items()
            }
            or {
                "ordinary_tree": 0.25,
                "duplicate_vs_silent": 0.25,
                "concurrent_vs_interleaved": 0.25,
                "m_nonfreechoice": 0.25,
            },
            representations=RepresentationConfig(
                variants_per_behavior=int(representations.get("variants_per_behavior", 2)),
                exact_equivalence_only_for_training=bool(
                    representations.get("exact_equivalence_only_for_training", True)
                ),
            ),
            logs=LogConfig(
                traces_per_behavior=int(logs.get("traces_per_behavior", 2)),
                clean_pool_size=int(logs.get("clean_pool_size", 16)),
                max_trace_length=int(logs.get("max_trace_length", 128)),
            ),
            noise=NoiseConfig(
                clean_fraction=float(noise.get("clean_fraction", 0.2)),
                edit_count_weights={int(key): float(value) for key, value in edits.items()}
                or {1: 0.5, 2: 0.3, 3: 0.2},
                operation_weights={
                    str(key): float(value)
                    for key, value in _mapping(noise.get("operation_weights")).items()
                }
                or NoiseConfig().operation_weights,
            ),
            validation=ValidationConfig(
                exact_language_max_states=int(
                    validation.get("exact_language_max_states", 5000)
                ),
                exact_language_max_traces=int(
                    validation.get("exact_language_max_traces", 10000)
                ),
                bounded_visible_length=int(validation.get("bounded_visible_length", 20)),
                reject_on_mismatch=bool(validation.get("reject_on_mismatch", True)),
            ),
        )

    @staticmethod
    def load(path: str | Path) -> "BehaviorFamilyConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return BehaviorFamilyConfig.from_dict(json.load(handle))

    @staticmethod
    def preset(name: str) -> "BehaviorFamilyConfig":
        presets: dict[str, dict[str, object]] = {
            "smoke": {
                "structure": {"max_visible_occurrences": 6},
                "logs": {"traces_per_behavior": 1, "clean_pool_size": 4},
            },
            "balanced_train": {},
            "iid_behavior": {},
            "equivalence_train": {
                "motifs": {
                    "ordinary_tree": 0.1,
                    "duplicate_vs_silent": 0.3,
                    "concurrent_vs_interleaved": 0.3,
                    "m_nonfreechoice": 0.3,
                }
            },
            "equivalence_test": {
                "motifs": {
                    "duplicate_vs_silent": 1.0,
                    "concurrent_vs_interleaved": 1.0,
                    "m_nonfreechoice": 1.0,
                }
            },
            "equivalence_seen": {
                "motifs": {
                    "duplicate_vs_silent": 1.0,
                    "concurrent_vs_interleaved": 1.0,
                    "m_nonfreechoice": 1.0,
                }
            },
            "equivalence_unseen": {
                "motifs": {
                    "duplicate_vs_silent": 1.0,
                    "concurrent_vs_interleaved": 1.0,
                    "m_nonfreechoice": 1.0,
                },
                "representations": {"variants_per_behavior": 4},
            },
            "nonblock_ood": {"motifs": {"m_nonfreechoice": 1.0}},
            "scale_ood": {
                "structure": {
                    "min_visible_occurrences": 12,
                    "max_visible_occurrences": 30,
                },
                "logs": {"max_trace_length": 256},
            },
            "noise_ood": {
                "noise": {
                    "clean_fraction": 0.0,
                    "edit_count_weights": {"3": 0.5, "4": 0.5},
                }
            },
            "sampling_ood": {
                "logs": {"traces_per_behavior": 1, "clean_pool_size": 4}
            },
            "loops_bounded": {
                "structure": {
                    "operator_weights": {
                        "seq": 0.3,
                        "xor": 0.2,
                        "and": 0.2,
                        "loop": 0.3
                    }
                },
                "motifs": {"ordinary_tree": 1.0},
                "representations": {
                    "variants_per_behavior": 2,
                    "exact_equivalence_only_for_training": False,
                },
            },
        }
        if name not in presets:
            raise ValueError(f"unknown behavior-family preset: {name}")
        return BehaviorFamilyConfig.from_dict(presets[name])


@dataclass(frozen=True)
class TraceEdit:
    kind: str
    position: int
    old_label: str | None
    new_label: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObservedTrace:
    trace_id: str
    clean_trace: tuple[str, ...]
    labels: tuple[str, ...]
    edits: tuple[TraceEdit, ...]


@dataclass
class ModelVariant:
    variant_id: str
    representation_kind: str
    net: PetriNet
    initial_marking: Marking
    final_marking: Marking
    transformation_sequence: tuple[str, ...]
    structural_statistics: dict[str, object]
    equivalence_level: str = "exact"


@dataclass(frozen=True)
class EquivalenceCertificate:
    status: str
    semantics: str
    reference_language_size: int
    checked_variants: tuple[str, ...]
    max_visible_length: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BehaviorFamily:
    behavior_id: str
    canonical_spec: BehaviorSpec
    model_variants: tuple[ModelVariant, ...]
    clean_trace_pool: tuple[tuple[str, ...], ...]
    noisy_traces: tuple[ObservedTrace, ...]
    equivalence_certificate: EquivalenceCertificate
    metadata: dict[str, object]


def stable_seed(global_seed: int, *parts: object) -> int:
    payload = "|".join([str(global_seed), *(str(part) for part in parts)])
    digest = blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def generate_behavior_family(
    config: BehaviorFamilyConfig,
    family_index: int,
    split: str,
) -> BehaviorFamily:
    seed = stable_seed(config.seed, split, family_index, "family")
    seed_bundle = {
        "model": stable_seed(seed, "model"),
        "representation": stable_seed(seed, "representation"),
        "playout": stable_seed(seed, "playout"),
        "log_view": stable_seed(seed, "log-view"),
        "noise": stable_seed(seed, "noise"),
    }
    rng = Random(seed_bundle["model"])
    motif = _weighted_choice(rng, config.motif_weights)
    behavior_id = f"{split}-{stable_seed(config.seed, split, family_index):016x}"

    if motif == "duplicate_vs_silent":
        spec, runtime = _duplicate_vs_silent(behavior_id)
    elif motif == "concurrent_vs_interleaved":
        spec, runtime = _concurrent_vs_interleaved(behavior_id)
    elif motif == "m_nonfreechoice":
        spec, runtime = _m_nonfreechoice(behavior_id)
    else:
        spec, runtime = _ordinary_tree(config, rng, behavior_id)

    context_labels: tuple[str, ...] = ()
    if motif != "ordinary_tree" and config.structure.motif_context_size > 0:
        used_labels = _spec_labels(spec)
        candidates = (_activity_name(index) for index in range(1000))
        context_labels = tuple(
            label
            for label in candidates
            if label not in used_labels
        )[: config.structure.motif_context_size]
        if context_labels:
            spec = BehaviorSpec(
                "seq",
                children=(
                    spec,
                    *(BehaviorSpec("activity", label) for label in context_labels),
                ),
            )
            runtime = tuple(
                _append_visible_suffix(values, context_labels) for values in runtime
            )

    runtime = list(runtime)
    requested_variants = max(1, config.representations.variants_per_behavior)
    if requested_variants >= 3:
        _, reference_net, reference_initial, reference_final, _, _ = runtime[0]
        finite_language, finite_complete = enumerate_visible_language(
            reference_net,
            reference_initial,
            reference_final,
            max_states=config.validation.exact_language_max_states,
            max_traces=config.validation.exact_language_max_traces,
            max_visible_length=config.validation.bounded_visible_length,
        )
        if finite_complete:
            trie = _make_prefix_trie_net(
                finite_language, f"{behavior_id}-prefix-trie"
            )
            runtime.append(
                (
                    "prefix_trie",
                    *trie,
                    ("exact_language_enumeration", "prefix_trie_compilation"),
                    None,
                )
            )
    if requested_variants >= 4:
        kind, net, initial, final, transformations, nonblock_reason = runtime[0]
        refined = _tau_prefix_refinement(
            net, initial, final, f"{behavior_id}-tau-refined"
        )
        runtime.append(
            (
                "tau_refinement",
                *refined,
                (*transformations, "invisible_prefix_refinement"),
                nonblock_reason,
            )
        )
    base_variants = tuple(runtime)
    while len(runtime) < requested_variants:
        kind, net, initial, final, transformations, nonblock_reason = base_variants[
            (len(runtime) - len(base_variants)) % len(base_variants)
        ]
        clone = _clone_isomorphic(
            net,
            initial,
            final,
            f"{behavior_id}-{kind}-iso-{len(runtime)}",
        )
        runtime.append(
            (
                f"{kind}_isomorphic_{len(runtime)}",
                *clone,
                (*transformations, "isomorphic_renaming"),
                nonblock_reason,
            )
        )

    languages: list[set[tuple[str, ...]]] = []
    exact = True
    for _, net, initial, final, _, _ in runtime:
        language, completed = enumerate_visible_language(
            net,
            initial,
            final,
            max_states=config.validation.exact_language_max_states,
            max_traces=config.validation.exact_language_max_traces,
            max_visible_length=config.validation.bounded_visible_length,
        )
        languages.append(language)
        exact = exact and completed
    reference = languages[0]
    if not reference:
        raise ValueError(f"family {behavior_id} has no complete visible traces")
    if any(language != reference for language in languages[1:]):
        if config.validation.reject_on_mismatch:
            raise ValueError(f"visible-language mismatch in family {behavior_id}")
        exact = False
    level = "exact" if exact else "bounded"
    if config.representations.exact_equivalence_only_for_training and split == "train" and not exact:
        raise ValueError(f"training family {behavior_id} is not exactly enumerable")

    variants = tuple(
        ModelVariant(
            variant_id=f"{behavior_id}:{kind}",
            representation_kind=kind,
            net=net,
            initial_marking=initial,
            final_marking=final,
            transformation_sequence=transformations,
            structural_statistics=structural_statistics(
                net,
                initial,
                final,
                nonblock_reason=nonblock_reason,
            ),
            equivalence_level=level,
        )
        for kind, net, initial, final, transformations, nonblock_reason in runtime[
            :requested_variants
        ]
    )
    if any(
        not variant.structural_statistics["final_marking_reachable"]
        or int(variant.structural_statistics["dead_transition_count"]) > 0
        for variant in variants
    ):
        raise ValueError(f"family {behavior_id} contains an unsound structural variant")
    complete_pool = tuple(sorted(reference))
    if len(complete_pool) > config.logs.clean_pool_size:
        selected = sorted(
            rng.sample(range(len(complete_pool)), config.logs.clean_pool_size)
        )
        trace_pool = tuple(complete_pool[index] for index in selected)
    else:
        trace_pool = complete_pool
    observed = _observed_traces(
        config, trace_pool, behavior_id, seed_bundle["noise"]
    )
    certificate = EquivalenceCertificate(
        status=level,
        semantics=EQUIVALENCE_SEMANTICS,
        reference_language_size=len(reference),
        checked_variants=tuple(variant.representation_kind for variant in variants),
        max_visible_length=config.validation.bounded_visible_length,
    )
    return BehaviorFamily(
        behavior_id=behavior_id,
        canonical_spec=spec,
        model_variants=variants,
        clean_trace_pool=trace_pool,
        noisy_traces=observed,
        equivalence_certificate=certificate,
        metadata={
            "motif": motif,
            "family_seed": seed,
            "seed_bundle": seed_bundle,
            "equivalence_semantics": EQUIVALENCE_SEMANTICS,
            "trace_language_equivalent": True,
            "partial_order_equivalent": (
                False if motif == "concurrent_vs_interleaved" else None
            ),
            "context_suffix": list(context_labels),
        },
    )


class TraceCorruptor:
    def __init__(self, config: NoiseConfig) -> None:
        self.config = config

    def corrupt(
        self,
        labels: Sequence[str],
        alphabet: Sequence[str],
        rng: Random,
        edit_count: int,
    ) -> tuple[tuple[str, ...], tuple[TraceEdit, ...]]:
        result = list(labels)
        edits: list[TraceEdit] = []
        operations = list(self.config.operation_weights)
        weights = [max(0.0, self.config.operation_weights[name]) for name in operations]
        for _ in range(edit_count):
            kind = rng.choices(operations, weights=weights, k=1)[0]
            if kind == "delete" and result:
                position = rng.randrange(len(result))
                old = result.pop(position)
                edits.append(TraceEdit(kind, position, old, None))
            elif kind == "insert":
                position = rng.randrange(len(result) + 1)
                new = rng.choice(list(alphabet))
                result.insert(position, new)
                edits.append(TraceEdit(kind, position, None, new))
            elif kind == "substitute" and result:
                position = rng.randrange(len(result))
                old = result[position]
                choices = [label for label in alphabet if label != old] or ["__OOD__"]
                new = rng.choice(choices)
                result[position] = new
                edits.append(TraceEdit(kind, position, old, new))
            elif kind == "swap" and len(result) >= 2:
                position = rng.randrange(len(result) - 1)
                old = result[position]
                new = result[position + 1]
                result[position], result[position + 1] = new, old
                edits.append(TraceEdit(kind, position, old, new))
            elif kind == "repeat" and result:
                position = rng.randrange(len(result))
                new = result[position]
                result.insert(position, new)
                edits.append(TraceEdit(kind, position, None, new))
            elif kind == "prefix_truncate" and result:
                amount = rng.randint(1, len(result))
                old = result[amount - 1]
                del result[:amount]
                edits.append(TraceEdit(kind, 0, old, None))
            elif kind == "suffix_truncate" and result:
                position = rng.randrange(len(result))
                old = result[position]
                del result[position:]
                edits.append(TraceEdit(kind, position, old, None))
            else:
                position = rng.randrange(len(result) + 1)
                result.insert(position, "__OOD__")
                edits.append(TraceEdit("outside_insert", position, None, "__OOD__"))
        return tuple(result), tuple(edits)


def enumerate_visible_language(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    *,
    max_states: int,
    max_traces: int,
    max_visible_length: int,
) -> tuple[set[tuple[str, ...]], bool]:
    places = sorted(net.places, key=lambda place: str(place.name))
    transitions = sorted(net.transitions, key=lambda transition: str(transition.name))
    initial = Counter({place: int(tokens) for place, tokens in initial_marking.items()})
    final = Counter({place: int(tokens) for place, tokens in final_marking.items()})
    stack: list[tuple[Counter, tuple[str, ...], int]] = [(initial, (), 0)]
    seen: set[tuple[tuple[int, ...], tuple[str, ...], int]] = set()
    language: set[tuple[str, ...]] = set()
    completed = True
    while stack:
        marking, trace, firings = stack.pop()
        key = (tuple(marking.get(place, 0) for place in places), trace, firings)
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > max_states or len(language) > max_traces:
            completed = False
            break
        if +marking == +final:
            language.add(trace)
            continue
        for transition in reversed(transitions):
            if not _enabled(transition, marking):
                continue
            next_trace = trace
            if transition.label is not None:
                if len(trace) >= max_visible_length:
                    completed = False
                    continue
                next_trace = (*trace, str(transition.label))
            if firings >= max_visible_length * 4 + len(transitions):
                completed = False
                continue
            stack.append((_fire(transition, marking), next_trace, firings + 1))
    return language, completed


def structural_statistics(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    *,
    nonblock_reason: str | None = None,
) -> dict[str, object]:
    labels = [str(transition.label) for transition in net.transitions if transition.label is not None]
    counts = Counter(labels)
    return {
        "num_places": len(net.places),
        "num_transitions": len(net.transitions),
        "num_arcs": len(net.arcs),
        "num_invisible": sum(1 for transition in net.transitions if transition.label is None),
        "duplicate_label_count": sum(count for count in counts.values() if count > 1),
        "free_choice_violation_count": _free_choice_violations(net),
        "initial_token_count": sum(int(value) for value in initial_marking.values()),
        "final_token_count": sum(int(value) for value in final_marking.values()),
        **_reachability_statistics(net, initial_marking, final_marking),
        "nonblock_reason": nonblock_reason,
    }


def transition_identity_identifiable(variant: ModelVariant, labels: Sequence[str]) -> bool:
    if variant.representation_kind != "duplicate_prefix":
        return True
    # The future B/C observation identifies which concrete A transition was intended.
    return ("B" in labels) ^ ("C" in labels)


def _duplicate_vs_silent(behavior_id: str):
    spec = BehaviorSpec(
        "xor",
        children=(
            BehaviorSpec("seq", children=(BehaviorSpec("activity", "A"), BehaviorSpec("activity", "B"))),
            BehaviorSpec("seq", children=(BehaviorSpec("activity", "A"), BehaviorSpec("activity", "C"))),
        ),
    )
    duplicate = _make_duplicate_prefix_net(f"{behavior_id}-duplicate")
    silent = _make_silent_routing_net(f"{behavior_id}-silent")
    return spec, (
        ("duplicate_prefix", *duplicate, ("duplicate_visible_prefix",), None),
        ("silent_routing", *silent, ("shared_activity_tau_routing",), None),
    )


def _concurrent_vs_interleaved(behavior_id: str):
    from lara_align.synthetic import _BlockNode

    spec = BehaviorSpec(
        "and", children=(BehaviorSpec("activity", "A"), BehaviorSpec("activity", "B"))
    )
    block = _BlockNode("and", children=[_BlockNode("leaf", "A"), _BlockNode("leaf", "B")])
    parallel = make_block_structured_net(block, f"{behavior_id}-parallel")
    interleaved = _make_interleaving_net(f"{behavior_id}-interleaved")
    return spec, (
        ("parallel", *parallel, ("true_token_concurrency",), None),
        ("explicit_interleaving", *interleaved, ("progress_state_interleaving",), None),
    )


def _m_nonfreechoice(behavior_id: str):
    from lara_align.synthetic import _BlockNode

    spec = BehaviorSpec(
        "xor",
        children=(
            BehaviorSpec("activity", "B"),
            BehaviorSpec(
                "and", children=(BehaviorSpec("activity", "A"), BehaviorSpec("activity", "C"))
            ),
        ),
    )
    block = _BlockNode(
        "xor",
        children=[
            _BlockNode("leaf", "B"),
            _BlockNode("and", children=[_BlockNode("leaf", "A"), _BlockNode("leaf", "C")]),
        ],
    )
    canonical = make_block_structured_net(block, f"{behavior_id}-block")
    nonfree = _make_m_nonfreechoice_net(f"{behavior_id}-m")
    return spec, (
        ("canonical_block", *canonical, ("block_compiler",), None),
        (
            "m_nonfreechoice",
            *nonfree,
            ("non_free_choice_m_pattern",),
            "non_free_choice_m_pattern",
        ),
    )


def _ordinary_tree(config: BehaviorFamilyConfig, rng: Random, behavior_id: str):
    leaves = rng.randint(
        config.structure.min_visible_occurrences,
        config.structure.max_visible_occurrences,
    )
    alphabet_size = max(2, int(round(leaves * config.structure.alphabet_ratio)))
    alphabet = [_activity_name(index) for index in range(alphabet_size)]
    tree = _random_block_tree(
        rng,
        leaves,
        alphabet,
        config.structure.operator_weights,
        max_depth=config.structure.max_depth,
    )
    spec = _block_to_spec(tree)
    canonical = make_block_structured_net(tree, f"{behavior_id}-block")
    renamed = _clone_isomorphic(*canonical, f"{behavior_id}-renamed")
    return spec, (
        ("canonical_block", *canonical, ("block_compiler",), None),
        ("isomorphic_renaming", *renamed, ("block_compiler", "isomorphic_renaming"), None),
    )


def _observed_traces(
    config: BehaviorFamilyConfig,
    trace_pool: tuple[tuple[str, ...], ...],
    behavior_id: str,
    seed: int,
) -> tuple[ObservedTrace, ...]:
    alphabet = tuple(sorted({label for trace in trace_pool for label in trace}))
    corruptor = TraceCorruptor(config.noise)
    observed: list[ObservedTrace] = []
    for index in range(max(1, config.logs.traces_per_behavior)):
        rng = Random(stable_seed(seed, "trace", index))
        clean = trace_pool[index % len(trace_pool)]
        if rng.random() < config.noise.clean_fraction:
            labels, edits = clean, ()
        else:
            counts = list(config.noise.edit_count_weights)
            weights = [max(0.0, config.noise.edit_count_weights[count]) for count in counts]
            edit_count = rng.choices(counts, weights=weights, k=1)[0]
            labels, edits = corruptor.corrupt(clean, alphabet, rng, edit_count)
        observed.append(
            ObservedTrace(
                trace_id=f"{behavior_id}:trace-{index:02d}",
                clean_trace=clean,
                labels=labels[: config.logs.max_trace_length],
                edits=edits,
            )
        )
    return tuple(observed)


def _make_duplicate_prefix_net(name: str):
    net = PetriNet(name)
    p0, pl, pr, pf = (PetriNet.Place(value) for value in ("p0", "pl", "pr", "pf"))
    net.places.update({p0, pl, pr, pf})
    transitions = (
        PetriNet.Transition("t_A_left", "A"),
        PetriNet.Transition("t_A_right", "A"),
        PetriNet.Transition("t_B", "B"),
        PetriNet.Transition("t_C", "C"),
    )
    net.transitions.update(transitions)
    for source, transition, target in (
        (p0, transitions[0], pl),
        (p0, transitions[1], pr),
        (pl, transitions[2], pf),
        (pr, transitions[3], pf),
    ):
        petri_utils.add_arc_from_to(source, transition, net)
        petri_utils.add_arc_from_to(transition, target, net)
    return net, Marking({p0: 1}), Marking({pf: 1})


def _make_silent_routing_net(name: str):
    net = PetriNet(name)
    p0, pc, pl, pr, pf = (
        PetriNet.Place(value) for value in ("p0", "pc", "pl", "pr", "pf")
    )
    net.places.update({p0, pc, pl, pr, pf})
    transitions = (
        PetriNet.Transition("t_A", "A"),
        PetriNet.Transition("tau_left", None),
        PetriNet.Transition("tau_right", None),
        PetriNet.Transition("t_B", "B"),
        PetriNet.Transition("t_C", "C"),
    )
    net.transitions.update(transitions)
    for source, transition, target in (
        (p0, transitions[0], pc),
        (pc, transitions[1], pl),
        (pc, transitions[2], pr),
        (pl, transitions[3], pf),
        (pr, transitions[4], pf),
    ):
        petri_utils.add_arc_from_to(source, transition, net)
        petri_utils.add_arc_from_to(transition, target, net)
    return net, Marking({p0: 1}), Marking({pf: 1})


def _make_interleaving_net(name: str):
    net = PetriNet(name)
    p0, pa, pb, pf = (PetriNet.Place(value) for value in ("p0", "pa", "pb", "pf"))
    net.places.update({p0, pa, pb, pf})
    transitions = (
        PetriNet.Transition("t_A_first", "A"),
        PetriNet.Transition("t_B_after_A", "B"),
        PetriNet.Transition("t_B_first", "B"),
        PetriNet.Transition("t_A_after_B", "A"),
    )
    net.transitions.update(transitions)
    for source, transition, target in (
        (p0, transitions[0], pa),
        (pa, transitions[1], pf),
        (p0, transitions[2], pb),
        (pb, transitions[3], pf),
    ):
        petri_utils.add_arc_from_to(source, transition, net)
        petri_utils.add_arc_from_to(transition, target, net)
    return net, Marking({p0: 1}), Marking({pf: 1})


def _make_m_nonfreechoice_net(name: str):
    net = PetriNet(name)
    p0, p1, p2, q1, q2, pf = (
        PetriNet.Place(value) for value in ("p0", "p1", "p2", "q1", "q2", "pf")
    )
    net.places.update({p0, p1, p2, q1, q2, pf})
    split = PetriNet.Transition("tau_split", None)
    a = PetriNet.Transition("t_A", "A")
    b = PetriNet.Transition("t_B", "B")
    c = PetriNet.Transition("t_C", "C")
    join = PetriNet.Transition("tau_join", None)
    net.transitions.update({split, a, b, c, join})
    petri_utils.add_arc_from_to(p0, split, net)
    petri_utils.add_arc_from_to(split, p1, net)
    petri_utils.add_arc_from_to(split, p2, net)
    petri_utils.add_arc_from_to(p1, a, net)
    petri_utils.add_arc_from_to(a, q1, net)
    petri_utils.add_arc_from_to(p2, c, net)
    petri_utils.add_arc_from_to(c, q2, net)
    petri_utils.add_arc_from_to(p1, b, net)
    petri_utils.add_arc_from_to(p2, b, net)
    petri_utils.add_arc_from_to(b, pf, net)
    petri_utils.add_arc_from_to(q1, join, net)
    petri_utils.add_arc_from_to(q2, join, net)
    petri_utils.add_arc_from_to(join, pf, net)
    return net, Marking({p0: 1}), Marking({pf: 1})


def _make_prefix_trie_net(language: set[tuple[str, ...]], name: str):
    net = PetriNet(name)
    source = PetriNet.Place("prefix_root")
    sink = PetriNet.Place("prefix_sink")
    net.places.update({source, sink})
    places: dict[tuple[str, ...], PetriNet.Place] = {(): source}
    edges: dict[tuple[tuple[str, ...], str], PetriNet.Transition] = {}
    terminals: set[tuple[str, ...]] = set()
    for trace in sorted(language):
        prefix: tuple[str, ...] = ()
        for label in trace:
            next_prefix = (*prefix, label)
            if next_prefix not in places:
                place = PetriNet.Place(f"prefix_{len(places):05d}")
                places[next_prefix] = place
                net.places.add(place)
            edge_key = (prefix, label)
            if edge_key not in edges:
                transition = PetriNet.Transition(f"edge_{len(edges):05d}_{label}", label)
                edges[edge_key] = transition
                net.transitions.add(transition)
                petri_utils.add_arc_from_to(places[prefix], transition, net)
                petri_utils.add_arc_from_to(transition, places[next_prefix], net)
            prefix = next_prefix
        terminals.add(prefix)
    for index, prefix in enumerate(sorted(terminals)):
        transition = PetriNet.Transition(f"tau_accept_{index:05d}", None)
        net.transitions.add(transition)
        petri_utils.add_arc_from_to(places[prefix], transition, net)
        petri_utils.add_arc_from_to(transition, sink, net)
    return net, Marking({source: 1}), Marking({sink: 1})


def _clone_isomorphic(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    name: str,
):
    clone = PetriNet(name)
    places = sorted(net.places, key=lambda place: str(place.name))
    transitions = sorted(net.transitions, key=lambda transition: str(transition.name))
    place_map = {
        place: PetriNet.Place(f"renamed_p_{len(places) - index:04d}")
        for index, place in enumerate(places)
    }
    transition_map = {
        transition: PetriNet.Transition(
            f"renamed_t_{len(transitions) - index:04d}", transition.label
        )
        for index, transition in enumerate(transitions)
    }
    clone.places.update(place_map.values())
    clone.transitions.update(transition_map.values())
    mapping = {**place_map, **transition_map}
    for arc in net.arcs:
        petri_utils.add_arc_from_to(mapping[arc.source], mapping[arc.target], clone, arc.weight)
    return (
        clone,
        Marking({place_map[place]: count for place, count in initial_marking.items()}),
        Marking({place_map[place]: count for place, count in final_marking.items()}),
    )


def _append_visible_suffix(values, labels: Sequence[str]):
    kind, net, initial, final, transformations, nonblock_reason = values
    current_marking = final
    for index, label in enumerate(labels):
        target = PetriNet.Place(f"context_sink_{index:03d}")
        transition = PetriNet.Transition(f"context_{index:03d}_{label}", label)
        net.places.add(target)
        net.transitions.add(transition)
        for place, tokens in current_marking.items():
            petri_utils.add_arc_from_to(place, transition, net, int(tokens))
        petri_utils.add_arc_from_to(transition, target, net)
        current_marking = Marking({target: 1})
    return (
        kind,
        net,
        initial,
        current_marking,
        (*transformations, "shared_sequence_context"),
        nonblock_reason,
    )


def _tau_prefix_refinement(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    name: str,
):
    clone, clone_initial, clone_final = _clone_isomorphic(
        net, initial_marking, final_marking, name
    )
    source = PetriNet.Place("tau_refinement_source")
    tau = PetriNet.Transition("tau_refinement_enter", None)
    clone.places.add(source)
    clone.transitions.add(tau)
    petri_utils.add_arc_from_to(source, tau, clone)
    for place, tokens in clone_initial.items():
        petri_utils.add_arc_from_to(tau, place, clone, int(tokens))
    return clone, Marking({source: 1}), clone_final


def _block_to_spec(node: Any) -> BehaviorSpec:
    if node.op == "leaf":
        return BehaviorSpec("activity", node.label)
    return BehaviorSpec(node.op, children=tuple(_block_to_spec(child) for child in node.children))


def _spec_labels(spec: BehaviorSpec) -> set[str]:
    labels = {spec.label} if spec.label is not None else set()
    for child in spec.children:
        labels.update(_spec_labels(child))
    return labels


def _activity_name(index: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    name = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = letters[remainder] + name
    return name


def _enabled(transition: PetriNet.Transition, marking: Counter) -> bool:
    return all(marking.get(arc.source, 0) >= int(arc.weight) for arc in transition.in_arcs)


def _fire(transition: PetriNet.Transition, marking: Counter) -> Counter:
    result = Counter(marking)
    for arc in transition.in_arcs:
        result[arc.source] -= int(arc.weight)
        if result[arc.source] == 0:
            del result[arc.source]
    for arc in transition.out_arcs:
        result[arc.target] += int(arc.weight)
    return result


def _free_choice_violations(net: PetriNet) -> int:
    violations: set[tuple[str, str]] = set()
    for place in net.places:
        outgoing = [arc.target for arc in place.out_arcs]
        for left_index, left in enumerate(outgoing):
            left_preset = {arc.source for arc in left.in_arcs}
            for right in outgoing[left_index + 1 :]:
                if left_preset != {arc.source for arc in right.in_arcs}:
                    violations.add(tuple(sorted((str(left.name), str(right.name)))))
    return len(violations)


def _reachability_statistics(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    max_states: int = 5000,
) -> dict[str, object]:
    places = sorted(net.places, key=lambda place: str(place.name))
    transitions = sorted(net.transitions, key=lambda transition: str(transition.name))
    initial = Counter({place: int(tokens) for place, tokens in initial_marking.items()})
    final = Counter({place: int(tokens) for place, tokens in final_marking.items()})
    stack = [initial]
    seen: set[tuple[int, ...]] = set()
    fired: set[PetriNet.Transition] = set()
    final_reachable = False
    max_tokens = 0
    arcs: dict[object, list[object]] = {}
    for arc in net.arcs:
        arcs.setdefault(arc.source, []).append(arc.target)
    has_cycle = _has_cycle(arcs)
    truncated = False
    while stack:
        marking = stack.pop()
        key = tuple(marking.get(place, 0) for place in places)
        if key in seen:
            continue
        seen.add(key)
        max_tokens = max(max_tokens, sum(key))
        final_reachable = final_reachable or (+marking == +final)
        if len(seen) >= max_states:
            truncated = bool(stack)
            break
        for transition in transitions:
            if _enabled(transition, marking):
                fired.add(transition)
                stack.append(_fire(transition, marking))
    return {
        "reachable_state_count": len(seen),
        "reachability_truncated": truncated,
        "final_marking_reachable": final_reachable,
        "dead_transition_count": len(net.transitions) - len(fired),
        "max_reachable_token_count": max_tokens,
        "has_cycle": has_cycle,
    }


def _has_cycle(adjacency: dict[object, list[object]]) -> bool:
    visiting: set[object] = set()
    visited: set[object] = set()

    def visit(node: object) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in adjacency.get(node, [])):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in adjacency)


def _weighted_choice(rng: Random, weights: dict[str, float]) -> str:
    normalized = {kind: max(0.0, float(weights.get(kind, 0.0))) for kind in MOTIF_KINDS}
    if sum(normalized.values()) <= 0:
        raise ValueError("at least one behavior-family motif weight must be positive")
    return rng.choices(list(normalized), weights=list(normalized.values()), k=1)[0]


def _mapping(value: object) -> dict:
    return dict(value) if isinstance(value, dict) else {}
