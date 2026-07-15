from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Sequence

import torch
from pm4py.objects.petri_net.obj import Marking, PetriNet

from lara_align.types import CostModel
from lara_align.verify import trace_labels


DEFAULT_HASH_VOCAB_SIZE = 8192


def stable_label_id(
    label: str | None, hash_vocab_size: int = DEFAULT_HASH_VOCAB_SIZE
) -> int:
    """Map labels into a stable bounded vocabulary for neural embeddings."""

    if label is None:
        return 0
    if hash_vocab_size <= 1:
        raise ValueError("hash_vocab_size must be greater than 1")
    digest = blake2b(str(label).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return 1 + (value % (hash_vocab_size - 1))


@dataclass
class PetriTraceFeatures:
    place_features: torch.Tensor
    transition_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    transition_label_ids: torch.Tensor
    event_label_ids: torch.Tensor
    compatibility: torch.Tensor
    place_names: list[str]
    transition_names: list[str]
    transition_labels: list[str | None]
    event_labels: list[str]

    @property
    def num_places(self) -> int:
        return len(self.place_names)

    @property
    def num_transitions(self) -> int:
        return len(self.transition_names)

    @property
    def trace_length(self) -> int:
        return len(self.event_labels)

    def to(self, device: torch.device | str) -> "PetriTraceFeatures":
        return PetriTraceFeatures(
            place_features=self.place_features.to(device),
            transition_features=self.transition_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            transition_label_ids=self.transition_label_ids.to(device),
            event_label_ids=self.event_label_ids.to(device),
            compatibility=self.compatibility.to(device),
            place_names=self.place_names,
            transition_names=self.transition_names,
            transition_labels=self.transition_labels,
            event_labels=self.event_labels,
        )


def _place_degree(place: PetriNet.Place) -> tuple[int, int]:
    return len(place.in_arcs), len(place.out_arcs)


def _transition_degree(transition: PetriNet.Transition) -> tuple[int, int]:
    return len(transition.in_arcs), len(transition.out_arcs)


def pm4py_to_features(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    trace: Sequence[object],
    cost_model: CostModel | None = None,
    activity_key: str = "concept:name",
    hash_vocab_size: int = DEFAULT_HASH_VOCAB_SIZE,
) -> PetriTraceFeatures:
    """Convert a pm4py Petri net and trace into typed tensors.

    The graph is represented as a single bipartite node set with places first
    and transitions second. Arcs are materialized in both directions so that
    message passing can propagate context forward and backward:

    - `edge_type=0`: place-to-transition (consumption, along the arc);
    - `edge_type=1`: transition-to-place (production, along the arc);
    - `edge_type=2`: transition-to-place reversal of consumption (a place
      hears from its consumers);
    - `edge_type=3`: place-to-transition reversal of production (a transition
      hears from its output places).

    Without the reverse types (2 and 3), duplicate-labeled transitions with
    symmetric presets receive provably identical embeddings regardless of
    their downstream context, which makes label-to-transition disambiguation
    impossible. Models trained before reverse edges existed simply ignore
    types 2 and 3.
    """

    costs = cost_model or CostModel()
    places = sorted(net.places, key=lambda place: str(place.name))
    transitions = sorted(net.transitions, key=lambda transition: str(transition.name))
    place_index = {place: index for index, place in enumerate(places)}
    transition_index = {transition: index for index, transition in enumerate(transitions)}

    place_rows: list[list[float]] = []
    for place in places:
        in_degree, out_degree = _place_degree(place)
        place_rows.append(
            [
                float(initial_marking.get(place, 0)),
                float(final_marking.get(place, 0)),
                float(in_degree),
                float(out_degree),
                float(in_degree + out_degree),
            ]
        )

    transition_rows: list[list[float]] = []
    transition_label_ids: list[int] = []
    transition_names: list[str] = []
    transition_labels: list[str | None] = []
    for transition in transitions:
        in_degree, out_degree = _transition_degree(transition)
        name = str(transition.name)
        label = None if transition.label is None else str(transition.label)
        transition_rows.append(
            [
                1.0 if label is None else 0.0,
                float(in_degree),
                float(out_degree),
                float(costs.model_cost(name, label)),
                float(costs.sync_cost(name)),
                1.0 if _has_self_loop_like_context(transition) else 0.0,
            ]
        )
        transition_label_ids.append(stable_label_id(label, hash_vocab_size))
        transition_names.append(name)
        transition_labels.append(label)

    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_types: list[int] = []
    transition_offset = len(places)
    for transition in transitions:
        t_node = transition_offset + transition_index[transition]
        for arc in transition.in_arcs:
            edge_sources.append(place_index[arc.source])
            edge_targets.append(t_node)
            edge_types.append(0)
            edge_sources.append(t_node)
            edge_targets.append(place_index[arc.source])
            edge_types.append(2)
        for arc in transition.out_arcs:
            edge_sources.append(t_node)
            edge_targets.append(place_index[arc.target])
            edge_types.append(1)
            edge_sources.append(place_index[arc.target])
            edge_targets.append(t_node)
            edge_types.append(3)

    labels = trace_labels(trace, activity_key=activity_key)
    event_label_ids = [stable_label_id(label, hash_vocab_size) for label in labels]
    # Vectorized exact label equality: distinct labels get distinct dense ids,
    # invisible transitions get -1 and never match an event.
    dense_ids: dict[str, int] = {}
    event_dense = [dense_ids.setdefault(label, len(dense_ids)) for label in labels]
    transition_dense = [
        -1 if label is None else dense_ids.setdefault(label, len(dense_ids))
        for label in transition_labels
    ]
    if labels and transitions:
        compatibility = torch.tensor(event_dense, dtype=torch.long).unsqueeze(
            1
        ) == torch.tensor(transition_dense, dtype=torch.long).unsqueeze(0)
    else:
        compatibility = torch.zeros((len(labels), len(transitions)), dtype=torch.bool)

    place_features = torch.tensor(place_rows, dtype=torch.float32).reshape(len(places), 5)
    transition_features = torch.tensor(transition_rows, dtype=torch.float32).reshape(
        len(transitions), 6
    )
    if edge_sources:
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    return PetriTraceFeatures(
        place_features=place_features,
        transition_features=transition_features,
        edge_index=edge_index,
        edge_type=edge_type,
        transition_label_ids=torch.tensor(transition_label_ids, dtype=torch.long),
        event_label_ids=torch.tensor(event_label_ids, dtype=torch.long),
        compatibility=compatibility,
        place_names=[str(place.name) for place in places],
        transition_names=transition_names,
        transition_labels=transition_labels,
        event_labels=labels,
    )


def _has_self_loop_like_context(transition: PetriNet.Transition) -> bool:
    preset = {arc.source for arc in transition.in_arcs}
    postset = {arc.target for arc in transition.out_arcs}
    return bool(preset.intersection(postset))
