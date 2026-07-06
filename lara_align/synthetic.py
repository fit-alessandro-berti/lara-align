from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, Sequence

from pm4py.objects.log.obj import Event, Trace
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils


@dataclass
class SyntheticExample:
    net: PetriNet
    initial_marking: Marking
    final_marking: Marking
    trace: Trace
    fitting_trace: Trace
    metadata: dict


def make_sequence_net(
    labels: Sequence[str],
    name: str = "sequence",
    silent_prefix: bool = False,
) -> tuple[PetriNet, Marking, Marking]:
    net = PetriNet(name)
    places = [PetriNet.Place(f"p{i}") for i in range(len(labels) + 1)]
    net.places.update(places)

    previous_place = places[0]
    if silent_prefix:
        prefix_place = PetriNet.Place("p_silent_prefix")
        silent = PetriNet.Transition("tau_prefix", None)
        net.places.add(prefix_place)
        net.transitions.add(silent)
        petri_utils.add_arc_from_to(previous_place, silent, net)
        petri_utils.add_arc_from_to(silent, prefix_place, net)
        previous_place = prefix_place

    for index, label in enumerate(labels):
        transition = PetriNet.Transition(f"t{index}_{label}", label)
        net.transitions.add(transition)
        target = places[index + 1]
        petri_utils.add_arc_from_to(previous_place, transition, net)
        petri_utils.add_arc_from_to(transition, target, net)
        previous_place = target

    return net, Marking({places[0]: 1}), Marking({places[-1]: 1})


def make_duplicate_label_choice_net(
    label: str = "A",
    suffixes: tuple[str, str] = ("B", "C"),
    name: str = "duplicate_label_choice",
) -> tuple[PetriNet, Marking, Marking]:
    """Create an XOR-like net where two concrete transitions share one label."""

    net = PetriNet(name)
    p0 = PetriNet.Place("p0")
    p_left = PetriNet.Place("p_left")
    p_right = PetriNet.Place("p_right")
    p_end_left = PetriNet.Place("p_end_left")
    p_end_right = PetriNet.Place("p_end_right")
    pf = PetriNet.Place("pf")
    net.places.update({p0, p_left, p_right, p_end_left, p_end_right, pf})

    t_left = PetriNet.Transition("t_left_A", label)
    t_right = PetriNet.Transition("t_right_A", label)
    t_left_suffix = PetriNet.Transition(f"t_left_{suffixes[0]}", suffixes[0])
    t_right_suffix = PetriNet.Transition(f"t_right_{suffixes[1]}", suffixes[1])
    tau_left = PetriNet.Transition("tau_left_join", None)
    tau_right = PetriNet.Transition("tau_right_join", None)
    net.transitions.update(
        {t_left, t_right, t_left_suffix, t_right_suffix, tau_left, tau_right}
    )

    petri_utils.add_arc_from_to(p0, t_left, net)
    petri_utils.add_arc_from_to(t_left, p_left, net)
    petri_utils.add_arc_from_to(p0, t_right, net)
    petri_utils.add_arc_from_to(t_right, p_right, net)
    petri_utils.add_arc_from_to(p_left, t_left_suffix, net)
    petri_utils.add_arc_from_to(t_left_suffix, p_end_left, net)
    petri_utils.add_arc_from_to(p_right, t_right_suffix, net)
    petri_utils.add_arc_from_to(t_right_suffix, p_end_right, net)
    petri_utils.add_arc_from_to(p_end_left, tau_left, net)
    petri_utils.add_arc_from_to(tau_left, pf, net)
    petri_utils.add_arc_from_to(p_end_right, tau_right, net)
    petri_utils.add_arc_from_to(tau_right, pf, net)

    return net, Marking({p0: 1}), Marking({pf: 1})


def trace_from_labels(labels: Iterable[str], activity_key: str = "concept:name") -> Trace:
    return Trace([Event({activity_key: label}) for label in labels])


def generate_sequence_example(
    rng: Random | None = None,
    min_len: int = 3,
    max_len: int = 8,
    alphabet: Sequence[str] = tuple("ABCDEFGH"),
    deviation_rate: float = 0.25,
) -> SyntheticExample:
    rng = rng or Random()
    length = rng.randint(min_len, max_len)
    labels = [alphabet[index % len(alphabet)] for index in range(length)]
    net, initial_marking, final_marking = make_sequence_net(
        labels,
        name=f"sequence_{length}",
        silent_prefix=rng.random() < 0.25,
    )
    fitting = trace_from_labels(labels)
    deviated_labels = inject_deviations(labels, rng, deviation_rate, alphabet)
    return SyntheticExample(
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        trace=trace_from_labels(deviated_labels),
        fitting_trace=fitting,
        metadata={
            "family": "sequence",
            "labels": labels,
            "deviated_labels": deviated_labels,
            "deviation_rate": deviation_rate,
        },
    )


def inject_deviations(
    labels: Sequence[str],
    rng: Random,
    deviation_rate: float,
    alphabet: Sequence[str],
) -> list[str]:
    result: list[str] = []
    for label in labels:
        if rng.random() < deviation_rate / 3:
            continue
        if rng.random() < deviation_rate / 3:
            result.append(rng.choice(alphabet))
        if rng.random() < deviation_rate / 3:
            result.append(label)
        result.append(label)

    if len(result) >= 2 and rng.random() < deviation_rate:
        index = rng.randrange(len(result) - 1)
        result[index], result[index + 1] = result[index + 1], result[index]
    if rng.random() < deviation_rate:
        result.append(rng.choice(alphabet))
    return result
