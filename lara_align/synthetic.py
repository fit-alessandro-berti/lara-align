from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class _BlockNode:
    """Node of a random block-structured process tree."""

    op: str  # "leaf" | "seq" | "xor" | "and" | "loop"
    label: str | None = None
    children: list["_BlockNode"] = field(default_factory=list)

    def count_ops(self, counts: dict[str, int] | None = None) -> dict[str, int]:
        counts = counts if counts is not None else {}
        counts[self.op] = counts.get(self.op, 0) + 1
        for child in self.children:
            child.count_ops(counts)
        return counts


def _random_block_tree(
    rng: Random,
    num_leaves: int,
    alphabet: Sequence[str],
    operator_weights: dict[str, float],
    depth: int = 0,
    max_depth: int = 6,
) -> _BlockNode:
    if num_leaves <= 1 or depth >= max_depth:
        return _BlockNode(op="leaf", label=rng.choice(list(alphabet)))

    operators = list(operator_weights)
    weights = [operator_weights[op] for op in operators]
    op = rng.choices(operators, weights=weights, k=1)[0]

    if op == "loop":
        body_leaves = max(1, num_leaves - 1)
        redo_leaves = num_leaves - body_leaves
        children_sizes = [body_leaves, max(1, redo_leaves)]
    else:
        max_children = min(4 if op != "seq" else 5, num_leaves)
        num_children = rng.randint(2, max(2, max_children))
        children_sizes = _partition(rng, num_leaves, num_children)

    children = [
        _random_block_tree(rng, size, alphabet, operator_weights, depth + 1, max_depth)
        for size in children_sizes
    ]
    return _BlockNode(op=op, children=children)


def _partition(rng: Random, total: int, parts: int) -> list[int]:
    parts = min(parts, total)
    sizes = [1] * parts
    for _ in range(total - parts):
        sizes[rng.randrange(parts)] += 1
    return sizes


def make_block_structured_net(
    tree: _BlockNode,
    name: str = "block_structured",
) -> tuple[PetriNet, Marking, Marking]:
    """Compile a block-structured process tree into a Petri net.

    Sequence blocks chain sub-blocks through fresh places, choice blocks share
    entry/exit places, parallel blocks fork and join through invisible
    transitions, and loop blocks isolate a body/redo pair behind invisible
    entry/exit transitions. Playouts of the tree are replayable on the net
    from the initial to the final marking by construction.
    """

    net = PetriNet(name)
    source = PetriNet.Place("p_source")
    sink = PetriNet.Place("p_sink")
    net.places.update({source, sink})
    counter = {"t": 0, "p": 0}
    _compile_block(net, tree, source, sink, counter)
    return net, Marking({source: 1}), Marking({sink: 1})


def _fresh_place(net: PetriNet, counter: dict[str, int]) -> PetriNet.Place:
    place = PetriNet.Place(f"p{counter['p']}")
    counter["p"] += 1
    net.places.add(place)
    return place


def _fresh_transition(
    net: PetriNet,
    counter: dict[str, int],
    label: str | None,
) -> PetriNet.Transition:
    suffix = label if label is not None else "tau"
    transition = PetriNet.Transition(f"t{counter['t']}_{suffix}", label)
    counter["t"] += 1
    net.transitions.add(transition)
    return transition


def _compile_block(
    net: PetriNet,
    node: _BlockNode,
    entry: PetriNet.Place,
    exit_place: PetriNet.Place,
    counter: dict[str, int],
) -> None:
    if node.op == "leaf":
        transition = _fresh_transition(net, counter, node.label)
        petri_utils.add_arc_from_to(entry, transition, net)
        petri_utils.add_arc_from_to(transition, exit_place, net)
        return

    if node.op == "seq":
        current = entry
        for index, child in enumerate(node.children):
            target = (
                exit_place
                if index == len(node.children) - 1
                else _fresh_place(net, counter)
            )
            _compile_block(net, child, current, target, counter)
            current = target
        return

    if node.op == "xor":
        for child in node.children:
            _compile_block(net, child, entry, exit_place, counter)
        return

    if node.op == "and":
        split = _fresh_transition(net, counter, None)
        join = _fresh_transition(net, counter, None)
        petri_utils.add_arc_from_to(entry, split, net)
        petri_utils.add_arc_from_to(join, exit_place, net)
        for child in node.children:
            child_entry = _fresh_place(net, counter)
            child_exit = _fresh_place(net, counter)
            petri_utils.add_arc_from_to(split, child_entry, net)
            petri_utils.add_arc_from_to(child_exit, join, net)
            _compile_block(net, child, child_entry, child_exit, counter)
        return

    if node.op == "loop":
        enter = _fresh_transition(net, counter, None)
        leave = _fresh_transition(net, counter, None)
        body_entry = _fresh_place(net, counter)
        body_exit = _fresh_place(net, counter)
        petri_utils.add_arc_from_to(entry, enter, net)
        petri_utils.add_arc_from_to(enter, body_entry, net)
        petri_utils.add_arc_from_to(body_exit, leave, net)
        petri_utils.add_arc_from_to(leave, exit_place, net)
        body, redo = node.children[0], node.children[1]
        _compile_block(net, body, body_entry, body_exit, counter)
        _compile_block(net, redo, body_exit, body_entry, counter)
        return

    raise ValueError(f"unknown block operator: {node.op}")


def _playout_block(
    node: _BlockNode,
    rng: Random,
    loop_redo_probability: float,
    max_loop_redos: int,
) -> list[str]:
    if node.op == "leaf":
        return [node.label] if node.label is not None else []

    if node.op == "seq":
        result: list[str] = []
        for child in node.children:
            result.extend(_playout_block(child, rng, loop_redo_probability, max_loop_redos))
        return result

    if node.op == "xor":
        child = rng.choice(node.children)
        return _playout_block(child, rng, loop_redo_probability, max_loop_redos)

    if node.op == "and":
        sequences = [
            _playout_block(child, rng, loop_redo_probability, max_loop_redos)
            for child in node.children
        ]
        return _random_interleaving(sequences, rng)

    if node.op == "loop":
        body, redo = node.children[0], node.children[1]
        result = _playout_block(body, rng, loop_redo_probability, max_loop_redos)
        redos = 0
        while redos < max_loop_redos and rng.random() < loop_redo_probability:
            result.extend(_playout_block(redo, rng, loop_redo_probability, max_loop_redos))
            result.extend(_playout_block(body, rng, loop_redo_probability, max_loop_redos))
            redos += 1
        return result

    raise ValueError(f"unknown block operator: {node.op}")


def _random_interleaving(sequences: list[list[str]], rng: Random) -> list[str]:
    pending = [list(sequence) for sequence in sequences if sequence]
    result: list[str] = []
    while pending:
        index = rng.randrange(len(pending))
        result.append(pending[index].pop(0))
        if not pending[index]:
            pending.pop(index)
    return result


def generate_block_structured_example(
    rng: Random | None = None,
    num_leaves: int = 12,
    alphabet_size: int | None = None,
    deviation_rate: float = 0.25,
    operator_weights: dict[str, float] | None = None,
    loop_redo_probability: float = 0.3,
    max_loop_redos: int = 2,
    max_trace_len: int = 200,
    max_depth: int = 6,
) -> SyntheticExample:
    """Generate a larger block-structured net with concurrency, choices, loops,
    duplicate labels, and invisible transitions, plus a deviating trace.

    `alphabet_size` below `num_leaves` forces duplicate labels; it defaults to
    roughly 70% of the leaves. Deviations are injected with the same
    perturbation model used for the simple families.
    """

    rng = rng or Random()
    if alphabet_size is None:
        alphabet_size = max(2, int(round(num_leaves * 0.7)))
    alphabet = [_activity_name(index) for index in range(alphabet_size)]
    weights = operator_weights or {"seq": 0.45, "xor": 0.25, "and": 0.2, "loop": 0.1}

    tree = _random_block_tree(rng, num_leaves, alphabet, weights, max_depth=max_depth)
    net, initial_marking, final_marking = make_block_structured_net(
        tree, name=f"block_{num_leaves}"
    )

    fitting_labels = _playout_block(tree, rng, loop_redo_probability, max_loop_redos)
    while not fitting_labels:
        fitting_labels = _playout_block(tree, rng, loop_redo_probability, max_loop_redos)
    fitting_labels = fitting_labels[:max_trace_len]
    deviated_labels = inject_deviations(fitting_labels, rng, deviation_rate, alphabet)
    deviated_labels = deviated_labels[:max_trace_len]

    return SyntheticExample(
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        trace=trace_from_labels(deviated_labels),
        fitting_trace=trace_from_labels(fitting_labels),
        metadata={
            "family": "block_structured",
            "num_leaves": num_leaves,
            "alphabet_size": alphabet_size,
            "operator_counts": tree.count_ops(),
            "num_places": len(net.places),
            "num_transitions": len(net.transitions),
            "num_invisible": sum(1 for t in net.transitions if t.label is None),
            "num_duplicate_labels": _duplicate_label_count(net),
            "labels": fitting_labels,
            "deviated_labels": deviated_labels,
            "deviation_rate": deviation_rate,
        },
    )


def _activity_name(index: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    name = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = letters[remainder] + name
    return name


def _duplicate_label_count(net: PetriNet) -> int:
    label_counts: dict[str, int] = {}
    for transition in net.transitions:
        if transition.label is not None:
            label = str(transition.label)
            label_counts[label] = label_counts.get(label, 0) + 1
    return sum(count for count in label_counts.values() if count > 1)
