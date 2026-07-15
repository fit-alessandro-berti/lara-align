from __future__ import annotations

def _dot_quote(value: object) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def petri_net_dot(
    net,
    initial_marking,
    final_marking,
    alignment=None,
    current_marking=None,
    fired_transition: str | None = None,
    enabled: set[str] | None = None,
) -> str:
    alignment_transitions = {
        move.transition_name
        for move in alignment.moves
        if move.transition_name is not None
    } if alignment is not None else set()
    enabled = enabled or set()
    marking = current_marking if current_marking is not None else initial_marking
    duplicate_counts: dict[str, int] = {}
    for transition in net.transitions:
        if transition.label is not None:
            duplicate_counts[str(transition.label)] = duplicate_counts.get(str(transition.label), 0) + 1

    lines = [
        "digraph PetriNet {",
        'rankdir="LR"; bgcolor="transparent"; graph [pad="0.2", nodesep="0.35"];',
        'node [fontname="Arial", fontsize="10"]; edge [color="#64748b"];',
    ]
    for place in sorted(net.places, key=lambda item: str(item.name)):
        tokens = int(marking.get(place, 0))
        final = int(final_marking.get(place, 0))
        label = str(place.name) + (f"\n● {tokens}" if tokens else "")
        border = 3 if final else 1
        fill = "#ecfeff" if tokens else "#ffffff"
        tooltip = f"Place {place.name}; tokens={tokens}; final tokens={final}"
        lines.append(
            f"p_{id(place)} [shape=circle, label={_dot_quote(label)}, "
            f"style=filled, fillcolor={_dot_quote(fill)}, penwidth={border}, "
            f"tooltip={_dot_quote(tooltip)}];"
        )
    for transition in sorted(net.transitions, key=lambda item: str(item.name)):
        name = str(transition.name)
        visible = transition.label is not None
        label = f"{transition.label}\n{name}" if visible else f"τ\n{name}"
        fill = "#dcfce7" if name in alignment_transitions else "#f8fafc"
        color = "#dc2626" if name == fired_transition else "#16a34a" if name in enabled else "#475569"
        duplicate = visible and duplicate_counts.get(str(transition.label), 0) > 1
        tooltip = (
            f"Transition ID: {name}; label: {transition.label or 'tau'}; "
            f"inputs: {', '.join(str(arc.source.name) for arc in transition.in_arcs)}; "
            f"outputs: {', '.join(str(arc.target.name) for arc in transition.out_arcs)}; "
            f"duplicate label: {duplicate}"
        )
        lines.append(
            f"t_{id(transition)} [shape=box, label={_dot_quote(label)}, "
            f"style=filled, fillcolor={_dot_quote(fill)}, color={_dot_quote(color)}, "
            f"penwidth={3 if name == fired_transition else 2 if name in enabled else 1}, "
            f"tooltip={_dot_quote(tooltip)}];"
        )
    for arc in net.arcs:
        source = f"p_{id(arc.source)}" if arc.source in net.places else f"t_{id(arc.source)}"
        target = f"p_{id(arc.target)}" if arc.target in net.places else f"t_{id(arc.target)}"
        label = "" if int(arc.weight) == 1 else str(arc.weight)
        lines.append(f"{source} -> {target} [label={_dot_quote(label)}];")
    lines.append("}")
    return "\n".join(lines)
