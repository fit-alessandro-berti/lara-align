from __future__ import annotations

from html import escape

from lara_align.types import MoveKind
from lara_ui.replay_service import anchored_alignment


def _move_class(move) -> str:
    if move.kind == MoveKind.SYNCHRONOUS:
        return "sync"
    if move.kind == MoveKind.LOG:
        return "log"
    if move.is_silent_model_move:
        return "silent"
    return "model"


def _move_card(move) -> str:
    kind = _move_class(move)
    icon = {"sync": "✓", "log": "!", "model": "M", "silent": "τ"}[kind]
    observed = escape(move.log_label or ">>")
    identity = escape(move.transition_name or ">>")
    label = escape(move.transition_label if move.transition_label is not None else ("τ" if move.transition_name else ">>"))
    return (
        f'<div class="lara-move {kind}" title="{kind} move">'
        f'<b>{icon} {observed}</b><small>{identity} / {label}</small></div>'
    )


def alignment_ribbon_html(
    labels: list[str],
    candidate=None,
    exact=None,
    mode: str = "Candidate and exact stacked",
    difference_positions: set[int] | None = None,
) -> str:
    difference_positions = difference_positions or set()
    rows = []
    if mode in {"Candidate only", "Candidate and exact stacked", "Difference-focused"}:
        rows.append(("Neural candidate", candidate))
    if mode in {"Exact only", "Candidate and exact stacked", "Difference-focused"}:
        rows.append(("Exact", exact))
    column_count = len(labels) + 1
    html = [
        """<style>
        .lara-ribbon {overflow-x:auto; padding:.25rem 0 1rem;}
        .lara-row {display:grid; gap:.35rem; margin:.45rem 0; min-width:max-content;}
        .lara-label {position:sticky;left:0;background:white;z-index:2;font-weight:700;padding:.4rem;}
        .lara-slot {width:150px;min-height:96px;border:1px solid #cbd5e1;border-radius:8px;padding:5px;background:#f8fafc;}
        .lara-slot.diff {border:3px solid #eab308;}
        .lara-observed {font-size:.78rem;color:#334155;border-bottom:1px solid #cbd5e1;margin-bottom:4px;padding-bottom:3px;}
        .lara-move {padding:4px;border-radius:5px;margin:3px 0;border-left:5px solid;line-height:1.05;}
        .lara-move small {display:block;font-size:.68rem;margin-top:3px;overflow-wrap:anywhere;}
        .lara-move.sync {background:#dcfce7;border-color:#16a34a}.lara-move.log {background:#ffedd5;border-color:#dc2626}
        .lara-move.model {background:#dbeafe;border-color:#2563eb}.lara-move.silent {background:#ede9fe;border-color:#7c3aed}
        </style>""",
        '<div class="lara-ribbon">',
    ]
    for row_name, alignment in rows:
        slots = anchored_alignment(alignment, len(labels))
        visible_indices = range(column_count)
        if mode == "Difference-focused":
            visible_indices = [index for index in range(column_count) if index in difference_positions]
        grid_columns = len(list(visible_indices)) + 1
        html.append(f'<div class="lara-row" style="grid-template-columns:130px repeat({max(grid_columns - 1, 1)},150px)">')
        html.append(f'<div class="lara-label">{escape(row_name)}</div>')
        for index in visible_indices:
            slot = slots[index]
            heading = f"Event {index + 1}: {labels[index]}" if index < len(labels) else "Completion"
            css = "lara-slot diff" if index in difference_positions else "lara-slot"
            html.append(f'<div class="{css}"><div class="lara-observed">{escape(heading)}</div>')
            for move in slot["before"]:
                html.append(_move_card(move))
            if slot["consume"] is not None:
                html.append(_move_card(slot["consume"]))
            elif index < len(labels):
                html.append('<div class="lara-move log"><b>! missing</b></div>')
            html.append("</div>")
        html.append("</div>")
    html.append("</div>")
    return "".join(html)
