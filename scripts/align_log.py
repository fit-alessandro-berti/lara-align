from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Sequence

os.environ.setdefault("PM4PY_SHOW_PROGRESS_BAR", "False")
sys._pm4py_welcome_shown = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer

from lara_align.certifier import CertifyingAlignmentSystem, LARAMode
from lara_align.checkpoint import load_checkpoint
from lara_align.types import Alignment, AlignmentMove
from lara_align.verify import trace_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align each trace in an event log against a PNML Petri net with a trained LARA model."
    )
    parser.add_argument("event_log", type=Path, help="Path to the XES event log.")
    parser.add_argument("pnml", type=Path, help="Path to the PNML Petri net.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/lara/best.pt"),
        help="Path to the trained LARA checkpoint.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda.")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in LARAMode],
        default=LARAMode.FAST.value,
        help="Alignment mode. fast prints the model reconstruction; certified may repair via pm4py.",
    )
    parser.add_argument(
        "--activity-key",
        default="concept:name",
        help="Event attribute containing the activity label.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional pm4py timeout per trace in certified/anytime mode.",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Maximum number of traces to read from the log.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=0,
        help="Maximum moves shown per alignment. Use 0 to print all moves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)

    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    net, initial_marking, final_marking = pnml_importer.apply(str(args.pnml))
    log = _load_event_log(args.event_log, args.max_traces)

    system = CertifyingAlignmentSystem(model=model, device=device)
    model.eval()

    start = perf_counter()
    lines = [
        "LARA Trace Alignments",
        "=====================",
        f"log:              {args.event_log}",
        f"pnml:             {args.pnml}",
        f"checkpoint:       {args.checkpoint}",
        f"checkpoint epoch: {checkpoint.get('epoch')}",
        f"device:           {device}",
        f"mode:             {args.mode}",
        f"traces:           {len(log)}",
    ]

    with torch.no_grad():
        for index, trace in enumerate(log, start=1):
            result = system.align(
                net,
                initial_marking,
                final_marking,
                trace,
                mode=args.mode,
                activity_key=args.activity_key,
                timeout_seconds=args.timeout_seconds,
            )
            lines.extend(_format_trace_result(index, trace, result, args))

    lines.append("")
    lines.append(f"elapsed: {perf_counter() - start:.2f}s")
    print("\n".join(lines))


def _validate_args(args: argparse.Namespace) -> None:
    for path_name in ("event_log", "pnml", "checkpoint"):
        path = getattr(args, path_name)
        if not path.exists():
            raise SystemExit(f"{path_name.replace('_', ' ')} does not exist: {path}")
        if not path.is_file():
            raise SystemExit(f"{path_name.replace('_', ' ')} is not a file: {path}")
    if args.max_traces is not None and args.max_traces <= 0:
        raise SystemExit("--max-traces must be greater than zero")
    if args.max_moves < 0:
        raise SystemExit("--max-moves must be zero or greater")


def _load_event_log(path: Path, max_traces: int | None) -> Sequence[object]:
    parameters: dict[str, Any] = {"show_progress_bar": False}
    if max_traces is not None:
        parameters["max_traces"] = max_traces
    return xes_importer.apply(str(path), parameters=parameters)


def _format_trace_result(
    index: int,
    trace: object,
    result,
    args: argparse.Namespace,
) -> list[str]:
    labels = trace_labels(trace, activity_key=args.activity_key)
    trace_name = _trace_name(trace)
    title = f"[{index}]"
    if trace_name is not None:
        title += f" trace={trace_name}"

    lines = [
        "",
        title,
        f"events: {' '.join(labels)}",
        (
            f"legal={result.legal} cost={result.cost} "
            f"certified_optimal={result.certified_optimal}"
        ),
    ]
    if result.verifier is not None and result.verifier.reason:
        lines.append(f"failure_reason={result.verifier.reason}")
    if result.diagnostics:
        source = result.diagnostics.get("candidate_source")
        if source is not None:
            lines.append(f"candidate_source={source}")

    lines.append("alignment:")
    if result.alignment is None:
        lines.append("  <no alignment>")
    else:
        max_moves = None if args.max_moves == 0 else args.max_moves
        lines.extend(_format_alignment(result.alignment, max_moves))
    return lines


def _trace_name(trace: object) -> str | None:
    attributes = getattr(trace, "attributes", None)
    if not attributes:
        return None
    name = attributes.get("concept:name")
    return None if name is None else str(name)


def _format_alignment(alignment: Alignment, max_moves: int | None) -> list[str]:
    moves = alignment.moves if max_moves is None else alignment.moves[:max_moves]
    rows_as_text = [
        (
            str(index),
            _move_log(move),
            _move_transition(move),
            _move_model_label(move),
            move.kind.value,
        )
        for index, move in enumerate(moves)
    ]
    widths = _column_widths(
        rows_as_text,
        ("idx", "log", "model_transition", "model_label", "kind"),
    )
    rows = [
        (
            f"  {'idx':>{widths[0]}}  {'log':<{widths[1]}}  "
            f"{'model_transition':<{widths[2]}}  "
            f"{'model_label':<{widths[3]}}  {'kind':<{widths[4]}}"
        )
    ]
    for index, log_label, transition, model_label, kind in rows_as_text:
        rows.append(
            f"  {index:>{widths[0]}}  {log_label:<{widths[1]}}  "
            f"{transition:<{widths[2]}}  {model_label:<{widths[3]}}  "
            f"{kind:<{widths[4]}}"
        )
    if max_moves is not None and len(alignment.moves) > max_moves:
        rows.append(f"  ... {len(alignment.moves) - max_moves} more moves")
    return rows


def _column_widths(
    rows: list[tuple[str, str, str, str, str]],
    headers: tuple[str, str, str, str, str],
) -> tuple[int, int, int, int, int]:
    return tuple(
        max(len(row[column]) for row in [headers, *rows])
        for column in range(len(headers))
    )


def _move_log(move: AlignmentMove) -> str:
    return move.log_label if move.log_label is not None else ">>"


def _move_transition(move: AlignmentMove) -> str:
    return move.transition_name if move.transition_name is not None else ">>"


def _move_model_label(move: AlignmentMove) -> str:
    if move.transition_name is None:
        return ">>"
    return move.transition_label if move.transition_label is not None else "tau"


if __name__ == "__main__":
    main()
