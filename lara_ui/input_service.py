from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence

from pm4py.objects.log.obj import Event, Trace
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer


@dataclass
class ParsedLog:
    traces: list[Trace]
    case_ids: list[str]
    activity_key: str
    case_id_key: str
    source_name: str
    empty_trace_count: int


@dataclass
class ParsedNet:
    net: Any
    initial_marking: Any
    final_marking: Any
    source_name: str
    source_kind: str = "uploaded_pnml"
    initial_marking_guessed: bool = False
    final_marking_guessed: bool = False


def content_hash(data: bytes) -> str:
    return sha256(data).hexdigest()


def _temporary_input(data: bytes, suffix: str) -> Path:
    handle = NamedTemporaryFile(prefix="lara_ui_", suffix=suffix, delete=False)
    try:
        handle.write(data)
        return Path(handle.name)
    finally:
        handle.close()


def parse_xes_bytes(
    data: bytes,
    source_name: str,
    activity_key: str = "concept:name",
    case_id_key: str = "concept:name",
    max_cases: int | None = None,
    ignore_empty: bool = True,
    include_lifecycle: bool = False,
) -> ParsedLog:
    path = _temporary_input(data, ".xes")
    try:
        parameters: dict[str, Any] = {"show_progress_bar": False}
        if max_cases is not None:
            parameters["max_traces"] = max_cases
        log = xes_importer.apply(str(path), parameters=parameters)
    finally:
        path.unlink(missing_ok=True)
    return normalize_log(
        log,
        source_name=source_name,
        activity_key=activity_key,
        case_id_key=case_id_key,
        max_cases=max_cases,
        ignore_empty=ignore_empty,
        include_lifecycle=include_lifecycle,
    )


def parse_xes_path(path: str | Path, **kwargs) -> ParsedLog:
    source = Path(path)
    return parse_xes_bytes(source.read_bytes(), source.name, **kwargs)


def normalize_log(
    log: Sequence[object],
    source_name: str,
    activity_key: str,
    case_id_key: str,
    max_cases: int | None,
    ignore_empty: bool,
    include_lifecycle: bool,
) -> ParsedLog:
    traces: list[Trace] = []
    case_ids: list[str] = []
    empty_count = 0
    for original_index, original in enumerate(log):
        if max_cases is not None and len(traces) >= max_cases:
            break
        attributes = dict(getattr(original, "attributes", {}) or {})
        case_id = str(attributes.get(case_id_key, f"case-{original_index + 1}"))
        events: list[Event] = []
        for raw_event in original:
            event = dict(raw_event)
            if activity_key not in event:
                raise KeyError(
                    f"Event in case {case_id!r} has no activity attribute {activity_key!r}"
                )
            label = str(event[activity_key])
            if include_lifecycle and event.get("lifecycle:transition"):
                label = f"{label}+{event['lifecycle:transition']}"
            event["concept:name"] = label
            events.append(Event(event))
        if not events:
            empty_count += 1
            if ignore_empty:
                continue
        traces.append(Trace(events, attributes=attributes))
        case_ids.append(case_id)
    return ParsedLog(
        traces=traces,
        case_ids=case_ids,
        activity_key="concept:name",
        case_id_key=case_id_key,
        source_name=source_name,
        empty_trace_count=empty_count,
    )


def parse_pnml_bytes(data: bytes, source_name: str) -> ParsedNet:
    path = _temporary_input(data, ".pnml")
    try:
        net, initial_marking, final_marking = pnml_importer.apply(str(path))
    finally:
        path.unlink(missing_ok=True)
    return ParsedNet(
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        source_name=source_name,
    )


def parse_pnml_path(path: str | Path) -> ParsedNet:
    source = Path(path)
    return parse_pnml_bytes(source.read_bytes(), source.name)


def save_uploaded_checkpoint(data: bytes, original_name: str) -> Path:
    suffix = Path(original_name).suffix or ".pt"
    digest = content_hash(data)[:20]
    path = Path("/tmp") / f"lara_checkpoint_{digest}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return path


def trusted_checkpoint_paths(root: str | Path = "runs") -> list[Path]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(
        [path for path in base.rglob("*.pt") if path.is_file()],
        key=lambda path: str(path),
    )


def checkpoint_metadata(checkpoint: dict[str, Any], path: str | Path) -> dict[str, Any]:
    metrics = checkpoint.get("metrics") or {}
    config = checkpoint.get("model_config") or {}
    source = Path(path)
    return {
        "identifier": source.name,
        "path": str(source),
        "epoch": checkpoint.get("epoch"),
        "checkpoint_date": source.stat().st_mtime if source.exists() else None,
        "hidden_dimension": config.get("hidden_dim"),
        "latent_regions": config.get("num_regions"),
        "edge_types": config.get("num_edge_types", 2),
        "graph_layers": config.get("graph_layers"),
        "trace_layers": config.get("trace_layers"),
        "training_loss": metrics.get("train_total", metrics.get("train_loss")),
        "validation_loss": metrics.get("val_total", metrics.get("val_loss")),
        "model_config": config,
        "metrics": metrics,
    }
