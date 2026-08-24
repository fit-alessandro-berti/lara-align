from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pm4py  # noqa: E402
from pm4py.objects.log.exporter.xes import exporter as xes_exporter  # noqa: E402
from pm4py.objects.log.importer.xes import importer as xes_importer  # noqa: E402
from pm4py.objects.log.obj import Event, EventLog, Trace  # noqa: E402
from pm4py.objects.petri_net.obj import Marking, PetriNet  # noqa: E402

from lara_align.data import SPLITS, AlignmentSample, load_metadata, load_split  # noqa: E402


SCHEMA_VERSION = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export every LARA foundation-model sample to a self-contained identifier "
            "folder with PNML, XES, and PM4Py-converted BPMN files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/lara_synthetic"),
        help="Directory containing train.pkl, val.pkl, test.pkl, and metadata.pkl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/foundation_model_pm4py_export"),
        help="Destination directory for the portable PM4Py export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination after a new export validates successfully.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    output_dir = args.output.resolve()
    _validate_locations(data_dir, output_dir)
    _validate_source_files(data_dir)
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"{output_dir} already exists; pass --overwrite to replace it")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging_dir = Path(temporary)
        manifest = export_dataset(data_dir, staging_dir)
        _write_manifest_files(staging_dir, manifest)
        if output_dir.exists():
            if output_dir.is_dir() and not output_dir.is_symlink():
                shutil.rmtree(output_dir)
            else:
                output_dir.unlink()
        staging_dir.replace(output_dir)

    totals = manifest["totals"]
    print(
        "export complete: "
        f"{totals['samples']} samples, "
        f"{totals['petri_nets']} PNML files, "
        f"{totals['event_logs']} XES files, and "
        f"{totals['bpmn_models']} BPMN files"
    )
    print(f"output: {output_dir}")
    print(f"manifest: {output_dir / 'manifest.json'}")


def export_dataset(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    dataset_metadata = load_metadata(data_dir)
    expected_counts = dict(dataset_metadata.split_counts)
    source_files = {
        f"{split}.pkl": _source_file_record(data_dir / f"{split}.pkl", data_dir)
        for split in SPLITS
    }
    for name in ("metadata.pkl", "metadata.json"):
        path = data_dir / name
        if path.exists():
            source_files[name] = _source_file_record(path, data_dir)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Portable, self-contained PM4Py export of every LARA foundation-model "
            "train, validation, and test sample."
        ),
        "source_dataset": {
            "directory": str(data_dir),
            "metadata": asdict(dataset_metadata),
            "files": source_files,
        },
        "generator": {
            "script": "scripts/export_foundation_data.py",
            "pm4py_version": pm4py.__version__,
        },
        "organization": {
            "unit": "sample_id",
            "layout": "<split>/<sample_id>/<sample_id>.{pnml,bpmn,xes}",
            "coverage": (
                "every source sample has its own folder containing its PNML, BPMN, "
                "and single-trace XES files"
            ),
        },
        "splits": {},
        "totals": {
            "samples": 0,
            "petri_nets": 0,
            "event_logs": 0,
            "bpmn_models": 0,
            "distinct_petri_nets": 0,
            "distinct_event_logs": 0,
        },
        "petri_nets": [],
        "event_logs": [],
        "samples": [],
        "validation": {
            "status": "passed",
            "method": (
                "Every PNML, BPMN, and XES artifact was read back with PM4Py. "
                "PNML structure/markings, BPMN node/flow counts, and XES event data "
                "were compared with their in-memory sources."
            ),
            "pnml_files_read_back": 0,
            "bpmn_files_read_back": 0,
            "xes_files_read_back": 0,
        },
    }
    checksum_lines: list[tuple[str, str]] = []

    for split in SPLITS:
        samples = load_split(data_dir, split)
        expected = expected_counts.get(split)
        if expected is None or len(samples) != expected:
            raise ValueError(
                f"{split}.pkl contains {len(samples)} samples; metadata expects {expected}"
            )
        _validate_samples(samples, split)
        split_result = _export_split(samples, split, output_dir, checksum_lines)
        manifest["splits"][split] = split_result["summary"]
        manifest["petri_nets"].extend(split_result["petri_nets"])
        manifest["event_logs"].extend(split_result["event_logs"])
        manifest["samples"].extend(split_result["samples"])
        manifest["totals"]["samples"] += split_result["summary"]["samples"]
        manifest["totals"]["petri_nets"] += split_result["summary"]["petri_nets"]
        manifest["totals"]["event_logs"] += split_result["summary"]["event_logs"]
        manifest["totals"]["bpmn_models"] += split_result["summary"]["bpmn_models"]
        manifest["totals"]["distinct_petri_nets"] += split_result["summary"][
            "distinct_petri_nets"
        ]
        manifest["totals"]["distinct_event_logs"] += split_result["summary"][
            "distinct_event_logs"
        ]
        manifest["validation"]["pnml_files_read_back"] += split_result["summary"][
            "petri_nets"
        ]
        manifest["validation"]["bpmn_files_read_back"] += split_result["summary"][
            "bpmn_models"
        ]
        manifest["validation"]["xes_files_read_back"] += split_result["summary"][
            "event_logs"
        ]
        print(
            f"{split}: {len(samples)} samples -> "
            f"{split_result['summary']['petri_nets']} PNML/BPMN pairs and "
            f"{split_result['summary']['event_logs']} XES logs"
        )

    if len({row["sample_id"] for row in manifest["samples"]}) != manifest["totals"][
        "samples"
    ]:
        raise ValueError("sample_id values are not unique across dataset splits")

    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(checksum_lines)),
        encoding="utf-8",
    )
    return manifest


def _export_split(
    samples: list[AlignmentSample],
    split: str,
    output_dir: Path,
    checksum_lines: list[tuple[str, str]],
) -> dict[str, Any]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, AlignmentSample] = {}
    logs: dict[str, AlignmentSample] = {}
    for sample in samples:
        variant_id = _required_metadata(sample, "variant_id")
        trace_id = _required_metadata(sample, "trace_id")
        previous_model = models.setdefault(variant_id, sample)
        if _petri_fingerprint(
            previous_model.net,
            previous_model.initial_marking,
            previous_model.final_marking,
        ) != _petri_fingerprint(sample.net, sample.initial_marking, sample.final_marking):
            raise ValueError(f"variant_id {variant_id!r} refers to different Petri nets")
        previous_log = logs.setdefault(trace_id, sample)
        if _trace_fingerprint(previous_log.trace) != _trace_fingerprint(sample.trace):
            raise ValueError(f"trace_id {trace_id!r} refers to different event logs")

    sample_stems = _artifact_stems({sample.sample_id: sample for sample in samples})
    petri_records: list[dict[str, Any]] = []
    log_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    bpmn_cache: dict[str, Any] = {}

    for sample in samples:
        variant_id = _required_metadata(sample, "variant_id")
        trace_id = _required_metadata(sample, "trace_id")
        stem = sample_stems[sample.sample_id]
        sample_dir = split_dir / stem
        sample_dir.mkdir()
        pnml_path = sample_dir / f"{stem}.pnml"
        bpmn_path = sample_dir / f"{stem}.bpmn"
        xes_path = sample_dir / f"{stem}.xes"

        pm4py.write_pnml(
            sample.net,
            sample.initial_marking,
            sample.final_marking,
            str(pnml_path),
        )
        bpmn = bpmn_cache.get(variant_id)
        if bpmn is None:
            bpmn = pm4py.convert_to_bpmn(
                sample.net, sample.initial_marking, sample.final_marking
            )
            bpmn_cache[variant_id] = bpmn
        pm4py.write_bpmn(bpmn, str(bpmn_path))

        event_log = _single_trace_event_log(sample, split, trace_id)
        expected_trace_attributes = dict(event_log[0].attributes)
        # The object exporter preserves valid zero-event traces. The optional
        # Rust-backed convenience writer converts through a DataFrame and drops
        # precisely those traces because they have no event rows.
        xes_exporter.apply(
            event_log,
            str(xes_path),
            variant=xes_exporter.Variants.LINE_BY_LINE,
            parameters={"show_progress_bar": False},
        )

        _validate_pnml_round_trip(
            pnml_path, sample.net, sample.initial_marking, sample.final_marking
        )
        _validate_bpmn_round_trip(bpmn_path, bpmn)
        _validate_xes_round_trip(xes_path, sample.trace, expected_trace_attributes)

        pnml_relative = pnml_path.relative_to(output_dir).as_posix()
        bpmn_relative = bpmn_path.relative_to(output_dir).as_posix()
        xes_relative = xes_path.relative_to(output_dir).as_posix()
        pnml_digest = _sha256_file(pnml_path)
        bpmn_digest = _sha256_file(bpmn_path)
        xes_digest = _sha256_file(xes_path)
        checksum_lines.extend(
            [
                (pnml_relative, pnml_digest),
                (bpmn_relative, bpmn_digest),
                (xes_relative, xes_digest),
            ]
        )
        petri_records.append(
            {
                "sample_id": sample.sample_id,
                "split": split,
                "behavior_id": _required_metadata(sample, "behavior_id"),
                "variant_id": variant_id,
                "representation_kind": sample.metadata.get("representation_kind"),
                "pnml_path": pnml_relative,
                "pnml_sha256": pnml_digest,
                "bpmn_path": bpmn_relative,
                "bpmn_sha256": bpmn_digest,
                "places": len(sample.net.places),
                "transitions": len(sample.net.transitions),
                "arcs": len(sample.net.arcs),
                "bpmn_nodes": len(bpmn.get_nodes()),
                "bpmn_flows": len(bpmn.get_flows()),
            }
        )
        log_records.append(
            {
                "sample_id": sample.sample_id,
                "split": split,
                "behavior_id": _required_metadata(sample, "behavior_id"),
                "trace_id": trace_id,
                "xes_path": xes_relative,
                "xes_sha256": xes_digest,
                "traces": 1,
                "events": len(sample.trace),
            }
        )
        sample_records.append(
            {
                "sample_id": sample.sample_id,
                "split": split,
                "behavior_id": _required_metadata(sample, "behavior_id"),
                "variant_id": variant_id,
                "trace_id": trace_id,
                "representation_kind": sample.metadata.get("representation_kind"),
                "optimal_cost": sample.optimal_cost,
                "pnml_path": pnml_relative,
                "bpmn_path": bpmn_relative,
                "xes_path": xes_relative,
            }
        )

    return {
        "summary": {
            "source_file": f"{split}.pkl",
            "samples": len(samples),
            "petri_nets": len(samples),
            "event_logs": len(samples),
            "bpmn_models": len(samples),
            "distinct_petri_nets": len(models),
            "distinct_event_logs": len(logs),
        },
        "petri_nets": petri_records,
        "event_logs": log_records,
        "samples": sample_records,
    }


def _single_trace_event_log(
    sample: AlignmentSample, split: str, trace_id: str
) -> EventLog:
    trace = Trace(
        [Event(dict(event)) for event in sample.trace],
        attributes={
            "concept:name": trace_id,
            "lara:split": split,
            "lara:behavior_id": _required_metadata(sample, "behavior_id"),
            "lara:trace_id": trace_id,
            "lara:sample_id": sample.sample_id,
        },
    )
    return EventLog(
        [trace],
        attributes={
            "concept:name": f"LARA foundation-model {split} log {trace_id}",
        },
    )


def _write_manifest_files(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sample_fields = [
        "sample_id",
        "split",
        "behavior_id",
        "variant_id",
        "trace_id",
        "representation_kind",
        "optimal_cost",
        "pnml_path",
        "bpmn_path",
        "xes_path",
    ]
    with (output_dir / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(manifest["samples"])
    (output_dir / "README.md").write_text(_export_readme(manifest), encoding="utf-8")


def _export_readme(manifest: dict[str, Any]) -> str:
    totals = manifest["totals"]
    split_lines = "\n".join(
        f"- `{split}`: {values['samples']} samples, {values['petri_nets']} PNML/BPMN "
        f"pairs, {values['event_logs']} XES logs"
        for split, values in manifest["splits"].items()
    )
    return f"""# LARA foundation-model PM4Py export

This folder contains every sample from the train, validation (`val`), and test
splits. Each `<split>/<sample_id>/` directory is self-contained and holds
`<sample_id>.pnml`, `<sample_id>.bpmn`, and `<sample_id>.xes`. PM4Py
{manifest['generator']['pm4py_version']} exported the PNML/XES files and converted
every Petri net to BPMN.

The source dataset contains {totals['samples']} samples, so this export has
{totals['petri_nets']} PNML files, {totals['bpmn_models']} BPMN files, and
{totals['event_logs']} single-trace XES files. The folders contain
{totals['distinct_petri_nets']} distinct Petri nets and
{totals['distinct_event_logs']} distinct event logs; shared source objects are
repeated so each sample folder can be used independently.

{split_lines}

`samples.csv` and `manifest.json` link every source sample to its PNML, BPMN, and
XES artifacts. `SHA256SUMS` contains checksums for every exported model/log file.
All artifacts were read back and checked with PM4Py before this folder was
published.
"""


def _validate_samples(samples: Iterable[AlignmentSample], split: str) -> None:
    seen: set[str] = set()
    for sample in samples:
        if sample.split != split:
            raise ValueError(
                f"sample {sample.sample_id!r} declares split {sample.split!r}, expected {split!r}"
            )
        if sample.sample_id in seen:
            raise ValueError(f"duplicate sample_id in {split}: {sample.sample_id!r}")
        seen.add(sample.sample_id)
        for key in ("behavior_id", "variant_id", "trace_id"):
            _required_metadata(sample, key)


def _validate_pnml_round_trip(
    path: Path, net: PetriNet, initial_marking: Marking, final_marking: Marking
) -> None:
    imported_net, imported_initial, imported_final = pm4py.read_pnml(str(path))
    expected = _petri_fingerprint(net, initial_marking, final_marking)
    actual = _petri_fingerprint(imported_net, imported_initial, imported_final)
    if actual != expected:
        raise ValueError(f"PNML round-trip changed Petri-net semantics: {path}")


def _validate_bpmn_round_trip(path: Path, bpmn: Any) -> None:
    imported = pm4py.read_bpmn(str(path))
    expected_shape = (len(bpmn.get_nodes()), len(bpmn.get_flows()))
    actual_shape = (len(imported.get_nodes()), len(imported.get_flows()))
    if actual_shape != expected_shape or expected_shape[0] == 0:
        raise ValueError(
            f"BPMN round-trip shape mismatch for {path}: "
            f"expected {expected_shape}, got {actual_shape}"
        )


def _validate_xes_round_trip(
    path: Path, expected_trace: Trace, expected_trace_attributes: dict[str, Any]
) -> None:
    # Likewise, use PM4Py's object importer so an empty trace remains a trace.
    imported = xes_importer.apply(
        str(path),
        variant=xes_importer.Variants.ITERPARSE,
        parameters={"show_progress_bar": False},
    )
    if len(imported) != 1:
        raise ValueError(f"XES file must contain exactly one trace: {path}")
    if _trace_fingerprint(imported[0]) != _trace_fingerprint(expected_trace):
        raise ValueError(f"XES round-trip changed event data: {path}")
    for key in (
        "concept:name",
        "lara:split",
        "lara:behavior_id",
        "lara:trace_id",
        "lara:sample_id",
    ):
        if imported[0].attributes.get(key) != expected_trace_attributes.get(key):
            raise ValueError(f"XES round-trip changed trace attribute {key!r}: {path}")


def _petri_fingerprint(
    net: PetriNet, initial_marking: Marking, final_marking: Marking
) -> tuple[Any, ...]:
    places = tuple(sorted(str(place.name) for place in net.places))
    transitions = tuple(
        sorted((str(transition.name), transition.label) for transition in net.transitions)
    )
    arcs = tuple(
        sorted(
            (
                _petri_node_key(arc.source),
                _petri_node_key(arc.target),
                arc.weight,
            )
            for arc in net.arcs
        )
    )
    initial = tuple(sorted((str(place.name), tokens) for place, tokens in initial_marking.items()))
    final = tuple(sorted((str(place.name), tokens) for place, tokens in final_marking.items()))
    return places, transitions, arcs, initial, final


def _petri_node_key(node: Any) -> tuple[str, str]:
    node_type = "place" if isinstance(node, PetriNet.Place) else "transition"
    return node_type, str(node.name)


def _trace_fingerprint(trace: Trace) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(
        tuple(sorted((str(key), _json_scalar(value)) for key, value in event.items()))
        for event in trace
    )


def _json_scalar(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _required_metadata(sample: AlignmentSample, key: str) -> str:
    value = sample.metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"sample {sample.sample_id!r} has no valid metadata[{key!r}]")
    return value


def _artifact_stems(records: dict[str, AlignmentSample]) -> dict[str, str]:
    result: dict[str, str] = {}
    claimed: dict[str, str] = {}
    for identifier in records:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "__", identifier).strip("._-")
        if not stem:
            stem = "artifact"
        previous = claimed.get(stem)
        if previous is not None and previous != identifier:
            stem = f"{stem}__{sha256(identifier.encode('utf-8')).hexdigest()[:12]}"
        claimed[stem] = identifier
        result[identifier] = stem
    return result


def _source_file_record(path: Path, data_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(data_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_locations(data_dir: Path, output_dir: Path) -> None:
    if data_dir == output_dir or output_dir in data_dir.parents:
        raise SystemExit("output must not be the dataset directory or one of its parents")


def _validate_source_files(data_dir: Path) -> None:
    missing = [
        path
        for path in [
            *(data_dir / f"{split}.pkl" for split in SPLITS),
            data_dir / "metadata.pkl",
        ]
        if not path.is_file()
    ]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise SystemExit(f"foundation-model dataset files are missing: {formatted}")


if __name__ == "__main__":
    main()
