from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import pickle
from typing import Any, Iterable

from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import Marking, PetriNet

from lara_align.types import Alignment

SPLITS = ("train", "val", "test")
DATASET_VERSION = 2


@dataclass
class AlignmentSample:
    sample_id: str
    split: str
    net: PetriNet
    initial_marking: Marking
    final_marking: Marking
    trace: Trace
    optimal_alignment: Alignment
    optimal_cost: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetMetadata:
    version: int
    split_counts: dict[str, int]
    seed: int
    generator: str
    parameters: dict[str, Any] = field(default_factory=dict)


def save_split(data_dir: str | Path, split: str, samples: Iterable[AlignmentSample]) -> Path:
    _validate_split(split)
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    split_path = path / f"{split}.pkl"
    with split_path.open("wb") as handle:
        pickle.dump(list(samples), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return split_path


def load_split(data_dir: str | Path, split: str) -> list[AlignmentSample]:
    _validate_split(split)
    split_path = Path(data_dir) / f"{split}.pkl"
    with split_path.open("rb") as handle:
        samples = pickle.load(handle)
    return list(samples)


def save_metadata(data_dir: str | Path, metadata: DatasetMetadata) -> Path:
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / "metadata.pkl"
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    json_path = path / "metadata.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(metadata), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata_path


def load_metadata(data_dir: str | Path) -> DatasetMetadata:
    metadata_path = Path(data_dir) / "metadata.pkl"
    with metadata_path.open("rb") as handle:
        return pickle.load(handle)


def split_counts(data_dir: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        split_path = Path(data_dir) / f"{split}.pkl"
        counts[split] = len(load_split(data_dir, split)) if split_path.exists() else 0
    return counts


def _validate_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
