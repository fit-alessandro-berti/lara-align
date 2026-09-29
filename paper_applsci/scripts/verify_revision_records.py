"""Replay the packaged reviewer-study witnesses without loading any checkpoint.

Run with the parent repository's dependencies. The original test inputs are
read directly from the bundled reference corpus; no training run is required.
The stored exact witnesses prove feasibility, not optimality by themselves.
Optimality comes from the completed exact searches recorded by the experiment.
"""
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_applsci"
sys.path.insert(0, str(ROOT))
from lara_align.types import Alignment, AlignmentMove
from lara_align.verify import verify_alignment


def main():
    summary = json.loads((PAPER / "data/reviewer_experiments.json").read_text())
    with gzip.open(PAPER / "data/reviewer_experiment_records.json.gz", "rt") as handle:
        records = json.load(handle)
    novel_path = PAPER / "data/reviewer_inputs.pkl.gz"
    assert hashlib.sha256(novel_path.read_bytes()).hexdigest() == summary["input_manifest"]["inputs_sha256"]
    with gzip.open(novel_path, "rb") as handle:
        samples = {s.sample_id: s for s in pickle.load(handle)}
    with tarfile.open(PAPER / "data/reference_corpus.tar.gz") as archive:
        matches = [m for m in archive.getmembers() if m.name.endswith("/test.pkl")]
        assert len(matches) == 1
        raw = archive.extractfile(matches[0]).read()
        # load_split stores a plain list of AlignmentSample objects.
        original_test = pickle.load(io.BytesIO(raw))
    samples.update({s.sample_id: s for s in original_test})
    assert len(records["quality"]) == len(samples) == 1676
    assert {r["id"] for r in records["quality"]} == set(samples)
    checked = 0
    grouped = defaultdict(list)
    for row in records["quality"]:
        sample = samples[row["id"]]
        grouped[row["group"]].append(row)
        for name, result in row.items():
            if not isinstance(result, dict) or "moves" not in result or not result["legal"]:
                continue
            alignment = Alignment([AlignmentMove(*move) for move in result["moves"]])
            replay = verify_alignment(alignment, sample.net, sample.initial_marking,
                                      sample.final_marking, sample.trace)
            assert replay.legal and replay.cost == result["cost"], (row["id"], name)
            if row["exact"]["legal"]:
                assert replay.cost >= row["exact"]["cost"]
            checked += 1
        if row["group"] == "test":
            assert row["exact"]["cost"] == sample.optimal_cost
        for base in ["guided", "unguided"]:
            if row[base]["legal"] and row["repair_" + base]["legal"]:
                assert row["repair_" + base]["cost"] <= row[base]["cost"]
    assert checked == summary["replayed_witnesses"]
    for group, rows in grouped.items():
        for method, counts in summary["groups"][group].items():
            legal = [r for r in rows if r[method]["legal"]]
            comparable = [r for r in legal if r["exact"]["legal"]]
            optimal = [r for r in comparable if r[method]["cost"] == r["exact"]["cost"]]
            assert counts["n"] == len(rows) and counts["legal"] == len(legal)
            assert counts["compared"] == len(comparable) and counts["optimal"] == len(optimal)
            assert counts["optimal_cases"] == sum(r["frequency"] for r in optimal)
    for name in ["seed17", "seed29", "seed17_no_log", "seed29_no_log"]:
        path = PAPER / "data" / f"revision_{name}_metrics.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == summary["models"][name]["metrics_sha256"]
    timings = records["timing"]
    assert len(timings) == len({(r["id"], r["method"], r["repeat"]) for r in timings}) == 20480
    for row in timings:
        assert row["seconds"] >= 0 and row["repeat"] in range(5)
        assert row["method"] in summary["timing"]
        if row["legal"]:
            assert row["cost"] >= samples[row["id"]].optimal_cost
    print(json.dumps({"inputs": len(samples), "replayed_witnesses": checked,
                      "timed_calls": len(timings), "checkpoint_weights_loaded": 0}))


if __name__ == "__main__":
    main()
