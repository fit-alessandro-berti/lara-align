"""Audit reviewer experiment witnesses and export publication data, without weights."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

from evaluate_revision import ROOT, OUT, DATA, inputs, read_records, decode, sha
from lara_align.verify import verify_alignment

PAPER = ROOT / "paper_applsci"


def summary(rows, method):
    legal = [r for r in rows if r[method]["legal"]]
    compared = [r for r in legal if r["exact"]["legal"]]
    optimal = [r for r in compared if r[method]["cost"] == r["exact"]["cost"]]
    return {"n": len(rows), "legal": len(legal),
            "exact_available": sum(r["exact"]["legal"] for r in rows),
            "compared": len(compared), "optimal": len(optimal),
            "optimal_percent": 100 * len(optimal)/len(compared) if compared else None,
            "mean_gap": statistics.mean(r[method]["cost"] - r["exact"]["cost"] for r in compared) if compared else None,
            "max_gap": max((r[method]["cost"] - r["exact"]["cost"] for r in compared), default=None),
            "cases": sum(r["frequency"] for r in rows),
            "compared_cases": sum(r["frequency"] for r in compared),
            "optimal_cases": sum(r["frequency"] for r in optimal),
            "case_optimal_percent": (100 * sum(r["frequency"] for r in optimal) /
                                     sum(r["frequency"] for r in compared)) if compared else None}


def paired_interval(rows, first, second, seed=13):
    families = defaultdict(lambda: defaultdict(list))
    for row in rows:
        assert row[first]["legal"] and row[second]["legal"] and row["exact"]["legal"]
        families[row["motif"]][row["family"]].append(
            int(row[first]["cost"] == row["exact"]["cost"]) -
            int(row[second]["cost"] == row["exact"]["cost"]))
    rng = np.random.default_rng(seed)
    draws = np.zeros(10000)
    difference = 0.
    assert len(families) == 4
    for motif in sorted(families):
        grouped = families[motif]
        assert len(grouped) == 32 and all(len(values) == 4 for values in grouped.values())
        values = np.array([np.mean(grouped[key]) for key in sorted(grouped)])
        indices = rng.integers(len(values), size=(10000, len(values)))
        draws += values[indices].mean(axis=1) * 100 / len(families)
        difference += values.mean() * 100 / len(families)
    return {"difference_points": float(difference),
            "ci95_points": np.quantile(draws, [.025, .975]).tolist(),
            "family_count": 128, "strata": "motif", "families_per_stratum": 32,
            "resamples": 10000, "seed": seed}


def main():
    samples = {s.sample_id: s for s in inputs()}
    reference = read_records(OUT / "reference_quality.jsonl")
    assert len(reference) == len(samples) and {r["id"] for r in reference} == set(samples)
    by_id = {r["id"]: r for r in reference}
    models = {"seed13": json.loads((OUT / "reference_checkpoint.json").read_text())}
    for name in ["seed17", "seed29", "seed17_no_log", "seed29_no_log"]:
        folder = OUT / name
        model = json.loads((folder / "evaluated_checkpoint.json").read_text())
        assert sha(folder / "best.pt") == model["sha256"]
        history = list(csv.DictReader((folder / "metrics.csv").open()))
        selected_epoch = int([r for r in history if r["improved"] == "1"][-1]["epoch"])
        assert selected_epoch == model["epoch"]
        assert model["config"] == models["seed13"]["config"]
        model["training_epochs"] = len(history)
        model["metrics_sha256"] = sha(folder / "metrics.csv")
        model["environment"] = json.loads((folder / "train.log").read_text().splitlines()[0])
        expected_log_weight = 0 if name.endswith("no_log") else 1
        assert model["args"]["log_loss_weight"] == expected_log_weight
        for key, value in models["seed13"]["args"].items():
            if key not in {"output_dir", "seed", "no_progress"}:
                assert model["args"][key] == value, (name, key)
        models[name] = model
        records = read_records(folder / "quality.jsonl")
        expected = {s.sample_id for s in samples.values() if s.group == "test" or not name.endswith("no_log")}
        assert len(records) == len(expected) and {r["id"] for r in records} == expected
        for row in records:
            by_id[row["id"]][name] = row["guided"]
        # The numeric history is publishable; no model state is copied.
        shutil.copyfile(folder / "metrics.csv", PAPER / "data" / f"revision_{name}_metrics.csv")

    checked = 0
    groups = defaultdict(list)
    for row in reference:
        sample = samples[row["id"]]
        if sample.group == "test":
            row["motif"] = sample.metadata["motif"]
        groups[row["group"]].append(row)
        for key, value in row.items():
            if not isinstance(value, dict) or "moves" not in value:
                continue
            if value["legal"]:
                replay = verify_alignment(decode(value["moves"]), sample.net,
                    sample.initial_marking, sample.final_marking, sample.trace)
                assert replay.legal and replay.cost == value["cost"], (row["id"], key)
                checked += 1
                if row["exact"]["legal"]:
                    assert value["cost"] >= row["exact"]["cost"]
        if sample.group == "test":
            assert row["exact"]["cost"] == sample.optimum
        elif sample.view == "clean":
            assert row["exact"]["legal"] and row["exact"]["cost"] == 0
    paired = defaultdict(list)
    for r in reference:
        if r["group"].startswith(("acyclic/", "loop/")):
            paired[(r["family"], r["view"])].append(r)
    for pair in paired.values():
        assert len(pair) == 2
        if all(r["exact"]["legal"] for r in pair):
            assert pair[0]["exact"]["cost"] == pair[1]["exact"]["cost"]

    methods = ["guided", "unguided", "repair_guided", "repair_unguided", "seed17", "seed29"]
    stats = {name: {method: summary(rows, method) for method in methods}
             for name, rows in groups.items()}
    for name in ["seed17_no_log", "seed29_no_log"]:
        stats["test"][name] = summary(groups["test"], name)
    seed_percent = [stats["test"][name]["optimal_percent"] for name in ["guided", "seed17", "seed29"]]
    seed_gap = [stats["test"][name]["mean_gap"] for name in ["guided", "seed17", "seed29"]]

    timing = read_records(OUT / "common_timing.jsonl")
    timing_groups = defaultdict(list)
    for row in timing:
        assert row["id"] in samples and samples[row["id"]].group == "test"
        timing_groups[row["method"]].append(row)
    timing_summary = {}
    assert len(timing_groups) == 8 and len(timing) == 8*512*5
    for name, rows in timing_groups.items():
        assert len({(r["id"], r["repeat"]) for r in rows}) == 512*5
        per_input = defaultdict(list)
        for row in rows:
            per_input[row["id"]].append(row["seconds"]*1000)
        medians = [statistics.median(values) for values in per_input.values()]
        legal = [r for r in rows if r["legal"]]
        timing_summary[name] = {"calls": len(rows), "legal_calls": len(legal),
            "optimal_calls": sum(r["cost"] == samples[r["id"]].optimum for r in legal),
            "median_ms": statistics.median(medians), "iqr_ms": np.quantile(medians, [.25,.75]).tolist(),
            "repeat_medians_ms": [statistics.median(r["seconds"]*1000 for r in rows if r["repeat"] == i) for i in range(5)]}
    result = {"groups": stats, "models": models,
              "real_guidance_cost_equality": {
                  group: {method: all(r[method]["cost"] == r["unguided"]["cost"] for r in rows)
                          for method in ["guided", "seed17", "seed29"]}
                  for group, rows in groups.items() if group.startswith("real/")},
              "seed_summary": {"optimal_percent_mean": statistics.mean(seed_percent),
                               "optimal_percent_sample_sd": statistics.stdev(seed_percent),
                               "mean_gap_mean": statistics.mean(seed_gap),
                               "mean_gap_sample_sd": statistics.stdev(seed_gap)},
              "paired_intervals": {name: paired_interval(groups["test"], name, "unguided")
                                   for name in ["guided", "seed17", "seed29", "repair_guided"]},
              "repair_gain": paired_interval(groups["test"], "repair_guided", "guided"),
              "loss_ablation": {name: paired_interval(groups["test"], name, name+"_no_log")
                                for name in ["seed17", "seed29"]},
              "timing": timing_summary,
              "timing_protocol": json.loads((OUT / "common_timing_protocol.json").read_text()),
              "input_manifest": json.loads((DATA / "inputs_manifest.json").read_text()),
              "replayed_witnesses": checked, "paired_compiler_checks": len(paired),
              "record_count": len(reference),
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                                [OUT / "reference_quality.jsonl", OUT / "common_timing.jsonl",
                                 DATA / "inputs.pkl.gz", DATA / "inputs_manifest.json",
                                 ROOT / "lara_align/decode.py", ROOT / "lara_align/training.py",
                                 ROOT / "scripts/train_model.py", ROOT / "lara_align/verify.py",
                                 ROOT / "lara_align/exact.py", ROOT / "lara_align/certifier.py",
                                 PAPER / "scripts/evaluate_revision.py",
                                 PAPER / "scripts/prepare_revision_inputs.py",
                                 PAPER / "scripts/train_revision.py"]}}
    (PAPER / "data/reviewer_experiments.json").write_text(json.dumps(result, indent=2) + "\n")
    with (PAPER / "data/reviewer_experiment_records.json.gz").open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as zipped:
            zipped.write(json.dumps({"quality": reference, "timing": timing}).encode())
    shutil.copyfile(DATA / "inputs.pkl.gz", PAPER / "data/reviewer_inputs.pkl.gz")
    (PAPER / "data/sepsis_source.json").write_text((DATA / "sepsis-source.json").read_text())
    print(json.dumps({"groups": stats, "seed_summary": result["seed_summary"],
                      "timing": timing_summary, "replayed_witnesses": checked}, indent=2))


if __name__ == "__main__":
    main()
