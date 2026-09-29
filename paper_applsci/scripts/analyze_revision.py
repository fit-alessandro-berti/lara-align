"""Summarize fixed-checkpoint uncertainty and the existing real-log records.

Run after collect_evidence.py to use freshly verified synthetic candidate costs.
This analysis neither retrains the scorer nor measures new runtimes.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


PAPER = Path(__file__).resolve().parents[1]


def main():
    source = PAPER / "data/evidence.json"
    evidence = json.loads(source.read_text())
    records = evidence["checked_test_records"]
    families = defaultdict(lambda: defaultdict(list))
    for row in records:
        families[row["motif"]][row["family"]].append(row)
    assert len(records) == 512 and len(families) == 4

    # Resample whole paired families within each motif, preserving the corpus
    # balance and the dependence between the two observations and two nets.
    rng = np.random.default_rng(13)
    resamples = 10000
    draws = np.zeros((resamples, 3))
    totals = np.zeros(3)
    for motif in sorted(families):
        grouped = families[motif]
        assert len(grouped) == 32
        values = []
        for family in sorted(grouped):
            rows = grouped[family]
            assert len(rows) == 4
            guided = np.mean([r["guided"] == r["optimum"] for r in rows])
            unguided = np.mean([r["unguided"] == r["optimum"] for r in rows])
            gap_reduction = np.mean([r["unguided"] - r["guided"] for r in rows])
            values.append([guided, unguided, gap_reduction])
        values = np.array(values)
        indices = rng.integers(len(values), size=(resamples, len(values)))
        draws += values[indices].mean(axis=1) / len(families)
        totals += values.mean(axis=0) / len(families)

    def interval(values):
        return np.quantile(values, [0.025, 0.975]).tolist()

    assert totals[0] * 512 == 466 and totals[1] * 512 == 445
    summary = {
        "source_evidence_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checkpoint_sha256": evidence["source_sha256"]["best.pt"],
        "scope": "Fixed checkpoint and generator; no retraining or domain uncertainty",
        "bootstrap": {"seed": 13, "resamples": resamples, "method": "percentile",
                      "unit": "whole paired behavior family", "strata": "motif",
                      "families_per_motif": 32, "rows_per_family": 4},
        "guided_optimal_percent": float(totals[0] * 100),
        "unguided_optimal_percent": float(totals[1] * 100),
        "guidance_gain_points": float((totals[0] - totals[1]) * 100),
        "guidance_gain_ci95_points": interval((draws[:, 0] - draws[:, 1]) * 100),
        "mean_gap_reduction": float(totals[2]),
        "mean_gap_reduction_ci95": interval(draws[:, 2]),
        "real_logs": {},
    }
    for name, result in sorted(evidence["original_results"].items()):
        if not name.startswith("real_"):
            continue
        rows = result["records"]
        assert all(r["guided_legal"] and r["unguided_legal"] and r["exact_solved"]
                   for r in rows)
        assert all(r["guided_cost"] == r["unguided_cost"] for r in rows)
        summary["real_logs"][name] = {
            "variants": len(rows), "cases": sum(r["trace_count"] for r in rows),
            "optimal_variants": sum(r["guided_gap"] == 0 for r in rows),
            "optimal_cases": sum(r["trace_count"] for r in rows if r["guided_gap"] == 0),
            "guided_unguided_costs_identical": True,
        }
    target = PAPER / "data/revision_analysis.json"
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
