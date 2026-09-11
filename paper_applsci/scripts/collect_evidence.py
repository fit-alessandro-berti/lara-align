"""Check manuscript examples and snapshot the existing experiment evidence.

Run from any directory with the repository's Python dependencies installed.
This replays the fixed checkpoint on the existing test split; it does not train
a model or replace historical timing measurements.
"""
from pathlib import Path
import collections
import hashlib
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_applsci"
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from lara_align.checkpoint import load_checkpoint
from lara_align.certifier import CertifyingAlignmentSystem
from lara_align.data import load_split
from lara_align.exact import Pm4PyExactAligner
from lara_align.features import pm4py_to_features
from lara_align.training import targets_from_alignment
from lara_align.verify import verify_alignment
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to


def read(name):
    return json.loads((ROOT / "runs/lara" / name).read_text())


def main():
    torch.set_num_threads(10)
    torch.set_num_interop_threads(10)
    model, checkpoint = load_checkpoint(ROOT / "runs/lara/best.pt")
    model.eval()
    systems = {name: CertifyingAlignmentSystem(model=model, use_guidance=guided)
               for name, guided in [("guided", True), ("unguided", False)]}
    splits = {name: load_split(ROOT / "data/lara_synthetic", name)
              for name in ("train", "val", "test")}
    corpus_checks = {}
    seen_families, seen_seeds = set(), set()
    for split, samples in splits.items():
        families = collections.defaultdict(list)
        for sample in samples:
            assert sample.split == split
            assert sample.metadata["equivalence_level"] == "exact"
            assert sample.metadata["equivalence_certificate"]["status"] == "exact"
            witness = verify_alignment(sample.optimal_alignment, sample.net,
                                       sample.initial_marking, sample.final_marking,
                                       sample.trace)
            assert witness.legal and witness.cost == sample.optimal_cost
            families[sample.metadata["behavior_id"]].append(sample)
        assert not seen_families.intersection(families)
        seen_families.update(families)
        seeds = {rows[0].metadata["family_seed"] for rows in families.values()}
        assert len(seeds) == len(families) and not seen_seeds.intersection(seeds)
        seen_seeds.update(seeds)
        motif_counts = collections.Counter()
        for rows in families.values():
            assert len(rows) == 4
            assert len({r.metadata["representation_kind"] for r in rows}) == 2
            observations = collections.defaultdict(list)
            for row in rows:
                observations[row.metadata["trace_id"]].append(row)
            assert len(observations) == 2
            for pair in observations.values():
                assert len(pair) == 2
                assert {r.metadata["representation_slot"] for r in pair} == {0, 1}
                assert len({tuple(e["concept:name"] for e in r.trace) for r in pair}) == 1
                assert len({r.optimal_cost for r in pair}) == 1
            motif_counts[rows[0].metadata["motif"]] += 1
        assert len(motif_counts) == 4 and len(set(motif_counts.values())) == 1
        corpus_checks[split] = {"rows": len(samples), "families": len(families),
                               "families_per_motif": dict(motif_counts),
                               "stored_exact_language_certificates": len(samples),
                               "replayed_teacher_witnesses": len(samples)}
    assert {k: v["rows"] for k, v in corpus_checks.items()} == {"train": 2048, "val": 512, "test": 512}
    samples = splits["test"]
    exact = Pm4PyExactAligner()
    records = []
    for sample in samples:
        result = exact.align_trace(sample.net, sample.initial_marking,
                                   sample.final_marking, sample.trace, timeout_seconds=30)
        assert result.optimal and result.cost == sample.optimal_cost
        row = {"id": sample.sample_id, "family": sample.metadata["behavior_id"],
               "motif": sample.metadata["motif"],
               "representation": sample.metadata["representation_kind"],
               "trace_id": sample.metadata["trace_id"], "optimum": sample.optimal_cost}
        for name, system in systems.items():
            result = system.align(sample.net, sample.initial_marking,
                                  sample.final_marking, sample.trace, mode="fast")
            assert result.legal
            row[name] = result.cost
        records.append(row)
    for name in systems:
        saved = read(f"test_{name}.json")["metrics"]
        count = sum(r[name] == r["optimum"] for r in records)
        gap = statistics.mean(r[name]-r["optimum"] for r in records)
        assert count == saved["optimal_cost_count"], (name, count)
        assert gap == saved["gap_mean"]

    groups = collections.defaultdict(list)
    for row in records:
        groups[(row["motif"], row["representation"])].append(row)
    classes = []
    for (motif, rep), rows in sorted(groups.items()):
        classes.append({"motif": motif, "representation": rep, "n": len(rows),
                        **{name: {"optimal": sum(r[name]==r["optimum"] for r in rows),
                                  "gap_mean": statistics.mean(r[name]-r["optimum"] for r in rows)}
                           for name in systems}})
    rng = np.random.default_rng(13)
    for cell in classes:
        families = collections.defaultdict(list)
        for row in groups[(cell["motif"], cell["representation"])]:
            families[row["family"]].append(int(row["guided"] == row["optimum"])
                                           - int(row["unguided"] == row["optimum"]))
        differences = np.array([np.mean(v) for v in families.values()])
        draws = rng.choice(differences, (10000, len(differences))).mean(axis=1)*100
        cell["delta_ci95"] = np.quantile(draws, [.025, .975]).tolist()

    # Check every component of the worked stored input and its supervision.
    sample = next(s for s in splits["train"]
                  if s.sample_id == "train-000048")
    feat = pm4py_to_features(sample.net, sample.initial_marking, sample.final_marking, sample.trace)
    targets = targets_from_alignment(sample.optimal_alignment, feat, sample.optimal_cost)
    assert feat.event_label_ids.tolist() == [8075, 5878, 4475, 2587]
    assert feat.transition_label_ids.tolist() == [2587, 4475, 8075, 8075, 6408, 5878]
    assert targets.sync_transition_targets.tolist() == [3, 5, -1, 0]
    assert targets.log_move_targets.tolist() == [0, 0, 1, 0]
    assert targets.model_move_targets.tolist() == [0, 1, 0, 0, 0, 0]
    assert sample.optimal_cost == 2
    assert feat.place_features.tolist() == [[0,0,1,1,2],[0,1,1,0,1],[1,0,0,2,2],
                                          [0,0,2,1,3],[0,0,1,1,2],[0,0,1,1,2]]
    assert feat.transition_features.tolist() == [[0,1,1,1,0,0]]*6
    assert feat.compatibility.int().tolist() == [[0,0,1,1,0,0],[0,0,0,0,0,1],
                                               [0,1,0,0,0,0],[1,0,0,0,0,0]]
    actual_edges = set(zip(feat.edge_index[0].tolist(), feat.edge_index[1].tolist(), feat.edge_type.tolist()))
    expected_edges = set()
    for p,t,q in [(3,6,0),(0,7,1),(2,8,4),(2,9,5),(4,10,3),(5,11,3)]:
        expected_edges.update([(p,t,0),(t,p,2),(t,q,1),(q,t,3)])
    assert actual_edges == expected_edges

    # Independently instantiate the net in the running-example drawing.
    net = PetriNet("running-example")
    ps = {i: PetriNet.Place(f"p{i}") for i in range(6)}
    net.places.update(ps.values())
    ts = {}
    for name,label,a,b in [("t_a","a",0,1),("t_1","b",1,2),("t_2","b",1,3),
                           ("t_c","c",2,4),("t_d","d",3,4),("t_tau",None,4,5)]:
        t = PetriNet.Transition(name,label);net.transitions.add(t);ts[name]=t
        add_arc_from_to(ps[a],t,net);add_arc_from_to(t,ps[b],net)
    features = pm4py_to_features(net,Marking({ps[0]:1}),Marking({ps[5]:1}),
                                [{"concept:name": a} for a in ["a","b","c"]])
    with torch.inference_mode():
        out = model(features)
    scores = {"transition_order": features.transition_names,
              "sync": out.sync_logits.tolist(), "model": out.model_move_logits.tolist(),
              "log": out.log_move_logits.tolist()}
    assert [round(v,3) for v in scores["model"]] == [-1.357,-1.401,-1.030,-1.830,-1.837,2.519]
    assert [round(v,3) for v in scores["log"]] == [-1.682,-1.618,-1.527]
    for i,j,value in [(0,2,.353),(1,0,.067),(1,1,.039),(2,3,.876)]:
        assert round(scores["sync"][i][j],3) == value

    names = ["test_guided.json","test_unguided.json","test_analysis_summary.json",
             "scaling_guided.json","scaling_unguided.json","pm4py_approx_comparison.json",
             "pm4py_approx_scaling_comparison.json","real_receipt_noise0.json",
             "real_receipt_noise05.json","real_roadtraffic_noise0.json","real_roadtraffic_noise05.json"]
    snapshots = {}
    for name in names:
        d = read(name)
        if name == "pm4py_approx_comparison.json":
            d = {k:v for k,v in d.items() if k != "records"}
        snapshots[name] = d
    inputs = {"original_results": snapshots, "checked_classes": classes,
              "corpus_checks": corpus_checks, "recomputed_test_optima": len(records),
              "bootstrap": {"seed":13,"resamples":10000,"unit":"behavior family","method":"percentile"},
              "checked_test_records": records, "running_example_scores": scores,
              "checkpoint_config": checkpoint["model_config"],
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "source_sha256": {name:hashlib.sha256((ROOT/"runs/lara"/name).read_bytes()).hexdigest()
                                for name in names + ["best.pt","metrics.csv"]}}
    (PAPER/"data/evidence.json").write_text(json.dumps(inputs,indent=2)+"\n")
    (PAPER/"data/metrics.csv").write_text((ROOT/"runs/lara/metrics.csv").read_text())
    print(json.dumps({"checked_test_rows":len(records),"corpus_checks":corpus_checks,"classes":classes,
                      "parameter_count":inputs["parameter_count"],"running_example":scores},indent=2))


if __name__ == "__main__":
    main()
