"""Materialize reviewer-study inputs before evaluating any trained checkpoint.

Only activity sequences are retained from the public logs. The 70/30 case split
is fixed before model discovery; all held-out variants are evaluated. Synthetic
trees use two compilers and an explicitly cyclic structural class absent from
the training corpus. Graph-isomorphic training structures are excluded.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
from random import Random
import sys
from types import SimpleNamespace

os.environ.setdefault("PM4PY_SHOW_PROGRESS_BAR", "False")
sys._pm4py_welcome_shown = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import networkx as nx
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.process_tree.obj import Operator, ProcessTree
from pm4py.objects.conversion.process_tree import converter
from lara_align.data import load_split
from lara_align.synthetic import (_BlockNode, _random_block_tree, _playout_block,
                                  inject_deviations, make_block_structured_net,
                                  trace_from_labels)

OUT = ROOT / "data/reviewer_20260929"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph(net, initial, final):
    result = nx.DiGraph()
    for p in net.places:
        result.add_node(p, color=f"p:{initial.get(p, 0)}:{final.get(p, 0)}")
    for t in net.transitions:
        result.add_node(t, color="silent" if t.label is None else "visible")
    for arc in net.arcs:
        result.add_edge(arc.source, arc.target, weight=int(arc.weight))
    return result


def graph_hash(g):
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="color", edge_attr="weight")


def as_process_tree(block, parent=None):
    ops = {"seq": Operator.SEQUENCE, "xor": Operator.XOR,
           "and": Operator.PARALLEL, "loop": Operator.LOOP}
    result = ProcessTree(operator=ops.get(block.op), parent=parent, label=block.label)
    result.children = [as_process_tree(child, result) for child in block.children]
    return result


def net_stats(net, initial, final):
    counts = Counter(t.label for t in net.transitions if t.label is not None)
    g = graph(net, initial, final)
    return {"places": len(net.places), "transitions": len(net.transitions),
            "visible": sum(counts.values()), "labels": len(counts),
            "duplicate_label_groups": sum(n > 1 for n in counts.values()),
            "cyclic": not nx.is_directed_acyclic_graph(g), "structure_hash": graph_hash(g)}


def main():
    target = OUT / "inputs.pkl.gz"
    if target.exists():
        raise SystemExit(f"Inputs already materialized at {target}; preserve them for evaluation.")
    OUT.mkdir(parents=True, exist_ok=True)
    samples = load_split(ROOT / "data/lara_synthetic", "train")
    training_graphs = defaultdict(list)
    seen_nets = set()
    for sample in samples:
        if id(sample.net) in seen_nets:
            continue
        seen_nets.add(id(sample.net))
        g = graph(sample.net, sample.initial_marking, sample.final_marking)
        assert nx.is_directed_acyclic_graph(g), "The loop holdout requires acyclic training nets"
        training_graphs[graph_hash(g)].append(g)

    def seen_training_structure(net_tuple):
        g = graph(*net_tuple)
        return any(nx.is_isomorphic(g, other,
                   node_match=nx.algorithms.isomorphism.categorical_node_match("color", ""),
                   edge_match=nx.algorithms.isomorphism.categorical_edge_match("weight", 1))
                   for other in training_graphs[graph_hash(g)])

    records = []
    manifest = {"design": {"structure_seed": 703, "case_split_seed": 37,
                           "trees_per_class": 64, "observations_per_tree": 2,
                           "visible_sizes": [6, 10, 14, 18], "noise_thresholds": [0., .5],
                           "discovery_fraction": .7, "synthetic_deviation_rate": .25,
                           "loop_redo_probability": .5, "max_loop_redos": 2},
                "software": {"pm4py": pm4py.__version__, "networkx": nx.__version__},
                "training_nets_checked_acyclic": len(seen_nets),
                "training_sha256": sha(ROOT / "data/lara_synthetic/train.pkl"),
                "logs": {}, "structural_classes": {}}
    rng = Random(703)
    for class_name in ["acyclic", "loop"]:
        rejected = 0
        for i in range(64):
            size = [6, 10, 14, 18][i % 4]
            alphabet = [f"revision_activity_{j}" for j in range(max(2, round(size*.7)))]
            while True:
                tree = _random_block_tree(rng, size - (1 if class_name == "loop" else 0),
                                          alphabet, {"seq": .4, "xor": .3, "and": .3})
                if class_name == "loop":
                    tree = _BlockNode("loop", children=[tree, _BlockNode("leaf", rng.choice(alphabet))])
                nets = {"custom": make_block_structured_net(tree, f"revision_{class_name}_{i}"),
                        "pm4py": converter.apply(as_process_tree(tree))}
                if not any(seen_training_structure(value) for value in nets.values()):
                    break
                rejected += 1
                assert rejected < 10000
            clean = _playout_block(tree, rng, .5, 2)
            noisy = inject_deviations(clean, rng, .25, alphabet)
            assert len(clean) <= 200 and len(noisy) <= 200
            for compiler, (net, initial, final) in nets.items():
                assert net_stats(net, initial, final)["cyclic"] == (class_name == "loop")
                for view, labels in [("clean", clean), ("noisy", noisy)]:
                    records.append(SimpleNamespace(
                        sample_id=f"{class_name}-{i:03d}-{compiler}-{view}",
                        group=f"{class_name}/{compiler}", family=f"{class_name}-{i:03d}",
                        view=view, net=net, initial_marking=initial, final_marking=final,
                        trace=trace_from_labels(labels), frequency=1, optimum=None,
                        metadata={"size": size, "compiler": compiler, "class": class_name,
                                  "net": net_stats(net, initial, final)},
                    ))
        manifest["structural_classes"][class_name] = {"trees": 64, "rows": 256,
                                                      "rejected_training_isomorphs": rejected}
        print(f"Prepared {class_name}: 64 paired trees, rejected {rejected} training isomorphs", flush=True)

    for name, path in [("roadtraffic", ROOT / "files/roadtraffic100traces.xes"),
                       ("receipt", ROOT / "files/receipt.xes"),
                       ("sepsis", OUT / "sepsis.xes.gz")]:
        original = xes_importer.apply(str(path), parameters={"show_progress_bar": False})
        log = EventLog([trace_from_labels([str(e["concept:name"]) for e in case]) for case in original])
        indices = list(range(len(log)))
        Random(37).shuffle(indices)
        cut = int(.7 * len(log))
        discovery_indices, holdout_indices = sorted(indices[:cut]), sorted(indices[cut:])
        assert not set(discovery_indices) & set(holdout_indices)
        assert set(discovery_indices) | set(holdout_indices) == set(range(len(log)))
        discovery = EventLog([log[i] for i in discovery_indices])
        counts = Counter(tuple(e["concept:name"] for e in log[i]) for i in holdout_indices)
        manifest["logs"][name] = {"source": str(path.relative_to(ROOT)), "sha256": sha(path),
                                  "cases": len(log), "events": sum(map(len, log)),
                                  "discovery_indices": discovery_indices,
                                  "holdout_indices": holdout_indices,
                                  "holdout_variants": len(counts), "nets": {}}
        for noise in [0., .5]:
            net, initial, final = pm4py.discover_petri_net_inductive(discovery, noise_threshold=noise)
            group = f"real/{name}/{noise:.1f}"
            manifest["logs"][name]["nets"][str(noise)] = net_stats(net, initial, final)
            for i, (labels, count) in enumerate(sorted(counts.items())):
                records.append(SimpleNamespace(
                    sample_id=f"{name}-{noise:.1f}-{i:04d}", group=group,
                    family=f"{name}-{i:04d}", view="held_out", net=net,
                    initial_marking=initial, final_marking=final,
                    trace=trace_from_labels(labels), frequency=count, optimum=None,
                    metadata={"log": name, "noise": noise, "net": net_stats(net, initial, final)},
                ))
        print(f"Prepared {name}: {len(discovery_indices)} discovery cases, {len(holdout_indices)} held out, {len(counts)} variants", flush=True)
    with gzip.open(target, "wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest["inputs_sha256"] = sha(target)
    manifest["rows"] = len(records)
    (OUT / "inputs_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(records), "sha256": manifest["inputs_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
