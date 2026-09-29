"""Run reviewer experiments with resumable per-input records and explicit timers.

Stages: reference (exact references and the public checkpoint), checkpoints
(two new seeds and paired log-loss ablations), timing (common end to end calls).
Run timing only after training has completed, without concurrent experiment jobs.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
from random import Random
import signal
import sys
from time import perf_counter
from types import SimpleNamespace

os.environ.setdefault("PM4PY_SHOW_PROGRESS_BAR", "False")
sys._pm4py_welcome_shown = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pm4py
import torch
from pm4py.algo.conformance.alignments.petri_net import algorithm as pn_alignments
from lara_align.checkpoint import load_checkpoint
from lara_align.certifier import CertifyingAlignmentSystem
from lara_align.data import load_split
from lara_align.decode import GreedyCandidateDecoder, SingleLogRepairDecoder
from lara_align.exact import Pm4PyExactAligner, parse_pm4py_alignment
from lara_align.types import Alignment, AlignmentMove, CostModel
from lara_align.verify import verify_alignment
from scripts.benchmark_pm4py_approx import PETRI_VARIANTS, _petri_parameters

OUT = ROOT / "runs/reviewer_20260929"
DATA = ROOT / "data/reviewer_20260929"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def deadline(seconds):
    def expired(*_):
        raise TimeoutError(f"Wall time budget {seconds}s exceeded")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def checkpoint(path):
    model, state = load_checkpoint(path)
    model.eval()
    return model, {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                   "epoch": state["epoch"], "config": state["model_config"],
                   "args": state["metrics"]["args"]}


def systems(model):
    return {name: CertifyingAlignmentSystem(
        model=model, use_guidance=guided,
        decoder=(SingleLogRepairDecoder(max_final_model_depth=64)
                 if repair else GreedyCandidateDecoder(max_final_model_depth=64)))
        for name, guided, repair in [("guided", True, False), ("unguided", False, False),
                                     ("repair_guided", True, True), ("repair_unguided", False, True)]}


def inputs():
    with gzip.open(DATA / "inputs.pkl.gz", "rb") as handle:
        novel = pickle.load(handle)
    test = []
    for sample in load_split(ROOT / "data/lara_synthetic", "test"):
        test.append(SimpleNamespace(
            sample_id=sample.sample_id, group="test", family=sample.metadata["behavior_id"],
            view=sample.metadata["representation_kind"], frequency=1,
            net=sample.net, initial_marking=sample.initial_marking, final_marking=sample.final_marking,
            trace=sample.trace, optimum=sample.optimal_cost, metadata=sample.metadata,
        ))
    return test + novel


def base_row(sample):
    return {"id": sample.sample_id, "group": sample.group, "family": sample.family,
            "view": sample.view, "frequency": sample.frequency, "trace_length": len(sample.trace)}


def encode(alignment):
    return [[m.log_label, m.transition_name, m.transition_label] for m in alignment.moves]


def decode(moves):
    return Alignment([AlignmentMove(*m) for m in moves])


def read_records(path):
    if not path.exists():
        return []
    # A truncated last line is an interrupted write, never a completed result.
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"Incomplete last record in {path}; inspect before resuming")
    return [json.loads(line) for line in raw.splitlines() if line]


def record_alignment(alignment, sample):
    verification = verify_alignment(alignment, sample.net, sample.initial_marking,
                                    sample.final_marking, sample.trace, CostModel())
    return {"legal": verification.legal, "cost": verification.cost if verification.legal else None,
            "moves": encode(alignment), "error": verification.reason}


def exact_call(sample, timeout=30.):
    try:
        with deadline(timeout + 1):
            result = Pm4PyExactAligner().align_trace(
                sample.net, sample.initial_marking, sample.final_marking, sample.trace,
                timeout_seconds=timeout)
        if not result.optimal or result.alignment is None:
            return {"legal": False, "cost": None, "moves": [], "error": result.error or "exact timeout/failure"}
        return record_alignment(result.alignment, sample)
    except Exception as exc:
        return {"legal": False, "cost": None, "moves": [], "error": f"{type(exc).__name__}: {exc}"}


def candidate_call(system, sample, timeout=5.):
    try:
        with deadline(timeout):
            result = system.align(sample.net, sample.initial_marking, sample.final_marking,
                                  sample.trace, mode="fast")
        if result.alignment is None:
            return {"legal": False, "cost": None, "moves": [], "error": str(result.diagnostics)}
        return record_alignment(result.alignment, sample)
    except Exception as exc:
        return {"legal": False, "cost": None, "moves": [], "error": f"{type(exc).__name__}: {exc}"}


def run_reference(samples):
    path = OUT / "reference_quality.jsonl"
    done = {r["id"] for r in read_records(path)}
    model, meta = checkpoint(ROOT / "runs/lara/best.pt")
    (OUT / "reference_checkpoint.json").write_text(json.dumps(meta, indent=2) + "\n")
    runners = systems(model)
    # Preserve the original synthetic test completion depth for that comparison.
    with path.open("a", buffering=1) as handle:
        for index, sample in enumerate(samples):
            if sample.sample_id in done:
                continue
            for runner in runners.values():
                runner.decoder.max_final_model_depth = 32 if sample.group == "test" else 64
            row = base_row(sample)
            row["exact"] = exact_call(sample)
            if sample.optimum is not None:
                assert row["exact"]["cost"] == sample.optimum and row["exact"]["legal"]
            for name, runner in runners.items():
                row[name] = candidate_call(runner, sample)
                if row[name]["legal"] and row["exact"]["legal"]:
                    assert row[name]["cost"] >= row["exact"]["cost"]
            for guided in ["guided", "unguided"]:
                if row[guided]["legal"] and row["repair_" + guided]["legal"]:
                    assert row["repair_" + guided]["cost"] <= row[guided]["cost"]
            handle.write(json.dumps(row) + "\n")
            if index % 50 == 0:
                print(f"reference {index+1}/{len(samples)}: {sample.sample_id}", flush=True)


def run_checkpoints(samples):
    for name in ["seed17", "seed29", "seed17_no_log", "seed29_no_log"]:
        folder = OUT / name
        # Completion is proved by the training process result and its final log,
        # not by the mere existence of a best checkpoint during a live run.
        assert "best checkpoint:" in (folder / "train.log").read_text(), name
        model, meta = checkpoint(folder / "best.pt")
        (folder / "evaluated_checkpoint.json").write_text(json.dumps(meta, indent=2) + "\n")
        path = folder / "quality.jsonl"
        done = {r["id"] for r in read_records(path)}
        runner = systems(model)["guided"]
        selected = [s for s in samples if s.group == "test" or not name.endswith("no_log")]
        with path.open("a", buffering=1) as handle:
            for i, sample in enumerate(selected):
                if sample.sample_id in done:
                    continue
                runner.decoder.max_final_model_depth = 32 if sample.group == "test" else 64
                row = base_row(sample)
                row["guided"] = candidate_call(runner, sample)
                handle.write(json.dumps(row) + "\n")
                if i % 100 == 0:
                    print(f"{name} {i+1}/{len(selected)}", flush=True)


def timed_call(name, sample, runners, timeout=5.):
    # All timers start from the same in-memory net and trace. They include
    # method-specific preparation, candidate production, parsing, and replay.
    start = perf_counter()
    try:
        with deadline(timeout):
            if name in runners:
                result = runners[name].align(sample.net, sample.initial_marking,
                                              sample.final_marking, sample.trace, mode="fast")
                alignment = result.alignment
            elif name == "exact":
                result = Pm4PyExactAligner().align_trace(
                    sample.net, sample.initial_marking, sample.final_marking,
                    sample.trace, timeout_seconds=timeout)
                alignment = result.alignment if result.optimal else None
            else:
                args = SimpleNamespace(timeout=timeout, max_expansions=100000,
                                       sliding_window_size=20, sliding_max_candidates=5,
                                       fixed_horizon=4)
                raw = pn_alignments.apply_trace(sample.trace, sample.net,
                    sample.initial_marking, sample.final_marking,
                    parameters=_petri_parameters(sample, args, name), variant=PETRI_VARIANTS[name])
                alignment = parse_pm4py_alignment(raw, sample.net) if raw is not None else None
            if alignment is None:
                row = {"legal": False, "cost": None, "error": "no complete result"}
            else:
                verification = verify_alignment(alignment, sample.net, sample.initial_marking,
                                                sample.final_marking, sample.trace, CostModel())
                row = {"legal": verification.legal,
                       "cost": verification.cost if verification.legal else None,
                       "error": verification.reason}
    except Exception as exc:
        row = {"legal": False, "cost": None, "error": f"{type(exc).__name__}: {exc}"}
    row["seconds"] = perf_counter() - start
    return row


def run_timing(samples):
    model, model_meta = checkpoint(ROOT / "runs/lara/best.pt")
    runners = systems(model)
    names = ["guided", "unguided", "repair_guided", "exact", *PETRI_VARIANTS]
    selected = [s for s in samples if s.group == "test"]
    for runner in runners.values():
        runner.decoder.max_final_model_depth = 32
    metadata = {"checkpoint": model_meta, "repeats": 5, "shuffle_seed": 902,
                "warmup_calls_per_method": 3, "inputs": len(selected), "timeout_seconds": 5,
                "python": platform.python_version(), "torch": torch.__version__,
                "pm4py": pm4py.__version__, "intra_threads": 1, "inter_threads": 1,
                "cpu": next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                            if l.startswith("model name")),
                "boundary": "in-memory net/trace through final independent replay; loading excluded"}
    path = OUT / "common_timing.jsonl"
    if path.exists():
        raise RuntimeError("Timing records already exist; do not mix interrupted and new timing runs")
    (OUT / "common_timing_protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for sample in selected[:3]:
        for name in names:
            timed_call(name, sample, runners)
    rng = Random(902)
    with path.open("w", buffering=1) as handle:
        for repeat in range(5):
            order = list(selected)
            rng.shuffle(order)
            for sample in order:
                methods = list(names)
                rng.shuffle(methods)
                for name in methods:
                    row = {"id": sample.sample_id, "method": name, "repeat": repeat,
                           **timed_call(name, sample, runners)}
                    if row["legal"]:
                        assert row["cost"] >= sample.optimum
                    handle.write(json.dumps(row) + "\n")
            print(f"timing repeat {repeat+1}/5 complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["reference", "checkpoints", "timing"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    samples = inputs()
    assert len({s.sample_id for s in samples}) == len(samples)
    print(json.dumps({"stage": args.stage, "groups": Counter(s.group for s in samples)}), flush=True)
    {"reference": run_reference, "checkpoints": run_checkpoints, "timing": run_timing}[args.stage](samples)


if __name__ == "__main__":
    main()
