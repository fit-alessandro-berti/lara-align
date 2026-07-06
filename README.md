# lara-align

Learned Adaptive Recombined Alignments (LARA) is an experimental system for
neural-guided, certifying Petri-net trace alignment.

## Abstract

Alignment-based conformance checking compares an observed event trace with the
behavior allowed by a process model. For Petri nets, the standard formulation is
a shortest-path problem over a synchronous product: synchronous moves, log-only
moves, and model-only moves are assigned costs, and the optimal alignment is the
minimum-cost path. Exact methods are trustworthy, but their search space can grow
quickly in the presence of concurrency, loops, duplicate labels, and invisible
transitions.

This repository studies a hybrid alternative. LARA trains a PyTorch model to
encode a Petri net and trace, predict trace-dependent latent regions, score local
alignment sketches, and produce a strong legal candidate. A pm4py-based exact
layer remains responsible for legality checks, repair, and optimality
certification. The neural model is therefore used as guidance, not as an
uncertified replacement for exact conformance checking.

## Problem

Given a Petri net

```text
N = (P, T, F, lambda, m0, mf)
```

and an observed trace

```text
sigma = <a1, ..., an>
```

the task is to construct an alignment

```text
gamma = <(x1, y1), ..., (xk, yk)>
```

where each move is one of:

- synchronous: `(a, t)`
- log-only: `(a, >>)`
- model-only: `(>>, t)`

The log projection must reconstruct the trace exactly, and the model projection
must be fireable from the initial marking `m0` to the final marking `mf`.

## Hypothesis

The central hypothesis is that a learned model can reduce the practical burden
of exact alignment search by learning:

- which duplicate-labeled transition is likely intended;
- where invisible transitions are needed;
- where local deviations are likely;
- which regions of the net and trace should be considered together;
- which candidate alignments are promising enough to verify first.

Correctness is preserved by keeping exact Petri-net simulation and pm4py
alignment in the loop. Learned scores may rank candidates, but they are not used
as admissible lower bounds.

## Proposed Solution

LARA separates candidate generation from certification.

1. `LARANeuralModel` encodes a typed Petri-net graph and event sequence.
2. A learned router assigns transitions and events to latent regions.
3. Local expert heads score alignment sketches, lower/upper cost estimates, and
   uncertainty.
4. A recomposer head scores synchronous/log/model moves.
5. `GreedyCandidateDecoder` builds a replayable candidate when possible.
6. `verify_alignment` simulates the transition projection and checks trace
   reconstruction.
7. `CertifyingAlignmentSystem` compares the candidate with pm4py's exact
   state-equation A* alignment and certifies optimality when costs match.

The important design constraint is that neural output is never trusted as a
certificate. Fast mode evaluates the learned candidate. Certified mode invokes
pm4py and returns an exact repair if the candidate is not already optimal.

## Research Questions

This prototype is organized around the following research questions.

**RQ1. Legality**
Can a neural-guided decoder produce alignments whose model projection can be
replayed and whose log projection exactly reconstructs the trace?

**RQ2. Near-optimality**
How often does the reconstructed alignment have the same cost as pm4py's exact
optimal alignment, and what is the distribution of cost gaps when it does not?

**RQ3. Ambiguity**
How well does transition-identity decoding handle duplicate labels and invisible
transitions compared with label-only decoding?

**RQ4. Generalization**
How does performance change across synthetic families, deviation types, and
held-out train/validation/test splits?

**RQ5. Certification**
When the learned candidate is legal, how often can it be certified optimal by
cost equality against the exact backend, and when does exact repair remain
necessary?

## Experimental Protocol

The default experiment is intentionally larger than a smoke test while still
being small enough to run on a workstation.

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Initialize Data

```bash
python scripts/init_data.py --overwrite
```

Default split sizes:

- train: 400 examples
- validation: 100 examples
- test: 100 examples

The initializer generates exact-labeled synthetic Petri-net/trace pairs using
pm4py. It currently mixes sequential nets and duplicate-label choice nets, then
injects controlled deviations. A single exact-labeled pool is stratified by
synthetic family and optimal-cost bucket before writing:

- `data/lara_synthetic/train.pkl`
- `data/lara_synthetic/val.pkl`
- `data/lara_synthetic/test.pkl`
- `data/lara_synthetic/metadata.pkl`

For a quick smoke dataset:

```bash
python scripts/init_data.py \
  --output /tmp/lara_smoke_data \
  --train-size 32 \
  --val-size 8 \
  --test-size 8 \
  --overwrite
```

### 3. Train

```bash
python scripts/train_model.py
```

Default training configuration:

- epochs: 50
- hidden dimension: 128
- graph transformer layers: 3
- trace transformer layers: 2
- latent regions: 8
- local sketches per region: 6
- gradient-accumulation batch size: 16
- optimizer: AdamW
- learning rate: `5e-4`
- weight decay: `1e-3`
- dropout: `0.25`
- validation-plateau learning-rate scheduler
- early stopping patience: 8 epochs

The script writes:

- `runs/lara/best.pt`
- `runs/lara/last.pt`
- `runs/lara/metrics.csv`

`metrics.csv` is updated once per epoch with train and validation losses,
elapsed time, learning rate, best validation loss, and early-stopping state. Use
`--metrics-csv PATH` to write it somewhere else.

Training also displays tqdm progress bars while loading the train/validation
split files, advancing epochs, and processing train/validation batches. Use
`--no-progress` to disable them.

On CPU, this is a research run rather than a unit test; one epoch over the
default 400-example training split can take a few minutes. For a quick training
smoke test, use the small dataset command above and override the model size:

```bash
python scripts/train_model.py \
  --data-dir /tmp/lara_smoke_data \
  --output-dir /tmp/lara_smoke_run \
  --epochs 2 \
  --hidden-dim 32 \
  --graph-layers 1 \
  --trace-layers 1 \
  --num-regions 2 \
  --sketches-per-region 2 \
  --batch-size 4
```

### 4. Evaluate

```bash
python scripts/test_model.py \
  --data-dir data/lara_synthetic \
  --checkpoint runs/lara/best.pt \
  --split test \
  --num-examples 5
```

The evaluator prints a human-readable report containing:

- replayable alignment rate;
- optimal-cost rate against the pm4py label;
- exact transition-sequence match rate;
- label-level alignment match rate;
- mean, median, min, max, p90, p95, and standard deviation of cost gaps;
- per-family legality and optimality metrics;
- example traces with side-by-side pm4py optimal and LARA reconstructed
  alignments.

Machine-readable output is also available:

```bash
python scripts/test_model.py \
  --format json \
  --metrics-output runs/lara/test_metrics.json
```

Use `--run-certified` to also run the pm4py certifying layer for each test
sample. The main learned-method metrics use fast mode because certified mode may
repair the candidate with exact search.

### 5. Scaling Benchmark

```bash
python scripts/benchmark_scaling.py \
  --checkpoint runs/lara/best.pt \
  --sizes 5,10,20,40,80 \
  --deviation-rates 0.15,0.35 \
  --samples-per-config 10 \
  --exact-timeout 30 \
  --output runs/lara/scaling.json
```

The benchmark generates random block-structured nets (nested sequence, choice,
parallel, and loop blocks with duplicate labels and invisible transitions) of
increasing size, injects deviations, and measures per trace: LARA fast-mode
wall time, legality, and cost gap versus pm4py's exact aligner, plus exact
wall time and timeouts. The report includes a speed/quality threshold summary
indicating at which size (if any) the learned fast path overtakes exact
search.

## Minimal API Example

```python
from pm4py.objects.log.obj import Event, Trace

from lara_align import CertifyingAlignmentSystem, LARAMode
from lara_align.synthetic import make_sequence_net

net, im, fm = make_sequence_net(["A", "B", "C"])
trace = Trace([
    Event({"concept:name": "A"}),
    Event({"concept:name": "X"}),
    Event({"concept:name": "B"}),
    Event({"concept:name": "C"}),
])

lara = CertifyingAlignmentSystem()
result = lara.align(net, im, fm, trace, mode=LARAMode.CERTIFIED)

print(result.certified_optimal)
print(result.cost)
print(result.alignment.to_pm4py_label_alignment())
```

## Package Map

- `lara_align/features.py`: typed pm4py-to-tensor conversion.
- `lara_align/model.py`: graph/trace encoder, learned router, local experts,
  and recomposer heads.
- `lara_align/decode.py`: constrained greedy candidate decoder.
- `lara_align/verify.py`: Petri-net replay and alignment legality checks.
- `lara_align/exact.py`: pm4py exact alignment backend.
- `lara_align/certifier.py`: fast/certified alignment wrapper.
- `lara_align/synthetic.py`: synthetic process-model and trace generators,
  including random block-structured nets with concurrency, loops, duplicate
  labels, and invisible transitions.
- `lara_align/training.py`: training targets and losses.
- `scripts/init_data.py`: exact-labeled dataset creation.
- `scripts/train_model.py`: training and validation.
- `scripts/test_model.py`: human-readable and JSON evaluation.
- `scripts/benchmark_scaling.py`: speed/quality scaling benchmark against the
  exact backend on progressively larger block-structured nets.

## Current Limitations

This is a research prototype. The synthetic generator is still small compared
with the process-model diversity required for a foundation model, and the fast
decoder is a constrained greedy decoder rather than a full neural-guided A*
implementation. The exact pm4py backend remains the source of truth for
optimality certification.
