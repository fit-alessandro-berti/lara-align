# lara-align

Learned Adaptive Recombined Alignments: a neural-guided, certifying alignment
prototype for Petri-net conformance checking.

LARA separates candidate generation from certification:

- `LARANeuralModel` encodes a pm4py Petri net plus trace with PyTorch, learns
  trace-dependent latent regions, scores local alignment sketches, and produces
  move logits for recomposition.
- `GreedyCandidateDecoder` turns those scores into a concrete alignment by
  replaying enabled transitions and bounded model-prefix repairs.
- `verify_alignment` simulates the decoded model projection and checks that the
  log projection reconstructs the trace exactly.
- `CertifyingAlignmentSystem` uses pm4py's exact Petri-net alignment backend as
  the certifying layer. Learned scores can rank candidates, but only pm4py's
  exact result certifies optimality.

## Install

```bash
pip install -e ".[dev]"
```

## Minimal Usage

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

## Modes

- `fast`: return the neural/greedy candidate after legality verification. If the
  candidate is illegal, the result is marked illegal.
- `certified`: run pm4py's exact state-equation A* backend. If the legal neural
  candidate has the exact optimal cost, return it with `certified_optimal=True`;
  otherwise return the exact repair.
- `anytime`: currently uses the same exact backend with optional pm4py timeout.
  If exact search does not return an alignment, LARA reports the legal candidate
  with a trivial lower bound.

## Training Hooks

The package includes:

- `pm4py_to_features` for typed graph-plus-sequence tensors.
- `LARALoss` for move imitation, cost shaping, and router regularization.
- `targets_from_alignment` for converting exact alignments into supervision.
- `synthetic.generate_sequence_example` plus small Petri-net generators for
  smoke tests and curriculum scaffolding.

This is a foundation implementation: the neural model is trainable, but the
default instance is randomly initialized. Use certified mode for trustworthy
alignments unless you have trained weights.

## Data, Training, And Test Scripts

Initialize exact-labeled synthetic data split into train/validation/test:

```bash
python scripts/init_data.py \
  --output data/lara_synthetic \
  --train-size 128 \
  --val-size 32 \
  --test-size 32
```

This writes:

- `data/lara_synthetic/train.pkl`
- `data/lara_synthetic/val.pkl`
- `data/lara_synthetic/test.pkl`
- `data/lara_synthetic/metadata.pkl`

The initializer generates one exact-labeled pool, then stratifies the splits by
synthetic family and optimal-cost bucket.

Train with validation checkpointing:

```bash
python scripts/train_model.py \
  --data-dir data/lara_synthetic \
  --output-dir runs/lara \
  --epochs 15
```

The best validation checkpoint is written to `runs/lara/best.pt`; the latest
epoch is written to `runs/lara/last.pt`.

The default training configuration is intentionally modest for the default
128-example synthetic training split: 64 hidden units, minibatch-style gradient
accumulation over 8 variable-size examples, Smooth L1 cost regression, moderate
dropout/weight decay, validation-plateau learning-rate reduction, and
patience-based early stopping.

Evaluate on the held-out test split:

```bash
python scripts/test_model.py \
  --data-dir data/lara_synthetic \
  --checkpoint runs/lara/best.pt \
  --split test \
  --num-examples 5
```

The test script prints a human-readable report by default. It compares pm4py's
stored optimal alignment with LARA's fast reconstructed alignment for a few
examples, then reports replayable alignment rate, optimal-cost rate, exact
transition-sequence match rate, label-level alignment match rate, cost-gap
statistics, losses, and per-family metrics.

Useful evaluation options:

```bash
python scripts/test_model.py --format json --metrics-output runs/lara/test_metrics.json
python scripts/test_model.py --example-selection first --num-examples 10
python scripts/test_model.py --run-certified
```

`--run-certified` also runs the pm4py certifying layer for every sample. For
judging the learned method itself, the main report uses fast mode, because
certified mode may repair the candidate with exact pm4py search.
