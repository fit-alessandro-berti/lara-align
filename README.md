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

Train with validation checkpointing:

```bash
python scripts/train_model.py \
  --data-dir data/lara_synthetic \
  --output-dir runs/lara \
  --epochs 10
```

The best validation checkpoint is written to `runs/lara/best.pt`; the latest
epoch is written to `runs/lara/last.pt`.

Evaluate on the held-out test split:

```bash
python scripts/test_model.py \
  --data-dir data/lara_synthetic \
  --checkpoint runs/lara/best.pt \
  --split test
```

The test script reports held-out loss, fast-mode legal alignment rate,
fast-mode optimal-cost rate against the stored exact label, and mean cost gap
over legal fast-mode candidates. Add `--run-certified` if you also want to run
the pm4py certifying layer for every test sample.
