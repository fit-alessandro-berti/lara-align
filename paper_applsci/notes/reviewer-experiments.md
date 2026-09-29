# Added reviewer experiments

The original reference results and checkpoint are unchanged. The added study
is in Section 7.9 and Tables 18 to 22. No files were uploaded. The four new
checkpoints remain in local folders under `runs/reviewer_20260929`.

## Inspect results without new weights

From the repository root, using its Python dependencies:

```bash
make -C paper_applsci verify-revision
make -C paper_applsci revision-tables
```

The first command reads the original test split directly from the included
`data/reference_corpus.tar.gz`, the frozen added inputs from
`data/reviewer_inputs.pkl.gz`, and `data/reviewer_experiment_records.json.gz`.
It independently replays 12,756 witnesses across 1,676 inputs, checks reported
costs and counts, and audits 20,480 timing records. It loads no checkpoint.
Replay proves feasibility and cost; the optimality references were established
by the completed exact searches, not by replay alone.

`data/reviewer_experiments.json` contains the aggregate results, conditional
paired bootstrap intervals, all checkpoint hashes and selected epochs,
training arguments and software metadata, the input manifest and case split
indices, and timing protocol. Four `data/revision_*_metrics.csv` files contain
the complete new training histories. `data/sepsis_source.json` records the
primary dataset metadata and downloaded file hash.

## Training runs

The new runs use the original 2,048 training and 512 validation rows, the
reference architecture and optimization settings, a maximum of 50 epochs,
and validation-based selection with the original patience and minimum
improvement. Test outputs do not influence selection. The ablated runs set
only the log-loss coefficient to zero and select by their own validation
objective. The schedule and selected epoch can consequently differ.

CSV component losses are unweighted diagnostic values. In particular,
`train_move` and `val_move` retain the sum of all three move-head losses even
when log supervision has weight zero. `train_total` and `val_total` apply
the configured weights; only `val_total` controls scheduling and selection.

| Folder | Seed | Log-loss weight | Completed epochs | Selected epoch |
| --- | ---: | ---: | ---: | ---: |
| `seed17` | 17 | 1 | 33 | 27 |
| `seed29` | 29 | 1 | 50 | 48 |
| `seed17_no_log` | 17 | 0 | 50 | 50 |
| `seed29_no_log` | 29 | 0 | 50 | 48 |

The CPU is an Intel Xeon Platinum 8468, with Python 3.12.13, PyTorch
2.10.0+cu128, and PM4Py 2.7.23.6. All computations are CPU-only with one
intra-operation and one inter-operation thread. The original seed-13 run
used an i7-1355U, ten threads, and PM4Py 2.7.23.2. The three-run standard
deviation describes those observed runs and is not a pure estimate of seed
variance under one execution environment.

In a fresh checkout with the original corpus restored as documented in the
main README, the commands are:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p runs/reviewer_20260929/{seed17,seed29,seed17_no_log,seed29_no_log}
python -u paper_applsci/scripts/train_revision.py --seed 17 --output-dir runs/reviewer_20260929/seed17 --no-progress > runs/reviewer_20260929/seed17/train.log 2>&1
python -u paper_applsci/scripts/train_revision.py --seed 29 --output-dir runs/reviewer_20260929/seed29 --no-progress > runs/reviewer_20260929/seed29/train.log 2>&1
python -u paper_applsci/scripts/train_revision.py --seed 17 --log-loss-weight 0 --output-dir runs/reviewer_20260929/seed17_no_log --no-progress > runs/reviewer_20260929/seed17_no_log/train.log 2>&1
python -u paper_applsci/scripts/train_revision.py --seed 29 --log-loss-weight 0 --output-dir runs/reviewer_20260929/seed29_no_log --no-progress > runs/reviewer_20260929/seed29_no_log/train.log 2>&1
```

Preserve existing experiments rather than rerunning these commands over them. The independent
training commands may run concurrently. The wrapper records hardware,
thread settings, arguments, and process ID before invoking the training CLI.

## Frozen evaluation inputs

`scripts/prepare_revision_inputs.py` materializes the study before evaluation
and refuses to replace existing inputs. The study contains:

- 64 acyclic and 64 cyclic trees, balanced over sizes 6, 10, 14, and 18,
  generated with seed 703. A clean and corrupted observation is paired across
  the original compiler and PM4Py, giving 512 rows. Both compiled graphs must
  differ from every training graph under isomorphism preserving node types,
  arc weights, and initial/final tokens. Activity strings and identifiers are
  ignored by this check. All training graphs are checked acyclic.
- Road traffic, receipt, and sepsis, split by case with seed 37 into 70%
  discovery and 30% evaluation. Discovery uses only the former cases. All
  evaluation variants are retained at two discovery thresholds, giving 652
  variant/model rows. Only activity sequences are retained from the logs.

For exact reproduction use the frozen input snapshot, including its stored
transition identifiers. PM4Py can generate fresh identifiers on regeneration;
identifier-based tie breaking can then change candidate costs even when
behavior is unchanged. No input is filtered using candidate quality.

To recover the evaluation inputs in a fresh checkout:

```bash
mkdir -p data/reviewer_20260929
cp paper_applsci/data/reviewer_inputs.pkl.gz data/reviewer_20260929/inputs.pkl.gz
cp paper_applsci/data/sepsis_source.json data/reviewer_20260929/sepsis-source.json
python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('paper_applsci/data/reviewer_experiments.json').read_text())
Path('data/reviewer_20260929/inputs_manifest.json').write_text(
    json.dumps(summary['input_manifest'], indent=2) + '\n')
PY
```

The original public checkpoint is needed for inference; its restoration is
documented in the main README. The added input SHA-256 is
`4cb3e01c3c342119055251458b773cf806a88398cc37a91be837ef50afd482e1`.
The sepsis source DOI is
`10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460`; its downloaded XES gzip
SHA-256 is
`709c52340306415952811b9b9c5dc6bcc8f8d47d583eba39df9a538459dc543a`.

## Quality and timing

Run these stages sequentially after training has finished:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u paper_applsci/scripts/evaluate_revision.py reference
python -u paper_applsci/scripts/evaluate_revision.py checkpoints
python -u paper_applsci/scripts/evaluate_revision.py timing
make -C paper_applsci revision-evidence
```

Quality records are resumable by completed input ID; a truncated line is an
error requiring inspection. Checkpoint evaluation requires the training log's
completion marker and never evaluates a merely existing live checkpoint.
Timing refuses to mix an existing output file with a new run. Preserve prior
records if a separate rerun is needed.

Quality uses five-second candidate budgets and 30-second exact budgets, with
a 31-second external exact cutoff. Prefix depth is eight; completion depth is
32 on the reference test and 64 on added inputs. Repair tries up to eight
single forced-log alternatives using one forward pass and accepts only a
cheaper complete replayed candidate. It does not establish optimality.

Timing runs only after training and quality jobs have finished. Eight methods
process all 512 reference test inputs, with three warmup inputs per method,
five repetitions, randomized order (seed 902), and a common five-second
per-call budget. Timers include preparation, inference/search, parsing, and
final independent replay; existing internal checks are retained. Loading
and discovery are excluded. The batch subset method is outside this per-trace
comparison. Each input is summarized by its median of five calls, and the
table reports the median and interquartile range across those input medians.
All attempted calls remain in coverage and optimality denominators.

The summary script checks new checkpoint hashes and training arguments,
validation selections, witness legality/cost, paired compiler optima, and
timing coverage before exporting records. Bootstrap intervals use 10,000
whole-family resamples, stratified by the four motifs, with seed 13. They are
conditional on the fitted models; they are not intervals over training seeds.
