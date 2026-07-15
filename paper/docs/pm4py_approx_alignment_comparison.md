# Experimental Comparison: LARA vs. PM4Py Approximate Alignments

This document compares LARA against the approximate alignment variants in the
installed PM4Py `2.7.23.2` by actually running the variants and replay-checking
their outputs. It supersedes the earlier source-level-only assessment.

## What Was Run

Two experiments were executed.

### Experiment 1: LARA Held-Out Test Split

Command:

```bash
python3 scripts/benchmark_pm4py_approx.py \
  --timeout 5 \
  --output runs/lara/pm4py_approx_comparison.json
```

Input:

- split: `data/lara_synthetic/test.pkl`
- samples: 512
- trace length: mean 3.625, max 11
- exact reference: stored PM4Py state-equation A* optimal cost per sample
- independent legality/cost check: `lara_align.verify.verify_alignment`
- costs: unit model/log costs, zero synchronous and invisible costs

PM4Py variants run:

- `VERSION_DISCOUNTED_A_STAR`
- `APPROX_TANDEM_REPEATS`
- `APPROX_SLIDING_WINDOW`
- `APPROX_FIXED_HORIZON`
- `edit_distance.Variants.APPROX_SUBSET`, grouped by shared Petri-net variant

LARA baselines are the existing same-split artifacts:

- `runs/lara/test_guided.json`
- `runs/lara/test_unguided.json`

Important caveat: the test split is short. With `window_size = 20`,
`APPROX_SLIDING_WINDOW` processes every test trace in one window, so on this
split it behaves like a direct bounded search rather than a real windowing
stress test.

### Experiment 2: Generated Scaling Stress Grid

Command:

```bash
python3 scripts/benchmark_pm4py_approx_scaling.py \
  --approx-timeout 5 \
  --exact-timeout 30 \
  --output runs/lara/pm4py_approx_scaling_comparison.json
```

Input:

- generated block-structured stress nets from `generate_block_structured_example`
- sizes: 5, 10, 20, 40 visible activities
- deviation rates: 0.15 and 0.35
- samples per configuration: 10
- total generated samples: 80
- exact reference: PM4Py state-equation A* with a 30 s timeout
- exact solved: 79/80 samples
- independent legality/cost check: `lara_align.verify.verify_alignment`

Methods run on the same generated examples:

- LARA guided fast mode
- `VERSION_DISCOUNTED_A_STAR`
- `APPROX_TANDEM_REPEATS`
- `APPROX_SLIDING_WINDOW`
- `APPROX_FIXED_HORIZON`

The edit-distance subset method is log-level and representative-based, so it
was run on the held-out split where shared-net trace groups exist. It is not a
meaningful per-singleton competitor on the generated scaling grid.

## Experiment 1 Results: Held-Out Test Split

All PM4Py outputs below were parsed into transition-aware LARA alignments and
replayed with the LARA verifier. `median ms` is the per-call median for PM4Py
variants. The existing LARA artifacts report total evaluation time rather than
the same per-call median, so their speed is not put into this column.

| method | legal | exact-cost | mean gap | p95 gap | max gap | median ms | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| LARA guided fast | 100.0% | 91.0% | 0.180 | 2 | 4 | n/a | existing `test_guided.json`; total eval 11.35 s |
| LARA unguided fast | 100.0% | 86.9% | 0.258 | 2 | 4 | n/a | existing `test_unguided.json`; total eval 4.80 s |
| discounted A* | 100.0% | 95.7% | 0.090 | 0 | 5 | 0.57 | fastest high-quality Petri-net approximation on this split |
| tandem repeats | 100.0% | 98.0% | 0.039 | 0 | 2 | 0.45 | strong despite few long repeats |
| sliding window | 100.0% | 100.0% | 0.000 | 0 | 0 | 0.32 | one-window/direct-search regime because max trace length is 11 |
| fixed horizon | 100.0% | 98.0% | 0.047 | 0 | 4 | 4.62 | high quality but much slower than other PM4Py variants |
| edit-distance subset | 100.0% | 70.3% | 0.637 | 3 | 7 | 0.18 | very fast, but weak transition-level quality |

Gap histograms:

| method | gap 0 | gap 1 | gap 2 | gap 3 | gap 4 | gap 5 | gap 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LARA guided fast | 466 | 14 | 24 | 2 | 6 | 0 | 0 |
| LARA unguided fast | 445 | 16 | 43 | 2 | 6 | 0 | 0 |
| discounted A* | 490 | 12 | 4 | 0 | 4 | 2 | 0 |
| tandem repeats | 502 | 0 | 10 | 0 | 0 | 0 | 0 |
| sliding window | 512 | 0 | 0 | 0 | 0 | 0 | 0 |
| fixed horizon | 502 | 0 | 8 | 0 | 2 | 0 | 0 |
| edit-distance subset | 360 | 32 | 78 | 36 | 4 | 0 | 2 |

By family, the largest PM4Py losses on this split are concentrated as follows:

| method | weakest family | exact-cost in weakest family | mean gap |
|---|---|---:|---:|
| discounted A* | ordinary tree | 85.9% | 0.328 |
| tandem repeats | ordinary tree | 92.2% | 0.156 |
| sliding window | none | 100.0% | 0.000 |
| fixed horizon | concurrent/interleaved | 96.1% | 0.109 |
| edit-distance subset | M-pattern non-free-choice | 57.8% | 0.922 |

Interpretation:

- On the current short held-out split, PM4Py Petri-net approximations beat
  LARA guided fast quality. This is especially clear for sliding-window,
  tandem-repeat, and fixed-horizon variants.
- The sliding-window result should not be overclaimed: because every trace is
  shorter than the 20-event window, the approximation does not face a real
  window-boundary tradeoff here.
- The edit-distance subset approximation loses clearly to LARA guided fast on
  alignment quality, despite being very fast.

## Experiment 2 Results: Scaling Stress Grid

Aggregate across all 80 generated stress samples:

| method | legal | exact-comparable | exact-cost on comparable | mean gap | p95 gap | max gap | median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| LARA guided fast | 100.0% | 79 | 59.5% | 3.797 | 19 | 44 | 7.69 |
| discounted A* | 100.0% | 79 | 75.9% | 1.468 | 5 | 45 | 1.17 |
| tandem repeats | 97.5% | 78 | 93.6% | 0.128 | 2 | 2 | 1.37 |
| sliding window | 98.8% | 79 | 100.0% | 0.000 | 0 | 0 | 1.27 |
| fixed horizon | 95.0% | 76 | 82.9% | 2.263 | 6 | 44 | 82.37 |

`exact-comparable` excludes the one sample where exact A* timed out and any
method call that did not return a legal alignment within the configured
timeout.

Full stress grid:

| size | dev | method | legal | comparable | exact-cost | mean gap | p95 gap | median ms |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | 0.15 | LARA guided fast | 100% | 10 | 90.0% | 0.20 | 2 | 6.87 |
| 5 | 0.15 | discounted A* | 100% | 10 | 90.0% | 0.10 | 1 | 0.70 |
| 5 | 0.15 | tandem repeats | 100% | 10 | 100.0% | 0.00 | 0 | 0.38 |
| 5 | 0.15 | sliding window | 100% | 10 | 100.0% | 0.00 | 0 | 0.31 |
| 5 | 0.15 | fixed horizon | 100% | 10 | 100.0% | 0.00 | 0 | 11.00 |
| 5 | 0.35 | LARA guided fast | 100% | 10 | 60.0% | 1.60 | 10 | 6.60 |
| 5 | 0.35 | discounted A* | 100% | 10 | 80.0% | 0.30 | 2 | 0.64 |
| 5 | 0.35 | tandem repeats | 100% | 10 | 90.0% | 0.20 | 2 | 0.47 |
| 5 | 0.35 | sliding window | 100% | 10 | 100.0% | 0.00 | 0 | 0.45 |
| 5 | 0.35 | fixed horizon | 100% | 10 | 90.0% | 0.10 | 1 | 7.15 |
| 10 | 0.15 | LARA guided fast | 100% | 10 | 80.0% | 1.20 | 11 | 7.25 |
| 10 | 0.15 | discounted A* | 100% | 10 | 90.0% | 0.10 | 1 | 0.85 |
| 10 | 0.15 | tandem repeats | 100% | 10 | 100.0% | 0.00 | 0 | 0.52 |
| 10 | 0.15 | sliding window | 100% | 10 | 100.0% | 0.00 | 0 | 0.46 |
| 10 | 0.15 | fixed horizon | 100% | 10 | 100.0% | 0.00 | 0 | 17.61 |
| 10 | 0.35 | LARA guided fast | 100% | 10 | 50.0% | 1.70 | 6 | 7.26 |
| 10 | 0.35 | discounted A* | 100% | 10 | 70.0% | 0.40 | 2 | 0.94 |
| 10 | 0.35 | tandem repeats | 100% | 10 | 90.0% | 0.20 | 2 | 0.81 |
| 10 | 0.35 | sliding window | 100% | 10 | 100.0% | 0.00 | 0 | 0.71 |
| 10 | 0.35 | fixed horizon | 100% | 10 | 90.0% | 0.10 | 1 | 97.41 |
| 20 | 0.15 | LARA guided fast | 100% | 10 | 60.0% | 3.60 | 18 | 8.68 |
| 20 | 0.15 | discounted A* | 100% | 10 | 70.0% | 1.40 | 7 | 0.99 |
| 20 | 0.15 | tandem repeats | 100% | 10 | 90.0% | 0.20 | 2 | 1.26 |
| 20 | 0.15 | sliding window | 100% | 10 | 100.0% | 0.00 | 0 | 1.35 |
| 20 | 0.15 | fixed horizon | 100% | 10 | 90.0% | 0.40 | 4 | 245.52 |
| 20 | 0.35 | LARA guided fast | 100% | 10 | 60.0% | 3.10 | 15 | 7.88 |
| 20 | 0.35 | discounted A* | 100% | 10 | 90.0% | 0.30 | 3 | 2.62 |
| 20 | 0.35 | tandem repeats | 100% | 10 | 80.0% | 0.40 | 2 | 5.45 |
| 20 | 0.35 | sliding window | 100% | 10 | 100.0% | 0.00 | 0 | 3.83 |
| 20 | 0.35 | fixed horizon | 100% | 10 | 60.0% | 0.90 | 3 | 139.83 |
| 40 | 0.15 | LARA guided fast | 100% | 10 | 30.0% | 8.10 | 27 | 10.77 |
| 40 | 0.15 | discounted A* | 100% | 10 | 60.0% | 3.90 | 19 | 4.15 |
| 40 | 0.15 | tandem repeats | 90% | 9 | 100.0% | 0.00 | 0 | 23.76 |
| 40 | 0.15 | sliding window | 100% | 10 | 100.0% | 0.00 | 0 | 40.02 |
| 40 | 0.15 | fixed horizon | 80% | 8 | 62.5% | 11.75 | 44 | 1398.70 |
| 40 | 0.35 | LARA guided fast | 100% | 9 | 44.4% | 11.67 | 44 | 9.04 |
| 40 | 0.35 | discounted A* | 100% | 9 | 55.6% | 5.67 | 45 | 2.86 |
| 40 | 0.35 | tandem repeats | 90% | 9 | 100.0% | 0.00 | 0 | 12.51 |
| 40 | 0.35 | sliding window | 90% | 9 | 100.0% | 0.00 | 0 | 13.69 |
| 40 | 0.35 | fixed horizon | 80% | 8 | 62.5% | 7.88 | 38 | 437.50 |

Interpretation:

- Sliding window is the best quality method in these runs: every legal,
  exact-comparable sliding-window result has exact optimal cost. Its weakness
  is not cost quality here, but non-return on one large high-deviation case and
  higher median time than LARA on the size-40 cells.
- Tandem repeats is also strong: small gaps and perfect exact-cost rate on the
  legal comparable size-40 cases. It failed to return a legal result in two
  large cases under the configured timeout.
- Discounted A* is the fastest PM4Py Petri-net approximation with 100% legal
  return rate in these runs, and it beats LARA's cost quality on aggregate.
  However, its tail can be very bad: max gap 45 on the stress grid.
- Fixed horizon is mixed. It improves over LARA on several small and medium
  cells, but it is much slower and degrades badly at size 40, including no
  legal return for 20% of size-40 cases.
- LARA's current fast decoder is not competitive on optimality in this
  experiment. Its real advantage in this grid is predictable legal output and
  low size-40 latency versus the high-quality sliding-window/tandem/fixed
  alternatives, not lower cost.

## Where PM4Py Approximate Variants Win

The experiments show several concrete PM4Py wins.

### Alignment Quality on the Tested Distributions

On the held-out split, all Petri-net PM4Py approximations beat LARA guided fast
on exact-cost rate. On the stress grid, sliding window and tandem repeats are
substantially closer to the exact optimum than LARA.

This is the main correction to the earlier source-level assessment: the
current LARA checkpoint is not the strongest approximate aligner on these
benchmarks.

### Runtime for Small and Medium Cases

Discounted A*, tandem repeats, and sliding window are all sub-millisecond to
low-millisecond on the short held-out split. On stress sizes 5-20, they are
usually faster than LARA guided fast and more accurate.

### Structural Shortcuts Work Well Here

The generated stress nets contain block structure, loops, duplicate labels,
and invisible transitions, but the PM4Py approximations exploit enough
structure to stay close to exact. Sliding window in particular dominates cost
quality under the tested `window_size = 20`, `max_candidates = 5` setting.

### No Checkpoint Dependency

PM4Py's methods require no training corpus, no checkpoint, and no neural
inference. Given these results, that is a practical advantage: the simpler
methods are not merely simpler, they are also better on these measured
benchmarks.

## Where PM4Py Approximate Variants Lose

The PM4Py results are strong, but they do not win every criterion.

### Legal Return Reliability at Larger Size

LARA fast returned legal alignments for 100% of the stress samples. Under the
configured timeout:

- tandem repeats returned legal alignments for 97.5%;
- sliding window returned legal alignments for 98.8%;
- fixed horizon returned legal alignments for 95.0%.

For applications where a legal upper bound must always be returned quickly,
LARA's current decoder still has a reliability advantage over those variants
at the tested timeout.

### Large-Case Latency Versus High-Quality PM4Py Variants

On size-40 cells, LARA fast has lower median latency than sliding window,
tandem repeats, and fixed horizon:

| size | dev | LARA ms | tandem ms | sliding ms | fixed-horizon ms |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.15 | 10.77 | 23.76 | 40.02 | 1398.70 |
| 40 | 0.35 | 9.04 | 12.51 | 13.69 | 437.50 |

This is a speed-quality tradeoff, not a total win for LARA: the PM4Py methods
have much better costs when they return.

### Edit-Distance Subset Quality

The edit-distance subset method is very fast and useful at log level, but on
the held-out trace groups it is far below LARA guided fast:

- LARA guided exact-cost rate: 91.0%
- edit-distance subset exact-cost rate: 70.3%
- LARA guided mean gap: 0.180
- edit-distance subset mean gap: 0.637

The visible-string representative proxy is weak for transition-level alignment
diagnostics on these samples.

### Certification Still Needs an Exact Layer

The PM4Py approximate variants return upper-bound candidates, not optimality
certificates. LARA certified mode is still architecturally useful: it wraps a
candidate with exact PM4Py certification or exact repair. The experiments
suggest that PM4Py approximate candidates should be considered as candidate
generators inside the same certification wrapper, because several of them are
better candidates than the current LARA fast decoder.

## Where LARA Loses

The measured losses are clear.

### Current Fast-Mode Cost Quality

LARA guided fast loses to PM4Py Petri-net approximations on both experiments.
On the stress grid, LARA's aggregate exact-cost rate is 59.5% on comparable
cases, versus:

- discounted A*: 75.9%
- tandem repeats: 93.6%
- sliding window: 100.0%
- fixed horizon: 82.9%

### Greedy Decoder Myopia

The stress gaps are large for LARA: p95 gap 19 and max gap 44. This confirms
that the current greedy decoder is the bottleneck. It returns legal alignments,
but the local commitments are too expensive compared with PM4Py's bounded
search procedures.

### Learned Guidance Is Not Yet Paying for Itself Broadly

The existing LARA ablation already showed guided fast beats unguided mainly on
duplicate-prefix cases. The new comparison shows that PM4Py approximations
beat both on broader alignment quality. The current checkpoint should not be
positioned as a generally better approximated aligner.

## Consequences for the Paper

The paper should not claim that LARA beats PM4Py approximate alignments in the
current implementation. The data supports a more precise claim:

1. LARA is a certifying learned candidate-generation architecture.
2. The current greedy LARA candidate is legal and predictable, but often lower
   quality than PM4Py's new approximate Petri-net variants.
3. PM4Py sliding-window and tandem-repeat candidates are strong baselines and
   should be integrated into future certified/anytime comparisons.
4. The most promising next LARA step is not another small checkpoint tweak; it
   is replacing greedy decoding with beam search or neural-guided A* and
   comparing against PM4Py approximate candidates under the same certification
   wrapper.

## Practical Selection Guide Based on These Runs

| Requirement | Best supported choice from these experiments |
|---|---|
| Best cost quality on the tested Petri-net benchmarks | PM4Py sliding window |
| Strong quality with simple repeated-trace compression | PM4Py tandem repeats |
| Fastest legal Petri-net approximation with decent quality | PM4Py discounted A* |
| Log-level representative approximation | PM4Py edit-distance subset |
| Always return a legal fast candidate under tested stress timeout | LARA fast |
| Certified optimality or exact repair | LARA certified mode or a wrapper around any approximate candidate plus PM4Py exact A* |
| Best next research direction | use PM4Py approximate variants as candidate baselines and improve LARA decoding/search |

## Reproducibility Artifacts

Scripts added:

- `scripts/benchmark_pm4py_approx.py`
- `scripts/benchmark_pm4py_approx_scaling.py`

Result artifacts produced:

- `runs/lara/pm4py_approx_comparison.json`
- `runs/lara/pm4py_approx_scaling_comparison.json`

Both scripts independently replay and rescore returned alignments with LARA's
verifier, so the reported gaps use the same unit-cost semantics as the LARA
assessment.
