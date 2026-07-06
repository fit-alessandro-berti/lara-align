# Approach: The LARA Foundation Model

LARA (Learned Adaptive Recombined Alignments) is a neural-guided, *certifying*
system for Petri-net trace alignment. The design separates **candidate
generation** (learned, fast, uncertified) from **certification** (exact,
pm4py-based, trusted). This document details (1) the architecture of the
neural foundation model, (2) how the exact-labeled training data is created,
and (3) the training procedure, its hyperparameters, and the metrics tracked
during training. Notation follows `preliminaries.md`.

## 1. System Overview

```mermaid
flowchart LR
    subgraph Input
        PN["Petri net (N, m0, mf)"]
        TR["Trace sigma"]
    end
    PN --> FE["Feature extraction<br/>(typed tensors)"]
    TR --> FE
    FE --> ENC["Petri/Trace encoder<br/>(typed graph transformer +<br/>trace transformer + cross-attention)"]
    ENC --> ROU["Learned router<br/>(latent regions)"]
    ROU --> EXP["Local alignment experts<br/>(sketches, cost bounds, uncertainty)"]
    ENC --> REC["Recomposer heads<br/>(sync / log / model move scores)"]
    REC --> DEC["Greedy candidate decoder<br/>(constrained by firing rule)"]
    DEC --> VER["Verifier<br/>(replay + trace reconstruction)"]
    VER --> CERT["Certifier<br/>(pm4py state-equation A*)"]
    CERT --> OUT["Legal alignment +<br/>optimality certificate or exact repair"]
```

The invariant maintained throughout: **neural output is never trusted as a
certificate.** The model proposes; exact Petri-net replay decides legality; the
pm4py exact backend decides optimality. In *fast* mode the system returns the
verified learned candidate (an upper bound on the optimal cost); in *certified*
mode it additionally runs the exact aligner and either certifies the candidate
by cost equality or returns the exact alignment as a repair.

## 2. Input Encoding

`pm4py_to_features` converts a pm4py net/trace pair into typed tensors
(`lara_align/features.py`). The net is represented as a bipartite graph with
places first and transitions second.

**Place features** (5 per place): initial-marking tokens, final-marking tokens,
in-degree, out-degree, total degree.

**Transition features** (6 per transition): invisibility flag
($\lambda(t)=\tau$), in-degree, out-degree, model-move cost $c(\gg,t)$,
synchronous-move cost, and a self-loop-context flag
($\bullet t \cap t\bullet \neq \emptyset$).

**Typed edges (bidirectional)**: every arc is materialized in both directions
with four relation types — `0`: place→transition along consumption arcs, `1`:
transition→place along production arcs, `2`: the reversal of consumption (a
place hears from its consumers), `3`: the reversal of production (a transition
hears from its output places). The reverse types are essential for
duplicate-label resolution: the context that distinguishes two transitions
sharing a label (e.g., which branch suffix follows each of them) lies
*downstream*, and messages restricted to arc direction can never deliver it —
duplicate transitions with symmetric presets would receive provably identical
embeddings, and global attention cannot separate them because identical
embeddings issue identical queries. Bidirectional typed messages carry the
disambiguating context in two hops (suffix transition → shared place →
duplicate transition), which is what allows the sync head to discriminate
transition identities (`assessment.md` §4).

**Labels via hashing.** Activity labels are mapped into a bounded vocabulary of
8,192 ids by a stable BLAKE2b hash (`stable_label_id`), with id 0 reserved for
$\tau$. Hashing makes the model *label-vocabulary-free*: it can embed activity
names it has never seen during training, a prerequisite for foundation-model
use across event logs. Crucially, an event and a transition with the same label
receive the same embedding id, which is the inductive bias that lets the model
tie trace positions to candidate transitions.

**Compatibility mask.** A boolean matrix
$C \in \{0,1\}^{n \times |T|}$ with $C_{ij} = 1$ iff event $i$ and transition
$j$ carry the same visible label. It hard-masks synchronous-move scores so the
model can only propose label-consistent synchronous moves.

## 3. Neural Architecture

`LARANeuralModel` (`lara_align/model.py`) has four components.

### 3.1 Petri/Trace Encoder

- **Node embedding.** Place and transition feature rows are linearly projected
  to the hidden dimension $d = 128$; transitions additionally receive their
  hashed label embedding; a 2-way node-type embedding distinguishes places from
  transitions.
- **Typed graph transformer (3 layers).** Each `TypedGraphTransformerLayer`
  combines (i) full self-attention over all net nodes (4 heads) with (ii)
  relation-specific message passing: per edge type (4 types, forward and
  reverse), source embeddings pass through a dedicated linear map and are
  mean-aggregated into targets. The sum of attention output and typed messages
  enters a pre-norm residual block with a 4× GELU feed-forward.
- **Trace transformer (2 layers).** Events are embedded via the shared label
  embedding plus learned positional embeddings (max length 4,096) and encoded
  by a standard Transformer encoder (4 heads, 4× GELU feed-forward).
- **Cross-attention coupling.** One bidirectional multi-head cross-attention
  round lets events attend to transitions and transitions attend to events,
  each followed by a residual LayerNorm. This is where trace context flows into
  transition representations (and vice versa), making all downstream heads
  *trace-dependent*.

### 3.2 Learned Router (latent regions)

Two linear heads map transition and event embeddings to soft assignments over
$R = 8$ latent regions (softmax). The router is the learned analogue of
decomposition-based alignment: instead of a fixed structural decomposition of
the net, region membership is predicted per (net, trace) pair. Three
regularizers (Section 5) push the assignments toward decompositions that are
balanced, confident, and structurally coherent.

### 3.3 Local Alignment Experts

For each region, transition and event embeddings are pooled by
assignment-probability-weighted averaging, concatenated, and passed through a
2-layer GELU MLP. Per region the experts predict:

- **sketch logits** over $S = 6$ local alignment sketches per region
  (a coarse classification of the local deviation pattern);
- a **lower cost bound** (Softplus, non-negative);
- an **upper cost bound**, parameterized as lower bound + non-negative
  Softplus increment so that $\widehat{ub} \geq \widehat{lb}$ by construction;
- an **uncertainty** estimate (Softplus).

The summed upper bounds $\sum_r \widehat{ub}_r$ serve as the model's global
cost estimate and are supervised against the optimal cost. Predicted bounds
are *diagnostic* signals for search prioritization; they are never used as
admissible heuristics, because a learned bound carries no admissibility
guarantee.

### 3.4 Recomposer Heads

Global move scores over the full (net, trace) pair:

- **Synchronous-move logits** $\mathrm{sync} \in \mathbb{R}^{n \times |T|}$,
  computed as a scaled bilinear form
  $(W e_i)^\top h_j / \sqrt{d}$ between event embedding $e_i$ and transition
  embedding $h_j$, hard-masked by the compatibility matrix ($-10^9$ where
  labels differ). Row $i$ is the model's belief about *which concrete
  transition* event $i$ should synchronize with — this is where duplicate-label
  resolution happens.
- **Log-move logits** $\in \mathbb{R}^{n}$: per event, the belief that it is
  a deviation to be skipped.
- **Model-move logits** $\in \mathbb{R}^{|T|}$: per transition, the belief
  that it must fire without log support.

### 3.5 Parameter Budget

Default configuration (hidden 128, 4 heads, 3 graph layers, 2 trace layers,
8 regions, 6 sketches/region, 4 edge types):

| component | parameters |
|---|---:|
| Petri/Trace encoder | 2,895,360 |
| Learned router | 2,064 |
| Local alignment experts | 50,569 |
| Recomposer heads | 16,642 |
| **total** | **2,964,635** |

The encoder dominates; within it, the hashed label embedding
(8,192 × 128 ≈ 1.05M) and positional embedding (4,096 × 128 ≈ 0.52M) account
for more than half of all parameters. At ~3M parameters the model runs
comfortably on CPU.

## 4. Candidate Decoding, Verification, and Certification

**Greedy constrained decoder** (`GreedyCandidateDecoder`). The decoder walks
the trace left to right while tracking the current marking, so every emitted
move is fireable by construction:

1. For event $a_i$, collect *enabled* transitions with label $a_i$ and pick
   the one with the highest sync logit (duplicate-label resolution).
2. If none is enabled, run a bounded breadth-first search over markings (depth
   ≤ 8) for a model-move path that enables label $a_i$; successor transitions
   are ordered invisible-first, then by model-move logit. If found, emit the
   path as model moves and synchronize.
3. Otherwise emit a log move for $a_i$.
4. After the last event, search a model-move path to $m_f$ (depth ≤ 32).

For speed, the decoder precomputes a per-net runtime before the walk: indexed
presets/postsets, place-to-consumer lists (so only transitions consuming from
marked places are tested for enabledness), zero-free dict markings, and neural
scores extracted once into Python floats. This keeps the search cost mild in
net size; the neural forward pass dominates fast-mode wall time
(`assessment.md` §7.1).

**Verifier** (`verify_alignment`). Replays the transition projection from
$m_0$, checks that $m_f$ is reached and that the log projection reconstructs
$\sigma$ exactly, and computes the candidate's cost under the unit cost model.

**Certifier** (`CertifyingAlignmentSystem`). Three modes: `FAST` returns the
verified candidate (its cost is an upper bound on $\delta$); `CERTIFIED` also
runs pm4py's state-equation A* — if the legal candidate's cost equals the exact
cost, the candidate is certified optimal, otherwise the exact alignment is
returned as the repair; `ANYTIME` degrades gracefully when the exact backend
fails or times out, returning the best available bounds.

## 5. Training Objective

`LARALoss` (`lara_align/training.py`) is a composite of imitation, cost
shaping, and router regularization. Targets are derived from the pm4py optimal
alignment of each sample (`targets_from_alignment`): for each trace position, a
synchronous-target transition index (or −1), a binary log-move flag; for each
transition, a binary model-move flag.

$$\mathcal{L} = w_m\,(\mathcal{L}_{\text{sync}} + \mathcal{L}_{\text{log}} + \mathcal{L}_{\text{model}}) + w_c\,\mathcal{L}_{\text{cost}} + w_b\,\mathcal{L}_{\text{boundary}} + w_\ell\,\mathcal{L}_{\text{balance}} + w_e\,\mathcal{L}_{\text{entropy}}$$

| term | definition | weight |
|---|---|---:|
| $\mathcal{L}_{\text{sync}}$ | cross-entropy of sync logits vs. the optimal transition identity, over positions aligned synchronously | 1.0 |
| $\mathcal{L}_{\text{log}}$ | binary cross-entropy of log-move logits vs. optimal log moves (optional label smoothing) | 1.0 |
| $\mathcal{L}_{\text{model}}$ | binary cross-entropy of model-move logits vs. transitions fired as model moves | 1.0 |
| $\mathcal{L}_{\text{cost}}$ | smooth-L1 ($\beta = 1$) between $\sum_r \widehat{ub}_r$ and the optimal cost $\delta$ | 0.03 |
| $\mathcal{L}_{\text{boundary}}$ | for transitions sharing a place, expected probability of falling in different regions (structural coherence of the learned decomposition) | 0.01 |
| $\mathcal{L}_{\text{balance}}$ | variance of mean region loads across transitions and events (avoid region collapse) | 0.01 |
| $\mathcal{L}_{\text{entropy}}$ | mean assignment entropy (push toward confident routing) | 0.001 |

Supervising the *transition identity* (not merely the label) in
$\mathcal{L}_{\text{sync}}$ is what teaches duplicate-label disambiguation.

## 6. Training Data Creation

`scripts/init_data.py` generates a fully *exact-labeled* synthetic corpus.

**Generation pipeline** (per example):

1. **Sample a model family.** With probability 0.8 a *sequence* net over a
   sampled label sequence (length 3–8 from an 8-letter alphabet; with
   probability 0.25 prefixed by an invisible transition); with probability 0.2
   a *duplicate-label choice* net: an XOR of two branches whose first
   transitions share label `A`, followed by distinguishing suffixes `B`/`C`
   and invisible join transitions — by construction the trace suffix, not the
   ambiguous event itself, reveals the intended transition.
2. **Inject controlled deviations** into the fitting trace
   (`inject_deviations`, rate 0.25): per event, independent deletion, random
   insertion, and duplication (each with rate/3); plus trace-level adjacent
   swap and random append (each with the full rate).
3. **Label exactly.** pm4py's state-equation A* computes an optimal alignment;
   examples whose exact search fails or is non-optimal are discarded.
4. **Verify.** The optimal alignment must pass `verify_alignment` (replay to
   $m_f$ + exact trace reconstruction); the verified cost becomes the label.
5. **Stratified split.** The pooled examples are split into train/val/test
   stratified jointly by family and by optimal-cost bucket
   ($\min(\delta, 4)$), so the difficulty and family mix are matched across
   splits.

**Resulting dataset** (seed 13, defaults):

| split | examples | sequence | duplicate-label choice | trace length (min/mean/max) | optimal cost range |
|---|---:|---:|---:|---|---|
| train | 400 | 327 | 73 | 1 / 5.30 / 12 | 0–6 |
| val | 100 | 82 | 18 | 1 / 5.16 / 11 | 0–6 |
| test | 100 | 82 | 18 | 1 / 5.25 / 12 | 0–6 |

Optimal-cost distribution (train): $\delta{=}0$: 86, $1$: 116, $2$: 98,
$3$: 65, $4$: 22, $5$: 10, $6$: 3. Nets range over 3–9 transitions and 4–10
places. Every sample stores the net, markings, trace, the pm4py-optimal
alignment with transition identities, and the exact optimal cost.

## 7. Training Procedure and Hyperparameters

`scripts/train_model.py` trains on CPU by default with per-sample forward
passes accumulated into batched optimizer steps.

| hyperparameter | value |
|---|---|
| epochs (max) | 50, early stopping patience 8 (min-delta $10^{-4}$) |
| optimizer | AdamW, lr $5 \times 10^{-4}$, weight decay $10^{-3}$ |
| LR schedule | ReduceLROnPlateau on val total loss (factor 0.5, patience 3, min lr $10^{-5}$) |
| gradient-accumulation batch size | 16 |
| gradient clipping | max norm 1.0 |
| hidden dim / heads | 128 / 4 |
| graph / trace layers | 3 / 2 |
| latent regions / sketches per region | 8 / 6 |
| dropout | 0.25 |
| loss weights | move 1.0, cost 0.03, boundary 0.01, balance 0.01, entropy 0.001 |
| seed | 13 |

**Checkpointing and metrics.** After every epoch the trainer writes `last.pt`,
updates `best.pt` when the validation total loss improves, and appends a row to
`metrics.csv` containing: per-split component losses (sync/log/model moves,
cost, router boundary/balance/entropy, aggregate move loss, total), epoch wall
time, current learning rate, best validation loss, and the early-stopping
counter. These per-epoch curves are analyzed in `assessment.md`.

**Reference run.** The checkpoint evaluated in the paper
(`runs/lara_bidir/best.pt`) comes from a run of 22 recorded epochs at
~32 s/epoch on CPU; the best validation total loss, 0.4614, was reached at
epoch 18 (the run was stopped manually before the early-stopping criterion
fired).

## 8. Design Rationale

- **Guidance, not replacement.** Learned scores rank decisions; exact replay
  and exact search retain sole authority over legality and optimality. This
  keeps the system's answers as trustworthy as classical conformance checking.
- **Identity-level supervision** attacks the duplicate-label and invisible-
  transition ambiguities that inflate exact-search effort.
- **Hash-based label embeddings** decouple the model from any fixed activity
  vocabulary — a necessary property for a process-mining foundation model.
- **Trace-dependent routing** generalizes static net decomposition: the same
  net can be partitioned differently depending on where the trace deviates.
- **Bounded greedy decoding** guarantees that every emitted model move is
  fireable, so illegality can only arise from unreachable final markings —
  and in practice the verifier catches even that (see `assessment.md`).
