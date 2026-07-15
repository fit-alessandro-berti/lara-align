# Main Contribution and Research Questions

This document states, at a high level, what the paper contributes and which
research questions it answers. Details of the method are in `approach.md`;
the supporting evidence is in `assessment.md`; background and notation are in
`preliminaries.md`.

## The Problem

Alignment-based conformance checking is the gold standard for explaining how
an observed trace deviates from a process model: it produces an optimal
sequence of synchronous, log, and model moves. Its guarantee is also its
burden — optimality is computed by exact shortest-path search over a
synchronous product, whose state space degrades quickly with concurrency,
loops, invisible transitions, and duplicate labels. Machine learning promises
speed, but a purely learned aligner is unusable for conformance checking: an
uncertified alignment may be illegal (not replayable on the model) or
silently suboptimal, and conformance conclusions drawn from it cannot be
trusted.

The tension, then, is between *trust* and *tractability*. This paper asks
whether that tension can be dissolved rather than traded off.

## Main Contribution

**LARA (Learned Adaptive Recombined Alignments): a neural-guided alignment
system in which learning proposes and exact reasoning certifies — so that
speed comes from the model, but correctness never depends on it.**

The contribution decomposes into four parts:

1. **A certifying architecture.** A strict separation of roles: a neural
   model scores alignment decisions; a constrained greedy decoder turns those
   scores into a candidate that is fireable by construction; an exact replay
   verifier decides legality; and pm4py's exact aligner remains the sole
   authority on optimality, certifying the candidate by cost equality or
   repairing it. Neural output is never trusted as a certificate — the system
   degrades from "fast and certified optimal" to "fast with a safe upper
   bound", never to "wrong".

2. **A vocabulary-free foundation model for Petri nets and traces.** A typed
   graph transformer over the net (with bidirectional arc messages, which are
   provably necessary for duplicate-label disambiguation), a trace
   transformer, and cross-attention coupling — with activity labels embedded
   through a stable hash, so the same ~1.7M-parameter model applies zero-shot
   to event logs and models whose vocabulary it has never seen.

3. **An exact-labeled synthetic training pipeline.** Generators for
   balanced behavior families (including duplicate-prefix versus silent
   routing, parallel versus interleaved behavior, ordinary block trees,
   isomorphic renamings, and non-free-choice motifs) whose traces are labeled
   with provably optimal alignments and split before representation expansion
   — making supervised training on *transition identities*, not just labels,
   possible without leaking equivalent variants across splits.

4. **An evaluation methodology for learned alignment.** Beyond accuracy: a
   guidance ablation that isolates what the learned component contributes
   over constrained search alone; a scaling benchmark that locates the
   speed/quality threshold where learned decoding overtakes exact A*; and
   zero-shot validation on real-life event logs against discovered models.

## Research Questions

The central question is:

> **Can a learned model make trace alignment fast without making it less
> trustworthy?**

It is refined into five concrete questions.

**RQ1 — Legality.** Can neural-guided decoding produce alignments that are
*always* legal — replayable from the initial to the final marking while
reconstructing the trace exactly — even on nets and vocabularies never seen
in training?
*Why it matters:* legality is the minimum bar for any alignment to be usable;
a learned aligner that sometimes emits illegal output cannot back a
conformance verdict.

**RQ2 — Near-optimality.** How often does the learned candidate attain the
exact optimal cost, and when it does not, how large are the gaps?
*Why it matters:* every optimal candidate is certified for free by one cost
comparison; every gap bounds the extra work the exact layer must do — the
closer to optimal, the cheaper trust becomes.

**RQ3 — Ambiguity.** Can the model resolve what classical heuristics cannot
see locally: which of several identically labeled transitions an event should
synchronize with, and where invisible transitions must fire?
*Why it matters:* duplicate labels and silent routing are the principal
sources of search-space blowup in exact alignment, and the sub-problems where
learned context has the clearest comparative advantage.

**RQ4 — Generalization.** How does performance transfer across model
families, difficulty levels, scales, and from synthetic training data to
real-life event logs with unseen activity vocabularies?
*Why it matters:* the foundation-model premise — train once, align anywhere —
stands or falls with zero-shot transfer.

**RQ5 — Certification.** How often can the learned candidate be certified
optimal by cost equality against the exact backend, and what does
certification cost when repair is needed?
*Why it matters:* certification is the mechanism that converts a heuristic
into a trustworthy tool; its hit rate determines how much of the exact
solver's burden the learned system actually removes.

A sixth question is implied by the methodology and answered by the ablation:

**RQ6 — Attribution.** Of the system's quality, how much is owed to the
*learned* scores rather than to the constrained search around them?
*Why it matters:* without this control, any claimed benefit of learning could
be an artifact of the decoder; with it, the learned contribution becomes a
measured, falsifiable quantity — and the unguided decoder becomes the
baseline that future models must beat.

## Answers in One Paragraph

On held-out synthetic data the system produces legal alignments for 100% of
traces and exactly optimal ones for 91.0%, certifying them by cost equality;
the learned contribution is concentrated precisely where classical heuristics
are blind — duplicate-label resolution (95.3% vs. 68.8% optimality on
duplicate-prefix representations). Legality transfers perfectly to stress
nets and, zero-shot, to two real-life logs (74.6–100% trace-weighted
optimality), where the fast path outpaces exact search on the
invisible-transition-heavy receipt models. On random stress nets, learned
decoding overtakes exact A* at size 40 for deviation rate 0.15 and size 20 for
deviation rate 0.35. What remains open is quality — not legality — at scale:
the unguided baseline shows that future checkpoints must improve the guided
quality curve, not merely the decoder's legality.
