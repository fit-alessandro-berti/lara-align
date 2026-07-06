# Preliminaries

This document collects the process-mining background needed to read the paper.
It fixes the notation used in `approach.md` and `assessment.md` and matches the
conventions implemented in the `lara_align` package.

## 1. Events, Traces, and Event Logs

Let $\mathcal{A}$ be a finite alphabet of *activity labels*. An *event* is an
occurrence of an activity; in XES-style logs the label of an event is stored
under the attribute key `concept:name`. A *trace*

$$\sigma = \langle a_1, a_2, \ldots, a_n \rangle \in \mathcal{A}^*$$

is a finite sequence of events describing one execution (case) of a process. An
*event log* $L$ is a multiset of traces. Because conformance checking treats
each trace independently, this paper considers the alignment of a single trace
against a model; log-level results follow by aggregation over the traces of the
log.

## 2. Petri Nets

A *Petri net* is a tuple $N = (P, T, F)$ where $P$ is a finite set of *places*,
$T$ is a finite set of *transitions* with $P \cap T = \emptyset$, and
$F \subseteq (P \times T) \cup (T \times P)$ is the *flow relation* (the arcs).
For a node $x \in P \cup T$, the *preset* is
$\bullet x = \{y \mid (y, x) \in F\}$ and the *postset* is
$x \bullet = \{y \mid (x, y) \in F\}$.

A *marking* $m : P \to \mathbb{N}$ assigns a number of *tokens* to each place.
A transition $t$ is *enabled* in $m$ iff every input place carries at least the
required tokens, i.e., $m(p) \geq 1$ for all $p \in \bullet t$ (weight-1 arcs;
the implementation supports integer arc weights). Firing an enabled transition
$t$ yields the marking $m'$ with

$$m'(p) = m(p) - |\{(p,t)\} \cap F| + |\{(t,p)\} \cap F|,$$

written $m \xrightarrow{t} m'$. A sequence
$\theta = \langle t_1, \ldots, t_k \rangle \in T^*$ is a *firing sequence* from
$m_0$ to $m_k$ if there exist markings with
$m_0 \xrightarrow{t_1} m_1 \xrightarrow{t_2} \cdots \xrightarrow{t_k} m_k$.

## 3. Labeled Nets, Invisible Transitions, and Duplicate Labels

A *labeled Petri net* extends $N$ with a labeling function

$$\lambda : T \to \mathcal{A} \cup \{\tau\},$$

where $\tau \notin \mathcal{A}$ denotes the *invisible* (silent) label.
Transitions with $\lambda(t) = \tau$ (in pm4py: `label = None`) represent
routing behavior that leaves no trace in the log — e.g., the silent join
transitions `tau_left_join` / `tau_right_join` used by our synthetic choice
nets. Two distinct transitions $t \neq t'$ may share a visible label,
$\lambda(t) = \lambda(t') \neq \tau$; these are *duplicate labels*. Both
phenomena are central to this paper: an observed event determines the label of
a matching transition, but not necessarily its *identity*, so alignment must
resolve label-to-transition ambiguity.

A *system net* (accepting Petri net) is a triple $SN = (N, m_0, m_f)$ with
initial marking $m_0$ and final marking $m_f$. Its language is the set of
visible label sequences of complete firing sequences,

$$\mathcal{L}(SN) = \{\, \lambda_v(\theta) \mid m_0 \xrightarrow{\theta} m_f \,\},$$

where $\lambda_v$ projects a firing sequence onto its non-$\tau$ labels.

## 4. Conformance Checking and Alignments

*Conformance checking* quantifies to what extent observed behavior ($\sigma$)
agrees with modeled behavior ($SN$). The de-facto standard technique is
*alignments* (Adriansyah et al.), which explain each observed trace in terms of
the model by pairing trace positions with model steps.

Let $\gg$ denote a "skip" symbol. A *move* is a pair
$(x, y) \in (\mathcal{A} \cup \{\gg\}) \times (T \cup \{\gg\})$, excluding
$(\gg, \gg)$, of one of three kinds:

| move kind | form | meaning |
|---|---|---|
| synchronous | $(a, t)$ with $\lambda(t) = a$ | log and model agree |
| log move | $(a, \gg)$ | event observed but not replayed in the model |
| model move | $(\gg, t)$ | model executes a step with no matching event |

An *alignment* of $\sigma$ and $SN$ is a sequence of moves

$$\gamma = \langle (x_1, y_1), \ldots, (x_k, y_k) \rangle$$

such that:

1. **Log projection.** The sequence of non-$\gg$ log components equals $\sigma$
   exactly.
2. **Model projection (replayability).** The sequence of non-$\gg$ model
   components $\theta$ is a firing sequence of $N$ from $m_0$ that reaches
   $m_f$.

We call an alignment satisfying both conditions *legal* (in the code:
`verify_alignment` checks both and reports a `VerificationResult`). Note that
moves carry the *transition identity* $t$, not just its label: with duplicate
labels, two alignments can have identical label projections yet fire different
transitions.

## 5. Cost Functions and Optimal Alignments

A *move cost function* $c$ assigns a non-negative cost to every move. The
standard (unit) cost function, which is also the default `CostModel` of the
implementation, is:

| move | cost |
|---|---|
| synchronous move $(a, t)$ | $0$ |
| log move $(a, \gg)$ | $1$ |
| model move $(\gg, t)$, $\lambda(t) \neq \tau$ | $1$ |
| invisible model move $(\gg, t)$, $\lambda(t) = \tau$ | $0$ |

The cost of an alignment is $c(\gamma) = \sum_{i=1}^{k} c(x_i, y_i)$. An
alignment $\gamma^*$ is *optimal* iff it has minimal cost among all legal
alignments of $\sigma$ and $SN$:

$$c(\gamma^*) = \min \{\, c(\gamma) \mid \gamma \text{ legal alignment of } \sigma, SN \,\} =: \delta(\sigma, SN).$$

Optimal alignments need not be unique; the optimal *cost* $\delta(\sigma, SN)$
is. The unit cost model gives cost an interpretable reading: it counts the
minimum number of deviations (skipped events plus missing events) needed to
explain the trace. Alignment costs also induce the standard *fitness* measure

$$\mathit{fitness}(\sigma, SN) = 1 - \frac{\delta(\sigma, SN)}{c_{\text{ref}}(\sigma, SN)},$$

where $c_{\text{ref}}$ is the cost of a worst-case reference alignment (all
log moves plus the cheapest complete model run).

## 6. Computing Optimal Alignments: Synchronous Product and A*

The classical construction builds the *synchronous product net*
$SN \otimes \sigma$: the trace is encoded as a linear "trace net" whose $i$-th
transition emits $a_i$; the product contains a copy of the model, a copy of the
trace net, and one synchronous transition for every pair $(a_i, t)$ with
$\lambda(t) = a_i$. Legal alignments of $\sigma$ and $SN$ correspond exactly to
complete firing sequences of the product, and move costs turn the reachability
graph of the product into a weighted directed graph.

Finding an optimal alignment is therefore a *shortest-path problem* from the
product's initial marking to its final marking. State-of-the-art exact solvers
use A* search guided by an admissible heuristic derived from the *state
equation* (marking equation): a linear-programming relaxation of reachability
whose optimal value never overestimates the true remaining cost. This is the
algorithm implemented by pm4py's state-equation A* aligner, which we use as
the exact backend and as the source of ground-truth labels.

Exact search is trustworthy but can be expensive: the reachable state space of
the synchronous product grows quickly with concurrency (interleavings), loops,
duplicate labels (branching over transition identities), and invisible
transitions (silent paths that must be explored). This cost motivates learned
guidance — while the final answer remains anchored to the exact layer.

## 7. Decision Problems Addressed by Learned Guidance

Within alignment computation, the concrete decisions a learned component can
inform are:

- **Label-to-transition resolution.** For an event $a$, which of the
  transitions in $\{t \mid \lambda(t) = a\}$ should fire (duplicate labels)?
- **Silent routing.** Where must invisible transitions fire to enable the next
  synchronous move or to reach $m_f$?
- **Deviation localization.** Which events are best explained as log moves,
  and which model parts must be traversed as model moves?
- **Decomposition.** Which regions of net and trace interact, so they can be
  aligned locally and recomposed?
- **Candidate ranking.** Which complete candidate alignments are promising
  enough to verify first?

## 8. Certification

A *certificate of optimality* for a candidate alignment $\gamma$ is a proof
that $c(\gamma) = \delta(\sigma, SN)$. Since the exact solver returns an
optimal cost, cost equality with an exact solution certifies a legal candidate:
if $\gamma$ is legal and $c(\gamma) = c(\gamma^*_{\text{exact}})$, then
$\gamma$ is optimal (even when $\gamma \neq \gamma^*_{\text{exact}}$, e.g., a
different optimal alignment). This is the certification rule used throughout
the paper: neural outputs are treated as *candidates* whose legality is checked
by replay and whose optimality is certified — or repaired — by the exact layer.

## 9. Notation Summary

| symbol | meaning |
|---|---|
| $\mathcal{A}$ | activity alphabet |
| $\sigma = \langle a_1, \ldots, a_n \rangle$ | observed trace |
| $N = (P, T, F)$ | Petri net (places, transitions, arcs) |
| $\lambda : T \to \mathcal{A} \cup \{\tau\}$ | transition labeling; $\tau$ = invisible |
| $m_0, m_f$ | initial and final marking |
| $SN = (N, m_0, m_f)$ | system net |
| $\gg$ | skip symbol in moves |
| $\gamma$ | alignment (sequence of moves) |
| $c(\gamma)$ | alignment cost under the unit cost model |
| $\delta(\sigma, SN)$ | optimal alignment cost |
| $\gamma^*$ | an optimal alignment |
