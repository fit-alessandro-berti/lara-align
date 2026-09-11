# Final manuscript validation

Checked on 11 September 2026.

## Delivered manuscript

- `main.pdf`: 40 pages in the official MDPI Applied Sciences submission class.
- Nine main sections, 11 vector figures, 18 tables, and 34 cited references.
- Research questions and contributions appear explicitly in Section 2.
- Section 2 introduces the process and machine-learning concepts used later.
- Section 3 expands related work and distinguishes competing outputs and guarantees.
- Sections 4 and 5 pair the formal method with explanations of its purpose.
- Section 5 contains the complete data-generation and training account: behavior
  families, four motif pairs, language and structural checks, trace edits, exact
  supervision, family acceptance, full worked input tensors and targets,
  relabeling, corpus composition, all losses, and optimization settings.
- Section 6 describes the implementation and supports later screenshot insertion.
- No appendix or separate data-generation document exists in the journal project.
- The original manuscript, experimental code, data, and checkpoint are unchanged.

## Typesetting and artifact checks

`make` completes successfully. A separate build using only the manuscript's
LaTeX sources, bibliography, figures, and supplied class files also completes
successfully; its extracted PDF text and pagination match the delivered PDF.
All 40 final pages were rendered and visually reviewed. Figures and tables were
also examined at larger scale during revision. Labels, arrows, captions,
equations, and table columns fit within the intended layout.

All 98 labels are unique. All citations and cross-references resolve; every
bibliography entry is cited. There are no overfull boxes, missing references,
missing glyph reports, empty screenshot boxes, or publication-date/DOI
placeholders. The remaining build messages are the supplied class's redundant
xcolor-load warning and two harmless underfull text boxes.

Delivered PDF SHA-256:

```
0e1d18ec93a36217ebb23522ecbab2730a634a6c898c95c509dfd8b8f56194c8
```

## Numerical and data checks

`make evidence` completed successfully against the fixed checkpoint and existing
corpus. The saved results in `data/evidence.json` record these checks:

- All 3,072 stored teacher witnesses replay legally at their stated costs.
- Training, validation, and test contain 512, 128, and 128 families respectively;
  family IDs and seeds are disjoint across splits, and motifs are balanced.
- Every family has two representations and two observations, expanded into four
  rows. Paired representations share observations and exact teacher costs.
- Every stored language certificate has status `exact`. This checks the stored
  certificates; it is not a new enumeration of every net language.
- Fresh exact searches reproduce all 512 stored test optima.
- Guided and unguided candidate runs each return 512 legal results and reproduce
  466 and 445 optimal-cost candidates, respectively.
- The worked training row's hashes, feature matrices, compatibility mask, graph
  edges, and move targets match the manuscript. Running-example neural scores
  also reproduce the printed values.
- Family bootstrap intervals are recomputed with 10,000 resamples and seed 13.
- The five empirical plots regenerate from the included evidence and training CSV.

The expanded explanation records the implemented decoder depths, static-score
limitations, unused log head, experimental timing boundaries, exact-timeout
denominators, and the absence of a guided-quality gain on the real logs.
Historical benchmark timings are retained; this verification does not claim a
new training run or reproduction of the original timing environment.

Screenshots and personal author declarations can be supplied later as described
in `README.md`. No author approval, submission, or publication is asserted.
