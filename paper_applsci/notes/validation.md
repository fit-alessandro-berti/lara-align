# Final manuscript validation

Checked on 11 September 2026.

## Delivered manuscript

- `build/main.pdf`: 41 pages in the official MDPI Applied Sciences submission class.
- `build/submission.zip`: self-contained LaTeX source archive for submission.
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

`make submission` completes successfully. The resulting source ZIP was extracted
into a new temporary directory and compiled independently with latexmk. Both
builds produce 41 pages with identical extracted text. Both final LaTeX and
BibTeX logs contain no errors, warnings, missing glyph reports, or overfull or
underfull boxes. All 41 final pages were rendered and visually reviewed; the
changed text, declarations, and bibliography were also checked at larger scale.
Labels, arrows, captions, equations, and table columns fit within the layout.

All 98 labels are unique. All citations and cross-references resolve; every
bibliography entry is cited. There are no empty screenshot boxes or dummy
publication dates or DOIs. Removing the redundant global `pdftex` option resolves
the xcolor warning; two prose edits resolve the underfull boxes. The supplied
MDPI class files are unmodified.

Build products are ignored by Git. The previously removed `main.pdf` remains
absent from this directory's root and is not restored to the repository.

Delivered PDF SHA-256:

```
f42f0bffe603dea9d56eaa5b0abe39e91f8f6d8da4ab68c7040ba567aa521622
```

## Bibliography verification

All 34 cited works were located and checked against primary records. The source
URLs and verified metadata for every entry are in `reference-metadata.json`.
The audit covers authors and their order, titles, publication years, venues or
source types, volumes, pages or article numbers where applicable, and persistent
links. Publisher and repository metadata were supplemented by publisher pages
and author-posted papers when a record was incomplete.

Corrections include book and proceedings metadata, chapter series volumes,
publication types, article identifiers, compound surnames, accents, and initials.
The six cited arXiv works are explicitly identified as preprints. Every entry
has a DOI or publisher URL, and the rendered bibliography was checked for name
formatting, edition wording, duplicated words, and link wrapping.

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

The exact reference corpus is now included in `data/reference_corpus.tar.gz`.
All five archived files match the checksums recorded in `data/artifacts.json`.
The separately hosted checkpoint was downloaded and its `best.pt` checksum
matched the model used for the reported evaluation.

An isolated reproduction used the packaged corpus, downloaded checkpoint, and
included result snapshot, without the original local benchmark JSON files. It
completed successfully and reproduced `data/evidence.json` exactly. Historical
comparison records use the included snapshot when their original files are
absent; their original source hashes remain recorded as provenance.

The expanded explanation records the implemented decoder depths, static-score
limitations, unused log head, experimental timing boundaries, exact-timeout
denominators, and the absence of a guided-quality gain on the real logs.
Historical benchmark timings are retained; this verification does not claim a
new training run or reproduction of the original timing environment.

## Author statements

Funding and the AI-use acknowledgment follow the original manuscript. The
contribution statement records supported software authorship and manuscript
review, and the conflict statement records the declared Celonis affiliation.
Ethics and consent statements identify the synthetic and published-data scope.
Data availability now distinguishes the packaged corpus from the separately
hosted checkpoint. No final author approval, submission, or publication is
asserted.
