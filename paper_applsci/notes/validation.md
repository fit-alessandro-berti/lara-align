# Final manuscript validation

## Machine learning preliminaries and availability revision (14 September 2026)

- Section 2.4 is titled "Machine Learning Preliminaries". It introduces input
  features and tensors, embeddings and encoders, layers and prediction heads,
  graph message passing, attention, transformers, normalization, masks, logits,
  calibration, losses, auxiliary regions, data splits, optimization, and
  regularization before their detailed use. Section 2.5 retains the explanations
  of pretraining, fine tuning, zero shot reuse, and distribution shift.
- First-use explanations also cover recurrent models in Section 3, unused
  auxiliary outputs in Section 5, family bootstrap intervals in Section 7, and
  beam search in Section 8. The numerical method and results are unchanged.
- Six primary references support further reading on deep learning, GELU, layer
  normalization, calibration, label smoothing, and dropout. All 40 bibliography
  entries are cited and have source records in `reference-metadata.json`. The
  six added records were verified on 14 September; the earlier audit remains
  dated 11 September 2026.
- Section 6 links to https://lara-pm.app and has one footnote with the author's
  exact HTTP download URLs for the data and trained checkpoints. The data
  availability statement also names the public application and points to the
  downloads. All three endpoints returned HTTP 200 when checked on 14 September;
  the three exact URLs are clickable in the PDF. The footnote appears on page 28.
- `make submission` succeeds, producing a 45-page `build/main.pdf` and a valid
  `build/submission.zip`. All 45 archived files match the working sources.
- Pages 6--9, 28, 42, and 43 were rendered and visually reviewed. The preliminaries,
  footnote, availability statement, acknowledgments, and added references fit
  the layout. The final LaTeX and BibTeX logs contain no unresolved references,
  missing glyphs, or overfull/underfull boxes. The only pdfTeX warning remains
  the supplied MDPI logo's PDF version (1.7 versus output 1.5).
- All 101 source labels are unique and all cross-references resolve. The 15
  figures and 17 tables retain starred floats at `[!t]` or `[!b]`. Both screenshot
  assets still match the supplied originals byte for byte. Figure 4 is now on
  page 17, and Figures 9 and 10 are on pages 30 and 31.
- Numerical experiments were not rerun for these manuscript-only changes.

Current PDF SHA-256:

```
99b420abfe2153594f474b5ef9f73dbe0ca4df87844f275cbc44588cc5a6abc5
```

The records below describe earlier versions. Their page counts, page positions,
and PDF hashes are historical; the checks above describe the current manuscript.

## Earlier manuscript revisions (14 September 2026)

- Figure 1 on page 2 now outlines the method: synthetic supervision, a reusable
  scorer, candidate decoding, independent replay, and optional exact certification.
  It contains no paper-section labels or reading-order arrows.
- The introduction explicitly states contributions C1--C4 and research questions
  RQ1--RQ3, consistently with their detailed treatment in Section 2. Its final
  paragraph begins "The rest of the paper is organized as follows" and describes
  Sections 2--9.
- Figure 4 on page 16 replaces the checkpoint-output table with the adapted
  complete move-score figure from `paper/`: Petri net, highlighted branch choice,
  decoded alignment, marking sequence, and full score arrays. Every score was
  checked against `data/evidence.json` to three decimals.
- Figures 9 and 10 on pages 29 and 30 show the Live Run and Trace Inspector.
  The PNG files match the supplied originals byte for byte. Both screenshots
  use the MDPI full-page figure width and are referenced in Section 6.
- The manuscript now has 15 figures (13 vector figures and two screenshots)
  and 17 tables. All use starred floats with `[!t]` or `[!b]` placement.
- `make submission` succeeds and produces the updated 43-page `build/main.pdf`
  and `build/submission.zip`. All archived manuscript and figure sources match
  the working files, and the ZIP passes its integrity check.
- Rendered pages 2--4 were reviewed for the introduction revision; pages 16,
  29, and 30 were reviewed for the score figure and screenshots. Labels, arrows,
  captions, and images fit the layout. There are no LaTeX
  warnings, unresolved references, missing glyphs, or overfull/underfull boxes.
  pdfTeX reports the supplied MDPI logo's PDF version (1.7 versus output 1.5).
- All 101 source labels are unique and their references resolve. Numerical
  experiments were not rerun for this editorial change.
- The acknowledgments use the author's requested two-sentence wording, verified
  in the extracted PDF text after rebuilding the PDF and submission archive.

Earlier PDF SHA-256:

```
3e6fde750fc5db6ddf8b730a55818d0b375dbdd42e9e16edd8c5c533c7b5a30d
```

The remaining record describes the earlier 41-page manuscript. Its page
positions, paper-outline description, PDF hash, and full-document visual review
are historical; the revisions above supersede those details.

Layout and editorial revision checked on 14 September 2026. Bibliography and
numerical verification were completed on 11 September 2026; their source records
and results are unchanged in this revision.

## Delivered manuscript

- `build/main.pdf`: 41 pages in the official MDPI Applied Sciences submission class.
- `build/submission.zip`: self-contained LaTeX source archive for submission.
- Nine main sections, 12 vector figures, 18 tables, and 34 cited references.
- Figure 1 provides a paper outline within the introduction, on page 2.
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

All 99 labels are unique. All citations and cross-references resolve; every
bibliography entry is cited. There are no empty screenshot boxes or dummy
publication dates or DOIs. Removing the redundant global `pdftex` option resolves
the xcolor warning; two prose edits resolve the underfull boxes. The supplied
MDPI class files are unmodified.

Build products are ignored by Git. The previously removed `main.pdf` remains
absent from this directory's root and is not restored to the repository.

Delivered PDF SHA-256:

```
26a369a8ffd6a8017aa9762b98aa8af13c93c8bc21595f7b3e37e45ea84762e9
```

## Requested editorial and layout changes

1. All 12 numbered figures use `figure*`, and all 18 numbered tables use
   `table*`. Twenty-nine floats use `[!t]`; the related-work comparison table
   uses `[!b]` near its first mention. The algorithm also uses `[!t]`.
2. Automatic section float barriers and the restriction against moving a float
   upward on its source page were removed. Existing figure widths, table type,
   and body type sizes are unchanged; no shrinking or negative space commands
   were introduced. The manuscript remains 41 pages with the added material.
3. Every numbered figure and table has an explicit reference in the prose,
   checked independently of captions. Missing scaling callouts were added,
   and the worked failure now has a callout in its own subsection.
4. The introduction contains an editable TikZ outline of Sections 2--9, with
   arrows showing the reading order. It appears before Section 2 begins.
5. The Method section opens with exactly three paragraphs covering the reused
   checkpoint, graph and trace encoding, move scoring, semantic decoding,
   independent replay, and optional exact certification. All three paragraphs
   appear together on page 11.
6. Each of the five Method subsections contains explicit Input, Processing, and
   Output blocks, in that order. Equations and the verification proposition
   retain their scientific meaning.
7. Manuscript prose and TikZ text contain no unqualified uses of network,
   learning, or mining. Generic compound phrases no longer have unnecessary
   hyphens. Formal terms such as non-free-choice, the big-endian convention,
   proper names, URLs, and code identifiers retain their required spelling.
   Published bibliography titles are unchanged.

For a quantitative spacing comparison against commit `6c1acca`, horizontal empty
bands at least 36 points high were measured across the text area, between page
coordinates 85 and 782 points. The title and final reference pages were excluded,
as was the template's left margin. The total height of those large bands fell
from 1199.6 to 116.0 points (90.3%); their count fell from seven to two, and the
largest fell from 388.9 to 72.5 points. The remaining spaces precede material that
needs to stay together, including a displayed matrix. The page images were also
reviewed to check float placement, legibility, and the new outline.

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
