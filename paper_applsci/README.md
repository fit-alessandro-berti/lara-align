# Applied Sciences submission source

The journal manuscript contains nine main sections, 15 figures (13 vector
figures and two tool screenshots), 17 tables, and 34 verified references.
Data generation and training are integrated into
Section 5; there is no appendix or separate data-generation document.

The introduction includes a visual overview of the method, from synthetic
training to candidate construction, replay, and optional exact certification.
It explicitly states the four contributions and three research questions and
ends with a paragraph describing the organization of the paper. Section 4 begins
with a three-paragraph overview, and each of its five subsections identifies its
input, processing, and output. All numbered figures and tables use `figure*` or
`table*` with `[!t]` or `[!b]` placement and are mentioned in the prose.

The target is the Applied Sciences special issue
[Process Mining: Theory and Applications](https://www.mdpi.com/journal/applsci/special_issues/87G1T64B1M).
The project uses the official [MDPI LaTeX class](https://mdpi-res.com/data/MDPI_template.zip),
downloaded on 11 September 2026. The supplied class files are unmodified.

## Build and submission files

Run from this directory:

```bash
make submission
```

This creates `build/main.pdf` and `build/submission.zip`. The ZIP contains the
LaTeX source, bibliography, required class files, and figure assets. Upload the
PDF and source ZIP as the manuscript files. The PDF and ZIP are build products
and are excluded from Git; `main.pdf` is not recreated in this directory's root.
Use `make` to compile without packaging, and `make clean` to remove build output.

The build requires TeX Live with pdfLaTeX, BibTeX, latexmk, and the packages used
by the MDPI class. Python 3 packages the source ZIP. No image-generation service
is needed. The final LaTeX and BibTeX logs are checked for errors, warnings,
missing glyphs, and overfull or underfull boxes; see `notes/validation.md`.

Funding and the AI-use acknowledgment are based on the original manuscript.
The author contribution statement records software authorship and manuscript
review; the declared Celonis affiliation is included in the conflict statement.
Ethics and consent statements identify the study's synthetic and public-data
scope. Final author approval and the submission-system confirmations remain
with the corresponding author.

The Implementation section includes the supplied Live Run and Trace Inspector
screenshots through `figures/implementation_screenshots.tex`. Both original PNG
files are included in the source archive; see `figures/README.md`.

## Bibliography verification

`notes/reference-metadata.json` records primary sources for all 34 references.
Sources include publisher pages and metadata, DataCite dataset records, PMLR
and NeurIPS proceedings, and author-posted arXiv papers. The bibliography
preserves compound surnames and initials, identifies preprints explicitly,
and includes a DOI or persistent publisher URL for every entry.

## Reproducing figures

The numerical snapshot and training history are included. To redraw the five
empirical plots without rerunning experiments:

```bash
make figures
make
```

This requires Python, NumPy, and Matplotlib. Figures are exported as PDF and SVG;
the conceptual diagrams use editable TikZ source.

## Rechecking the reported results

`data/reference_corpus.tar.gz` contains the exact training, validation, and test
splits and their generation metadata. `data/artifacts.json` records checksums
for the corpus files and the reference checkpoint. The publicly distributed
checkpoint was downloaded and its `best.pt` hash matched the evaluated model.

From the root of a fresh repository checkout, recover the inputs with:

```bash
mkdir -p data
tar -xzf paper_applsci/data/reference_corpus.tar.gz -C data
curl --fail --location https://www.alessandroberti.it/checkpoint_lara_latest.tar.gz --output /tmp/lara-checkpoint.tar.gz
tar -xf /tmp/lara-checkpoint.tar.gz
make -C paper_applsci evidence
```

These extraction commands are intended for a fresh checkout; preserve existing experimental files when working in an
active research directory. Use the parent project's Python dependencies.

The verification checks all 3,072 teacher witnesses, family separation and
balance, paired observations and costs, 512 freshly recomputed exact test
costs, guided and unguided candidates, the worked example, and bootstrap
intervals. It reproduces 466 guided and 445 unguided optimal candidates.

Historical comparison results are read from the original files when present
and otherwise from the included snapshot. Their original file hashes are
preserved as provenance. Verification does not rerun those timing experiments
or claim a new training run. The entire check was also run in an isolated copy
using only the packaged corpus, downloaded checkpoint, and included snapshot;
its resulting evidence matched the supplied snapshot exactly.

Generation and experiment interfaces remain in the parent repository's
`scripts/` directory. Their settings and measurement boundaries are explained
in the manuscript. Public real-log sources are cited in Section 7.
