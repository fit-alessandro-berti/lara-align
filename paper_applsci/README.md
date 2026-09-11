# Expanded Applied Sciences manuscript

Open `main.pdf` to read the 40-page journal version, with 11 figures, 18 tables,
and 34 cited references. `main.tex` is the entry point for
the self-contained LaTeX project; the manuscript text is split into `sections/`.
The original `paper/` manuscript and its standalone data-generation document
are left untouched. The journal version integrates data generation and all
training losses into its main text and has no appendix or separate data paper.

The target is the Applied Sciences special issue
[Process Mining: Theory and Applications](https://www.mdpi.com/journal/applsci/special_issues/87G1T64B1M).
It uses the official MDPI class, downloaded from the
[MDPI template distribution](https://mdpi-res.com/data/MDPI_template.zip)
on 11 September 2026, with `applsci,article,submit` options. The class files in
`Definitions/` are supplied by MDPI and are not custom layout substitutes.

## Build

From this directory:

```bash
make
```

This requires a TeX Live installation with pdfLaTeX, BibTeX, latexmk, and the
packages loaded by the MDPI class. All figures and references needed for the
PDF are included. The source can also be uploaded as a folder to a LaTeX editor
with `main.tex` selected as the main document.

To regenerate only the result plots from the included numerical snapshot:

```bash
make figures
make
```

This requires Python, NumPy, and Matplotlib. Plots are exported as PDF and SVG.
The conceptual illustrations use editable TikZ sources and need no external
image service.

## Evidence and reproduction

`data/evidence.json` contains the original result summaries and stress/real-log
records, source hashes, a fresh check of all 512 guided/unguided test outputs,
512 recomputed exact test costs, the worked-example neural scores, and
family-bootstrap settings. The evidence script also checks family split
separation and balance, the stored language-certificate statuses, shared
observations and equal teacher costs within representation pairs, and the
replay of all 3,072 teacher witnesses. The original
benchmark timings remain historical measurements; they are not replaced by the
quality verification run. `data/metrics.csv` is the recorded training history.
See `notes/validation.md` for the final manuscript checks and their scope.

To recheck the fixed checkpoint, worked input tensors, and test-set quality in
the parent repository's Python environment:

```bash
make evidence
make figures
make
```

The original generation and experiment commands run from the repository root.
Use a new output directory when regenerating data or training so that the
reference artifacts remain available. The relevant command interfaces are:

```bash
python scripts/init_data.py --help
python scripts/train_model.py --help
python scripts/test_model.py --help
python scripts/benchmark_scaling.py --help
python scripts/benchmark_pm4py_approx.py --help
python scripts/benchmark_pm4py_approx_scaling.py --help
python scripts/evaluate_real_log.py --help
streamlit run streamlit_app.py
```

The paper explains the generation settings, losses, model configuration,
decoder depths, comparison parameters, and timing boundaries. Quality
verification and plot regeneration do not imply reproduction of the original
CPU timings. No new training run is claimed.

## Screenshots and author declarations

The implementation section is complete as prose. Screenshots can be added later
through `figures/implementation_screenshots.tex`; see `figures/README.md`.
There are no empty screenshot placeholders in the PDF.

Before actual journal submission, the authors should supply their funding and
APC declaration, CRediT contributions, conflicts of interest, any applicable
ethics statements, and final author-approved AI-use disclosure. These personal
declarations cannot be inferred from the repository and have not been invented.
The title-page authors, affiliations, and ORCIDs are inherited from the original
manuscript. No acceptance date or article DOI is asserted.

For the AI-use disclosure, the factual preparation record is: OpenAI Codex
assisted with restructuring and expanding the manuscript, explaining notation
and methodology, literature verification, editable figure code, numerical
cross-checks, and LaTeX preparation. The empirical results come from the existing
project artifacts and a fixed-checkpoint verification, not generated experimental
data. The authors can use this record to write the disclosure after reviewing
the manuscript; their review or approval is not asserted here.
