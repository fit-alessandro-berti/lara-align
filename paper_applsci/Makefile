.PHONY: all submission figures evidence clean

all:
	latexmk -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex

submission: all
	python3 -m zipfile -c build/submission.zip main.tex references.bib Definitions sections figures

figures:
	python3 scripts/make_figures.py

evidence:
	python3 scripts/collect_evidence.py

clean:
	latexmk -C -outdir=build main.tex
	rm -f build/submission.zip
