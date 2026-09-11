.PHONY: all figures evidence clean

all: main.pdf

main.pdf: main.tex references.bib Definitions/mdpi.cls Definitions/mdpi.bst Definitions/journalnames.tex $(wildcard Definitions/*.pdf) $(wildcard sections/*.tex) $(wildcard figures/*.tex) $(wildcard figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures:
	python3 scripts/make_figures.py

evidence:
	python3 scripts/collect_evidence.py

clean:
	latexmk -c main.tex
