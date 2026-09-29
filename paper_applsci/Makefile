.PHONY: all submission responses figures evidence revision-evidence revision-tables verify-revision clean

all:
	latexmk -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex

responses:
	python3 scripts/render_responses.py
	latexmk -pdf -outdir=build -interaction=nonstopmode -halt-on-error reviewer_response.tex

submission: all responses
	python3 -m zipfile -c build/submission.zip main.tex references.bib Definitions sections figures
	python3 scripts/package_revision_evidence.py

figures:
	python3 scripts/make_figures.py

evidence:
	python3 scripts/collect_evidence.py
	python3 scripts/analyze_revision.py

revision-evidence:
	python3 scripts/summarize_revision.py
	python3 scripts/render_revision_tables.py
	python3 scripts/verify_revision_records.py

revision-tables:
	python3 scripts/render_revision_tables.py

verify-revision:
	python3 scripts/verify_revision_records.py

clean:
	latexmk -C -outdir=build main.tex
	latexmk -C -outdir=build reviewer_response.tex
	rm -f build/submission.zip
	rm -f build/revision_evidence.zip
