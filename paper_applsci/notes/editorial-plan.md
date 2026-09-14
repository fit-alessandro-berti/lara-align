# Journal expansion: scope and evidence

Target: *Applied Sciences*, special issue [Process Mining: Theory and Applications](https://www.mdpi.com/journal/applsci/special_issues/87G1T64B1M).
Official class downloaded from https://mdpi-res.com/data/MDPI_template.zip on 2026-09-11.

The original `paper/` remains the source manuscript. The journal version is a separate, self-contained LaTeX project. It preserves the existing empirical study and does not present editorial work as new experiments.

Requirements:

- Explain the problem for process mining researchers without assuming machine learning background.
- Motivate every section, define all needed notation, and pair formal methodology with its purpose and limitations.
- State the research questions and contributions explicitly in the preliminaries.
- Expand related work with verified primary references and comparisons by task, guarantee, and reuse.
- Include a dedicated implementation section; leave an optional source insertion point for screenshots supplied later.
- Integrate all data generation and training explanations into the main article. No separate appendix or data-generation document in the journal version.
- Redesign vector illustrations and empirical charts for reading at the journal's natural column width.
- Compile the complete PDF, inspect all rendered pages, verify citations and cross-references, and check numerical claims against stored artifacts.

Evidence discovered during inspection:

- `runs/lara/test_guided.json`: 466/512 candidates optimal; 512/512 legal. Certified mode invokes the exact backend on every case.
- `runs/lara/test_unguided.json`: 445/512 candidates optimal. Machine learning adds 21 optimal candidates, or 4.10 percentage points.
- Stress and real-log scripts use final model-search depth 64; the default decoder uses 32. Prefix depth is 8.
- The real-log guided and unguided quality values coincide. These data establish interface transfer, not an empirical learned-quality advantage.
- Synthetic family IDs are split before expansion; this is not a held-out-motif or globally deduplicated structural split.
- Current certification does not use the candidate as an incumbent in exact search. Exact timeouts retain only the available replay-verified upper bound.

Submission preparation restores the funding and AI-use statements supplied in the original manuscript. The contribution line records software authorship and manuscript review, and the conflict statement discloses the existing Celonis affiliation. Broader roles or claims about funder involvement are not inferred.

The revision of 14 September 2026 adds an introduction roadmap and a
three-paragraph Method overview, followed by explicit input, processing, and
output descriptions in all five Method subsections. Numbered figures and tables
use starred environments with explicit top or bottom placement and prose
callouts. Removing section float barriers reduces blank space without reducing
type or graphic sizes. The prose consistently uses neural network, machine
learning, and process mining; published reference titles and code identifiers
retain their exact wording.
