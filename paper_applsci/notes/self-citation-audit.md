# Editorial self-citation revision (29 September 2026)

The editor requested a significant reduction in references authored by either
manuscript author or the PADS research group. The request was relayed by the
user and is identified that way in Comment E.1 of the response letter. Its
wording is a summary of that request, not a newly recovered editorial letter.

The comparison starts from Git commit `c1baccf`. Counts cover the manuscript
and its included sections and figures, excluding the response letter and
historical audit notes. A distinct reference is one cited bibliography entry.
An occurrence is one cited key at one location; a grouped citation contributes
one occurrence for each key.

| Measure | Before | Revised |
| --- | ---: | ---: |
| Distinct bibliography entries, all cited | 42 | 33 |
| References with either manuscript author | 13 | 5 |
| Additional PADS references without either manuscript author | 1 | 0 |
| Total author/PADS references | 14 | 5 |
| Author/PADS share of bibliography | 33.3% | 15.2% |
| Author/PADS citation occurrences | 17 | 6 |
| All citation occurrences | 59 | 51 |

Nine of fourteen author/PADS references were removed, a 64.3% reduction.
Eleven of seventeen author/PADS citation occurrences were removed, a 64.7%
reduction. No new bibliography entries were added; all 28 independent entries
remain. The five retained entries are needed to identify the underlying
alignment formulation, software, discovery algorithms, and evaluated baseline.

## Classification and decisions

The manuscript authors are Alessandro Berti and Wil M. P. van der Aalst.
Their presence anywhere in an author list qualifies a reference. PADS
membership is also considered independently of manuscript authorship:
`Zelst2018` qualifies through Sebastiaan van Zelst. The
[official RWTH PADS directory](https://rwthcontacts.rwth-aachen.de/organization/ORG-95GKM)
lists him alongside Daniel Schuster, Sander Leemans, and Wil van der Aalst.
Coauthorship on another publication or affiliation with TU Eindhoven alone
does not make an author a PADS member.

| Reference key | Before / after occurrences | Decision |
| --- | ---: | --- |
| `Aalst2016` | 1 / 0 | General introduction now uses the existing independent conformance-checking textbook. |
| `Adriansyah2011` | 2 / 1 | Retained once in Section 3.1 for the original cost-based alignment formulation; removed the duplicate introductory citation. |
| `Zelst2018` | 2 / 0 | Section 2.3 cites Chapter 7 of Carmona et al.; Section 3.2 points back to that explanation. |
| `Schwanen2026` | 1 / 0 | Removed the peripheral complexity-preprint discussion and its hardness claims; the replacement refers to the manuscript's own timing evidence. |
| `Aalst2013` | 1 / 0 | Removed the specific decomposition-method discussion; a shorter taxonomy is attributed explicitly to the independent survey. |
| `Lee2018` | 1 / 0 | Removed the specific recomposition-method discussion; no method-specific claim remains without its source. |
| `Berti2023` | 1 / 1 | Retained in Section 6.1 to credit PM4Py, used throughout the implementation and experiments. |
| `Leemans2013` | 1 / 1 | Retained in Section 7.7 for the original Inductive Miner at noise threshold zero. |
| `Leemans2014` | 1 / 1 | Retained separately in Section 7.7 for the infrequent-behavior variant at positive noise threshold. |
| `FaniSani2020` | 2 / 2 | Retained in Sections 3.3 and 7.6 to describe and identify the subset-selection/edit-distance baseline actually evaluated. |
| `Rozinat2008` | 1 / 0 | Replaced historical token-replay discussion with a general comparison supported by Chapter 4 of Carmona et al. |
| `Berti2021` | 1 / 0 | Removed the peripheral discussion of the authors' token-replay implementation. |
| `Schuster2021` | 1 / 0 | Removed the peripheral process-tree approximation discussion; tandem repeats remain attributed to the independent source. |
| `Munoz2014` | 1 / 0 | Removed the specific SESE decomposition discussion; the survey supports the remaining general taxonomy. |

## Source support

The existing independent textbook *Conformance Checking: Relating Processes
and Models* supports the introduction. Its publisher identifies token replay
and alignment in [Chapter 4](https://link.springer.com/chapter/10.1007/978-3-319-99414-7_4)
and synchronous-product shortest-path search in
[Chapter 7](https://link.springer.com/chapter/10.1007/978-3-319-99414-7_7).
The manuscript now names those chapters in the relevant citations.

The shorter decomposition paragraph explicitly reports the taxonomy in
[Genga and Winter's survey](https://link.springer.com/article/10.1007/s44311-025-00015-7).
Its full text distinguishes partial alignments from global alignments obtained
by recomposition. The publisher lists both authors at TU Eindhoven.
The existing [extended marking equation paper](https://link.springer.com/chapter/10.1007/978-3-319-98648-7_12)
continues to support exact-search guidance. No removed hardness, implementation,
or specific decomposition result is reassigned to an unrelated source.

`reference-metadata.json` records these support checks and the counting rule.
The historical metadata of each removed entry remains available, with its
previous citation context separated from its current removed status.

## Delivery and verification

The introduction, Section 2.3, and Sections 3.1 to 3.3 are revised in dark blue.
The TXT response contains Comment E.1 and its quantified reply; the renderer
produces the matching TEX. Space is reserved before editor/reviewer headings
to keep them with the following assessment after the extra reply is inserted.

`make submission` rebuilds the 52-page manuscript, seven-page response,
manuscript source ZIP, and evidence ZIP. All 33 references are cited and
resolve. The nine removed keys appear in neither the active manuscript nor
its bibliography. All existing equations, figures, tables, results, and
declarations are preserved. No experiment or training run is needed for this
editorial change.

The LaTeX/BibTeX checks find no unresolved citations or cross-references,
missing glyphs, or overfull/underfull boxes. The supplied MDPI logo generates
one pdfTeX version warning (PDF 1.7 asset, PDF 1.5 output); the logo and class
files are unmodified. Artifact counts and hashes are refreshed in
`data/revision_artifact_audit.json`.
