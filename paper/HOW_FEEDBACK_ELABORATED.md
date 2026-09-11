This note explains how each feedback point was addressed in [main.tex](main.tex) and [appendix.tex](appendix.tex).

1. **Explain what `sync(i,t)` and `model(t)` depend on and how they are computed.**

   The notation now includes the trained parameters, system net, labeling, and complete trace. The paper gives the equations that turn event and transition representations into scores. It also states that scores are computed once per input pair and remain fixed during decoding. The current marking affects enabledness checks, not the neural scores. Model scores have no event-position index.

   Location: main Section 4.2, Equation 1, and Algorithm 1; appendix Section B.1.

2. **Show complete score tables instead of selected scores.**

   Figure 2 now shows the full synchronous score matrix, model-move vector, and auxiliary log-move vector for the drawn net and trace. The values come from the epoch-49 checkpoint, with dropout disabled, and are rounded to three decimals. Masked entries are identified explicitly.

   Location: main Figure 2 and Section 4.2.

3. **Make the network inputs concrete.**

   The vague description of three input objects was replaced with an explicit tuple of tensors. The paper defines the place features, transition features, edge endpoints, edge types, label IDs, and compatibility matrix, including their dimensions. It distinguishes initial and final markings from intermediate decoder markings.

   Location: main Section 4.1; appendix Section A.2.

4. **Explain hashing and the 8,192 identifiers.**

   The exact hash formula is given, including the reserved ID zero for silent transitions. Numeric examples show how activity labels become IDs and select learned embeddings. Compatibility uses original label equality, so hash collisions cannot create a false label match. The appendix also shows a concrete example of training-time ID remapping.

   Location: main Sections 4.1 and 4.5; appendix Section A.2.

5. **Give an actual training example, including in the appendix.**

   Added the stored row `train-000048`. The main paper shows its net, observed trace, label IDs, exact alignment, optimal cost, and target vectors. The appendix provides the full feature matrices, edge specification, compatibility matrix, stored identifier order, and the edits that produced the observation. Inputs and supervision targets are clearly separated.

   Location: main Table 2; appendix Section A.2.

6. **Explain the training targets and loss more precisely.**

   Earlier loss descriptions were checked against the implementation. The revised text defines synchronous transition targets, per-event log targets, and per-transition model targets. A model target records whether a transition appears as a model move anywhere in the alignment, not its position or number of occurrences. The appendix gives all loss equations, weights, smoothing rules, masking rules, and optimization settings. It also explains which outputs the decoder uses.

   Location: main Section 4.5; appendix Sections A.8 and B.2 to B.4.

7. **Explain what a behavior family means.**

   A family is now defined as two nets with the same complete visible language and two shared observations. The four resulting net and trace pairs are listed explicitly. Each pair receives its own exact alignment. The appendix explains that keeping families within a split does not guarantee that all languages are different across splits.

   Location: main Section 4.4; appendix Section A.3.

8. **Explain the two fresh sequential activities.**

   The text now says that both representations append activity D followed by E, with labels absent from the core. It gives the resulting language explicitly. For example, AB and AC become ABDE and ACDE. This adds two events to each clean trace while preserving language equality between the paired nets.

   Location: main Section 4.4; appendix Section A.3 and Figure 6.

9. **Make data generation clearer in both files.**

   The main paper explains the representation pairs, shared observations, rejection checks, and exact labeling. The appendix connects these steps to a stored training row and its numeric inputs. It also clarifies that ambiguous rows lose synchronous identity supervision but keep their other targets. Replay remains a separate check, not a training loss.

   Location: main Sections 4.4 and 4.5; appendix Section A.

10. **Use the term "foundation model" and explain the role of abstract labels.**

    The term now appears in the abstract, introduction, keywords, and conclusion. It refers to one scorer pretrained across synthetic process structures and reused on unseen nets and vocabularies. The text explains that business meanings of activity names are not inputs. It also avoids claiming exact invariance to every relabeling or competence across all process mining tasks.

    Location: abstract, introduction, and conclusion.

11. **Replace "unpredictable" in the abstract and introduction.**

    The text now describes the concrete runtime issue: most exact alignments are fast, but a few difficult cases can dominate processing time and delay interactive use.

    Location: abstract and introduction.

12. **Do not present replay and certification as unique to LARA.**

    The claim about a distinctive approach to trust was replaced. The paper explicitly states that replay and exact certification can also be applied to classical approximate candidates. LARA's contribution is the reusable learned candidate scorer.

    Location: related work and discussion.

13. **Correct the terminology and vague wording.**

    "Symmetric presets" became "identical presets". The phrase about giving evidence to duplicates was replaced with an explanation of reverse edges: transitions receive messages from output places, so different outgoing branches can produce different representations. "Cross attention" became "cross-attention".

    Location: main Sections 4.1 and 4.2 and Figure 1.

14. **Improve the figure showing a gap of four.**

    The figure was enlarged for readability. Its caption now describes a suboptimal candidate with cost gap four and explains the event subscripts. It states the optimal cost of two and candidate cost of six, and identifies early model completion as the cause.

    Location: main Figure 5.

15. **Emphasize that approximation should target slow exact cases.**

    The conclusion now recommends learned or classical approximation for difficult cases, for example after an exact-search time budget. The paper states that this selection policy has not been evaluated and that the current certified mode still runs exact A* for every input. Alternative improvements remain possible, including joint move ranking, beam search, incumbent bounds, and method portfolios.

    Location: discussion and conclusion.

16. **Do not claim that only LARA enables progressive alignment services.**

    The conclusion now says that progressive services can combine any suitable candidate generator with later exact certification. LARA is presented as one pretrained option.

    Location: conclusion.

17. **Keep the assessment of the results honest.**

    The experimental result tables and plotted values were preserved. The text continues to acknowledge stronger classical approximations, limited gains from guidance, and the lack of a demonstrated speedup for certified mode. The new numerical examples were checked against the stored data and implementation.

    Location: assessment, discussion, and conclusion.

18. **Keep the paper at 13 pages and update the linked appendix PDF.**

    Repeated discussion and explanatory prose were shortened to make room for the concrete neural details. The compiled main paper remains exactly 13 pages, with the original font and margin settings. Both documents compile without warnings. The revised appendix is 12 pages, and `data_generation.pdf` is a recompilation of `paper/appendix.tex`, identical to `paper/appendix.pdf`.
