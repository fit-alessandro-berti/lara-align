# Related Work for `lara-align`

This document identifies related work for **Learned Alignment with Replay Assurance (LARA)**, an experimental system for neural-guided, certifying Petri-net trace alignment. The project sits at the intersection of alignment-based conformance checking, efficient exact search, approximate replay, process-mining tooling, and neural models for event-log/process-model data.

The key positioning is that LARA does **not** replace exact conformance checking with a neural network. Instead, it learns to propose and rank candidate alignments, while exact Petri-net replay and a pm4py state-equation A* backend remain responsible for legality checking, repair, and optimality certification.

## 1. Process Mining, Petri Nets, and Conformance Foundations

In [Murata1989], Petri nets are presented as a formalism for concurrent, asynchronous, distributed, parallel, nondeterministic, and stochastic systems. This is the formal background behind the project’s explicit treatment of places, transitions, markings, invisible transitions, and fireability. LARA inherits the Petri-net view that a candidate model projection must be executable from an initial marking to a final marking, not merely label-compatible with the trace.

In [Aalst1998], Petri nets are connected directly to workflow management, making them a natural representation for business processes. This line of work explains why conformance checking over Petri nets remains central in process mining: Petri nets provide both graphical intuition and executable semantics. LARA relies on these semantics in its verifier, especially when distinguishing synchronous moves, log-only moves, model-only moves, duplicate labels, and invisible transitions.

In [Aalst2016], process mining is organized around discovering, checking, and improving processes from event data. Conformance checking is the part of this pipeline that relates recorded behavior to modeled behavior. The book [Carmona2018] gives a focused treatment of conformance checking as the discipline of relating process models and observed executions. LARA’s contribution should be read within this conformance-checking tradition: it uses learning to reduce the practical burden of alignment search, but it still evaluates candidates against the formal relation between log behavior and model behavior.

The early conformance-checking work in [Rozinat2008] emphasizes replaying observed behavior on process models and diagnosing deviations. As discussed in [Aalst2012], replay-based conformance checking creates a link between trace events and process-model elements, enabling both conformance measurement and performance analysis. LARA follows this idea, but it makes the transition identity problem more explicit: a label-level match is insufficient when several transitions share the same label or when invisible transitions are needed for a legal replay.

## 2. Alignment-Based Conformance Checking

The paper [Adriansyah2011] is one of the core references for cost-based alignment and fitness analysis. It introduced the now-standard view in which observed behavior and modeled behavior are related through synchronous moves, log-only moves, and model-only moves, each with a cost. This is exactly the move vocabulary used by LARA: a learned decoder proposes a sequence of moves, and the verifier checks that the log projection reconstructs the trace and that the model projection is fireable.

In [Adriansyah2014], alignments are developed into a comprehensive framework for relating observed and modeled behavior. This thesis is especially relevant to LARA because it frames alignments as more than scalar fitness values: they are diagnostic objects that show where the model and log agree or disagree. LARA preserves this diagnostic role, since its output is an alignment object rather than only a predicted cost or binary conformance label.

The paper [Zelst2018] formalizes the computation of alignments between event data and process models, including the shortest-path view over a synchronous product. This is the main exact-search setting that LARA tries to accelerate in practice. However, LARA’s learned scores are deliberately not treated as admissible proof obligations. They are used to generate promising legal candidates, while exact search is still used to certify optimality when needed.

Alignment-based conformance has also been extended beyond fitness. In [Adriansyah2013], alignments are used for precision checking, showing that the alignment concept can support several conformance dimensions. This matters for LARA because a neural-guided candidate generator could later be used not only for fitness-oriented alignments, but also as a building block for richer diagnostic measures that depend on reliable correspondences between log events and model behavior.

## 3. Exact Optimal Alignment Search and Search Acceleration

A* search, introduced in [Hart1968], is a foundational shortest-path method and underlies many optimal alignment algorithms. In the alignment setting, the main challenge is that the synchronous-product state space can grow quickly with concurrency, loops, long traces, duplicate labels, and invisible transitions. This is the computational pain point that motivates LARA.

The paper [Dongen2018] studies efficient alignment computation using the extended marking equation. This line is directly related to pm4py’s state-equation A* backend, which LARA uses as the exact source of truth for certification. The important conceptual distinction is that state-equation and marking-equation methods provide formal search guidance, whereas LARA’s neural model provides empirical guidance that must still be checked.

In [Zelst2017], alignment computation is experimentally tuned, showing that implementation choices and search parameters can materially affect runtime. The paper [Casas2024] continues the search-efficiency direction with REACH, an efficient optimal alignment-based conformance-checking tool that reduces the number of states explored by A*. These works are close to LARA’s motivation, but they remain algorithmic search optimizations rather than learned candidate-generation systems.

Recent complexity results further clarify why the problem is difficult. In the preprint [Schwanen2026], alignment computation is analyzed from a computational-complexity perspective. This supports the project’s premise that exact methods are trustworthy but may become practically expensive. LARA’s response is not to abandon exactness; it separates fast learned proposal from exact legality and optimality checks.

## 4. Decomposition, Recomposition, and Local Alignment

In [Aalst2013], Petri nets are decomposed for process mining so that large problems can be split into smaller ones. The paper [Munoz2014] introduces single-entry single-exit decomposed conformance checking, showing that structural decomposition can speed up conformance analysis while preserving useful diagnostics.

The recomposition problem is central to this project. In [Verbeek2016], local alignments are merged for decomposed replay. The paper [Lee2018] closes the loop on decomposed alignment-based conformance checking by studying how decomposed results can be recomposed and bounded. These works are particularly relevant to LARA because LARA’s architecture has a learned router, local experts, and a recomposer. The analogy is strong: both families of work try to exploit locality. The difference is that classical decomposition usually relies on structural properties of the model, while LARA learns trace-dependent latent regions that may or may not coincide with structural decompositions.

The paper [Reissner2020] studies scalable alignment using automata and S-components. This is another important reference for LARA because it shows that decomposition can trade global optimality guarantees for scalability in controlled ways. LARA makes a different trade: it lets the neural model produce a candidate quickly, then invokes exact replay and exact alignment as needed. Therefore, LARA’s learned decomposition-like behavior must be evaluated not only by speed but also by legality, cost gap, and certification rate.

## 5. Approximate, Token-Based, and Heuristic Conformance Checking

Token-based replay predates optimal alignment as a practical conformance-checking method. In [Rozinat2008], token replay is used to compare real behavior with process models and diagnose deviations. The paper [Berti2021] revisits token-based replay to improve scalability and diagnostics, explicitly addressing the practical limitations of exact alignment-based methods. This is highly relevant to LARA because both approaches respond to the same runtime pressure: exact optimal alignments are informative, but may be expensive on complex models or long logs.

The paper [Schuster2021] studies alignment approximation for process trees, while [Padro2022] computes alignments through relaxation labeling and local optimal search. These works occupy the same speed-accuracy space as LARA. The distinctive feature of LARA is the certification boundary: a learned or approximate candidate is useful only if its replay legality can be checked, its cost can be compared with an exact backend, and exact repair remains available when the candidate is not optimal.

For this project, approximate and heuristic replay work provides the most important baseline family. LARA should be compared against such methods using at least replayability rate, optimal-cost rate, cost-gap distribution, transition-sequence match, label-level match, and runtime. Because LARA currently trains on exact-labeled synthetic data, it should also be compared against simple non-neural heuristics for duplicate labels and invisible-transition insertion.

## 6. Multi-Perspective, Declarative, Object-Centric, and Context-Aware Alignments

The paper [DeLeoni2013] extends alignments to multi-perspective conformance checking using integer linear programming. This is relevant because real conformance problems often involve data, resources, and other attributes, not only control flow. LARA currently focuses on control-flow Petri-net alignments, but its graph/trace encoding architecture could later be extended with event attributes, transition guards, resource information, or data-aware costs.

The paper [DeLeoni2015] develops an alignment-based framework for declarative process models and event-log preprocessing. This line shows that alignments are not limited to procedural Petri nets. In [Acitelli2022], context-aware trace alignment is formulated with automated planning. These works are useful for positioning LARA as one point in a broader design space: the project uses Petri-net execution semantics and pm4py exact alignment, whereas other alignment frameworks rely on declarative constraints, ILP, or planning encodings.

In [Wil2020], object-centric Petri nets are introduced to handle event data where events may refer to multiple interacting objects rather than a single case identifier. The paper [Gianola2024] studies object-centric conformance alignments with synchronization, addressing conformance when multiple object identities interact. The paper [Berti2025] introduces CPN-Py, a Python-based tool for modeling and analyzing colored Petri nets, including token data. These works are not direct baselines for the present LARA prototype, but they identify natural future extensions: learned candidate generation for colored, data-aware, or object-centric Petri-net alignments.

## 7. Tool Support and Reproducibility

The ProM framework in [Dongen2005] established a major open platform for process-mining research. The work [Verbeek2011] discusses XES, XESame, and ProM 6, which are important for event-log interoperability and tooling. LARA belongs to this tool-oriented tradition in the sense that it is not only a theoretical proposal; it implements feature extraction, neural decoding, legality checking, exact alignment invocation, synthetic data generation, training, and evaluation scripts.

As discussed in [Berti2019], PM4Py bridges process mining and data science in Python, making it suitable for algorithmic experimentation and integration with libraries such as pandas, NumPy, SciPy, and scikit-learn. The paper [Berti2023] presents PM4Py as a mature Python process-mining library. LARA’s dependence on pm4py is therefore not incidental: pm4py supplies the exact alignment layer and process-mining data structures, while LARA adds neural candidate generation and a certification wrapper on top.

This tool-support literature also implies an evaluation expectation. LARA should expose outputs in standard process-mining terms: pm4py-compatible alignments, replay fitness/cost, diagnostic moves, and reproducible benchmark splits. Its synthetic-data generator is useful for controlled experiments, but broader comparison should eventually include public event logs and model collections used in the alignment-computation literature.

## 8. Neural Models in Process Mining

Neural process mining has mostly focused on prediction rather than post-hoc conformance alignment. In [Evermann2017], recurrent neural networks are used to predict process behavior. In [Tax2017], LSTM networks are used for predictive business process monitoring, including next-event and remaining-time prediction. The paper [Camargo2019] improves LSTM modeling of business processes. These works are relevant because they show that event traces can be represented and learned from, but they do not solve LARA’s exact problem: constructing a legal, replayable, costed alignment against an explicit Petri net.

Transformers are relevant to LARA’s trace encoder. In [Vaswani2017], the Transformer architecture is introduced using self-attention. The paper [Bukhsh2021] applies transformer-based modeling to predictive business process monitoring. LARA uses similar representation-learning ideas, but in a different task setting: the model must jointly encode a process model graph and an observed trace, then propose alignment moves that can survive formal replay checks.

Graph neural methods are relevant because LARA encodes the Petri net as a typed graph. In [Sommers2021], graph neural networks are used for process discovery, translating event-log information into Petri-net structures. In [Chiorrini2022], instance graphs and graph neural networks are used for next-activity prediction. In [Stierle2021], graph-based neural networks support relevance scoring of process activities. These works show that graph representations are useful in process-mining tasks. LARA differs by applying graph/trace neural encoding to conformance alignment rather than discovery, prediction, or performance explanation.

The survey [Genga2025] is especially close to LARA’s motivation because it studies artificial intelligence in conformance checking and identifies opportunities for AI to address open conformance-checking challenges. LARA can be positioned as a concrete instance of this research agenda: a learned component helps with the hard search/selection part of conformance checking, while exact replay and exact alignment remain responsible for correctness.

## 9. Neural-Guided Search and Learning-Augmented Optimization

Outside process mining, [Bengio2021] surveys machine learning for combinatorial optimization. This literature is relevant because optimal alignment is a combinatorial search problem, and many exact algorithms spend most of their time making search decisions. LARA fits this learning-augmented view: it learns from exact alignment labels to propose promising moves and regions.

The paper [Yonetani2021] introduces Neural A* for path planning, where neural guidance is integrated with search. This is conceptually close to LARA’s long-term direction, because alignments can also be seen as shortest paths in a synchronous-product search space. The present LARA prototype, however, is more conservative than a fully neural-guided A*: it uses a greedy candidate decoder and then verifies/certifies the result with an exact backend. A natural future extension would be to use neural scores inside a bounded or exact search procedure while preserving admissibility or retaining a separate certificate.

The main risk in neural-guided conformance checking is overtrusting learned scores. LARA explicitly avoids this risk by treating the neural model as a proposal mechanism rather than a proof mechanism. This design makes it easier to compare LARA with classical exact methods: if the candidate cost equals the exact pm4py cost and the replay verifier accepts the alignment, the candidate can be certified optimal; otherwise, exact repair is still available.

## 10. Positioning of LARA Against the Literature

The closest related-work clusters are exact Petri-net alignments [Adriansyah2011], [Adriansyah2014], [Zelst2018], efficient A*/state-equation methods [Dongen2018], [Casas2024], decomposition and recomposition [Aalst2013], [Munoz2014], [Lee2018], approximate replay [Berti2021], [Padro2022], and neural process-mining representation learning [Evermann2017], [Tax2017], [Bukhsh2021], [Sommers2021].

LARA’s novelty is best stated as a hybrid systems contribution:

1. It learns graph/trace representations for Petri-net alignment candidate generation.
2. It explicitly predicts transition identity rather than only activity labels, which matters for duplicate-labeled transitions.
3. It treats invisible transitions as part of replay legality rather than as a post-hoc label artifact.
4. It uses learned latent regions and local experts in a way that resembles decomposition, but without assuming a fixed structural decomposition.
5. It keeps a formal verifier and exact pm4py alignment backend in the loop, so neural output is guidance rather than a certificate.
6. It evaluates both fast-mode behavior and certified-mode behavior, making legality, cost gap, and exact repair central metrics.

The main gap that LARA addresses is therefore not “alignment computation” in general, which is already well established. The gap is **certifiable neural guidance for Petri-net trace alignment**: using learning to propose strong candidates while preserving the trust model of exact conformance checking.

## BIBLIOGRAPHY

[Aalst1998] Wil M.P. van der Aalst, "The Application of Petri Nets to Workflow Management," *Journal of Circuits, Systems and Computers*, 8(1), pp. 21–66, 1998. DOI: 10.1142/S0218126698000043.

[Aalst2012] Wil M.P. van der Aalst, Arya Adriansyah, and Boudewijn F. van Dongen, "Replaying History on Process Models for Conformance Checking and Performance Analysis," *WIREs Data Mining and Knowledge Discovery*, 2(2), pp. 182–192, 2012. DOI: 10.1002/widm.1045.

[Aalst2013] Wil M.P. van der Aalst, "Decomposing Petri Nets for Process Mining: A Generic Approach," *Distributed and Parallel Databases*, 31(4), pp. 471–507, 2013. DOI: 10.1007/s10619-013-7127-5.

[Aalst2016] Wil M.P. van der Aalst, *Process Mining: Data Science in Action*, 2nd ed., Springer, 2016. DOI: 10.1007/978-3-662-49851-4.

[Acitelli2022] Giacomo Acitelli, Marco Angelini, Silvia Bonomi, Fabrizio M. Maggi, Andrea Marrella, and Alessandro Palma, "Context-Aware Trace Alignment with Automated Planning," *Proceedings of the 4th International Conference on Process Mining (ICPM 2022)*, IEEE, pp. 104–111, 2022. DOI: 10.1109/ICPM57379.2022.9980649.

[Adriansyah2011] Arya Adriansyah, Boudewijn F. van Dongen, and Wil M.P. van der Aalst, "Conformance Checking Using Cost-Based Fitness Analysis," *Proceedings of the 15th IEEE International Enterprise Distributed Object Computing Conference (EDOC 2011)*, IEEE, pp. 55–64, 2011. DOI: 10.1109/EDOC.2011.12.

[Adriansyah2013] Arya Adriansyah, Jorge Munoz-Gama, Josep Carmona, Boudewijn F. van Dongen, and Wil M.P. van der Aalst, "Alignment Based Precision Checking," in *Business Process Management Workshops*, LNBIP 132, Springer, pp. 137–149, 2013. DOI: 10.1007/978-3-642-36285-9_15.

[Adriansyah2014] Arya Adriansyah, *Aligning Observed and Modeled Behavior*, PhD thesis, Eindhoven University of Technology, 2014. DOI: 10.6100/IR770080.

[Bengio2021] Yoshua Bengio, Andrea Lodi, and Antoine Prouvost, "Machine Learning for Combinatorial Optimization: A Methodological Tour d’Horizon," *European Journal of Operational Research*, 290(2), pp. 405–421, 2021. DOI: 10.1016/j.ejor.2020.07.063.

[Berti2019] Alessandro Berti, Sebastiaan J. van Zelst, and Wil M.P. van der Aalst, "Process Mining for Python (PM4Py): Bridging the Gap Between Process- and Data Science," arXiv:1905.06169, 2019.

[Berti2021] Alessandro Berti and Wil M.P. van der Aalst, "A Novel Token-Based Replay Technique to Speed Up Conformance Checking and Process Enhancement," *Transactions on Petri Nets and Other Models of Concurrency XV*, LNCS 12530, Springer, pp. 1–26, 2021. DOI: 10.1007/978-3-662-63079-2_1.

[Berti2023] Alessandro Berti, Sebastiaan van Zelst, and Daniel Schuster, "PM4Py: A Process Mining Library for Python," *Software Impacts*, 17, article 100556, 2023. DOI: 10.1016/j.simpa.2023.100556.

[Berti2025] Alessandro Berti and Wil M.P. van der Aalst, "CPN-Py: A Python-Based Tool for Modeling and Analyzing Colored Petri Nets," arXiv:2506.12238, 2025.

[Bukhsh2021] Zaharah A. Bukhsh, Aaqib Saeed, and Remco M. Dijkman, "ProcessTransformer: Predictive Business Process Monitoring with Transformer Network," arXiv:2104.00721, 2021.

[Camargo2019] Manuel Camargo, Marlon Dumas, and Oscar González-Rojas, "Learning Accurate LSTM Models of Business Processes," in *Business Process Management*, LNCS 11675, Springer, pp. 286–302, 2019.

[Carmona2018] Josep Carmona, Boudewijn van Dongen, Andreas Solti, and Matthias Weidlich, *Conformance Checking: Relating Processes and Models*, Springer, 2018. DOI: 10.1007/978-3-319-99414-7.

[Casas2024] Jacobo Casas-Ramos, Manuel Mucientes, and Manuel Lama, "REACH: Researching Efficient Alignment-Based Conformance Checking," *Expert Systems with Applications*, 241, article 122467, 2024. DOI: 10.1016/j.eswa.2023.122467.

[Chiorrini2022] Andrea Chiorrini, Claudia Diamantini, Alex Mircoli, and Domenico Potena, "Exploiting Instance Graphs and Graph Neural Networks for Next Activity Prediction," in *Process Mining Workshops: ICPM 2021*, LNBIP 433, Springer, 2022. DOI: 10.1007/978-3-030-98581-3_9.

[DeLeoni2013] Massimiliano de Leoni and Wil M.P. van der Aalst, "Aligning Event Logs and Process Models for Multi-Perspective Conformance Checking: An Approach Based on Integer Linear Programming," in *Business Process Management*, LNCS 8094, Springer, pp. 113–129, 2013. DOI: 10.1007/978-3-642-40176-3_10.

[DeLeoni2015] Massimiliano de Leoni, Fabrizio M. Maggi, and Wil M.P. van der Aalst, "An Alignment-Based Framework to Check the Conformance of Declarative Process Models and to Preprocess Event-Log Data," *Information Systems*, 47, pp. 258–277, 2015. DOI: 10.1016/j.is.2013.12.005.

[Dongen2005] Boudewijn F. van Dongen, Ana Karla A. de Medeiros, H.M.W. Verbeek, A.J.M.M. Weijters, and Wil M.P. van der Aalst, "The ProM Framework: A New Era in Process Mining Tool Support," in *Applications and Theory of Petri Nets 2005*, LNCS 3536, Springer, pp. 444–454, 2005. DOI: 10.1007/11494744_25.

[Dongen2018] Boudewijn F. van Dongen, "Efficiently Computing Alignments: Using the Extended Marking Equation," in *Business Process Management*, LNCS 11080, Springer, pp. 197–214, 2018. DOI: 10.1007/978-3-319-98648-7_12.

[Evermann2017] Joerg Evermann, Jana-Rebecca Rehse, and Peter Fettke, "Predicting Process Behaviour Using Deep Learning," *Decision Support Systems*, 100, pp. 129–140, 2017. DOI: 10.1016/j.dss.2017.04.003.

[Genga2025] Laura Genga and Karolin Winter, "Artificial Intelligence in Conformance Checking: State of the Art and Research Agenda," *Process Science*, 2, article 9, 2025. DOI: 10.1007/s44311-025-00015-7.

[Gianola2024] Alessandro Gianola, Marco Montali, and Sarah Winkler, "Object-Centric Conformance Alignments with Synchronization," in *Application and Theory of Petri Nets and Concurrency*, Springer, 2024. DOI: 10.1007/978-3-031-61057-8_1.

[Hart1968] Peter E. Hart, Nils J. Nilsson, and Bertram Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," *IEEE Transactions on Systems Science and Cybernetics*, 4(2), pp. 100–107, 1968. DOI: 10.1109/TSSC.1968.300136.

[Lee2018] Wai Lam Jonathan Lee, H.M.W. Verbeek, Jorge Munoz-Gama, Wil M.P. van der Aalst, and Marcos Sepúlveda, "Recomposing Conformance: Closing the Circle on Decomposed Alignment-Based Conformance Checking in Process Mining," *Information Sciences*, 466, pp. 55–91, 2018. DOI: 10.1016/j.ins.2018.07.026.

[Murata1989] Tadao Murata, "Petri Nets: Properties, Analysis and Applications," *Proceedings of the IEEE*, 77(4), pp. 541–580, 1989. DOI: 10.1109/5.24143.

[Munoz2014] Jorge Munoz-Gama, Josep Carmona, and Wil M.P. van der Aalst, "Single-Entry Single-Exit Decomposed Conformance Checking," *Information Systems*, 46, pp. 102–122, 2014. DOI: 10.1016/j.is.2014.04.003.

[Padro2022] Lluís Padró and Josep Carmona, "Computation of Alignments of Business Processes Through Relaxation Labeling and Local Optimal Search," *Information Systems*, 104, article 101703, 2022. DOI: 10.1016/j.is.2020.101703.

[Reissner2020] Daniel Reißner, Abel Armas-Cervantes, Raffaele Conforti, Marlon Dumas, Dirk Fahland, and Marcello La Rosa, "Scalable Alignment of Process Models and Event Logs: An Approach Based on Automata and S-Components," *Information Systems*, 94, article 101561, 2020. DOI: 10.1016/j.is.2020.101561.

[Rozinat2008] Anne Rozinat and Wil M.P. van der Aalst, "Conformance Checking of Processes Based on Monitoring Real Behavior," *Information Systems*, 33(1), pp. 64–95, 2008. DOI: 10.1016/j.is.2007.07.001.

[Schuster2021] Daniel Schuster, Sebastiaan J. van Zelst, and Wil M.P. van der Aalst, "Alignment Approximation for Process Trees," in *Process Mining Workshops: ICPM 2020*, LNBIP 406, Springer, pp. 247–259, 2021. DOI: 10.1007/978-3-030-72693-5_19.

[Schwanen2026] Christopher T. Schwanen, Wied Pakusa, and Wil M.P. van der Aalst, "Computational Complexity of Alignments," arXiv:2603.05331, 2026.

[Sommers2021] Dominique Sommers, Vlado Menkovski, and Dirk Fahland, "Process Discovery Using Graph Neural Networks," *Proceedings of the 3rd International Conference on Process Mining (ICPM 2021)*, IEEE, pp. 40–47, 2021. DOI: 10.1109/ICPM53251.2021.9576849.

[Stierle2021] Matthias Stierle, Sven Weinzierl, Maximilian Harl, and Martin Matzner, "A Technique for Determining Relevance Scores of Process Activities Using Graph-Based Neural Networks," *Decision Support Systems*, 144, article 113511, 2021. DOI: 10.1016/j.dss.2021.113511.

[Tax2017] Niek Tax, Ilya Verenich, Marcello La Rosa, and Marlon Dumas, "Predictive Business Process Monitoring with LSTM Neural Networks," in *Advanced Information Systems Engineering*, LNCS 10253, Springer, pp. 477–492, 2017. DOI: 10.1007/978-3-319-59536-8_30.

[Vaswani2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin, "Attention Is All You Need," *Advances in Neural Information Processing Systems 30*, 2017.

[Verbeek2011] H.M.W. Verbeek, J.C.A.M. Buijs, Boudewijn F. van Dongen, and Wil M.P. van der Aalst, "XES, XESame, and ProM 6," in *Information Systems Evolution*, LNBIP 72, Springer, pp. 60–75, 2011. DOI: 10.1007/978-3-642-17722-4_5.

[Verbeek2016] H.M.W. Verbeek and Wil M.P. van der Aalst, "Merging Alignments for Decomposed Replay," in *Application and Theory of Petri Nets and Concurrency*, LNCS 9698, Springer, pp. 219–239, 2016. DOI: 10.1007/978-3-319-39086-4_14.

[Wil2020] Wil M.P. van der Aalst and Alessandro Berti, "Discovering Object-Centric Petri Nets," *Fundamenta Informaticae*, 175(1–4), pp. 1–40, 2020. DOI: 10.3233/FI-2020-1946.

[Yonetani2021] Ryo Yonetani, Tatsunori Taniai, Mohammadamin Barekatain, Mai Nishimura, and Asako Kanezaki, "Path Planning Using Neural A* Search," *Proceedings of the 38th International Conference on Machine Learning (ICML 2021)*, PMLR 139, 2021.

[Zelst2017] Sebastiaan J. van Zelst, Alfredo Bolt, and Boudewijn F. van Dongen, "Tuning Alignment Computation: An Experimental Evaluation," *Proceedings of the International Workshop on Algorithms & Theories for the Analysis of Event Data (ATAED 2017)*, CEUR-WS Vol. 1847, pp. 6–20, 2017.

[Zelst2018] Sebastiaan J. van Zelst, Alfredo Bolt, and Boudewijn F. van Dongen, "Computing Alignments of Event Data and Process Models," *Transactions on Petri Nets and Other Models of Concurrency XIII*, LNCS 11090, Springer, pp. 1–26, 2018. DOI: 10.1007/978-3-662-58381-4_1.
