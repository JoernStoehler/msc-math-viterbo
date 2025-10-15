# Meeting Notes — Project Vision Sync

- Date: 2025-10-13 (Mon, afternoon)
- Type: meeting (paraphrased; no verbatim quotes)
- Participants: Academic Advisor (Kai Cieliebak), Project Owner/Maintainer (Jörn Stöhler)

## Summary

- The advisor aligned with the overall project vision and acknowledged current progress and next steps. We proceed with a data-driven exploration of 4D polytopes (and beyond), combining exact/approximate symplectic invariants with modern data science/ML workflows to search for structure and counterexamples (systolic ratio > 1), while validating against known theory and examples.

## Agreed Vision and Scope

- Approach: Implement literature-grounded and custom algorithms, leveraging “vibecoding” agents to accelerate implementation. Validate via hand-checked examples and property tests (e.g., invariances, transformation behavior, known relations/lemmas).
- Dataset: Generate a large corpus of 4D polytopes first (random; structured families like Lagrangian products; canonical examples such as 5‑gon × rotated 5‑gon), then scale to higher dimensions as feasible.
 - Dataset: Generate a large corpus of 4D polytopes first (random; structured families like Lagrangian products; canonical examples such as 5‑gon × rotated 5‑gon), then scale to higher dimensions as feasible. Plan to reach ~1e5 distinct base polytopes, and enrich via augmentations (Sp(2n)+translations; local gradient‑informed vertex moves preserving combinatorics; light noise with superfluous vertices and small jitters).
- Quantities: Compute and store symplectic and non‑symplectic quantities and pairwise relations (e.g., EHZ capacity/minimal action; volume; systolic ratio; geometry/topology similarity metrics, symplectomorphism‑invariant features).
- Algorithms: Start with PyTorch + C++ baselines to avoid MATLAB bottlenecks. Explore a specialized 4D algorithm that may sidestep hardness results proved for unbounded dimensions.
- Analysis/ML: Run PCA and topology/similarity analyses, search for large systolic ratio examples, perform gradient‑based refinement at fixed/variable vertex counts, and use NNs/RL for surrogate modeling and constructive search (e.g., learn faster predictors or synthesize polytopes with target properties). Also search for the simplest/smallest counterexample (minimal vertex count) per advisor suggestion.
- Philosophy: Focus on generic polytopes (by density/continuity arguments) rather than symmetry‑heavy cases, except for key families (e.g., central symmetry, Lagrangian products) where structure is informative.
- Parallel tracks: Continuous visualization, write‑ups, and conjecture formation based on empirical findings.
- Conjectures/math to probe: Viterbo’s conjecture (systolic ratio); Mahler’s conjecture for K × K^o (advisor mentioned possible recent results in 4D/6D to verify against our data).

## Next Steps (aligned)

- Complete verified, performant implementations for the core quantities (EHZ capacity, volume, systolic ratio). Notify the advisor when the first verified datapoints are ready and when performance permits building a large dataset within a ~1 CPU‑day budget.
- Build the initial 4D dataset and run the standard first‑pass analyses (search for large ratios, PCA, topology/similarity, gradient‑based refinement).
- Stage ML experiments (NN surrogates, RL construction) after initial signals and verification.

## Milestones (planned)

- Weeks −2..0 (past): Infrastructure across ML/C++ established; slow versions worked (unverified).
- Week of 2025‑10‑13 (“this week”): Verify and harden fast implementations; then C++ rewrite for hotspots; start dataset generation.
- Week of 2025‑10‑20 (“next week”): Produce a moderately large dataset; parallelize first‑pass experiments; gather results.
- Following weeks: Formulate conjectures, develop new algorithms if vertex counts limit discovery (need ~25 vertices for the known counterexample; exploit efficient paths for Lagrangian products, e.g., 5×5). Begin NN‑based experiments (student selects architectures; agent implements/runs, including generic hyperparameter sweeps).
- Thereafter: Thesis writeup and/or proofs if conjectures crystallize; deepen experiments.

## Initial Scope: Datasets, Tools, Codebases

- Datasets: 4D polytopes (random; Lagrangian products; canonical counterexamples); extend to higher‑D as feasible.
- Compute stack: PyTorch + C++ (pybind11) for speed; scikit‑learn + standard Python libs for analysis; internal library under `src/viterbo/` adhering to project conventions.
- Algorithms: Implement EHZ capacity (minimal action), volume, derived systolic ratio; topology/similarity metrics; gradient‑based refinement; exploratory surrogates (NNs) and constructive RL.

## Evaluation Criteria (updated)

- Correctness: Reproduce known values on hand‑crafted examples; property‑based tests of invariances/relations; cross‑check against literature examples.
- Performance: Achieve throughput to compute a “moderately large” 4D dataset of ~1e5 4D polytopes within ~1 CPU‑day (or a comparable budget via batching); C++ accelerations where Python/Torch bottlenecks appear.
- Discovery: Existence of new examples with systolic ratio ≥ 1; search coverage across families; ability to climb to local maxima via refinement.
- Surrogate quality: NN predictors improve throughput accuracy/speed on larger polytopes; constructive RL yields candidates matching targets.

## Risks and Constraints

- Rarity: Polytopes with systolic ratio > 1 may be rare; discovery may require larger datasets/targeted search.
- Complexity: Algorithms may be too slow for large vertex counts; need specialized 4D methods/surrogates to scale.

## Literature and Contacts

- Reeb orbits on surfaces (to understand where minimal action cycles may lie in generic settings) — advisor suggested revisiting this thread; integrate into the reading list and dataset checks.
- Cross‑check Mahler (4D/6D status) and align dataset‑based validations.

## Cadence

- Weekly reporting cadence continues.
- Next in‑person meeting: 2025‑10‑21 (Tue) 09:00.
- Student to send: (a) literature list mail this week; (b) a note once fast verified calculations and first datapoints are ready.

## Approvals / Permissions (to confirm)

- Attachment “Optimization on Symplectic Embeddings” appears to be a private preprint. Treat as non‑public; seek authors’ consent before redistribution/citation beyond private notes.
- Final thesis text: advisor sign‑off required.

## Clarifications (provided after meeting)

- Sharing: Owner is fine sharing results broadly; GitHub is used only by Owner and Codex agents. Advisor sees results in person or via email (attachments preferred; links acceptable).

## Additional Notes

- Advisor was positively surprised by progress and acknowledged the student’s deeper ML planning.

## Action Items

- Student: Finish verified, fast implementations for EHZ/volume; start C++ acceleration where needed — target this week.
- Student: Generate initial 4D dataset; run first‑pass analyses (search for large ratios, PCA, topology/similarity, gradient refinement) — target next week.
- Student: Send literature list email to advisor this week; notify advisor upon first verified datapoints and performance confirmation.
- Student: Prepare architecture choices for early NN surrogate experiments; agent to implement/run once green‑lit.
- Agent: Maintain project conventions (PyTorch‑first, math purity in `viterbo.math`, no silent dtype/device moves); keep notes updated under `mail/archive/`.

## Remaining Open Question

- Surrogates: define acceptable error/tolerance targets and validation protocol for NN predictors (given dataset target ~1e5 and augmentation plan).
