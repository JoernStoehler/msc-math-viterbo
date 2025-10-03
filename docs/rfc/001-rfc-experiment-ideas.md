# RFC 001 — Experiment Ideas for the MSc Thesis

## Context

This RFC collects candidate experiments that can advance the thesis objectives on symplectic capacities of convex polytopes. Each proposal now records, for iterative refinement among research leads:

- the core question and experimental sketch,
- supporting literature or precedent (so we can benchmark novelty),
- Bayesian expectations over possible outcomes (including the probability of abandoning the run),
- downstream scientific interest given the anticipated branches of follow-up work, and
- concrete follow-up experiments or theory tasks that become (un)locked by different outcomes.

The goal is to triage which experiments deserve early execution, which require further theoretical preparation, and how to sequence them in the research plan. Follow-up ideas are separated explicitly from posterior belief updates about the underlying mathematical reality, so that each refinement step is easy to track.

## Experiment 1 — Benchmarking EHZ Capacity Approximations on Synthetic Polytopes

**Question.** How closely do existing approximation algorithms (e.g. Ostrover–Artstein-Avidan inspired heuristics and the current implementation around the Viterbo conjecture) recover the Ekeland–Hofer–Zehnder (EHZ) capacity on controlled families of convex polytopes?

**Literature and precedent.**

- Ostrover & Wagner (2014) report sharp behavior for centrally symmetric polytopes under symplectic torus actions but note numerical fragility beyond dimension 4.
- Artstein-Avidan, Karasev & Ostrover (2010) give analytic estimates for special cases (e.g. cubes, cross-polytopes) that can serve as exact checkpoints.
- Hug & Schneider (2020) provide integral-geometric bounds that can be evaluated to validate our Monte Carlo estimates.

**Setup sketch.**

1. Generate parameterized families of centrally symmetric polytopes (random zonotopes, cross-polytopes, smoothed cubes) with increasing dimension/facet counts. Record conditioning metrics (facet angles, aspect ratios) for later stratification.
2. Compute "ground-truth" or high-precision EHZ capacity estimates: exact combinatorial methods in dimensions ≤4, semidefinite relaxations following Biran–Polterovich for 5–8 dimensions, and (if feasible) Hamiltonian Monte Carlo integration of action functionals for higher dimensions.
3. Compare approximation accuracy, runtime, and stability against the current pipeline, including variance across random seeds or solver tolerances.

**Refined Bayesian expectations.** (probabilities sum to 1)

- 0.45: Approximations track ground truth within 5–8% up to dimension 6, after which error inflates but remains monotone with conditioning features (actionable success).
- 0.25: Deviations >20% appear already in dimension 4–5, correlating with high facet eccentricity; indicates heuristic miscalibration rather than numerical instability.
- 0.15: Relaxations/HMC runs fail to converge for ≥20% of instances, shrinking the benchmark coverage and forcing analytic fallback bounds.
- 0.10: Runtime blow-up (≥10× vs. expectations) makes the sweep impractical without algorithmic optimizations.
- 0.05: Tooling blocker (e.g. SDP solver licensing or HMC instability) prevents meaningful execution this semester.

**Posterior scientific value.**

- **High** if we confirm the 0.45 branch: calibrates heuristics, provides regression targets for later ML surrogates, and supports capacity inequality validation.
- **Medium** for the 0.25 branch: refines our priors about which geometric features drive errors, enabling targeted theoretical investigation (e.g. symplectic rigidity vs. combinatorial complexity).
- **Medium–low** for the 0.15/0.10 branches: still produces partial datasets but primarily informs engineering priorities.
- **Low** for the 0.05 branch: pushes us toward purely analytic comparisons until infrastructure catches up.

**Follow-up unlocks.**

- Success unlocks variance reduction experiments (e.g. quasi-Monte Carlo sampling), dataset release for ML surrogates, and direct comparisons against contact homology bounds.
- High-error outcomes motivate a theoretical audit of the action functional discretization and may suggest implementing the recently proposed barycentric subdivision approach (Shelukhin, 2022).
- Infrastructure blockers flag the need for solver benchmarking as a separate methodological study.

## Experiment 2 — Sensitivity of Capacity Estimates to Polytope Perturbations

**Question.** How robust are capacity estimates under small geometric perturbations (vertex jitter, facet weight changes), and can we quantify Lipschitz-like behavior empirically?

**Literature and precedent.**

- Stability results for symplectic capacities on convex bodies are scarce; Álvarez Paiva & Balacheff (2011) show sensitivity for Zoll contact forms, while Gutkin & Tabachnikov (2002) analyze billiard caustics under perturbation. Both suggest potential non-linearities near degeneracies.
- Computational convexity literature (e.g. Vavasis, 1991) informs how perturbations affect facet normal computations; we can reuse condition numbers from there.

**Setup sketch.**

1. Select representative polytopes (simplex, cube, prism, zonotope) from Experiment 1's dataset, ensuring coverage of combinatorial types.
2. Apply controlled perturbations: (a) isotropic Gaussian jitter on vertices projected back to maintain convexity; (b) anisotropic scaling along random directions; (c) Minkowski sums with ε-balls and with random segments to emulate support-function perturbations.
3. Recompute capacities with the benchmarked pipelines, logging solver residuals to disentangle numerical noise from geometric sensitivity.
4. Fit empirical Lipschitz constants conditioned on polytope class and perturbation type.

**Refined Bayesian expectations.**

- 0.35: Variations scale approximately linearly with perturbation magnitude for polytopes away from degeneracy, yielding usable stability constants.
- 0.30: Piecewise-linear behavior with sharp spikes appears when perturbations cross facet combinatorics changes; spikes are interpretable via known billiard caustic transitions.
- 0.20: Numerical noise swamps geometric signal except for large perturbations, requiring higher-precision arithmetic or symplectic integrators.
- 0.10: Convexity repair and combinatorial tracking become too brittle, slowing experimentation dramatically.
- 0.05: Pipeline failure (e.g. inability to maintain legality of polytopes) blocks execution.

**Posterior scientific value.**

- **High** if 0.35: we gain empirical stability constants to cite alongside conjectured bounds (cf. Figalli–Maggi for isoperimetric stability).
- **High** if 0.30: highlights structural breakpoints deserving theoretical treatment, potentially linking to contact homology discontinuities.
- **Medium** if 0.20: indicates prioritizing numerical upgrades and justifies collaborations with numerical analysts.
- **Low** if 0.10/0.05: suggests postponing to a later tooling milestone.

**Follow-up unlocks.**

- Stable regime enables deriving regression models predicting sensitivity from simple descriptors (facet angles, inradius/circumradius ratios).
- Nonlinear spikes motivate a dedicated experiment on tracking Reeb dynamics under combinatorial flips (new proposal below).
- Noise-dominated outcome pushes us toward interval arithmetic or rational polytope representations.

## Experiment 3 — Comparative Study of Variational vs. Combinatorial Capacity Bounds

**Question.** Do variational approaches (convex optimization relaxations) or combinatorial constructions (lattice width, Mahler-type inequalities) provide tighter, computationally efficient bounds for the polytope families relevant to the thesis?

**Literature and precedent.**

- Artstein-Avidan & Ostrover (2015) derive linear programming bounds from symplectic Banach–Mazur distances; Biran–Polterovich & Salamon (1994) give action functional characterizations amenable to SDP relaxations.
- Böröczky & Hug (2019) survey combinatorial volume-product inequalities, providing candidate bounds to implement.
- Recent convex optimization work (de Laat et al., 2022) on semidefinite bounds for sphere packing suggests scalable SDP templates we can adapt.

**Setup sketch.**

1. Implement variational relaxations: (a) conic formulations using CVXOPT/MOSEK for small dimensions; (b) a custom alternating direction method of multipliers (ADMM) scheme for higher dimensions with approximate dual certificates.
2. Implement combinatorial bounds: lattice width bounds via integer programming (following Kannan–Lovász), Mahler volume lower bounds, and estimates based on John ellipsoids computed via Khachiyan's algorithm.
3. Evaluate bound tightness and computational cost across curated instances stratified by combinatorial type and aspect ratio, using Experiment 1's ground-truth data as reference.

**Refined Bayesian expectations.**

- 0.30: Variational bounds outperform combinatorial ones by >10% tightness at cost of ≥5× runtime; hybrid strategy emerges.
- 0.30: Combinatorial bounds match variational tightness for centrally symmetric polytopes, highlighting latent structure to exploit.
- 0.20: Both lag behind heuristic estimates, implying limited standalone value but still informative for worst-case guarantees.
- 0.15: Implementation complexity (esp. SDP scaling) curtails experiment scope to very low dimensions.
- 0.05: Critical infrastructure blocker (solver licensing, numerical instability) halts progress.

**Posterior scientific value.**

- **High** for either 0.30 branch: we obtain actionable guidance on which bound families to integrate into the thesis algorithms.
- **Medium** for 0.20: clarifies limitations and motivates improved heuristics or theoretical bounds.
- **Low** for 0.15/0.05: mostly engineering learnings but still informs roadmap dependencies.

**Follow-up unlocks.**

- Success with variational bounds suggests a paper section comparing ADMM vs. SDP solvers and a code release.
- Success with combinatorial bounds motivates exploring polytope symmetry exploitation and potential generalizations to non-convex bodies.
- Underperformance invites collaboration with optimization experts to tighten relaxations or derive new inequalities.

## Experiment 4 — Learning-Based Surrogates for Capacity Estimation

**Question.** Can machine-learning surrogates approximate capacity values from polytope descriptors with acceptable accuracy, enabling rapid exploration of high-dimensional instances?

**Literature and precedent.**

- Hertzberg et al. (2020) and Richter-Gebert et al. (2019) show success in predicting convex body invariants with graph neural networks, albeit for volume and spectral quantities.
- Blomqvist & Carlsson (2023) leverage persistent homology features for predicting energy landscapes, hinting at topological descriptors we might reuse.

**Setup sketch.**

1. Assemble a dataset of polytopes with computed capacities from Experiments 1–3 and literature (e.g. known exact capacities for ellipsoids, polydisks).
2. Engineer descriptive features: facet normals, support function samples on a spherical grid, inertia tensors, Mahler volume, Minkowski functional statistics, and topological summaries via persistence diagrams.
3. Train regression models: baseline gradient-boosted trees on engineered features, Gaussian processes for uncertainty, and GNNs operating on adjacency/facet graphs.
4. Validate on held-out polytope families and out-of-distribution tests (e.g. random projections of higher-dimensional cubes).

**Refined Bayesian expectations.**

- 0.25: Models achieve <10% relative error with calibrated uncertainty on in-distribution data; extrapolation remains moderate.
- 0.35: 10–20% error with overconfident uncertainty estimates, but feature importance reveals interpretable geometric correlates.
- 0.20: Models overfit due to limited data volume; adding synthetic perturbations helps marginally but not enough.
- 0.15: Data generation cost or compute (especially for GNNs) outweighs benefit relative to heuristic estimators.
- 0.05: Tooling blocker (e.g. lack of GPU access) prevents completion.

**Posterior scientific value.**

- **High** for 0.25: supports interactive exploration and may inspire new conjectures from feature saliency.
- **Medium** for 0.35: still offers qualitative insights and can prioritize future data collection.
- **Low–medium** for 0.20/0.15: primarily informs resource allocation and dataset generation strategy.
- **Low** for 0.05: minimal immediate benefit.

**Follow-up unlocks.**

- Strong performance enables active learning loops where the surrogate guides which polytopes to compute exactly.
- Feature importance analyses may point to new analytic invariants worth formalizing.
- Underperformance pushes us toward better data augmentation or exploring equivariant neural architectures.

## Experiment 5 — Empirical Validation of Symplectic Capacity Inequalities

**Question.** Do empirical observations across diverse polytopes support conjectured inequalities (Viterbo conjecture ratios, Mahler-type bounds, symplectic Banach–Mazur comparisons) within measurable tolerance?

**Literature and precedent.**

- Artstein-Avidan & Ostrover (2013) outline key inequalities relating capacity and volume; Viterbo (2000) conjecture gives the canonical ratio to test.
- Alvarez Paiva, Balacheff & Tsanev (2012) provide near-tight constructions for the Viterbo conjecture using convex bodies close to polydisks.
- Nazarov, Petrov & Ryabogin (2019) analyze Mahler-type extremals which can be reinterpreted symplectically.

**Setup sketch.**

1. Compile inequalities with precise constants and required ancillary quantities (volume, polar body volume, widths) alongside references.
2. For each polytope from the Experiment 1 dataset plus selected literature extremals, compute the required invariants with interval arithmetic to bound numerical error.
3. Check inequality satisfaction, quantify slack, and perform sensitivity analysis via bootstrapped perturbations to determine robustness of near-tight examples.

**Refined Bayesian expectations.**

- 0.40: Inequalities hold with measurable slack; near-tightness appears in known families (e.g. Hanner polytopes), affirming literature priors.
- 0.30: New near-tight examples emerge, possibly indicating overlooked structure or numerical artifacts requiring validation.
- 0.15: Violations appear but vanish after accounting for numerical error, highlighting the need for certified computation.
- 0.10: Dataset coverage too sparse to draw conclusions; we only replicate known checks.
- 0.05: Pipeline issues (e.g. computing polar body volumes accurately) prevent execution.

**Posterior scientific value.**

- **High** for 0.40/0.30: contributes empirical evidence, informs theoretical focus, and could yield publishable counterexample searches.
- **Medium** for 0.15: still clarifies the importance of certification and may motivate adopting exact arithmetic libraries.
- **Low** for 0.10/0.05: little new knowledge beyond implementation notes.

**Follow-up unlocks.**

- Near-tight examples trigger theoretical classification work or targeted symplectic embedding constructions.
- Discrepancies motivate a dedicated certified computation project using rational polytopes.
- Sparse coverage outcome pushes to expand dataset or integrate with Experiment 6 (below) to generate new candidates algorithmically.

## Experiment 6 — Tracking Reeb Dynamics Across Combinatorial Transitions

**Question.** How do the Reeb flow characteristics (closed orbit actions, spectral invariants) of a polytope's contact boundary change as we traverse combinatorial transitions (e.g. flips in the normal fan)?

**Literature and precedent.**

- Pelayo & Vũ Ngọc (2011) analyze focus–focus singularities in toric integrable systems, suggesting qualitative changes when the polytope normal fan changes.
- Shelukhin (2022) discusses Reeb dynamics on contact-type boundaries of polytopes, indicating possible discontinuities in action spectra.

**Setup sketch.**

1. Start from a baseline polytope (e.g. cube) and trace a path in parameter space that triggers a known combinatorial change (vertex truncation, facet addition).
2. For discrete steps along the path, compute approximations of the Reeb flow invariants (e.g. closed orbit actions via contact homology numerics) using tools from the existing pipeline.
3. Correlate jumps in invariants with combinatorial data (changes in face lattice, normal fan) and with sensitivity findings from Experiment 2.

**Refined Bayesian expectations.**

- 0.30: Action spectra vary smoothly except near transitions, where identifiable jumps occur, aligning with Shelukhin's predictions.
- 0.30: Variability is dominated by numerical noise, preventing clear attribution to combinatorial events.
- 0.20: Unexpected discontinuities appear even without combinatorial flips, indicating hidden geometric factors.
- 0.15: Computing Reeb invariants proves too expensive, limiting the experiment to toy examples.
- 0.05: Tooling blockers (lack of contact homology implementation) halt progress.

**Posterior scientific value.**

- **High** for 0.30/0.20: directly informs the thesis narrative on how combinatorics influence symplectic invariants and could seed a follow-up paper.
- **Medium** for 0.30 (noise-dominated): suggests prioritizing numerical stabilization.
- **Low** for 0.15/0.05: primarily implementation feedback.

**Follow-up unlocks.**

- Clear jumps motivate targeted theoretical analysis using toric degeneration techniques.
- Noise issues link back to Experiment 2 and may require improved integrators or certified algorithms.
- Unexpected discontinuities prompt a search for new invariants capturing the observed behavior.

## Experiment 7 — Automated Discovery of Extremal Polytopes via Evolutionary Search

**Question.** Can evolutionary or reinforcement-learning search over polytope parameters uncover extremal examples for symplectic capacity inequalities or stress-test heuristics?

**Literature and precedent.**

- Böröczky & Campi (2019) explore extremal convex bodies via gradient flows, hinting that algorithmic search may find non-obvious candidates.
- In computational geometry, evolutionary strategies have been used to find counterexamples (e.g. to the Hirsch conjecture); we can adapt similar pipelines.

**Setup sketch.**

1. Parameterize polytopes via support functions or vertex sets with convexity constraints enforced through projection methods.
2. Define objective functions: minimize/maximize capacity-to-volume ratios, violate inequality margins, or maximize approximation error of heuristics.
3. Run evolutionary algorithms (CMA-ES, simulated annealing) seeded with known extremals, using Experiment 1's calibrated capacity computations for fitness evaluation.
4. Cluster and analyze discovered candidates, verifying convexity and uniqueness up to symplectic transformations.

**Refined Bayesian expectations.**

- 0.25: Search uncovers new near-extremal polytopes distinct from known families, warranting theoretical investigation.
- 0.30: Re-discovers known extremals but provides richer data on local landscapes, useful for confirming robustness.
- 0.20: Optimization stagnates due to noisy fitness evaluations; limited insights gained.
- 0.20: Computational cost too high; each fitness evaluation is prohibitive without surrogate models.
- 0.05: Implementation issues (convexity enforcement) derail the experiment.

**Posterior scientific value.**

- **High** for 0.25: potential publishable counterexamples or new conjectures.
- **Medium** for 0.30: still enriches understanding and validates the pipeline.
- **Low–medium** for 0.20 outcomes: informs whether to invest in surrogate-assisted search.
- **Low** for 0.05: mostly engineering feedback.

**Follow-up unlocks.**

- Novel candidates feed directly into Experiment 5 and theoretical work on extremals.
- High computational cost outcome motivates combining with Experiment 4's surrogates or using adaptive sampling strategies.
- Implementation challenges highlight where to integrate convex optimization libraries for feasibility projection.

## Prioritization Summary

1. **Experiment 1** — remains the highest leverage for calibrating existing tools and seeding datasets for most downstream work.
2. **Experiment 3** — crucial for understanding analytic vs. numerical trade-offs; can begin once Experiment 1 produces baseline data.
3. **Experiment 2** — follows naturally to assess robustness using the benchmarked pipeline; also prerequisite for Experiment 6.
4. **Experiment 5** — depends on curated datasets but directly informs thesis arguments and potential publications.
5. **Experiment 6** — contingent on sensitivity insights; high upside for theoretical narrative.
6. **Experiment 4** — valuable long-term but data-hungry; schedule after Experiments 1–3 populate datasets.
7. **Experiment 7** — exploratory and compute-intensive; postpone until surrogates or efficient fitness approximations are available.

