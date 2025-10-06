# Task & Experiment Evaluation Methodology

This methodology guides Codex research and engineering agents when scoping, assessing, and
prioritising work on the MSc project for Viterbo's conjecture. It unifies research experiments and
software-engineering tasks because both feed the same decision queue and rely on the shared brief
structure under `docs/tasks/`.

## 1. Mental model and workflow

1. **Collect context**
   - Re-read the relevant task briefs, roadmap notes, algorithm implementation plan, and the latest
     progress reports before scoring any proposal. Note the conjectures, code modules, datasets, or
     infrastructure the work touches.
1. **Clarify the task nucleus**
   - State the core question or capability the work should deliver and the artefacts produced
     (datasets, counterexamples, refactored modules, benchmark harnesses, etc.). Keep the proposal
     narrow enough that success, failure, or blockers are unambiguous.
1. **Map outcomes → information value**
   - Enumerate mutually exclusive outcomes at two levels:
     1. _High-level categories_ (e.g., "blocked by SWE complexity", "formalisation infeasible",
        "inconclusive", "supports conjecture", "finds counterexample").
     1. _Concrete sub-outcomes_ capturing how our knowledge or infrastructure changes (e.g.,
        "dataset of systolic ratios for 200 polytopes reveals new clusters", "reference algorithm
        layout too entangled to split").
   - Assign odds using gut feeling backed by BOTECs when helpful; explicitly mark high-variance
     estimates instead of forcing false precision.
1. **Score utilities**
   - Rate each outcome's _relative information value_ on an arbitrary utility scale. Suggested
     anchors:
     - `0`: negligible insight (e.g., rediscovering known engineering pain).
     - `+3`: useful but bounded insight (e.g., a clean dataset seeding follow-up conjectures).
     - `+6`: major leverage (e.g., counterexamples, modular refactors unlocking several experiments,
       or benchmark suites that harden regression detection).
     - `-3` or lower: harmful outcomes (e.g., false confidence from undetected bugs).
   - Utilities should reflect downstream leverage: the number of dependent tasks, publications, or
     PI decisions the result unlocks.
1. **Estimate costs**
   - Track costs across three axes: agent effort, compute, and expert/PI time. Use qualitative bins
     (`low`, `medium`, `high`) tied to rough effort bands: `low` ≈ ≤ 0.5 agent-weeks on commodity
     CPU, `medium` ≈ 0.5–2 agent-weeks or GPU/special tooling setup, `high` ≥ 2 agent-weeks or
     requires PI time.
   - Highlight early reevaluation checkpoints—experiments with early "stop/go" decisions are safer
     even when full execution is expensive.
   - Treat commercial solvers as out of scope; the team committed to open-source backends even if
     runs take longer.
1. **Compute expected scores**
   - Compute an _expected utility_ (`Σ probability × utility`) and optionally adjust by a cost
     heuristic. Use these as guides, not mandates; override them when qualitative reasoning
     dominates.
1. **Document dependencies**
   - Record prerequisites (datasets, lemmas, module refactors) and downstream tasks whose priority
     depends on this work. Express dependencies visually via Mermaid graphs in
     `docs/tasks/02-task-portfolio.md`.
1. **Decide and communicate**
   - Prioritise work with high expected value, manageable risk, and limited dependencies. Log open
     questions or blockers directly inside the owning task brief so future agents inherit the
     context.

## 2. Outcome taxonomy

| Category                       | Description                                                          | Typical next steps                                    |
| ------------------------------ | -------------------------------------------------------------------- | ----------------------------------------------------- |
| Success                        | Work delivers the intended artefact or insight.                      | Trigger follow-on briefs; integrate code or datasets. |
| Rich partial                   | Results differ from expectations yet surface new leads.              | Draft follow-up proposals; refine conjectures.        |
| Inconclusive                   | Signal drowned in noise or evaluation insufficient.                  | Collect more data; improve diagnostics.               |
| Aborted – SWE                  | Blocked by engineering complexity or unstable numerics.              | Decide whether to request expert help or archive.     |
| Aborted – formalisation        | Discretisation/theory gap prevents execution.                        | Escalate to PI; consider reframing.                   |
| Aborted – prioritisation shift | Task paused because dependencies moved or higher-value work emerged. | Revisit when blockers lift; record rationale.         |
| Safety failure                 | Outputs misleading results without detection.                        | Halt dependents; add diagnostics and safeguards.      |

Use this taxonomy before customising finer-grained outcomes for individual briefs.

## 3. Probability heuristics

- Start with base rates from prior Codex efforts (reference implementations tend to succeed within
  two iterations ≈ 60%; ambitious optimisation or ML tracks may succeed ≈ 20% without PI support).
- Adjust odds multiplicatively for unknown libraries, theoretical novelty, data availability,
  reliance on non-deterministic optimisation, or breadth of code touched.
- Prefer ranges (`30–50%`) when uncertainty is decision-relevant; large ranges signal that we should
  gather clarifying information first.
- Record correlations when tasks share dependencies—success is rarely independent if they rely on
  the same dataset or module refactor.

## 4. Utility heuristics

- Focus on information value and downstream unlocks:
  - Counterexample discovery: `+8` or higher because it redirects the research.
  - Well-curated dataset enabling statistical exploration: `+5`.
  - Establishing that a code path is brittle: `+1` (hygiene insight).
  - Landing a clean modularisation enabling multiple algorithm families: `+4`.
- Penalise misleading or unverified results heavily (`≤ -6`).
- When utilities depend on expert review (e.g., PI inspection of datasets), state the dependency
  explicitly in the brief.

## 5. Cost estimation bins

| Cost level | Definition                                                 | Examples                                                     |
| ---------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| Low        | ≤ 0.5 agent-weeks; CPU only; no expert help.               | Deterministic reference algorithms, documentation passes.    |
| Medium     | 0.5–2 agent-weeks; may need GPU or moderate refactors.     | Dataset generation, adaptive relaxations, JAX ports.         |
| High       | ≥ 2 agent-weeks or expert/PI time or specialised hardware. | Large MILP sweeps, transformer training, sweeping refactors. |

Log costs as tuples `(agent_time, compute, expert_time)` and mark reevaluation points.

## 6. Decision heuristics and stopping rules

- Stop refining numbers once ordering stabilises; note explicitly when additional precision is
  non-decision-relevant.
- Prefer cheaper or prerequisite tasks that unlock expensive ones.
- Flag tasks that can run in parallel versus those that must stay serial.
- Update the Mermaid dependency graph whenever outcomes change ordering so the queue stays accurate.

## 7. Communication norms

- Write evaluations in full sentences referencing the source documents informing your belief
  (algorithm plan sections, prior PRs, benchmarks).
- Use tables to compare multiple tasks when practical.
- Capture subjective confidence levels; "high variance" or "needs PI review" is acceptable
  shorthand.
- Keep documents living: append dated updates when new evidence shifts odds or utilities.
- Store full briefs under `docs/tasks/` using `template.md`; link to them from the portfolio so
  execution notes live with their owning document.

## 8. Integrating software-engineering-first work

- Treat refactors, benchmarking harnesses, or infrastructure upgrades as first-class items—many
  experiments depend on trustworthy code paths.
- Document success criteria for SWE tasks (shared signatures, tests, profiling hooks, benchmark
  tiers) inside the brief.
- Capture regression risk in the safety-failure bucket; undetected numerical drift is as harmful as
  misleading data.

## 9. Testing and benchmarking cadence

- Maintain a **three-tier loop**:
  1. _Inner loop_ (\<2 minutes): format, lint, typecheck, plus targeted smoke tests covering touched
     modules. Agents should run these repeatedly while iterating.
  1. _CI loop_ (\<5 minutes): `just ci` on every PR. Restrict benchmark coverage here to
     deterministic smoke cases (e.g., one small polytope per algorithm family) to keep runtime
     predictable.
  1. _Deep loop_ (5–20 minutes for broader unit/benchmark suites; up to several hours for profiling
     or HPC/ML experiments). Run these locally before making performance claims or when touching hot
     paths. Document commands and summaries in PR descriptions or task notes.
- Schedule **long-haul (>1 hour) benchmark or profiling passes roughly monthly** (align with roadmap
  milestones or completion of major infrastructure tasks such as T1/T2). Record results by:
  - Saving raw metrics using `pytest --benchmark-autosave` (stored under `.benchmarks/`), and
  - Adding a short narrative summary (command, hardware, notable deltas) to the relevant progress
    report or task brief. No separate logbook is required—the autosaved artefacts plus the
    brief-level notes keep history auditable.
- For HPC or ML experiments, capture: dataset size, hardware, precision, and any reproducibility
  seeds. Prefer comparing against the latest `.benchmarks/` artefact or a dedicated baseline branch
  rather than ad-hoc spreadsheets.
- Use pytest markers to distinguish smoke vs. deep benchmark sets so CI can stay fast while longer
  runs remain opt-in.
- Prefer selective execution over skipping suites; when a benchmark is temporarily too slow,
  document a follow-up task or waiver.

Following this methodology keeps prioritisation reproducible while leaving room for expert judgement
and future revisions.
