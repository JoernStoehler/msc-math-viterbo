# Review 09 — Advisor-Facing Documentation/Presentation

Provenance
- Topic: 09 — Advisor-Facing Documentation/Presentation
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 5065563
- Timebox: ~90 minutes
- Scope: Advisor-facing narrative clarity, minimal run instructions, reproducibility notes, doc drift risks. No edits outside adding this review file.
- Version: v0.1

Context
- Review the repository from the perspective of the thesis advisor: Is there a clear, minimal narrative explaining what this is, how to run a tiny demonstration, how results are made reproducible, and where doc drift may occur. Capture everything worth mentioning (unsorted, redundant OK) to enable focused follow-up work.

Inputs & Method
- `uv run python scripts/load_skills_metadata.py` — loaded skills summaries to verify guidance freshness.
- `rg --files -n` — scanned repository structure.
- Read: `README.md`, `AGENTS.md`, `mkdocs.yml`, `docs/README.md`, `docs/charter.md`, `docs/architecture/overview.md`, `docs/math/**`, `docs/workflows-project-owner-vibekanban.md`, `docs/reviews/**`, `mail/**`, `notebooks/**`, and selected `src/viterbo/**` modules.
- `uv run mkdocs build --strict` — validated docs build locally.
- `git rev-parse HEAD` — captured commit hash for provenance.

Findings (unsorted)
- The top-level README explains the project succinctly (purpose: numerical experiments around the Viterbo conjecture; stack: PyTorch + C++; quickstart with `uv` + `just`) and links to Architecture and Charter; this is an effective first touch for an advisor skimming the repo.
- The Project Charter (`docs/charter.md`) provides a strong thesis-level narrative: purpose, scope, non‑negotiables, goals, success metrics, and milestones; it’s the right anchor for advisor alignment.
- Charter includes explicit freshness and reproducibility signals (docs strict build “0 warnings”, API-doc update within 48h, commit-hash recording for experiments); this is excellent for advisor trust.
- Minor Charter polish issue: duplicated acceptance-criteria bullet for the 2025‑10‑21 “Atlas Tiny” demo; small doc‑hygiene nit but not confusing.
- The Docs index (`docs/README.md`) is clear about what’s in the published site vs. “browse tree” content; it intentionally keeps nav sparse and relies on discoverability for briefs/workflows. This is reasonable but can hide advisor‑relevant notes if the advisor sticks to the nav.
- MkDocs strict build is green locally; build output lists many pages “not included in nav” (info, not warnings). The strict gate is doing its job without being noisy.
- The site nav prioritizes: Charter, Home, Reviews (index), Architecture, Math API. Advisor content lives primarily in Charter + Reviews; math sections are excellent references but deep for advisor skims.
- Reviews framework (`docs/reviews/**`) is robust: a template enforces provenance, validation, and checklists; prior reviews exist (01–08, 13). Advisors benefit indirectly when reviews surface concrete action lists.
- There is no single “Advisor Overview” page that consolidates: what to know this month, how to run a tiny demo locally, and pointers to most relevant artifacts. Today, this is spread across Charter, Docs index, reviews, and mail templates.
- The weekly mail system (`mail/`) is thoughtfully designed for advisor communication: stable context, template, windowing rules, human‑in‑the‑loop QA, and private artefacts location. Strong foundation for advisor‑facing reporting cadence.
- The mail template targets a crisp reading budget (10s → 30s → ≤4min) and enforces one link per claim, which aligns with advisor expectations.
- There’s a clear Owner workflow doc for VibeKanban + devcontainer + remote access. This is excellent for operations, but likely too heavy for an advisor who just wants to read or run a tiny demo.
- Minimal run instructions for a tiny demo exist but are not curated into an advisor‑friendly snippet. README’s Quickstart gives `uv` + `just` basics and two dummy notebooks; however, it does not show a minimal capacity/systolic example in one or two commands.
- Notebooks: `notebooks/dummy_generate.py` + `dummy_plot.py` demonstrate artefact I/O in a simple, reproducible way. This is good for a quick smoke demo (plots saved under `artefacts/`).
- Notebook `notebooks/atlas_tiny_profile.py` profiles the dataset builder; it’s dev‑facing (profiling) and not something an advisor would run without guidance.
- Dataset demo: `src/viterbo/datasets/atlas_tiny.py` returns deterministic small polytopes and attaches invariants (volume, optional 2D capacity, systolic ratio) and cycles. Tests validate determinism and shapes. This is the best candidate for an advisor‑sized “demo pipeline”.
- There is no dedicated `docs/datasets/atlas_tiny.md` page yet; the Charter acceptance criteria mention such a short page. This gap is visible for advisor onboarding to the “tomorrow demo”.
- Math API docs are unusually strong: per‑module pages with semantics, invariants, shapes/dtypes, and concise theory context. This helps advisors cross‑check terminology and algorithmic intent.
- The EHZ section (`docs/math/capacity_ehz.md`) includes both plan (algorithms list) and theory (definitions, facet‑multiplier program, 4D enumeration arguments), plus references. This is advisor‑appropriate depth with the right caveats.
- The oriented‑edge CH page (`docs/math/oriented_edge_spectrum_4d.md`) claims “Status: Implemented (deterministic, CPU‑only)” and provides API, tuning knobs, complexity, pruning guarantees, and a roadmap; code backs this up in `src/viterbo/math/capacity_ehz/stubs.py`.
- The certified C*(X) budget spec (`docs/math/cstar_constant_spec.md`) is rigorous and the implementation exists as `compute_cF_constant_certified(...)` in stubs; good doc‑code alignment on a tricky concept.
- The math layer documents invariants and conventions (even‑dimension policy, symplectic form) that match code (e.g., `_symplectic_form_matrix`, dtype/device handling); this reduces advisor confusion when skimming code for plausibility.
- Tests structure (smoke + deep) aligns with the narrative of fast iteration and incremental selection; advisors can trust that small invariants and sanity checks guard regressions.
- Quick minimal “advisor demo” is currently implicit: run `uv run mkdocs build --strict` (to see the docs) and possibly run `uv run python -c '...'` snippets using `rotated_regular_ngon2d` + `volume` + 2D capacity, or use the `atlas_tiny` helpers — but there’s no copy‑paste block in docs aimed at advisors.
- The top of the repo avoids overwhelming the reader; deeper details live under `docs/`. This is a good posture for advisors, provided a small “for advisors” page collects the current state and a minimal demo.
- Governance and roles are clear in the Charter; advisors know who decides scope and metrics (Project Owner), who enforces code quality (Maintainers), and how agents operate; reduces process confusion.
- The AGENTS skills map is auto‑generated and fresh; while aimed at agents, it shows discipline and consistency that advisors often care about (especially freshness timestamps).
- CI parity `just ci` includes docs build; strict docs policy is enforced; this is a positive signal that narrative changes are treated as first‑class.
- The Docs Home page is intentionally sparse; advisors may benefit from a “Start Here (Advisor)” anchor visible in nav, even if it only links to Charter + latest published review.
- The “Reviews index” (`docs/reviews/README.md`) is the hub for published reviews; currently only a subset is linked. As per process, linking is deferred to a separate consolidation task — good separation, but it does mean an advisor browsing the site may miss recent reviews until publishing happens.
- The math pages are readable and free of build warnings; equations are minimal and kept in plaintext/inline math where needed. Good balance for a code‑heavy thesis.
- The `README.md` runs on the assumption that `uv` is installed; for advisors, a short line reminding how to install uv (pipx, homebrew) is already present (link). Good.
- `pyproject.toml` pins versions (Torch, SciPy, matplotlib, ninja) with sane ranges; this helps reduce environment drift for reproducible demos.
- The C++ extension scaffold is present but safely optional (Python fallbacks); advisors will not be blocked by native builds when running basic demos.
- The `runtime` module exposes a time‑budget decorator (env `VITERBO_SOLVER_TIMEOUT`); docs (oriented‑edge page) mention timeouts and verified vs. memoized modes. Good for setting advisor expectations about long runs.
- Tests for `atlas_tiny` validate determinism and shape contracts; this makes it a safe demo target.
- The documentation and code consistently prefer float64 in the math layer and preserve dtype/device without implicit moves; this is called out in docs. Advisors care when numerical stability is discussed.
- The `docs/papers/*` directory holds TeX sources and notes; these are not deeply integrated into site nav (by design). For advisor reading, a curated “reading list” exists with annotations — helpful.
- Thesis sources exist under `thesis/` with a skeletal `main.tex` and chapter stubs; no build/run instructions are provided in‑repo (Overleaf sync is mentioned). Advisors who want a local PDF may need a tiny “How to build locally” note.
- The skills include a “writing” guide and “collaboration/reporting”; this shows awareness of advisor communication hygiene.
- Owner environment (`docs/environment.md`) documents devcontainer, tunnels, Cloudflare, and VibeKanban. It’s operationally dense and probably not for advisors — fine, but ensure advisor links don’t route here by default.
- The Docs charter sets an aggressive “0 warnings” target; current build emits only INFO lines (pages not in nav). Consider whether “0 warnings” means “no warnings” vs. “no INFO”; right now OK.
- Navigation gaps: “Owner workflow” and “Environment” are off‑nav; “Briefs” are off‑nav; “Papers” reading list is off‑nav. This is intentional, but the Advisor may miss them without deep browsing.
- The Reviews template mandates “record commands and outcomes” which aids reproducibility of the review process itself; good pattern for advisor trust.
- No explicit “latest results snapshot” is published in nav; advisors typically appreciate a periodically updated page with the current best numbers/plots and their provenance.
- The dataset acceptance criteria in the Charter promise a docs page for schema/invariants; that page is missing; this is a doc drift risk relative to the milestone.
- The `docs/architecture/overview.md` has one formatting hiccup (a stray list dash before “Author small, self‑contained tickets…”); purely cosmetic.
- The MkDocs build is fast (~1–2s locally via uv), which encourages frequent doc updates; low friction reduces drift.
- The math pages sometimes label sections as “stubs” or “planned backends”; this transparency is good and prevents over‑promising to advisors.
- The code and docs consistently use deterministic ordering and `tol ~ sqrt(eps)` heuristics; advisors reading proofs or numerical discussions benefit from seeing this explicitly stated.
- No “reproducible run descriptor” exists (e.g., a YAML that records commit hash + env + command + outputs) for demo runs. Weekly mail asks to record commit hashes, but a standard artefact format could help advisors rerun later.
- The notebooks demonstrate saving artefacts under `artefacts/` and printing “Wrote:” lines; this is a clear pattern that can be echoed in advisor demos.
- The `Justfile` offers `docs-build` but not a one‑step “advisor demo” target; adding a `just demo-advisor` that builds docs and runs a 2D capacity example would reduce friction.
- There’s no “single-command” check that a minimal advisor workflow works (build docs, generate small artefact, render a plot). Chaining existing pieces would be straightforward.
- The `skills/testing-and-ci.md` emphasizes incremental selection and strict docs in CI; advisor risks from doc drift are already mitigated by CI gates.
- The `skills/skill-authoring.md` encourages metadata freshness and auto‑generated summaries; this habit reduces stale guidance and helps advisors trust the process.
- The math layer is Torch‑first but uses SciPy for convex hulls in `vertices_to_halfspaces`; this pulls SciPy into runtime dependencies. It’s pinned and acceptable, but mention in advisor run notes helps preempt “why SciPy?” questions.
- The oriented‑edge implementation is CPU‑only and has heuristics toggles. Docs clearly state when results are certified vs. heuristic (memo); this is important for advisor interpretation of early results.
- The `mail/archive/` includes meeting notes; good provenance trail for advisor communications.
- `mkdocs.yml` uses Material theme with search; the site is readable and familiar for academics.
- The site omits detailed PR templates from nav (good), but includes a “PR template” file in docs; harmless.
- The docs home references VibeKanban workflows; this keeps process/self‑care aligned but is secondary for advisors.
- The math docs include references (CH, Haim‑Kislev, etc.) with enough context to look up details; this helps advisors cross‑validate claims quickly.
- The Charter sets success metrics for coverage and CI timing; advisors appreciate concrete targets.
- The Charter’s “Reproducibility: experiments record last commit hash” is strong, but a tiny helper function to stamp commit/hash into artefacts would make adherence trivial.
- Tests include performance tiers (`tests/performance/`), allowing the team to communicate measured progress to advisors credibly.
- No “figures” folder for published plots yet in docs; when results mature, a `docs/results/` or `docs/figures/` would give advisors a place to browse without notebooks.
- The docs make good use of constraints: pure math layer, no IO/state; advisors can understand the separation and trust the math APIs in isolation.
- The repo layout is orthodox and easy to explain in a slide: math → datasets → models; C++ with fallbacks; tests; docs; notebooks; thesis; mail.
- The Reviews process explicitly forbids modifying README/mkdocs/nav during review; this reduces merge risk but means advisors won’t see new reviews until consolidation — intentional tradeoff.
- Strict doc build means typos in links or front‑matter will break CI; this is healthy, and advisors benefit from fewer broken links.
- The minimal dataset (`atlas_tiny`) uses float64 and deterministic geometry; advisors can compare to literature values for polygons easily.
- `systolic_ratio(volume, capacity, 2n)` enforces positivity and dimension checks; docs point to invariants; good for advisor sanity checks in demonstrations.
- Deep math modules (`similarity`, `polar`) are marked stubs; this sets expectations correctly.
- The `skills/repo-onboarding.md` offers a quick command palette; if an advisor wanted to test locally, they could realistically follow Quickstart without devcontainer.
- There is no explicit Windows/macOS note; `uv` supports both, but C++ extensions may require compilers. Because fallbacks exist, minimal demos remain viable on major platforms.
- The `site/` folder is configured as MkDocs build output; CI pages badge is present in README, giving advisors a link to the live docs.
- Thesis TeX uses generic packages and a simple report class; building locally should be trivial with `latexmk`, but instructions are not provided.
- Mail system excludes private artefacts from git and encourages summaries; good privacy posture for advisor correspondence.
- The repo deliberately avoids exporting a public API via `__init__` re‑exports; docs explain this. Advisors will see “import from concrete module” examples in pages — consistent.
- The `AGENTS.md` boot sequence (load skills metadata, then act) is visible in this review process; including that provenance in reviews helps advisors trust methodical workflows.

Actions (first pass)
- Add a small “For Advisors” page (title: “Advisor Overview”) under `docs/` that links: Charter, latest published review, minimal demo instructions (two copy‑paste blocks), and reading list.
- Publish a minimal demo block in docs: (a) build docs; (b) run `atlas_tiny` and print a tiny table of `polytope_id`, `volume`, `capacity_ehz`, `systolic_ratio`; (c) optional plot. Keep CPU‑only and `<10s` on typical laptop.
- Create `docs/datasets/atlas_tiny.md` with schema/invariants and a “how to collate” snippet, fulfilling the Charter acceptance criteria.
- Add a helper to stamp commit hash + env into dataset artefacts (e.g., `utils/stamp_provenance(out, commit, env)`); reference it from the demo and notebooks.
- Consider a `just demo-advisor` command that runs the minimal demo end‑to‑end and prints where outputs were written.
- Add a tiny “How to build thesis locally” note in `thesis/README.md` (e.g., `latexmk -pdf main.tex`) for advisors who want a PDF without Overleaf.
- Decide whether to surface an “Advisor Overview” link in nav (one line in `mkdocs.yml`) during consolidation; otherwise ensure Docs Home clearly points to it.

Cross-References
- Topics: T01 (Project design principles), T02 (Onboarding), T05 (Developer docs), T08 (Review process)
- Files: `mkdocs.yml`, `README.md`, `docs/README.md`, `docs/charter.md`, `docs/architecture/overview.md`, `docs/math/index.md`, `docs/math/capacity_ehz.md`, `docs/math/oriented_edge_spectrum_4d.md`, `docs/math/cstar_constant_spec.md`, `src/viterbo/math/capacity_ehz/stubs.py`, `src/viterbo/math/polytope.py`, `src/viterbo/datasets/atlas_tiny.py`, `tests/datasets/test_atlas_tiny.py`, `mail/README.md`, `mail/template-weekly-mail.md`, `skills/experiments-notebooks-artefacts.md`.

Validation
- `uv run python scripts/load_skills_metadata.py` — OK; printed up‑to‑date skills (last‑updated dates in 2025‑10‑17/18 range).
- `uv run mkdocs build --strict` — OK; INFO lines list non‑nav pages; no warnings/errors; site builds in ~1–2s locally.
- `rg --files -n` — OK; enumerated repo layout for this audit.
- `git rev-parse HEAD` — OK; commit `5065563c868e786bfaeb3cabebf21265347d8932` captured.
- `just checks` — N/A for this docs‑only review (not run).

Limitations
- Did not execute long‑running math pipelines or the oriented‑edge solver; relied on documentation and code inspection for status.
- Did not modify nav/index as per task constraints; advisor discoverability improvements are proposed but not enacted in this review.
- Thesis TeX was not built; local build instructions are suggested but unverified here.
- Did not validate CI on GitHub; only local MkDocs build verified.

Status
- Draft

Pre-Submit Checklist
- Linked from `docs/reviews/README.md` under Published reviews — PENDING (intentionally deferred by policy; to be done in consolidation)
- `uv run mkdocs build --strict` green — DONE
- `just checks` green (or N/A for docs-only) — N/A
- Actions extracted (even if minimal) — DONE
- Cross-refs noted (topics/files) — DONE
- Provenance filled — DONE
- Title matches pattern `Review XX — <Topic Title>` — DONE
