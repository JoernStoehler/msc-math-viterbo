# Review 10 — MSc Thesis

Provenance
- Topic: 10 — MSc Thesis (thesis-readiness)
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 5065563c868e
- Timebox: 75 minutes
- Scope: Structure, section coverage, references, reproducibility, and alignment with charter timeline/success metrics; no content rewrites.
- Version: v0.1

Context
- Review thesis readiness with focus on document structure, coverage of expected sections, bibliographic completeness, reproducibility affordances, and alignment to the project charter’s timeline and success metrics.

Inputs & Method
- Read thesis sources: `thesis/main.tex:1`, `thesis/frontmatter/abstract.tex:1`, `thesis/chapters/introduction.tex:1`, `thesis/chapters/literature-review.tex:1`, `thesis/references.bib:1`.
- Consult charter: `docs/charter.md:1`.
- Repo scan: `rg --files` to inventory the tree.
- Skills metadata surfaced: `uv run python scripts/load_skills_metadata.py --quiet` — OK.
- Docs build: `uv run mkdocs build --strict` — OK.
- Captured commit: `git rev-parse --short=12 HEAD`.

Findings (unsorted)
- Thesis skeleton compiles conceptually: `report` class with `hyperref`/`cleveref`, `amsmath`/`amsthm`, and sane margins via `geometry`.
- Title is placeholder (“Working Title”); author is placeholder (“Author Name”); both must be finalized for any external handoff.
- Date uses `\today`; consider pinning to the archival submission date at camera‑ready time for reproducibility of the PDF metadata.
- Frontmatter includes only `abstract`; no acknowledgements, declarations, or copyright/ethics pages common to institutional templates.
- Table of contents is enabled early; good for navigation while drafting.
- Only two chapters are active: `introduction` and `literature-review`.
- Core chapters `methodology`, `results`, `conclusion` are commented out in `thesis/main.tex`; appendices are also commented out.
- Abstract is a placeholder (“Summarise … in ~250 words”); does not yet communicate problem, approach, results, or contributions.
- Introduction is a thin placeholder; lacks problem statement, precise research questions, contributions, and roadmap.
- Literature chapter is the most developed; it is structured, topical, and already encodes normalization choices and audience.
- Literature chapter correctly reminds capacity axioms, normalization (`c(B)=\pi`, conformality), and gives ball/ellipsoid equality check.
- “Post‑2024 Update” correctly flags the Haim‑Kislev–Ostrover counterexample for EHZ in 4D and cites `HaimKislevOstrover2024` and `HaimKislev2019`.
- Literature chapter uses starred sections (`\section*{Audience and Scope}`, `Notation and Conventions`) which are unnumbered and not in the ToC; decide if they should be numbered or added to ToC via `\addcontentsline`.
- Several bullets are marked “from abstract”; these need verification and full citations or removal before submission.
- “Selected Reading Guide (names without full citations yet)” signals pending bibliography work; acceptable at draft phase, not for final.
- Citations used in text mostly exist in `references.bib` (e.g., `Viterbo2000`, `CHLS2007`, `CristofaroGardiner2015`, `Hutchings2014`, `Irie2022`, `AbbondandoloBenedettiEdtmair2023`, `ArtsteinAvidanMilmanOstrover2008`, `ArtsteinAvidanKarasevOstrover2014`, `Wormleighton2021`).
- The line “Cieliebak, Frauenfelder, Paternain: identification of c1^SH with c_EHZ” appears without a corresponding BibTeX entry; add the precise reference (CFP or appropriate source) or adjust to `CHLS2007` if that was intended.
- Normalization and notation are clear and consistent with mainstream conventions; choice of Liouville form acknowledged with equivalence remark.
- The ECH capacities discussion cites both asymptotics and expository sources; good coverage for 4D context.
- Relations to Mahler’s product are included and cite `ArtsteinAvidanKarasevOstrover2014`; nice cross‑bridge to convex geometry.
- Computational/algorithmic aspects section mentions Minkowski billiards, toric combinatorics, and SH proxies; aligns with repo capabilities.
- “Positioning of This Thesis” narrows scope to convex polytopes/star‑shaped and algorithmic lens on EHZ/related capacities; consistent with repo.
- Bibliography style `alpha` is fine for math; consider `amsalpha`/`abbrvnat` if name sorting or compressed labels are desired by the institution.
- No custom theorem environments are defined yet (although `amsthm` is loaded); later chapters will need shared theorem styles and numbering.
- No macros package file (e.g., `defs.tex`) exists; defining common commands for capacities, bodies, norms, and asymptotics will reduce repetition and typos.
- No figures/tables are included yet; algorithmic or geometric sections will likely need schematics and tables for datasets/results.
- The literature file contains unnumbered bullet lists; ensure prose integration or consistent list styling in final.
- Small LaTeX spacing artifact present (`see also\\ \cite{...}` with explicit space); minor cleanup pass needed for `\,` vs `\ ` usage.
- No institutional thesis template is referenced; if the university mandates one, migrate to it early to avoid last‑minute formatting churn.
- `bibliography{references}` is set; `references.bib` contains a focused but incomplete set; missing entries for some named sources in text.
- No `latexmkrc`/Makefile is provided; adding a build script with `latexmk -pdf -interaction=nonstopmode` would improve reproducibility.
- No CI job compiles the thesis PDF; optional, but a nightly PDF build could catch missing citations and broken refs early.
- Reproducibility section (methods/runtime/env) is absent; should document `uv`/`just` flows, dataset seeds, compute budgets, and commit hashes for experiments.
- Charter requires experiments to record the last commit hash; thesis should reflect this in a “Reproducibility” appendix.
- Charter’s “Docs strict build (0 warnings) on main” is currently satisfied locally; keep it green as chapters are added.
- Charter milestone “2025‑10‑21 — Atlas Tiny demo” implies near‑term results; thesis `results` chapter should at least outline planned tables/plots.
- The repo contains `tests/datasets/test_atlas_tiny.py` signaling a ready demo path; thesis should reference this dataset and expected invariants.
- Codebase has math layer modules (`src/viterbo/math/**`) with strong test coverage goals (≥95%); thesis should cite invariants/tests when relying on algorithms.
- Device/dtype policies are specified in the charter; any numerical results in the thesis should state dtype, tolerances, and device explicitly.
- No ethics/data policy needed (synthetic convex polytopes), but license and archival policy of generated datasets should be briefly stated.
- No discussion yet on computational budgets/cutoffs (e.g., oriented‑edge budgets); literature chapter hints at this—methodology should make them explicit with citations.
- Literature chapter includes 4D‑specific ECH content; ensure the thesis maintains a clear distinction between general 2n and n=2 cases.
- Lagrangian products are referenced in the counterexample; methodology should specify how such products are parameterized in experiments.
- Many sections promise “to be instantiated with precise constants”; track a checklist to resolve all placeholders before freeze.
- The `introduction` should contain an explicit contributions list (bulleted), clearly mapping to the deliverables and success metrics from the charter.
- Consider a “Background and Notation” chapter separate from literature if the literature becomes dense; otherwise keep concise reminders inline.
- Consider a “Threats to Validity” or “Limitations” section capturing numerical approximation risks and algorithmic coverage limits.
- Ensure `cleveref` names are configured for custom theorem types and for appendix numbering once appendices are enabled.
- Appendix stubs (proofs, extended tables, environment details, dataset schema) should be added early to absorb growing details without bloating main chapters.
- The bibliography includes both arXiv and journal entries; ensure consistency of fields (doi, eprint, url) per style guide.
- Some BibTeX entries lack DOIs/URLs; add where available for reproducibility and linking.
- Verify capitalization in titles that include acronyms (ECH, SH); protect with `{}` in BibTeX.
- No cross‑refs from literature to math modules/docs pages yet; adding pointers to `docs/math/*.md` could strengthen cohesion.
- The repo has `docs/math/*` pages (e.g., `volume.md`, `polytope.md`); consider citing these in the thesis for implementation details.
- The repo already uses `uv` and `just`; thesis should include a one‑page reproducibility recipe with exact commands.
- Consider including a “Glossary of Symbols” early; capacity notations, body classes, operators, and asymptotic symbols benefit from a quick map.
- Methodology should pre‑declare statistical practices for dataset analysis (splits, metrics, CIs), even if experiments are primarily geometric/computational.
- The writing tone in the literature chapter is concise and survey‑style; maintain consistent voice across chapters.
- Section ordering within literature is reasonable but can be revisited once methodology/results clarify story flow.
- No index is planned; optional but sometimes helpful in theses with heavy notation; consider a short index if time permits.
- No `\listoffigures`/`\listoftables`; add if figures/tables grow beyond a handful.
- No `\hypersetup` tweaks (colors, PDF metadata) are present; defaults are OK for drafts but verify institutional print requirements.
- Ensure `geometry` margins comply with the university guidelines (1in currently typical but must match policy).
- Ensure theorem numbering resets and cross‑chapter consistency once multiple chapters are enabled.
- The literature chapter uses both prose and lists; before final, convert key bullet lists into cohesive narratives where appropriate.
- No mention yet of verification against literature values (e.g., capacities for ellipsoids, polydisks); results should include sanity tables.
- Dataset schema and versioning are not yet documented in the thesis; point to the corresponding docs page and include a condensed schema in an appendix.
- No shared figure style (TikZ/Matplotlib) is defined; standardize to maintain visual coherence.
- No guidance yet on how to regenerate every figure/table from code with a single make target; recommended for reproducibility.
- The LaTeX sources are minimal and clean; good starting point for growth without legacy clutter.

Actions (first pass)
- Un‑comment and create stubs for `methodology`, `results`, `conclusion`, and an `appendix` (reproducibility + environment).
- Replace placeholders in `title`, `author`, and `abstract` with draft content; pin `\date{YYYY-MM-DD}` on release.
- Add missing BibTeX entries (e.g., Cieliebak–Frauenfelder–Paternain or correct source) and remove “from abstract” markers by verifying or deleting.
- Add `latexmk`/Makefile target and optional GitHub CI job to compile the PDF nightly.
- Add a `Reproducibility` appendix: environment (`uv`, `just`), datasets, seeds, device/dtype, commit hashes, and one‑click rebuild instructions.
- Add theorem environments and shared macros file; include glossaries if needed.
- Draft a results outline for the 2025‑10‑21 “Atlas Tiny” demo: planned tables/plots and sanity checks vs. literature values.

Cross-References
- Topics: T01 (Design Principles), T06 (Math Used), T07 (Algorithms Used), T08 (Review Process), T13 (Repo Readiness — AI vs Humans)
- Files: `thesis/main.tex:1`, `thesis/chapters/literature-review.tex:1`, `thesis/references.bib:1`, `docs/charter.md:1`, `docs/math/volume.md:1`, `docs/math/polytope.md:1`

Validation
- `uv run python scripts/load_skills_metadata.py --quiet` — OK
- `rg --files` — OK (repo inventory obtained)
- `uv run mkdocs build --strict` — OK (no strict errors)

Limitations
- Did not compile the LaTeX thesis to PDF; focused on source inspection.
- Did not run full `just checks` to avoid unrelated compute; scope limited to docs readiness.
- Literature content not fact‑checked line‑by‑line; this is a structural/readiness review.

Status
- Draft

Pre-Submit Checklist
- Do not link this review in nav; consolidation will handle publishing.
- `uv run mkdocs build --strict` green locally.
- Actions listed for follow‑up.
- Provenance filled; title follows pattern.
