# Review 05 — Developer Documentation

Provenance
- Topic: 05 — Developer documentation
- Author: Codex CLI (agent)
- Date: 2025-10-20
- Commit: 96e7f5acdfb3
- Timebox: ~90 minutes
- Scope: Documentation breadth, freshness, cross-linking, discoverability across `docs/`, `AGENTS.md`, `skills/`, `mkdocs.yml`; no code changes beyond linking this review from the index
- Version: v0.1

Context
- Focused review of developer documentation as it exists today to assess breadth, freshness, cross-linking, and discoverability. Goal is to surface specific observations (unsorted, unfiltered) to guide future fixes and consolidation.

Inputs & Method
- Read MkDocs config and built site locally: `uv run mkdocs build --strict` — OK
- Searched and skimmed `docs/` (architecture, math API, workflows, reviews, briefs), `AGENTS.md`, and `skills/`
- Looked for update signals (timestamps, auto-generated sections), navigation coverage in `mkdocs.yml`, and cross-links between sections
- Commands used (representative):
  - `rg --files -n`
  - `sed -n '1,200p' mkdocs.yml`
  - `uv run python scripts/load_skills_metadata.py --quiet`
  - `uv run mkdocs build --strict`

Findings (unsorted)
- MkDocs strict build is green and fast; current config uses `theme: material`, `plugins: [search]`, and a minimal set of Markdown extensions (admonition, tables, toc with permalinks). This is good for baseline quality and portability. (mkdocs.yml)
- The nav is intentionally sparse: Charter, Home, Reviews (index), Architecture overview, and a grouped Math API menu. Many docs exist outside nav by design, surfaced as “exist but not included in nav” during builds; these include briefs, environment notes, workflows, PR template, and specific paper notes. This keeps the top-level clean but can reduce discoverability for non-authors. (mkdocs.yml: nav; build output)
- Reviews are indexed via `docs/reviews/README.md`, which explains conventions, numbering, and a Published reviews section. Only two reviews were previously linked (01 and 08). Adding new reviews requires manually adding a single bullet here; this process is explicit and matches the template instructions. (docs/reviews/README.md)
- A standard review template exists (`docs/reviews/TEMPLATE.md`) with a comprehensive “Provenance” and validation checklist. It emphasizes strict MkDocs build and optional `just checks`. This template is aligned with the requested deliverable and promotes reproducibility of review context. (docs/reviews/TEMPLATE.md)
- `AGENTS.md` operates as a hub for “skills” with auto-generated sections maintained by `scripts/load_skills_metadata.py`. The “Skills Overview” includes last-updated timestamps and succinct descriptions. This provides a fresh index for developer-facing skills and appears up-to-date (2025-10-17/18 for most). (AGENTS.md)
- The boot sequence in `AGENTS.md` explicitly instructs loading skills metadata first, then consulting skill guides. This establishes a good invariant for freshness and navigation for agents. (AGENTS.md)
- Skills are not published in the MkDocs site (they live under `skills/`), but are the authoritative developer guides for agents. Discoverability for external developers reading the public site may be limited, but repo readers see `AGENTS.md` early (linked from `README.md`). This split is deliberate and acceptable, but could be highlighted more in the public docs’ Home page for non-agent contributors. (README.md, docs/README.md)
- The `docs/README.md` (Docs index) states that published docs mirror what most contributors need after reading `AGENTS.md` and points to Architecture overview, Math API, and Owner workflow. It also clarifies that other content is discoverable by browsing the tree. This is clear but relies on readers understanding that some content is intentionally off-nav. (docs/README.md)
- The top-level `README.md` is concise, with clear quickstart commands (uv + just), layout overview, and links to Architecture and Charter. It defers policy/workflow details to `AGENTS.md`, which is appropriate for developers. (README.md)
- The “Project Owner Workflow” page is comprehensive and operationally detailed, targeted at the Owner and not general developers. It’s not in nav, but is discoverable through Docs index and briefs/workflow references. It reads like a runbook with commands and security notes; good separation of audience. (docs/workflows-project-owner-vibekanban.md)
- Documentation freshness goals are encoded in `docs/charter.md` (e.g., “Docs freshness: MkDocs strict build (0 warnings) on main. API-affecting changes update docs within ≤ 48 hours.”). Good policy signal; actual enforcement relies on CI (mkdocs build present in `just ci`) and maintainer discipline. (docs/charter.md)
- The `Justfile` includes `docs-build` and incorporates a docs build in `ci`. This directly supports the freshness policy and early detection of link/nav issues. (Justfile)
- The Math API docs are well-structured: a clear `docs/math/index.md` and per-module pages (polytope, constructions, volume, symplectic, capacity_ehz, similarity, polar). The overview states conventions (Torch-first, shapes/dtypes, even-dimension policy), which is helpful. (docs/math/*.md)
- The Math API content focuses on semantics and invariants rather than implementation details; this aligns with stated intent. Shapes/dtypes are promised but some pages are stubs or brief, especially similarity and polar; content length varies. (docs/math/similarity.md, docs/math/polar.md)
- `capacity_ehz.md` exists and is the most detailed among math pages; matches presence of capacity-related Python modules and tests. This suggests coverage follows actual code maturity. (docs/math/capacity_ehz.md)
- Architecture overview is crisp, covering devices, dtypes, layering, ragged patterns, C++ extension policy, testing/CI philosophy, imports, and tasks. It includes practical local C++ build commands and troubleshooting. This is strong developer documentation. (docs/architecture/overview.md)
- The PR template in `docs/PR_TEMPLATE.md` is minimal (summary bullets + `just test`). It could reference the broader validation gates recommended by `Justfile` (`just checks`, `just ci`, docs build) to standardize expectations. (docs/PR_TEMPLATE.md)
- Discoverability: The MkDocs nav does not surface Docs index; “Home” points to repository `README.md`, not `docs/README.md`. While “Reviews” and “Architecture” are one click away, the Docs index (which explains what’s off-nav) is not directly linked from nav. Consider adding “Docs Index” to nav or linking it from Home. (mkdocs.yml, docs/README.md)
- Build output notes many pages not included in nav (environment, workflows, briefs, papers, prompts, and reviews themselves). This is fine but requires conscious linking strategy from tickets and indexes; otherwise, content risks becoming orphaned. (build output)
- The reviews process is documented (Topic 8 review) and the index enumerates topics 1–14; the presence of only a subset of published reviews is clearly called out. This signals an ongoing program rather than stale. (docs/reviews/README.md, docs/reviews/08-review-process.md)
- Skills are meticulously organized with frontmatter (`name`, `description`, `last-updated`) and cross-links between skills (related skills). They cover environment/tooling, testing/CI, good-code-loop, devcontainer ops, repo layout, performance discipline, writing, math layer, etc. This breadth is excellent for developer productivity and correctness. (skills/*)
- The skills metadata update script (`scripts/load_skills_metadata.py`) maintains auto-generated sections in `AGENTS.md` and validates metadata (`--check`). This automation is a strong freshness mechanism. (scripts/load_skills_metadata.py, AGENTS.md)
- The `skills/testing-and-ci.md` captures incremental selection, CI parity, and troubleshooting patterns. Guidance is actionable and aligns with the Justfile. Good cross-link to `good-code-loop`. (skills/testing-and-ci.md)
- The “Always-On Essentials” in `AGENTS.md` codifies navigation preferences (`rg`, ≤250 line streams), golden commands, escalation, and safety (no silent dtype/device hops). This guidance is repeated across skills and appears consistently applied. (AGENTS.md)
- The mkdocs theme is Material; no custom CSS is configured here (Cloudflare Workers inject font/CSS for VibeKanban UI, not the docs). Docs readability is fine by default; optional theme tuning could improve long-form review pages readability (font size/line length) if needed. (mkdocs.yml)
- CI badge in `README.md` includes a Docs workflow; pages are published to GitHub Pages. This increases discoverability for external readers. (README.md)
- The Math docs reference even-dimension policy and symplectic conventions; cross-links to tests do not exist directly, which is okay but linking key tests can help newcomers map docs to executable examples. (docs/math/index.md, tests/math/*)
- The docs site includes “Reviews” as a single index page; individual review pages are not added to nav on purpose. Publishing reviews relies on the index list. This is consistent but adds a manual step; the index provides instructions to add “a single bullet” — clear enough. (mkdocs.yml, docs/reviews/README.md)
- There is a clear repository layout section in top-level README and also in `skills/repo-layout.md`. Duplication is acceptable because audiences differ (public site vs agent skills). Content is consistent. (README.md, skills/repo-layout.md)
- The Charter encodes “Docs strict build (0 warnings)” as a goal. Current build emits “The following pages exist in the docs directory, but are not included in the nav configuration:” lines as INFO, not warnings. Strict mode doesn’t fail, so the goal is met; consider enabling additional link-checking if stricter guarantees are desired (via plugins like mkdocs-link-checker or mkdocs-htmlproofer-plugin). (docs/charter.md, mkdocs.yml)
- No search hints or tags beyond default search are used. Material supports improved search UX (metadata, sections), but default is sufficient for the current scale. (mkdocs.yml)
- No explicit contribution guide (`CONTRIBUTING.md`) is present in docs; guidance is spread across `AGENTS.md`, skills, and PR template. This is fine for an agent-centric project, but external contributors may expect a CONTRIBUTING page (even if it simply points to `AGENTS.md` + skills). (repo root)
- The `docs/papers/` subtree contains LaTeX sources and notes; these are part of the site but mostly off-nav. That is reasonable (reference material) but consider a landing page or a short index page to orient readers to the collection and its purpose. (docs/papers/*)
- `docs/prompts/orchestrator.md` exists and is off-nav. If it’s intended for agents only, consider referencing it from `AGENTS.md` or a relevant skill to make its role clearer. (docs/prompts/orchestrator.md)
- Owner-oriented environment details live in `docs/environment.md`, also off-nav. The presence of both `docs/environment.md` and `docs/workflows-project-owner-vibekanban.md` is logical; the latter is a runbook, the former is a high-level environment map. Cross-link between these two could help. (docs/environment.md, docs/workflows-project-owner-vibekanban.md)
- The Math docs pages are short and readable, with a preference for high-level guarantees. Some sections are marked as “stubs” implicitly (short pages) rather than explicitly (“Stub” banner). Consider adding small admonitions where content is intentionally incomplete to set expectations. (docs/math/*)
- `docs/architecture/overview.md` mentions specific local build commands for C++ — nice touch. It could reference the existence of Python fallbacks in `src/viterbo/_cpp/__init__.py` explicitly for readers to discover how it wires up. (docs/architecture/overview.md, src/viterbo/_cpp/*.py)
- The nav includes “Charter: charter.md” at top, which is good for context, but the Charter is quite detailed; new developers might benefit from a “Start Here” page that triangulates `AGENTS.md`, Docs Index, and Quickstart commands in one place (even a short section on Home could suffice). (mkdocs.yml, README.md, docs/README.md)
- `skills/writing.md` exists and can support long-form review writing standards. Linking it from the review template might improve consistency of tone/format. (skills/writing.md, docs/reviews/TEMPLATE.md)
- The repo has a `thesis/` directory with LaTeX; not part of docs nav (and shouldn’t be). No confusion arises because nav is explicit. (thesis/*)
- `skills/performance-discipline.md` sets an escalation path to C++ only with evidence. This is echoed in architecture overview. The reinforcement across docs is good; consistency helps prevent premature optimization. (skills/performance-discipline.md, docs/architecture/overview.md)
- `skills/devcontainer-ops.md` and `docs/workflows-project-owner-vibekanban.md` overlap in theme (environment operations), but targets differ (agents vs owner). Cross-links between them would reduce context-jumping for readers who toggle roles. (skills/devcontainer-ops.md, docs/workflows-project-owner-vibekanban.md)
- `skills/repo-onboarding.md` provides a startup checklist; it is not surfaced in public docs. The README points to `AGENTS.md`, which then points to skills. This path is clear for agents; for non-agents reading the site, the path is less obvious. (skills/repo-onboarding.md, README.md, AGENTS.md)
- The docs include a “Docs index” page (`docs/README.md`) but it is not in nav; someone landing on the site sees Home (repo README) rather than Docs index. This is fine but a mild discoverability gap for newcomers. (mkdocs.yml)
- “Papers” include several LaTeX sources that build outside the docs pipeline; no build integration is attempted, which keeps the docs build fast and robust. Good separation of concerns. (docs/papers/*)
- The docs site uses toc permalink anchors; this improves shareability of sub-sections. All pages I inspected render anchors correctly. (mkdocs.yml: toc)
- The Math docs do not currently cross-link to specific code files (e.g., `src/viterbo/math/polytope.py`) — adding a short “See also” box per page pointing to the module path can accelerate onboarding. (docs/math/*.md, src/viterbo/math/*)
- The “Briefs” directory is off-nav but acts as a durable record of decisions and task workflows. Linking selected, still-relevant briefs from the Charter or Architecture overview could surface high-value context without cluttering nav. (docs/briefs/*.md)
- `docs/reviews/README.md` numbering includes “5. Developer documentation”, which aligns with this review. The process text instructs adding a single bullet after creating the file, which has been followed here. (docs/reviews/README.md)
- The nav contains “Home: README.md”, which pulls the root README into the site. This is convenient but can slightly confuse expectations (Home content vs Docs index). The Home page could include a prominent link to “Docs index” to clarify the site’s structure. (mkdocs.yml, docs/README.md)
- No dead internal links were encountered during manual browsing; strict build didn’t surface link errors. That said, `mkdocs` core does not check all inline links; a dedicated link checker plugin would catch more issues proactively if desired. (mkdocs.yml)
- The repository’s “Golden commands” are present in `AGENTS.md` and skills, and echoed in README. Consistency of command names and flows is high. (AGENTS.md, skills/basic-environment.md, README.md)
- The reviews template asks for “Actions (first pass)” and “Cross-References”; this encourages reviews to end with actionable outcomes and provenance — a good practice for iterative documentation improvements. (docs/reviews/TEMPLATE.md)
- There is no explicit “Glossary” for symplectic geometry terms; the math docs assume background. For developer documentation, a short glossary or “Conventions” page could help new contributors align faster. (docs/math/*)
- Many pages have a high signal-to-noise ratio and avoid verbosity; this is good for execution. Where pages are sparse by design (stubs), small banners or TODO markers would help set expectations and invite contributions. (docs/math/similarity.md, docs/math/polar.md)
- Docs for datasets and models are not part of the docs site; models are experiment-only; datasets have tests. If these areas expand, consider adding a small “Data & Models” section with policy notes to prevent drift (e.g., device/dtype boundaries, collation). (README.md layout section)
- The repo-level onramp is efficient: “Ensure uv, just sync, just checks, just ci” — minimal overhead. This reduces friction for developers and keeps docs lean. (README.md)
- Security/permissions and persistence concerns are well documented for the Owner workflow (Cloudflare, Tailscale, bind mounts, token locations). This is precise but mostly for the Owner. Developers get the benefit indirectly via stable infra. (docs/workflows-project-owner-vibekanban.md, docs/environment.md)
- The docs site doesn’t currently expose a sitemap for the “off-nav” content. Search works, but adding a lightweight “Other documents” index (auto-generated list) could improve discoverability. (mkdocs.yml)
- Cross-link density between skills and docs is decent but could be improved: e.g., adding a “Related skills” section to public docs pages (Architecture overview, Docs index) would connect two worlds for hybrid audiences. (docs/architecture/overview.md, docs/README.md, skills/*)
- The “Charter” under nav sets expectations clearly (gates, SLOs for docs freshness). Linking the Charter from README (already present) is helpful; additionally, a short “How this repo stays healthy” section on Home could summarize gates for quick scanning. (docs/charter.md, README.md)

Actions (first pass)
- Add a “Docs Index” link to MkDocs nav (or add a prominent Home link to `docs/README.md`) to improve discoverability of off-nav content.
- Expand `docs/PR_TEMPLATE.md` to reference `just checks`, `just ci`, and `uv run mkdocs build --strict` so PRs include full validation evidence.
- Add a short “See also: module path” box to each Math API page linking to the corresponding `src/viterbo/math/<module>.py` file for faster triangulation.
- Cross-link `docs/environment.md` and `docs/workflows-project-owner-vibekanban.md` with a one-line “Related” note near the top of each.
- Consider enabling a link-check plugin in MkDocs CI to proactively catch broken internal/external links.
- Add a small “Conventions & Glossary” page (symplectic terms, dtype/device policy) and link it from Math API index.
- Optionally surface a generated “Other documents” index (off-nav docs) or link to `docs/README.md` from Home to set expectations.

Cross-References
- mkdocs.yml:1
- mkdocs.yml:18
- docs/reviews/README.md:1
- docs/reviews/TEMPLATE.md:1
- AGENTS.md:1
- README.md:1
- docs/README.md:1
- docs/architecture/overview.md:1
- docs/math/index.md:1
- docs/math/capacity_ehz.md:1
- docs/workflows-project-owner-vibekanban.md:1
- docs/environment.md:1
- docs/charter.md:1
- skills/testing-and-ci.md:1
- skills/good-code-loop.md:1
- skills/repo-layout.md:1

Validation
- `uv run mkdocs build --strict` — OK (no errors)
- `uv run python scripts/load_skills_metadata.py --quiet` — OK (skills overview refreshed)
- `just checks` — N/A (docs-only scope; not required)

Limitations
- Did not attempt to reorganize nav or add plugins; assessment focuses on current content and structure.
- Did not validate external links across all pages; a link-check plugin would do this better.
- Math docs content depth reviewed at a high level; did not validate every invariant against tests.

Status
- Draft

Pre-Submit Checklist
- Linked from `docs/reviews/README.md` under Published reviews — DONE
- `uv run mkdocs build --strict` green — DONE
- `just checks` green (or N/A for docs-only) — N/A
- Actions extracted (even if minimal) — DONE
- Cross-refs noted (topics/files) — DONE
- Provenance filled — DONE
- Title matches pattern `Review XX — <Topic Title>` — DONE

