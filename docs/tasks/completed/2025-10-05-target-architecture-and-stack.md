# Task Brief — Target Architecture, Conventions, and Tooling (T0)

- Status: Completed
- Last updated: 2025-10-08
- Owner / DRI: Unassigned
- Reviewers: Maintainer (PI), module maintainers
- Related docs: `AGENTS.md`, `README.md`, `pyproject.toml`, `pyrightconfig.json`,
  `.github/workflows/ci.yml`, `.devcontainer/post-start.sh`,
  `docs/rfc/002-jax-pyright-typing-options.md`,
  `docs/tasks/completed/001-bring-repo-into-alignment-with-AGENTS-conventions.md`

## One‑liner checklist (for this brief)

- Scope: Decide whether any supplemental guidance beyond `AGENTS.md` is needed for architecture,
  conventions, DevOps, or tooling.
- Acceptance criteria: Clear decision documented plus follow-up actions (if any). No redundant
  restatement of `AGENTS.md`.
- Links to context: See Related docs above (source of truth lives in `AGENTS.md`).
- Constraints: JAX‑first; x64; strict typing; deterministic tests; SciPy/NumPy only via wrappers; no
  optional stacks without rationale.
- Expected tests/benchmarks: None beyond doc linting (Ruff pydocstyle) if a new note is introduced.
- Escalation triggers: Conflicts with existing policy; ambiguous invariants; environment or CI
  mismatches.

## 1. Context and intent

Contributors already rely on `AGENTS.md` as the single source of truth. Earlier drafts suggested a
separate “Target Architecture & Stack” document, but that risks duplicating policy unless additional
explanatory material exists. This brief re-evaluates the need for auxiliary documentation and
records any decision to add clarifying references (or to confirm that none are required right now).

## 2. Objectives and non‑goals

### In scope

- Audit `AGENTS.md` and related configs to verify whether critical details would benefit from
  elaboration (diagrams, worked examples, onboarding heuristics, etc.).
- If supplemental material is justified, outline the unique content it would cover and link it
  explicitly to existing policy.
- If no new insights are necessary, document the “no additional doc needed” decision so future
  agents know T0 is satisfied without action.
- Keep `README.md` pointing to `AGENTS.md` as the authoritative policy entry point (already
  completed).

### Out of scope

- Rewriting or duplicating sections of `AGENTS.md` without new substance.
- Introducing new policies or tooling decisions without Maintainer approval.
- Editing code, configuration files, or CI beyond references needed for documentation.

## 3. Deliverables and exit criteria

- Decision note logged in this brief (or a short companion Markdown file) explaining whether extra
  documentation is warranted and why.
- If proceeding with supplemental material, list the concrete topics and references that will make
  it non-redundant; otherwise, close the brief with the decision rationale (“no extra docs
  required”).
- Any new doc passes lint checks (`just lint`) and links to source configs.

## 4. Dependencies and prerequisites

- Inputs: `AGENTS.md:1`, `pyproject.toml:1`, `pyrightconfig.json:1`, `.github/workflows/ci.yml:1`,
  `.devcontainer/post-start.sh:1`, `docs/rfc/002-jax-pyright-typing-options.md:1`.
- No runtime dependencies; this is a documentation/decision task.

## 5. Execution plan and checkpoints

1. ✅ Inventory current policies and configs; collect file anchors for cross-references (completed
   2025‑10‑08).
1. ✅ Update `README.md` with a pointer to `AGENTS.md` so newcomers land on the source of truth.
1. ✅ Evaluate whether there are real gaps that merit supplemental explanatory docs or diagrams.
   - Review of `AGENTS.md`, `docs/22-code-structure.md`, task briefs, and CI/Justfile configs found
     no missing guidance that would benefit from a standalone architecture note.
1. ✅ Record the decision:
   - No additional documentation is required. Existing materials already cover architecture, coding
     conventions, and operational workflow. Future teams should reopen T0 only if new explanatory
     content is identified.

## Outcome

No supplemental “Target Architecture & Stack” document is needed. `AGENTS.md`, supporting task
briefs, and configuration files already provide the necessary guidance. T0 can be considered closed;
reopen only if future work uncovers new explanatory requirements.

## 6. Effort and resource estimates

- Agent time: Low
- Compute budget: None
- Expert/PI involvement: Low (consult only if new guidance is proposed)

## 7. Testing, benchmarks, and verification

- If new docs are written, run `just lint` to validate formatting/linking.
- Manual: ensure any added guidance truly augments `AGENTS.md` instead of copying it (currently
  N/A).

## 8. Risks, mitigations, and escalation triggers

- Risk: Duplicating policy text and causing drift.
  - Mitigation: Use this brief to block redundant docs; reference `AGENTS.md` instead of restating
    it.
- Risk: Missing real onboarding pain points by declaring “no change” prematurely.
  - Mitigation: Survey recent tasks/readmes to confirm contributors understand the current policies.
    (Completed: no gaps surfaced during review.)

## 9. Follow‑on work

- Optional: If future experiments reveal onboarding gaps, reopen T0 (or author a scoped design note)
  with concrete additions.
- Optional: Add diagrams or workflow examples only when they deliver new clarity beyond `AGENTS.md`.

## Relationship to recent briefs

T0 now acts as a holding brief ensuring documentation remains DRY. It complements the completed
alignment work (`docs/tasks/completed/001-bring-repo-into-alignment-with-AGENTS-conventions.md:1`)
by confirming the repository continues to match `AGENTS.md` without adding redundant policy
documents.
