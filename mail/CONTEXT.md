---
version: 1
project: msc-thesis-viterbo
timezone: Europe/Berlin
subject_format: "Weekly Report — Jörn Stöhler — {{from}}–{{to}}{{action_tag}}"
action_tag_rule: "append [Action Needed] iff the 'Actions for Kai' section is non-empty"
---

# Reporter Context (Stable)

## Purpose
Keep **Prof. Kai (Uni Augsburg)** synced on the **mathematical takeaways** of the thesis, plus plan deltas and explicit asks. Code/process details appear only when they help the *math* narrative.

## Audience & reading budget
- 10 s skim (executive bullets) → 30 s action triage → ≤4 min details.

## Scope & sources of truth
- Single repo: `JoernStoehler/msc-thesis-viterbo` (default branch `main`).
- Include: meeting notes, thesis notes, reports, experiment writeups, GH Pages; wandb summary pages only if they clarify a takeaway.
- Exclude: raw code/PR/commit streams from the main body; they can appear in **Appendix** if useful.

## Reporting window
- Default: **“since last mail”**. Derive from the `to:` field in the most recent folder `mail/<date>/mail.md`.
- Fallback if none exists: last week ending **Sunday 23:59 (Europe/Berlin)**.

## What makes the cut (selection heuristic)
- **Research >> everything else.** Include items that update Kai’s model of the symplectic geometry landscape or your plan.
- Avoid vanity metrics (LOC, commit counts). CI status appears only as a one-liner if it directly explains a plan adjustment.

## Structure of the mail
1) **Executive Summary** — ≤5 bullets, ≤80 chars each.  
2) **Actions for Kai** — only if you have explicit questions/decisions.  
3) **Weekly Update** — one subheading per takeaway; short paragraph + one link.  
4) **Plan Adjustments** — what changed and why (2–5 bullets).  
5) **Next Up** — concrete next steps (2–6 bullets).  
6) **Appendix (links)** — human-readable links/paths.

## Evidence & trust order
Prefer one **human-readable** artifact per claim.
**Trust stack:** current curated notes (this week’s folder) > what’s visible in HEAD now > past mails (approved) > PR messages > docstrings > older docs.

## Components (orientation only)
- `src/viterbo/` — library;  
- `src/viterbo/experiments/` — maintained experiment notes;  
- `tests/viterbo/` — TDD;  
- `tests/performance/viterbo/` — benchmarks/profiling;  
- notebooks (TBD) — disposable experiments.

## Per-week folder layout
`mail/<YYYY-MM-DD>/`
- `curated-takeaways.md` — short list of selected takeaways (+ evidence links).
- `gathered-changes.md` — brief diff of relevant notes/figures/docs.
- `mail.md` — the draft you will send (Markdown with YAML header).
- optional: figures/scripts referenced by the mail.

## Runbook (what the agent does)
1. **Window**: compute from previous folder’s `mail.md` `to:` → now; else fallback (Sun 23:59).  
2. **Gather**: scan changed/added Markdown under `src/viterbo/**`, `reports/**`, `docs/**`, repo root notes; harvest lines starting with `Conjecture:`, `Lemma:`, `Observation:`, `Counterexample:`, `Example:`, `Open Q:`.  
3. **Curate**: draft candidate takeaways (type, one-liner, why-Kai-cares, one link). Keep only items that inform Kai; drop code-only deltas.  
4. **Decide** if **Actions for Kai** exists (explicit questions/decisions).  
5. **Compose** `mail.md` from the template; keep links human-readable.  
6. **QA**: subject format, ≤5 summary bullets, one link per claim, contiguous window.  
7. **Persist** folder & open PR: `mail/<YYYY-MM-DD>` as `Weekly report <YYYY-MM-DD>`. Stop. Human review & send.

## QA checklist (hard fail if violated)
- Subject matches `Weekly Report — Jörn Stöhler — {{from}}–{{to}}{{action_tag}}`.
- Every claim has exactly **one** resolvable, human-readable link.
- Executive Summary ≤5 bullets; each ≤80 chars.
- Window is contiguous with previous report.
- No vanity metrics in the main body.

## Definition of Done
- Draft saved at `mail/<date>/mail.md` with YAML header (`from`, `to`, `action_needed`, `status: draft`).
- `curated-takeaways.md` and `gathered-changes.md` present.
- PR opened: “Weekly report <YYYY-MM-DD>”.
