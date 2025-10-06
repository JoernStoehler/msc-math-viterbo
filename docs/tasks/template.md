# Task Brief Template

Use this scaffold when drafting a new task or experiment brief. Delete sections that are not
relevant, but keep the headings order so other agents can parse documents quickly.

- **Status**: Draft | RFC | Scheduled | In Progress | Completed
- **Last updated**: YYYY-MM-DD
- **Owner / DRI**: \<name or "Unassigned">
- **Reviewers**: <optional>
- **Related docs**: \<links to roadmap, progress reports, prior briefs, AGENTS items>

## 1. Context and intent

Summarise the motivating question or problem, the background references, and why the task matters
now. Mention upstream docs (e.g., algorithm plans, meeting notes) that frame the work.

## 2. Objectives and non-goals

List the concrete outcomes we expect (what success looks like). Under a separate sub-list, state
explicit non-goals to prevent scope creep.

### In scope

- itemised deliverables

### Out of scope

- intentional exclusions and deferrals

## 3. Deliverables and exit criteria

Describe artefacts that must exist at completion (code modules, datasets, documents, benchmarks).
Provide crisp acceptance tests or observable signals.

## 4. Dependencies and prerequisites

Identify tasks, datasets, theoretical results, or environment features required before starting.
Note blocking vs. soft prerequisites.

## 5. Execution plan and checkpoints

Lay out the major steps, ordered where possible. Include checkpoints or review gates (e.g., initial
inventory, mid-task validation, final review). Mention expected iteration counts when helpful.

## 6. Effort and resource estimates

Use qualitative bins (low/medium/high) for:

- Agent time
- Compute budget
- Expert/PI involvement

Note any reevaluation points where we can stop early.

## 7. Testing, benchmarks, and verification

Specify which automated checks (format, lint, typecheck, unit, benchmark tiers) must run in CI vs.
locally. Mention manual validation (e.g., spot-check datasets, visual inspections) as needed.

## 8. Risks, mitigations, and escalation triggers

Enumerate major risks or unknowns, mitigations, and when to escalate (e.g., unclear invariants,
tooling gaps, missing data). Align with AGENTS escalation policy.

## 9. Follow-on work

List potential subsequent tasks unlocked by this work or items deferred for later. Use bullet points
for easy grooming.

---

Add additional sections (e.g., glossary, research questions) when specialised tasks warrant them.
Keep the brief concise enough that another agent can onboard within minutes.
