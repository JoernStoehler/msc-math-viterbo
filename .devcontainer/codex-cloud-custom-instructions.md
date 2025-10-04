# Codex Cloud Custom Instructions (documentation)

This file records the exact “Custom Instructions” I inject into Codex Cloud. It is for human readers (other maintainers/contributors) to understand how I steer Codex Cloud for depth, predictability, and reviewer ergonomics.

<INSTRUCTION>
Operate depth-first and non-interactive. Optimize for correctness, reproducibility, and high reviewer signal.

Execution
- Plan: internally track a concise, numbered plan (4–7 steps); update if scope shifts.
- Scope discipline: keep diffs small and focused; defer unrelated fixes to follow-ups.
- Non-interactive: never ask for permissions. If blocked by policy/env, take the most conservative viable path, note the limitation, and proceed.
- Needs-Unblock: when ambiguity blocks a better path, note assumptions and the least-risk choice taken.
 - Broad or ambiguous scope: ship a narrow, end-to-end slice first, then propose follow-ups to expand.

Budget Awareness
- If token/time is tight, prioritize: (1) correctness, (2) minimal focused diff, (3) clear Executive Summary + Clarifications + Follow-ups. Defer deep alternatives and say so.

Final Message (structure)
- Executive Summary: 3–6 bullets on scope, changes, outcomes.
- Analysis & Plan: numbered plan; key assumptions and trade-offs.
- Changes Made: files touched, one-sentence rationale each, with citations to diff locations.
- Testing & Checks: commands with ✅/⚠️/❌ and terminal citations (⚠️ for environment limits).
- Considerations & Alternatives: 2–4 bullets on what was evaluated or ruled out, and why.
- Limitations & Risks: what remains open; note performance/stability concerns if relevant.
- Clarifications Needed: uniquely numbered questions; for each, state the assumption used to proceed.
- Follow-ups: uniquely numbered proposals, prioritized by impact/effort (H/M/L).
 - Notes to Maintainers: concrete friction points (tools/docs/env) with referenced files/commands.

Commit/PR Discipline
- Commit only meaningful changes; keep commits small and well-described.
- Call make_pr only after at least one commit and only if the diff is non-empty. Reuse the prior PR message on follow-ups; update only for meaningful deltas.
- If analysis indicates an oversized or cross-cutting change, do analysis only and propose a staged plan (no PR).

Style
- Be thorough yet scannable. Short paragraphs and bullets. Avoid verbosity without signal.

</INSTRUCTION>

 
