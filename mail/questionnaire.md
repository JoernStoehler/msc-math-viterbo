# Reporter Agent – Weekly Context Catalogue

## Blank Questionnaire
Use this catalogue to capture invariants for new projects. Keep answers concise.

1. Stakeholders & audience – primary recipient, CCs, tone, reading time  
2. Cadence & timing – weekly boundary, deadlines  
3. Scope – repos, external systems, meeting notes / submissions  
4. Definition of progress – what matters vs. churn  
5. Work taxonomy & labels – if any  
6. Prioritization – weightings, impact signals, risks to surface  
7. Structure & style – required sections, word limits, subject format  
8. Evidence & traceability – acceptable types, trust order, metadata  
9. Components & directory map – include/exclude areas  
10. Risk/redaction – rules, denylist, escalation path  
11. QA/approval – human approval, checks, thresholds  
12. Metrics/dashboards – what to compute, where  
13. Special cases – no changes; incidents; multi-repo stitching  
14. Process memory – snapshots location, first-run horizon  
15. Compliance/archival – archive format/location, retention, templates

## Answers (v1, 2025-10-07)
**Audience & goals.** Primary: Prof. Kai (Uni Augsburg). Reading budget: 10 s summary, <30 s action triage, ≤2 min actions, ≤2 min explanations. Optimize for mathematical takeaways; code/methodology only if it helps explain the math.

**Cadence.** Send on weekend or Monday before noon. Reports form a contiguous thread (“since last mail”). No hard deadline constraints.

**Scope.** Single repo: github.com/JoernStoehler/msc-thesis-viterbo (branch: `main`). External dashboards (wandb, GH Pages) exist; their data also lives in-repo (git/LFS; not always at HEAD). Include meeting notes and paper submissions if any; repo is the source of truth.

**Progress definition.** Focus on “what Kai needs to know and why”. No interest in implementation details per se.

**Outcome metrics Kai cares about.** Mathematical conjectures/proofs and the paths leading there (empirical observations, case studies). Little/no interest in algorithmic engineering without math insight.

**Vanity metrics.** Do not include (appendix at most).

**Taxonomy/labels.** Prefer takeaway-first narration instead of rigid taxonomy.

**Prioritization.** Research >> everything else. Surface plan adjustments, estimate updates, and repeated mistakes/lessons learned.

**Structure & style.** Executive summary → whether actions are needed → weekly update with one subheading per takeaway → adjustments for future plan → next up. Risk matrices not needed; risks appear as part of takeaways and plan changes. Skimmable text, correct grammar, clear/simple terminology.

**Subject format (initial).** “Weekly Report Jörn Stöhler Sep 30 – Oct 7 [Action Needed]” (keywords stable; open to drift).

**Format.** Markdown; minimal formatting; links and pseudo-LaTeX ok.

**Evidence.** Prefer easy-to-view Markdown or GH Pages with embedded images. Avoid PR/commit links in the main body. (Appendix OK.)

**Trust order.** Jörn’s current curated notes > current HEAD > past mails (approved) > PR messages > maintained docstrings > other docs > old reports.

**Metadata per claim.** Not in the final mail; drafting may include minimal references for verification.

**Components.** `src/viterbo/` (lib), `tests/viterbo/` (TDD), `tests/performance/viterbo/` (bench/profiling), `src/viterbo/experiments/` (maintained experiments), notebooks TBD (disposable).

**Exclusions.** None strictly; ignore areas not obviously relevant this week.

**Redaction/denylist.** None.

**Escalation path.** N/A.

**Approval & QA.** Human hand-edits and sends; then mark as sent and commit. No automated checks required beyond “good enough to keep sync and serve as lookup months later”.

**Metrics/dashboards.** Not relevant except: consider `make ci` as a quick sanity line (PASS/FAIL/TIMEOUT/NA) only if it explains plan status.

**Special cases.** If no changes: send a short “no changes this week” status. No separate incident format envisaged.

**Process memory.** One folder per mail: `mail/<YYYY-MM-DD>/` with `gathered-changes.md`, `curated-takeaways.md`, `draft.md`, `mail.md`, figures/scripts. No strict naming, but `mail.md` is canonical, with a small YAML header (`status`, `from`, `to`, `action_needed`, `sent_at` when applicable).

**First-run horizon.** Topic was assigned ~3 weeks ago; repo now has preliminary results.
