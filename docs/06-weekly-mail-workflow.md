# Weekly Progress Mail Workflow

This document describes how to generate and archive the weekly progress mail that is sent to the thesis supervisor.

## Purpose
Summarise the past week’s achievements, upcoming plans, and any blockers in a concise email that can be refined and sent each Monday.

## Trigger
Run this workflow once per week, ideally after wrapping up Friday’s work so the update is ready for Monday morning.

## Steps
1. **Prepare context**
   - Pull the latest changes from `main`.
   - Review the roadmap (`docs/02-project-roadmap.md`) and any relevant meeting notes.
   - Skim recent commits: `git log --since="last friday" --oneline`.
2. **Generate a draft**
   - Open `progress-reports/weekly-mail-prompt.md`.
   - Launch Codex (or your preferred agent) with that prompt and supply any extra notes from meetings or experiments.
   - Ask the agent to produce a draft following `progress-reports/weekly-mail-template.md`.
3. **Edit and finalise**
   - Copy the draft into `progress-reports/drafts/YYYY-MM-DD-weekly-mail.md`.
   - Polish the tone, verify dates, and ensure the word count stays below 250 words.
   - Update the “Plans” section to reflect the next concrete milestones.
4. **Send and archive**
   - Send the edited email to the supervisor from your mail client.
   - Commit the archived draft along with any artefacts referenced in the update.

## Tips
- Keep achievements outcome-focused (e.g., “Validated lemmas 2.1–2.3 by reproducing the proof in SageMath”).
- Flag blockers early so they can be discussed during weekly supervision.
- Link back to thesis chapters by name (e.g., “Drafted Section~2.1 in the Introduction chapter”) when useful for continuity.
