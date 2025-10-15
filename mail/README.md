# Weekly Mail System

This folder holds the stable context and per-week artefacts for thesis progress mails.

## For the Project Owner
When you decide it is time to write a report and the repo is ready, send this line to your codex agent:

> Please prepare this week’s report using mail/CONTEXT.md. No report folder exists yet. Create mail/<today-YYYY-MM-DD>/, derive the window from the last report’s `to:` value, then generate curated-takeaways.md, gathered-changes.md, and mail.md from mail/template-weekly-mail.md. Open a PR “Weekly report <YYYY-MM-DD>” and stop before sending.

After the agent runs:
- Review `mail/<date>/curated-takeaways.md` for a quick sanity check.
- Edit `mail/<date>/mail.md` as needed, including whether an **Actions for Kai** section is required.
- Send the email manually to the Academic Advisor; once sent, update the YAML header (`status: sent`) and merge the PR.

Notes:
- The Academic Advisor does not routinely use the repo; prefer attaching figures/pdfs to the email (links are acceptable if preferred).

## For the codex agent
Use `CONTEXT.md` as the single source of truth for the workflow, selection heuristics, and QA checks. The weekly draft should be rendered from `template-weekly-mail.md` and stored under `mail/<YYYY-MM-DD>/` alongside the curated notes.

## Folder Layout
- Context files: `CONTEXT.md`, `template-weekly-mail.md`, `questionnaire.md`.
- Weekly folders (created on demand): `mail/<YYYY-MM-DD>/` with `curated-takeaways.md`, `gathered-changes.md`, `mail.md`, optional figures/scripts.
- Private artefacts: `mail/private/` for raw emails and attachments (e.g., `.eml`, `.pdf`). This folder is git-ignored; only `.gitkeep` is tracked. Summarize private content in `mail/archive/` without verbatim quotes.

Automation scripts are intentionally out of scope; weekly reports stay human-in-the-loop.
