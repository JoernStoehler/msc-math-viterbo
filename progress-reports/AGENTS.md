# AGENTS.md — Weekly Progress Mail Workflow

This directory stores prompts, templates, and archived drafts for the weekly status mail that is sent to the supervisor.

## File Layout
- `weekly-mail-template.md` — editable scaffold for each week’s message.
- `weekly-mail-prompt.md` — instruction set Codex should follow to prepare the first draft.
- `drafts/` — store committed drafts named `YYYY-MM-DD-weekly-mail.md` after editing.

## Weekly Workflow
1. **Collect context**
   - Ensure the repository is up to date (pull the latest changes).
   - Review recent commits, open issues, and the roadmap (`docs/02-project-roadmap.md`).
2. **Generate a draft with Codex**
   - Run the weekly workflow (described in `docs/06-weekly-mail-workflow.md`).
   - Provide Codex with `weekly-mail-prompt.md` and any additional notes you want highlighted.
3. **Edit and finalise**
   - Save the generated draft into `drafts/YYYY-MM-DD-weekly-mail.md`.
   - Polish the tone, add personal comments, and double-check dates or meeting actions before sending.
4. **Archive**
   - Commit the final draft for that week together with any supplementary materials discussed with the supervisor.

## Style Notes
- Use British English consistently across emails unless the supervisor specifies otherwise.
- Keep paragraphs short and scannable: lead with outcomes, follow with supporting detail, end with next steps.
- Use bullet lists for enumerating tasks, blockers, or upcoming milestones.
- Refer to thesis sections using their working titles (e.g. “Chapter~\ref{chap:introduction} (Introduction)”).

