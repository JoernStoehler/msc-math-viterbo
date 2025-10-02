# AGENTS.md — Thesis Authoring Conventions

This folder contains the LaTeX sources for the MSc thesis. Follow these conventions when editing any file under `thesis/`.

## Tooling & Workflow
- Draft and compile the thesis on Overleaf (or another LaTeX editor) — do **not** install LaTeX toolchains inside the devcontainer.
- Keep `main.tex` as the single entry point. Include chapter files with `\include{chapters/<name>}` and shorter front‑matter pieces with `\input{frontmatter/<name>}`.
- Commit only source assets (`.tex`, `.bib`, vector graphics such as `.pdf`/`.svg`, and bitmap images such as `.png`). Never commit build artefacts (e.g. `.aux`, `.log`, `.pdf`).
- Store figures in `thesis/figures/`; prefer vector formats when possible. Use descriptive filenames that match their caption slugs.

## Structure & Style
- Keep each major section or chapter in its own file inside `chapters/`. Name files with the pattern `<nn>-<slug>.tex` once the structure stabilises (e.g. `01-introduction.tex`).
- Use `% TODO:` comments sparingly to flag unfinished prose; remove them before submission.
- Use LaTeX macros for recurring notation. Declare them in `thesis/macros.tex` (create the file if needed) and include it from `main.tex`.
- Place bibliography entries in `references.bib`. Prefer `@article`/`@book` entries with full metadata and DOIs where available.
- When referencing images, use `\label` and `\ref` to keep cross‑references stable.

## Version Control Notes
- Describe substantive content changes in commit messages (e.g. “Add preliminaries chapter outline”).
- When importing figures generated from notebooks, export the final artefact into `thesis/figures/` and document the source notebook path in the figure caption or a comment.

