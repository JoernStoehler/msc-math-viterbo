# Codex Initial Contexts — CLI vs IDE vs Cloud

> Non‑normative comparison: This document contrasts past context captures. Policies and workflows
> are defined in `AGENTS.md` and repository configs; if anything here conflicts with those sources,
> prefer `AGENTS.md` and configs.

This document compares the initial contexts of three related agent surfaces — Codex CLI, Codex IDE,
and Codex Cloud — with a focus on how each product frames and constrains an autonomous agent at
session start. Each statement below is explicitly labeled as one of: Observation, Quote (with source
file and start line), Interpretation, Hypothesis, or External Observation. The goal is a neutral,
sourced comparison that separates what the repository shows from reasoned inferences.

---

## Scope and Sources

- Observation — This comparison draws on five in‑repo sources:
  - Context snapshots: `docs/94-codex-cli-context.md`, `docs/96-codex-ide-context.md`,
    `docs/98-codex-cloud-context.md`.
  - Companion discussions: `docs/95-codex-cli-context-discussion.md`,
    `docs/97-codex-ide-context-discussion.md`, `docs/99-codex-cloud-context-discussion.md`.
  - Policy baseline: `AGENTS.md` (repo‑wide conventions and workflows).

- Interpretation — The three “context” files are verbatim captures of the initial session
  prompts/configs for their respective surfaces. The “discussion” files explain those captures from
  prior runs and help disambiguate intent and precedence.

- Hypothesis — Where CLI and IDE contexts appear nearly identical, it likely reflects shared harness
  logic rather than accidental duplication. Differences in the Cloud context suggest a distinct,
  PR‑oriented harness.

---

## High‑Level Identity and Role

- Quote — CLI identity: `docs/94-codex-cli-context.md:57`

  > You are a coding agent running in the Codex CLI, a terminal-based coding assistant. Codex CLI is
  > an open source project led by OpenAI. You are expected to be precise, safe, and helpful.

- Quote — IDE identity (note that it references CLI): `docs/96-codex-ide-context.md:76`

  > You are a coding agent running in the Codex CLI, a terminal-based coding assistant. Codex CLI is
  > an open source project led by OpenAI. You are expected to be precise, safe, and helpful.

- Quote — Cloud identity: `docs/98-codex-cloud-context.md:1`

  > You are ChatGPT, a large language model trained by OpenAI.

- Observation — CLI and IDE present the agent explicitly as “a coding agent running in the Codex
  CLI,” whereas Cloud presents the agent as “ChatGPT.”

- Interpretation — CLI/IDE frame the agent as a task‑oriented coding assistant embedded in a
  harness; Cloud frames a more general ChatGPT‑style agent adapted for repository work.

---

## Tooling and Namespaces

- Quote — CLI tools (functions namespace): `docs/94-codex-cli-context.md:17`

  > ## Namespace: functions

- Quote — IDE tools (functions namespace): `docs/96-codex-ide-context.md:1` and `:22`

  > # Codex IDE — Initial Context Snapshot (verbatim)
  >
  > ...
  >
  > # Valid channels: analysis, commentary, final. Channel must be included for every message.

  And earlier sections enumerate `functions.shell`, `functions.update_plan`, and
  `functions.view_image` (see `docs/96-codex-ide-context.md` around lines 30–70 for the full tool
  schemas).

- Quote — Cloud tools (container and browser): `docs/98-codex-cloud-context.md:92` onward shows the
  `container` namespace, and `:132` onward shows `browser_container`.

- Observation — CLI/IDE expose a compact toolset via a single `functions` namespace: `shell` (for
  commands and file edits via `apply_patch`), `update_plan` (task plans), and `view_image`.

- Observation — Cloud exposes broader orchestration via `container` (processes, filesystem, network)
  and `browser_container` (Playwright‑driven screenshots), with an explicit PR workflow tool
  (`make_pr`) governed by the system instructions.

- Interpretation — CLI/IDE tools target quick, iterative coding and planning inside a terminal
  harness; Cloud tools target end‑to‑end repository automation including commits, PRs, and optional
  UI validation.

---

## Sandbox, Approvals, and Interactivity

- Quote — CLI sandbox/approvals section header: `docs/94-codex-cli-context.md:205`

  > ## Sandbox and approvals

- Quote — IDE sandbox/approvals section header: `docs/96-codex-ide-context.md:224`

  > ## Sandbox and approvals

- Quote — Cloud non‑interactive directive: `docs/98-codex-cloud-context.md:54`

  > This is a non-interactive environment. Never ask for permissions to run a command, just do it.

- Observation — CLI/IDE define a matrix of filesystem sandboxing (read‑only, workspace‑write,
  danger‑full‑access), network sandboxing (restricted/enabled), and approval policies
  (untrusted/on‑failure/on‑request/never). Cloud eliminates approvals and prescribes non‑interactive
  execution.

- Interpretation — CLI/IDE assume conversational escalation patterns suitable for live developer
  collaboration; Cloud favors autonomous runs where the agent executes without pausing for user
  approval.

---

## Output Style and Citation Rules

- Quote — CLI channels and token budget: `docs/94-codex-cli-context.md:13` and `:15`

  > # Valid channels: analysis, commentary, final. Channel must be included for every message.
  >
  > # Juice: 200

- Quote — IDE channels and token budget: `docs/96-codex-ide-context.md:22` and `:24`

  > # Valid channels: analysis, commentary, final. Channel must be included for every message.
  >
  > # Juice: 200

- Quote — Cloud channels and token budget: `docs/98-codex-cloud-context.md:162` and `:164`

  > # Valid channels: analysis, commentary, final. Channel must be included for every message.
  >
  > # Juice: 64

- Quote — CLI/IDE final answer style: `docs/96-codex-ide-context.md:305`

  > You are producing plain text that will later be styled by the CLI. Follow these rules exactly.
  > Formatting should make results easy to scan, but not feel mechanical. Use judgment to decide how
  > much structure adds value.

- Quote — Cloud citation and formatting mandate: `docs/98-codex-cloud-context.md:18`

  > # Citations instructions
  >
  > - You must add citations to the final response (not the body of the PR message) where relevant.
  >   Citations reference file paths and terminal outputs with the following formats:
  >   `【F:<file_path>†L<line_start>(-L<line_end>)?】` ...

- Observation — CLI/IDE encourage light structure (bullets, short headers) and discourage heavy
  Markdown in chat; Cloud requires structured Markdown, formal citations using `【F:...】`/terminal
  chunk IDs, and even emoji status markers for test results.

- Interpretation — Cloud’s stricter output/citation regime aligns with auditability in PR workflows;
  CLI/IDE optimize for fast, readable collaboration in a terminal/IDE pane.

---

## File Editing and Change Management

- Quote — CLI/IDE “apply_patch” usage: `docs/94-codex-cli-context.md:188`

  > Use the `apply_patch` tool to edit files (NEVER try `applypatch` or `apply-patch`, only
  > `apply_patch`): {"command":["apply_patch","\*\*\* Begin Patch\\n\*\*\* Update File: >
  > path/to/file.py\\n@@ def example():\\n- pass\\n+ return 123\\n\*\*\* End Patch"]}

- Observation — In CLI/IDE, `apply_patch` is invoked as a shell subcommand through
  `functions.shell`; there is no dedicated `functions.apply_patch` tool.

- Quote — Cloud commit/PR workflow: `docs/98-codex-cloud-context.md:3`

  > Commit your changes on the current branch.

- Quote — Cloud PR creation and rules (excerpted): `docs/98-codex-cloud-context.md:36`–`:53` (see
  full file for details)

  > Use the make_pr tool to create a pull request after running git commit, with an appropriate
  > title and body.
  >
  > - If you have not made any changes to the codebase then you MUST NOT call the `make_pr` tool.
  > - I.e. it is strictly forbidden to end the turn either of these states: ...

- Observation — Cloud prescribes “commit then make_pr” with strict sequencing and forbids
  out‑of‑sequence PR calls. CLI/IDE never mention a PR tool within the initial context.

- Interpretation — CLI/IDE prefer local diffs applied via patches; Cloud elevates to repository
  lifecycle primitives (commit, PR) with strict guardrails.

---

## Personality, Preambles, and Planning

- Quote — CLI/IDE personality (excerpt): `docs/94-codex-cli-context.md:43`

  > Your default personality and tone is concise, direct, and friendly. You communicate efficiently,
  > always keeping the user clearly informed about ongoing actions without unnecessary detail.

- Quote — CLI/IDE preambles: `docs/94-codex-cli-context.md:84`–`:111` include concrete preamble
  examples such as “I’ve explored the repo; now checking the API route definitions.”

- Quote — CLI/IDE planning tool: `docs/94-codex-cli-context.md:113` onward describes `update_plan`
  use and quality standards for plans.

- Observation — CLI/IDE dedicate substantial guidance to writing brief preambles and maintaining a
  stepwise plan via `update_plan`. Cloud’s initial context provides templates for PR‑style summaries
  and test reporting instead.

- Interpretation — CLI/IDE align with interactive, stepwise development; Cloud aligns with batch
  execution with structured end‑of‑run reporting.

---

## Environment Framing and AGENTS.md

- Quote — AGENTS.md mentions Codex Cloud explicitly: `AGENTS.md:25`

  > Agents run in fresh, ephemeral containers (Codex Cloud) with this AGENTS.md and the task brief.
  > The maintainer merges or closes PRs.

- Observation — The repository’s AGENTS.md assumes a Golden Path applicable to all agents and
  explicitly acknowledges Codex Cloud as the execution substrate for ephemeral containers. CLI/IDE
  sessions load AGENTS.md into their developer instructions; Cloud also includes its own “AGENTS.md
  spec” section tailored to containerized discovery, including system‑wide AGENTS.md locations.

- Quote — Cloud AGENTS.md discovery scope (excerpt): `docs/98-codex-cloud-context.md:5`–`:17`

  > AGENTS.md files may provide instructions about PR messages ... AGENTS.md files need not live
  > only in Git repos. For example, you may find one in your home directory.

- Interpretation — Cloud anticipates cross‑filesystem discovery of policy files and integrates
  AGENTS.md into PR messaging; CLI/IDE emphasize conformance to project policy when touching files
  but do not couple to PR messaging.

---

## Session Budgets and Rigor

- Observation — Token budgets differ: CLI/IDE set `Juice: 200` while Cloud sets `Juice: 64`. See
  `docs/94-codex-cli-context.md:15`, `docs/96-codex-ide-context.md:24`, and
  `docs/98-codex-cloud-context.md:164`.

- Interpretation — Cloud encourages concise, templated outputs with strict citations; CLI/IDE allow
  more expansive reasoning and planning during interactive development.

---

## Captured Session Environment Differences

- Quote — CLI discussion (prior run) environment: `docs/95-codex-cli-context-discussion.md:9`

  > Environment context: Effective runtime constraints for this session:
  > `cwd=/workspaces/msc-math-viterbo`, approvals=`on-request`, sandbox=`workspace-write`,
  > network=`restricted`, shell=`bash`.

- Quote — IDE discussion (this repo) environment: `docs/97-codex-ide-context-discussion.md:23`–`:29`

  > - cwd: `/workspaces/msc-math-viterbo`
  > - approvals: `never`
  > - sandbox: `danger-full-access`
  > - network: `enabled`
  > - shell: `bash`

- Observation — CLI vs IDE differences here reflect specific run configurations, not hardwired
  product semantics. The initial context documents treat these as variables communicated at session
  start.

- Interpretation — The harness supports multiple approval/sandbox modes; individual sessions on
  CLI/IDE can legitimately differ without implying product‑level divergence.

---

## Safety, Determinism, and Error Handling

- Quote — CLI/IDE caution against guessing: `docs/94-codex-cli-context.md:172`

  > Do NOT guess or make up an answer.

- Quote — Cloud thoroughness and structure for answers: `docs/98-codex-cloud-context.md:58`–`:76`

  > If you are answering a question, you MUST cite the files referenced and terminal commands you
  > used to answer the question. Be EXTREMELY thorough in your answer, and structure your response
  > using Markdown ... The user really likes detailed answers to questions--you should not be terse!

- Observation — All three surfaces emphasize correctness; Cloud adds auditability via citations and
  templates, while CLI/IDE emphasize conversational discipline and minimal formatting in chat.

---

## Side‑by‑Side Summary (Selected Axes)

- Observation — Identity
  - CLI: “coding agent running in the Codex CLI” (`docs/94-codex-cli-context.md:57`).
  - IDE: Same phrasing as CLI (`docs/96-codex-ide-context.md:76`).
  - Cloud: “You are ChatGPT ...” (`docs/98-codex-cloud-context.md:1`).

- Observation — Tools
  - CLI/IDE: `functions.shell`, `functions.update_plan`, `functions.view_image`.
  - Cloud: `container`, `browser_container`, plus PR flow (`make_pr`) in system instructions.

- Observation — Approvals & Interactivity
  - CLI/IDE: Rich approvals model and preambles; interactive style.
  - Cloud: Non‑interactive; “never ask for permissions.”

- Observation — Output Style
  - CLI/IDE: Plain text with light structure; discourage heavy Markdown in chat.
  - Cloud: Strong Markdown + citations + emoji status in final messages.

- Observation — Change Management
  - CLI/IDE: File edits via `apply_patch` patches.
  - Cloud: Git commits and PR creation via `make_pr` after commit.

- Observation — Token Budget
  - CLI/IDE: 200
  - Cloud: 64

---

## What’s Likely Shared vs. Truly Different

- Interpretation — Shared core behaviors across CLI and IDE:
  - Personality and preamble guidance for interactive collaboration.
  - The planning model (`update_plan`) and the same patch/edit flow (`apply_patch` via shell).
  - Output structure rules optimized for terminal/IDE chat rendering.

- Interpretation — Material differences in Cloud:
  - Role framing (“ChatGPT”), explicit PR lifecycle, mandatory citations, and non‑interactive
    execution.
  - Broader tools (container/browser) suitable for end‑to‑end automation, including UI checks.

- Hypothesis — IDE appears to reuse CLI’s initial context for consistency inside editor panes;
  differences in behavior vs. pure CLI are likely to be environmental (e.g., approvals mode) rather
  than spec differences.

- Hypothesis — Cloud’s constraints are designed to produce review‑ready artifacts (commits, PRs,
  reproducible citations) without human‑in‑the‑loop approvals, trading interactivity for
  auditability.

---

## External Observations

- External Observation — In the current session (from the developer context), we see
  `approval_policy=never`, `sandbox_mode=danger-full-access`, and `network_access=enabled`, which
  matches the IDE discussion’s captured environment and the non‑interactive stance seen in Cloud.
  This highlights that per‑session environment settings can blur product lines operationally even
  when the initial context templates differ.

- External Observation — The repository’s `AGENTS.md` is explicitly intended to govern “all tasks”
  and “all agents,” which helps explain why many behavioral constraints are consistent across
  surfaces (types, tests, docs). See `AGENTS.md:245`.

---

## Limitations

- Observation — These snapshots reflect specific runs; the harnesses may evolve. Even within a
  surface (e.g., CLI), approvals/sandbox/network modes are variable and communicated at session
  start.

- Hypothesis — Some textual overlaps between CLI and IDE (IDE referencing “Codex CLI” verbatim)
  likely indicate a deliberate choice to impose a single behavioral spec across local developer
  tools, not an omission.

- Observation — This comparison does not rely on external documentation; it is grounded in the
  repository files listed above.

---

## Conclusion

- Observation — Codex CLI and Codex IDE share essentially the same initial behavioral specification:
  a concise, interactive coding agent with preambles, stepwise planning, and patch‑based edits.
  Differences across sessions stem mainly from runtime environment settings
  (approvals/sandbox/network) rather than distinct specs.

- Observation — Codex Cloud presents a different initial context: a ChatGPT‑framed agent with
  non‑interactive execution, mandatory citations, PR/commit workflow enforcement, and broader
  orchestration tools. The Cloud context optimizes for auditable, end‑to‑end automation rather than
  conversational iteration.

- Interpretation — The three surfaces emphasize different trade‑offs: CLI/IDE for interactive
  developer productivity with clear, minimal formatting and live planning; Cloud for reproducible,
  review‑ready outputs with strict structure and lifecycle integration.

- Hypothesis — Unifying repository policy (AGENTS.md) and shared core conventions reduce cognitive
  overhead when moving between surfaces; the primary differences are the guardrails and ergonomics
  appropriate to each environment (terminal/IDE vs. CI‑like Cloud runs).
