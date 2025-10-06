# Codex IDE — Initial Context Discussion and Interpretation

This note explains the parts of the initial context, how they interact and take precedence, and
highlights any potential sources of confusion or drift.

## Parts of the Context

- System message
  - Global runtime facts and constraints: knowledge cutoff, date, channel rules (`analysis`,
    `commentary`, `final`), verbosity default, image capability.
  - Applies to the entire session with highest precedence.

- Tool definitions (namespace `functions`)
  - Declares available tools: `shell`, `update_plan`, `view_image` with their parameter shapes.
  - These are the concrete APIs the agent can call.

- Developer message (Codex CLI agent guide)
  - Defines how to behave in this harness: preambles, planning, execution, validating work,
    approvals, formatting of final answers, tool usage (`apply_patch` via shell), and safety
    constraints.
  - Establishes precedence/interaction with repo `AGENTS.md` (direct system/developer/user
    instructions take precedence over AGENTS.md instructions; deeper AGENTS.md files override
    shallower ones for files in their scope).

- Repository policy (AGENTS.md) — provided inline by the user and also present at repo root
  - Project-wide conventions for Python code, docs, testing, performance, and CI. Applies to all
    tasks and files touched in this repo.
  - Emphasizes determinism, purity for math code, strict typing, Google-style docstrings, jaxtyping
    shapes, Ruff/Pyright gates, and `uv` for deps.

- Environment context (user-provided)
  - cwd: `/workspaces/msc-math-viterbo`
  - approvals: `never`
  - sandbox: `danger-full-access`
  - network: `enabled`
  - shell: `bash`
  - Interpretation: we must not request escalations; we can write files anywhere in the workspace;
    network is available; commands run under bash.

- User request (this task)
  - Create two docs: a verbatim snapshot of the initial context and a discussion/interpretation.
    Call out possible bugs or misunderstanding.

## Precedence and Interaction

- Order of precedence for behavior:
  1. System instructions
  1. Developer (Codex CLI) instructions
  1. User instructions/request for this task
  1. Repo `AGENTS.md` (policy) for code/style/testing when touching repo files

- Tooling and edits:
  - The harness exposes `functions.shell` (which can run `apply_patch`), `functions.update_plan`,
    and `functions.view_image`.
  - File edits are performed by invoking the `apply_patch` command via `shell` as documented.

- Approvals/sandbox:
  - With `approval_policy=never`, we cannot ask for escalations; we must proceed autonomously within
    the provided privileges.
  - `danger-full-access` means normal filesystem constraints are lifted. We still act conservatively
    and keep diffs minimal per repo policy.

## What I Did Based on This Context

- Created `docs/96-codex-ide-context.md` containing the verbatim snapshot (system, tools, developer,
  AGENTS.md inline, environment context, and your task request).
- Authored this discussion in `docs/97-codex-ide-context-discussion.md` explaining parts,
  precedence, and potential drift.

## Potential Confusions, Drift, or Minor Gotchas

- apply_patch invocation
  - The developer guide says “Use the apply_patch tool,” but there is no separate
    `functions.apply_patch`; instead, `apply_patch` is a shell command invoked via
    `functions.shell`. This is expected for Codex CLI, but can be confusing if you look only at the
    tool list.

- Approvals guidance vs. current mode
  - The developer guide describes multiple approval modes (untrusted/on-failure/on-request/never)
    with escalation scenarios. In this session the mode is `never`, so the escalation guidance does
    not apply; we proceed without requesting approvals.

- Formatting guidance vs. minimal formatting
  - The system message discourages heavy formatting in chat unless needed; the developer guide
    contains detailed formatting rules for chat answers. These coexist: keep chat succinct and
    structured when helpful; heavy formatting is fine in repo docs (like these files).

- AGENTS.md duplication
  - You supplied AGENTS.md inline, and the repo also contains `AGENTS.md`. Treat the repository file
    as the source of truth for code changes. The inline copy appears consistent with the repo
    version.

- Image capabilities
  - System says image input is enabled and a `view_image` tool exists, but this task does not use
    image inputs. Mentioned only to clarify capabilities.

## Clarifications on How Context Works

- The “initial context” comprises: system message, tool definitions, developer message,
  user-provided repo policy (AGENTS.md), environment context, and your request. Those are the levers
  that shape agent behavior.
- Channel discipline: messages are emitted using `analysis` (internal), `commentary` (preambles/tool
  calls), and `final` (user-facing results). Only `final` is intended as the polished handoff;
  `analysis` is not surfaced.
- Repo policy is enforced for files we modify; since this task is docs-only, code-style constraints
  (Ruff/Pyright) are not engaged, but we still keep changes small and focused.

## Bottom Line

- Your mental model is correct: the session is defined by system + tools + developer instructions,
  then the repo’s AGENTS.md and the environment context, and finally your task request. The only
  subtlety is that `apply_patch` is a shell-invoked command rather than a first-class tool;
  otherwise the context appears consistent with the Codex CLI design.
