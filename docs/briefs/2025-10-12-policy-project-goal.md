---
status: adopted
created: 2025-10-12
workflow: policy
summary: State the overarching research aim and roadmap for probing the Viterbo conjecture.
---

# Policy: Project Goal and Context — Viterbo Conjecture

## Context

- The project investigates the Viterbo conjecture, which compares symplectic capacities and volume for convex domains in even-dimensional Euclidean space.
- Modern code lives under the `viterbo` namespace; legacy stacks such as `viterbo.symplectic` were retired during the modernization programme.
- Higher-dimensional (≥6D) capacity, cycle, and spectrum support remains deferred while we control combinatorial blow-up and runtime budgets.

## Objectives

- Survey the landscape of symplectic capacities (Gromov width, Hofer–Zehnder, Ekeland–Hofer, ECH) and the precise formulation of the Viterbo conjecture.
- Catalogue known partial results, equality cases, and counterexamples, highlighting proof strategies across generating functions, symplectic embeddings, and convex-geometric techniques.
- Produce worked examples and computational experiments on convex bodies to build intuition around the volume–capacity relationship.
- Keep release messaging explicit about supported dimensions and roadmap items, especially when higher-dimensional features are deferred.

## Execution

1. Foundations — compile precise statements of the conjecture, identify invariants, and align notation across code and docs.
2. Results & Techniques — study the strongest known results and their methods; map implications and dependencies between them.
3. Examples & Experiments — implement or bound capacities for canonical convex bodies and explore parametric families numerically.
4. Synthesis — assemble a concise survey with diagrams, results maps, and optional computational experiments that surface insights.

## Dependencies & Cross-links

- Reading priorities live in [2025-10-12-workflow-reading-list.md](2025-10-12-workflow-reading-list.md).
- Active exploration programmes are tracked in briefs such as [2025-10-12-task-spectrum-4d-baseline.md](2025-10-12-task-spectrum-4d-baseline.md).
- Thesis topic transcription and acceptance criteria sit in [2025-10-12-workflow-thesis-topic-probing-viterbo.md](2025-10-12-workflow-thesis-topic-probing-viterbo.md).

## Acceptance

- The policy remains aligned with the adopted modernization architecture and up-to-date roadmap briefs.
- Public APIs referenced here point to the `viterbo` namespace and document any deferred dimensions explicitly.

## Status Log

- 2025-02-14 — Migrated legacy `docs/01-project-goal.md` into the briefs tree; removed references to the deprecated `viterbo.symplectic` namespace.
