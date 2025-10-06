# RFC: Managing Experimental Training Artifacts

- **Status**: RFC
- **Last updated**: 2025-10-05
- **Owner / DRI**: Unassigned
- **Reviewers**: Maintainer (PI)
- **Related docs**: AGENTS.md

## 1. Context and motivation

- Upcoming research work spans logistic regression baselines through heavier graph and neural
  models; we need a single way to track runs, reuse artefacts, and communicate results.
- Synthetic datasets may be generated with costly numerical routines; reusing them avoids duplicated
  compute and ensures reproducibility.
- Downstream collaborators (data scientists, ML researchers) should be able to read artefacts
  without reverse-engineering bespoke tooling, even when we mix JAX with PyTorch or TensorFlow
  components.
- The project has a ~6-month lifetime, so any workflow should favour turnkey services and minimise
  long-term upkeep.

## 2. Requirements and constraints

- **Reproducibility**: Record provenance for datasets, code revisions, random seeds, and training
  hyperparameters. Runs must be recreatable months later.
- **Standard formats**: Metrics, checkpoints, and datasets should land in widely recognised formats
  (Parquet, HDF5, TensorBoard event files, JSON/YAML configs) rather than bespoke binaries.
- **Artefact portability**: Facilitate sharing across machines and cloud storage; support a
  promotion path from local experiments to long-term storage (object store or registry).
- **JAX-first training; standard analysis formats**: Training code remains JAX-first. We emit a
  small set of common, widely readable formats (TensorBoard, CSV/JSONL metrics, Parquet datasets).
  Optional exports (e.g., ONNX) are opt-in, not default.
- **Scalability**: Handle small logistic-regression sweeps as well as multi-hour GPU training runs
  without changing workflows.
- **Security & governance**: Authenticate against private storage, manage large binaries outside Git
  history, and respect AGENTS.md escalation triggers when deviating from policy.
- **Operational simplicity**: Minimise maintenance overhead; prefer hosted, mainstream services with
  widely adopted APIs that have remained stable for 6+ months.
- **Agent ergonomics**: Codex agents execute day-to-day workflows, so tooling should align with
  popular, well-documented APIs instead of bespoke scripts.
- **Secret management**: Keep W&B and GitHub credentials in GitHub Actions secrets plus Codex Cloud
  secrets, and surface them locally via `.env` guided by `.env.example`.
- **Project horizon**: Optimise for a 6-month research window; avoid platforms that demand long
  ramp-up or dedicated DevOps ownership.
- **Artefact size expectations**: Datasets and checkpoints stay under ~1 GB (<=50M parameter models,
  synthetic polytope corpora), keeping GitHub-centric workflows feasible.

Clarifying “standard analysis formats”

- Training loops remain JAX-first. We do not mix frameworks inside training.
- Outputs target common tools, not “every possible tool”:
  - Metrics: TensorBoard event files + `metrics.csv`/`metrics.jsonl` in
    `{step, split, metric, value}` schema.
  - Datasets: Parquet (tabular); HDF5/Zarr (dense arrays).
  - Checkpoints: Orbax (primary) only by default. Optional ONNX or PyTorch `state_dict` exports are
    opt-in and produced only when explicitly requested.
- We never dump full parameter tensors to CSV/JSON.
- Each run includes `metadata.json` covering commit, seed(s), resolved config, environment, and
  brief architecture description.
- Rationale: teammates can inspect results with TensorBoard/pandas/NumPy without bespoke code, while
  we avoid supporting every downstream stack.

## 3. Tooling landscape

### 3.1 Experiment tracking platforms

Hosted trackers dominate the landscape and align with the 6-month timeline; we prioritise offerings
with mature SDKs that Codex agents have seen before. With Weights & Biases selected as the default,
the remaining entries act as contingency options if constraints shift (budget, offline mode, etc.).

| Platform                    | Strengths                                                                                                                   | Considerations                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Weights & Biases (selected) | SaaS with rich dashboards, sweeps, reports; Python API works for JAX/Flax, PyTorch, TensorFlow; good collaboration features | Default choice; monitor private-project spend and keep exporting snapshots for longevity |
| MLflow (managed)            | Familiar API with large community adoption; Databricks Community/Managed MLflow reduces ops load; supports model registry   | Managed tiers can be limited; self-host path likely too costly for this project horizon  |
| Neptune                     | SaaS / hybrid; strong metadata schema, comparison tools; integrates with major frameworks                                   | Paid plans for private usage; smaller community than W&B                                 |
| Comet                       | SaaS experiment tracker; metric visualisations, experiment comparison                                                       | Similar trade-offs to W&B; smaller market share but stable API                           |
| Aim (self-host)             | Open-source UI with fast local search; simple integration                                                                   | Requires ongoing ops; deprioritise unless a low-touch hosted option becomes available    |

#### Weighted decision matrix (hosted trackers)

| Criterion                                    | Weight | Weights & Biases | Managed MLflow | Neptune / Comet |
| -------------------------------------------- | ------ | ---------------- | -------------- | --------------- |
| Operational overhead (lower is better)       | 0.30   | 5                | 4              | 4               |
| Ecosystem maturity & agent familiarity       | 0.25   | 5                | 4              | 3               |
| Cost predictability for private work         | 0.15   | 3                | 4              | 3               |
| Feature depth (dashboards, sweeps, registry) | 0.15   | 5                | 3              | 4               |
| Portability & export friendliness            | 0.15   | 4                | 5              | 4               |
| **Weighted score (0–5)**                     | —      | **4.55**         | **4.00**       | **3.60**        |

Scoring uses a 1–5 scale (5 = best against the criterion). Weighted totals reflect our current
priorities: minimal ops and widely known tooling dominate, while cost and portability remain notable
but lighter factors.

With privacy out of scope and W&B’s leading score, Weights & Biases is locked in as the default
tracker. Managed MLflow or Neptune/Comet stay as fallback options if budget, connectivity, or policy
changes make W&B impractical.

### 3.2 Dataset and artefact storage

Git LFS is the canonical storage path. Datasets and checkpoints stay under GitHub limits, so pointer
files in the repo plus LFS objects give us versioned artefacts with minimal ceremony. Optional
add-ons (GitHub Releases milestones, Hugging Face Hub for public sharing) can be layered later, but
they are not part of the default golden path. Heavier systems (DVC, object stores, LakeFS) remain
out of scope unless artefacts unexpectedly outgrow Git LFS or policy changes demand them.

### 3.3 Model checkpoints and serialisation

Note: We are not proposing to mix frameworks in training loops. Interoperability is limited to
exporting/consuming artefacts in widely supported formats when beneficial. Exports beyond Orbax
(e.g., ONNX, PyTorch `state_dict`) are opt-in only, not part of the default path.

- **JAX/Flax + Orbax**: Preferred for JAX-first; stores PyTree params and optimizer state with
  structured metadata and checksums.
- **PyTorch `state_dict`**: When using PyTorch modules, store as `.pt` plus a JSON sidecar
  describing architecture and training details.
- **ONNX**: Framework-neutral graph format; useful for deployment artefacts and compatibility
  checks.
- **TensorFlow SavedModel**: If TensorFlow components appear, follow SavedModel conventions to
  capture graph + weights.
- **Accompanying metadata**: Always emit a `metadata.json` containing commit hash, dataset
  fingerprint, config, metrics summary, and training environment (hardware, library versions).

### 3.4 Metric logging and training curves

- **TensorBoard event files**: De-facto standard; widely supported across frameworks and tools
  (MLflow, W&B import). Compatible with JAX via `tensorboardX` or Flax metrics helpers.
- **CSV / JSONL exports**: Provide tabular `metrics.csv` or `metrics.jsonl` with schema
  `{step, split, metric, value}` for quick analysis and archival.
- **MLflow metrics / W&B history**: Native storage when using those platforms; ensure we mirror key
  metrics to the standard CSV/JSONL to keep portability.
- **Visualization notebooks**: Store notebooks separately from raw metrics; notebooks rely on stable
  artefact URIs.

### 3.5 Configuration and reproducibility tooling

- **Hydra / OmegaConf**: Hierarchical YAML configs with sweep support; integrates with JAX training
  scripts.
- **absl.flags / gin-config**: Lightweight options popular in JAX ecosystems (e.g., Flax examples).
  gin-config aids declarative architecture definitions.
- **Sacred + Omniboard**: Combines config management with experiment logging; overlaps with
  MLflow/Aim functionality.
- **uv + Makefile integration**: Keep golden-path commands (e.g., `make train EXPERIMENT=<name>`)
  reproducible; optionally emit `run.json` with resolved config.

## 4. Popular modular stacks

The stacks below emphasise low-maintenance combinations first, reflecting the constraint that the
Maintainer's time is better spent on research than platform admin.

- **Default: W&B + GitHub-native storage**: Weights & Biases (hosted) for runs, metrics, and
  artefact registry + Git LFS / release assets for datasets and checkpoints under 1 GB. Quick to
  adopt, minimal ops, and well-known to Codex agents.
- **Managed MLflow pathway**: Use Databricks Community/Managed MLflow for tracking and registry +
  Git LFS for artefacts, exporting TensorBoard logs for interoperability. Evaluate only if hosted
  MLflow features match needs; skip self-host.
- **Hugging Face sharing layer**: Pair a hosted tracker (W&B or Neptune) with Hugging Face Hub for
  publishing curated datasets/models. Keeps collaboration light-weight while leaning on standard
  dataset cards.
- **Minimalist local**: Git + Git LFS + structured folder layout (`artefacts/<run_id>/...`) +
  CSV/JSON metrics + TensorBoard logs. Lowest friction for quick experiments; lacks dashboards but
  zero vendor dependency.

## 5. Conventions on the golden path

- **Directory structure**: `artefacts/<dataset|model>/<slug>/<version>/` with sidecar
  `metadata.json` and checksums (`SHA256`). Symlink latest successful run for quick access.
- **Dataset format & provenance**: Use Apache Parquet for tabular data, HDF5/Zarr for dense arrays;
  include README.md outlining generation process and parameters plus a `metadata.json` capturing
  commit, seeds, config hash, and generation params. Store dataset generation script and random
  seeds.
- **Training configs**: Persist resolved config per run as YAML/JSON plus git commit hash. Avoid
  relying solely on CLI flags.
- **Type support**: Favour client libraries with distributed type hints; when absent, add local
  `typings/` stubs so Pyright stays informative.
- **Metrics bundle**: Emit TensorBoard event file, `metrics.jsonl`, and `summary.yaml` (best
  metrics, early stopping step, training duration).
- **No raw-weights CSV**: Do not export parameters/gradients to CSV/JSON; rely on checkpoints.
- **Model export**: Save primary framework checkpoint (Orbax, `state_dict`, etc.) plus optional ONNX
  export for interoperability. Tag each with schema version.
- **Run registry**: Maintain an index file (W&B tables plus a lightweight `artefacts/runs.parquet`)
  to cross-link datasets, configs, metrics, and checkpoints.
- **W&B project conventions**:
  - Single project named `viterbo-experiments` for all training runs.
  - Mandatory run tags: `experiment`, `dataset`, `model`, `run_type`; optional `sweep` for batch
    jobs.
  - Run names follow `YYYYMMDD_<dataset>_<model>_<variant>`.
  - Artifact labels: `<YYYYMMDD>-<slug>@<git-sha>` where the slug matches the run name semantics.
  - Use W&B tables for curated evaluation summaries; raw artefacts remain in Git LFS.
- **Secrets bootstrap**: Track `.env.example` at repo root with a `WANDB_API_KEY` placeholder. Each
  contributor copies to `.env`, fills their key, and loads it before running Make targets. GitHub
  credentials continue to come from the provisioned environment per AGENTS.md.

## 6. Decisions locked in (with rationale)

For each area below, we evaluated realistic options against our criteria (5 = best). The selected
option in each table achieved the highest weighted score and is now part of the golden path.

### 6.1 W&B workspace layout

| Criterion (weight)                    | Single project `viterbo-experiments` **(selected)** | Project per major task |
| ------------------------------------- | --------------------------------------------------- | ---------------------- |
| Setup/admin overhead (0.35)           | **5**                                               | 3                      |
| Cross-run comparison (0.30)           | **5**                                               | 3                      |
| Agent familiarity & docs reuse (0.20) | **5**                                               | 3                      |
| Dashboard focus (0.15)                | 3                                                   | **5**                  |
| **Weighted score (0–5)**              | **4.70**                                            | 3.60                   |

Outcome: Single W&B project `viterbo-experiments` with mandatory tags (`experiment`, `dataset`,
`model`, `run_type`) and run naming `YYYYMMDD_dataset_model_variant` remains the standard.
Project-per-task stays as contingency if we ever need stricter dashboard isolation.

### 6.2 Dataset provenance bundle

| Criterion (weight)                      | Minimal `metadata.json` + README **(selected)** | Frictionless datapackage |
| --------------------------------------- | ----------------------------------------------- | ------------------------ |
| Implementation effort (0.40)            | **5**                                           | 3                        |
| Expressiveness / reproducibility (0.35) | 3                                               | **5**                    |
| Popularity in similar projects (0.15)   | **4**                                           | 3                        |
| Tooling support in repo (0.10)          | **4**                                           | 3                        |
| **Weighted score (0–5)**                | **4.30**                                        | 3.85                     |

Outcome: Lightweight `metadata.json` + README bundle is locked in. Revisit Frictionless only if
external publishing or stricter validation requirements arise.

### 6.3 Credential distribution

| Criterion (weight)                    | W&B key via GitHub/Codex secrets + `.env` **(selected)** | Central secrets manager |
| ------------------------------------- | -------------------------------------------------------- | ----------------------- |
| Setup time (0.40)                     | **5**                                                    | 2                       |
| Security for six-month horizon (0.30) | 4                                                        | **5**                   |
| Agent ergonomics (0.20)               | **5**                                                    | 3                       |
| Maintenance overhead (0.10)           | **5**                                                    | 3                       |
| **Weighted score (0–5)**              | **4.70**                                                 | 3.05                    |

Outcome: Store the W&B API key in GitHub Actions secrets and Codex Cloud secrets; surface it locally
through `.env` (documented via `.env.example`). Git credentials continue to come from the
provisioned environment. A dedicated secrets manager stays optional for future expansions.

### 6.4 Checkpoint exports

| Criterion (weight)                         | Orbax only + on-demand converters **(selected)** | Orbax + automatic ONNX |
| ------------------------------------------ | ------------------------------------------------ | ---------------------- |
| Alignment with JAX-first constraint (0.35) | **5**                                            | 4                      |
| Operational simplicity (0.30)              | **5**                                            | 3                      |
| Downstream portability (0.25)              | 3                                                | **5**                  |
| Failure surface / deps (0.10)              | **5**                                            | 3                      |
| **Weighted score (0–5)**                   | **4.50**                                         | 3.95                   |

Outcome: Save only Orbax checkpoints by default; document a manual conversion recipe for data
scientists who need other formats.

### 6.5 Automation & retention

| Criterion (weight)                            | Make targets + monthly manual tidy **(selected)** | CI sweeps + automated retention |
| --------------------------------------------- | ------------------------------------------------- | ------------------------------- |
| Implementation effort (0.35)                  | **5**                                             | 3                               |
| Repeatability for agents (0.30)               | **4**                                             | **5**                           |
| Risk of accidental deletion / over-run (0.20) | **5**                                             | 3                               |
| CI cost / compute usage (0.15)                | **5**                                             | 2                               |
| **Weighted score (0–5)**                      | **4.60**                                          | 3.35                            |

Outcome: Extend the Makefile with `make train`, `make evaluate`, `make publish`, and rely on a
monthly reminder (already scheduled) to prune W&B runs and large LFS blobs. CI-driven sweeps remain
a future option if volume grows.

## 7. Suggested next steps

1. Implement the W&B logging path on a logistic-regression toy project, capturing configs,
   TensorBoard events, CSV metrics, and an Orbax checkpoint.
1. Prototype dataset publishing pipeline: generate a synthetic dataset, store as Parquet with
   metadata, and push via Git LFS to validate workflow friction.
1. Finalise the `metadata.json` / `summary.yaml` schema using at least two training scripts (JAX
   baseline + PyTorch reference consumer) to ensure consistency.
1. Extend the Makefile with `make train`, `make evaluate`, and `make publish`; document W&B usage,
   tag requirements, and secrets loading steps in the developer guide.
1. Confirm SaaS account needs (e.g., W&B seats, Git LFS quota) with the Maintainer before
   operationalising.

## 8. Risks and mitigations

- **Operational overhead**: Self-hosted trackers/storage demand upkeep. Mitigate by defaulting to
  managed offerings and keeping contingency scripts only as a fallback.
- **Vendor lock-in**: SaaS trackers can become hard dependencies. Keep export scripts (e.g.,
  download metrics to CSV/TensorBoard) and archive snapshots in GitHub storage to reduce switching
  cost.
- **Data governance**: Large datasets may include sensitive simulated data. Define access controls
  early and include redaction guidelines in metadata.
- **Tool sprawl**: Multiple trackers/conf systems can confuse contributors. Enforce a minimal golden
  path once the evaluation concludes, favouring hosted services with long-lived APIs.

---

This RFC now locks in the full experiment stack: W&B (`viterbo-experiments`) with mandated tags, Git
LFS artefact storage, lightweight metadata bundles, GitHub/Codex secrets with `.env` loading,
Orbax-only checkpoints, and Makefile-driven workflows. The remaining work is execution: implement
the logging path, finalise the metadata schema, extend the Makefile, and validate the Git LFS
pipeline while keeping contingency options noted if future constraints shift.
