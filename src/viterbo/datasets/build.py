"""Dataset build harness for atlas snapshots and benchmarks (datasets layer)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import jax.random
import jax.numpy as jnp
from datasets import Dataset

from viterbo.datasets import atlas
from viterbo.datasets import generators as basic_generators
import viterbo.datasets.cycles as cycles
import viterbo.datasets.systolic as systolic
from viterbo.math import volume
from viterbo.math.capacity.facet_normals import ehz_capacity_fast_facet_normals
from viterbo.datasets.types import Polytope


QuantValue = float | list[list[float]]


@dataclass(slots=True)
class QuantityConfig:
    """Configuration for a derived quantity stored in the atlas."""

    name: str
    target_field: str
    implementation: str
    compute: Callable[[Polytope], QuantValue]
    failure_value: QuantValue


@dataclass(slots=True)
class GeneratorConfig:
    """Description of a polytope generator family."""

    name: str
    distribution_name: str
    entrypoint: str
    parameters: Mapping[str, object]
    samples: Mapping[str, int]
    base_seed: int
    requires_key: bool = True

    def requested_samples(self, preset: str, *, overrides: Mapping[str, int] | None = None) -> int:
        """Return the number of requested samples for ``preset``."""

        if overrides and self.name in overrides:
            return overrides[self.name]
        try:
            return int(self.samples[preset])
        except KeyError as exc:  # pragma: no cover - defensive
            msg = f"Unknown preset '{preset}' for generator '{self.name}'."
            raise ValueError(msg) from exc

    def generate(
        self,
        *,
        preset: str,
        completed: int,
        overrides: Mapping[str, int] | None = None,
        seed_offset: int = 0,
    ) -> list[Polytope]:
        """Materialise the next batch of polytopes for ``preset``."""

        requested = self.requested_samples(preset, overrides=overrides)
        if requested <= completed:
            return []

        generator: Callable[..., Sequence[Polytope]] = getattr(basic_generators, self.entrypoint)
        params = dict(self.parameters)

        if self.requires_key:
            total = requested
            key = jax.random.PRNGKey(self.base_seed + seed_offset)
            params.setdefault("num_samples", total)
            samples = list(generator(key=key, **params))
            return list(samples[completed:requested])

        samples = list(generator(**params))
        return samples[completed:requested]


@dataclass(slots=True)
class AtlasBuildPlan:
    """Plan containing generator and quantity coverage."""

    base_name: str
    generators: list[GeneratorConfig]
    quantities: list[QuantityConfig]
    schema_version: str = "atlas-v1"

    def dataset_name(self, preset: str) -> str:
        """Return the dataset name for ``preset``."""
        return f"{self.base_name}-{preset}"


@dataclass(slots=True)
class BuildOutputs:
    """Paths and counters produced by a build run."""

    dataset_path: Path
    manifest_path: Path
    log_path: Path
    summary_path: Path
    rows_written: int


def default_plan() -> AtlasBuildPlan:
    """Return the default atlas build plan covering core generators."""

    generators = [
        GeneratorConfig(
            name="halfspace",
            distribution_name="sample_halfspace",
            entrypoint="sample_halfspace",
            parameters={"dimension": 4, "num_facets": 8},
            samples={"tiny": 1, "small": 200},
            base_seed=11,
        ),
        GeneratorConfig(
            name="halfspace-tangent",
            distribution_name="sample_halfspace_tangent",
            entrypoint="sample_halfspace_tangent",
            parameters={"dimension": 4, "num_facets": 8},
            samples={"tiny": 1, "small": 200},
            base_seed=23,
        ),
        GeneratorConfig(
            name="uniform-sphere",
            distribution_name="sample_uniform_sphere",
            entrypoint="sample_uniform_sphere",
            parameters={"dimension": 4},
            samples={"tiny": 1, "small": 200},
            base_seed=37,
        ),
        GeneratorConfig(
            name="uniform-ball",
            distribution_name="sample_uniform_ball",
            entrypoint="sample_uniform_ball",
            parameters={"dimension": 4},
            samples={"tiny": 1, "small": 200},
            base_seed=41,
        ),
        GeneratorConfig(
            name="product-ngons",
            distribution_name="enumerate_product_ngons",
            entrypoint="enumerate_product_ngons",
            parameters={"max_ngon_P": 7, "max_ngon_Q": 7, "max_rotation_Q": 2},
            samples={"tiny": 1, "small": 250},
            base_seed=0,
            requires_key=False,
        ),
    ]

    def _cycle(poly: Polytope) -> list[list[float]]:
        try:
            cycle_points = cycles.minimum_cycle_reference(poly)
        except NotImplementedError:
            return []
        except Exception as exc:  # pragma: no cover - defensive
            raise exc
        return jnp.asarray(cycle_points, dtype=jnp.float64).tolist()

    quantities = [
        QuantityConfig(
            name="volume",
            target_field="volume",
            implementation="viterbo.volume.volume_reference",
            compute=lambda poly: float(volume.volume_reference(poly.vertices)),
            failure_value=math.nan,
        ),
        QuantityConfig(
            name="ehz_capacity",
            target_field="ehz_capacity",
            implementation="viterbo.capacity.ehz_capacity_fast",
            compute=lambda poly: float(ehz_capacity_fast_facet_normals(poly.normals, poly.offsets)),
            failure_value=math.nan,
        ),
        QuantityConfig(
            name="systolic_ratio",
            target_field="systolic_ratio",
            implementation="viterbo.datasets.systolic.systolic_ratio",
            compute=lambda poly: float(systolic.systolic_ratio(poly)),
            failure_value=math.nan,
        ),
        QuantityConfig(
            name="minimum_action_cycle",
            target_field="minimum_action_cycle",
            implementation="viterbo.datasets.cycles.minimum_cycle_reference",
            compute=_cycle,
            failure_value=[],
        ),
    ]

    return AtlasBuildPlan(base_name="atlas", generators=generators, quantities=quantities)


def _empty_dataset() -> Dataset:
    return atlas.build_dataset([])


def _load_existing_dataset(dataset_dir: Path) -> Dataset:
    dataset_info = dataset_dir / "dataset_info.json"
    if dataset_info.exists():
        existing = atlas.load_dataset(str(dataset_dir))
        iterable = cast(Iterable[Mapping[str, object]], existing)
        rows = [dict(row) for row in iterable]
        return atlas.build_dataset(rows)
    return _empty_dataset()


def _initial_manifest(plan: AtlasBuildPlan, preset: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "dataset": plan.dataset_name(preset),
        "preset": preset,
        "schema_version": plan.schema_version,
        "created": now,
        "updated": now,
        "generators": [],
        "quantities": [
            {
                "name": quantity.name,
                "target_field": quantity.target_field,
                "implementation": quantity.implementation,
            }
            for quantity in plan.quantities
        ],
        "rows": 0,
    }
    return manifest


def _ensure_manifest_entry(
    manifest: dict[str, Any],
    generator: GeneratorConfig,
    preset: str,
    overrides: Mapping[str, int] | None,
) -> dict[str, Any]:
    records_obj = manifest.get("generators")
    if not isinstance(records_obj, list):
        records = []
        manifest["generators"] = records
    else:
        records = cast(list[dict[str, Any]], records_obj)
    for item in records:
        if item.get("name") == generator.name:
            return item
    entry: dict[str, Any] = {
        "name": generator.name,
        "distribution_name": generator.distribution_name,
        "entrypoint": generator.entrypoint,
        "parameters": dict(generator.parameters),
        "requested": generator.requested_samples(preset, overrides=overrides),
        "completed": 0,
        "base_seed": generator.base_seed,
    }
    records.append(entry)
    manifest["generators"] = records
    return entry


def _update_manifest_entry(entry: dict[str, Any], increment: int) -> None:
    entry["completed"] = int(entry.get("completed", 0)) + increment


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def _load_manifest(path: Path, plan: AtlasBuildPlan, preset: str) -> dict[str, Any]:
    if not path.exists():
        return _initial_manifest(plan, preset)
    with path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    manifest.setdefault("rows", 0)
    if not isinstance(manifest.get("generators"), list):
        manifest["generators"] = []
    if not isinstance(manifest.get("quantities"), list):
        manifest["quantities"] = []
    return manifest


def _append_dataset(dataset: Dataset, rows: Iterable[Mapping[str, object]]) -> Dataset:
    materialised = list(rows)
    if not materialised:
        return dataset
    return atlas.append_rows(dataset, materialised)


def _write_readme(dataset_dir: Path, plan: AtlasBuildPlan, preset: str) -> None:
    dataset_name = plan.dataset_name(preset)
    readme_path = dataset_dir / "README.md"
    content = f"""# {dataset_name}

This directory stores the Hugging Face dataset snapshot produced by the
atlas build harness in `scripts/build_atlas_small.py`.

- Manifest: `manifest.json`
- Raw timings: see `../benchmarks/{dataset_name}_timings.jsonl`
- Summary: `../benchmarks/{dataset_name}_summary.json`

To resume a partial build run:

```bash
python scripts/build_atlas_small.py --preset {preset} --resume
```

Use `--limit-generator` to focus on a subset of generator families.
"""
    with readme_path.open("w", encoding="utf-8") as fp:
        fp.write(content)


def _summarise_logs(log_path: Path, dataset_name: str) -> dict[str, Any]:
    summary: dict[tuple[str, str], dict[str, float | int]] = {}
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                payload = json.loads(line)
                key = (payload["generator"], payload["quantity"])
                record = summary.setdefault(
                    key,
                    {"count": 0, "errors": 0, "total_runtime_s": 0.0},
                )
                record["count"] += 1
                record["total_runtime_s"] += float(payload["runtime_s"])
                if payload.get("status") != "ok":
                    record["errors"] += 1

    aggregated = [
        {
            "generator": generator,
            "quantity": quantity,
            "count": record["count"],
            "errors": record["errors"],
            "total_runtime_s": record["total_runtime_s"],
            "mean_runtime_s": (
                record["total_runtime_s"] / record["count"] if record["count"] else 0.0
            ),
        }
        for (generator, quantity), record in sorted(summary.items())
    ]
    return {
        "dataset": dataset_name,
        "entries": aggregated,
        "aggregated": aggregated,
    }


def build_atlas_dataset(
    plan: AtlasBuildPlan,
    *,
    preset: str,
    dataset_dir: Path,
    benchmark_dir: Path,
    sample_overrides: Mapping[str, int] | None = None,
    seed_offset: int = 0,
) -> BuildOutputs:
    """Build or extend an atlas dataset according to ``plan``.

    Writes the dataset to ``dataset_dir`` and aggregates timing logs under
    ``benchmark_dir``. Idempotent per generator entry.
    """
    manifest_path = dataset_dir / "manifest.json"
    manifest = _load_manifest(manifest_path, plan, preset)
    dataset = _load_existing_dataset(dataset_dir)
    dataset_name = plan.dataset_name(preset)

    selected_generators = plan.generators
    if sample_overrides is None:
        sample_overrides = {}

    raw_logs: list[dict[str, object]] = []
    rows_to_append: list[dict[str, object]] = []
    total_written = 0

    for generator in selected_generators:
        entry = _ensure_manifest_entry(manifest, generator, preset, sample_overrides)
        requested = generator.requested_samples(preset, overrides=sample_overrides)
        completed = int(entry.get("completed", 0))
        batch = generator.generate(
            preset=preset,
            completed=completed,
            overrides=sample_overrides,
            seed_offset=seed_offset,
        )

        if not batch:
            continue

        for index, poly in enumerate(batch, start=completed):
            poly_id = f"{dataset_name}/{generator.name}/{index:05d}"
            normals = jnp.asarray(poly.normals, dtype=jnp.float64)
            offsets = jnp.asarray(poly.offsets, dtype=jnp.float64)
            vertices = jnp.asarray(poly.vertices, dtype=jnp.float64)
            dimension = int(normals.shape[1])
            num_facets = int(normals.shape[0])
            num_vertices = int(vertices.shape[0])
            notes: list[str] = []
            row: dict[str, object] = {
                "polytope_id": poly_id,
                "notes": "",
                "distribution_name": generator.distribution_name,
                "dimension": dimension,
                "num_facets": num_facets,
                "num_vertices": num_vertices,
                "normals": normals.tolist(),
                "offsets": offsets.tolist(),
                "vertices": vertices.tolist(),
            }

            for quantity in plan.quantities:
                start = perf_counter()
                status = "ok"
                error_message: str | None = None
                try:
                    value = quantity.compute(poly)
                except NotImplementedError as exc:
                    value = quantity.failure_value
                    status = "skipped"
                    error_message = str(exc)
                except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
                    value = quantity.failure_value
                    status = "error"
                    error_message = f"{type(exc).__name__}: {exc}"
                runtime = perf_counter() - start

                if isinstance(value, float):
                    row[quantity.target_field] = float(value)
                else:
                    row[quantity.target_field] = value

                raw_logs.append(
                    {
                        "dataset": dataset_name,
                        "polytope_id": poly_id,
                        "generator": generator.name,
                        "quantity": quantity.name,
                        "status": status,
                        "runtime_s": runtime,
                        "error": error_message,
                    }
                )

                if status != "ok" and error_message:
                    notes.append(f"{quantity.name}: {error_message}")

            row["notes"] = " | ".join(notes)
            rows_to_append.append(row)
            total_written += 1

        processed = len(batch)
        if processed:
            _update_manifest_entry(entry, processed)
            entry["requested"] = requested

    if rows_to_append:
        dataset = _append_dataset(dataset, rows_to_append)

    if dataset_dir.exists():
        import shutil

        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    atlas.save_dataset(dataset, str(dataset_dir))

    manifest["rows"] = int(dataset.num_rows)
    manifest["updated"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    _write_readme(dataset_dir, plan, preset)

    # Ensure benchmark directory exists for logs and summaries
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    log_path = benchmark_dir / f"{dataset_name}_timings.jsonl"
    if raw_logs:
        with log_path.open("a", encoding="utf-8") as fp:
            for entry in raw_logs:
                fp.write(json.dumps(entry) + "\n")
    elif not log_path.exists():
        log_path.touch()

    summary_path = benchmark_dir / f"{dataset_name}_summary.json"
    summary_payload = _summarise_logs(log_path, dataset_name)
    _write_json(summary_path, summary_payload)

    return BuildOutputs(
        dataset_path=dataset_dir,
        manifest_path=manifest_path,
        log_path=log_path,
        summary_path=summary_path,
        rows_written=total_written,
    )


__all__ = [
    "AtlasBuildPlan",
    "BuildOutputs",
    "GeneratorConfig",
    "QuantityConfig",
    "build_atlas_dataset",
    "default_plan",
]
