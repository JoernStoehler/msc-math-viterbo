"""Tests for the atlas dataset build harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from viterbo.datasets import atlas
from viterbo.datasets.build import QuantityConfig, build_atlas_dataset, default_plan


@pytest.mark.goal_code
@pytest.mark.smoke
def test_build_atlas_tiny_creates_manifest_and_logs(tmp_path: Path) -> None:
    """Building the tiny preset writes datasets, manifests, and timing logs."""

    plan = default_plan()
    overrides = {cfg.name: 1 for cfg in plan.generators}
    for cfg in plan.generators:
        params = dict(cfg.parameters)
        if "dimension" in params:
            params["dimension"] = 2
        if "num_facets" in params:
            params["num_facets"] = 4
        if cfg.entrypoint == "enumerate_product_ngons":
            params = {"max_ngon_P": 3, "max_ngon_Q": 3, "max_rotation_Q": 1}
        cfg.parameters = params
    plan.quantities = [
        QuantityConfig(
            name="volume",
            target_field="volume",
            implementation="tests.stub.volume",
            compute=lambda _poly: 1.0,
            failure_value=float("nan"),
        ),
        QuantityConfig(
            name="ehz_capacity",
            target_field="ehz_capacity",
            implementation="tests.stub.capacity",
            compute=lambda _poly: 1.0,
            failure_value=float("nan"),
        ),
        QuantityConfig(
            name="systolic_ratio",
            target_field="systolic_ratio",
            implementation="tests.stub.systolic",
            compute=lambda _poly: 1.0,
            failure_value=float("nan"),
        ),
        QuantityConfig(
            name="minimum_action_cycle",
            target_field="minimum_action_cycle",
            implementation="tests.stub.cycle",
            compute=lambda _poly: [],
            failure_value=[],
        ),
    ]

    outputs = build_atlas_dataset(
        plan,
        preset="tiny",
        dataset_dir=tmp_path / "datasets",
        benchmark_dir=tmp_path / "benchmarks",
        sample_overrides=overrides,
    )

    dataset = atlas.load_dataset(str(outputs.dataset_path))
    assert dataset.num_rows == len(plan.generators)

    with outputs.manifest_path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    assert manifest["rows"] == dataset.num_rows
    assert {entry["name"] for entry in manifest["generators"]} == set(overrides)

    with outputs.summary_path.open("r", encoding="utf-8") as fp:
        summary = json.load(fp)
    assert summary["dataset"].endswith("tiny")
    assert summary["entries"], "Summary should include aggregated timings."

    log_lines = outputs.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert log_lines, "Timings log should contain per-quantity measurements."


@pytest.mark.goal_code
@pytest.mark.smoke
def test_resume_skips_completed_generators(tmp_path: Path) -> None:
    """Resuming the build does not duplicate rows for completed generators."""

    plan = default_plan()
    overrides = {cfg.name: 1 for cfg in plan.generators}
    for cfg in plan.generators:
        params = dict(cfg.parameters)
        if "dimension" in params:
            params["dimension"] = 2
        if "num_facets" in params:
            params["num_facets"] = 4
        if cfg.entrypoint == "enumerate_product_ngons":
            params = {"max_ngon_P": 3, "max_ngon_Q": 3, "max_rotation_Q": 1}
        cfg.parameters = params
    plan.quantities = [
        QuantityConfig(
            name="volume",
            target_field="volume",
            implementation="tests.stub.volume",
            compute=lambda _poly: 1.0,
            failure_value=float("nan"),
        ),
        QuantityConfig(
            name="ehz_capacity",
            target_field="ehz_capacity",
            implementation="tests.stub.capacity",
            compute=lambda _poly: 1.0,
            failure_value=float("nan"),
        ),
        QuantityConfig(
            name="systolic_ratio",
            target_field="systolic_ratio",
            implementation="tests.stub.systolic",
            compute=lambda _poly: 1.0,
            failure_value=float("nan"),
        ),
        QuantityConfig(
            name="minimum_action_cycle",
            target_field="minimum_action_cycle",
            implementation="tests.stub.cycle",
            compute=lambda _poly: [],
            failure_value=[],
        ),
    ]
    output_dir = tmp_path / "datasets"
    log_dir = tmp_path / "benchmarks"

    initial = build_atlas_dataset(
        plan,
        preset="tiny",
        dataset_dir=output_dir,
        benchmark_dir=log_dir,
        sample_overrides=overrides,
    )
    assert initial.rows_written == len(plan.generators)

    rerun = build_atlas_dataset(
        plan,
        preset="tiny",
        dataset_dir=output_dir,
        benchmark_dir=log_dir,
        sample_overrides=overrides,
    )

    assert rerun.rows_written == 0
