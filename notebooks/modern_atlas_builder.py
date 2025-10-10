"""Demonstration notebook for building the modern atlas dataset.

This script-style notebook outlines the intended workflow for generating or
updating the polytope atlas without relying on legacy modules. Each step invokes
stub functions from :mod:`viterbo.modern` and therefore raises
:class:`NotImplementedError` today; the goal is to communicate the orchestration
pattern we will support once real implementations land.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from viterbo.modern import basic_generators, datasets, polytopes, volume

# ----------------------------------------------------------------------------
# Configuration knobs a practitioner might toggle when (re)building the atlas.
# ----------------------------------------------------------------------------
ATLAS_PATH = Path("artefacts/modern_atlas.parquet")
GENERATOR_DIMENSION = 4
NUM_SAMPLES = 32


# ----------------------------------------------------------------------------
# Step 1: Produce candidate polytopes using modern generators.
# ----------------------------------------------------------------------------
try:
    generator = basic_generators.sample_uniform_ball(
        key=None,  # placeholder until PRNG handling is wired in
        dimension=GENERATOR_DIMENSION,
        num_samples=NUM_SAMPLES,
    )
except NotImplementedError:
    generator = None


# ----------------------------------------------------------------------------
# Step 2: Convert raw outputs into structured bundles and quantities.
# ----------------------------------------------------------------------------
records: list[tuple] = []
if generator is not None:
    for bundle, metadata in generator:
        try:
            enriched_bundle = polytopes.complete_incidence(bundle)
            volume_estimate = volume.volume_reference(enriched_bundle)
        except NotImplementedError:
            break
        records.append((enriched_bundle, metadata, volume_estimate))


# ----------------------------------------------------------------------------
# Step 3: Materialise a Polars dataframe matching the atlas schema.
# ----------------------------------------------------------------------------
try:
    schema = datasets.atlas_schema()
    dataframe = datasets.records_to_dataframe([])
except NotImplementedError:
    schema = None
    dataframe = None


# ----------------------------------------------------------------------------
# Step 4: Merge with any on-disk snapshot and persist the result.
# ----------------------------------------------------------------------------
if dataframe is not None and schema is not None:
    try:
        if ATLAS_PATH.exists():
            existing = pl.read_parquet(ATLAS_PATH)
            merged = datasets.merge_results(existing, dataframe, conflict_policy="overwrite")
        else:
            merged = dataframe
        merged.write_parquet(ATLAS_PATH)
    except NotImplementedError:
        pass

print("Modern atlas builder executed placeholder workflow.")
