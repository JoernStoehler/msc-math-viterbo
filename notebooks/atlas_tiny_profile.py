# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Atlas Tiny dataset profiling
#
# This notebook profiles `viterbo.datasets.atlas_tiny.atlas_tiny_build` to surface
# hotspots across the supporting `viterbo.math` helpers.  The builder currently
# returns a list of completed row dictionaries; consumers can convert it to
# batched tensors via `atlas_tiny_collate_pad`.  The math and dataset helpers are
# expected to land in a follow-up PR, so this file focuses on the profiling
# harness and reporting utilities.
#
# ## Workflow
#
# 1. (Optional) edit the dataset/math code.
# 2. Re-run the profiling cell below.  It executes `atlas_tiny_build()` inside a
#    `cProfile` session and returns the list of rows for inspection.
# 3. Inspect the printed tables: first a global overview, then an extraction of
#    the functions that live under `viterbo/math`.
#
# Adjust the number of warm-up/profile runs in the configuration cell to match
# the fidelity you need.

# %%
from __future__ import annotations

import cProfile
import importlib
import io
import pstats
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import viterbo.datasets.atlas_tiny as atlas_module


# %% [markdown]
# ## Configuration
#
# `PROFILE_REPEATS` controls how often `atlas_tiny_build()` is executed under the
# profiler.  Extra repetitions improve stability at the cost of a longer run.
# Adjust `WARMUP_REPEATS` if the implementation benefits from warming up caches
# or JITed code (no-op for plain Python/Torch).

# %%
PROFILE_REPEATS = 1
WARMUP_REPEATS = 0


# %% [markdown]
# ## Profiling harness
#
# The helper below reloads the dataset module (useful while iterating), performs
# optional warm-up runs, and then executes the build function inside a
# `cProfile.Profile`.  The raw profiler object is returned alongside the dataset
# instance so that you can perform additional analyses (e.g. export to SnakeViz).


# %%
def run_profile(
    *,
    profile_repeats: int = PROFILE_REPEATS,
    warmup_repeats: int = WARMUP_REPEATS,
    build_fn: Callable[[], Any] | None = None,
) -> tuple[Any | None, cProfile.Profile]:
    """Profile ``atlas_tiny_build`` and return the dataset plus profiler."""

    module = importlib.reload(atlas_module)
    build: Callable[[], Any]
    if build_fn is None:
        build = module.atlas_tiny_build
    else:
        build = build_fn

    for _ in range(max(warmup_repeats, 0)):
        try:
            build()
        except NotImplementedError:
            break

    profiler = cProfile.Profile()
    dataset: Any | None = None

    def target() -> None:
        nonlocal dataset
        dataset = build()

    try:
        for _ in range(max(profile_repeats, 1)):
            profiler.runcall(target)
    except NotImplementedError:
        print("atlas_tiny_build is not implemented yet; profiling results will be empty.")
        return None, profiler

    return dataset, profiler


# %% [markdown]
# ## Execute profiling

# %%
dataset, profiler = run_profile()
if dataset is not None:
    print(f"Profiled AtlasTiny build with {len(dataset)} rows.")
else:
    print("Dataset creation did not return a value (likely due to stub implementation).")


# %% [markdown]
# ### Batching the rows
#
# Use `atlas_tiny_collate_pad(dataset)` to obtain padded tensors suitable for
# default PyTorch batching.  The helper returns masks alongside the padded
# arrays so you can trim back to the true vertex/facet counts when needed.


# %% [markdown]
# ## Reporting utilities
#
# `print_profile_overview` shows a global summary sorted by cumulative time.
# `collect_math_hotspots` extracts only the frames coming from modules under
# ``viterbo/math`` so we can focus optimization work on our own kernels.


# %%
def print_profile_overview(profile: cProfile.Profile, *, limit: int = 30) -> None:
    """Print a high-level summary sorted by cumulative time."""

    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).sort_stats("cumulative")
    stats.print_stats(limit)
    print(stream.getvalue())


class FuncStatSummary(NamedTuple):
    label: str
    primitive_calls: int
    total_calls: int
    self_seconds: float
    cumulative_seconds: float


def collect_math_hotspots(
    profile: cProfile.Profile,
    *,
    root: Path = Path("viterbo/math"),
    limit: int = 20,
) -> list[FuncStatSummary]:
    """Return the hottest frames originating from ``viterbo/math``."""

    stats = pstats.Stats(profile).sort_stats("cumulative")
    rows: list[FuncStatSummary] = []
    root_fragment = root.as_posix()
    for (filename, line, func_name), func_stat in stats.stats.items():
        path = Path(filename)
        if root_fragment not in path.as_posix():
            continue
        cc, nc, tt, ct, _ = func_stat
        label = f"{path.name}:{line}::{func_name}"
        rows.append(
            FuncStatSummary(
                label=label,
                primitive_calls=cc,
                total_calls=nc,
                self_seconds=tt,
                cumulative_seconds=ct,
            )
        )
    rows.sort(key=lambda item: item.cumulative_seconds, reverse=True)
    return rows[:limit]


def print_math_hotspots(profile: cProfile.Profile, *, limit: int = 20) -> None:
    """Pretty-print the cumulative time spent in ``viterbo.math`` helpers."""

    rows = collect_math_hotspots(profile, limit=limit)
    if not rows:
        print("No frames from viterbo.math detected in the profile (implementation pending?).")
        return

    header = (
        f"{'function':60} | {'cum. time (s)':>12} | {'self time (s)':>13} | "
        f"{'calls':>7} | {'prim calls':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.label:60} | {row.cumulative_seconds:12.6f} | "
            f"{row.self_seconds:13.6f} | {row.total_calls:7d} | {row.primitive_calls:10d}"
        )


# %% [markdown]
# ## Global overview

# %%
print_profile_overview(profiler)


# %% [markdown]
# ## Hotspots inside `viterbo.math`

# %%
print_math_hotspots(profiler)


# %% [markdown]
# ## Next steps
#
# * Implement the missing dataset/math functions.
# * Re-run the profiling cells after every optimization pass.
# * Consider exporting the profiler data with `profiler.dump_stats('atlas_tiny.prof')`
#   for visualization tools such as SnakeViz.
