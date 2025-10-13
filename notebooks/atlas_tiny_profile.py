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
# hotspots across the supporting `viterbo.math` helpers.  The math and dataset
# implementations are expected to land in a follow-up PR, so this file focuses on
# the profiling harness and reporting utilities.
#
# ## Workflow
#
# 1. (Optional) edit the dataset/math code.
# 2. Re-run the profiling cell below.  It executes `atlas_tiny_build()` inside a
#    `cProfile` session.
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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    print(f"Profiled dataset with {len(dataset)} rows.")
else:
    print("Dataset creation did not return a value (likely due to stub implementation).")


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


def collect_math_hotspots(
    profile: cProfile.Profile,
    *,
    root: Path = Path("viterbo/math"),
    limit: int = 20,
) -> list[tuple[str, pstats.func_stats]]:
    """Return the hottest frames originating from ``viterbo/math``."""

    stats = pstats.Stats(profile).sort_stats("cumulative")
    rows: list[tuple[str, pstats.func_stats]] = []
    for (filename, line, func_name), func_stat in stats.stats.items():
        if root.as_posix() in Path(filename).as_posix():
            rows.append((f"{Path(filename).name}:{line}::{func_name}", func_stat))
    rows.sort(key=lambda item: item[1][3], reverse=True)  # sort by cumulative time
    return rows[:limit]


def format_func_stat(func_stat: pstats.func_stats) -> tuple[float, float, int, int]:
    cc, nc, tt, ct, callers = func_stat
    return tt, ct, cc, nc


def print_math_hotspots(profile: cProfile.Profile, *, limit: int = 20) -> None:
    """Pretty-print the cumulative time spent in ``viterbo.math`` helpers."""

    rows = collect_math_hotspots(profile, limit=limit)
    if not rows:
        print("No frames from viterbo.math detected in the profile (implementation pending?).")
        return

    header = f"{'function':60} | {'cum. time (s)':>12} | {'self time (s)':>13} | {'calls':>7}"
    print(header)
    print("-" * len(header))
    for label, func_stat in rows:
        tt, ct, cc, nc = format_func_stat(func_stat)
        print(f"{label:60} | {ct:12.6f} | {tt:13.6f} | {nc:7d}")


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

