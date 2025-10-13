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
# # Proposal: Visualizing the 2024 Counterexample to Viterbo's Conjecture
#
# This notebook sketches the exploration tool we want once the geometry helpers in
# `viterbo.math` land.  The goal is to interactively inspect the 4D polytope and
# the piecewise linear Reeb trajectory underlying the 2024 counterexample.
#
# **Status:** proposal.  The data seeded below are placeholders that mimic the
# structure we expect from the future APIs.

# %% [markdown]
# ## Planned library support
#
# To make this notebook production ready we will need:
#
# 1. `viterbo.math.polytopes` helpers capable of returning the counterexample
#    polytope as
#    - vertex list (``torch.Tensor`` of shape ``(V, 4)``),
#    - facet normals (``(F, 4)``) and offsets (``(F,)``) so we can render and
#      export the polytope,
#    - symplectic splitting into ``p`` and ``q`` components.
# 2. `viterbo.math.restricted_contact` (or similar) routines that produce the
#    systolic data: volume, capacity, systolic ratio, and the closed characteristic
#    sampled as an ordered list of 4D vertices.
# 3. Convenience projectors ``split_pq(points: Tensor) -> tuple[Tensor, Tensor]``
#    to avoid hand slicing in every notebook.
# 4. Plotting utilities (matplotlib wrappers) that accept torch tensors and emit
#    publication-ready 2D projections with consistent styling.
#
# Once these land we will replace the mock data section with live calls into the
# library, e.g. ``polytope = counterexample_2024()``.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_PATH))

# %% [markdown]
# ## Mock counterexample data
#
# These arrays only demonstrate the expected shapes and layout.  Replace them
# with real outputs once the geometry code is available.

# %%
MOCK_SYSTOLIC_RATIO = 1.234
MOCK_CAPACITY = 3.14159
MOCK_VOLUME = 2.71828

MOCK_FACET_NORMALS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -1.0],
    ]
)
MOCK_FACET_OFFSETS = np.array([1.0, 1.0, 1.0, 1.0, 0.25, 0.25])

MOCK_VERTICES = np.array(
    [
        [0.8, 0.1, 0.0, 0.0],
        [0.4, 0.7, 0.0, 0.0],
        [0.0, 0.6, 0.2, -0.3],
        [-0.5, 0.2, 0.4, 0.5],
        [-0.3, -0.4, 0.6, 0.2],
        [0.2, -0.7, -0.2, 0.6],
        [0.7, -0.2, -0.4, -0.4],
    ]
)

MOCK_CYCLE_PATH = np.array(
    [
        [0.8, 0.0, 0.1, 0.0],
        [0.6, 0.4, 0.2, -0.2],
        [0.2, 0.8, 0.3, -0.1],
        [-0.2, 0.5, 0.5, 0.2],
        [-0.4, 0.1, 0.6, 0.4],
        [-0.1, -0.3, 0.4, 0.6],
        [0.3, -0.5, 0.1, 0.4],
        [0.6, -0.2, -0.1, 0.1],
        [0.8, 0.0, 0.1, 0.0],
    ]
)

# %% [markdown]
# ## Inspect the mock dataset


# %%
def print_cycle(path: np.ndarray) -> None:
    for idx, vertex in enumerate(path):
        label = f"v{idx:02d}"
        coords = ", ".join(f"{value:+.3f}" for value in vertex)
        print(f"{label}: [{coords}]")


print("Systolic ratio:", MOCK_SYSTOLIC_RATIO)
print("Capacity:", MOCK_CAPACITY)
print("Volume:", MOCK_VOLUME)
print("\nFacet normals and offsets:")
for normal, offset in zip(MOCK_FACET_NORMALS, MOCK_FACET_OFFSETS):
    normal_str = ", ".join(f"{value:+.2f}" for value in normal)
    print(f"  n = [{normal_str}], offset = {offset:+.2f}")

print("\nVertices:")
print_cycle(MOCK_VERTICES)

print("\nCycle path:")
print_cycle(MOCK_CYCLE_PATH)

# %% [markdown]
# ## Visualization helper


# %%
def regular_pentagon(scale: float = 1.0, rotation: float = 0.0) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 6)[:-1] + rotation
    return np.column_stack((np.cos(angles), np.sin(angles))) * scale


def plot_counterexample(
    path: np.ndarray,
    *,
    pentagon_scale: float = 0.9,
    line_alpha: float = 0.85,
    label_offset: float = 0.04,
) -> None:
    p_coords = path[:, :2]
    q_coords = path[:, 2:]
    colors = cm.rainbow(np.linspace(0.0, 1.0, len(path) - 1))

    fig, (ax_p, ax_q) = plt.subplots(1, 2, figsize=(12, 6))

    def draw_panel(ax: plt.Axes, coords: np.ndarray, name: str) -> None:
        ax.set_title(f"{name}-projection")
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0.0, color="lightgray", linewidth=0.8)
        ax.axvline(0.0, color="lightgray", linewidth=0.8)
        for idx, point in enumerate(coords[:-1]):
            ax.scatter(point[0], point[1], color="black", zorder=3)
            ax.text(
                point[0] + label_offset,
                point[1] + label_offset,
                str(idx + 1),
                fontsize=9,
                ha="left",
                va="bottom",
            )
        ax.scatter(coords[-1, 0], coords[-1, 1], color="black", zorder=3)
        ax.text(
            coords[-1, 0] + label_offset,
            coords[-1, 1] + label_offset,
            str(len(coords)),
            fontsize=9,
            ha="left",
            va="bottom",
        )

    for idx in range(len(path) - 1):
        ax_p.plot(
            p_coords[idx : idx + 2, 0],
            p_coords[idx : idx + 2, 1],
            color=colors[idx],
            alpha=line_alpha,
            linewidth=2.0,
        )
        ax_q.plot(
            q_coords[idx : idx + 2, 0],
            q_coords[idx : idx + 2, 1],
            color=colors[idx],
            alpha=line_alpha,
            linewidth=2.0,
        )

    draw_panel(ax_p, p_coords, "p")
    draw_panel(ax_q, q_coords, "q")

    pentagon = regular_pentagon(scale=pentagon_scale)
    rotated_pentagon = regular_pentagon(scale=pentagon_scale, rotation=np.pi / 2)

    ax_p.plot(
        np.append(pentagon[:, 0], pentagon[0, 0]),
        np.append(pentagon[:, 1], pentagon[0, 1]),
        color="dimgray",
        linewidth=1.5,
        linestyle="--",
        label="Lagrangian factor",
    )
    ax_q.plot(
        np.append(rotated_pentagon[:, 0], rotated_pentagon[0, 0]),
        np.append(rotated_pentagon[:, 1], rotated_pentagon[0, 1]),
        color="dimgray",
        linewidth=1.5,
        linestyle="--",
        label="Rotated factor",
    )

    for ax in (ax_p, ax_q):
        ax.legend(loc="upper right")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()


plot_counterexample(MOCK_CYCLE_PATH)

# %% [markdown]
# ## Next steps
#
# - Swap the mock arrays for real tensors coming from the `viterbo.math`
#   counterexample loader.
# - Thread through metadata (facet adjacency, action values) once they are
#   exposed.
# - Consider exporting interactive visualizations (Plotly/pythreejs) for deeper
#   inspection of the 4D structure.
