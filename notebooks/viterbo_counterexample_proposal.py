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
import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

from viterbo.math.constructions import lagrangian_product, rotated_regular_ngon2d
from viterbo.math.minimal_action import minimal_action_cycle_lagrangian_product

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_PATH))

# %% [markdown]
# ## Counterexample geometry in torch

# %%
torch.set_default_dtype(torch.float64)


def split_pq(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split 4D points into their (q, p) components."""
    if points.ndim != 2 or points.size(1) != 4:
        raise ValueError("expected points shaped (N, 4)")
    return points[:, :2], points[:, 2:]


def polygon_area(vertices: torch.Tensor) -> torch.Tensor:
    """Signed area of a planar polygon provided in counter-clockwise order."""
    if vertices.ndim != 2 or vertices.size(1) != 2:
        raise ValueError("expected planar vertices (N, 2)")
    rolled = torch.roll(vertices, shifts=-1, dims=0)
    cross = vertices[:, 0] * rolled[:, 1] - vertices[:, 1] * rolled[:, 0]
    return 0.5 * torch.sum(cross).abs()


def counterexample_geometry() -> dict[str, torch.Tensor]:
    """Construct the pentagon × rotated pentagon counterexample."""
    vertices_q, normals_q, offsets_q = rotated_regular_ngon2d(5, 0.0)
    vertices_p, normals_p, offsets_p = rotated_regular_ngon2d(5, -math.pi / 2)
    vertices_4d, normals_4d, offsets_4d = lagrangian_product(vertices_q, vertices_p)
    capacity, cycle = minimal_action_cycle_lagrangian_product(vertices_q, normals_p, offsets_p)
    area_q = polygon_area(vertices_q)
    area_p = polygon_area(vertices_p)
    volume_4d = area_q * area_p
    systolic_ratio_value = volume_4d / capacity.pow(2)
    return {
        "vertices_q": vertices_q,
        "normals_q": normals_q,
        "offsets_q": offsets_q,
        "vertices_p": vertices_p,
        "normals_p": normals_p,
        "offsets_p": offsets_p,
        "vertices_4d": vertices_4d,
        "normals_4d": normals_4d,
        "offsets_4d": offsets_4d,
        "capacity": capacity,
        "cycle": cycle,
        "area_q": area_q,
        "area_p": area_p,
        "volume_4d": volume_4d,
        "systolic_ratio": systolic_ratio_value,
    }


GEOMETRY = counterexample_geometry()

# %% [markdown]
# ## Inspect the counterexample dataset


# %%
def print_cycle(path: torch.Tensor) -> None:
    for idx, vertex in enumerate(path):
        label = f"v{idx:02d}"
        coords = ", ".join(f"{value.item():+.3f}" for value in vertex)
        print(f"{label}: [{coords}]")


print("Systolic ratio:", GEOMETRY["systolic_ratio"].item())
print("Capacity:", GEOMETRY["capacity"].item())
print("Volume:", GEOMETRY["volume_4d"].item())
print("Area(K):", GEOMETRY["area_q"].item())
print("Area(T):", GEOMETRY["area_p"].item())
print("\nFacet normals and offsets:")
for normal, offset in zip(GEOMETRY["normals_4d"], GEOMETRY["offsets_4d"]):
    normal_str = ", ".join(f"{value.item():+.2f}" for value in normal)
    print(f"  n = [{normal_str}], offset = {offset.item():+.2f}")

print("\nVertices:")
print_cycle(GEOMETRY["vertices_4d"])

print("\nCycle path:")
print_cycle(GEOMETRY["cycle"])

# %% [markdown]
# ## Visualization helper


# %%
def regular_pentagon(scale: float = 1.0, rotation: float = 0.0) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 6)[:-1] + rotation
    return np.column_stack((np.cos(angles), np.sin(angles))) * scale


def plot_counterexample(
    path: torch.Tensor,
    *,
    q_vertices: torch.Tensor | None = None,
    p_vertices: torch.Tensor | None = None,
    pentagon_scale: float = 0.9,
    line_alpha: float = 0.85,
    label_offset: float = 0.04,
) -> None:
    path_np = path.detach().cpu().numpy()
    p_coords = path_np[:, :2]
    q_coords = path_np[:, 2:]
    colors = cm.rainbow(np.linspace(0.0, 1.0, len(path_np) - 1))

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

    for idx in range(len(path_np) - 1):
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

    if q_vertices is None:
        base = regular_pentagon(scale=pentagon_scale)
        pentagon = np.vstack([base, base[0]])
    else:
        q_vertices_np = q_vertices.detach().cpu().numpy()
        pentagon = np.vstack([q_vertices_np, q_vertices_np[0]])
    if p_vertices is None:
        base_rot = regular_pentagon(scale=pentagon_scale, rotation=np.pi / 2)
        rotated_pentagon = np.vstack([base_rot, base_rot[0]])
    else:
        p_vertices_np = p_vertices.detach().cpu().numpy()
        rotated_pentagon = np.vstack([p_vertices_np, p_vertices_np[0]])

    ax_p.plot(
        pentagon[:, 0],
        pentagon[:, 1],
        color="dimgray",
        linewidth=1.5,
        linestyle="--",
        label="Lagrangian factor",
    )
    ax_q.plot(
        rotated_pentagon[:, 0],
        rotated_pentagon[:, 1],
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


plot_counterexample(
    GEOMETRY["cycle"],
    q_vertices=GEOMETRY["vertices_q"],
    p_vertices=GEOMETRY["vertices_p"],
)

# %% [markdown]
# ## Next steps
#
# - Swap the mock arrays for real tensors coming from the `viterbo.math`
#   counterexample loader.
# - Thread through metadata (facet adjacency, action values) once they are
#   exposed.
# - Consider exporting interactive visualizations (Plotly/pythreejs) for deeper
#   inspection of the 4D structure.
