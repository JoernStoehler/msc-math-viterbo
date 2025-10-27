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
# # Viterbo Counterexample (Haim–Kislev–Ostrover, 2024) — Standard 4D Instance
#
# This page presents the classical 4D counterexample to Viterbo’s volume–capacity
# inequality using a Lagrangian product of regular pentagons (one rotated by 90°).
# We report the Ekeland–Hofer–Zehnder capacity, the 4D volume, and the systolic
# ratio (normalised so the Euclidean ball has value 1), together with the closed
# characteristic (piecewise-linear Reeb orbit) and a clean two-panel figure of its
# projections to the two planar factors.
#
# References
# - P. Haim‑Kislev, Y. Ostrover (2024), “A Counterexample to Viterbo’s Conjecture”, arXiv:2405.16513.
# - C. Viterbo (2000), “Metric and isoperimetric problems in symplectic geometry”, JAMS.


# %%
from __future__ import annotations

from pathlib import Path
import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_PATH))

from viterbo.math.constructions import lagrangian_product, rotated_regular_ngon2d
from viterbo.math.capacity_ehz.ratios import systolic_ratio
from viterbo.math.capacity_ehz.lagrangian_product import minimal_action_cycle_lagrangian_product
from viterbo.math.polytope import support as support_value

# %% [markdown]
# ## Construction and Normalisation
#
# - Domain: K × T ⊂ R^4, where K and T are regular pentagons in R^2 and T is
#   obtained from K by a 90° rotation. We use unit-radius pentagons.
# - Capacity: c = c_EHZ(K × T) from the minimal-action (≤3-bounce) Minkowski
#   billiard on the product.
# - Volume: Vol_4D(K × T) = Area(K) · Area(T).
# - Systolic ratio (dimension 4): Sys = c^2 / (2 · Vol_4D).

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
    """Pentagon × (90°-rotated) pentagon counterexample in R^4."""
    vertices_q, normals_q, offsets_q = rotated_regular_ngon2d(5, 0.0)
    vertices_p, normals_p, offsets_p = rotated_regular_ngon2d(5, -math.pi / 2)
    vertices_4d, normals_4d, offsets_4d = lagrangian_product(vertices_q, vertices_p)
    capacity, cycle = minimal_action_cycle_lagrangian_product(vertices_q, normals_p, offsets_p)
    area_q = polygon_area(vertices_q)
    area_p = polygon_area(vertices_p)
    volume_4d = area_q * area_p
    # Literature-normalized systolic ratio via core function (n=2 in 4D)
    systolic_ratio_value = systolic_ratio(volume_4d, capacity, 4)
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
# ## Results Summary (dimension 4)


# %%
def _fmt(x: torch.Tensor, digits: int = 12) -> str:
    return f"{x.item():.{digits}f}"


def print_cycle(path: torch.Tensor) -> None:
    for idx, vertex in enumerate(path):
        label = f"v{idx:02d}"
        coords = ", ".join(f"{value.item():+.3f}" for value in vertex)
        print(f"{label}: [{coords}]")

def print_summary(data: dict[str, torch.Tensor]) -> None:
    n = 2  # 4D = 2n with n=2
    print("Definition: Sys = c^n / (n! · Vol_{2n}), here n=2 → Sys = c^2/(2·Vol).\n")
    print("Results (K × T ⊂ R^4):")
    print(f"  c_EHZ(K×T)     = {_fmt(data['capacity'])}")
    print(f"  Vol_4D(K×T)    = {_fmt(data['volume_4d'])}")
    print(f"  Area(K)        = {_fmt(data['area_q'])}")
    print(f"  Area(T)        = {_fmt(data['area_p'])}")
    print(f"  Sys(K×T)       = {_fmt(data['systolic_ratio'])}")


print_summary(GEOMETRY)

print("\nClosed characteristic (cycle vertices in R^4):")
print_cycle(GEOMETRY["cycle"])

print("\nFacet normals and offsets (H-representation of K×T):")
for normal, offset in zip(GEOMETRY["normals_4d"], GEOMETRY["offsets_4d"]):
    normal_str = ", ".join(f"{value.item():+.2f}" for value in normal)
    print(f"  n = [{normal_str}], offset = {offset.item():+.2f}")

# %% [markdown]
# ## Sanity checks — cycle structure and constants

# %%
# The minimal-action cycle for K×T (regular pentagons) is a 2-bounce orbit.
# Verify unique vertices in each projection and re-derive capacity/area constants.
cycle = GEOMETRY["cycle"]
q_cycle = cycle[:, :2]
p_cycle = cycle[:, 2:]

def _unique_rows(x: torch.Tensor) -> torch.Tensor:
    # Stable unique rows helper (float64 coordinates)
    if x.ndim != 2:
        raise ValueError("expected a 2D tensor")
    # Round to dampen printer artifacts and enable equality on expected vertices
    xr = torch.round(x * 1_000_000.0) / 1_000_000.0
    unique, _ = torch.unique(xr, dim=0, sorted=True, return_inverse=True)
    return unique

uq_q = _unique_rows(q_cycle)
uq_p = _unique_rows(p_cycle)
print(f"Unique q-vertices visited: {uq_q.size(0)} (expected 2)")
print(f"Unique p-vertices visited: {uq_p.size(0)} (expected 2)")

# Re-derive c_EHZ from the 2-bounce formula: h_T(v) + h_T(-v) with v = q_j - q_i
v = uq_q[1] - uq_q[0]
h_fwd = support_value(uq_p, v)
h_bwd = support_value(uq_p, -v)
action_two_bounce = h_fwd + h_bwd
print(f"Action from supports (2-bounce): {action_two_bounce.item():.12f}")
print(f"Capacity from solver:          {GEOMETRY['capacity'].item():.12f}")

# Analytic constants for regular pentagon (unit radius)
area_pentagon = (5.0 / 2.0) * math.sin(2.0 * math.pi / 5.0)
capacity_formula = 2.0 * math.cos(math.pi / 10.0) * (1.0 + math.cos(math.pi / 5.0))
sys_formula = (math.sqrt(5.0) + 3.0) / 5.0
print(f"Area(K)=Area(T) analytic:      {area_pentagon:.12f}")
print(f"Capacity analytic (HK–O 2024):  {capacity_formula:.12f}")
print(f"Sys analytic:                   {sys_formula:.12f}")

# %% [markdown]
# ## Figure — Projections of the orbit to p- and q-planes


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
) -> plt.Figure:
    path_np = path.detach().cpu().numpy()
    # Ordering in cycles is (q, p); respect that in projections
    q_coords = path_np[:, :2]
    p_coords = path_np[:, 2:]
    colors = cm.rainbow(np.linspace(0.0, 1.0, max(1, len(path_np) - 1)))

    fig, (ax_p, ax_q) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Closed characteristic on K×T — projections to q and p")

    def draw_panel(ax: plt.Axes, coords: np.ndarray, name: str) -> None:
        # name is either 'q' or 'p'
        if name == "q":
            ax.set_title("q-plane: K")
        else:
            ax.set_title(r"p-plane: T = R$_{\pi/2}$K")
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
        # Arrow to indicate direction (place near 70% along segment)
        for _ax, seg in ((ax_q, q_coords), (ax_p, p_coords)):
            x0, y0 = seg[idx]
            x1, y1 = seg[idx + 1]
            dx, dy = (x1 - x0, y1 - y0)
            _ax.annotate(
                "",
                xy=(x0 + 0.7 * dx, y0 + 0.7 * dy),
                xytext=(x0 + 0.55 * dx, y0 + 0.55 * dy),
                arrowprops=dict(arrowstyle="-|>", color=colors[idx], lw=1.5, alpha=line_alpha),
            )

    draw_panel(ax_q, q_coords, "q")
    draw_panel(ax_p, p_coords, "p")

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

    ax_q.plot(
        pentagon[:, 0],
        pentagon[:, 1],
        color="dimgray",
        linewidth=1.5,
        linestyle="--",
        label=r"K (q-plane)",
    )
    ax_p.plot(
        rotated_pentagon[:, 0],
        rotated_pentagon[:, 1],
        color="dimgray",
        linewidth=1.5,
        linestyle="--",
        label=r"T = R$_{\pi/2}$K (p-plane)",
    )

    # Axis labels reflect canonical coordinates on each factor
    ax_q.legend(loc="upper right")
    ax_q.set_xlabel("q1")
    ax_q.set_ylabel("q2")
    ax_p.legend(loc="upper right")
    ax_p.set_xlabel("p1")
    ax_p.set_ylabel("p2")

    plt.tight_layout()
    return fig


fig = plot_counterexample(
    GEOMETRY["cycle"],
    q_vertices=GEOMETRY["vertices_q"],
    p_vertices=GEOMETRY["vertices_p"],
)
plt.show()

# %% [markdown]
# ## Notes
# - Normalisation: Sys(B^4) = 1 for the Euclidean ball in R^4.
# - This instance (regular pentagons) violates Viterbo’s conjectured bound.
# - All computations are Torch-based and deterministic (CPU, float64).
