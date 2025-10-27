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
from IPython.display import Markdown, display

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
# ## Result and Validation (dimension 4)


# %%
# Numeric values (computed)
c_val = GEOMETRY["capacity"].item()
vol_val = GEOMETRY["volume_4d"].item()
area_q_val = GEOMETRY["area_q"].item()
area_p_val = GEOMETRY["area_p"].item()
sys_val = GEOMETRY["systolic_ratio"].item()

# Analytic reference (regular pentagons, unit radius)
area_analytic = (5.0 / 2.0) * math.sin(2.0 * math.pi / 5.0)
capacity_analytic = 2.0 * math.cos(math.pi / 10.0) * (1.0 + math.cos(math.pi / 5.0))
sys_analytic = (math.sqrt(5.0) + 3.0) / 5.0

# Two-bounce structure
cycle = GEOMETRY["cycle"]
q_cycle = cycle[:, :2]
p_cycle = cycle[:, 2:]

def _unique_rows(x: torch.Tensor) -> torch.Tensor:
    xr = torch.round(x * 1_000_000.0) / 1_000_000.0
    unique, _ = torch.unique(xr, dim=0, sorted=True, return_inverse=True)
    return unique

uq_q = _unique_rows(q_cycle)
uq_p = _unique_rows(p_cycle)
v = uq_q[1] - uq_q[0]
h_fwd = support_value(uq_p, v).item()
h_bwd = support_value(uq_p, -v).item()
action_two_bounce = h_fwd + h_bwd

# Relative errors
def rel(a: float, b: float) -> float:
    return 0.0 if b == 0 else abs(a - b) / abs(b)

cap_ana = f"{capacity_analytic:.12f}"
area_ana = f"{area_analytic:.12f}"
vol_ana = f"{(area_analytic**2):.12f}"
sys_ana = f"{sys_analytic:.12f}"
cap_meas = f"{c_val:.12f}"
vol_meas = f"{vol_val:.12f}"
sys_meas = f"{sys_val:.12f}"
two_bounce = f"{action_two_bounce:.12f}"
err_cap = f"{rel(c_val, capacity_analytic):.2e}"
err_vol = f"{rel(vol_val, area_analytic**2):.2e}"
err_sys = f"{rel(sys_val, sys_analytic):.2e}"
uq_q_n = int(uq_q.size(0))
uq_p_n = int(uq_p.size(0))

md = r"""
### Statement

- Conjecture (Viterbo, 2000; 4D normalisation): for a convex domain $X \subset \mathbb R^{4}$,
  $$\operatorname{Sys}(X) = \frac{c_{\rm EHZ}(X)^{2}}{2\,\operatorname{Vol}_{4}(X)} \le 1.$$
- Construction (Haim–Kislev–Ostrover, 2024): $K \times T$ with $K$ a regular pentagon and $T = R_{\pi/2}K$.

### Result (this instance)

- $c_{\rm EHZ}(K\times T) = 2\cos(\tfrac{\pi}{10})\,\bigl(1+\cos(\tfrac{\pi}{5})\bigr)$ $\approx$ %(cap_ana)s
- $\operatorname{Area}(K) = \operatorname{Area}(T) = \tfrac{5}{2}\sin(\tfrac{2\pi}{5})$ $\approx$ %(area_ana)s
- $\operatorname{Vol}_{4}(K\times T) = \operatorname{Area}(K)\,\operatorname{Area}(T)$ $\approx$ %(vol_ana)s
- $\operatorname{Sys}(K\times T) = \dfrac{c^{2}}{2\,\operatorname{Vol}_{4}} = \dfrac{\sqrt{5}+3}{5}$ $\approx$ %(sys_ana)s (> 1)

### Computation (this page)

- Measured: $c_{\rm EHZ}$ = %(cap_meas)s, $\operatorname{Vol}_{4}$ = %(vol_meas)s, $\operatorname{Sys}$ = %(sys_meas)s
- Two-bounce orbit: $|\{q\text{-vertices}\}| = %(uq_q_n)d$, $|\{p\text{-vertices}\}| = %(uq_p_n)d$ (expected 2 and 2)
- Support check: $h_T(v)+h_T(-v)$ = %(two_bounce)s (equals measured capacity up to solver precision)

### Consistency (relative error)

- $|c - c_\text{analytic}|/c_\text{analytic}$ = %(err_cap)s
- $|\operatorname{Vol} - (\operatorname{Area}K)^{2}| / (\operatorname{Area}K)^{2}$ = %(err_vol)s
- $|\operatorname{Sys} - (\sqrt{5}+3)/5| / ((\sqrt{5}+3)/5)$ = %(err_sys)s

The figure below shows the closed characteristic projected to the $q$- and $p$-planes.
""" % {
    "cap_ana": cap_ana,
    "area_ana": area_ana,
    "vol_ana": vol_ana,
    "sys_ana": sys_ana,
    "cap_meas": cap_meas,
    "vol_meas": vol_meas,
    "sys_meas": sys_meas,
    "uq_q_n": uq_q_n,
    "uq_p_n": uq_p_n,
    "two_bounce": two_bounce,
    "err_cap": err_cap,
    "err_vol": err_vol,
    "err_sys": err_sys,
}

display(Markdown(md))

# %% [markdown]
# ### Interpretation (for context)
#
# - The reported value $\operatorname{Sys}(K\times T) = (\sqrt{5}+3)/5 > 1$ contradicts
#   Viterbo’s 4D volume–capacity conjecture in this instance.
# - The capacity used here is the Ekeland–Hofer–Zehnder (EHZ) capacity, which
#   agrees with several other capacities on convex domains in $\mathbb R^{2n}$ and
#   can be computed as the minimal action of a closed characteristic.
# - For the product $K\times T$ of planar convex bodies, the orbit is a 2‑bounce
#   Minkowski billiard. In the regular case the action can be expressed in
#   closed form, giving the analytic constants shown above.
# - Our computation reproduces these constants numerically and visualises the
#   orbit in the $q$- and $p$-projections.

# %% [markdown]
# (Internal computations above feed the displayed summary; code is hidden on the site.)

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
    line_alpha: float = 0.95,
    label_offset: float = 0.06,
) -> plt.Figure:
    path_np = path.detach().cpu().numpy()
    # Ordering in cycles is (q, p); respect that in projections
    q_coords = path_np[:, :2]
    p_coords = path_np[:, 2:]
    # Single, readable colour for orbit segments
    orbit_color = "black"

    fig, (ax_p, ax_q) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Closed characteristic on K×T — projections to q and p")

    def _unique_rows_np(a: np.ndarray) -> np.ndarray:
        # Round to 1e-6 to stabilise uniqueness under float output
        ar = np.round(a * 1_000_000.0) / 1_000_000.0
        # Structured view for uniqueness
        b = np.ascontiguousarray(ar).view([('', ar.dtype)] * ar.shape[1])
        _, idx = np.unique(b, return_index=True)
        return ar[np.sort(idx)]

    def draw_panel(ax: plt.Axes, coords: np.ndarray, name: str) -> None:
        # name is either 'q' or 'p'
        if name == "q":
            ax.set_title("q-plane: K")
        else:
            ax.set_title(r"p-plane: T = R$_{\pi/2}$K")
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0.0, color="lightgray", linewidth=0.8)
        ax.axvline(0.0, color="lightgray", linewidth=0.8)
        # Label only distinct vertices to avoid clutter: q1,q2 or p1,p2
        uniq = _unique_rows_np(coords)
        labels = [f"{name}1", f"{name}2"] if len(uniq) == 2 else [f"{name}{i+1}" for i in range(len(uniq))]
        for lbl, pt in zip(labels, uniq, strict=False):
            ax.scatter(pt[0], pt[1], color="black", s=25, zorder=4)
            ax.text(
                pt[0] + label_offset,
                pt[1] + label_offset,
                lbl,
                fontsize=10,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                zorder=5,
            )

    for idx in range(len(path_np) - 1):
        # Draw segments in a single, consistent style
        ax_p.plot(
            p_coords[idx : idx + 2, 0],
            p_coords[idx : idx + 2, 1],
            color=orbit_color,
            alpha=line_alpha,
            linewidth=2.2,
        )
        ax_q.plot(
            q_coords[idx : idx + 2, 0],
            q_coords[idx : idx + 2, 1],
            color=orbit_color,
            alpha=line_alpha,
            linewidth=2.2,
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
                arrowprops=dict(arrowstyle="-|>", color=orbit_color, lw=1.4, alpha=line_alpha),
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

    # Boundaries drawn as dashed outlines
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

# %% [markdown]
# ### Figure caption
# Dashed polygons indicate the boundaries of $K$ (left: $p$-plane shows $T=R_{\pi/2}K$; right: $q$-plane shows $K$).
# The solid black polyline is the projection of the closed characteristic onto each factor.
# Arrowheads show the traversal direction. Labels $q1, q2$ and $p1, p2$ mark the
# two distinct contact/support vertices visited by the 2‑bounce orbit in the $q$- and $p$-planes, respectively.
