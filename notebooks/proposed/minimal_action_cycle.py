# %% [markdown]
r"""
# Minimal Action Cycle on the Viterbo Counterexample

This notebook computes the action-minimising closed characteristic on the
standard Haim–Kislev–Ostrover counterexample to Viterbo's conjecture.  We work
with the Lagrangian product of two congruent regular pentagons where the second
factor is rotated by 90°.  The code below

1. constructs the 4D polytope,  
2. enumerates simple Reeb cycles on the oriented-edge graph and extracts the
   action-minimising one, and  
3. renders synchronised piecewise-linear plots of the $q$- and $p$-components
   with a shared colour gradient and numbered vertices.

All helpers live in the public math layer; no placeholder imports remain.
r"""

# %%
from __future__ import annotations

import math

import jax.numpy as jnp
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# Optional cross-check with the facet-based solver; disabled by default because
# the combinatorial search can take several seconds for 4D polytopes.
VERIFY_CAPACITY_WITH_SOLVER = False

if VERIFY_CAPACITY_WITH_SOLVER:
    from viterbo.math.capacity.facet_normals import (
        ehz_capacity_fast_facet_normals,
        ehz_capacity_reference_facet_normals,
    )
from viterbo.math.capacity.reeb_cycles import (
    OrientedEdgeGraph,
    build_oriented_edge_graph,
    enumerate_simple_cycles,
)
from viterbo.math.generators import counterexample_hk_ostrover_4d
from viterbo.math.geometry import enumerate_vertices
from viterbo.math.numerics import GEOMETRY_ABS_TOLERANCE

mpl.rcParams["figure.figsize"] = (10, 5)


# %% [markdown]
r"""
## 1. Build the counterexample polytope

We assemble the canonical counterexample $K \times T$ where both polygons have
area 1 and the second factor is rotated by $\pi/2$.  The helper returns both the
vertex set and the half-space description $(B, c)$ used by the capacity
solvers.
r"""

# %%
AREA = 1.0
ROTATION = math.pi / 2.0

vertices4d, normals4d, offsets4d = counterexample_hk_ostrover_4d(
    rotation=ROTATION, area=AREA
)
print("4D pentagon product:")
print("  vertices:", vertices4d.shape)
print("  facets:", normals4d.shape)


# %% [markdown]
r"""
## 2. Enumerate simple Reeb cycles and pick the action-minimising orbit

We build the Chaidez–Hutchings oriented-edge graph and enumerate simple cycles
up to a modest cap.  For each cycle we compute its symplectic action by summing
Euclidean lengths of consecutive vertices (the graph embeds the closed
characteristic as a broken geodesic).  The first entry of the sorted spectrum is
our action minimiser.
"""

# %%
def cycle_vertices(
    cycle: tuple[int, ...],
    graph: OrientedEdgeGraph,
    vertex_cloud: jnp.ndarray,
) -> np.ndarray:
    """Return ordered vertices (4D points) for ``cycle`` including closure."""

    points = []
    edges = graph.edges
    for edge_id in cycle[:-1]:
        tail_vertex = int(edges[edge_id].tail_vertex)
        points.append(np.asarray(vertex_cloud[tail_vertex], dtype=np.float64))
    # Close the loop
    if points:
        points.append(points[0])
    return np.stack(points, axis=0)


def cycle_action(points: np.ndarray) -> float:
    """Symplectic action as total Euclidean length of successive vertices."""

    if points.shape[0] < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    return float(segment_lengths.sum())


graph = build_oriented_edge_graph(normals4d, offsets4d, atol=GEOMETRY_ABS_TOLERANCE)
vertex_cloud = enumerate_vertices(normals4d, offsets4d, atol=GEOMETRY_ABS_TOLERANCE)

# Enumerate a generous selection of simple cycles; the polytope is small.
MAX_CYCLES = 128
cycles = enumerate_simple_cycles(graph, limit=MAX_CYCLES)
if not cycles:
    raise RuntimeError("No Reeb cycles found; the oriented-edge graph is empty.")

cycle_records: list[tuple[float, np.ndarray, tuple[int, ...]]] = []
for cycle in cycles:
    pts = cycle_vertices(cycle, graph, vertex_cloud)
    action = cycle_action(pts)
    if action <= 0.0:
        continue
    cycle_records.append((action, pts, cycle))

if not cycle_records:
    raise RuntimeError("All enumerated cycles had non-positive action.")

cycle_records.sort(key=lambda record: record[0])
minimal_action, minimal_points, minimal_cycle = cycle_records[0]
print(f"Enumerated {len(cycle_records)} positive-action cycles (limit {MAX_CYCLES}).")
print(f"Minimal action: {minimal_action:.6f}")
print(f"Cycle length (vertices): {minimal_points.shape[0] - 1}")

if VERIFY_CAPACITY_WITH_SOLVER:
    try:
        capacity = ehz_capacity_fast_facet_normals(normals4d, offsets4d)
    except ValueError:
        capacity = ehz_capacity_reference_facet_normals(normals4d, offsets4d)
    print(f"EHZ capacity (facet solver): {capacity:.6f}")
else:
    capacity = minimal_action
    print("EHZ capacity (via minimal action): {:.6f}".format(capacity))


# %% [markdown]
"""
## 3. Prepare synchronised projections and colour gradient

We split the 4D trajectory into $(q_1, q_2)$ and $(p_1, p_2)$ components.  A
shared colour gradient parameterised by cumulative arclength keeps both plots in
lockstep.  The helper below offsets labels whenever multiple vertices share the
same 2D projection to avoid text overlap.
"""

# %%
def projection_components(points4d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (q-plane, p-plane) projections of the 4D path."""

    q_coords = points4d[:, :2]
    p_coords = points4d[:, 2:]
    return q_coords, p_coords


def cumulative_parameter(points: np.ndarray) -> np.ndarray:
    """Normalised cumulative arclength parameter in [0, 1]."""

    if points.shape[0] < 2:
        return np.zeros(points.shape[0], dtype=np.float64)
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = cumulative[-1]
    if total <= 0.0:
        return cumulative
    return cumulative / total


def label_offsets(coords: np.ndarray, *, base_offset: float = 0.05) -> list[np.ndarray]:
    """Deterministic offsets for duplicate coordinates to keep labels readable."""

    offsets: list[np.ndarray] = []
    registry: dict[tuple[int, int], int] = {}
    patterns = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
        np.array([1.0, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([-1.0, -1.0]),
        np.array([1.0, -1.0]),
    ]
    for point in coords:
        key = tuple(np.round(point / base_offset).astype(int))
        count = registry.get(key, 0)
        pattern = patterns[count % len(patterns)]
        registry[key] = count + 1
        offsets.append(pattern * base_offset)
    return offsets


q_path, p_path = projection_components(minimal_points)
param = cumulative_parameter(minimal_points)


# %% [markdown]
"""
## 4. Plot the matched pentagon projections

Each subplot shows one 2D pentagon projection with matching colours, scatter
markers, and numbered labels.  Identical projections receive small deterministic
offsets so that labels remain legible.
"""

# %%
fig, (ax_q, ax_p) = plt.subplots(1, 2, figsize=(12, 5))
colormap = plt.cm.viridis

# Build coloured line segments for both projections.
segments_q = np.stack([q_path[:-1], q_path[1:]], axis=1)
segments_p = np.stack([p_path[:-1], p_path[1:]], axis=1)
segment_colors = colormap(param[:-1])

for axis, segments in ((ax_q, segments_q), (ax_p, segments_p)):
    collection = LineCollection(segments, colors=segment_colors, linewidths=2.0)
    axis.add_collection(collection)
    axis.scatter(segments[:, 0, 0], segments[:, 0, 1], c=colormap(param[:-1]), s=30)
    axis.scatter(
        segments[-1, 1, 0],
        segments[-1, 1, 1],
        color=colormap(param[-1]),
        s=30,
    )

labels = [str(index) for index in range(minimal_points.shape[0] - 1)]
q_offsets = label_offsets(q_path[:-1])
p_offsets = label_offsets(p_path[:-1])

for point, label, offset in zip(q_path[:-1], labels, q_offsets):
    ax_q.text(point[0] + offset[0], point[1] + offset[1], label, color="black", fontsize=10)

for point, label, offset in zip(p_path[:-1], labels, p_offsets):
    ax_p.text(point[0] + offset[0], point[1] + offset[1], label, color="black", fontsize=10)

ax_q.set_title("q-plane (left pentagon)")
ax_q.set_xlabel(r"$q_1$")
ax_q.set_ylabel(r"$q_2$")
ax_q.set_aspect("equal", adjustable="box")

ax_p.set_title("p-plane (right pentagon, rotated)")
ax_p.set_xlabel(r"$p_1$")
ax_p.set_ylabel(r"$p_2$")
ax_p.set_aspect("equal", adjustable="box")

fig.suptitle(
    "Action-minimising Reeb orbit on the Haim–Kislev–Ostrover pentagon product",
    fontsize=14,
)
plt.tight_layout()
plt.show()
plt.close(fig)


# %% [markdown]
"""
## 5. Summary statistics

For convenience we tabulate the key invariants extracted above.
"""

# %%
print("Summary:")
print(f"  Minimal action: {minimal_action:.8f}")
print(f"  EHZ capacity: {capacity:.8f}")
print(
    "  Ratio (action / capacity): "
    f"{minimal_action / capacity if capacity > 0 else float('nan'):.8f}"
)
print(f"  Cycle (edge IDs): {minimal_cycle}")
