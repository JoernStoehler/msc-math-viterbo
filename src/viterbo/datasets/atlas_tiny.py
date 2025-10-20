"""AtlasTiny dataset helpers returning completed ragged rows.

This module assembles a deterministic roster of low-dimensional polytopes and
uses :mod:`viterbo.math` utilities to populate symplectic invariants. The
helpers return Python lists of typed dictionaries so callers can decide how to
batch (pad, collate, etc.) for their pipeline.

Schema (in-memory; float64 CPU tensors)
- Identity/meta:
  - ``polytope_id``: str
  - ``generator``: str (canonical label)
  - ``generator_config``: str (JSON string with params/seed)
  - ``dimension``: int
  - ``num_vertices``: int
  - ``num_facets``: int
- Geometry:
  - ``vertices``: (M, D) float64
  - ``normals``: (F, D) float64
  - ``offsets``: (F,) float64
  - ``minimal_action_cycle``: (C, D) float64 or None
- Quantities (nullable):
  - ``volume``: torch.Tensor (scalar) — always present
  - ``capacity_ehz``: torch.Tensor | None (present in 2D and 4D products only)
  - ``systolic_ratio``: torch.Tensor | None (present iff capacity present)
- Backend labels (nullable):
  - ``volume_backend``: str ("area2d" for D=2 else "facets")
  - ``capacity_ehz_backend``: str | None ("area2d" in 2D; "minkowski_lp3" in 4D
    product; None otherwise)
  - ``systolic_ratio_backend``: str | None ("formula" when computed)
- Walltimes (seconds; nullable where not run):
  - ``time_generator``, ``time_volume_area2d``, ``time_volume_facets``,
    ``time_capacity_area2d``, ``time_capacity_minkowski_lp3``,
    ``time_systolic_ratio`` (all float; None if not executed)
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import TypedDict

import torch

from viterbo.math.constructions import rotated_regular_ngon2d
from viterbo.math.polytope import vertices_to_halfspaces


class AtlasTinyRaggedRow(TypedDict):
    """AtlasTiny row spec before geometry and derived quantities are attached."""

    polytope_id: str
    generator: str
    generator_config: str


class AtlasTinyRow(TypedDict):
    """Completed AtlasTiny row with geometry, quantities, labels, and timings."""

    # Identity/meta
    polytope_id: str
    generator: str
    generator_config: str
    dimension: int
    num_vertices: int
    num_facets: int

    # Geometry
    vertices: torch.Tensor
    normals: torch.Tensor
    offsets: torch.Tensor
    minimal_action_cycle: torch.Tensor | None

    # Quantities (nullable)
    volume: torch.Tensor
    capacity_ehz: torch.Tensor | None
    systolic_ratio: torch.Tensor | None

    # Backend labels (nullable)
    volume_backend: str
    capacity_ehz_backend: str | None
    systolic_ratio_backend: str | None

    # Walltimes (seconds; nullable)
    time_generator: float
    time_volume_area2d: float | None
    time_volume_facets: float | None
    time_capacity_area2d: float | None
    time_capacity_minkowski_lp3: float | None
    time_systolic_ratio: float | None


def atlas_tiny_generate() -> list[AtlasTinyRaggedRow]:
    """Return deterministic roster specs for AtlasTiny v1."""

    def cfg(d: dict) -> str:
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    roster: list[AtlasTinyRaggedRow] = [
        # 2D
        {
            "polytope_id": "unit_square",
            "generator": "unit_square",
            "generator_config": cfg({}),
        },
        {
            "polytope_id": "triangle_area_one",
            "generator": "triangle_area_one",
            "generator_config": cfg({}),
        },
        {
            "polytope_id": "regular_pentagon",
            "generator": "regular_ngon",
            "generator_config": cfg({"k": 5, "angle": 0.0}),
        },
        {
            "polytope_id": "random_hexagon_seed41",
            "generator": "random_polygon",
            "generator_config": cfg({"seed": 41, "k": 6}),
        },
        # 4D
        {
            "polytope_id": "orthogonal_simplex_4d",
            "generator": "regular_simplex",
            "generator_config": cfg({"d": 4}),
        },
        {
            "polytope_id": "pentagon_product_counterexample",
            "generator": "lagrangian_product",
            "generator_config": cfg(
                {
                    "variant": "pentagon_product_counterexample",
                    "factors": [
                        {"type": "regular_ngon", "k": 5, "angle": 0.0},
                        {"type": "regular_ngon", "k": 5, "angle": -0.5 * float(torch.pi)},
                    ],
                }
            ),
        },
        {
            "polytope_id": "noisy_pentagon_product",
            "generator": "lagrangian_product",
            "generator_config": cfg(
                {
                    "variant": "noisy_pentagon_product",
                    "seed_q": 314159,
                    "seed_p": 271828,
                    "amp": 0.03,
                }
            ),
        },
        {
            "polytope_id": "mixed_nonproduct_from_product",
            "generator": "mixed_nonproduct_from_product",
            "generator_config": cfg(
                {
                    "base": "noisy_pentagon_product",
                    "seed_q": 314159,
                    "seed_p": 271828,
                    "amp": 0.03,
                    "mix": "default",
                }
            ),
        },
        {
            "polytope_id": "random_vrep_4d_seed20241017",
            "generator": "random_polytope_algorithm2",
            "generator_config": cfg({"seed": 20241017, "num_vertices": 10, "dimension": 4}),
        },
        {
            "polytope_id": "random_hrep_4d_seed20241017",
            "generator": "random_polytope_algorithm1",
            "generator_config": cfg({"seed": 20241017, "num_facets": 10, "dimension": 4}),
        },
    ]
    return roster


def atlas_tiny_complete_row(row: AtlasTinyRaggedRow) -> AtlasTinyRow:
    """Populate geometry, quantities, labels, and timings for a roster spec."""

    from viterbo.math import constructions as C
    from viterbo.math.capacity_ehz.cycle import minimal_action_cycle
    from viterbo.math.capacity_ehz.ratios import systolic_ratio
    from viterbo.math.volume import volume as volume_from_vertices

    def _time_call(fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        t1 = time.perf_counter()
        return out, float(t1 - t0)

    cfg = json.loads(row["generator_config"]) if row["generator_config"] else {}
    gen = row["generator"]

    # Generate geometry (float64 CPU) and record walltime.
    if gen == "unit_square":
        (vertices, normals, offsets), t_gen = _time_call(C.unit_square)
    elif gen == "triangle_area_one":
        (vertices, normals, offsets), t_gen = _time_call(C.triangle_area_one)
    elif gen == "regular_ngon":
        k = int(cfg.get("k", 3))
        angle = float(cfg.get("angle", 0.0))
        (v, _, _), t_gen = _time_call(rotated_regular_ngon2d, k, angle)
        vertices = v.to(dtype=torch.float64, device=torch.device("cpu"))
        normals, offsets = vertices_to_halfspaces(vertices)
    elif gen == "random_polygon":
        seed = int(cfg["seed"])  # required
        k = int(cfg["k"])  # required
        (vertices, normals, offsets), t_gen = _time_call(C.random_polygon, seed, k)
    elif gen == "regular_simplex":
        d = int(cfg["d"])  # required
        (vertices, normals, offsets), t_gen = _time_call(C.regular_simplex, d)
    elif gen == "lagrangian_product":
        variant = cfg.get("variant", "pentagon_product_counterexample")
        if variant == "pentagon_product_counterexample":
            (vertices, normals, offsets), t_gen = _time_call(C.pentagon_product_counterexample)
        elif variant == "noisy_pentagon_product":
            seed_q = int(cfg.get("seed_q", 314159))
            seed_p = int(cfg.get("seed_p", 271828))
            amp = float(cfg.get("amp", 0.03))
            (vertices, normals, offsets), t_gen = _time_call(
                C.noisy_pentagon_product, seed_q, seed_p, amp
            )
        else:
            raise ValueError(f"Unknown lagrangian_product variant: {variant}")
    elif gen == "mixed_nonproduct_from_product":
        (vertices, normals, offsets), t_gen = _time_call(C.mixed_nonproduct_from_product)
    elif gen == "random_polytope_algorithm1":
        seed = int(cfg["seed"])  # required
        num_facets = int(cfg.get("num_facets", 10))
        dimension = int(cfg.get("dimension", 4))
        (vertices, normals, offsets), t_gen = _time_call(
            C.random_polytope_algorithm1, seed, num_facets, dimension
        )
    elif gen == "random_polytope_algorithm2":
        seed = int(cfg["seed"])  # required
        num_vertices = int(cfg.get("num_vertices", 10))
        dimension = int(cfg.get("dimension", 4))
        (vertices, normals, offsets), t_gen = _time_call(
            C.random_polytope_algorithm2, seed, num_vertices, dimension
        )
    else:
        raise ValueError(f"Unknown generator label: {gen}")

    # Enforce float64 CPU discipline for dataset tensors.
    vertices = vertices.to(dtype=torch.float64, device=torch.device("cpu"))
    normals = normals.to(dtype=torch.float64, device=torch.device("cpu"))
    offsets = offsets.to(dtype=torch.float64, device=torch.device("cpu"))

    if vertices.ndim != 2:
        raise ValueError("vertices must be (M, D) tensor")
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, D) and offsets must be (F,) tensor")
    if vertices.device != normals.device or vertices.device != offsets.device:
        raise ValueError("vertices, normals, and offsets must share the same device")
    if vertices.dtype != normals.dtype or vertices.dtype != offsets.dtype:
        raise ValueError("vertices, normals, and offsets must share the same dtype")

    dim = int(vertices.size(1))

    # Volume + backend/time label
    if dim == 2:
        (volume, t_vol), volume_backend = _time_call(volume_from_vertices, vertices), "area2d"
        time_volume_area2d: float | None = t_vol
        time_volume_facets: float | None = None
    else:
        (volume, t_vol), volume_backend = _time_call(volume_from_vertices, vertices), "facets"
        time_volume_area2d = None
        time_volume_facets = t_vol

    # Capacity + cycle + backend/time label (nullable)
    capacity: torch.Tensor | None = None
    cycle: torch.Tensor | None = None
    capacity_backend: str | None = None
    time_capacity_area2d: float | None = None
    time_capacity_minkowski_lp3: float | None = None

    if dim == 2:
        (cap_and_cycle, t_cap) = _time_call(minimal_action_cycle, vertices, normals, offsets)
        capacity, cycle = cap_and_cycle
        capacity_backend = "area2d"
        time_capacity_area2d = t_cap
    elif dim == 4:
        try:
            (cap_and_cycle, t_cap) = _time_call(minimal_action_cycle, vertices, normals, offsets)
            capacity, cycle = cap_and_cycle
            capacity_backend = "minkowski_lp3"
            time_capacity_minkowski_lp3 = t_cap
        except NotImplementedError:
            capacity, cycle, capacity_backend = None, None, None

    # Systolic ratio (nullable)
    systolic: torch.Tensor | None = None
    systolic_backend: str | None = None
    time_systolic_ratio: float | None = None
    if capacity is not None:
        (systolic, t_sys) = _time_call(systolic_ratio, volume, capacity, dim)
        systolic_backend = "formula"
        time_systolic_ratio = t_sys

    return AtlasTinyRow(
        # Identity/meta
        polytope_id=row["polytope_id"],
        generator=row["generator"],
        generator_config=row["generator_config"],
        dimension=dim,
        num_vertices=int(vertices.size(0)),
        num_facets=int(normals.size(0)),
        # Geometry
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        minimal_action_cycle=cycle,
        # Quantities
        volume=volume,
        capacity_ehz=capacity,
        systolic_ratio=systolic,
        # Backend labels
        volume_backend=volume_backend,
        capacity_ehz_backend=capacity_backend,
        systolic_ratio_backend=systolic_backend,
        # Walltimes
        time_generator=t_gen,
        time_volume_area2d=time_volume_area2d,
        time_volume_facets=time_volume_facets,
        time_capacity_area2d=time_capacity_area2d,
        time_capacity_minkowski_lp3=time_capacity_minkowski_lp3,
        time_systolic_ratio=time_systolic_ratio,
    )


def atlas_tiny_build() -> list[AtlasTinyRow]:
    """Return completed AtlasTiny rows as a list of typed dictionaries."""

    rows = atlas_tiny_generate()
    return [atlas_tiny_complete_row(row) for row in rows]


def atlas_tiny_collate_pad(rows: Sequence[AtlasTinyRow]) -> dict[str, torch.Tensor | list[str]]:
    """Pad a batch of AtlasTiny rows to the maximum vertex/facet counts.

    Args:
      rows: sequence of completed AtlasTiny rows.

    Returns:
      Dictionary with padded tensors:
        - ``polytope_id``: list[str]
        - ``generator``: list[str]
        - scalar quantities ``volume``, ``capacity_ehz``, ``systolic_ratio`` of shape (B,)
        - ``vertices``: (B, V_max, D)
        - ``normals``: (B, F_max, D)
        - ``offsets``: (B, F_max)
        - ``minimal_action_cycle``: (B, C_max, D)
        - ``vertex_mask``: (B, V_max) bool
        - ``facet_mask``: (B, F_max) bool
        - ``cycle_mask``: (B, C_max) bool
      Other string/label/timing metadata remain per-row lists, not padded.
    """

    if not rows:
        raise ValueError("atlas_tiny_collate_pad requires a non-empty batch")

    dtype = rows[0]["vertices"].dtype
    device = rows[0]["vertices"].device
    dim = rows[0]["vertices"].size(1)

    max_vertices = max(row["vertices"].size(0) for row in rows)
    max_facets = max(row["normals"].size(0) for row in rows)
    max_cycle = max(
        row["minimal_action_cycle"].size(0) if row["minimal_action_cycle"] is not None else 0
        for row in rows
    )

    batch_size = len(rows)
    vertices = torch.zeros((batch_size, max_vertices, dim), dtype=dtype, device=device)
    normals = torch.zeros((batch_size, max_facets, dim), dtype=dtype, device=device)
    offsets = torch.zeros((batch_size, max_facets), dtype=dtype, device=device)
    cycle = torch.zeros((batch_size, max_cycle, dim), dtype=dtype, device=device)

    vertex_mask = torch.zeros((batch_size, max_vertices), dtype=torch.bool, device=device)
    facet_mask = torch.zeros((batch_size, max_facets), dtype=torch.bool, device=device)
    cycle_mask = torch.zeros((batch_size, max_cycle), dtype=torch.bool, device=device)

    volume = torch.zeros((batch_size,), dtype=dtype, device=device)
    capacity = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)
    systolic = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)

    polytope_ids: list[str] = []
    generators: list[str] = []

    for i, row in enumerate(rows):
        polytope_ids.append(row["polytope_id"])
        generators.append(row["generator"])

        v = row["vertices"]
        n = row["normals"]
        o = row["offsets"]
        vertices[i, : v.size(0)] = v
        normals[i, : n.size(0)] = n
        offsets[i, : o.size(0)] = o
        vertex_mask[i, : v.size(0)] = True
        facet_mask[i, : n.size(0)] = True

        if row["minimal_action_cycle"] is not None and row["minimal_action_cycle"].size(0) > 0:
            c = row["minimal_action_cycle"]
            cycle[i, : c.size(0)] = c
            cycle_mask[i, : c.size(0)] = True

        volume[i] = row["volume"]
        if row["capacity_ehz"] is not None:
            capacity[i] = row["capacity_ehz"]
        if row["systolic_ratio"] is not None:
            systolic[i] = row["systolic_ratio"]

    return {
        "polytope_id": polytope_ids,
        "generator": generators,
        "vertices": vertices,
        "normals": normals,
        "offsets": offsets,
        "minimal_action_cycle": cycle,
        "vertex_mask": vertex_mask,
        "facet_mask": facet_mask,
        "cycle_mask": cycle_mask,
        "volume": volume,
        "capacity_ehz": capacity,
        "systolic_ratio": systolic,
    }
