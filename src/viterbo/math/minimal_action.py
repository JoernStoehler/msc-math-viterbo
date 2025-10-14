"""EHZ capacity and minimal action cycles (4D focus, stubs included)."""

from __future__ import annotations

import torch

from viterbo.math.polytope import halfspaces_to_vertices, support, support_argmax


def minimal_action_cycle_lagrangian_product(
    vertices_q: torch.Tensor,
    normals_p: torch.Tensor,
    offsets_p: torch.Tensor,
    *,
    max_bounces: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal-action Reeb orbit for ``K × T`` with planar factors (≤3 bounces).

    Note:
        This combinatorial search assumes that the minimal orbit hits vertices of
        both polygons.  Centrally-symmetric pairs and the pentagon counterexample
        satisfy this, but generic Lagrangian products require facet-interior
        bounce points.  See :func:`minimal_action_cycle_lagrangian_product_generic`
        for the planned full solver.

    This implements the discrete search guaranteed by Rudolf (2022, Thm. 1.1):
    in ``R^{2}`` the EHZ capacity of a convex Lagrangian product is realised by
    a closed Minkowski billiard with at most three bounces.  We enumerate all
    two- and three-bounce candidates on the ``q``-polygon, enforce the strong
    billiard reflection rule at vertices, and return the cycle with minimal
    action, i.e. ``μ_{T^◦}(q_{j+1}-q_j)`` summed along the polygon.

    Args:
        vertices_q: (M, 2) tensor listing the ``q``-polygon vertices (unordered).
        normals_p: (F, 2) tensor of facet normals for the ``p``-polygon ``T``.
        offsets_p: (F,) tensor of offsets for ``T`` (``normals_p ⋅ x ≤ offsets_p``).
        max_bounces: Upper bound on the number of bounces to consider (2 or 3).

    Returns:
        (capacity, cycle):
          - capacity: scalar tensor equal to the minimal action.
          - cycle: (2 * bounces,) × 4 tensor listing successive points on the
            Reeb orbit with alternating ``q``- and ``p``-segments. The loop is
            represented without repeating the initial point at the end.

    Raises:
        ValueError: if the inputs are invalid or no admissible billiard exists.
    """
    if max_bounces not in (2, 3):
        raise ValueError("max_bounces must be 2 or 3")
    _validate_planar_vertices(vertices_q, "vertices_q")
    _validate_halfspaces_planar(normals_p, offsets_p, "normals_p", "offsets_p")

    dtype = vertices_q.dtype
    device = vertices_q.device
    vertices_q_ordered = _order_vertices_counter_clockwise(vertices_q)
    vertices_p = halfspaces_to_vertices(normals_p, offsets_p)
    tol = max(float(torch.finfo(dtype).eps) ** 0.5, 1e-9)

    best_action: torch.Tensor | None = None
    best_cycle: torch.Tensor | None = None

    if max_bounces >= 2:
        candidate = _minimal_action_two_bounce(vertices_q_ordered, vertices_p, tol)
        if candidate is not None:
            best_action, best_cycle = candidate

    if max_bounces >= 3:
        candidate = _minimal_action_three_bounce(vertices_q_ordered, vertices_p, tol)
        if candidate is not None:
            if best_action is None or candidate[0] < best_action - tol:
                best_action, best_cycle = candidate
            elif (
                best_action is not None
                and best_cycle is not None
                and torch.isclose(candidate[0], best_action, atol=tol, rtol=0.0)
            ):
                # Tie-break deterministically to favour lexicographically smaller cycles.
                if _cycle_lexicographic(candidate[1]) < _cycle_lexicographic(best_cycle):
                    best_action, best_cycle = candidate

    if best_action is None or best_cycle is None:
        raise ValueError("failed to locate an admissible Minkowski billiard trajectory")

    return best_action.to(dtype=dtype, device=device), best_cycle.to(dtype=dtype, device=device)


def _minimal_action_two_bounce(
    vertices_q: torch.Tensor, vertices_p: torch.Tensor, tol: float
) -> tuple[torch.Tensor, torch.Tensor] | None:
    count_q = vertices_q.size(0)
    best_action: torch.Tensor | None = None
    best_cycle: torch.Tensor | None = None

    for i in range(count_q):
        for j in range(i + 1, count_q):
            direction = vertices_q[j] - vertices_q[i]
            if torch.linalg.norm(direction) <= tol:
                continue

            forward_action, forward_idx = _support_with_index(vertices_p, direction)
            backward_action, backward_idx = _support_with_index(vertices_p, -direction)
            if forward_action <= tol or backward_action <= tol:
                continue
            total_action = forward_action + backward_action

            p_forward = vertices_p[forward_idx]
            p_backward = vertices_p[backward_idx]

            if not _satisfies_reflection_two_bounce(vertices_q, i, j, p_forward, p_backward, tol):
                continue

            cycle = torch.stack(
                [
                    torch.cat((vertices_q[i], p_forward)),
                    torch.cat((vertices_q[j], p_forward)),
                    torch.cat((vertices_q[j], p_backward)),
                    torch.cat((vertices_q[i], p_backward)),
                    torch.cat((vertices_q[i], p_forward)),
                ]
            )

            if best_action is None or total_action < best_action - tol:
                best_action = total_action
                best_cycle = cycle
            elif (
                best_action is not None
                and best_cycle is not None
                and torch.isclose(total_action, best_action, atol=tol, rtol=0.0)
            ):
                if _cycle_lexicographic(cycle) < _cycle_lexicographic(best_cycle):
                    best_action = total_action
                    best_cycle = cycle

    if best_action is None or best_cycle is None:
        return None
    return best_action, best_cycle


def _minimal_action_three_bounce(
    vertices_q: torch.Tensor, vertices_p: torch.Tensor, tol: float
) -> tuple[torch.Tensor, torch.Tensor] | None:
    count_q = vertices_q.size(0)
    if count_q < 3:
        return None

    best_action: torch.Tensor | None = None
    best_cycle: torch.Tensor | None = None

    for i in range(count_q - 2):
        for j in range(i + 1, count_q - 1):
            for k in range(j + 1, count_q):
                q_i = vertices_q[i]
                q_j = vertices_q[j]
                q_k = vertices_q[k]

                directions = [
                    q_j - q_i,
                    q_k - q_j,
                    q_i - q_k,
                ]

                actions: list[torch.Tensor] = []
                support_indices: list[int] = []
                skip = False
                for direction in directions:
                    if torch.linalg.norm(direction) <= tol:
                        skip = True
                        break
                    action, idx = _support_with_index(vertices_p, direction)
                    if action <= tol:
                        skip = True
                        break
                    actions.append(action)
                    support_indices.append(idx)
                if skip:
                    continue

                p_vertices = [vertices_p[idx] for idx in support_indices]

                if not _satisfies_reflection_three_bounce(vertices_q, (i, j, k), p_vertices, tol):
                    continue

                total_action = actions[0] + actions[1] + actions[2]

                cycle = torch.stack(
                    [
                        torch.cat((q_i, p_vertices[0])),
                        torch.cat((q_j, p_vertices[0])),
                        torch.cat((q_j, p_vertices[1])),
                        torch.cat((q_k, p_vertices[1])),
                        torch.cat((q_k, p_vertices[2])),
                        torch.cat((q_i, p_vertices[2])),
                        torch.cat((q_i, p_vertices[0])),
                    ]
                )

                if best_action is None or total_action < best_action - tol:
                    best_action = total_action
                    best_cycle = cycle
                elif (
                    best_action is not None
                    and best_cycle is not None
                    and torch.isclose(total_action, best_action, atol=tol, rtol=0.0)
                ):
                    if _cycle_lexicographic(cycle) < _cycle_lexicographic(best_cycle):
                        best_action = total_action
                        best_cycle = cycle

    if best_action is None or best_cycle is None:
        return None
    return best_action, best_cycle


def _support_with_index(
    vertices: torch.Tensor, direction: torch.Tensor
) -> tuple[torch.Tensor, int]:
    value, idx = support_argmax(vertices, direction)
    return value, idx


def _cycle_lexicographic(cycle: torch.Tensor) -> tuple[float, ...]:
    flattened = cycle.reshape(-1)
    return tuple(float(x.item()) for x in flattened)


def _satisfies_reflection_two_bounce(
    vertices_q: torch.Tensor,
    idx_a: int,
    idx_b: int,
    p_forward: torch.Tensor,
    p_backward: torch.Tensor,
    tol: float,
) -> bool:
    directions = [
        p_backward - p_forward,
        p_forward - p_backward,
    ]
    indices = [idx_a, idx_b]
    for direction, idx in zip(directions, indices, strict=True):
        if torch.linalg.norm(direction) <= tol:
            return False
        support_value = support(vertices_q, direction)
        candidate_value = torch.dot(vertices_q[idx], direction)
        if not torch.isclose(candidate_value, support_value, atol=tol, rtol=0.0):
            return False
    return True


def _satisfies_reflection_three_bounce(
    vertices_q: torch.Tensor,
    indices_q: tuple[int, int, int],
    p_vertices: list[torch.Tensor],
    tol: float,
) -> bool:
    directions = [
        -(p_vertices[0] - p_vertices[2]),
        -(p_vertices[1] - p_vertices[0]),
        -(p_vertices[2] - p_vertices[1]),
    ]
    for direction, idx in zip(directions, indices_q, strict=True):
        if torch.linalg.norm(direction) <= tol:
            return False
        support_value = support(vertices_q, direction)
        candidate_value = torch.dot(vertices_q[idx], direction)
        if not torch.isclose(candidate_value, support_value, atol=tol, rtol=0.0):
            return False
    return True


def _validate_planar_vertices(vertices: torch.Tensor, name: str) -> None:
    if vertices.ndim != 2 or vertices.size(1) != 2:
        raise ValueError(f"{name} must be a (N, 2) tensor")
    if vertices.size(0) < 3:
        raise ValueError(f"{name} must contain at least three vertices")


def _validate_halfspaces_planar(
    normals: torch.Tensor, offsets: torch.Tensor, normals_name: str, offsets_name: str
) -> None:
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError(f"{normals_name} must be (F, 2) and {offsets_name} must be (F,)")
    if normals.size(0) != offsets.size(0):
        raise ValueError(f"{normals_name} and {offsets_name} must share the first dimension")
    if normals.size(1) != 2:
        raise ValueError(f"{normals_name} must describe planar halfspaces")
    if torch.any(offsets <= 0):
        raise ValueError(f"{offsets_name} must be strictly positive for a valid convex body")


def minimal_action_cycle_lagrangian_product_generic(
    normals_q: torch.Tensor,
    offsets_q: torch.Tensor,
    normals_p: torch.Tensor,
    offsets_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Facet-interior Minkowski billiard solver (stub).

    Target behaviour: allow bounce points to move along facets, solving the
    stationarity conditions from Artstein-Avidan–Ostrover/Rudolf so the EHZ
    capacity is recovered for generic Lagrangian products where vertex contacts
    are non-minimal.  Implementation will require a small constrained optimiser
    over facet indices and barycentric coordinates.
    """
    raise NotImplementedError(
        "facet-interior Minkowski billiard solver is not implemented yet; "
        "the vertex-only solver can miss generic minimisers"
    )


def capacity_ehz_algorithm1(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    r"""Ekeland–Hofer–Zehnder capacity via the Artstein–Avidan–Ostrover program (2D placeholder)."""
    _ensure_planar(normals, offsets)
    vertices = halfspaces_to_vertices(normals, offsets)
    return _polygon_area(vertices)


def capacity_ehz_algorithm2(vertices: torch.Tensor) -> torch.Tensor:
    r"""EHZ capacity via discrete billiards on vertices (2D placeholder)."""
    if vertices.ndim != 2:
        raise ValueError("vertices must be a (M, d) tensor")
    d = vertices.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    if d != 2:
        raise NotImplementedError(
            "capacity_ehz_algorithm2 currently supports 2D only; 4D support is planned"
        )
    if vertices.size(0) < 3:
        raise ValueError("need at least three vertices for a 2D polygon")
    return _polygon_area(vertices)


def capacity_ehz_primal_dual(
    vertices: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    r"""Hybrid primal–dual EHZ capacity solver (2D placeholder)."""
    if vertices.ndim != 2:
        raise ValueError("vertices must be a (M, d) tensor")
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, d) and offsets must be (F,)")
    if normals.size(1) != vertices.size(1):
        raise ValueError("vertices and normals must share the same ambient dimension")
    d = normals.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    if d != 2:
        raise NotImplementedError(
            "capacity_ehz_primal_dual currently supports 2D only; 4D support is planned"
        )
    capacity = capacity_ehz_algorithm2(vertices)
    reference = capacity_ehz_algorithm1(normals, offsets)
    if not torch.allclose(capacity, reference, atol=1e-8, rtol=1e-8):
        raise ValueError("inconsistent primal and dual capacities for the provided polygon")
    return capacity


def minimal_action_cycle(
    vertices: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the minimal action Reeb orbit (2D placeholder)."""
    if vertices.ndim != 2:
        raise ValueError("vertices must be a (M, d) tensor")
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, d) and offsets must be (F,)")
    if vertices.size(1) != normals.size(1):
        raise ValueError("vertices and normals must share the same ambient dimension")
    d = vertices.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    if d != 2:
        raise NotImplementedError("minimal_action_cycle currently supports 2D only; 4D planned")
    ordered_vertices = _order_vertices_counter_clockwise(vertices)
    capacity = _polygon_area(ordered_vertices)
    return capacity, ordered_vertices


def systolic_ratio(
    volume: torch.Tensor, capacity_ehz: torch.Tensor, symplectic_dimension: int | None = None
) -> torch.Tensor:
    r"""Viterbo systolic ratio ``vol(K) / c_{EHZ}(K)^{n}`` for ``2n``-dimensional bodies."""
    if volume.ndim != 0 or capacity_ehz.ndim != 0:
        raise ValueError("volume and capacity_ehz must be scalar tensors")
    if torch.any(capacity_ehz <= 0):
        raise ValueError("capacity_ehz must be strictly positive")
    if symplectic_dimension is None:
        raise ValueError("symplectic_dimension must be provided for systolic_ratio")
    if symplectic_dimension % 2 != 0 or symplectic_dimension <= 0:
        raise ValueError("symplectic_dimension must be a positive even integer")
    n = symplectic_dimension // 2
    return volume / capacity_ehz.pow(n)


def _ensure_planar(normals: torch.Tensor, offsets: torch.Tensor) -> None:
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, d) and offsets must be (F,)")
    if normals.size(0) != offsets.size(0):
        raise ValueError("normals and offsets must have matching first dimension")
    if normals.size(1) != 2:
        raise NotImplementedError(
            "capacity_ehz_algorithm1 currently supports planar polytopes only"
        )


def _order_vertices_counter_clockwise(vertices: torch.Tensor) -> torch.Tensor:
    if vertices.ndim != 2 or vertices.size(1) != 2:
        raise ValueError("vertices must be (M, 2) tensor")
    if vertices.size(0) < 3:
        raise ValueError("need at least three vertices for a 2D polygon")
    centroid = vertices.mean(dim=0)
    shifted = vertices - centroid
    angles = torch.atan2(shifted[:, 1], shifted[:, 0])
    order = torch.argsort(angles)
    ordered = vertices[order]
    return ordered


def _polygon_area(vertices: torch.Tensor) -> torch.Tensor:
    ordered = _order_vertices_counter_clockwise(vertices)
    rolled = ordered.roll(shifts=-1, dims=0)
    cross = ordered[:, 0] * rolled[:, 1] - ordered[:, 1] * rolled[:, 0]
    area = 0.5 * torch.sum(cross)
    return area.abs()


# 4D-focused stubs -------------------------------------------------------------


def capacity_ehz_haim_kislev(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    r"""General EHZ formula (Haim–Kislev) operating on the H-representation (stub)."""
    d = normals.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    raise NotImplementedError


def oriented_edge_spectrum_4d(
    vertices: torch.Tensor,
    normals: torch.Tensor,
    offsets: torch.Tensor,
    *,
    k_max: int | None = None,
) -> torch.Tensor:
    r"""Hutchings-style oriented-edge action spectrum in R^4 (stub)."""
    if vertices.size(1) != 4:
        raise ValueError("oriented_edge_spectrum_4d expects vertices in R^4")
    raise NotImplementedError


def capacity_ehz_via_qp(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    r"""Facet-multiplier convex QP with QR-reduced constraints (stub)."""
    d = normals.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    raise NotImplementedError


def capacity_ehz_via_lp(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    r"""LP/SOCP relaxations (Krupp-style) providing bounds/warm-starts (stub)."""
    d = normals.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    raise NotImplementedError
