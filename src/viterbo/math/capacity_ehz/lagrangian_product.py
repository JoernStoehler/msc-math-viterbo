"""Minkowski billiards on Lagrangian products K × T (planar factors).

Implements the vertex-contact discrete search (≤3 bounces) and helpers.
"""

from __future__ import annotations

import torch

from viterbo.math.capacity_ehz.common import (
    order_vertices_ccw,
    validate_halfspaces_planar,
    validate_planar_vertices,
)
from viterbo.math.polytope import halfspaces_to_vertices, support_argmax


def minimal_action_cycle_lagrangian_product(
    vertices_q: torch.Tensor,
    normals_p: torch.Tensor,
    offsets_p: torch.Tensor,
    *,
    max_bounces: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal-action Reeb orbit for K × T with planar factors (≤3 bounces)."""
    if max_bounces not in (2, 3):
        raise ValueError("max_bounces must be 2 or 3")
    validate_planar_vertices(vertices_q, "vertices_q")
    validate_halfspaces_planar(normals_p, offsets_p, "normals_p", "offsets_p")

    dtype = vertices_q.dtype
    device = vertices_q.device
    vertices_q_ordered = order_vertices_ccw(vertices_q)
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
        support_value = torch.dot(vertices_q[idx], direction)
        expected = torch.max(vertices_q @ direction)
        if not torch.isclose(support_value, expected, atol=tol, rtol=0.0):
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
        support_value = torch.dot(vertices_q[idx], direction)
        expected = torch.max(vertices_q @ direction)
        if not torch.isclose(support_value, expected, atol=tol, rtol=0.0):
            return False
    return True
