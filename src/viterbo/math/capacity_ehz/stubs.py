"""Stubs and planned solvers for EHZ capacity in higher dimensions."""

from __future__ import annotations

import itertools
import math

import torch

from viterbo.math.capacity_ehz.common import polygon_area
from viterbo.math.polytope import halfspaces_to_vertices


def _symplectic_form_matrix(d: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if d % 2 != 0:
        raise ValueError("symplectic form is defined for even ambient dimensions only")
    n = d // 2
    eye = torch.eye(n, dtype=dtype, device=device)
    top = torch.cat((torch.zeros((n, n), dtype=dtype, device=device), eye), dim=1)
    bottom = torch.cat((-eye, torch.zeros((n, n), dtype=dtype, device=device)), dim=1)
    return torch.cat((top, bottom), dim=0)


def _nullspace_vectors(a: torch.Tensor, tol: float) -> torch.Tensor:
    """Return column-nullspace basis vectors of ``a`` (may be empty)."""
    if a.numel() == 0:
        return torch.empty((a.size(1), 0), dtype=a.dtype, device=a.device)
    svd_tol = tol * max(a.size(0), a.size(1))
    _, s, vh = torch.linalg.svd(a, full_matrices=True)
    if s.numel() == 0:
        return torch.eye(a.size(1), dtype=a.dtype, device=a.device)
    rank = int(torch.sum(s > svd_tol).item())
    if rank >= vh.size(0):
        return torch.empty((a.size(1), 0), dtype=a.dtype, device=a.device)
    return vh[rank:].T


def _candidate_betas(
    matrix: torch.Tensor,
    offsets: torch.Tensor,
    nullspace: torch.Tensor,
    tol: float,
    feas_tol: float,
) -> list[torch.Tensor]:
    """Enumerate non-negative ``beta`` vectors spanning ``nullspace``."""
    if nullspace.size(1) == 0:
        return []
    candidates: list[torch.Tensor] = []
    null_dim = nullspace.size(1)

    if null_dim == 1:
        ray = nullspace[:, 0]
        norm = torch.linalg.norm(ray)
        if norm <= tol:
            return []
        ray = ray / norm
        for sign in (1.0, -1.0):
            beta = sign * ray
            if torch.any(beta < -tol):
                continue
            beta = torch.where(beta >= 0, beta, torch.zeros_like(beta))
            if torch.all(beta <= tol):
                continue
            residual = matrix @ beta
            if torch.linalg.norm(residual, ord=float("inf")) > feas_tol:
                continue
            candidates.append(beta)
        return candidates

    if null_dim != 2:
        return []

    dtype = nullspace.dtype
    device = nullspace.device
    rows = nullspace  # (k, 2)

    boundaries: list[float] = []
    for row in rows:
        a = float(row[0].item())
        b = float(row[1].item())
        if abs(a) <= tol and abs(b) <= tol:
            continue
        angle = math.atan2(-a, b)
        angle_mod = (angle + math.pi) % math.pi
        boundaries.append(angle_mod)
    boundaries.extend([0.0, math.pi])
    # Deduplicate and sort.
    boundaries_sorted: list[float] = []
    for angle in sorted(boundaries):
        if not boundaries_sorted or abs(angle - boundaries_sorted[-1]) > 1e-8:
            boundaries_sorted.append(angle)
    if boundaries_sorted[-1] != math.pi:
        boundaries_sorted.append(math.pi)

    def feasible(theta: float) -> tuple[bool, torch.Tensor]:
        gamma = torch.tensor([math.cos(theta), math.sin(theta)], dtype=dtype, device=device)
        beta = rows @ gamma
        if torch.any(beta < -tol):
            return False, beta
        beta = torch.where(beta >= 0, beta, torch.zeros_like(beta))
        if torch.all(beta <= tol):
            return False, beta
        residual = matrix @ beta
        if torch.linalg.norm(residual, ord=float("inf")) > feas_tol:
            return False, beta
        return True, beta

    grid_points = 32
    for left, right in zip(boundaries_sorted[:-1], boundaries_sorted[1:]):
        if right - left <= 1e-8:
            continue
        mid = 0.5 * (left + right)
        ok, _ = feasible(mid)
        if not ok:
            continue
        for t in range(grid_points + 1):
            theta = left + (right - left) * (t / grid_points)
            ok, beta = feasible(theta)
            if not ok:
                continue
            candidates.append(beta)

    # Normalise candidates to avoid duplicates (scale invariance).
    filtered: list[torch.Tensor] = []
    for beta in candidates:
        scale = torch.dot(offsets, beta)
        if scale <= tol:
            continue
        beta_normed = beta / scale
        duplicate = False
        for existing in filtered:
            if torch.allclose(beta_normed, existing, atol=1e-6, rtol=0.0):
                duplicate = True
                break
        if not duplicate:
            filtered.append(beta_normed * scale)  # keep original scaling for downstream
    return [beta for beta in filtered]


def _maximum_triangular_sum(beta: torch.Tensor, omega: torch.Tensor) -> float:
    """Dynamic-programming maximum over permutations for ``beta``."""
    beta_np = beta.detach().cpu().numpy()
    omega_np = omega.detach().cpu().numpy()
    k = beta_np.shape[0]
    if k <= 1:
        return 0.0
    weights = beta_np[:, None] * beta_np[None, :] * omega_np
    size = 1 << k
    dp = [float("-inf")] * size
    dp[0] = 0.0
    for mask in range(1, size):
        best = float("-inf")
        for i in range(k):
            if not (mask & (1 << i)):
                continue
            prev = mask ^ (1 << i)
            base = dp[prev]
            if base == float("-inf"):
                continue
            contrib = base
            for j in range(k):
                if prev & (1 << j):
                    contrib += weights[i, j]
            if contrib > best:
                best = contrib
        dp[mask] = best
    return dp[-1]


def capacity_ehz_haim_kislev(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    r"""Haim–Kislev facet-multiplier programme for the EHZ capacity."""
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, d) and offsets must be (F,)")
    if normals.size(0) != offsets.size(0):
        raise ValueError("normals and offsets must share the first dimension")
    d = normals.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    facet_count = normals.size(0)
    if facet_count < d + 1:
        raise ValueError("need at least d + 1 facets to realise a non-trivial multiplier")
    dtype = normals.dtype
    device = normals.device
    eps = float(torch.finfo(dtype).eps)
    tol = max(eps**0.5, 1e-9)
    feas_tol = 10.0 * tol

    if d == 2:
        vertices = halfspaces_to_vertices(normals, offsets)
        area = polygon_area(vertices)
        return area.to(dtype=dtype, device=device)

    if torch.any(offsets <= 0):
        raise ValueError("offsets must be strictly positive for a bounded convex body")

    symplectic_form = _symplectic_form_matrix(d, dtype=dtype, device=device)
    omega_full = normals @ symplectic_form @ normals.T  # antisymmetric

    best_value: torch.Tensor | None = None
    if d >= 4:
        min_support = d + 1
        max_support = min(facet_count, d + 2)
    else:
        min_support = 2
        max_support = min(facet_count, d + 2)

    for support_size in range(min_support, max_support + 1):
        for support_indices in itertools.combinations(range(facet_count), support_size):
            idx = torch.tensor(support_indices, device=device, dtype=torch.long)
            sub_normals = normals[idx]
            sub_offsets = offsets[idx]
            rank = torch.linalg.matrix_rank(sub_normals, tol=tol)
            if int(rank.item()) < d:
                continue
            matrix = sub_normals.T  # (d, k)
            nullspace = _nullspace_vectors(matrix, tol)
            candidate_betas = _candidate_betas(matrix, sub_offsets, nullspace, tol, feas_tol)
            for beta in candidate_betas:
                normalisation = torch.dot(sub_offsets, beta)
                if normalisation <= tol:
                    continue
                beta = beta / normalisation
                omega_sub = omega_full[idx][:, idx]
                value_float = _maximum_triangular_sum(beta, omega_sub)
                if value_float <= tol:
                    continue
                value = torch.tensor(value_float, dtype=dtype, device=device)
                if best_value is None or value > best_value + tol:
                    best_value = value

    if best_value is None:
        raise ValueError(
            "Haim–Kislev programme failed to locate a feasible maximizing multiplier"
        )
    return (0.5 / best_value).to(dtype=dtype, device=device)


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
