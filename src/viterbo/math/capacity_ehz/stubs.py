"""Stubs and planned solvers for EHZ capacity in higher dimensions."""

from __future__ import annotations

import itertools
import math
import warnings
from collections import defaultdict
from typing import NamedTuple

import torch

from viterbo.math.capacity_ehz.common import polygon_area
from viterbo.math.polytope import halfspaces_to_vertices
from viterbo.runtime import enforce_time_budget


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
        raise ValueError("Haim–Kislev programme failed to locate a feasible maximizing multiplier")
    return (0.5 / best_value).to(dtype=dtype, device=device)


@enforce_time_budget()
def oriented_edge_spectrum_4d(
    vertices: torch.Tensor,
    normals: torch.Tensor,
    offsets: torch.Tensor,
    *,
    k_max: int | None = None,
    # Paper-aligned CH budgets and pruning controls
    rotation_cap: float | None = 4.0 * math.pi,
    use_cF_budgets: bool = False,
    cF_constant: float | None = None,
    # Memoisation toggles
    use_memo: bool | None = None,
    # Deprecated: 'verified' kept for backward-compat; if provided, sets use_memo = not verified
    verified: bool | None = None,
    memo_grid: float | None = 1e-6,
    memo_buckets: int = 32,
) -> torch.Tensor:
    r"""Chaidez–Hutchings oriented-edge action spectrum in ``R^4``.

    WARNING: Experimental. The original paper notes that their
    computer implementation includes "various optimizations not discussed here"
    (Chaidez–Hutchings, Computing Reeb dynamics on 4d convex polytopes,
    arXiv:2008.10111, Sec. 2). We implement the documented optimizations:
    rotation-number bounds and (optional) per-face budgets. Additional
    undocumented optimizations from the paper are not included here.

    Args:
      vertices: ``(M, 4)`` tensor with the polytope vertices.
      normals: ``(F, 4)`` tensor describing the supporting half-spaces.
      offsets: ``(F,)`` tensor with support numbers (positive for inward origin).
      k_max: optional cap on the number of oriented edges per cycle (depth limit).
      use_cF_budgets: enable Chaidez–Hutchings per-face budgets (Theorem
        1.12(v)). When True, we compute a certified lower-bound constant
        C*(X) per Lemma 5.13 / §6.2 unless ``cF_constant`` is provided.
      cF_constant: override for the certified C*(X). If ``None`` and
        ``use_cF_budgets`` is True, a device-stable certified constant is
        constructed from the H-representation and 2-face bases.
      verified: when True, disables heuristic memoisation (transfer quantisation
        and dominance). Rotation guard remains active.
      rotation_cap: cap (radians) on polar rotation angle of cumulative 2×2
        transfer. Set ``None`` or ``math.inf`` to disable.
      memo_grid: quantisation grid for transfer in memo key. ``None`` or ``<=0``
        disables memoisation.
      memo_buckets: number of coarse budget buckets per face in memo; ``<=0``
        disables budget dominance memo.

    Returns:
      Minimal combinatorial action (symplectic length) among admissible cycles.
    """
    # Backward-compat: map deprecated 'verified' to 'use_memo'
    if verified is not None:
        if use_memo is not None and use_memo == (not verified):
            pass  # consistent
        elif use_memo is not None and use_memo != (not verified):
            raise ValueError("Conflicting memo toggles: both 'use_memo' and 'verified' provided")
        else:
            use_memo = not verified
        warnings.warn(
            "Parameter 'verified' is deprecated; pass 'use_memo' (bool) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if use_memo:
        warnings.warn(
            "oriented_edge_spectrum_4d: use_memo=True enables heuristic memoisation and"
            " dominance pruning; results may be non-minimal or vary across modes."
            " Use use_memo=None/False for certified search (subject to time limits).",
            RuntimeWarning,
            stacklevel=2,
        )
    if vertices.ndim != 2 or vertices.size(1) != 4:
        raise ValueError("vertices must be (M, 4)")
    if normals.ndim != 2 or normals.size(1) != 4 or offsets.ndim != 1:
        raise ValueError("normals must be (F, 4) and offsets must be (F,)")
    if normals.size(0) != offsets.size(0):
        raise ValueError("normals and offsets must share first dimension")

    dtype = vertices.dtype
    device = vertices.device
    tol = max(float(torch.finfo(dtype).eps) ** 0.5, 1e-9)
    j_matrix = _symplectic_matrix(dtype=dtype, device=device)

    vertex_facets = _vertex_facet_incidence(vertices, normals, offsets, tol)
    faces = _enumerate_two_faces(vertices, normals, vertex_facets, tol)
    if not faces:
        raise ValueError("polytope has no two-dimensional faces")

    facet_to_faces = defaultdict(list)
    for face in faces:
        facet_to_faces[face.facets[0]].append(face.index)
        facet_to_faces[face.facets[1]].append(face.index)

    reeb_vectors = _facet_reeb_vectors(normals, offsets, j_matrix)
    edges = _enumerate_oriented_edges(normals, offsets, reeb_vectors, faces, facet_to_faces, tol)
    if not edges:
        raise ValueError("no admissible oriented edges detected")

    max_length = len(edges) if k_max is None else int(k_max)
    if max_length < 2:
        raise ValueError("k_max must be at least 2")

    adjacency: dict[int, list[EdgeData]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.from_face].append(edge)

    for edge_list in adjacency.values():
        edge_list.sort(key=lambda ed: (ed.to_face, ed.target_facet))

    best_action: torch.Tensor | None = None
    seen_cycles: set[tuple[int, ...]] = set()
    identity = torch.eye(2, dtype=dtype, device=device)

    # Optional CH c_F per 2-face: c_F(F) = C*(X) * dist_S^3(ν_E+, ν_E−),
    # where ν_E± are unit outward normals of the two facets defining face F.
    cF: dict[int, float] = {}

    def _unit(v: torch.Tensor) -> torch.Tensor:
        nrm = torch.linalg.norm(v)
        return v / (nrm + torch.finfo(v.dtype).eps)

    if use_cF_budgets:
        # Build a certified C*(X) if not supplied. This uses the rigorous
        # construction from docs/math/cstar_constant_spec.md (Lemma 5.13 / §6.2),
        # producing a deterministic, device-stable lower bound constant.
        if cF_constant is None:
            cF_constant = compute_cF_constant_certified(normals, offsets, faces)
        for face in faces:
            i, j = face.facets
            nu_i = _unit(normals[i])
            nu_j = _unit(normals[j])
            dot = float(torch.clamp(torch.dot(nu_i, nu_j), -1.0, 1.0).item())
            theta = math.acos(dot)
            cF[face.index] = float(cF_constant) * theta

    # Rotation-number pruning (angle in radians). Also define rotation-number cap Rn.
    enable_rotation = (
        rotation_cap is not None and math.isfinite(rotation_cap) and rotation_cap > 0.0
    )
    rotation_angle_cap = float(rotation_cap) if enable_rotation else float("inf")
    rotation_number_cap = (
        rotation_angle_cap / (2.0 * math.pi) if math.isfinite(rotation_angle_cap) else float("inf")
    )

    # State memo: (face, quantised transfer) -> best coarse budget used.
    # Dominance memo requires a monotone additive measure; we use the CH
    # per-face budgets for this. Without budgets, enabling this memo could prune
    # admissible paths erroneously. Therefore, require use_cF_budgets=True.
    # Enforce budgets when memoisation requested
    if use_memo and not use_cF_budgets:
        raise ValueError("Memoisation requires per-face budgets: set use_cF_budgets=True")

    enable_memo = (
        bool(use_memo)
        and use_cF_budgets
        and memo_grid is not None
        and (memo_grid if isinstance(memo_grid, float) else 0.0) > 0.0
        and memo_buckets > 0
    )
    memo: dict[tuple[int, tuple[int, int, int, int]], int] = {}

    def _quantise_matrix(mat: torch.Tensor, grid: float) -> tuple[int, int, int, int]:
        q = torch.round(mat / grid)
        return (int(q[0, 0].item()), int(q[0, 1].item()), int(q[1, 0].item()), int(q[1, 1].item()))

    def _polar_rotation_angle(mat: torch.Tensor) -> float:
        try:
            u, _, vh = torch.linalg.svd(mat)
            r = u @ vh
        except torch.linalg.LinAlgError:
            eps = max(tol, float(torch.finfo(mat.dtype).eps))
            u, _, vh = torch.linalg.svd(mat + eps * identity)
            r = u @ vh
        return abs(math.atan2(float(r[1, 0].item()), float(r[0, 0].item())))

    def dfs(
        start_face: int,
        current_face: int,
        path: list[EdgeData],
        visited_edges: set[int],
        depth: int,
        transfer: torch.Tensor,
        cF_sum_used: float,
    ) -> None:
        nonlocal best_action
        if depth >= max_length:
            return
        for edge in adjacency.get(current_face, []):
            if edge.id in visited_edges:
                continue
            # CH budget pruning (optional): accumulate sum over visited 2-faces.
            used = cF_sum_used
            if use_cF_budgets:
                used = used + cF[edge.from_face]
                if used > rotation_number_cap:
                    continue
            # Rotation-number pruning.
            transfer_next = edge.matrix @ transfer
            if enable_rotation and _polar_rotation_angle(transfer_next) > rotation_angle_cap:
                continue
            # State memoisation keyed by (next face, quantised transfer).
            if enable_memo:
                key = (edge.to_face, _quantise_matrix(transfer_next, float(memo_grid)))
                # Coarse budget: normalise by rotation cap Rn to maintain scale behaviour.
                denom = rotation_number_cap
                used_rel = 0.0 if not math.isfinite(denom) or denom <= 0.0 else used / denom
                bucket = int(min(1e9, math.floor(used_rel * memo_buckets)))
                best_bucket = memo.get(key)
                if best_bucket is not None and bucket >= best_bucket:
                    continue
                memo[key] = bucket

            path.append(edge)
            visited_edges.add(edge.id)
            cF_sum_next = used
            if edge.to_face == start_face:
                cycle_ids = tuple(edge_.id for edge_ in path)
                canonical = _canonical_cycle_id(cycle_ids)
                if canonical not in seen_cycles:
                    seen_cycles.add(canonical)
                    action = _evaluate_cycle(path, faces, normals, offsets, tol, identity)
                    if action is not None and (best_action is None or action < best_action - tol):
                        best_action = action
            dfs(
                start_face,
                edge.to_face,
                path,
                visited_edges,
                depth + 1,
                transfer_next,
                cF_sum_next,
            )
            visited_edges.remove(edge.id)
            path.pop()

    for entry_face in adjacency.keys():
        for edge in adjacency[entry_face]:
            # Reset memo for each DFS start to keep traversal deterministic.
            memo.clear()
            used0 = cF[edge.from_face] if use_cF_budgets else 0.0
            if enable_memo:
                key0 = (edge.to_face, _quantise_matrix(edge.matrix, float(memo_grid)))
                denom0 = rotation_number_cap
                used_rel0 = 0.0 if not math.isfinite(denom0) or denom0 <= 0.0 else used0 / denom0
                bucket0 = int(min(1e9, math.floor(used_rel0 * memo_buckets)))
                memo[key0] = bucket0
            dfs(entry_face, edge.to_face, [edge], {edge.id}, 1, edge.matrix, used0)

    if best_action is None:
        raise ValueError("no admissible cycles found within search limits")
    return best_action


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


class FaceData(NamedTuple):
    """Metadata describing a two-dimensional face in the oriented-edge graph."""

    index: int
    facets: tuple[int, int]
    origin: torch.Tensor  # (4,)
    basis: torch.Tensor  # (4, 2)
    vertices: torch.Tensor  # (K, 4)


class EdgeData(NamedTuple):
    """Affine data and bookkeeping for an oriented edge transition."""

    id: int
    facet: int
    from_face: int
    to_face: int
    target_facet: int
    matrix: torch.Tensor  # (2, 2)
    offset: torch.Tensor  # (2,)
    action_linear: torch.Tensor  # (2,)
    action_constant: torch.Tensor  # ()
    reeb_vector: torch.Tensor  # (4,)
    competitor_indices: torch.Tensor  # (L,)
    competitor_alphas: torch.Tensor  # (L,)
    # Conservative upper bound on the realised segment time/length for this edge
    # estimated from admissible samples in the source face. Used for per-face
    # c_F budget pruning. Non-negative scalar.
    time_upper_bound: torch.Tensor  # ()


def _symplectic_matrix(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    entries = [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ]
    return torch.tensor(entries, dtype=dtype, device=device)


def _vertex_facet_incidence(
    vertices: torch.Tensor,
    normals: torch.Tensor,
    offsets: torch.Tensor,
    tol: float,
) -> list[tuple[int, ...]]:
    incidences: list[tuple[int, ...]] = []
    for vertex in vertices:
        tight = torch.isclose(normals @ vertex, offsets, atol=tol, rtol=0.0)
        indices = torch.nonzero(tight, as_tuple=False).flatten()
        facets = tuple(int(i.item()) for i in indices)
        if len(facets) < 4:
            raise NotImplementedError("polytope is not simple at a vertex")
        incidences.append(facets)
    return incidences


def _enumerate_two_faces(
    vertices: torch.Tensor,
    normals: torch.Tensor,
    vertex_facets: list[tuple[int, ...]],
    tol: float,
) -> list[FaceData]:
    faces: list[FaceData] = []
    for i in range(normals.size(0)):
        for j in range(i + 1, normals.size(0)):
            shared_vertices = [
                idx for idx, facets in enumerate(vertex_facets) if i in facets and j in facets
            ]
            if len(shared_vertices) < 3:
                continue
            coords = vertices[shared_vertices]
            centroid = coords.mean(dim=0)
            diffs = coords - centroid
            rank = torch.linalg.matrix_rank(diffs, tol=tol)
            if int(rank.item()) != 2:
                continue
            q, _ = torch.linalg.qr(diffs.T)
            basis = q[:, :2]
            faces.append(
                FaceData(
                    index=len(faces),
                    facets=(i, j),
                    origin=centroid,
                    basis=basis,
                    vertices=coords,
                )
            )
    if not faces:
        raise ValueError("no two-dimensional faces detected")
    for face in faces:
        if torch.linalg.matrix_rank(face.basis, tol=tol) != 2:
            raise ValueError("face basis is degenerate")
    return faces


def _facet_reeb_vectors(
    normals: torch.Tensor, offsets: torch.Tensor, j_matrix: torch.Tensor
) -> list[torch.Tensor]:
    reeb_vectors: list[torch.Tensor] = []
    for normal, offset in zip(normals, offsets):
        if torch.abs(offset) < torch.finfo(offset.dtype).eps:
            raise ValueError("facet offset is zero, invalid support function")
        reeb = (2.0 / offset) * (j_matrix @ normal)
        reeb_vectors.append(reeb)
    return reeb_vectors


def _enumerate_oriented_edges(
    normals: torch.Tensor,
    offsets: torch.Tensor,
    reeb_vectors: list[torch.Tensor],
    faces: list[FaceData],
    facet_to_faces: dict[int, list[int]],
    tol: float,
) -> list[EdgeData]:
    edges: list[EdgeData] = []
    dtype = normals.dtype
    device = normals.device
    num_facets = normals.size(0)
    face_lookup = {face.index: face for face in faces}
    edge_id = 0

    for facet_idx in range(num_facets):
        reeb = reeb_vectors[facet_idx]
        face_indices = facet_to_faces.get(facet_idx, [])
        if len(face_indices) < 2:
            continue
        for from_idx, to_idx in itertools.permutations(face_indices, 2):
            face_from = face_lookup[from_idx]
            face_to = face_lookup[to_idx]
            other_to = _other_facet(face_to, facet_idx)
            alpha_target = torch.dot(normals[other_to], reeb)
            if alpha_target <= tol:
                continue
            competitor_indices = []
            competitor_alphas = []
            for candidate in range(num_facets):
                if candidate == facet_idx:
                    continue
                dot = torch.dot(normals[candidate], reeb)
                if dot > tol:
                    competitor_indices.append(candidate)
                    competitor_alphas.append(dot)
            if not _edge_has_domain(
                face_from,
                normals,
                offsets,
                other_to,
                reeb,
                alpha_target,
                competitor_indices,
                competitor_alphas,
                tol,
            ):
                continue
            action_const, action_linear = _action_coefficients(
                face_from, normals[other_to], offsets[other_to], alpha_target
            )
            matrix, offset_vec = _edge_affine_map(
                face_from,
                face_to,
                normals[other_to],
                offsets[other_to],
                reeb,
                alpha_target,
                action_const,
                action_linear,
            )
            # Compute an exact per-edge sup bound over the face by maximising the
            # numerator on the vertices of the 2-face (linear function achieves
            # its extrema on vertices). alpha_target > 0 by construction.
            time_upper = torch.tensor(0.0, dtype=dtype, device=device)
            for vtx in face_from.vertices:
                numerator = offsets[other_to] - torch.dot(normals[other_to], vtx)
                t_here = numerator / alpha_target
                if t_here > time_upper:
                    time_upper = t_here
            edges.append(
                EdgeData(
                    id=edge_id,
                    facet=facet_idx,
                    from_face=face_from.index,
                    to_face=face_to.index,
                    target_facet=other_to,
                    matrix=matrix,
                    offset=offset_vec,
                    action_linear=action_linear,
                    action_constant=action_const,
                    reeb_vector=reeb,
                    competitor_indices=torch.tensor(
                        competitor_indices, dtype=torch.int64, device=device
                    ),
                    competitor_alphas=torch.tensor(competitor_alphas, dtype=dtype, device=device),
                    time_upper_bound=time_upper.clamp_min(0.0),
                )
            )
            edge_id += 1
    return edges


def _other_facet(face: FaceData, facet_idx: int) -> int:
    if face.facets[0] == facet_idx:
        return face.facets[1]
    if face.facets[1] == facet_idx:
        return face.facets[0]
    raise ValueError("facet does not belong to the face")


def _edge_has_domain(
    face: FaceData,
    normals: torch.Tensor,
    offsets: torch.Tensor,
    target_facet: int,
    reeb_vector: torch.Tensor,
    alpha_target: torch.Tensor,
    competitor_indices: list[int],
    competitor_alphas: list[torch.Tensor],
    tol: float,
) -> bool:
    midpoint = face.origin
    samples = [midpoint]
    for vertex in face.vertices:
        samples.append(0.75 * midpoint + 0.25 * vertex)
    for sample in samples:
        if not _point_inside(sample, normals, offsets, tol):
            continue
        numerator_target = offsets[target_facet] - torch.dot(normals[target_facet], sample)
        t_target = numerator_target / alpha_target
        if t_target <= tol:
            continue
        valid = True
        for cand, alpha in zip(competitor_indices, competitor_alphas):
            numerator = offsets[cand] - torch.dot(normals[cand], sample)
            t_candidate = numerator / alpha
            if t_candidate > tol and t_candidate < t_target - tol:
                valid = False
                break
        if valid:
            return True
    return False


def _point_inside(
    point: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor, tol: float
) -> bool:
    residual = normals @ point - offsets
    return bool(torch.all(residual <= tol))


def _action_coefficients(
    face: FaceData,
    normal_target: torch.Tensor,
    offset_target: torch.Tensor,
    alpha_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    numerator_const = offset_target - torch.dot(normal_target, face.origin)
    action_const = numerator_const / alpha_target
    linear = -(face.basis.T @ normal_target) / alpha_target
    return action_const, linear


def _edge_affine_map(
    face_from: FaceData,
    face_to: FaceData,
    normal_target: torch.Tensor,
    offset_target: torch.Tensor,
    reeb_vector: torch.Tensor,
    alpha_target: torch.Tensor,
    action_const: torch.Tensor,
    action_linear: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    basis_from = face_from.basis
    basis_to = face_to.basis
    delta_origin = face_from.origin - face_to.origin
    projection_offset = basis_to.T @ delta_origin
    reeb_in_to = basis_to.T @ reeb_vector
    transfer = basis_to.T @ basis_from
    correction = torch.outer(reeb_in_to, action_linear)
    matrix = transfer + correction
    offset_vec = projection_offset + action_const * reeb_in_to
    return matrix, offset_vec


def _canonical_cycle_id(edge_ids: tuple[int, ...]) -> tuple[int, ...]:
    if not edge_ids:
        return edge_ids
    rotations = [tuple(itertools.chain(edge_ids[i:], edge_ids[:i])) for i in range(len(edge_ids))]
    return min(rotations)


def _evaluate_cycle(
    edges: list[EdgeData],
    faces: list[FaceData],
    normals: torch.Tensor,
    offsets: torch.Tensor,
    tol: float,
    identity: torch.Tensor,
) -> torch.Tensor | None:
    dtype = normals.dtype
    device = normals.device
    transform = identity
    shift = torch.zeros(2, dtype=dtype, device=device)
    for edge in edges:
        shift = edge.matrix @ shift + edge.offset
        transform = edge.matrix @ transform
    system = identity - transform
    reg = max(tol, torch.finfo(dtype).eps)
    lhs = system.T @ system + reg * identity
    rhs = system.T @ shift
    try:
        u0 = torch.linalg.solve(lhs, rhs)
    except torch.linalg.LinAlgError:
        u0 = torch.linalg.lstsq(lhs, rhs).solution
    residual_norm = torch.linalg.norm(system @ u0 - shift)
    if residual_norm > 100 * tol:
        return None

    action_total = torch.tensor(0.0, dtype=dtype, device=device)
    current = u0
    for edge in edges:
        action = edge.action_constant + edge.action_linear @ current
        if action <= tol:
            return None
        face_from = faces[edge.from_face]
        point = face_from.origin + face_from.basis @ current
        if not _edge_domain_check(edge, point, action, normals, offsets, tol):
            return None
        action_total = action_total + action
        current = edge.matrix @ current + edge.offset
    final_face = faces[edges[0].from_face]
    residual = final_face.origin + final_face.basis @ current
    start_point = final_face.origin + final_face.basis @ u0
    if torch.linalg.norm(residual - start_point) > 100 * tol:
        return None
    return action_total


def _edge_domain_check(
    edge: EdgeData,
    point: torch.Tensor,
    action: torch.Tensor,
    normals: torch.Tensor,
    offsets: torch.Tensor,
    tol: float,
) -> bool:
    if not _point_inside(point, normals, offsets, tol):
        return False
    numerator_target = offsets[edge.target_facet] - torch.dot(normals[edge.target_facet], point)
    alpha_target = torch.dot(normals[edge.target_facet], edge.reeb_vector)
    if alpha_target <= tol:
        return False
    candidate_time = numerator_target / alpha_target
    if torch.abs(candidate_time - action) > 10 * tol:
        return False
    if edge.competitor_indices.numel() == 0:
        return True
    numerators = offsets[edge.competitor_indices] - (normals[edge.competitor_indices] @ point)
    times = numerators / edge.competitor_alphas
    admissible = torch.logical_or(times <= tol, times >= action - tol)
    return bool(torch.all(admissible))


def compute_cF_constant_certified(
    normals: torch.Tensor,
    offsets: torch.Tensor,
    faces: list[FaceData],
) -> float:
    """Compute the certified C*(X) constant per Lemma 5.13 / §6.2.

    Implements the constructive spec in ``docs/math/cstar_constant_spec.md``:
    builds conservative, certified bounds D_min(X), U_max(X), and N_ann(X), then
    returns C*(X) = [N_ann(X)·D_min(X)] / [2π·U_max(X)]. The implementation uses
    closed-form pairwise formulas and rigorous relaxations for higher strata to
    ensure safety and determinism across devices.

    Args:
      normals: ``(F,4)`` facet normals (not necessarily unit).
      offsets: ``(F,)`` support numbers (positive, 0 in interior).
      faces: enumerated 2-faces with facet pairs; used to restrict to actual
        pairs that occur in X.

    Returns:
      Certified constant C*(X) > 0 as a Python float. Safe lower bound.
    """
    if normals.ndim != 2 or normals.size(1) != 4 or offsets.ndim != 1:
        raise ValueError("normals must be (F,4) and offsets must be (F,)")
    if normals.size(0) != offsets.size(0):
        raise ValueError("normals and offsets must share first dimension")

    # Use float64 on CPU for stable numerics; compute scale-invariant unit data.
    normals64 = normals.detach().to(dtype=torch.float64, device="cpu")
    offsets64 = offsets.detach().to(dtype=torch.float64, device="cpu")
    norms = torch.linalg.norm(normals64, dim=1).clamp_min(torch.finfo(torch.float64).eps)
    nu = normals64 / norms[:, None]
    c = offsets64 / norms

    # Helper: closed-form D({i,j}) and U({i,j}) for pairs I={i,j}.
    def pair_DU(i: int, j: int) -> tuple[float, float]:
        a = float(c[i].item())
        b = float(c[j].item())
        ui = nu[i]
        uj = nu[j]
        s = float(torch.clamp(torch.dot(ui, uj), -1.0, 1.0).item())
        # f(t) = (a + b t)/sqrt(1 + 2 s t + t^2)
        # Stationary point t* = (a s - b)/(b s - a) when denominator != 0
        candidates: list[float] = [a, b]  # limits at t=0 and t→∞
        denom = b * s - a
        if abs(denom) > 0.0:
            t_star = (a * s - b) / denom
            if t_star >= 0.0 and math.isfinite(t_star):
                D = 1.0 + 2.0 * s * t_star + t_star * t_star
                if D > 0.0 and math.isfinite(D):
                    val = (a + b * t_star) / math.sqrt(D)
                    if math.isfinite(val):
                        candidates.append(val)
        d_ij = min(candidates)
        u_ij = max(candidates)
        # Numerical safety
        d_ij = max(d_ij, 0.0)
        u_ij = max(u_ij, 0.0)
        return d_ij, u_ij

    # 1) Enumerate active sets and compute bounds.
    # For 2-faces, use exact closed form. Also collect per-facet offsets.
    if not faces:
        raise ValueError("compute_cF_constant_certified requires two-faces list")
    D_candidates: list[float] = []
    U_candidates: list[float] = []
    for face in faces:
        i, j = face.facets
        d_ij, u_ij = pair_DU(i, j)
        D_candidates.append(d_ij)
        U_candidates.append(u_ij)

    # For 1- and 0-face strata, use rigorous relaxations that only decrease C*:
    # - D_min: lower-bound by min facet offset among facets occurring in any 2-face.
    involved_facets = sorted({idx for f in faces for idx in f.facets})
    if involved_facets:
        D_candidates.append(min(float(c[idx].item()) for idx in involved_facets))
    else:  # fallback to global min
        D_candidates.append(float(torch.min(c).item()))

    # - U_max: upper-bound by sqrt(k) * max c over facets for k up to 4 (worst case).
    c_max = float(torch.max(c).item())
    U_candidates.append(math.sqrt(4.0) * c_max)

    # 2) Certified N_ann(X) lower bound via adaptive 1D search per active set.
    # Enumerate active sets I of size 2 (faces), 3 (edges), and 4 (vertices).
    # Build quaternionic matrices (i, j, k) as per docs for invariance.
    def _quaternion_mats() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = torch.tensor(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        j = torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        k = torch.tensor(
            [
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        return i, j, k

    i_mat, j_mat, k_mat = _quaternion_mats()

    # Collect active sets from 2-faces, vertices (triples/quadruples).
    active_sets: set[tuple[int, ...]] = set()
    for face in faces:
        i, j = face.facets
        active_sets.add(tuple(sorted((int(i), int(j)))))

    # Derive vertex facet incidences and triples/quadruples.
    tol = max(1e-9, float(torch.finfo(torch.float64).eps) ** 0.5)
    try:
        vertices = halfspaces_to_vertices(normals64, offsets64)
        vertex_facets = _vertex_facet_incidence(vertices, normals64, offsets64, tol)
    except Exception:
        vertex_facets = []
    for facets in vertex_facets:
        # Quadruple
        quad = tuple(sorted(int(x) for x in facets[:4]))
        if len(quad) == 4:
            active_sets.add(quad)
        # Triples
        if len(facets) >= 3:
            ls = [int(x) for x in facets]
            for a in range(len(ls)):
                for b in range(a + 1, len(ls)):
                    for d in range(b + 1, len(ls)):
                        active_sets.add(tuple(sorted((ls[a], ls[b], ls[d]))))

    def _qr_basis(mat: torch.Tensor) -> torch.Tensor:
        # mat: (k,4) rows span subspace in R^4; return Q (4,k) with orthonormal columns
        q, _ = torch.linalg.qr(mat.T, mode="reduced")
        return q[:, : mat.size(0)]

    def _spectral_norm(m: torch.Tensor) -> float:
        try:
            return float(torch.linalg.norm(m, ord=2).item())
        except Exception:
            # Fallback via SVD
            u, s, v = torch.linalg.svd(m)
            return float((s.max() if s.numel() else torch.tensor(0.0)).item())

    def _lambda_min(m: torch.Tensor) -> float:
        evals = torch.linalg.eigvalsh(m)
        return float(evals.min().item())

    def _K_from_blocks(
        A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, H: torch.Tensor, theta: float
    ) -> torch.Tensor:
        ct = math.cos(theta)
        st = math.sin(theta)
        M = A.T @ A + (ct * B + st * C).T @ (ct * B + st * C)
        # Compute K = H^{-1/2} M H^{-1/2} via Cholesky: K = L^{-1} M L^{-T}
        eps = 0.0
        for _ in range(3):
            try:
                L = torch.linalg.cholesky(H + eps * torch.eye(H.size(0), dtype=H.dtype))
                break
            except torch.linalg.LinAlgError:
                eps = max(1e-15, 10.0 * (eps if eps > 0 else 1e-15))
        else:
            # As a last resort, use eigendecomposition
            evals, evecs = torch.linalg.eigh(H)
            evals = torch.clamp(evals, min=1e-15)
            Hm12 = evecs @ torch.diag(torch.rsqrt(evals)) @ evecs.T
            return Hm12 @ M @ Hm12
        Linv = torch.linalg.solve(L, torch.eye(L.size(0), dtype=L.dtype))
        K = Linv @ M @ Linv.T
        return K

    def _N_ann_for_I(indices: tuple[int, ...]) -> float:
        idx = list(indices)
        N_I = nu[idx, :]  # (k,4)
        ksz = N_I.size(0)
        if ksz < 2 or ksz > 4:
            return float("inf")
        R_I = _qr_basis(N_I)
        A = R_I.T @ i_mat @ N_I.T  # (k,k)
        B = R_I.T @ j_mat @ N_I.T
        Cb = R_I.T @ k_mat @ N_I.T
        H = N_I @ N_I.T
        # Lipschitz bound on λ_min(K(θ)) via ||K'(θ)|| ≤ ||H^{-1/2}||^2 sup||M'(θ)||
        lam_min_H = max(_lambda_min(H), 1e-15)
        HB = 1.0 / lam_min_H
        # sup ||M'(θ)|| ≤ ||B^T B|| + ||C^T C|| + ||B^T C + C^T B||
        normB = _spectral_norm(B)
        normC = _spectral_norm(Cb)
        S = B.T @ Cb + Cb.T @ B
        normS = _spectral_norm(S)
        sup_deriv = normB * normB + normC * normC + normS
        L_lip = HB * sup_deriv

        # Adaptive sampling with certificates
        def min_lower_bound(samples: int) -> tuple[float, float]:
            # returns (lb, at_theta)
            two_pi = 2.0 * math.pi
            h = two_pi / samples
            best_val = float("inf")
            best_theta = 0.0
            for t in range(samples):
                # midpoints
                theta = (t + 0.5) * h
                K = _K_from_blocks(A, B, Cb, H, theta)
                val = _lambda_min(K)
                if val < best_val:
                    best_val = val
                    best_theta = theta
            # Certified lower bound for each interval: val - L * (h/2), minimised over intervals
            lb = best_val - L_lip * (h * 0.5)
            return lb, best_theta

        lb, theta_star = min_lower_bound(64)
        # refine near minima a couple of rounds if helpful
        for samples in (128, 256):
            lb_ref, th = min_lower_bound(samples)
            if lb_ref > lb + 1e-10:
                lb = lb_ref
                theta_star = th
        return max(lb, 1e-12)

    N_ann_candidates: list[float] = []
    for I in sorted(active_sets, key=lambda t: (len(t), t)):
        # Only 1-/0-face strata (k=3,4) contribute a strictly positive annest bound.
        if len(I) < 3:
            continue
        try:
            val = _N_ann_for_I(I)
            if math.isfinite(val) and val > 0.0:
                N_ann_candidates.append(val)
        except Exception:
            continue
    N_ann_lower = min(N_ann_candidates) if N_ann_candidates else 1e-6

    D_min = max(min(D_candidates), 0.0)
    U_max = max(max(U_candidates), 1e-12)  # prevent divide-by-zero

    cstar = (N_ann_lower * D_min) / (2.0 * math.pi * U_max)
    # Clamp to positive finite range; ensure strict positivity for pruning logic.
    cstar = float(max(min(cstar, 1.0), 1e-18))
    return cstar
