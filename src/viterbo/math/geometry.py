from __future__ import annotations

import math

import torch

from .halfspaces import vertices_to_halfspaces


def support(points: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Support function of a finite point set.

    Args:
      points: (N, D) float tensor.
      direction: (D,) float tensor. Not required to be normalized.

    Returns:
      Scalar tensor: max_i <points[i], direction>.
    """
    return (points @ direction).max()


def support_argmax(points: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Support value and index of the maximiser.

    Args:
      points: (N, D) float tensor.
      direction: (D,) float tensor.

    Returns:
      (value, index) where value is scalar tensor and index is Python int.
    """
    vals = points @ direction
    val, idx = torch.max(vals, dim=0)
    return val, int(idx.item())


def pairwise_squared_distances(points: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances.

    Args:
      points: (N, D) float tensor.

    Returns:
      (N, N) float tensor of squared distances.
    """
    # Using (x - y)^2 = x^2 + y^2 - 2xy trick for efficiency
    x2 = (points * points).sum(dim=1, keepdim=True)
    y2 = x2.transpose(0, 1)
    xy = points @ points.T
    d2 = x2 + y2 - 2.0 * xy
    return d2.clamp_min_(0.0)


def halfspace_violations(
    points: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Compute positive violations of halfspaces for each point.

    Given halfspaces Bx <= c (normals = B, offsets = c), returns
    relu(Bx - c) for all points.

    Args:
      points: (N, D)
      normals: (F, D)
      offsets: (F,)

    Returns:
      violations: (N, F) nonnegative.
    """
    bx = points @ normals.T  # (N, F)
    c = offsets.unsqueeze(0)  # (1, F)
    viol = bx - c
    return torch.clamp_min(viol, 0.0)


def bounding_box(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Axis-aligned bounding box (min, max) for a point cloud.

    Args:
      points: (N, D)

    Returns:
      (mins, maxs): each (D,)
    """
    return points.min(dim=0).values, points.max(dim=0).values


def _convex_hull_order_2d(vertices: torch.Tensor) -> torch.Tensor:
    unique_vertices = torch.unique(vertices, dim=0)
    centre = unique_vertices.mean(dim=0)
    rel = unique_vertices - centre
    angles = torch.atan2(rel[:, 1], rel[:, 0])
    order = torch.argsort(angles)
    return unique_vertices[order]


def _facet_polygon_area(vertices: torch.Tensor, normal: torch.Tensor, tol: float) -> torch.Tensor:
    if vertices.size(0) < 3:
        return torch.zeros((), dtype=vertices.dtype, device=vertices.device)
    centre = vertices.mean(dim=0)
    axis = torch.tensor([1.0, 0.0, 0.0], dtype=vertices.dtype, device=vertices.device)
    if torch.abs(normal[0]) > 0.9:
        axis = torch.tensor([0.0, 1.0, 0.0], dtype=vertices.dtype, device=vertices.device)
    u = torch.cross(normal, axis, dim=0)
    if torch.linalg.norm(u) <= tol:
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=vertices.dtype, device=vertices.device)
        u = torch.cross(normal, axis, dim=0)
    u = u / torch.linalg.norm(u)
    v = torch.cross(normal, u, dim=0)
    rel = vertices - centre
    coords = torch.stack((rel @ u, rel @ v), dim=1)
    ordered = _convex_hull_order_2d(coords)
    x = ordered[:, 0]
    y = ordered[:, 1]
    area = 0.5 * torch.abs(torch.sum(x * torch.roll(y, shifts=-1)) - torch.sum(y * torch.roll(x, shifts=-1)))
    return area


def matmul_vertices(matrix: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Apply linear map ``x -> A x`` to a vertex set.

    Args:
      matrix: (D, D) float tensor ``A``.
      vertices: (M, D) float tensor.

    Returns:
      new_vertices: (M, D) float tensor ``vertices @ A.T`` (implementation may differ).
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if vertices.ndim != 2:
        raise ValueError("vertices must be 2D")
    if matrix.size(0) != matrix.size(1) or matrix.size(1) != vertices.size(1):
        raise ValueError("matrix must be square with matching dimension to vertices")
    return vertices @ matrix.T


def translate_vertices(translation: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Translate a vertex set by ``t``.

    Args:
      translation: (D,) float tensor ``t``.
      vertices: (M, D) float tensor.

    Returns:
      new_vertices: (M, D) float tensor of translated points.
    """
    if translation.ndim != 1:
        raise ValueError("translation must be 1D tensor")
    if vertices.ndim != 2:
        raise ValueError("vertices must be 2D tensor")
    if translation.size(0) != vertices.size(1):
        raise ValueError("translation and vertices dimension mismatch")
    return vertices + translation


def volume(vertices: torch.Tensor) -> torch.Tensor:
    """Volume of a convex polytope specified by its vertices.

    The function currently supports ``D ∈ {1, 2, 3}`` by decomposing the input
    into intervals, polygons, or facet pyramids. To scale this routine beyond
    three dimensions we intend to ship three deterministic back ends whose
    docstrings spell out exact computational steps:

    1. :func:`volume_via_triangulation` — build an oriented convex hull from the
       vertices (Quickhull/Barber–Dobkin–Huhdanpaa 1996), extract a simplex
       fan, and accumulate signed simplex volumes via ``torch.linalg.det``.
    2. :func:`volume_via_lawrence` — operate directly on the H-representation
       ``Ax <= b`` and evaluate the Lawrence (1991) sign decomposition by
       iterating over facet bases and computing rational determinants.
    3. :func:`volume_via_monte_carlo` — offer a fallback using deterministic
       low-discrepancy sampling with stratified control variates when exact
       combinatorics become prohibitive.

    Keeping the current entry point small lets us dispatch to one of these
    specialised kernels depending on the available representation. Each backend
    will preserve torch dtype/device and provide hooks for gradient propagation
    (e.g., propagate barycentric coordinates for autodiff). All routines will
    reject unbounded polytopes by checking whether the hull or halfspaces span a
    full-dimensional cone.

    Args:
      vertices: ``(M, D)`` float tensor listing the vertex cloud. The points are
        assumed to span a full-dimensional convex body; duplicates are allowed
        and will be deduplicated internally once higher-dimensional support is
        added.

    Returns:
      Scalar float tensor with the Lebesgue volume of the convex hull of
      ``vertices``.

    References:
      Barber, Dobkin, Huhdanpaa (1996). ``The Quickhull algorithm for convex
      hulls``. ACM TOMS 22(4).
      Lawrence (1991). ``Polytope volume computation``. Mathematics of
      Computation 57(195).
      Büeler, Enge, Fukuda (2000). ``Exact volume computation for polytopes: A
      practical study``. Polytopes—Combinatorics and Computation.
    """
    if vertices.ndim != 2:
        raise ValueError("vertices must be (M, D)")
    dim = vertices.size(1)
    if dim == 0:
        raise ValueError("dimension must be positive")
    dtype = vertices.dtype
    device = vertices.device
    tol = max(float(torch.finfo(dtype).eps) ** 0.5, 1e-9)
    if dim == 1:
        return (vertices.max(dim=0).values - vertices.min(dim=0).values).abs()[0]
    if dim == 2:
        ordered = _convex_hull_order_2d(vertices)
        x = ordered[:, 0]
        y = ordered[:, 1]
        area = 0.5 * torch.abs(torch.sum(x * torch.roll(y, shifts=-1)) - torch.sum(y * torch.roll(x, shifts=-1)))
        return area
    if dim != 3:
        raise NotImplementedError("volume currently implemented for dimensions 1, 2, and 3 only")

    normals, offsets = vertices_to_halfspaces(vertices)
    centre = vertices.mean(dim=0)
    vol = torch.zeros((), dtype=dtype, device=device)
    for normal, offset in zip(normals, offsets):
        mask = torch.isclose(vertices @ normal, offset, atol=tol)
        facet_vertices = vertices[mask]
        area = _facet_polygon_area(facet_vertices, normal, tol)
        height = offset - (centre @ normal)
        vol = vol + area * height / 3.0
    return torch.abs(vol)


def volume_via_triangulation(vertices: torch.Tensor) -> torch.Tensor:
    """Deterministic convex hull triangulation volume estimator.

    The intended implementation constructs an oriented convex hull for the
    vertex cloud using a Quickhull-style incremental build, extracts a set of
    ``d``-simplices forming a fan around one interior point, and sums their
    signed volumes ``|det([v_1 - p, …, v_d - p])| / d!`` with PyTorch linear
    algebra. This backend is reliable up to ``d ≈ 7`` provided the hull is not
    massively degenerate.

    Args:
      vertices: ``(M, D)`` float tensor. Must span a bounded polytope of
        dimension ``D``.

    Returns:
      Scalar float tensor with the Lebesgue volume of the convex hull.

    Raises:
      NotImplementedError: pending triangulation kernel.
    """
    raise NotImplementedError


def volume_via_lawrence(
    normals: torch.Tensor, offsets: torch.Tensor, *, basis: torch.Tensor | None = None
) -> torch.Tensor:
    """Lawrence sign decomposition volume from supporting halfspaces.

    The Lawrence algorithm enumerates sets of ``d`` active facets, solves
    ``B x = c`` for each facet basis ``B`` (optionally preconditioned by
    ``basis``), and accumulates the signed contributions ``sign(det B)
    (⟨u, c⟩)^{d} / (d! det B)`` where ``u`` are the barycentric weights. We will
    follow the exact arithmetic roadmap of Büeler–Enge–Fukuda (2000) by
    rationalising the inputs and providing a fallback to 128-bit integers for
    certification in low dimensions.

    Args:
      normals: ``(F, D)`` float tensor ``A`` of outward normals.
      offsets: ``(F,)`` float tensor ``b`` with strictly positive entries.
      basis: Optional ``(D, D)`` float tensor providing a numeric basis change
        used to stabilise solves in near-degenerate configurations.

    Returns:
      Scalar float tensor with the polytope volume.

    Raises:
      NotImplementedError: pending Lawrence implementation.
    """
    raise NotImplementedError


def volume_via_monte_carlo(
    vertices: torch.Tensor,
    normals: torch.Tensor,
    offsets: torch.Tensor,
    *,
    samples: int,
    generator: torch.Generator | int,
) -> torch.Tensor:
    """Low-discrepancy Monte Carlo volume estimator for high dimensions.

    This stub will host a quasi-Monte Carlo estimator combining
    Sobol/Halton-generated samples with rejection sampling against the
    halfspaces ``Ax <= b``. We plan to add variance reduction by importance
    rescaling with the bounding box of ``vertices`` and adapt the sample budget
    based on concentration bounds (Dvoretzky–Kiefer–Wolfowitz inequality).

    Args:
      vertices: ``(M, D)`` float tensor used to build proposal distributions and
        bounding boxes.
      normals: ``(F, D)`` float tensor ``A`` of supporting normals.
      offsets: ``(F,)`` float tensor ``b``.
      samples: Number of quasi-random points to draw; must be ``>= D``.
      generator: ``torch.Generator`` or seed controlling the Sobol sequence.

    Returns:
      Scalar float tensor approximating the polytope volume.

    Raises:
      NotImplementedError: pending estimator implementation.
    """
    raise NotImplementedError


def rotated_regular_ngon2d(k: int, angle: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotated regular n-gon in 2D.

    Args:
      k: number of sides ``k >= 3``.
      angle: rotation angle in radians.

    Returns:
      (vertices, normals, offsets):
        - vertices: (k, 2)
        - normals: (k, 2)
        - offsets: (k,)
    """
    if k < 3:
        raise ValueError("k must be at least 3")
    dtype = torch.get_default_dtype()
    device = torch.device("cpu")
    angles = (torch.arange(k, dtype=dtype, device=device) * (2.0 * math.pi / k)) + angle
    vertices = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    normals = []
    offsets = []
    for i in range(k):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % k]
        edge = v1 - v0
        normal = torch.tensor([edge[1], -edge[0]], dtype=dtype, device=device)
        normal = normal / torch.linalg.norm(normal)
        offset = torch.dot(normal, v0)
        if offset < 0:
            normal = -normal
            offset = -offset
        normals.append(normal)
        offsets.append(offset)
    normals_tensor = torch.stack(normals)
    offsets_tensor = torch.stack(offsets)
    return vertices, normals_tensor, offsets_tensor
