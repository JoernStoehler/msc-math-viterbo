"""Symplectic-invariant polytope similarity primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from viterbo.types import Polytope


@dataclass(frozen=True)
class RadialProfileCache:
    """Softened radial profile of a centred polytope."""

    log_radii: Float[Array, " num_directions"]
    directions: Float[Array, " num_directions dimension"]
    softness: float
    epsilon: float


@dataclass(frozen=True)
class CorrelationSketch:
    """Sketch of the symplectic correlation distribution ω(x, y)."""

    samples: Float[Array, " num_pairs"]
    histogram: Float[Array, " num_bins"]
    bin_edges: Float[Array, " num_bins_plus_one"]
    num_pairs: int


@dataclass(frozen=True)
class JaccardMonteCarloCache:
    """Monte-Carlo artefacts for the symplectic Jaccard distance."""

    centred_normals: Float[Array, " num_facets dimension"]
    centred_offsets: Float[Array, " num_facets"]
    bounding_min: Float[Array, " dimension"]
    bounding_max: Float[Array, " dimension"]
    num_samples: int
    seed: int | None


@dataclass(frozen=True)
class SymplecticSpectrumCache:
    """Symplectic spectrum and Williamson transform for a polytope."""

    centred_covariance: Float[Array, " dimension dimension"]
    williamson_transform: Float[Array, " dimension dimension"]
    symplectic_eigenvalues: Float[Array, " n"]


def centre_polytope_vertices(polytope: Polytope) -> Float[Array, " dimension"]:
    """Return the translation that recentres ``polytope`` at the origin."""
    vertices = jnp.asarray(polytope.vertices, dtype=jnp.float64)
    if vertices.ndim != 2:
        raise ValueError("polytope vertices must be a 2D array")
    return jnp.mean(vertices, axis=0)


def symplectic_spectrum_from_covariance(
    covariance: Float[Array, " dimension dimension"],
) -> Float[Array, " n"]:
    """Compute the symplectic eigenvalues of a positive-definite matrix."""
    matrix = jnp.asarray(covariance, dtype=jnp.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix")
    dimension = int(matrix.shape[0])
    if dimension % 2 != 0:
        raise ValueError("symplectic eigenvalues require an even-dimensional matrix")

    # Symmetrise to guard against numerical asymmetry.
    matrix = 0.5 * (matrix + matrix.T)
    symplectic_matrix = _standard_symplectic_matrix(dimension)
    product = -matrix @ symplectic_matrix @ matrix @ symplectic_matrix
    eigenvalues = jnp.real(jnp.linalg.eigvals(product))
    eigenvalues = jnp.clip(eigenvalues, a_min=0.0)
    symplectic_eigenvalues = jnp.sqrt(eigenvalues)

    n = dimension // 2
    sorted_values = jnp.sort(symplectic_eigenvalues)
    return sorted_values[-n:]


def radial_profile_distance(
    polytope_a: Polytope,
    polytope_b: Polytope,
    *,
    directions: Float[Array, " num_directions dimension"],
    softness: float,
    epsilon: float,
) -> float:
    """Compare two polytopes via softened radial profiles."""
    cache_a = build_radial_profile_cache(
        polytope_a,
        directions=directions,
        softness=softness,
        epsilon=epsilon,
    )
    cache_b = build_radial_profile_cache(
        polytope_b,
        directions=directions,
        softness=softness,
        epsilon=epsilon,
    )
    return radial_profile_distance_cached(cache_a, cache_b)


def build_radial_profile_cache(
    polytope: Polytope,
    *,
    directions: Float[Array, " num_directions dimension"],
    softness: float,
    epsilon: float,
) -> RadialProfileCache:
    """Construct a reusable radial profile cache for ``polytope``."""
    directions = jnp.asarray(directions, dtype=jnp.float64)
    translation = centre_polytope_vertices(polytope)
    centred_offsets = _centre_offsets(polytope, translation)
    log_radii = _compute_log_radii(
        normals=jnp.asarray(polytope.normals, dtype=jnp.float64),
        offsets=centred_offsets,
        directions=directions,
        softness=softness,
        epsilon=epsilon,
    )
    return RadialProfileCache(
        log_radii=log_radii,
        directions=directions,
        softness=float(softness),
        epsilon=float(epsilon),
    )


def radial_profile_distance_cached(
    cache_a: RadialProfileCache,
    cache_b: RadialProfileCache,
) -> float:
    """Compare two polytopes using cached radial profiles."""
    if cache_a.log_radii.shape != cache_b.log_radii.shape:
        raise ValueError("radial caches must share the same directions")
    difference = jnp.abs(cache_a.log_radii - cache_b.log_radii)
    return float(jnp.mean(difference))


def symplectic_correlation_distance(
    polytope_a: Polytope,
    polytope_b: Polytope,
    *,
    num_pairs: int,
    histogram_edges: Float[Array, " num_bins_plus_one"],
    seed: int | None,
) -> float:
    """Estimate the Wasserstein-1 distance between correlation laws."""
    sketch_a = build_correlation_sketch(
        polytope_a,
        num_pairs=num_pairs,
        histogram_edges=histogram_edges,
        seed=seed,
    )
    sketch_b = build_correlation_sketch(
        polytope_b,
        num_pairs=num_pairs,
        histogram_edges=histogram_edges,
        seed=seed,
    )
    return symplectic_correlation_distance_cached(sketch_a, sketch_b)


def build_correlation_sketch(
    polytope: Polytope,
    *,
    num_pairs: int,
    histogram_edges: Float[Array, " num_bins_plus_one"],
    seed: int | None,
) -> CorrelationSketch:
    """Return the cached symplectic correlation sketch for ``polytope``."""
    samples = jnp.sort(
        _symplectic_correlation_samples(
            polytope=polytope,
            num_pairs=num_pairs,
            seed=seed,
        )
    )
    histogram, _ = jnp.histogram(samples, bins=histogram_edges)
    return CorrelationSketch(
        samples=samples,
        histogram=histogram,
        bin_edges=jnp.asarray(histogram_edges, dtype=jnp.float64),
        num_pairs=int(num_pairs),
    )


def symplectic_correlation_distance_cached(
    sketch_a: CorrelationSketch,
    sketch_b: CorrelationSketch,
) -> float:
    """Compare correlation sketches via Wasserstein-1 distance."""
    if sketch_a.samples.shape != sketch_b.samples.shape:
        raise ValueError("correlation sketches must share the same sample count")
    return float(jnp.mean(jnp.abs(sketch_a.samples - sketch_b.samples)))


def symplectic_jaccard_distance(
    polytope_a: Polytope,
    polytope_b: Polytope,
    *,
    num_samples: int,
    num_restarts: int,
    num_iterations: int,
    search_learning_rate: float,
    seed: int | None,
) -> float:
    """Estimate the symplectic Jaccard distance between two polytopes."""
    del num_restarts, num_iterations, search_learning_rate
    cache_a = build_jaccard_cache(polytope_a, num_samples=num_samples, seed=seed)
    cache_b = build_jaccard_cache(polytope_b, num_samples=num_samples, seed=seed)
    return symplectic_jaccard_distance_cached(
        cache_a,
        cache_b,
        num_restarts=1,
        num_iterations=1,
        search_learning_rate=1.0,
    )


def build_jaccard_cache(
    polytope: Polytope,
    *,
    num_samples: int,
    seed: int | None,
) -> JaccardMonteCarloCache:
    """Prepare Monte-Carlo artefacts for Jaccard comparisons."""
    if num_samples <= 0:
        raise ValueError("number of samples must be positive")
    normals = jnp.asarray(polytope.normals, dtype=jnp.float64)
    translation = centre_polytope_vertices(polytope)
    centred_offsets = _centre_offsets(polytope, translation)
    centred_vertices = jnp.asarray(polytope.vertices, dtype=jnp.float64) - translation
    bounding_min = jnp.min(centred_vertices, axis=0)
    bounding_max = jnp.max(centred_vertices, axis=0)
    return JaccardMonteCarloCache(
        centred_normals=normals,
        centred_offsets=centred_offsets,
        bounding_min=bounding_min,
        bounding_max=bounding_max,
        num_samples=int(num_samples),
        seed=seed,
    )


def symplectic_jaccard_distance_cached(
    cache_a: JaccardMonteCarloCache,
    cache_b: JaccardMonteCarloCache,
    *,
    num_restarts: int,
    num_iterations: int,
    search_learning_rate: float,
) -> float:
    """Estimate the Jaccard distance using cached Monte-Carlo artefacts."""
    del num_restarts, num_iterations, search_learning_rate
    if cache_a.num_samples != cache_b.num_samples:
        raise ValueError("cached Jaccard artefacts must share the same sample budget")

    dimension = cache_a.bounding_min.shape[0]
    if cache_b.bounding_min.shape[0] != dimension:
        raise ValueError("polytopes must share the same ambient dimension")

    key = jax.random.PRNGKey(_combine_seeds(cache_a.seed, cache_b.seed))
    base_samples = jax.random.uniform(
        key, (cache_a.num_samples, dimension), dtype=jnp.float64
    )

    union_min = jnp.minimum(cache_a.bounding_min, cache_b.bounding_min)
    union_max = jnp.maximum(cache_a.bounding_max, cache_b.bounding_max)
    extent = jnp.maximum(union_max - union_min, 0.0)
    points = union_min + base_samples * extent

    inside_a = _points_inside_polytope(
        points, cache_a.centred_normals, cache_a.centred_offsets
    )
    inside_b = _points_inside_polytope(
        points, cache_b.centred_normals, cache_b.centred_offsets
    )

    intersection = jnp.sum(jnp.logical_and(inside_a, inside_b))
    union = jnp.sum(jnp.logical_or(inside_a, inside_b))
    intersection = intersection.astype(jnp.float64)
    union = union.astype(jnp.float64)
    distance = jnp.where(union > 0.0, 1.0 - intersection / union, 0.0)
    return float(distance)


def staged_symplectic_similarity(
    spectrum_a: SymplecticSpectrumCache,
    spectrum_b: SymplecticSpectrumCache,
    radial_a: RadialProfileCache,
    radial_b: RadialProfileCache,
    correlation_a: CorrelationSketch,
    correlation_b: CorrelationSketch,
    *,
    weights: Float[Array, " 3"],
    near_threshold: float,
    far_threshold: float,
) -> float:
    """Combine cached invariants into a staged similarity estimate."""
    weights = jnp.asarray(weights, dtype=jnp.float64)
    if weights.shape != (3,):
        raise ValueError("weights must be a length-3 array")
    normalised_weights = weights / jnp.sum(weights)

    spec_a = jnp.log(jnp.asarray(spectrum_a.symplectic_eigenvalues, dtype=jnp.float64))
    spec_b = jnp.log(jnp.asarray(spectrum_b.symplectic_eigenvalues, dtype=jnp.float64))
    spec_distance = jnp.mean(jnp.abs(spec_a - spec_b))

    radial_distance = radial_profile_distance_cached(radial_a, radial_b)

    samples_a = jnp.sort(jnp.asarray(correlation_a.samples, dtype=jnp.float64))
    samples_b = jnp.sort(jnp.asarray(correlation_b.samples, dtype=jnp.float64))
    if samples_a.shape != samples_b.shape:
        raise ValueError("correlation sketches must share the same sample count")
    correlation_distance = float(jnp.mean(jnp.abs(samples_a - samples_b)))

    combined = (
        float(normalised_weights[0]) * float(spec_distance)
        + float(normalised_weights[1]) * float(radial_distance)
        + float(normalised_weights[2]) * float(correlation_distance)
    )

    if combined <= near_threshold:
        return 0.0
    if combined >= far_threshold:
        return 1.0
    scale = far_threshold - near_threshold
    if scale <= 0.0:
        return combined
    return (combined - near_threshold) / scale


def _standard_symplectic_matrix(dimension: int) -> Float[Array, " dimension dimension"]:
    if dimension % 2 != 0:
        raise ValueError("symplectic matrix requires even dimension")
    half = dimension // 2
    zero = jnp.zeros((half, half), dtype=jnp.float64)
    identity = jnp.eye(half, dtype=jnp.float64)
    top = jnp.concatenate([zero, identity], axis=1)
    bottom = jnp.concatenate([-identity, zero], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def _centre_offsets(
    polytope: Polytope, translation: Float[Array, " dimension"]
) -> Float[Array, " num_facets"]:
    normals = jnp.asarray(polytope.normals, dtype=jnp.float64)
    offsets = jnp.asarray(polytope.offsets, dtype=jnp.float64)
    return offsets - normals @ translation


def _compute_log_radii(
    *,
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    directions: Float[Array, " num_directions dimension"],
    softness: float,
    epsilon: float,
) -> Float[Array, " num_directions"]:
    if softness <= 0.0:
        raise ValueError("softness must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    projection_matrix = normals @ directions.T
    valid_facets = projection_matrix > epsilon
    if not bool(jnp.all(jnp.any(valid_facets, axis=0))):
        raise ValueError("every direction must intersect the polytope")

    def evaluate(direction: Float[Array, " dimension"]) -> Float[Array, ""]:
        projections = normals @ direction
        mask = projections > epsilon
        ratios = jnp.where(mask, offsets / projections, jnp.inf)
        min_val = jnp.min(ratios)
        shifted = jnp.where(mask, ratios - min_val, jnp.inf)
        weights = jnp.where(mask, jnp.exp(-shifted / softness), 0.0)
        weights_sum = jnp.maximum(jnp.sum(weights), jnp.finfo(jnp.float64).tiny)
        soft_min = min_val - softness * jnp.log(weights_sum)
        return jnp.log(soft_min)

    vectorised: Callable[[Float[Array, " num_directions dimension"]], Float[Array, " num_directions"]] = jax.vmap(evaluate)
    return vectorised(directions)


def _symplectic_correlation_samples(
    *,
    polytope: Polytope,
    num_pairs: int,
    seed: int | None,
) -> Float[Array, " num_pairs"]:
    if num_pairs <= 0:
        raise ValueError("number of correlation pairs must be positive")

    vertices = jnp.asarray(polytope.vertices, dtype=jnp.float64)
    if vertices.ndim != 2:
        raise ValueError("polytope vertices must be a 2D array")
    dimension = vertices.shape[1]
    if dimension % 2 != 0:
        raise ValueError("symplectic correlation requires an even-dimensional polytope")

    translation = centre_polytope_vertices(polytope)
    centred_vertices = vertices - translation
    num_vertices = centred_vertices.shape[0]
    if num_vertices == 0:
        raise ValueError("polytope must expose at least one vertex")

    base_seed = 0 if seed is None else int(seed)
    key = jax.random.PRNGKey(base_seed)
    key_y, key_weights = jax.random.split(key, 2)

    weights_x = jax.random.uniform(key_weights, (num_pairs, num_vertices), dtype=jnp.float64)
    weights_x = weights_x / jnp.sum(weights_x, axis=1, keepdims=True)
    samples_x = weights_x @ centred_vertices

    weights_y = jax.random.uniform(key_y, (num_pairs, num_vertices), dtype=jnp.float64)
    weights_y = weights_y / jnp.sum(weights_y, axis=1, keepdims=True)
    samples_y = weights_y @ centred_vertices

    symplectic_matrix = _standard_symplectic_matrix(dimension)
    products = jnp.einsum("bi,ij,bj->b", samples_x, symplectic_matrix, samples_y)
    return products


def _points_inside_polytope(
    points: Float[Array, " num_samples dimension"],
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
) -> Bool[Array, " num_samples"]:
    residuals = points @ normals.T - offsets
    return jnp.all(residuals <= 1e-9, axis=1)


def _combine_seeds(seed_a: int | None, seed_b: int | None) -> int:
    value_a = 0 if seed_a is None else int(seed_a)
    value_b = 0 if seed_b is None else int(seed_b)
    return (value_a ^ value_b) & 0xFFFF_FFFF
