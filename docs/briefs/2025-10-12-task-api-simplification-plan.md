---
status: draft
created: 2025-10-12
workflow: task
summary: Inventory of exported APIs and proposal for simplifying the Viterbo surface.
---

# API simplification opportunity

## Context

The public namespace currently exposes a large surface of loosely-related helpers for polytope
construction, symplectic capacities, similarity metrics, and dataset orchestration. The
interfaces grew organically and now include overlapping entry points (e.g. multiple "fast" vs
"reference" functions per solver, duplicate adapters across geometry layers, bespoke caches, and
wrappers around SciPy/JAX interop). Only a few downstream files consume these APIs, so we can
pursue aggressive consolidation without broad breakage.

### [important] Design directives captured from maintainer feedback

1. Maintain exactly one rich `Polytope` dataclass that captures geometry only (half-spaces,
   vertices, incidence, metadata) and keep symplectic quantities as computed outputs rather than
   stored fields.
2. Math-facing APIs must operate on arrays/tuples directly. The dataclass becomes a convenience for
   orchestration code, but algorithms present pure array signatures with jaxtyping annotations and
   documented tuple outputs.
3. Prefer explicit solver entry points rather than dispatch wrappers. Each algorithm is invoked by a
   distinct function name reflecting its inputs/outputs.
4. Keep `viterbo/_wrapped` as thin compatibility layers that adapt SciPy APIs to JAX arrays; avoid
   layering additional wrappers elsewhere.
5. `typings/` continues to mirror upstream JAX/SciPy typing gaps verbatim; it should not contain
   project-specific adapters or aliases.

These directives reshape the plan below.

## Inventory of functions, classes, and type signatures

The following longlist captures every callable or type defined under `src/viterbo/`.  Signatures are
quoted directly from the implementation modules to establish a baseline before refactoring.

### Core dataclasses and geometry building blocks

- `src/viterbo/types.py`
  - `HalfspaceGeometry(normals: Float[Array, " num_facets dimension"], offsets: Float[Array, " num_facets"])`
  - `VertexGeometry(vertices: Float[Array, " num_vertices dimension"])`
  - `Polytope(normals: Float[Array, " num_facets dimension"], offsets: Float[Array, " num_facets"], vertices: Float[Array, " num_vertices dimension"], incidence: Bool[Array, " num_vertices num_facets"])`
  - `GeneratorMetadata(identifier: str, parameters: dict[str, Any])`
  - `Cycle(points: Float[Array, " num_points dimension"], incidence: Bool[Array, " num_points num_facets"])`

- `src/viterbo/geom.py`
  - `NormalCone`
    - `__post_init__(self) -> None`
  - `PolytopeCombinatorics`
    - `__post_init__(self) -> None`
  - `Polytope`
    - `__post_init__(self) -> None`
    - `dimension(self) -> int`
    - `facets(self) -> int`
    - `halfspace_data(self) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]`
    - `with_metadata(self, name: str | None = None, description: str | None = None) -> "Polytope"`
  - `_halfspace_fingerprint(matrix: Float[Array, " num_facets dimension"], offsets: Float[Array, " num_facets"], decimals: int = 12) -> str`
  - `polytope_fingerprint(polytope: Polytope, decimals: int = 12) -> str`
  - `_tolerance_fingerprint(atol: float) -> str`
  - `polytope_cache_key(polytope: Polytope, atol: float) -> tuple[str, str]`
  - `clear_polytope_cache() -> None`
  - `_cache_enabled(use_cache: bool) -> bool`
  - `_cache_lookup(key: tuple[str, str]) -> PolytopeCombinatorics | None`
  - `_cache_store(key: tuple[str, str], value: PolytopeCombinatorics) -> None`
  - `_build_combinatorics(matrix: Float[Array, " num_facets dimension"], offsets: Float[Array, " num_facets"], vertices: Float[Array, " num_vertices dimension"], atol: float) -> PolytopeCombinatorics`
  - `_validate_halfspace_data(B: Float[Array, " num_facets dimension"], c: Float[Array, " num_facets"]) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]`
  - `_unique_rows(points: Float[Array, " num_points dimension"], atol: float) -> Float[Array, " num_unique dimension"]`
  - `_deduplicate_facets(matrix: Float[Array, " num_facets dimension"], offsets: Float[Array, " num_facets"], atol: float) -> tuple[Float[Array, " num_unique_facets dimension"], Float[Array, " num_unique_facets"]]`
  - `enumerate_vertices(B: Float[Array, " num_facets dimension"], c: Float[Array, " num_facets"], atol: float = 1e-09) -> Float[Array, " num_vertices dimension"]`
  - `remove_redundant_facets(B: Float[Array, " num_facets dimension"], c: Float[Array, " num_facets"], atol: float = 1e-09) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]`
  - `vertices_from_halfspaces(B: Float[Array, " num_facets dimension"], c: Float[Array, " num_facets"], atol: float = 1e-09) -> Float[Array, " num_vertices dimension"]`
  - `halfspaces_from_vertices(vertices: Float[Array, " num_vertices dimension"], qhull_options: str | None = None) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]`
  - `polytope_combinatorics(polytope: Polytope, atol: float = 1e-09, use_cache: bool = True) -> PolytopeCombinatorics`
  - `_volume_of_simplices(simplex_vertices: Float[Array, " num_simplices vertices dimension"]) -> float`
  - `polytope_volume_reference(B: Float[Array, " num_facets dimension"], c: Float[Array, " num_facets"], atol: float = 1e-09) -> float`
  - `polytope_volume_fast(B: Float[Array, " num_facets dimension"], c: Float[Array, " num_facets"], atol: float = 1e-09) -> float`
  - `cartesian_product(first: Polytope, second: Polytope, name: str | None = None, description: str | None = None) -> Polytope`
  - `affine_transform(polytope: Polytope, matrix: Float[Array, " dimension dimension"], translation: Float[Array, " dimension"] | None = None, name: str | None = None, description: str | None = None) -> Polytope`
  - `translate_polytope(polytope: Polytope, translation: Float[Array, " dimension"], name: str | None = None, description: str | None = None) -> Polytope`
  - `mirror_polytope(polytope: Polytope, axes: Sequence[bool], name: str | None = None, description: str | None = None) -> Polytope`
  - `rotate_polytope(polytope: Polytope, plane: tuple[int, int], angle: float, name: str | None = None, description: str | None = None) -> Polytope`
  - `_regular_polygon_normals(sides: int) -> jnp.ndarray`
  - `_rotation_matrix(angle: float) -> jnp.ndarray`
  - `regular_polygon_product(sides_first: int, sides_second: int, rotation: float = 0.0, radius_first: float = 1.0, radius_second: float | None = None, name: str | None = None, description: str | None = None) -> Polytope`
  - `cross_polytope(dimension: int, radius: float = 1.0, name: str | None = None) -> Polytope`
  - `hypercube(dimension: int, radius: float = 1.0, name: str | None = None) -> Polytope`
  - `simplex_with_uniform_weights(dimension: int, last_offset: float | None = None, name: str | None = None) -> Polytope`
  - `truncated_simplex_four_dim() -> Polytope`
  - `viterbo_counterexample(radius: float = 1.0) -> Polytope`
  - `random_affine_map(dimension: int, key: Array, scale_range: tuple[float, float] = (0.6, 1.4), shear_scale: float = 0.25, translation_scale: float = 0.3) -> tuple[Float[Array, " dimension dimension"], Float[Array, " dimension"]]`
  - `random_polytope(dimension: int, key: Array, facets: int | None = None, offset_range: tuple[float, float] = (0.5, 1.5), translation_scale: float = 0.2, name: str | None = None, description: str | None = None, max_attempts: int = 64) -> Polytope`
  - `random_transformations(polytope: Polytope, key: Array, count: int, scale_range: tuple[float, float] = (0.6, 1.4), translation_scale: float = 0.3, shear_scale: float = 0.25) -> list[Polytope]`
  - `catalog() -> tuple[Polytope, ...]`

- `src/viterbo/polytopes.py`
  - `build_from_halfspaces(normals: Float[Array, " num_facets dimension"], offsets: Float[Array, " num_facets"]) -> Polytope`
  - `build_from_vertices(vertices: Float[Array, " num_vertices dimension"]) -> Polytope`
  - `incidence_matrix(normals: Float[Array, " num_facets dimension"], offsets: Float[Array, " num_facets"], vertices: Float[Array, " num_vertices dimension"], rtol: float = INCIDENCE_REL_TOLERANCE, atol: float = INCIDENCE_ABS_TOLERANCE) -> Bool[Array, " num_vertices num_facets"]`
  - `pad_polytope_bundle(bundle: Polytope, target_facets: int, target_vertices: int) -> Polytope`

- `src/viterbo/polytopes_similarity.py`
  - `RadialProfileCache`
  - `CorrelationSketch`
  - `JaccardMonteCarloCache`
  - `SymplecticSpectrumCache`
  - `centre_polytope_vertices(polytope: Polytope) -> Float[Array, " dimension"]`
  - `symplectic_spectrum_from_covariance(covariance: Float[Array, " dimension dimension"]) -> Float[Array, " n"]`
  - `radial_profile_distance(polytope_a: Polytope, polytope_b: Polytope, directions: Float[Array, " num_directions dimension"], softness: float, epsilon: float) -> float`
  - `build_radial_profile_cache(polytope: Polytope, directions: Float[Array, " num_directions dimension"], softness: float, epsilon: float) -> RadialProfileCache`
  - `radial_profile_distance_cached(cache_a: RadialProfileCache, cache_b: RadialProfileCache) -> float`
  - `symplectic_correlation_distance(polytope_a: Polytope, polytope_b: Polytope, num_pairs: int, histogram_edges: Float[Array, " num_bins_plus_one"], seed: int | None) -> float`
  - `build_correlation_sketch(polytope: Polytope, num_pairs: int, histogram_edges: Float[Array, " num_bins_plus_one"], seed: int | None) -> CorrelationSketch`
  - `symplectic_correlation_distance_cached(sketch_a: CorrelationSketch, sketch_b: CorrelationSketch) -> float`
  - `symplectic_jaccard_distance(polytope_a: Polytope, polytope_b: Polytope, num_samples: int, num_restarts: int, num_iterations: int, search_learning_rate: float, seed: int | None) -> float`
  - `build_jaccard_cache(polytope: Polytope, num_samples: int, seed: int | None) -> JaccardMonteCarloCache`
  - `symplectic_jaccard_distance_cached(cache_a: JaccardMonteCarloCache, cache_b: JaccardMonteCarloCache, num_restarts: int, num_iterations: int, search_learning_rate: float) -> float`
  - `staged_symplectic_similarity(spectrum_a: SymplecticSpectrumCache, spectrum_b: SymplecticSpectrumCache, radial_a: RadialProfileCache, radial_b: RadialProfileCache, correlation_a: CorrelationSketch, correlation_b: CorrelationSketch, weights: Float[Array, " 3"], near_threshold: float, far_threshold: float) -> float`
  - `_standard_symplectic_matrix(dimension: int) -> Float[Array, " dimension dimension"]`
  - `_centre_offsets(polytope: Polytope, translation: Float[Array, " dimension"]) -> Float[Array, " num_facets"]`
  - `_compute_log_radii(normals: Float[Array, " num_facets dimension"], offsets: Float[Array, " num_facets"], directions: Float[Array, " num_directions dimension"], softness: float, epsilon: float) -> Float[Array, " num_directions"]`
  - `_symplectic_correlation_samples(polytope: Polytope, num_pairs: int, seed: int | None) -> Float[Array, " num_pairs"]`
  - `_points_inside_polytope(points: Float[Array, " num_samples dimension"], normals: Float[Array, " num_facets dimension"], offsets: Float[Array, " num_facets"]) -> Bool[Array, " num_samples"]`
  - `_combine_seeds(seed_a: int | None, seed_b: int | None) -> int`

### Capacity computations and billiards

- `src/viterbo/capacity/__init__.py`
  - `available_solvers() -> tuple[str, ...]`
  - `ehz_capacity_reference(bundle: Polytope, *, tol: float = FACET_SOLVER_TOLERANCE) -> float`
  - `ehz_capacity_fast(bundle: Polytope, *, tol: float = FACET_SOLVER_TOLERANCE) -> float`
  - Aliased exports: `ehz_capacity_reference_milp`, `ehz_capacity_fast_milp`

- `src/viterbo/capacity/facet_normals.py`
  - `_FacetSubset`
  - `support_radii(bundle: Polytope) -> Float[Array, " num_facets"]`
  - `_bundle_arrays(bundle: Polytope | tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]`
  - `_iter_index_combinations(count: int, size: int) -> Iterator[tuple[int, ...]]`
  - `_prepare_subset(B_matrix: Float[Array, " num_facets dimension"], c: Float[Array, " num_facets"], indices: Sequence[int], J: Float[Array, " dimension dimension"], tol: float) -> _FacetSubset | None`
  - `_maximum_antisymmetric_order_value(weights: Array) -> float`
  - `_subset_capacity_candidate_dynamic(subset: _FacetSubset, tol: float) -> float | None`
  - `_subset_capacity_candidate(subset: _FacetSubset, tol: float) -> float | None`
  - `_compute_ehz_capacity_reference(B_matrix: Float[Array, " num_facets dimension"], c: Float[Array, " num_facets"], tol: float) -> float`
  - `_compute_ehz_capacity_fast(B_matrix: Float[Array, " num_facets dimension"], c: Float[Array, " num_facets"], tol: float) -> float`
  - `ehz_capacity_reference_facet_normals(bundle: Polytope | tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]], tol: float = FACET_SOLVER_TOLERANCE) -> float`
  - `ehz_capacity_fast_facet_normals(bundle: Polytope | tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]], tol: float = FACET_SOLVER_TOLERANCE) -> float`

- `src/viterbo/capacity/milp.py`
  - `MilpCapacityResult`
  - `_capacity_upper_bound(bundle: Polytope) -> float`
  - `ehz_capacity_reference_milp(bundle: Polytope, max_nodes: int = 1024) -> MilpCapacityResult`
  - `ehz_capacity_fast_milp(bundle: Polytope, node_limit: int = 256) -> MilpCapacityResult`

- `src/viterbo/capacity/minkowski_billiards.py`
  - `MinkowskiNormalFan`
    - `dimension(self) -> int`
    - `vertex_count(self) -> int`
  - `_to_geometry_polytope(bundle: Polytope, name: str) -> _GeometryPolytope`
  - `build_normal_fan(bundle: Polytope, atol: float = GEOMETRY_ABS_TOLERANCE) -> MinkowskiNormalFan`
  - `_vertex_adjacency(cones: Sequence[Any], dimension: int) -> Array`
  - `_coordinate_blocks(matrix: Float[Array, " num_facets dimension"], tol: float) -> tuple[tuple[int, ...], ...]`
  - `_support_function(vertices: Float[Array, " num_vertices dimension"], direction: Float[Array, " dimension"]) -> float`
  - `minkowski_billiard_length_reference(table: Polytope, geometry: Polytope, max_bounces: int | None = None, atol: float = GEOMETRY_ABS_TOLERANCE) -> float`
  - `minkowski_billiard_length_fast(table: Polytope, geometry: Polytope, max_bounces: int | None = None, atol: float = GEOMETRY_ABS_TOLERANCE) -> float`
  - `_enumerate_cycles(fan: MinkowskiNormalFan, max_length: int) -> Iterable[tuple[int, ...]]`
  - `_component_containing(start: int, neighbor_lists: list[list[int]], min_vertex: int) -> set[int] | None`
  - `_strongly_connected_components(neighbor_lists: list[list[int]], min_vertex: int) -> list[set[int]]`
  - `_canonical_cycle(sequence: tuple[int, ...]) -> tuple[int, ...]`
  - `_cycle_length(cycle: tuple[int, ...], vertices: Float[Array, " num_vertices dimension"], geometry_vertices: Float[Array, " num_vertices_t dimension"]) -> float`
  - `_pairwise_lengths(fan: MinkowskiNormalFan, geometry_vertices: Float[Array, " num_vertices_t dimension"]) -> tuple[list[list[float]], float]`
  - `_dfs_prefix(start: int, current: int, visited_mask: int, steps_remaining: int, current_length: float, neighbors: tuple[tuple[int, ...], ...], length_matrix: list[list[float]], min_edge_length: float, completion_bounds: list[list[float]], best_overall_ref: list[float], prefix_best: dict[tuple[int, int, int, int], float]) -> None`
  - `_completion_bounds(start: int, neighbors: tuple[tuple[int, ...], ...], length_matrix: list[list[float]], max_length: int) -> list[list[float]]`
  - `_pair_to_polytope(polytope: _GeometryPolytope, blocks: tuple[tuple[int, ...], ...]) -> list[_GeometryPolytope] | None`
  - `_try_product_decomposition(billiard_table: _GeometryPolytope, geometry: _GeometryPolytope, max_bounces: int | None, atol: float) -> float | None`

- `src/viterbo/capacity/reeb_cycles.py`
  - `OrientedEdge`
  - `OrientedEdgeGraph`
    - `successors(self, edge_id: int) -> tuple[int, ...]`
    - `predecessors(self, edge_id: int) -> tuple[int, ...]`
    - `edge_count(self) -> int`
  - `OrientedEdgeDiagnostics`
  - `_to_geometry_polytope(bundle: Polytope) -> _GeometryPolytope`
  - `build_oriented_edge_graph(bundle: Polytope, atol: float = GEOMETRY_ABS_TOLERANCE) -> OrientedEdgeGraph`
  - `_vertex_key(vertex: Float[Array, " dimension"], atol: float) -> tuple[int, ...]`
  - `_graph_diagnostics(graph: OrientedEdgeGraph) -> OrientedEdgeDiagnostics`
  - `ehz_capacity_reference_reeb(bundle: Polytope, atol: float = GEOMETRY_ABS_TOLERANCE, tol: float = FACET_SOLVER_TOLERANCE) -> float`
  - `ehz_capacity_fast_reeb(bundle: Polytope, atol: float = GEOMETRY_ABS_TOLERANCE, tol: float = FACET_SOLVER_TOLERANCE) -> float`

- `src/viterbo/capacity/support_relaxation.py`
  - `SupportRelaxationDiagnostics`
  - `SupportRelaxationResult`
  - `_polygon_area(vertices: Float[Array, " num_vertices 2"]) -> float`
  - `_sample_directions(dimension: int, count: int) -> Float[Array, " count dimension"]`
  - `_support_values(bundle: Polytope, directions: Float[Array, " num_samples dimension"]) -> Float[Array, " num_samples"]`
  - `support_relaxation_capacity_reference(bundle: Polytope, grid_density: int = 12, smoothing_parameters: Sequence[float] = (0.6, 0.3, 0.1), tolerance_sequence: Sequence[float] = (0.001,), solver: str | None = None, center_vertices: bool = True) -> SupportRelaxationResult`
  - `support_relaxation_capacity_fast(bundle: Polytope, initial_density: int = 6, refinement_steps: int = 1, smoothing_parameters: Sequence[float] = (0.5, 0.25), jit_compile: bool = True) -> SupportRelaxationResult`

- `src/viterbo/capacity/symmetry_reduced.py`
  - `FacetPairingMetadata`
  - `detect_opposite_facet_pairs(bundle: Polytope, angle_tolerance: float = 1e-06) -> FacetPairingMetadata`
  - `_reduced_radii(bundle: Polytope, pairing: FacetPairingMetadata | None) -> jnp.ndarray`
  - `ehz_capacity_reference_symmetry_reduced(bundle: Polytope, pairing: FacetPairingMetadata | None = None) -> float`
  - `ehz_capacity_fast_symmetry_reduced(bundle: Polytope, pairing: FacetPairingMetadata | None = None) -> float`

- `src/viterbo/cycles.py`
  - `minimum_cycle_reference(bundle: Polytope) -> Float[Array, " num_points dimension"]`

### Symplectic and spectrum utilities

- `src/viterbo/symplectic.py`
  - `standard_symplectic_matrix(dimension: int) -> Float[Array, " dim dim"]`
  - `random_symplectic_matrix(key: jax.Array, dimension: int, scale: float = 0.1) -> Float[Array, " dim dim"]`
  - `symplectic_product(first: Float[Array, " dim"], second: Float[Array, " dim"], matrix: Float[Array, " dim dim"] | None = None) -> float`
  - `support_function(vertices: Float[Array, " num_vertices dim"], direction: Float[Array, " dim"]) -> float`
  - `minkowski_sum(first_vertices: Float[Array, " m dim"], second_vertices: Float[Array, " n dim"]) -> Float[Array, " k dim"]`
  - `normalize_vector(vector: Float[Array, " dim"]) -> Float[Array, " dim"]`

- `src/viterbo/spectrum.py`
  - `ehz_spectrum_reference(bundle: Polytope, head: int, atol: float = GEOMETRY_ABS_TOLERANCE) -> Sequence[float]`
  - `_to_geometry_polytope(bundle: Polytope) -> _GeometryPolytope`
  - `_enumerate_simple_cycles(graph: OrientedEdgeGraph, limit: int) -> Iterable[tuple[int, ...]]`
  - `_cycle_action(cycle: tuple[int, ...], graph: OrientedEdgeGraph, vertices: Float[Array, " num_vertices dimension"]) -> float`

- `src/viterbo/systolic.py`
  - `systolic_ratio(polytope: Polytope | GeometryPolytope, /) -> float`
  - `systolic_ratio(B_matrix: Float[Array, " num_facets dimension"], c: Float[Array, " num_facets"], /) -> float`
  - `systolic_ratio(arg: Polytope | GeometryPolytope | Float[Array, " num_facets dimension"], c: Float[Array, " num_facets"] | None = None) -> float`

### Volume adapters

- `src/viterbo/volume.py`
  - `volume_reference(bundle: Polytope) -> float`
  - `volume_padded(normals: Float[Array, " batch num_facets dimension"], offsets: Float[Array, " batch num_facets"], method: str) -> Float[Array, " batch"]`

### Dataset helpers and generators

- `src/viterbo/atlas.py`
  - `atlas_features() -> Features`
  - `build_dataset(rows_iterable: Iterable[Mapping[str, object]]) -> Dataset`
  - `append_rows(dataset: Dataset, rows_iterable: Iterable[Mapping[str, object]]) -> Dataset`
  - `save_dataset(dataset: Dataset, path: str) -> None`
  - `load_dataset(path: str) -> Dataset`
  - `map_quantities(dataset: Dataset, fn: Callable[[Mapping[str, object]], Mapping[str, object]]) -> Dataset`
  - `AtlasRow`
    - `as_polytope(dimension: int, num_facets: int, num_vertices: int, normals: list[Float[Array, " dimension"]], offsets: list[float], vertices: list[Float[Array, " dimension"]]) -> Polytope`
    - `as_cycle(dimension: int, num_points: int, points: list[Float[Array, " dimension"]], polytope: Polytope) -> Cycle`

- `src/viterbo/basic_generators.py`
  - `sample_halfspace(key: PRNGKeyArray, dimension: int, num_facets: int, num_samples: int) -> list[Polytope]`
  - `sample_halfspace_tangent(key: PRNGKeyArray, dimension: int, num_facets: int, num_samples: int) -> list[Polytope]`
  - `sample_uniform_sphere(key: PRNGKeyArray, dimension: int, num_samples: int) -> list[Polytope]`
  - `sample_uniform_ball(key: PRNGKeyArray, dimension: int, num_samples: int) -> list[Polytope]`
  - `enumerate_product_ngons(max_ngon_P: int, max_ngon_Q: int, max_rotation_Q: int) -> list[Polytope]`

### Wrapped SciPy/Numpy adapters

- `src/viterbo/_wrapped/cvx.py`
  - `_load_cvxpy() -> Any`
  - `solve_epigraph_minimum(values: Any, solver: str, tolerance: float) -> float`

- `src/viterbo/_wrapped/highs.py`
  - `HighsResources`
  - `load_highs() -> HighsResources`

- `src/viterbo/_wrapped/linalg.py`
  - `expm(matrix: Any) -> _npt.NDArray[_np.float64]`

- `src/viterbo/_wrapped/numpy_bytes.py`
  - `fingerprint_halfspace(matrix: object, offsets: object, decimals: int = 12) -> str`

- `src/viterbo/_wrapped/optimize.py`
  - `_OptimizeResultProtocol`
  - `linprog(c: Any, A_ub: Any | None, b_ub: Any | None, A_eq: Any | None, b_eq: Any | None, bounds: Sequence[BoundTuple] | None, **options) -> _OptimizeResultProtocol`

- `src/viterbo/_wrapped/spatial.py`
  - `convex_hull_volume(points: Any, qhull_options: str | None = 'QJ') -> float`
  - `convex_hull_equations(points: Any, qhull_options: str | None = None) -> _np.ndarray`
  - `convex_hull_vertices(points: Any, qhull_options: str | None = None) -> _np.ndarray`
  - `delaunay_simplices(points: Any, qhull_options: str | None = 'QJ') -> _np.ndarray`
  - `halfspace_intersection_vertices(B: Any, c: Any, atol: float = 1e-12) -> _np.ndarray`

### Experiments and training utilities

- `src/viterbo/experiments/logreg_toy.py`
  - `LogisticRegressionConfig`
  - `TrainingTrace(NamedTuple)`
  - `LogisticRegressionWeights(NamedTuple)`
  - `train_logistic_regression(features: Float[Array, " batch features"], labels: Float[Array, " batch"], config: LogisticRegressionConfig, key: jax.Array) -> tuple[LogisticRegressionWeights, TrainingTrace]`
  - `_predict_logits(params: LogisticRegressionWeights, features: Float[Array, " batch features"]) -> Float[Array, " batch"]`
  - `evaluate(params: LogisticRegressionWeights, features: Float[Array, " batch features"], labels: Float[Array, " batch"], l2_penalty: float = 0.0) -> dict[str, float]`

### Volume, spectrum, and cycle exports via `src/viterbo/__init__.py`

- Package `__init__` currently re-exports nothing itself.

## Proposed canonical `Polytope` dataclass for `src/viterbo/types.py`

The maintained dataclass should remain the single source of truth for geometric metadata. It should
be lightweight (slots, no symplectic payloads) and expose optional provenance fields so downstream
code can attach generator metadata without creating parallel structures.

```python
@dataclass(slots=True)
class Polytope:
    """Rich geometric description of a convex polytope."""

    normals: Float[Array, " num_facets dimension"]
    offsets: Float[Array, " num_facets"]
    vertices: Float[Array, " num_vertices dimension"]
    incidence: Bool[Array, " num_vertices num_facets"]
    name: str | None = None
    description: str | None = None
    provenance: GeneratorMetadata | None = None
```

Companion constructors (planned for `polytopes.py`) will produce this dataclass from arrays, but
core algorithms will accept `(normals, offsets[, vertices[, incidence]])` tuples directly.

## Simplification and consolidation plan

1. **Consolidate geometry types**
   - Update `types.py` to match the dataclass above, delete redundant geometry dataclasses, and move
     optional metadata into the single `Polytope`. Provide `HalfspaceGeometry`/`VertexGeometry`
     aliases only when strictly needed for typing helper functions.
   - Refactor `geom.py` and `polytopes.py` constructors to emit array tuples and expose pure
     functions such as `compute_incidence(normals, offsets, vertices) -> Bool[...]`. Keep
     `Polytope` creation as a thin orchestration layer that simply packages those arrays.

2. **Reframe capacity solvers around array-first APIs**
   - For every solver module, define explicit entry points like
     `ehz_capacity_facet_normals(normals, offsets, *, tol) -> float` and
     `ehz_capacity_facet_normals_with_cycle(normals, offsets, *, tol) -> tuple[float, Cycle]` when
     extra outputs exist. Remove dispatch wrappers (`ehz_capacity_fast`, `_to_geometry_polytope`,
     etc.) and let orchestration code convert dataclasses to arrays.
   - Update supporting helpers (`support_radii`, `_coordinate_blocks`, etc.) to accept raw arrays to
     eliminate repeated `bundle` unpacking logic.

3. **Simplify similarity and spectrum tooling**
   - Replace cached-object classes with functions returning tuples of arrays, e.g.
     `build_radial_profile(normals, offsets, directions, softness, epsilon) -> tuple[...]` and make
     cached variants thin wrappers in calling code rather than exported functions.
   - Expose separate top-level metrics (`radial_profile_distance`, `symplectic_correlation_distance`,
     `symplectic_jaccard_distance`) that all accept arrays. Dataset/CLI layers can compose these for
     staged evaluation without additional wrappers.

4. **Clarify spectrum, cycles, and volume utilities**
   - Ensure `spectrum.py`, `cycles.py`, and `volume.py` operate on array tuples, leaving dataclass use
     to CLI/test fixtures. Remove multi-signature wrappers like the current overloaded
     `systolic_ratio`; instead publish explicit functions per signature
     (`systolic_ratio_from_halfspaces`, `systolic_ratio_from_geometry`).
   - Normalize return types so functions that produce auxiliary data return tuples with documented
     elements (cycle, diagnostics, etc.).

5. **Data and generator alignment**
   - Collapse `atlas.py` and `basic_generators.py` around a registry that returns arrays and optional
     metadata, leaving `Polytope` instantiation to dataset writers. Replace `AtlasRow.as_polytope`
     with pure functions returning array tuples.
   - Introduce small utility functions that convert between dataset rows and array bundles to keep
     serialization isolated from math code.

6. **Interop and typing boundaries**
   - Keep `_wrapped` modules as the only SciPy/NumPy touchpoints. Document each wrapper as "array in,
     array out" without additional dispatch layers.
   - Audit `typings/` to ensure it contains only upstream type shims; any project-specific typing
     helpers move next to the modules they serve.
   - Co-locate `ehz_spectrum_reference`, `minimum_cycle_reference`, and `systolic_ratio` routines
     under a clearly named `spectral.py` module while ensuring each continues to return tuples of
     arrays/scalars per the math-API contract.

## Next steps

- Draft a concrete refactoring tree that (a) updates `types.py`/`polytopes.py` constructors to the
  proposed dataclass contract, (b) ports capacity/volume/spectrum algorithms to array-first
  signatures, and (c) revises dataset + similarity tooling to consume the new helpers.
- Create migration notes for the small number of downstream call sites once the new API surface is
  finalised, including guidance on choosing explicit solver entry points.
