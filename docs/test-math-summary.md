# Test Suite Mathematical Summary

This document captures every test under `tests/` and classifies it as either **Math-level correctness** (establishes or
regresses mathematical statements about the algorithms), **Code-level correctness** (guards implementation details or
API/validation contracts without probing new mathematical facts), or **Performance**.

## Geometry — Volume (`tests/viterbo/geometry/volume/test_volume.py`)
- **test_hypercube_volume_matches_closed_form** — Confirms that both volume estimators agree with the analytic
  \((2r)^4\) volume of a 4-D cube of radius 1.5. Classification: Math-level correctness.
- **test_random_polytope_volumes_agree** — Uses a random 4-D polytope to check the reference and fast estimators match,
  validating numerical stability. Classification: Math-level correctness.
- **test_simplex_volume_positive** — Ensures the computed volume of a weighted simplex is strictly positive, catching
  orientation sign errors. Classification: Math-level correctness.
- **test_hypercube_samples_match_expected_volume** — Verifies that helper-generated sample matrices yield the closed-form
  3-D cube volume for both estimators. Classification: Math-level correctness.

## Geometry — Polytopes (Transforms) (`tests/viterbo/geometry/polytopes/test_transforms.py`)
- **test_translate_polytope_updates_offsets** — Checks that translating a polytope updates the offset vector via
  \(c \mapsto c + Bx\). Classification: Math-level correctness.
- **test_cartesian_product_dimensions_add** — Verifies that the Cartesian product polytope has additive dimension and
  facet counts. Classification: Math-level correctness.
- **test_mirror_polytope_flips_coordinate** — Confirms mirroring negates the specified coordinate in the halfspace
  matrix. Classification: Math-level correctness.
- **test_rotate_polytope_consistency_in_plane** — Ensures rotations conjugate the halfspace matrix by the planar rotation
  matrix, preserving constraints. Classification: Math-level correctness.
- **test_random_affine_map_is_deterministic_per_seed** — Validates PRNG determinism and conditioning of the sampled
  affine maps; focuses on reproducibility. Classification: Code-level correctness.
- **test_random_polytope_facets_are_active** — Confirms randomly generated polytopes keep all facets active (each touches
  a vertex), preventing degenerate sampling. Classification: Math-level correctness.

## Geometry — Polytopes (Combinatorics) (`tests/viterbo/geometry/polytopes/test_combinatorics.py`)
- **test_polytope_combinatorics_square_facets** — Checks combinatorial metadata for a square: four vertices with facet
  degree two. Classification: Math-level correctness.
- **test_polytope_combinatorics_properties** — Ensures a cube’s combinatorics report the correct adjacency degrees.
  Classification: Math-level correctness.
- **test_vertex_enumeration_matches_reference_shape** — Confirms vertex enumeration from halfspaces returns the expected
  cube vertices. Classification: Math-level correctness.
- **test_polytope_combinatorics_cached_instances_are_reused** — Ensures caching reuses instances to avoid recomputation.
  Classification: Code-level correctness.
- **test_halfspace_vertex_roundtrip** — Validates that converting between vertices and halfspaces is round-trip
  consistent. Classification: Math-level correctness.
- **test_polytope_fingerprint_invariant_to_metadata** — Ensures fingerprints ignore metadata changes, focusing on object
  identity semantics. Classification: Code-level correctness.

## Geometry — Haim–Kislev Action (`tests/viterbo/geometry/polytopes/test_haim_kislev_action.py`)
- **test_haim_kislev_action_valid_order_matches_reference_capacity** — Checks the Haim–Kislev action matches the known
  EHZ capacity for the truncated simplex when using a valid subset ordering. Classification: Math-level correctness.
- **test_haim_kislev_action_invalid_order_raises_value_error** — Ensures invalid permutations trigger input validation.
  Classification: Code-level correctness.

## Geometry — Halfspaces (`tests/viterbo/geometry/halfspaces/test_halfspaces.py`)
- **test_reference_enumeration_matches_expected_square** — Confirms the reference vertex enumerator produces the square’s
  four vertices. Classification: Math-level correctness.
- **test_fast_enumerator_matches_reference** — Verifies the fast vertex enumerator agrees with the reference results.
  Classification: Math-level correctness.
- **test_remove_redundant_facets_discards_duplicates** — Ensures both redundant-facet removers shrink duplicated rows to
  the canonical four facets. Classification: Math-level correctness.

## Symplectic Core (`tests/viterbo/symplectic/test_core.py`)
- **test_standard_symplectic_matrix_structure** — Validates the explicit \(J\) matrix form. Classification:
  Math-level correctness.
- **test_standard_symplectic_matrix_requires_even_dimension** — Ensures odd dimensions raise errors; this protects the
  mathematical precondition. Classification: Code-level correctness.
- **test_symplectic_product_default_matrix** — Checks canonical basis vectors yield expected symplectic pairings.
  Classification: Math-level correctness.
- **test_symplectic_product_custom_matrix** — Confirms custom antisymmetric matrices scale the product appropriately.
  Classification: Math-level correctness.
- **test_support_function_simplex** — Ensures the simplex support function attains the expected directional value.
  Classification: Math-level correctness.
- **test_minkowski_sum_pairwise_vertices** — Verifies Minkowski addition enumerates all pairwise vertex sums.
  Classification: Math-level correctness.
- **test_support_function_validates_inputs** — Confirms invalid shapes trigger validation errors. Classification:
  Code-level correctness.
- **test_minkowski_sum_validates_inputs** — Ensures shape and dimensionality validation for Minkowski sums.
  Classification: Code-level correctness.
- **test_normalize_vector_unit_length** — Checks normalization yields unit norm, catching scaling bugs. Classification:
  Math-level correctness.
- **test_normalize_vector_zero_vector_raises** — Ensures zero vectors are rejected. Classification: Code-level correctness.
- **test_normalize_vector_accepts_list_input** — Confirms list inputs are accepted to smooth API ergonomics.
  Classification: Code-level correctness.
- **test_zero_tolerance_reasonable** — Ensures the global tolerance constant stays within a numerically safe range.
  Classification: Code-level correctness.

## Symplectic Capacity — Reference Implementation (`tests/viterbo/symplectic/capacity/facet_normals/test_reference.py`)
- **test_capacity_matches_baseline** — Cross-checks reference capacities against catalogued baselines for multiple
  polytopes. Classification: Math-level correctness.
- **test_capacity_scales_quadratically_under_dilation** — Verifies \(c_{EHZ}\) scales with the square of dilation.
  Classification: Math-level correctness.
- **test_capacity_is_translation_invariant** — Confirms translational invariance of the capacity. Classification:
  Math-level correctness.
- **test_truncated_simplex_matches_known_subset_action** — Ensures the truncated simplex’s capacity matches its known
  optimal action. Classification: Math-level correctness.
- **test_two_dimensional_simplex_matches_fast_capacity** — Cross-validates the reference and fast implementations on a
  2-D simplex, ensuring finiteness. Classification: Math-level correctness.

## Symplectic Capacity — Optimized Implementation (`tests/viterbo/symplectic/capacity/facet_normals/test_fast.py`)
- **test_dynamic_program_matches_bruteforce** — Compares the dynamic program’s antisymmetric order value against a brute
  force enumeration. Classification: Math-level correctness.
- **test_fast_implementation_matches_reference** — Parameterized regression ensuring the optimized kernel matches the
  reference (including shared failure cases). Classification: Math-level correctness.

## Symplectic Capacity — Algorithmic Invariants (`tests/viterbo/symplectic/capacity/facet_normals/test_algorithms.py`)
- **test_fast_matches_reference** — Cross-check between reference and fast implementations on curated polytopes.
  Classification: Math-level correctness.
- **test_symplectic_invariance_square** — Verifies symplectic coordinate changes leave the capacity invariant.
  Classification: Math-level correctness.
- **test_rejects_odd_dimension** — Ensures both implementations reject polytopes in odd dimensions. Classification:
  Code-level correctness.
- **test_prepare_subset_respects_tol** — Asserts the subset pre-processing honours caller-provided tolerances, focusing
  on numerical safeguards. Classification: Code-level correctness.

## Symplectic Systolic Ratio (`tests/viterbo/symplectic/systolic/test_systolic.py`)
- **test_systolic_ratio_translation_invariant** — Confirms translation invariance of the systolic ratio for a simplex.
  Classification: Math-level correctness.
- **test_systolic_ratio_scale_invariant** — Checks scaling invariance by dilating the halfspace offsets. Classification:
  Math-level correctness.
- **test_simplex_ratio_positive** — Ensures the ratio is strictly positive on the simplex. Classification:
  Math-level correctness.
- **test_raw_halfspace_input_validates_shapes** — Validates shape checking for halfspace inputs. Classification:
  Code-level correctness.

## Optimization — Solvers (`tests/viterbo/optimization/test_solvers.py`)
- **test_linear_program_validation_rejects_mismatched_rhs** — Ensures constructor validation for linear programs.
  Classification: Code-level correctness.
- **test_scipy_backend_solves_simple_problem** — Solves a canonical LP and checks optimal status and solution; validates
  the API-to-backend wiring. Classification: Code-level correctness.
- **test_solve_linear_program_uses_default_backend** — Ensures the convenience wrapper delegates to the default backend.
  Classification: Code-level correctness.
- **test_mixed_integer_solver_respects_integrality_constraints** — Checks integer-enforced variables snap to integers in
  an MILP. Classification: Code-level correctness.
- **test_mixed_integer_solver_supports_maximisation** — Verifies maximisation mode and bound handling in the MILP
  wrapper. Classification: Code-level correctness.
- **test_normalize_bounds_expands_scalars** — Ensures scalar bounds broadcast correctly when normalizing constraints.
  Classification: Code-level correctness.
- **test_normalize_bounds_validates_sequence** — Confirms invalid-length bounds or inverted intervals raise ValueError.
  Classification: Code-level correctness.
- **test_normalize_bounds_rejects_nan_entries** — Ensures NaNs are rejected during bounds normalization. Classification:
  Code-level correctness.

## Optimization — Search (`tests/viterbo/optimization/test_search.py`)
- **test_enumerate_search_space_deterministic** — Checks deterministic enumeration when PRNG seed and knobs are fixed.
  Classification: Code-level correctness.
- **test_search_space_contains_catalog** — Ensures the search space contains all catalogued base polytopes, protecting
  coverage. Classification: Math-level correctness.
- **test_iter_search_space_respects_max_candidates** — Validates iterator truncation behaviour. Classification:
  Code-level correctness.
- **test_iter_search_space_rejects_unknown_kwargs** — Ensures unexpected keyword arguments raise errors. Classification:
  Code-level correctness.
- **test_iter_search_space_honours_dimension_cap** — Checks enumerator honours the maximum dimension cap. Classification:
  Code-level correctness.

## Performance Benchmarks — Volume (`tests/performance/viterbo/geometry/volume/test_volume_benchmarks.py`)
- **test_volume_fast_matches_reference** — Benchmarks the fast volume estimator against the reference while asserting
  numerical agreement. Classification: Performance (with embedded math check).

## Performance Benchmarks — EHZ Capacity (`tests/performance/viterbo/symplectic/capacity/facet_normals/test_ehz_capacity_benchmarks.py`)
- **test_fast_ehz_capacity_matches_reference_and_tracks_speed** — Benchmarks the optimized EHZ capacity kernel on the
  catalogued polytopes, ensuring timing regressions are observable while confirming results match the reference (or
  share failure messages). Classification: Performance (with embedded math check).
