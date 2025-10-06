# Test Suite Review Notes

## tests/viterbo/symplectic/test_core.py

- tests/viterbo/symplectic/test_core.py:21 `def test_standard_symplectic_matrix_structure() -> None`
  — default smoke (no explicit marker); verifies closed-form structure of `J` matrix, anchors
  logical correctness and API stability for shape/dtype; no tunable parameters beyond dimension 4;
  tiny runtime; cheap to maintain unless matrix layout changes; alternative: could add jaxtyping
  guard but current array equality already strong.
- tests/viterbo/symplectic/test_core.py:35
  `def test_standard_symplectic_matrix_requires_even_dimension(dimension: int) -> None` —
  parametrized over odd `dimension` values [1, 3]; default smoke; enforces error contract (logical correctness/API)
  with negligible runtime; maintenance risk is low but remember to extend cases if API allows higher
  odd dims.
- tests/viterbo/symplectic/test_core.py:40 `def test_symplectic_product_default_matrix() -> None` —
  smoke; checks bilinear form invariants under default `J`; covers logical correctness/numerical
  fidelity via `math.isclose`; runtime minimal; future variant could add nontrivial vectors to cover
  antisymmetry property.
- tests/viterbo/symplectic/test_core.py:52 `def test_symplectic_product_custom_matrix() -> None` —
  smoke; exercises optional matrix argument (API stability) and numerical scaling; depends on
  hard-coded values; negligible runtime; alternative: include tolerance awareness for non-integer
  entries.
- tests/viterbo/symplectic/test_core.py:58 `def test_support_function_simplex() -> None` — smoke;
  deterministic simplex fixture; validates logical correctness and numerical fidelity of support
  function; cost low; potential extension: include negative direction or higher dims.
- tests/viterbo/symplectic/test_core.py:71 `def test_minkowski_sum_pairwise_vertices() -> None` —
  smoke; small combinatorial expansion verifies logical correctness; runtime linear in vertex count
  (8); ready to promote to deep if we ever scale sizes; alternatives: property-based check for
  commutativity.
- tests/viterbo/symplectic/test_core.py:97 `def test_support_function_validates_inputs() -> None` —
  smoke; focuses on API validation errors; multiple bad inputs; no runtime concerns; complements
  type hints by exercising runtime guards.
- tests/viterbo/symplectic/test_core.py:108 `def test_minkowski_sum_validates_inputs() -> None` —
  smoke; same rationale for Minkowski sum; minimal cost.
- tests/viterbo/symplectic/test_core.py:119 `def test_normalize_vector_unit_length() -> None` —
  smoke; covers numerical correctness (norm=1) and dtype conversion; uses double precision default;
  tiny runtime; maybe add orientation preservation check later.
- tests/viterbo/symplectic/test_core.py:126 `def test_normalize_vector_zero_vector_raises() -> None`
  — smoke; ensures domain validation (logical correctness/API); no runtime issues.
- tests/viterbo/symplectic/test_core.py:132 `def test_normalize_vector_accepts_list_input() -> None`
  — smoke; ensures flexible input types (API usability); warns that pyright ignores list type;
  trivial runtime; ok.
- tests/viterbo/symplectic/test_core.py:139 `def test_zero_tolerance_reasonable() -> None` — smoke;
  guards constant range (observability, downstream comparisons); zero runtime; ensures future
  refactors keep tolerance small.

## tests/viterbo/symplectic/capacity/facet_normals/test_reference.py

- tests/viterbo/symplectic/capacity/facet_normals/test_reference.py:34
  `def test_capacity_matches_baseline(polytope_name: str) -> None` — smoke (default) but runtime
  depends on size of baseline catalog; iterates through JSON baselines, ensures numerical fidelity
  and observability; parameterized dataset sensitive to baseline upkeep; cost moderate because
  baseline refresh needs PI sign-off; alternative: shard by tier once catalog expands.
- tests/viterbo/symplectic/capacity/facet_normals/test_reference.py:44
  `def test_capacity_scales_quadratically_under_dilation() -> None` — smoke; checks mathematical
  invariance (logical correctness/numerical fidelity); uses scalar `scale`; runtime low; constant
  1.5 is a tweakable knob.
- tests/viterbo/symplectic/capacity/facet_normals/test_reference.py:57
  `def test_capacity_is_translation_invariant() -> None` — smoke; uses fixed translation vector to
  exercise invariance; ensures algorithmic correctness; runtime minimal; knob: translation values if
  needing stress.
- tests/viterbo/symplectic/capacity/facet_normals/test_reference.py:72
  `def test_truncated_simplex_matches_known_subset_action() -> None` — smoke; relies on catalog
  metadata `reference_capacity`; guards regression of stored baselines; runtime moderate because
  truncated simplex evaluation is heavier; should probably move to deep tier if runtime observed >10
  s; alternative: convert to snapshot JSON.
- tests/viterbo/symplectic/capacity/facet_normals/test_reference.py:83
  `def test_two_dimensional_simplex_matches_fast_capacity() -> None` — smoke; cross-check reference
  vs fast pipeline for 2D case; covers logical/numerical correctness and API cross-consistency;
  runtime very low; no extra knobs.

## tests/viterbo/symplectic/capacity/facet_normals/test_algorithms.py

- tests/viterbo/symplectic/capacity/facet_normals/test_algorithms.py:24
  `@pytest.fixture name="subset_utils_close_records"` — fixture instrumentation; used in smoke/deep
  contexts; captures tolerance arguments to assess observability; cost: monkeypatching adds
  complexity but needed for verifying closeness configuration.
- tests/viterbo/symplectic/capacity/facet_normals/test_algorithms.py:67
  `def test_fast_matches_reference(B: np.ndarray, c: np.ndarray) -> None` — default smoke currently
  but runtime heavy: iterates through sample polytopes (variant_count=0) yet still large;
  classification should likely be deep; quality dimension: numerical fidelity & API parity; handles
  both ValueError parity asserts; maintenance heavy when dataset grows; consider splitting into
  smoke subset vs deep sweep.
- tests/viterbo/symplectic/capacity/facet_normals/test_algorithms.py:81
  `def test_symplectic_invariance_square() -> None` — smoke; verifies symplectic invariance within
  reference algorithm; includes skip branch when algorithm undefined; runtime moderately high due to
  matrix operations but acceptable; ensures logical correctness; knob: rotation angle `np.pi/6`.
- tests/viterbo/symplectic/capacity/facet_normals/test_algorithms.py:107
  `def test_rejects_odd_dimension() -> None` — smoke; ensures odd dimension raises ValueError in
  both implementations; tiny runtime; classification ok.
- tests/viterbo/symplectic/capacity/facet_normals/test_algorithms.py:117
  `def test_prepare_subset_respects_tol(...) -> None` — smoke but more like deep because of
  monkeypatch overhead; quality: observability (tolerance propagation) and API; runtime low; ensures
  closeness thresholds fixed at 1e-6; knob: `tol` argument.

## tests/viterbo/symplectic/capacity/facet_normals/test_fast.py

- tests/viterbo/symplectic/capacity/facet_normals/test_fast.py:22
  `def test_dynamic_program_matches_bruteforce() -> None` — smoke; tests combinatorial DP vs brute
  force on 5x5 antisymmetric weights; quality: logical correctness and numerical fidelity; runtime
  notable due to 5! permutation loop (~120 iterations) but still quick; knobs: matrix size
  (currently 5), PRNG seed (0); possible deep variant for larger size.
- tests/viterbo/symplectic/capacity/facet_normals/test_fast.py:47
  `def test_fast_implementation_matches_reference(B: np.ndarray, c: np.ndarray) -> None` — currently
  smoke (no marker) but dataset load is large (all variants); high runtime and repeated JAX traces;
  quality: numerical fidelity & API parity; handles ValueError message parity; should migrate to
  deep or longhaul due to timeouts; alternative: keep single cheap instance in smoke and move sweep
  elsewhere.

## tests/viterbo/symplectic/systolic/test_systolic.py

- tests/viterbo/symplectic/systolic/test_systolic.py:15
  `def test_systolic_ratio_translation_invariant() -> None` — smoke; ensures invariance under
  translation; uses deterministic offset vector; quality: logical correctness & API; runtime low.
- tests/viterbo/symplectic/systolic/test_systolic.py:21
  `def test_systolic_ratio_scale_invariant() -> None` — smoke; uses scalar 1.5 to verify scaling
  invariance; runtime low; knob: scale factor.
- tests/viterbo/symplectic/systolic/test_systolic.py:28 `def test_simplex_ratio_positive() -> None`
  — smoke; sanity guard for positivity; minimal cost.
- tests/viterbo/symplectic/systolic/test_systolic.py:33
  `def test_raw_halfspace_input_validates_shapes() -> None` — smoke; covers API validation for
  incorrect shapes; ensures ValueError surfaces; runtime trivial; good complement to type hints.

## tests/viterbo/geometry/volume/test_volume.py

- tests/viterbo/geometry/volume/test_volume.py:27
  `def test_hypercube_volume_matches_closed_form() -> None` — smoke; verifies closed-form volume
  equality; guards numerical fidelity; runtime small; knobs: radius=1.5, dimension=4.
- tests/viterbo/geometry/volume/test_volume.py:35 `def test_random_polytope_volumes_agree() -> None`
  — smoke; draws random polytope via fixed PRNG seed; cross-checks fast vs reference; runtime
  moderate due to sampling; ensures numerical fidelity; knob: seed 42; potential deep candidate for
  multiple samples.
- tests/viterbo/geometry/volume/test_volume.py:42 `def test_simplex_volume_positive() -> None` —
  smoke; simple positivity assertion; minimal cost; logical correctness.
- tests/viterbo/geometry/volume/test_volume.py:49
  `def test_hypercube_samples_match_expected_volume() -> None` — smoke; uses helper producing
  matrix/offsets plus theoretical volume; ensures helper fidelity; runtime low; quality: logical and
  numerical; knob: radius 2.0.

## tests/viterbo/geometry/polytopes/test_transforms.py

- tests/viterbo/geometry/polytopes/test_transforms.py:25
  `def test_translate_polytope_updates_offsets() -> None` — smoke; ensures translation updates
  offsets consistent with linear algebra; quality: logical correctness & API invariants; runtime
  minimal.
- tests/viterbo/geometry/polytopes/test_transforms.py:35
  `def test_cartesian_product_dimensions_add() -> None` — smoke; ensures metadata updates for
  Cartesian product; logical correctness; cost low.
- tests/viterbo/geometry/polytopes/test_transforms.py:43
  `def test_mirror_polytope_flips_coordinate() -> None` — smoke; ensures axis flip only affects
  selected coordinate; tests invariance; runtime low; uses `axes` tuple knob.
- tests/viterbo/geometry/polytopes/test_transforms.py:51
  `def test_rotate_polytope_consistency_in_plane() -> None` — smoke; checks rotation by π/4; ensures
  linear algebra alignment; runtime small; knob: angle and plane selection.
- tests/viterbo/geometry/polytopes/test_transforms.py:62
  `def test_random_affine_map_is_deterministic_per_seed() -> None` — smoke; asserts keyed
  determinism and randomness after splitting; quality: logical correctness & API; runtime moderate
  due to random generation; ensures condition number below 1e6; knobs: PRNG seed 10, cond threshold.
- tests/viterbo/geometry/polytopes/test_transforms.py:79
  `def test_random_polytope_facets_are_active() -> None` — smoke; ensures each facet active by
  checking vertices equality; quality: numerical fidelity and observability; runtime higher because
  enumerates vertices (convex hull); candidate to shift to deep if slow; uses seed 2024,
  dimension 3.

## tests/viterbo/geometry/polytopes/test_haim_kislev_action.py

- tests/viterbo/geometry/polytopes/test_haim_kislev_action.py:12
  `def test_haim_kislev_action_valid_order_matches_reference_capacity() -> None` — smoke; ensures
  action matches stored reference capacity; quality: numerical fidelity; runtime depends on
  truncated simplex solver (~heavy but manageable); maintenance: baseline coupling to
  `reference_capacity` attr.
- tests/viterbo/geometry/polytopes/test_haim_kislev_action.py:22
  `def test_haim_kislev_action_invalid_order_raises_value_error() -> None` — smoke; validates error
  handling for invalid permutations; logical correctness; trivial runtime.

## tests/viterbo/geometry/polytopes/test_combinatorics.py

- tests/viterbo/geometry/polytopes/test_combinatorics.py:27
  `def test_polytope_combinatorics_square_facets() -> None` — smoke; ensures combinatorics object
  shapes and adjacency degrees; quality: logical correctness & API stability; runtime minimal.
- tests/viterbo/geometry/polytopes/test_combinatorics.py:39
  `def test_polytope_combinatorics_properties() -> None` — smoke; same for cube; note adjacency
  degree check tolerance; runtime small.
- tests/viterbo/geometry/polytopes/test_combinatorics.py:56
  `def test_vertex_enumeration_matches_reference_shape() -> None` — smoke; ensures vertex
  enumeration stable; numerical fidelity; runtime moderate due to enumeration but ok.
- tests/viterbo/geometry/polytopes/test_combinatorics.py:64
  `def test_polytope_combinatorics_cached_instances_are_reused() -> None` — smoke; ensures caching
  semantics; API stability; trivial runtime; caution: relies on object identity.
- tests/viterbo/geometry/polytopes/test_combinatorics.py:71
  `def test_halfspace_vertex_roundtrip() -> None` — smoke; ensures roundtrip from vertices back to
  halfspaces; numerical fidelity & API compatibility; runtime medium; knobs: sorting helper ensures
  deterministic comparison.
- tests/viterbo/geometry/polytopes/test_combinatorics.py:82
  `def test_polytope_fingerprint_invariant_to_metadata() -> None` — smoke; ensures fingerprint
  independent of metadata; API stability; minimal runtime.

## tests/viterbo/geometry/halfspaces/test_halfspaces.py

- tests/viterbo/geometry/halfspaces/test_halfspaces.py:33
  `def test_reference_enumeration_matches_expected_square() -> None` — smoke; verifies reference
  enumerator correctness; numerical fidelity; runtime trivial; relies on helper `_run_enumerator`
  (unit square).
- tests/viterbo/geometry/halfspaces/test_halfspaces.py:47
  `def test_fast_enumerator_matches_reference() -> None` — smoke; cross-check fast vs reference
  enumerators; numerical fidelity; runtime low; ensures both produce same vertices.
- tests/viterbo/geometry/halfspaces/test_halfspaces.py:62
  `def test_remove_redundant_facets_discards_duplicates() -> None` — smoke; ensures both
  implementations drop redundant facets; logical correctness; runtime low; knob: duplication count
  (currently one extra facet).

## tests/viterbo/optimization/test_search.py

- tests/viterbo/optimization/test_search.py:9
  `def test_enumerate_search_space_deterministic() -> None` — smoke; ensures deterministic
  enumeration for fixed seed and parameters; logical correctness & API stability; runtime moderate
  because constructs search space; knobs: `max_dimension`, `transforms_per_base`,
  `random_polytopes_per_dimension`.
- tests/viterbo/optimization/test_search.py:29 `def test_search_space_contains_catalog() -> None` —
  smoke; ensures core catalog subset property; quality: coverage for regressions in enumeration;
  runtime moderate; may become heavy if catalog grows; alternative: sample subset or move to deep
  when dataset balloons.

## tests/viterbo/optimization/test_solvers.py

- tests/viterbo/optimization/test_solvers.py:20
  `def test_linear_program_validation_rejects_mismatched_rhs() -> None` — smoke; validates argument
  length checking; logical correctness; runtime trivial.
- tests/viterbo/optimization/test_solvers.py:26
  `def test_scipy_backend_solves_simple_problem() -> None` — smoke; ensures backend solves small LP;
  quality: numerical fidelity and API; runtime modest due to SciPy call (may spin up solver); knob:
  constraint setup; note reliance on SciPy availability.
- tests/viterbo/optimization/test_solvers.py:41
  `def test_solve_linear_program_uses_default_backend() -> None` — smoke; ensures convenience
  wrapper uses default backend; logical correctness; runtime similar to previous; coupling to
  default backend configuration.
- tests/viterbo/optimization/test_solvers.py:54
  `def test_mixed_integer_solver_respects_integrality_constraints() -> None` — smoke; ensures
  integrality enforcement; numerical fidelity; runtime moderate because MIP solver may be slower;
  watch for CI time; knob: bounds/integrality vector.
- tests/viterbo/optimization/test_solvers.py:72
  `def test_mixed_integer_solver_supports_maximisation() -> None` — smoke; tests maximize flag;
  ensures solver returns proper sum; runtime moderate; may belong to deep tier if solver latency
  increases; alternative: shrink to smaller search for smoke.

## tests/performance/viterbo/symplectic/capacity/facet_normals/test_ehz_capacity_benchmarks.py

- tests/performance/viterbo/symplectic/capacity/facet_normals/test_ehz_capacity_benchmarks.py:35
  `def test_fast_ehz_capacity_matches_reference_and_tracks_speed(... benchmark, B, c)` — explicitly
  marked `@pytest.mark.benchmark`, `pytestmark = [smoke, deep]`; belongs to both smoke and deep
  tiers, though runtime may exceed smoke budgets due to multiple instances; quality: performance
  envelopes and numerical fidelity; knobs: dataset from `load_polytope_instances`, tolerance
  `atol=1e-8`; cost: high because of repeated benchmarking per instance; consider dropping smoke
  marker if runtime remains high; alternative: sample minimal subset for smoke, keep full for deep.

## tests/performance/viterbo/geometry/volume/test_volume_benchmarks.py

- tests/performance/viterbo/geometry/volume/test_volume_benchmarks.py:17
  `def test_volume_fast_matches_reference(... benchmark)` — marked smoke+deep, plus benchmark;
  quality: performance envelope & numerical fidelity for random polytope; knobs: PRNG seed 314;
  runtime moderate (benchmark loops); consider demoting to deep-only if CI budget tight;
  alternative: add reference to `.benchmarks` artifact usage.
