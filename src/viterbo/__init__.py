"""Top-level package for the Viterbo tools (reference + fast variants)."""

from __future__ import annotations

from viterbo.geometry import NormalCone as NormalCone
from viterbo.geometry import Polytope as Polytope
from viterbo.geometry import PolytopeCombinatorics as PolytopeCombinatorics
from viterbo.geometry import affine_transform as affine_transform
from viterbo.geometry import cartesian_product as cartesian_product
from viterbo.geometry import catalog as catalog
from viterbo.geometry import cross_polytope as cross_polytope
from viterbo.geometry import enumerate_vertices as enumerate_vertices
from viterbo.geometry import haim_kislev_action as haim_kislev_action
from viterbo.geometry import halfspaces_from_vertices as halfspaces_from_vertices
from viterbo.geometry import hypercube as hypercube
from viterbo.geometry import mirror_polytope as mirror_polytope
from viterbo.geometry import polytope_combinatorics as polytope_combinatorics
from viterbo.geometry import polytope_fingerprint as polytope_fingerprint
from viterbo.geometry import (
    polytope_volume_fast as polytope_volume_fast,
)
from viterbo.geometry import (
    polytope_volume_reference as polytope_volume_reference,
)
from viterbo.geometry import random_affine_map as random_affine_map
from viterbo.geometry import random_polytope as random_polytope
from viterbo.geometry import random_transformations as random_transformations
from viterbo.geometry import regular_polygon_product as regular_polygon_product
from viterbo.geometry import remove_redundant_facets as remove_redundant_facets
from viterbo.geometry import rotate_polytope as rotate_polytope
from viterbo.geometry import simplex_with_uniform_weights as simplex_with_uniform_weights
from viterbo.geometry import translate_polytope as translate_polytope
from viterbo.geometry import truncated_simplex_four_dim as truncated_simplex_four_dim
from viterbo.geometry import vertices_from_halfspaces as vertices_from_halfspaces
from viterbo.geometry import viterbo_counterexample as viterbo_counterexample
from viterbo.optimization import LinearProgram as LinearProgram
from viterbo.optimization import LinearProgramBackend as LinearProgramBackend
from viterbo.optimization import LinearProgramSolution as LinearProgramSolution
from viterbo.optimization import (
    MixedIntegerLinearProgram as MixedIntegerLinearProgram,
)
from viterbo.optimization import (
    MixedIntegerLinearProgramSolution as MixedIntegerLinearProgramSolution,
)
from viterbo.optimization import ScipyLinearProgramBackend as ScipyLinearProgramBackend
from viterbo.optimization import enumerate_search_space as enumerate_search_space
from viterbo.optimization import iter_search_space as iter_search_space
from viterbo.optimization import solve_linear_program as solve_linear_program
from viterbo.optimization import (
    solve_mixed_integer_linear_program as solve_mixed_integer_linear_program,
)
from viterbo.symplectic.capacity import (
    compute_ehz_capacity_reference as compute_ehz_capacity_reference,
)
from viterbo.symplectic.capacity.facet_normals.fast import (
    compute_ehz_capacity_fast as compute_ehz_capacity_fast,
)
from viterbo.symplectic.core import ZERO_TOLERANCE as ZERO_TOLERANCE
from viterbo.symplectic.core import minkowski_sum as minkowski_sum
from viterbo.symplectic.core import normalize_vector as normalize_vector
from viterbo.symplectic.core import standard_symplectic_matrix as standard_symplectic_matrix
from viterbo.symplectic.core import support_function as support_function
from viterbo.symplectic.core import symplectic_product as symplectic_product
from viterbo.symplectic.systolic import systolic_ratio as systolic_ratio
