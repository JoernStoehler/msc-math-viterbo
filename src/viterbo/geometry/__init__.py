"""Geometry primitives and polytope helpers."""

from viterbo.geometry.halfspaces import (
    enumerate_vertices as enumerate_vertices,
)
from viterbo.geometry.halfspaces import (
    remove_redundant_facets as remove_redundant_facets,
)
from viterbo.geometry.polytopes import (
    NormalCone as NormalCone,
)
from viterbo.geometry.polytopes import (
    Polytope as Polytope,
)
from viterbo.geometry.polytopes import (
    PolytopeCombinatorics as PolytopeCombinatorics,
)
from viterbo.geometry.polytopes import (
    affine_transform as affine_transform,
)
from viterbo.geometry.polytopes import (
    cartesian_product as cartesian_product,
)
from viterbo.geometry.polytopes import (
    catalog as catalog,
)
from viterbo.geometry.polytopes import (
    cross_polytope as cross_polytope,
)
from viterbo.geometry.polytopes import (
    haim_kislev_action as haim_kislev_action,
)
from viterbo.geometry.polytopes import (
    halfspaces_from_vertices as halfspaces_from_vertices,
)
from viterbo.geometry.polytopes import (
    hypercube as hypercube,
)
from viterbo.geometry.polytopes import (
    mirror_polytope as mirror_polytope,
)
from viterbo.geometry.polytopes import (
    polytope_combinatorics as polytope_combinatorics,
)
from viterbo.geometry.polytopes import (
    polytope_fingerprint as polytope_fingerprint,
)
from viterbo.geometry.polytopes import (
    random_affine_map as random_affine_map,
)
from viterbo.geometry.polytopes import (
    random_polytope as random_polytope,
)
from viterbo.geometry.polytopes import (
    random_transformations as random_transformations,
)
from viterbo.geometry.polytopes import (
    regular_polygon_product as regular_polygon_product,
)
from viterbo.geometry.polytopes import (
    rotate_polytope as rotate_polytope,
)
from viterbo.geometry.polytopes import (
    simplex_with_uniform_weights as simplex_with_uniform_weights,
)
from viterbo.geometry.polytopes import (
    translate_polytope as translate_polytope,
)
from viterbo.geometry.polytopes import (
    truncated_simplex_four_dim as truncated_simplex_four_dim,
)
from viterbo.geometry.polytopes import (
    vertices_from_halfspaces as vertices_from_halfspaces,
)
from viterbo.geometry.polytopes import (
    viterbo_counterexample as viterbo_counterexample,
)
from viterbo.geometry.volume import (
    polytope_volume_fast as polytope_volume_fast,
)
from viterbo.geometry.volume import (
    polytope_volume_reference as polytope_volume_reference,
)
