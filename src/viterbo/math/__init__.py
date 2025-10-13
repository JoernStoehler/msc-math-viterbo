"""Pure math utilities (Torch tensors in/out)."""

from __future__ import annotations

# Features we need:

## Polytope Geometry

# Representation
#   normals: (n, d) tensor
#   offsets: (n,) tensor
#   => normals @ x <= offsets
#   vertices: (m, d) tensor
#   => x \in ConvexHull(vertices)


# Functions
def halfspaces_to_vertices(normals, offsets):
    """Convert halfspace representation (normals, offsets) to vertices.

    Args:
        normals: (n, d) tensor of facet normals.
        offsets: (n,) tensor of facet offsets.

    Returns:
        vertices: (m, d) tensor of vertices.
    """
    raise NotImplementedError


def vertices_to_halfspaces(vertices):
    """Convert vertices to halfspace representation (normals, offsets).

    Args:
        vertices: (m, d) tensor of vertices.

    Returns:
        normals: (n, d) tensor of facet normals.
        offsets: (n,) tensor of facet offsets.
    """
    raise NotImplementedError
