"""Top-level package for the Viterbo tools (flat modern namespace)."""

from __future__ import annotations

# Public capacity API
from viterbo.capacity import (
    MinkowskiNormalFan as MinkowskiNormalFan,
)
from viterbo.capacity import (
    build_normal_fan as build_normal_fan,
)
from viterbo.capacity import (
    ehz_capacity_reference as compute_ehz_capacity_reference,
)
from viterbo.capacity import (
    minkowski_billiard_length_fast as compute_minkowski_billiard_length_fast,
)
from viterbo.capacity import (
    minkowski_billiard_length_reference as compute_minkowski_billiard_length_reference,
)

# Symplectic helpers
from viterbo.symplectic import ZERO_TOLERANCE as ZERO_TOLERANCE
from viterbo.symplectic import minkowski_sum as minkowski_sum
from viterbo.symplectic import normalize_vector as normalize_vector
from viterbo.symplectic import standard_symplectic_matrix as standard_symplectic_matrix
from viterbo.symplectic import support_function as support_function
from viterbo.symplectic import symplectic_product as symplectic_product

# Systolic ratio
from viterbo.systolic import systolic_ratio as systolic_ratio


__all__ = [
    "MinkowskiNormalFan",
    "build_normal_fan",
    "compute_ehz_capacity_reference",
    "compute_minkowski_billiard_length_fast",
    "compute_minkowski_billiard_length_reference",
    "ZERO_TOLERANCE",
    "minkowski_sum",
    "normalize_vector",
    "standard_symplectic_matrix",
    "support_function",
    "symplectic_product",
    "systolic_ratio",
]
