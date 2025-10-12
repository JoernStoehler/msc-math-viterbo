"""Top-level package for the Viterbo tools (thin convenience namespace)."""

from __future__ import annotations

# Symplectic helpers
from viterbo.math.symplectic import ZERO_TOLERANCE as ZERO_TOLERANCE
from viterbo.math.symplectic import minkowski_sum as minkowski_sum
from viterbo.math.symplectic import normalize_vector as normalize_vector
from viterbo.math.symplectic import standard_symplectic_matrix as standard_symplectic_matrix
from viterbo.math.symplectic import support_function as support_function
from viterbo.math.symplectic import symplectic_product as symplectic_product

# Systolic ratio
from viterbo.systolic import systolic_ratio as systolic_ratio


__all__ = [
    "ZERO_TOLERANCE",
    "minkowski_sum",
    "normalize_vector",
    "standard_symplectic_matrix",
    "support_function",
    "symplectic_product",
    "systolic_ratio",
]
