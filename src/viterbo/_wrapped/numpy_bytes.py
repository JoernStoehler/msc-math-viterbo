"""Utilities that require NumPy byte-level conversions (hashing, etc.)."""

from __future__ import annotations

import hashlib

import numpy as _np


def fingerprint_halfspace(matrix: object, offsets: object, *, decimals: int = 12) -> str:
    """Return a deterministic fingerprint for a half-space system.

    Converts to contiguous float64 NumPy arrays, rounds to ``decimals`` and hashes
    the shapes and raw bytes. Accepts JAX arrays or array-like inputs.
    """
    rounded_matrix = _np.round(_np.asarray(matrix, dtype=float), decimals=decimals)
    rounded_offsets = _np.round(_np.asarray(offsets, dtype=float), decimals=decimals)

    contiguous_matrix = _np.ascontiguousarray(rounded_matrix)
    contiguous_offsets = _np.ascontiguousarray(rounded_offsets)

    hasher = hashlib.sha256()
    hasher.update(_np.array(contiguous_matrix.shape, dtype=_np.int64).tobytes())
    hasher.update(_np.array(contiguous_offsets.shape, dtype=_np.int64).tobytes())
    hasher.update(contiguous_matrix.tobytes())
    hasher.update(contiguous_offsets.tobytes())
    return hasher.hexdigest()
