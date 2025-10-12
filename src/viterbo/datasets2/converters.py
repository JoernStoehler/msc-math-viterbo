"""Helpers for converting between JAX arrays and HuggingFace-friendly types."""

from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


Serializable = Any


def array_to_python(value: Array | Sequence[float]) -> Serializable:
    """Return a Python-native representation for ``value``.

    HuggingFace datasets expect Python scalars, lists, or dictionaries. This helper
    mirrors ``numpy.asarray(value).tolist()`` while accepting JAX arrays and
    sequences interchangeably.
    """

    if isinstance(value, jnp.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [array_to_python(v) for v in value]
    return value


def bool_array_to_python(value: Array | Sequence[bool]) -> Serializable:
    """Convert a boolean JAX/NumPy array or sequence into nested Python lists."""

    if isinstance(value, jnp.ndarray):
        return np.asarray(value, dtype=bool).tolist()
    if isinstance(value, np.ndarray):
        return value.astype(bool).tolist()
    if isinstance(value, (list, tuple)):
        return [bool_array_to_python(v) for v in value]
    return bool(value)


def python_to_array(value: Serializable, *, dtype=jnp.float64) -> Array:
    """Convert a Python nested sequence back into a JAX array."""

    return jnp.asarray(value, dtype=dtype)


def python_to_bool_array(value: Serializable) -> Array:
    """Convert a Python nested sequence of booleans back into a JAX array."""

    return jnp.asarray(value, dtype=bool)


__all__ = [
    "Serializable",
    "array_to_python",
    "bool_array_to_python",
    "python_to_array",
    "python_to_bool_array",
]
