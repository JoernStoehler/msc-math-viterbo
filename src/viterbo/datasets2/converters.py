"""Helpers for converting between JAX arrays and HuggingFace-friendly types."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


def array_to_python(value: Array | Sequence[float] | float) -> object:
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


def bool_array_to_python(value: Array | Sequence[bool] | bool) -> object:
    """Convert a boolean JAX/NumPy array or sequence into nested Python lists."""

    if isinstance(value, jnp.ndarray):
        return np.asarray(value, dtype=bool).tolist()
    if isinstance(value, np.ndarray):
        return value.astype(bool).tolist()
    if isinstance(value, (list, tuple)):
        return [bool_array_to_python(v) for v in value]
    return bool(value)


def python_to_array(value: object, *, dtype: object = jnp.float64) -> Array:
    """Convert a Python nested sequence back into a JAX array."""

    return jnp.asarray(value, dtype=dtype)


def python_to_bool_array(value: object) -> Array:
    """Convert a Python nested sequence of booleans back into a JAX array."""

    return jnp.asarray(value, dtype=bool)
