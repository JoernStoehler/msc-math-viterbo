from __future__ import annotations

from typing import Sequence

from jax import Array

def PRNGKey(seed: int) -> Array: ...
def split(key: Array, num: int = 2) -> tuple[Array, ...]: ...
def normal(
    key: Array, shape: Sequence[int] | tuple[int, ...] | None = ..., dtype: object | None = ...
) -> Array: ...
def uniform(
    key: Array,
    shape: Sequence[int] | tuple[int, ...] | None = ...,
    *,
    minval: float | Array = 0.0,
    maxval: float | Array = 1.0,
    dtype: object | None = ...,
) -> Array: ...
def randint(
    key: Array,
    shape: Sequence[int] | tuple[int, ...],
    minval: int,
    maxval: int,
    *,
    dtype: object | None = ...,
) -> Array: ...
def bernoulli(
    key: Array,
    p: float | Array = 0.5,
    shape: Sequence[int] | tuple[int, ...] | None = ...,
    *,
    dtype: object | None = ...,
) -> Array: ...
