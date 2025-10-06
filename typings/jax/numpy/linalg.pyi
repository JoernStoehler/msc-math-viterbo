from __future__ import annotations

from typing import Any, Literal, overload

from jax import Array

def inv(a: Array) -> Array: ...
def det(a: Array) -> Array: ...
def solve(a: Array, b: Array) -> Array: ...
@overload
def svd(a: Array, compute_uv: Literal[False] = ..., full_matrices: bool = ...) -> Array: ...
@overload
def svd(
    a: Array,
    compute_uv: Literal[True],
    full_matrices: bool = ...,
) -> tuple[Array, Array, Array]: ...
def qr(a: Array, mode: str = ...) -> tuple[Array, Array]: ...
def norm(
    x: Array,
    ord: Any | None = ...,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool = ...,
) -> Array: ...
