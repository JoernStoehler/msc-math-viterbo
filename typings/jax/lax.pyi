from __future__ import annotations

from typing import Any, Callable, TypeVar

from jax import Array

R = TypeVar("R")

def select(pred: Array | bool, on_true: Any, on_false: Any) -> Any: ...
def cond(
    pred: Array | bool,
    true_fun: Callable[[Any], R],
    false_fun: Callable[[Any], R],
    operand: Any = ...,
) -> R: ...
def scan(f: Callable[[Any, Any], tuple[Any, Any]], init: Any, xs: Any) -> tuple[Any, Any]: ...
