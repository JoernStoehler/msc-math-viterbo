"""Runtime helpers and guardrails shared across the project."""

from __future__ import annotations

import functools
import os
import signal
import time
from collections.abc import Callable
from types import FrameType
from typing import ParamSpec, TypeVar

__all__ = ["TimeBudgetExceededError", "enforce_time_budget"]

P = ParamSpec("P")
R = TypeVar("R")


class TimeBudgetExceededError(RuntimeError):
    """Raised when a runtime budget (wall-clock limit) is exceeded."""


def _default_timeout(seconds: float | None) -> float:
    if seconds is not None:
        return float(seconds)
    env_value = os.getenv("VITERBO_SOLVER_TIMEOUT", "30")
    try:
        return float(env_value)
    except ValueError:
        return 30.0


def enforce_time_budget(seconds: float | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator enforcing a per-call wall-clock budget.

    Args:
      seconds: Maximum wall time in seconds (environment fallback:
        ``VITERBO_SOLVER_TIMEOUT``). Non-positive values disable the guard.
    """

    timeout = _default_timeout(seconds)
    if timeout <= 0:

        def _noop(func: Callable[P, R]) -> Callable[P, R]:
            return func

        return _noop

    supports_signals = hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if not supports_signals:

            @functools.wraps(func)
            def slow_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start = time.monotonic()
                result = func(*args, **kwargs)
                if time.monotonic() - start > timeout:
                    raise TimeBudgetExceededError(
                        f"{func.__name__} exceeded allotted {timeout} seconds "
                        f"(timed out post-execution on unsupported platform)"
                    )
                return result

            return slow_wrapper

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            def _handle_timeout(
                signum: int, frame: FrameType | None
            ) -> None:  # pragma: no cover - signal path
                raise TimeBudgetExceededError(
                    f"{func.__name__} exceeded allotted {timeout} seconds"
                )

            previous_handler = signal.getsignal(signal.SIGALRM)
            previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
            signal.signal(signal.SIGALRM, _handle_timeout)

            try:
                return func(*args, **kwargs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_timer[0] > 0.0:
                    signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])

        return wrapper

    return decorator
