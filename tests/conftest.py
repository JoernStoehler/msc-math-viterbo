"""Test configuration.

Provides an optional runtime shape/type checker for jaxtyping when
the environment variable `JAXTYPING_CHECKS=1` is set. This keeps
production runs fast while allowing strict checks locally.
"""

from __future__ import annotations

import os
from typing import Iterator

import jax
import pytest

SMOKE_TIMEOUT_SECONDS = 10


def _maybe_enable_jaxtyping_checks() -> None:
    if os.getenv("JAXTYPING_CHECKS", "0") != "1":
        return

    try:
        import jaxtyping as jt  # type: ignore

        # Prefer the simple toggle if available in the installed jaxtyping.
        if hasattr(jt, "enable"):
            try:
                # Some versions accept `checks=True` or no args; be permissive.
                jt.enable(checks=True)  # type: ignore[call-arg]
                return
            except (TypeError, ValueError, RuntimeError):
                try:
                    jt.enable()  # type: ignore[misc]
                    return
                except (TypeError, ValueError, RuntimeError):
                    pass

        # Fallback: instrument our package only if a runtime checker is available.
        try:
            jt.install_import_hook("viterbo", "beartype.beartype")  # type: ignore[attr-defined]
        except (AttributeError, ImportError):
            # If the optional runtime checker isn't available, skip quietly.
            pass
    except ImportError:
        # If jaxtyping is unavailable for any reason, skip quietly.
        pass


_maybe_enable_jaxtyping_checks()


@pytest.fixture
def rng_key() -> Iterator[jax.Array]:
    """Deterministic JAX PRNG key for tests; split per-test if needed.

    Yields a base key; tests can call ``jax.random.split`` to derive subkeys.
    """
    yield jax.random.PRNGKey(0)


@pytest.fixture
def tol() -> dict[str, float]:
    """Default numeric tolerances used in tests."""
    return {"rtol": 1e-9, "atol": 0.0}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tag smoke/deep markers based on policy defaults."""

    smoke_marker = pytest.mark.smoke
    deep_marker = pytest.mark.deep
    for item in items:
        if item.get_closest_marker("longhaul"):
            continue
        if item.get_closest_marker("deep"):
            # Respect explicit deep marks; still ensure slow implies deep.
            continue
        if item.get_closest_marker("slow"):
            item.add_marker(deep_marker)
            continue
        # Auto-mark non-performance tests as smoke for CI defaults.
        if "tests/performance" in str(item.fspath):
            continue
        if not item.get_closest_marker("smoke"):
            item.add_marker(smoke_marker)
        if not item.get_closest_marker("timeout"):
            item.add_marker(pytest.mark.timeout(SMOKE_TIMEOUT_SECONDS, method="thread"))
