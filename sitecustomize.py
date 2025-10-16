"""Repository-wide startup configuration for Python interpreters.

This module is discovered automatically (via ``PYTHONPATH``) and lets us
enforce lightweight guardrails such as per-process CPU limits and default
runtime flags for optional dependencies.
"""

from __future__ import annotations

import os
import warnings

# ---------------------------------------------------------------------------
# CPU time budget
#
# We cap the total CPU time available to any Python process so accidental
# runaway jobs terminate automatically instead of consuming hours of compute.
# The default cap can be tuned via ``VITERBO_CPU_LIMIT`` (seconds).  A value
# of 0 disables the guard.
# ---------------------------------------------------------------------------

try:
    import resource
except ImportError:  # pragma: no cover - e.g. Windows
    resource = None  # type: ignore[assignment]

_cpu_budget_env = os.getenv("VITERBO_CPU_LIMIT")
if _cpu_budget_env is not None:
    try:
        _cpu_budget = int(_cpu_budget_env)
    except ValueError:
        warnings.warn(
            "Ignoring invalid VITERBO_CPU_LIMIT value; expected integer seconds.",
            RuntimeWarning,
            stacklevel=1,
        )
        _cpu_budget = 0
else:
    _cpu_budget = 180  # seconds

if resource is not None and _cpu_budget > 0:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        target_soft = _cpu_budget
        target_hard = _cpu_budget
        if hard != resource.RLIM_INFINITY:
            target_soft = min(target_soft, hard)
            target_hard = min(target_hard, hard)
        if soft == resource.RLIM_INFINITY or soft > target_soft:
            resource.setrlimit(resource.RLIMIT_CPU, (target_soft, target_hard))
    except OSError:
        # Some environments (e.g. constrained containers) may forbid tightening
        # the limit; fail silently so we do not mask legitimate imports.
        pass

# ---------------------------------------------------------------------------
# Optional dependency defaults
#
# For JAX we enable float64 precision by default to avoid subtle dtype bugs.
# ---------------------------------------------------------------------------

if os.getenv("VITERBO_NO_SITE_CUSTOMIZE") != "1":
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
    except ModuleNotFoundError:
        pass
    except Exception:  # noqa: BLE001
        # Other import-time failures should not block startup; surface them to aid
        # debugging without stopping the interpreter completely.
        warnings.warn("sitecustomize failed to configure JAX", RuntimeWarning, stacklevel=1)
