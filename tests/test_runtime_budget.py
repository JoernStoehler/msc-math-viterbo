from __future__ import annotations

import signal
import time

import pytest

from viterbo.runtime import TimeBudgetExceededError, _default_timeout, enforce_time_budget

pytestmark = pytest.mark.smoke


def test_signal_path_timeout_raises() -> None:
    if not (hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")):
        pytest.skip("Signal timers not supported on this platform")

    @enforce_time_budget(0.02)
    def sleepy() -> None:
        time.sleep(0.2)

    with pytest.raises(TimeBudgetExceededError):
        sleepy()


def test_fallback_path_post_execution_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a platform without setitimer/SIGALRM so the decorator picks the fallback path
    monkeypatch.delattr(signal, "setitimer", raising=False)
    monkeypatch.delattr(signal, "SIGALRM", raising=False)

    @enforce_time_budget(0.01)
    def sleepy() -> None:
        time.sleep(0.05)

    with pytest.raises(TimeBudgetExceededError):
        sleepy()


def test_env_var_invalid_timeout_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Invalid value should fall back to the 30.0s default
    monkeypatch.setenv("VITERBO_SOLVER_TIMEOUT", "not-a-number")
    assert _default_timeout(None) == 30.0
