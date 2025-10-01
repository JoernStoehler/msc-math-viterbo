"""Tests for the public greeting helper."""

from __future__ import annotations

import numpy as np

from viterbo.hello import hello_numpy


def test_hello_numpy_formats_message() -> None:
    message = hello_numpy("Researcher", np.array([1.0, 1.0]))
    assert message.startswith("Hello, Researcher!")
    assert "Unit sample: [0.707" in message
