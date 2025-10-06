from __future__ import annotations

import jax.numpy as jnp
import pytest


@pytest.fixture()
def unit_disk_vertices() -> jnp.ndarray:
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 128, endpoint=False)
    vertices = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=1)
    return jnp.asarray(vertices, dtype=jnp.float64)
