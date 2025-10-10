"""Validate that generator stubs currently raise NotImplementedError."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.modern import basic_generators
from viterbo.modern.types import GeneratorMetadata, PolytopeBundle


@pytest.mark.goal_code
@pytest.mark.smoke
def test_generator_stubs_raise_not_implemented() -> None:
    """Every generator surface should raise until we provide real logic."""

    key = jnp.zeros((2,), dtype=jnp.uint32)
    bundle = PolytopeBundle(halfspaces=None, vertices=None)
    metadata = GeneratorMetadata(identifier="dummy", parameters={})

    with pytest.raises(NotImplementedError):
        next(iter(basic_generators.sample_uniform_ball(key, 3, num_samples=1)))

    product_input = [(bundle, metadata)]
    with pytest.raises(NotImplementedError):
        next(iter(basic_generators.sample_product(product_input, product_input)))

    bounds = jnp.zeros((3, 2))
    with pytest.raises(NotImplementedError):
        next(iter(basic_generators.enumerate_lattice(3, bounds)))
