"""Euclidean volume estimators grouped by implementation strategy."""

from viterbo.geometry.volume import jax as _jax_impl
from viterbo.geometry.volume import optimized as _optimized_impl
from viterbo.geometry.volume import reference as _reference_impl
from viterbo.geometry.volume import samples as _samples

polytope_volume_reference = _reference_impl.polytope_volume
polytope_volume_optimized = _optimized_impl.polytope_volume
polytope_volume_jax = _jax_impl.polytope_volume

hypercube_volume_inputs = _samples.hypercube_volume_inputs

# Backwards-compatible aliases retained for downstream callers.
polytope_volume_fast = polytope_volume_optimized
