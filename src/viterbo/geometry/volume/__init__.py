"""Euclidean volume estimators exposing reference and fast variants."""

from viterbo.geometry.volume import fast as _fast
from viterbo.geometry.volume import reference as _reference
from viterbo.geometry.volume import samples as _samples

polytope_volume_reference = _reference.polytope_volume
polytope_volume_fast = _fast.polytope_volume
hypercube_volume_inputs = _samples.hypercube_volume_inputs
