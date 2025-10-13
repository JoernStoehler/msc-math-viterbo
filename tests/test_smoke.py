import pytest
import torch
from torch.utils.data import DataLoader

from viterbo.datasets.core import RaggedPointsDataset, collate_list, collate_pad
from viterbo.math.polytope import (
    bounding_box,
    halfspace_violations,
    pairwise_squared_distances,
    support,
    support_argmax,
)
from viterbo.models.demo import run_probe

pytestmark = pytest.mark.smoke


def test_imports_and_basic_geometry():
    """Ensure core modules import and basic geometry works (support, distances)."""
    pts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([1.0, 1.0])
    s = support(pts, d)
    assert torch.is_tensor(s)
    assert pytest.approx(float(s)) == 1.0

    d2 = pairwise_squared_distances(pts)
    assert d2.shape == (2, 2)
    assert pytest.approx(float(d2[0, 1])) == 2.0

    val, idx = support_argmax(pts, d)
    assert idx in (0, 1) and pytest.approx(float(val)) == 1.0

    mins, maxs = bounding_box(pts)
    assert mins.shape == maxs.shape == (2,)
    assert float(mins.min()) <= float(maxs.max())

    normals = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    offsets = torch.tensor([1.0, 1.0])
    viol = halfspace_violations(pts, normals, offsets)
    assert viol.shape == (2, 2)
    assert torch.all(viol >= 0)


def test_datasets_and_collate_and_models(tmp_path):
    """Ragged dataset + collate functions + demo probe produce a scalar metric."""
    ds = RaggedPointsDataset(num_samples=8, dim=3, min_points=2, max_points=5, seed=123)

    # List collate
    loader_list = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_list)
    out_list = run_probe(loader_list)
    assert "avg_support" in out_list and isinstance(out_list["avg_support"], float)

    # Padded collate
    loader_pad = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_pad)
    out_pad = run_probe(loader_pad)
    assert "avg_support" in out_pad and isinstance(out_pad["avg_support"], float)
