import os
import tempfile

import pytest

from viterbo.exp1 import store


@pytest.mark.goal_code
@pytest.mark.smoke
def test_ensure_and_scan_smoke() -> None:
    """Imperative storage functions exist; scan returns a lazy object.

    Covers: ensure_dataset, append_rows (no-op), and scan_lazy contract.
    """

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dataset.parquet")
        store.ensure_dataset(path)
        # Append zero rows should be a no-op once implemented.
        store.append_rows(path, [])
        lf = store.scan_lazy(path)
        # Expect a lazy frame-like object exposing collect/select in implementation.
        assert hasattr(lf, "collect"), "scan_lazy() should return a lazy, collectable frame"
