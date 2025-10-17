from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def _set_default_torch_dtype() -> None:
    torch.set_default_dtype(torch.float64)

