"""Placeholder workflow for inspecting the modern atlas artefact.

The real consumer notebook will exercise :mod:`viterbo.atlas` helpers
once the builder pipeline produces artefacts. Until then we surface the intended
entry points and keep the script runnable for documentation and smoke checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

try:  # Imported for feature detection only; execution remains a stub.
    from viterbo import atlas as modern_atlas
    from viterbo import capacity as modern_capacity
except ImportError:  # pragma: no cover - defensive guard for docs builds.
    modern_atlas = None  # type: ignore[assignment]
    modern_capacity = None  # type: ignore[assignment]


ATLAS_PATH = Path("artefacts/modern_atlas.parquet")


@dataclass(frozen=True)
class PlaceholderAction:
    """Describe a consumer action that will be implemented later."""

    slug: str
    summary: str
    pending_apis: Sequence[str]


def _atlas_snapshot() -> pl.DataFrame | None:
    """Load the atlas snapshot if present; returns ``None`` when missing."""

    if not ATLAS_PATH.exists():
        return None
    return pl.read_parquet(ATLAS_PATH)


def _actions() -> Iterable[PlaceholderAction]:
    """Yield the intended consumer actions for future implementation."""

    yield PlaceholderAction(
        slug="schema",
        summary=(
            "Inspect ``atlas.atlas_pl_schema`` to confirm the builder emitted the "
            "expected columns and dtypes."
        ),
        pending_apis=("atlas.atlas_pl_schema",),
    )
    yield PlaceholderAction(
        slug="polytope",
        summary=(
            "Convert a dataframe row into a ``Polytope`` instance and feed it into "
            "the modern capacity solvers for verification."
        ),
        pending_apis=("atlas.as_polytope", "capacity.ehz_capacity_reference"),
    )
    yield PlaceholderAction(
        slug="filters",
        summary=(
            "Apply reproducible filters (dimension, symmetry class, diagnostics) to "
            "prepare experiment-ready subsets."
        ),
        pending_apis=("atlas.filter_frame",),
    )


def describe_actions() -> list[PlaceholderAction]:
    """Return the placeholder actions for external documentation or tests."""

    return list(_actions())


def main() -> None:
    """Emit a human-readable outline of the pending atlas consumer."""

    snapshot = _atlas_snapshot()
    print("Modern atlas consumer placeholder")
    if snapshot is None:
        print(f"No atlas snapshot found at {ATLAS_PATH}")
    else:
        print(f"Loaded atlas snapshot with {snapshot.height} rows and {snapshot.width} columns")

    module_status = {
        "viterbo.atlas": modern_atlas is not None,
        "viterbo.capacity": modern_capacity is not None,
    }
    print("Module availability:")
    for module_name, present in module_status.items():
        flag = "available" if present else "missing"
        print(f"  - {module_name}: {flag}")

    print("\nPlanned consumer actions:")
    for action in _actions():
        apis = ", ".join(action.pending_apis)
        print(f"  · [{action.slug}] {action.summary}")
        print(f"      pending APIs: {apis}")


if __name__ == "__main__":
    main()
