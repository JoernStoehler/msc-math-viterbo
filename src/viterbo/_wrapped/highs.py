"""Thin wrapper around :mod:`highspy` to centralise optional imports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HighsResources:
    """Expose the HiGHS classes required by callers."""

    Highs: type
    HighsStatus: type
    HighsModelStatus: type
    HighsVarType: type


def load_highs() -> HighsResources:
    """Import :mod:`highspy` lazily and return the relevant classes."""

    from highspy import Highs, HighsModelStatus, HighsStatus, HighsVarType

    return HighsResources(
        Highs=Highs,
        HighsStatus=HighsStatus,
        HighsModelStatus=HighsModelStatus,
        HighsVarType=HighsVarType,
    )
