import pytest

from viterbo.exp1 import store as rows_mod


@pytest.mark.goal_code
@pytest.mark.smoke
def test_log_row_mvp_keys_present() -> None:
    """log_row returns a dict with MVP schema keys.

    Covers: row assembly for current quantities and representations.
    """

    dummy_poly = object()
    row = rows_mod.log_row(
        dummy_poly,
        quantities={
            "polytope_id": "p-000",
            "generator": "products.km_xtk",
            "dimension": 4,
            "volume": 1.0,
            "capacity_ehz": 1.0,
            "systolic_ratio": 1.0,
            "min_action_orbit": [0, 1, 2, 3],
        },
    )
    for key in (
        "polytope_id",
        "generator",
        "dimension",
        "volume",
        "capacity_ehz",
        "systolic_ratio",
        "min_action_orbit",
    ):
        assert key in row, f"missing key in row: {key}"
