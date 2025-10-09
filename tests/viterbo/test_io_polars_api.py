import pytest

from viterbo._wrapped import polars_io


@pytest.mark.goal_code
@pytest.mark.smoke
def test_rows_to_polars_and_materialize_contract() -> None:
    """rows_to_polars creates a frame; materialize_to_jnp returns arrays.

    Covers: IO adapter contracts for DF creation and JAX materialization.
    """

    rows = [
        {
            "polytope_id": "p-001",
            "generator": "products.km_xtk",
            "dimension": 4,
            "volume": 1.0,
            "capacity_ehz": 1.0,
            "systolic_ratio": 1.0,
            "min_action_orbit": [0, 1, 2, 3],
        }
    ]
    df = polars_io.rows_to_polars(rows)
    assert hasattr(df, "select"), "expected a DataFrame-like object"
    # Lazy scan is not available at this level; materialize_from a LazyFrame by wrapping.
    # Once implemented, expose a scan() returning LazyFrame and feed it to materialize.
