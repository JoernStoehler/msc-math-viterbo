import pytest

try:
    import pytest_benchmark  # noqa: F401
except ImportError:  # pragma: no cover
    pytestmark = pytest.mark.skip(reason="pytest-benchmark not installed")
else:
    pytestmark = [pytest.mark.smoke, pytest.mark.benchmark]


def test_atlas_tiny_build_benchmark(benchmark, request) -> None:
    """Micro-benchmark atlas_tiny_build() and validate basic invariants.

    - Avoid strict thresholds to keep CI non-flaky.
    - Validate minimal schema invariants and timing fields.
    """
    from viterbo.datasets.atlas_tiny import atlas_tiny_build

    # Run only under explicit benchmark invocation to avoid slowing smoke tests.
    try:
        bench_only = bool(request.config.getoption("--benchmark-only"))
    except (AttributeError, ValueError):
        bench_only = False
    if not bench_only:
        pytest.skip("run via just bench (passes --benchmark-only)")

    rows = benchmark(atlas_tiny_build)

    # Basic structure and roster sanity
    assert isinstance(rows, list)
    assert len(rows) > 0

    for row in rows:
        # Geometry present and non-empty
        assert row["vertices"].ndim == 2 and row["vertices"].size(0) > 0
        assert row["normals"].ndim == 2 and row["normals"].size(0) > 0

        # Volume is scalar; dimension limited to {2, 4} for this roster
        assert getattr(row["volume"], "ndim", None) == 0
        assert row["dimension"] in (2, 4)

        # Timings: executed >= 0, non-executed are None
        assert isinstance(row["time_generator"], float) and row["time_generator"] >= 0.0

        t_area = row["time_volume_area2d"]
        t_facets = row["time_volume_facets"]
        if row["dimension"] == 2:
            assert isinstance(t_area, float) and t_area >= 0.0
            assert t_facets is None
        else:
            assert isinstance(t_facets, float) and t_facets >= 0.0
            assert t_area is None

        t_cap_area = row["time_capacity_area2d"]
        t_cap_mink = row["time_capacity_minkowski_lp3"]

        if row["capacity_ehz"] is not None:
            # If capacity computed, systolic must be computed and timed as well
            t_sys = row["time_systolic_ratio"]
            assert isinstance(t_sys, float) and t_sys >= 0.0

            if row["dimension"] == 2:
                assert isinstance(t_cap_area, float) and t_cap_area >= 0.0
                assert t_cap_mink is None
            elif row["dimension"] == 4:
                assert isinstance(t_cap_mink, float) and t_cap_mink >= 0.0
        else:
            assert t_cap_area is None
            assert t_cap_mink is None
            assert row["time_systolic_ratio"] is None
