"""Smoke tests for the test metadata reporting CLI."""

from __future__ import annotations

import textwrap

import pytest
from scripts import report_test_metadata


@pytest.mark.goal_code
@pytest.mark.smoke
def test_report_test_metadata_outputs_deterministic_summary(tmp_path, capsys):
    """Reporter prints deterministic listing with goal marker and docstring summary."""
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        textwrap.dedent(
            """
            import pytest


            @pytest.mark.goal_math
            def test_example():
                '''Check docstring summary.'''
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = report_test_metadata.main([str(tmp_path)])
    captured = capsys.readouterr().out.strip().splitlines()

    assert exit_code == 0
    assert captured[0] == "LINES:1"
    assert captured[1] == f"{test_file}::test_example [goal_math] - Check docstring summary."


@pytest.mark.goal_code
@pytest.mark.smoke
def test_report_test_metadata_filters_markers(tmp_path, capsys):
    """Reporter filters tests when marker selector is provided."""
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        textwrap.dedent(
            """
            import pytest


            @pytest.mark.goal_code
            def test_example_code():
                '''Code test.'''


            @pytest.mark.goal_math
            def test_example_math():
                '''Math test.'''
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = report_test_metadata.main(["--marker", "goal_math", str(tmp_path)])
    captured = capsys.readouterr().out.strip().splitlines()

    assert exit_code == 0
    assert captured[0] == "LINES:1"
    assert captured[1] == f"{test_file}::test_example_math [goal_math] - Math test."
