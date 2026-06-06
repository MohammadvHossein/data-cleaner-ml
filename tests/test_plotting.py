"""Tests for plotting module (without matplotlib)."""

import pandas as pd
import pytest

from clean_data_ml import DataCleaner
from clean_data_ml.plotting import _HAS_MPL, plot_null_report


def test_plot_null_report_noop_without_mpl():
    """Should not crash when matplotlib is absent."""
    if _HAS_MPL:
        pytest.skip("matplotlib is installed — skipping no-mpl test")
    dc = DataCleaner()
    dc.load_df(pd.DataFrame({"a": [1, None]}))
    # Should log a warning and return without error
    result = plot_null_report(dc)
    assert result is None


def test_plot_null_report_no_data():
    from clean_data_ml.plotting import plot_null_report

    dc = DataCleaner()
    result = plot_null_report(dc)
    assert result is None


def test_plot_null_report_no_nulls():
    from clean_data_ml.plotting import plot_null_report

    dc = DataCleaner()
    dc.load_df(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = plot_null_report(dc)
    assert result is None
