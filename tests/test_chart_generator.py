"""Tests for backend/chart_generator.py — validation and generation
for every supported chart type, including the dtype-checking that
used to surface as raw Plotly tracebacks."""

import pytest
import pandas as pd
from backend.chart_generator import generate_chart, validate_chart_config


THEME = {"plotly_template": "plotly_white", "plotly_colors": ["#2563EB", "#7C3AED"]}


class TestValidation:
    def test_missing_column_gives_friendly_error(self, sample_df):
        fig, err = generate_chart(sample_df, {"type": "bar", "x": "produt", "y": "revenue"}, theme=THEME)
        assert fig is None
        assert "not found" in err
        assert "product" in err  # suggestion should surface the real column name

    def test_non_numeric_y_for_bar_rejected(self, sample_df):
        fig, err = generate_chart(sample_df, {"type": "bar", "x": "product", "y": "region"}, theme=THEME)
        assert fig is None
        assert "numeric" in err.lower()

    def test_non_numeric_column_for_scatter_rejected(self, sample_df):
        fig, err = generate_chart(sample_df, {"type": "scatter", "x": "product", "y": "revenue"}, theme=THEME)
        assert fig is None
        assert "numeric" in err.lower()

    def test_non_numeric_column_for_histogram_rejected(self, sample_df):
        fig, err = generate_chart(sample_df, {"type": "histogram", "x": "product"}, theme=THEME)
        assert fig is None
        assert "numeric" in err.lower()

    def test_valid_bar_chart_passes(self, sample_df):
        fig, err = generate_chart(sample_df, {"type": "bar", "x": "product", "y": "revenue"}, theme=THEME)
        assert err is None
        assert fig is not None


class TestEachChartTypeGenerates:
    """Every supported chart type should produce a figure with no error
    given valid, well-typed columns."""

    @pytest.mark.parametrize("config", [
        {"type": "bar", "x": "product", "y": "revenue"},
        {"type": "line", "x": "date", "y": "revenue"},
        {"type": "scatter", "x": "quantity", "y": "revenue"},
        {"type": "pie", "x": "region", "y": "revenue"},
        {"type": "histogram", "x": "revenue"},
        {"type": "box", "x": "region", "y": "revenue"},
    ])
    def test_chart_type_generates_without_error(self, sample_df, config):
        fig, err = generate_chart(sample_df, {**config, "title": "Test"}, theme=THEME)
        assert err is None, f"{config['type']} chart failed: {err}"
        assert fig is not None

    def test_hist_alias_normalizes_to_histogram(self, sample_df):
        fig, err = generate_chart(sample_df, {"type": "hist", "x": "revenue"}, theme=THEME)
        assert err is None
        assert fig is not None


class TestScatterTrendlineFallback:
    def test_scatter_works_even_without_statsmodels(self, sample_df, monkeypatch):
        # Simulate statsmodels being unavailable by forcing the trendline
        # branch to raise, confirming the fallback plain-scatter path works
        import backend.chart_generator as cg

        original = cg.px.scatter
        call_count = {"n": 0}

        def flaky_scatter(*args, **kwargs):
            call_count["n"] += 1
            if kwargs.get("trendline"):
                raise ModuleNotFoundError("no statsmodels")
            return original(*args, **kwargs)

        monkeypatch.setattr(cg.px, "scatter", flaky_scatter)
        fig, err = generate_chart(sample_df, {"type": "scatter", "x": "quantity", "y": "revenue"}, theme=THEME)
        assert err is None
        assert fig is not None
        assert call_count["n"] == 2  # first call raised, fallback call succeeded


class TestLargeDatasetSampling:
    """Covers the fix found while stress-testing on a 300K-row dataset:
    scatter/line charts used to plot every raw point with no cap,
    which is fast in Python but would make a browser struggle."""

    def test_scatter_samples_down_large_datasets(self):
        import numpy as np
        rng = np.random.default_rng(0)
        big_df = pd.DataFrame({"x": rng.standard_normal(50_000), "y": rng.standard_normal(50_000)})
        fig, err = generate_chart(big_df, {"type": "scatter", "x": "x", "y": "y", "title": "Test"}, theme=THEME)
        assert err is None
        assert len(fig.data[0].x) <= 5000
        assert "sample of" in fig.layout.title.text

    def test_small_scatter_is_not_sampled_or_annotated(self, sample_df):
        fig, err = generate_chart(sample_df, {"type": "scatter", "x": "quantity", "y": "revenue", "title": "Test"}, theme=THEME)
        assert err is None
        assert len(fig.data[0].x) == len(sample_df)
        assert fig.layout.title.text == "Test"  # no sampling annotation added

    def test_line_chart_sampling_preserves_chronological_order(self):
        import numpy as np
        big_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=20_000, freq="h").astype(str),
            "value": np.random.default_rng(0).standard_normal(20_000),
        })
        fig, err = generate_chart(big_df, {"type": "line", "x": "date", "y": "value", "title": "Trend"}, theme=THEME)
        assert err is None
        x_values = list(fig.data[0].x)
        assert x_values == sorted(x_values), "sampling must not shuffle a time series"


class TestEmptyAfterFilter:
    def test_empty_result_after_dropna_gives_friendly_message(self):
        # Both columns must be genuinely numeric (float, all-NaN) so the
        # request passes dtype validation and reaches the dropna-empty
        # check specifically — an all-None column without an explicit
        # numeric dtype gets correctly rejected earlier as "not numeric",
        # which is the right precedence, just not what this test targets.
        df = pd.DataFrame({"x": pd.Series([None, None], dtype=float),
                            "y": pd.Series([None, None], dtype=float)})
        fig, err = generate_chart(df, {"type": "bar", "x": "x", "y": "y"}, theme=THEME)
        assert fig is None
        assert "nothing to plot" in err.lower()
