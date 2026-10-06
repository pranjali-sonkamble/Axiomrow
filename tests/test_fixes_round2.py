import numpy as np, pandas as pd
from backend.forecasting import forecast_dataframe
from backend.report_generator import (render_forecast_png, render_chart_config_png,
                                      generate_markdown_report)

def _monthly():
    return pd.DataFrame({"date": pd.date_range("2023-01-01", periods=36, freq="MS"),
                         "revenue": 1000 + np.arange(36) * 25.0,
                         "region": ["N", "S", "E"] * 12})

def test_forecast_png_renders():
    png = render_forecast_png(forecast_dataframe("predict next 6 months revenue", _monthly()))
    assert png and png[:4] == b"\x89PNG"

def test_chart_png_does_not_need_kaleido():
    png = render_chart_config_png(_monthly(), {"type": "bar", "x": "region", "y": "revenue",
                                               "aggregation": "sum", "title": "t"})
    assert png and png[:4] == b"\x89PNG"





def test_markdown_report_numbering_has_methodology_and_overview():
    md = generate_markdown_report(_monthly(), {"dataset_name": "x.csv"}, [], {}, {},
                                  include_charts=False)
    assert "## 1. Analysis Methodology" in md and "## 2. Dataset Overview" in md
