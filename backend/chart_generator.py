# backend/chart_generator.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def _generate_advanced_chart(df: pd.DataFrame, chart_config: dict, theme: dict = None):
    """Render deterministic chart configs that carry filters, top-N, date grain or a second group."""
    chart_type = str(chart_config.get("type", "bar")).lower()
    x_col = chart_config.get("x")
    y_col = chart_config.get("y")
    title = chart_config.get("title", "Chart")
    agg = str(chart_config.get("aggregation", "sum")).lower()
    orientation = str(chart_config.get("orientation", "vertical")).lower()
    group_by = chart_config.get("group_by")
    date_granularity = chart_config.get("date_granularity")
    top_n = chart_config.get("top_n")
    year = chart_config.get("date_filter_year")

    work = df.copy()
    if year is not None and x_col in work.columns:
        dates = pd.to_datetime(work[x_col], errors="coerce")
        work = work.loc[dates.dt.year == int(year)].copy()

    if x_col not in work.columns or (y_col and y_col not in work.columns):
        return None
    if group_by and group_by not in work.columns:
        return None
    if y_col:
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=[y_col, x_col])

    def aggregate(g):
        if agg in ("mean", "average", "avg"):
            return g.mean()
        if agg == "count":
            return g.count()
        if agg == "min":
            return g.min()
        if agg == "max":
            return g.max()
        return g.sum()

    # Date charts use a stable textual period axis after filtering.
    if date_granularity:
        dates = pd.to_datetime(work[x_col], errors="coerce")
        if date_granularity == "year":
            work["__chart_x__"] = dates.dt.year.astype("Int64").astype(str)
        else:
            work["__chart_x__"] = dates.dt.to_period("M").astype(str)
        x_plot = "__chart_x__"
    else:
        x_plot = x_col

    if group_by:
        grouped = work.groupby([x_plot, group_by], dropna=False)[y_col].apply(aggregate).reset_index(name=y_col)
        if chart_type == "bar":
            fig = px.bar(grouped, x=x_plot if orientation != "horizontal" else y_col,
                         y=y_col if orientation != "horizontal" else x_plot,
                         color=group_by, barmode="group", title=title,
                         template=(theme or {}).get("plotly_template", "plotly_white"),
                         color_discrete_sequence=(theme or {}).get("plotly_colors"))
        else:
            fig = px.line(grouped, x=x_plot, y=y_col, color=group_by, title=title,
                          markers=True, template=(theme or {}).get("plotly_template", "plotly_white"),
                          color_discrete_sequence=(theme or {}).get("plotly_colors"))
    else:
        grouped = work.groupby(x_plot, dropna=False)[y_col].apply(aggregate).reset_index(name=y_col)
        if top_n:
            grouped = grouped.sort_values(y_col, ascending=False).head(int(top_n))
        elif date_granularity:
            grouped = grouped.sort_values(x_plot)
        if chart_type == "pie":
            fig = px.pie(grouped, names=x_plot, values=y_col, title=title,
                         template=(theme or {}).get("plotly_template", "plotly_white"),
                         color_discrete_sequence=(theme or {}).get("plotly_colors"))
        elif chart_type == "line":
            fig = px.line(grouped, x=x_plot, y=y_col, title=title, markers=True,
                          template=(theme or {}).get("plotly_template", "plotly_white"),
                          color_discrete_sequence=(theme or {}).get("plotly_colors"))
        else:
            if orientation == "horizontal":
                fig = px.bar(grouped, x=y_col, y=x_plot, orientation="h", title=title,
                             template=(theme or {}).get("plotly_template", "plotly_white"),
                             color_discrete_sequence=(theme or {}).get("plotly_colors"))
            else:
                fig = px.bar(grouped, x=x_plot, y=y_col, title=title,
                             template=(theme or {}).get("plotly_template", "plotly_white"),
                             color_discrete_sequence=(theme or {}).get("plotly_colors"))
    return apply_layout(fig, (theme or {}).get("plotly_template", "plotly_white"), "rgba(0,0,0,0)")


def generate_chart(df: pd.DataFrame, chart_config: dict, theme: dict = None):
    """
    Main entry point. Takes a DataFrame, chart config dict, and optional
    theme dict. Returns (plotly_figure, error_string).
    """
    chart_type = chart_config.get("type", "bar").lower()
    if chart_type in ["hist"]:
        chart_type = "histogram"
    x_col = chart_config.get("x")
    y_col = chart_config.get("y")
    title = chart_config.get("title", "Chart")
    orientation = str(chart_config.get("orientation", "vertical")).lower()

    if any(k in chart_config for k in ("top_n", "date_granularity", "date_filter_year", "group_by")):
        try:
            advanced = _generate_advanced_chart(df, chart_config, theme=theme)
            if advanced is not None:
                return advanced, None
        except Exception as e:
            return None, f"Could not create chart: {str(e)}"

    error = validate_chart_config(df, chart_type, x_col, y_col)
    if error:
        return None, error

    template = theme["plotly_template"] if theme else "plotly_white"
    colors = theme["plotly_colors"] if theme else None
    paper_bg = "rgba(0,0,0,0)"

    try:
        if chart_type == "bar":
            fig = make_bar_chart(df, x_col, y_col, title, template, colors, orientation)
        elif chart_type == "line":
            fig = make_line_chart(df, x_col, y_col, title, template, colors)
        elif chart_type == "scatter":
            fig = make_scatter_chart(df, x_col, y_col, title, template, colors)
        elif chart_type == "pie":
            fig = make_pie_chart(df, x_col, y_col, title, template, colors)
        elif chart_type == "histogram":
            fig = make_histogram(df, x_col, title, template, colors)
        elif chart_type == "box":
            fig = make_box_chart(df, x_col, y_col, title, template, colors)
        else:
            fig = make_bar_chart(df, x_col, y_col, title, template, colors)

        fig = apply_layout(fig, template, paper_bg)
        return fig, None

    except Exception as e:
        return None, f"Could not create chart: {str(e)}"


def validate_chart_config(df, chart_type: str, x_col, y_col) -> str:
    """
    Returns a friendly error string, or None if the request is valid.
    Catches the failure modes that used to surface as raw, confusing
    Plotly/pandas tracebacks: missing columns, empty data after
    dropping nulls, and wrong data types for the requested chart type.
    """

    def _column_error(col_name):
        close = [c for c in df.columns if col_name.lower() in c.lower()]
        suggestion = f" Did you mean '{close[0]}'?" if close else ""
        available = ", ".join(df.columns[:8])
        return f"Column '{col_name}' not found in the dataset.{suggestion} Available columns: {available}"

    if x_col and x_col not in df.columns:
        return _column_error(x_col)
    if y_col and y_col not in df.columns:
        return _column_error(y_col)

    # Chart-type-specific dtype requirements
    if chart_type in ("bar", "line", "pie", "box") and y_col:
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            return (f"'{y_col}' isn't a numeric column, so it can't be used as the "
                    f"value axis for a {chart_type} chart. Try a numeric column instead.")

    if chart_type == "scatter":
        if x_col and not pd.api.types.is_numeric_dtype(df[x_col]):
            return f"Scatter plots need numeric columns — '{x_col}' isn't numeric."
        if y_col and not pd.api.types.is_numeric_dtype(df[y_col]):
            return f"Scatter plots need numeric columns — '{y_col}' isn't numeric."

    if chart_type == "histogram" and x_col:
        if not pd.api.types.is_numeric_dtype(df[x_col]):
            return f"Histograms need a numeric column — '{x_col}' isn't numeric. Try a count or value column."

    # Empty-after-dropna guard — avoids a confusing blank chart
    cols_to_check = [c for c in (x_col, y_col) if c]
    if cols_to_check:
        non_null_rows = df[cols_to_check].dropna()
        if len(non_null_rows) == 0:
            return f"No rows have data in both '{x_col}' and '{y_col}' — nothing to plot."

    return None


def make_bar_chart(df, x_col, y_col, title, template, colors, orientation="vertical"):
    # Same version-agnostic check as data_loader.py — pandas 3.0+
    # uses a native "str" dtype instead of "object" for text columns.
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        plot_df = df.groupby(x_col)[y_col].sum().reset_index()
        plot_df = plot_df.sort_values(y_col, ascending=False)
    else:
        plot_df = df[[x_col, y_col]].dropna()

    if orientation == "horizontal":
        fig = px.bar(
            plot_df, x=y_col, y=x_col, title=title,
            orientation="h",
            template=template,
            color_discrete_sequence=colors or px.colors.qualitative.Set2
        )
    else:
        fig = px.bar(
            plot_df, x=x_col, y=y_col, title=title,
            template=template,
            color_discrete_sequence=colors or px.colors.qualitative.Set2
        )
    return fig


MAX_PLOT_POINTS = 5000  # browsers struggle rendering far more raw SVG points/markers than this


def _sample_for_plotting(plot_df: pd.DataFrame, title: str, preserve_order: bool = False):
    """
    Caps the number of rendered points on very large datasets — Plotly's
    default SVG renderer visibly struggles well before 100K+ points,
    even though generating the figure in Python itself is fast. Returns
    (possibly-sampled dataframe, possibly-annotated title) so the user
    is told when they're looking at a sample, not silently shown one.

    preserve_order=True uses even-interval downsampling instead of
    random sampling — important for line charts, where random sampling
    would distort the visual shape of a trend.
    """
    total = len(plot_df)
    if total <= MAX_PLOT_POINTS:
        return plot_df, title

    if preserve_order:
        step = total // MAX_PLOT_POINTS
        sampled = plot_df.iloc[::step]
    else:
        sampled = plot_df.sample(n=MAX_PLOT_POINTS, random_state=42)

    annotated_title = f"{title} (sample of {len(sampled):,} of {total:,} points)"
    return sampled, annotated_title


def make_line_chart(df, x_col, y_col, title, template, colors):
    plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)
    plot_df, title = _sample_for_plotting(plot_df, title, preserve_order=True)
    fig = px.line(
        plot_df, x=x_col, y=y_col, title=title,
        template=template, markers=True,
        color_discrete_sequence=colors or px.colors.qualitative.Set2
    )
    return fig


def make_scatter_chart(df, x_col, y_col, title, template, colors):
    plot_df = df[[x_col, y_col]].dropna()
    plot_df, title = _sample_for_plotting(plot_df, title, preserve_order=False)
    try:
        # Trendline requires the optional 'statsmodels' package.
        # If it isn't installed, fall back to a plain scatter instead
        # of crashing the whole chart (this was silently failing every
        # scatter request before this fix).
        fig = px.scatter(
            plot_df, x=x_col, y=y_col, title=title,
            template=template, trendline="ols",
            color_discrete_sequence=colors or px.colors.qualitative.Set2
        )
    except (ImportError, ModuleNotFoundError, ValueError):
        fig = px.scatter(
            plot_df, x=x_col, y=y_col, title=title,
            template=template,
            color_discrete_sequence=colors or px.colors.qualitative.Set2
        )
    return fig


def make_pie_chart(df, x_col, y_col, title, template, colors):
    plot_df = df.groupby(x_col)[y_col].sum().reset_index()
    plot_df = plot_df.sort_values(y_col, ascending=False)
    if len(plot_df) > 8:
        top = plot_df.head(7)
        other_sum = plot_df.iloc[7:][y_col].sum()
        other_row = pd.DataFrame({x_col: ["Other"], y_col: [other_sum]})
        plot_df = pd.concat([top, other_row], ignore_index=True)
    fig = px.pie(
        plot_df, names=x_col, values=y_col, title=title,
        template=template,
        color_discrete_sequence=colors or px.colors.qualitative.Set2
    )
    return fig


def make_histogram(df, x_col, title, template, colors):
    plot_df = df[[x_col]].dropna()
    fig = px.histogram(
        plot_df, x=x_col, title=title,
        template=template, nbins=30,
        color_discrete_sequence=colors or px.colors.qualitative.Set2
    )
    return fig


def make_box_chart(df, x_col, y_col, title, template, colors):
    plot_df = df[[x_col, y_col]].dropna()
    fig = px.box(
        plot_df, x=x_col, y=y_col, title=title,
        template=template,
        color_discrete_sequence=colors or px.colors.qualitative.Set2
    )
    return fig


def apply_layout(fig, template: str, paper_bg: str):
    """Applies consistent layout to all charts — styling only. Does
    NOT set the title text: every make_*_chart() function already
    sets it correctly via its px call (including the sampling
    annotation added by _sample_for_plotting for large scatter/line
    charts). Re-setting it here from a stale outer variable used to
    silently overwrite that annotation."""
    fig.update_layout(
        title={"x": 0.5, "xanchor": "center", "font": {"size": 16, "weight": "bold"}},
        margin=dict(t=70, b=40, l=40, r=40),
        font=dict(family="sans-serif", size=13),
        paper_bgcolor=paper_bg,
        plot_bgcolor=paper_bg,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                    xanchor="center", x=0.5),
    )
    return fig