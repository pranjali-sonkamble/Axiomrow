# backend/chart_png.py
"""
Kaleido-free PNG rendering for charts (used by chat history + reports).

Takes the SAME chart_config dict that generate_chart() uses and renders a
PNG with matplotlib. No Chrome / kaleido needed, so it works on Streamlit
Cloud. Uses matplotlib's Figure object directly (not pyplot) so it is safe
inside Streamlit's threads.

Returns (png_bytes, error_string) - exactly one is None.
"""

import io

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

_DEFAULT_COLORS = ["#2563EB", "#7C3AED", "#1E40AF", "#60A5FA", "#334155"]
_AGGS = {"sum": "sum", "mean": "mean", "average": "mean", "avg": "mean",
         "count": "count", "min": "min", "max": "max"}
_MAX_POINTS = 5000


def _fmt_num(v, _pos=None):
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:.1f}B"
    if a >= 1e6:
        return f"{v/1e6:.1f}M"
    if a >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{v:g}"


def chart_config_to_png(df: pd.DataFrame, cfg: dict, theme: dict = None,
                        width: float = 8.0, height: float = 4.8, dpi: int = 130):
    try:
        return _render(df, cfg or {}, theme, width, height, dpi), None
    except Exception as e:  # never let a chart break a report
        return None, f"Could not render chart image: {e}"


def _render(df, cfg, theme, width, height, dpi) -> bytes:
    ctype = str(cfg.get("type", "bar")).lower()
    if ctype == "hist":
        ctype = "histogram"
    x, y = cfg.get("x"), cfg.get("y")
    title = cfg.get("title") or "Chart"
    agg = _AGGS.get(str(cfg.get("aggregation", "sum")).lower(), "sum")
    group_by = cfg.get("group_by")
    top_n = cfg.get("top_n")
    gran = cfg.get("date_granularity")
    year = cfg.get("date_filter_year")
    horizontal = str(cfg.get("orientation", "vertical")).lower() == "horizontal"
    colors = (theme or {}).get("plotly_colors") or _DEFAULT_COLORS

    if x not in df.columns:
        raise ValueError(f"column '{x}' not found")
    if y and y not in df.columns:
        raise ValueError(f"column '{y}' not found")
    if group_by and group_by not in df.columns:
        raise ValueError(f"column '{group_by}' not found")

    work = df
    xcol = x

    # Date handling (mirrors chart_generator._generate_advanced_chart)
    if year is not None or gran:
        dates = pd.to_datetime(work[x], errors="coerce")
        if year is not None:
            work = work.loc[dates.dt.year == int(year)]
            dates = dates.loc[work.index]
        if gran:
            label = (dates.dt.year.astype("Int64").astype(str) if gran == "year"
                     else dates.dt.to_period("M").astype(str))
            work = work.assign(__x__=label)
            xcol = "__x__"

    fig = Figure(figsize=(width, height), dpi=dpi)
    ax = fig.subplots()

    if ctype == "histogram":
        vals = pd.to_numeric(work[x], errors="coerce").dropna()
        ax.hist(vals, bins=30, color=colors[0], edgecolor="white")
        ax.set_xlabel(x)
        ax.set_ylabel("Count")
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_num))

    elif ctype == "scatter":
        pts = work[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pts) > _MAX_POINTS:
            pts = pts.sample(_MAX_POINTS, random_state=42)
            title += f" (sample of {_MAX_POINTS:,})"
        ax.scatter(pts[x], pts[y], s=8, alpha=.5, color=colors[0])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_num))
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_num))

    elif ctype == "box":
        tmp = work[[x, y]].copy()
        tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
        tmp = tmp.dropna()
        cats = tmp[x].value_counts().head(15).index.tolist()
        data = [tmp.loc[tmp[x] == c, y].values for c in cats]
        ax.boxplot(data, tick_labels=[str(c) for c in cats]) \
            if _supports_tick_labels() else ax.boxplot(data, labels=[str(c) for c in cats])
        ax.set_ylabel(y)
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_num))

    else:  # bar / line / pie
        if not y:
            raise ValueError("a value column (y) is required")
        w = work.assign(**{y: pd.to_numeric(work[y], errors="coerce")}).dropna(subset=[y, xcol])

        if group_by and ctype != "pie":
            piv = w.groupby([xcol, group_by], dropna=False)[y].agg(agg).unstack(group_by)
            piv = piv.sort_index()
            if ctype == "line":
                piv.plot(ax=ax, marker="o", color=colors * 5)
            else:
                piv.plot(kind="barh" if horizontal else "bar", ax=ax, color=colors * 5, width=.8)
            ax.legend(title=group_by, frameon=False, fontsize=8)
        else:
            s = w.groupby(xcol, dropna=False)[y].agg(agg)
            if top_n:
                s = s.sort_values(ascending=False).head(int(top_n))
            elif gran:
                s = s.sort_index()
            elif ctype in ("bar", "pie"):
                s = s.sort_values(ascending=False)

            if ctype == "pie":
                if len(s) > 8:
                    s = s.head(7)._append(pd.Series({"Other": s.iloc[7:].sum()})) \
                        if hasattr(s, "_append") else s.head(7)
                ax.pie(s.values, labels=[str(i) for i in s.index], autopct="%1.1f%%",
                       colors=colors * 3, startangle=90, textprops={"fontsize": 8})
                ax.axis("equal")
            elif ctype == "line":
                ax.plot([str(i) for i in s.index], s.values, marker="o",
                        color=colors[0], linewidth=2)
            else:
                s = s.head(25)
                labels = [str(i) for i in s.index]
                if horizontal:
                    ax.barh(labels, s.values, color=colors[0])
                    ax.invert_yaxis()
                else:
                    ax.bar(labels, s.values, color=colors[0])

        if ctype != "pie":
            val_axis = ax.xaxis if horizontal and ctype == "bar" else ax.yaxis
            val_axis.set_major_formatter(FuncFormatter(_fmt_num))
            if horizontal and ctype == "bar":
                ax.set_xlabel(f"{agg} of {y}")
                ax.set_ylabel(xcol if xcol != "__x__" else x)
            else:
                ax.set_ylabel(f"{agg} of {y}")
                ax.set_xlabel(x if xcol == "__x__" else xcol)
                n = len(ax.get_xticklabels())
                if n > 12:  # thin out crowded date/category labels
                    step = max(1, n // 12)
                    for i, lbl in enumerate(ax.get_xticklabels()):
                        lbl.set_visible(i % step == 0)
                fig.autofmt_xdate(rotation=40, ha="right") if n > 6 else None

    if ctype != "pie":
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y" if not horizontal else "x", alpha=.25)
        ax.set_axisbelow(True)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def _supports_tick_labels() -> bool:
    import matplotlib
    major, minor = (int(p) for p in matplotlib.__version__.split(".")[:2])
    return (major, minor) >= (3, 9)
