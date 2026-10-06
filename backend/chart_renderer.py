# backend/chart_renderer.py
"""
Self-contained chart renderer for reports.

WHY THIS EXISTS
The report used matplotlib (and, upstream, Plotly + Kaleido) to draw its
charts. If any of those is missing or broken in the deployment
environment the charts silently vanish from the report. This module draws
charts with Pillow + numpy ONLY. Pillow is already a hard dependency of
fpdf2 (which builds the PDF), so if a PDF can be produced at all, these
charts can be drawn. No matplotlib, no Plotly, no Kaleido, no browser.

Every public function takes plain Python/numpy data and returns PNG bytes.
"""

import io
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 650          # logical size of every chart
S = 2                     # supersampling factor (drawn at 2x, downsampled)

BG = (255, 255, 255)
INK = (15, 23, 42)
MUTED = (100, 116, 139)
GRID = (226, 232, 240)
AXIS = (148, 163, 184)


def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


COLORS = [_hex(c) for c in (
    "#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626",
    "#0891B2", "#DB2777", "#65A30D", "#475569", "#EA580C",
)]


# ---------------------------------------------------------------- fonts
_font_cache = {}


def _font(size, bold=False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
    candidates = [
        os.path.join(here, "Poppins-Bold.ttf" if bold else "Poppins-Regular.ttf"),
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
        "Arial.ttf",
    ]
    font = None
    for c in candidates:
        try:
            font = ImageFont.truetype(c, int(size * S))
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=int(size * S))
        except TypeError:           # very old Pillow
            font = ImageFont.load_default()
    _font_cache[key] = font
    return font


# --------------------------------------------------------------- helpers
def _fmt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(v):
        return "N/A"
    a = abs(v)

    def trim(s):
        return s.rstrip("0").rstrip(".") if "." in s else s

    if a >= 1e9:
        return trim(f"{v / 1e9:.2f}") + "B"
    if a >= 1e6:
        return trim(f"{v / 1e6:.2f}") + "M"
    if a >= 1e4:
        return trim(f"{v / 1e3:.1f}") + "K"
    if a >= 1000:
        return f"{v:,.0f}"
    if a == int(a):
        return str(int(v))
    if a >= 1:
        return trim(f"{v:.2f}")
    return f"{v:.3g}"


def _nice_ticks(vmin, vmax, n=5):
    vmin, vmax = float(vmin), float(vmax)
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        pad = abs(vmin) * 0.1 or 1.0
        vmin, vmax = vmin - pad, vmax + pad
    raw = (vmax - vmin) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = 10 * mag
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    count = int(round((hi - lo) / step))
    ticks = [round(lo + i * step, 10) for i in range(count + 1)]
    return ticks[0], ticks[-1], ticks


class _Canvas:
    def __init__(self):
        self.img = Image.new("RGBA", (W * S, H * S), BG + (255,))
        self.d = ImageDraw.Draw(self.img, "RGBA")

    # shapes ------------------------------------------------------
    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1):
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        kw = {"fill": fill}
        if outline:
            kw.update(outline=outline, width=max(1, int(width * S)))
        self.d.rectangle([x0 * S, y0 * S, x1 * S, y1 * S], **kw)

    def line(self, x0, y0, x1, y1, fill=GRID, width=1):
        self.d.line([x0 * S, y0 * S, x1 * S, y1 * S], fill=fill, width=max(1, int(width * S)))

    def polyline(self, pts, fill, width=2):
        if len(pts) >= 2:
            self.d.line([(x * S, y * S) for x, y in pts], fill=fill,
                        width=max(1, int(width * S)), joint="curve")

    def dot(self, cx, cy, r, fill, outline=None):
        self.d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S],
                       fill=fill, outline=outline)

    def pie(self, cx, cy, r, start, end, fill):
        self.d.pieslice([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S],
                        start, end, fill=fill, outline=(255, 255, 255), width=S)

    # text --------------------------------------------------------
    def tw(self, s, size=12, bold=False):
        return self.d.textlength(str(s), font=_font(size, bold)) / S

    def fit(self, s, maxw, size=12, bold=False):
        s = str(s)
        if self.tw(s, size, bold) <= maxw:
            return s
        while len(s) > 1 and self.tw(s + "..", size, bold) > maxw:
            s = s[:-1]
        return s + ".."

    def text(self, x, y, s, size=12, fill=INK, align="l", valign="t", bold=False):
        s = str(s)
        w = self.tw(s, size, bold)
        if align == "m":
            x -= w / 2
        elif align == "r":
            x -= w
        if valign == "m":
            y -= size * 0.62
        elif valign == "b":
            y -= size * 1.25
        self.d.text((x * S, y * S), s, font=_font(size, bold), fill=fill)

    def png(self):
        out = self.img.convert("RGB").resize((W, H), Image.LANCZOS)
        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def _title(cv, title):
    cv.text(W / 2, 16, cv.fit(title, W - 80, 20, True), size=20, align="m", bold=True)


def _legend(cv, names, x_right=None, y=52):
    items = [(n, COLORS[i % len(COLORS)]) for i, n in enumerate(names)]
    widths = [cv.tw(str(n), 12) + 30 for n, _ in items]
    x = (x_right if x_right else W - 30) - sum(widths)
    x = max(x, 20)
    for (n, col), w in zip(items, widths):
        cv.rect(x, y + 2, x + 12, y + 14, fill=col)
        cv.text(x + 18, y, n, size=12, fill=MUTED)
        x += w


def _axis_labels(cv, xlabel, ylabel, L, T, R, B):
    if xlabel:
        cv.text((L + R) / 2, H - 28, cv.fit(xlabel, R - L, 13), size=13, fill=MUTED, align="m")
    if ylabel:
        cv.text(L, T - 24, cv.fit(ylabel, 400, 13), size=13, fill=MUTED)


def _vaxis(cv, L, T, R, B, lo, hi, ticks):
    """Horizontal gridlines + y tick labels (value axis on the left)."""
    span = (hi - lo) or 1.0
    for t in ticks:
        y = B - (t - lo) / span * (B - T)
        cv.line(L, y, R, y, GRID, 1)
        cv.text(L - 10, y, _fmt(t), size=12, fill=MUTED, align="r", valign="m")
    cv.line(L, T, L, B, AXIS, 1)
    cv.line(L, B, R, B, AXIS, 1)
    return lambda v: B - (v - lo) / span * (B - T)


def _haxis(cv, L, T, R, B, lo, hi, ticks):
    """Vertical gridlines + x tick labels (value axis along the bottom)."""
    span = (hi - lo) or 1.0
    for t in ticks:
        x = L + (t - lo) / span * (R - L)
        cv.line(x, T, x, B, GRID, 1)
        cv.text(x, B + 8, _fmt(t), size=12, fill=MUTED, align="m")
    cv.line(L, T, L, B, AXIS, 1)
    cv.line(L, B, R, B, AXIS, 1)
    return lambda v: L + (v - lo) / span * (R - L)


def _clean(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


# ------------------------------------------------------------ bar (horiz)
def render_bar_h(title, labels, values, xlabel="", ylabel="", color=None):
    """Ranked horizontal bars; the FIRST item is drawn at the top."""
    labels = [str(l) for l in labels]
    values = [float(v) if v is not None and math.isfinite(float(v)) else 0.0 for v in values]
    n = len(values)
    if n == 0:
        raise ValueError("no data")
    cv = _Canvas()
    _title(cv, title)
    label_w = min(260, max(cv.tw(l, 12) for l in labels) + 6)
    L, R, T, B = 40 + label_w, W - 70, 80, H - 80
    lo, hi, ticks = _nice_ticks(min(0.0, min(values)), max(0.0, max(values)))
    xm = _haxis(cv, L, T, R, B, lo, hi, ticks)
    band = (B - T) / n
    bh = min(band * 0.7, 46)
    base = xm(0)
    col = color or COLORS[0]
    for i, (lab, v) in enumerate(zip(labels, values)):
        cy = T + band * (i + 0.5)
        cv.rect(base, cy - bh / 2, xm(v), cy + bh / 2, fill=col)
        cv.text(L - 10, cy, cv.fit(lab, label_w, 12), size=12, align="r", valign="m")
        if n <= 15:
            if v >= 0:
                cv.text(xm(v) + 6, cy, _fmt(v), size=11, fill=MUTED, valign="m")
            else:
                cv.text(xm(v) - 6, cy, _fmt(v), size=11, fill=MUTED, align="r", valign="m")
    _axis_labels(cv, xlabel, "", L, T, R, B)
    if ylabel:
        cv.text(L - label_w - 5, T - 24, cv.fit(ylabel, 300, 13), size=13, fill=MUTED)
    return cv.png()


# --------------------------------------------------------- bar (vertical)
def render_grouped_bar(title, categories, series, xlabel="", ylabel=""):
    """Vertical bars. `series` is {name: [values aligned with categories]}.
    One series -> plain bar chart; several -> grouped bars with a legend."""
    categories = [str(c) for c in categories]
    names = list(series.keys())
    n, k = len(categories), len(names)
    if n == 0 or k == 0:
        raise ValueError("no data")
    data = [[float(v) if v is not None and math.isfinite(float(v)) else 0.0
             for v in series[nm]] for nm in names]
    allv = [v for row in data for v in row]
    cv = _Canvas()
    _title(cv, title)
    L, R, T, B = 100, W - 40, 100 if k > 1 else 85, H - 95
    lo, hi, ticks = _nice_ticks(min(0.0, min(allv)), max(0.0, max(allv)))
    ym = _vaxis(cv, L, T, R, B, lo, hi, ticks)
    band = (R - L) / n
    gw = band * 0.78
    bw = gw / k
    zero = ym(0)
    for ci in range(n):
        x0 = L + band * ci + (band - gw) / 2
        for si in range(k):
            v = data[si][ci]
            cv.rect(x0 + bw * si + 1, ym(v), x0 + bw * (si + 1) - 1, zero,
                    fill=COLORS[si % len(COLORS)])
            if k == 1 and n <= 14:
                cv.text(x0 + bw / 2, ym(v) - 16, _fmt(v), size=11, fill=MUTED, align="m")
    step = max(1, math.ceil(n / max(1, int((R - L) // 70))))
    for ci in range(0, n, step):
        cx = L + band * (ci + 0.5)
        cv.text(cx, B + 10, cv.fit(categories[ci], band * step - 6, 12), size=12,
                fill=MUTED, align="m")
    if k > 1:
        _legend(cv, names)
    _axis_labels(cv, xlabel, ylabel, L, T, R, B)
    return cv.png()


# ------------------------------------------------------------------- line
def render_line(title, labels, series, xlabel="", ylabel=""):
    """`series` is {name: [values aligned with labels]} (NaN = gap)."""
    labels = [str(l) for l in labels]
    names = list(series.keys())
    n, k = len(labels), len(names)
    if n == 0 or k == 0:
        raise ValueError("no data")
    data = [np.asarray(series[nm], dtype=float) for nm in names]
    finite = np.concatenate([d[np.isfinite(d)] for d in data]) if data else np.array([])
    if finite.size == 0:
        raise ValueError("no finite data")
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin >= 0:
        vmin = 0.0
    cv = _Canvas()
    _title(cv, title)
    L, R, T, B = 100, W - 50, 100 if k > 1 else 85, H - 95
    lo, hi, ticks = _nice_ticks(vmin, vmax)
    ym = _vaxis(cv, L, T, R, B, lo, hi, ticks)

    def xm(i):
        return L + (R - L) * (0.5 if n == 1 else (i + 0.0) / (n - 1)) if n > 1 else (L + R) / 2

    if n > 1:
        pad = 18
        xm = lambda i: L + pad + (R - L - 2 * pad) * i / (n - 1)  # noqa: E731
    step = max(1, math.ceil(n / 10))
    for i in range(0, n, step):
        cv.line(xm(i), B, xm(i), B + 5, AXIS, 1)
        cv.text(xm(i), B + 10, cv.fit(labels[i], 90, 12), size=12, fill=MUTED, align="m")
    for si, d in enumerate(data):
        col = COLORS[si % len(COLORS)]
        seg = []
        for i, v in enumerate(d):
            if np.isfinite(v):
                seg.append((xm(i), ym(v)))
            else:
                cv.polyline(seg, col, 3)
                seg = []
        cv.polyline(seg, col, 3)
        if n <= 40:
            for i, v in enumerate(d):
                if np.isfinite(v):
                    cv.dot(xm(i), ym(v), 4.5, col, outline=(255, 255, 255))
    if k > 1:
        _legend(cv, names)
    _axis_labels(cv, xlabel, ylabel, L, T, R, B)
    return cv.png()


# -------------------------------------------------------------- histogram
def render_hist(title, values, xlabel="", bins=None):
    v = _clean(values)
    if v.size < 2:
        raise ValueError("not enough data")
    if bins is None:
        bins = int(min(40, max(10, math.sqrt(v.size))))
        uniq = np.unique(v)
        if uniq.size <= 4:
            bins = max(2, uniq.size)
    counts, edges = np.histogram(v, bins=bins)
    cv = _Canvas()
    _title(cv, title)
    L, R, T, B = 100, W - 40, 100, H - 95
    lo, hi, yt = _nice_ticks(0, counts.max())
    ym = _vaxis(cv, L, T, R, B, lo, hi, yt)
    xlo, xhi, xt = _nice_ticks(edges[0], edges[-1])
    xspan = (xhi - xlo) or 1.0

    def xm(val):
        return L + (val - xlo) / xspan * (R - L)

    for t in xt:
        cv.text(xm(t), B + 10, _fmt(t), size=12, fill=MUTED, align="m")
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        cv.rect(xm(e0) + 1, ym(c), xm(e1) - 1, ym(0), fill=COLORS[0])
    mean, median = float(v.mean()), float(np.median(v))
    cv.line(xm(mean), T, xm(mean), B, COLORS[4], 2)
    cv.line(xm(median), T, xm(median), B, COLORS[2], 2)
    cv.rect(R - 230, T + 6, R - 218, T + 18, fill=COLORS[4])
    cv.text(R - 212, T + 4, f"Mean: {_fmt(mean)}", size=12, fill=MUTED)
    cv.rect(R - 230, T + 28, R - 218, T + 40, fill=COLORS[2])
    cv.text(R - 212, T + 26, f"Median: {_fmt(median)}", size=12, fill=MUTED)
    _axis_labels(cv, xlabel, "Frequency", L, T, R, B)
    return cv.png()


# ---------------------------------------------------------------- scatter
def render_scatter(title, x, y, xlabel="", ylabel="", max_points=6000):
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 2:
        raise ValueError("not enough data")
    if xa.size > max_points:
        idx = np.random.default_rng(42).choice(xa.size, max_points, replace=False)
        xa, ya = xa[idx], ya[idx]
    cv = _Canvas()
    _title(cv, title)
    L, R, T, B = 100, W - 40, 85, H - 95
    ylo, yhi, yt = _nice_ticks(ya.min(), ya.max())
    ym = _vaxis(cv, L, T, R, B, ylo, yhi, yt)
    xlo, xhi, xt = _nice_ticks(xa.min(), xa.max())
    xspan = (xhi - xlo) or 1.0
    for t in xt:
        px = L + (t - xlo) / xspan * (R - L)
        cv.line(px, T, px, B, GRID, 1)
        cv.text(px, B + 10, _fmt(t), size=12, fill=MUTED, align="m")
    col = COLORS[0] + (120,)
    for a, b in zip(xa, ya):
        cv.dot(L + (a - xlo) / xspan * (R - L), ym(b), 4, col)
    _axis_labels(cv, xlabel, ylabel, L, T, R, B)
    return cv.png()


# -------------------------------------------------------------------- pie
def render_pie(title, labels, values, max_slices=8):
    pairs = [(str(l), float(v)) for l, v in zip(labels, values)
             if v is not None and math.isfinite(float(v)) and float(v) > 0]
    if not pairs:
        raise ValueError("no positive data")
    pairs.sort(key=lambda p: p[1], reverse=True)
    if len(pairs) > max_slices:
        other = sum(v for _, v in pairs[max_slices - 1:])
        pairs = pairs[:max_slices - 1] + [("Other", other)]
    total = sum(v for _, v in pairs)
    cv = _Canvas()
    _title(cv, title)
    cx, cy, r = 330, 355, 215
    ang = -90.0
    for i, (lab, v) in enumerate(pairs):
        sweep = 360.0 * v / total
        cv.pie(cx, cy, r, ang, ang + sweep, COLORS[i % len(COLORS)])
        ang += sweep
    y = 130
    for i, (lab, v) in enumerate(pairs):
        cv.rect(660, y + 2, 676, y + 18, fill=COLORS[i % len(COLORS)])
        cv.text(688, y, f"{cv.fit(lab, 300, 14)}  {v / total * 100:.1f}%", size=14)
        y += 34
    return cv.png()


# ---------------------------------------------------------------- heatmap
def render_heatmap(title, labels, matrix):
    labels = [str(l) for l in labels]
    m = np.asarray(matrix, dtype=float)
    n = len(labels)
    if n == 0 or m.shape != (n, n):
        raise ValueError("bad matrix")
    cv = _Canvas()
    _title(cv, title)
    lab_w = min(170, max(cv.tw(l, 12) for l in labels) + 6)
    L, T = 40 + lab_w, 75
    size = min((W - L - 120) / n, (H - T - 95) / n)
    pos, neg, mid = COLORS[0], COLORS[4], (255, 255, 255)

    def colour(v):
        if not math.isfinite(v):
            return (240, 240, 240)
        t = max(-1.0, min(1.0, v))
        end = pos if t >= 0 else neg
        a = abs(t)
        return tuple(int(mid[i] + (end[i] - mid[i]) * a) for i in range(3))

    for i in range(n):
        cv.text(L - 8, T + size * (i + 0.5), cv.fit(labels[i], lab_w, 12), size=12,
                align="r", valign="m")
        cv.text(L + size * (i + 0.5), T + size * n + 8, cv.fit(labels[i], size - 4, 11),
                size=11, align="m", fill=MUTED)
        for j in range(n):
            v = m[i, j]
            cv.rect(L + size * j, T + size * i, L + size * (j + 1), T + size * (i + 1),
                    fill=colour(v), outline=(255, 255, 255), width=1)
            if math.isfinite(v):
                cv.text(L + size * (j + 0.5), T + size * (i + 0.5), f"{v:.2f}",
                        size=12 if n <= 8 else 10, align="m", valign="m",
                        fill=(255, 255, 255) if abs(v) > 0.6 else INK)
    # colour bar
    bx = L + size * n + 30
    for k in range(100):
        v = 1 - k / 49.5
        cv.rect(bx, T + k * (size * n / 100), bx + 18, T + (k + 1) * (size * n / 100) + 1,
                fill=colour(v))
    cv.text(bx + 26, T - 4, "1.0", size=11, fill=MUTED)
    cv.text(bx + 26, T + size * n / 2 - 7, "0", size=11, fill=MUTED)
    cv.text(bx + 26, T + size * n - 12, "-1.0", size=11, fill=MUTED)
    return cv.png()


# --------------------------------------------------------------- box plot
def render_box(title, groups, ylabel=""):
    """`groups` is {name: values}."""
    names = list(groups.keys())
    arrs = [_clean(groups[n]) for n in names]
    arrs = [a for a in arrs if a.size >= 2]
    if not arrs:
        raise ValueError("not enough data")
    names = [n for n, a in zip(names, [_clean(groups[n]) for n in names]) if a.size >= 2]
    allv = np.concatenate(arrs)
    cv = _Canvas()
    _title(cv, title)
    L, R, T, B = 100, W - 40, 85, H - 95
    lo, hi, ticks = _nice_ticks(allv.min(), allv.max())
    ym = _vaxis(cv, L, T, R, B, lo, hi, ticks)
    band = (R - L) / len(arrs)
    for i, (nm, a) in enumerate(zip(names, arrs)):
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        iqr = q3 - q1
        wl = a[a >= q1 - 1.5 * iqr].min()
        wh = a[a <= q3 + 1.5 * iqr].max()
        cx = L + band * (i + 0.5)
        bw = min(90, band * 0.4)
        col = COLORS[i % len(COLORS)]
        cv.line(cx, ym(wl), cx, ym(q1), AXIS, 2)
        cv.line(cx, ym(q3), cx, ym(wh), AXIS, 2)
        cv.line(cx - bw / 4, ym(wl), cx + bw / 4, ym(wl), AXIS, 2)
        cv.line(cx - bw / 4, ym(wh), cx + bw / 4, ym(wh), AXIS, 2)
        cv.rect(cx - bw / 2, ym(q3), cx + bw / 2, ym(q1), fill=col + (60,), outline=col, width=2)
        cv.line(cx - bw / 2, ym(med), cx + bw / 2, ym(med), col, 3)
        for o in a[(a < q1 - 1.5 * iqr) | (a > q3 + 1.5 * iqr)][:200]:
            cv.dot(cx, ym(o), 3, col)
        cv.text(cx, B + 10, cv.fit(nm, band - 6, 12), size=12, fill=MUTED, align="m")
    _axis_labels(cv, "", ylabel, L, T, R, B)
    return cv.png()


# --------------------------------------------------------------- forecast
def render_forecast(title, hist_labels, hist_values, fc_labels, fc_values,
                    lower=None, upper=None, ylabel=""):
    """History line, forecast line (dashed look via lighter colour) and an
    optional shaded uncertainty band. Labels are plain strings."""
    hist_labels = [str(x) for x in hist_labels]
    fc_labels = [str(x) for x in fc_labels]
    hv = [float(v) for v in hist_values]
    fv = [float(v) for v in fc_values]
    if not fv:
        raise ValueError("no forecast data")
    lo_b = [float(v) for v in lower] if lower is not None else fv
    up_b = [float(v) for v in upper] if upper is not None else fv
    labels = hist_labels + fc_labels
    n, nh = len(labels), len(hv)
    allv = hv + lo_b + up_b + fv
    cv = _Canvas()
    _title(cv, title)
    L, R, T, B = 100, W - 50, 100, H - 95
    lo, hi, ticks = _nice_ticks(min(0.0, min(allv)), max(allv))
    ym = _vaxis(cv, L, T, R, B, lo, hi, ticks)
    pad = 18
    xm = (lambda i: L + pad + (R - L - 2 * pad) * i / (n - 1)) if n > 1 else (lambda i: (L + R) / 2)

    # uncertainty band
    top = [(xm(nh + i), ym(v)) for i, v in enumerate(up_b)]
    bot = [(xm(nh + i), ym(v)) for i, v in enumerate(lo_b)][::-1]
    poly = top + bot
    if len(poly) >= 3:
        cv.d.polygon([(x * S, y * S) for x, y in poly], fill=COLORS[1] + (45,))

    step = max(1, math.ceil(n / 10))
    for i in range(0, n, step):
        cv.line(xm(i), B, xm(i), B + 5, AXIS, 1)
        cv.text(xm(i), B + 10, cv.fit(labels[i], 90, 12), size=12, fill=MUTED, align="m")

    hist_pts = [(xm(i), ym(v)) for i, v in enumerate(hv)]
    cv.polyline(hist_pts, COLORS[0], 3)
    # connect last actual point to the first forecast point
    fc_pts = ([hist_pts[-1]] if hist_pts else []) + [(xm(nh + i), ym(v)) for i, v in enumerate(fv)]
    cv.polyline(fc_pts, COLORS[1], 3)
    if n <= 40:
        for x, y in hist_pts:
            cv.dot(x, y, 4.5, COLORS[0], outline=(255, 255, 255))
        for x, y in fc_pts[1 if hist_pts else 0:]:
            cv.dot(x, y, 4.5, COLORS[1], outline=(255, 255, 255))
    _legend(cv, ["Actual", "Forecast"])
    _axis_labels(cv, "", ylabel, L, T, R, B)
    return cv.png()
