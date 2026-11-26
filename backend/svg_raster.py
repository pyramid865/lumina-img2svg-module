"""Rasterise vtracer 0.6 SVG into the PNG compatibility fallback.

Workshop hosts use ``pngBytes`` as the compatible fallback when native SVG is
unavailable.  That PNG must show the *traced* artwork, not the source photo.
Shipping a native SVG rasteriser is not allowed (backend/ is text-only), so
this small pure-Pillow renderer handles exactly what vtracer emits: absolute
``M/L/C/Z`` paths, one fill colour per path, and per-path ``translate()``.

Cubic curves are flattened and strokes are approximated with an outline of the
same colour.  It is a preview approximation only; the SVG itself is byte-exact
and remains the authoritative output.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw

_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
_PATH_RE = re.compile(r"<path\b([^>]*)/?>", re.I)
_D_RE = re.compile(r'\bd="([^"]*)"', re.I)
_FILL_RE = re.compile(r'\bfill="([^"]*)"', re.I)
_VIEWBOX_RE = re.compile(r'\bviewBox="([^"]*)"', re.I)
_WIDTH_RE = re.compile(r'\bwidth="([\d.]+)(?:px)?"', re.I)
_HEIGHT_RE = re.compile(r'\bheight="([\d.]+)(?:px)?"', re.I)
_TRANSLATE_RE = re.compile(
    r'transform="translate\(\s*(' + _NUM + r")[,\s]+(" + _NUM + r')\s*\)"', re.I)
_TOKEN_RE = re.compile(r"([MLCZmlcz])|(" + _NUM + r")")
_CURVE_STEPS = 8


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.strip()
    if not v.startswith("#"):
        return (0, 0, 0)
    if len(v) == 4:
        v = "#" + "".join(c * 2 for c in v[1:])
    try:
        return (int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16))
    except ValueError:
        return (0, 0, 0)


def _flatten_cubic(p0, p1, p2, p3, steps: int = _CURVE_STEPS):
    for i in range(1, steps + 1):
        t = i / steps
        u = 1.0 - t
        yield (
            u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
            u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
        )


def _segments(d: str):
    """Yield ``(command, [numbers])`` in order."""
    cmd, nums = None, []
    for m in _TOKEN_RE.finditer(d):
        letter, number = m.group(1), m.group(2)
        if letter:
            if cmd is not None:
                yield cmd.upper(), nums
            cmd, nums = letter.upper(), []
        else:
            nums.append(float(number))
    if cmd is not None:
        yield cmd.upper(), nums


def parse_subpaths(d: str) -> list[list[tuple[float, float]]]:
    """Return one polygon per ``M ... Z`` group in a path."""
    polys: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    pos = (0.0, 0.0)

    def flush() -> None:
        nonlocal cur
        if len(cur) >= 3:
            polys.append(cur)
        cur = []

    for cmd, n in _segments(d):
        if cmd == "M":
            flush()
            if len(n) >= 2:
                pos = (n[0], n[1])
                cur = [pos]
            for i in range(2, len(n) - 1, 2):
                pos = (n[i], n[i + 1])
                cur.append(pos)
        elif cmd == "L":
            for i in range(0, len(n) - 1, 2):
                pos = (n[i], n[i + 1])
                cur.append(pos)
        elif cmd == "C":
            for i in range(0, len(n) - 5, 6):
                end = (n[i + 4], n[i + 5])
                cur.extend(_flatten_cubic(pos, (n[i], n[i + 1]),
                                          (n[i + 2], n[i + 3]), end))
                pos = end
        elif cmd == "Z":
            flush()
    flush()
    return polys


def render(svg_text: str, max_size: int | None = None) -> Image.Image:
    """Render SVG source text to an RGB PIL image."""
    vb = _VIEWBOX_RE.search(svg_text)
    wm = _WIDTH_RE.search(svg_text)
    hm = _HEIGHT_RE.search(svg_text)
    # 两套尺寸必须分开看：
    #   viewBox  = 路径坐标所在的画布（linefirst 用 1600 宽的工作网格）
    #   width/height = 这张图真正的显示尺寸（通常就是原图像素）
    # 输出图按显示尺寸走，路径坐标再按比例缩到显示尺寸上，这样 meta 报的
    # 像素尺寸才和用户的原图一致（否则 683×916 的图会被报成 1600×2146）。
    if vb:
        n = [float(x) for x in re.findall(_NUM, vb.group(1))]
        vb_w, vb_h = n[2], n[3]
    else:
        vb_w = vb_h = 0.0
    if wm and hm:
        out_w, out_h = float(wm.group(1)), float(hm.group(1))
    elif vb_w and vb_h:
        # vtracer's fallback output declares width/height but no viewBox.
        out_w, out_h = vb_w, vb_h
    else:
        out_w = out_h = 1024.0
    if not (vb_w and vb_h):
        vb_w, vb_h = out_w, out_h

    width, height = max(1, int(round(out_w))), max(1, int(round(out_h)))
    scale = 1.0
    if max_size and max(width, height) > max_size:
        scale = max_size / max(width, height)
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
    sx = width / vb_w
    sy = height / vb_h

    img = Image.new("RGB", (max(1, width), max(1, height)), "white")
    draw = ImageDraw.Draw(img)
    for tag in _PATH_RE.finditer(svg_text):
        attrs = tag.group(1)
        dm, fm = _D_RE.search(attrs), _FILL_RE.search(attrs)
        if not dm or not fm:
            continue
        colour = _hex_to_rgb(fm.group(1))
        tm = _TRANSLATE_RE.search(attrs)
        dx, dy = (float(tm.group(1)), float(tm.group(2))) if tm else (0.0, 0.0)
        for poly in parse_subpaths(dm.group(1)):
            pts = [((x + dx) * sx, (y + dy) * sy) for x, y in poly]
            draw.polygon(pts, fill=colour, outline=colour)
    return img


def render_file(svg_path: str | Path, max_size: int | None = None) -> Image.Image:
    return render(Path(svg_path).read_text(encoding="utf-8"), max_size)
