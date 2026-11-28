"""Rasterise the traced SVG into the PNG compatibility fallback.

Workshop hosts use ``pngBytes`` as the compatible fallback when native SVG is
unavailable.  That PNG must show the *traced* artwork, not the source photo.
Shipping a native SVG rasteriser is not allowed (backend/ is text-only), so
this small pure-Pillow renderer handles the subset of SVG our tracers emit:
absolute and relative ``M/L/H/V/C/S/Q/T/A/Z`` paths, one fill colour per path,
and per-path ``translate()``.

Cubic and quadratic curves and elliptical arcs are flattened, and strokes are
approximated with an outline of the same colour.  It is a preview
approximation only; the SVG itself is byte-exact and remains the authoritative
output.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

# Long-edge cap for the preview PNG.  The SVG keeps its own dimensions; only
# this raster preview is bounded, so an oversized artwork cannot allocate
# hundreds of megabytes just to build a thumbnail.
MAX_PREVIEW_EDGE = 4096  # optional cap; callers decide, default keeps the SVG's own size

_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
_NUM_RE = re.compile(_NUM)
_SVG_ROOT_RE = re.compile(r"<svg\b([^>]*)>", re.I)
_PATH_RE = re.compile(r"<path\b([^>]*)/?>", re.I)
_TRANSLATE_RE = re.compile(
    r"translate\(\s*(" + _NUM + r")(?:[,\s]+(" + _NUM + r"))?\s*\)", re.I)
_TOKEN_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(" + _NUM + r")")
_CURVE_STEPS = 8
_ARC_STEPS = 16
_DEFAULT_SIZE = 1024.0
_ATTR_CACHE: dict[str, re.Pattern[str]] = {}
_HEX_CACHE: dict[str, tuple[int, int, int]] = {}

_ATTR_PAIR_RE = re.compile(
    r"(?<![-\w])([A-Za-z][\w:.-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')",
)


def _attr(attrs: str, name: str) -> str | None:
    """Read one attribute value, accepting single or double quotes.

    The lookbehind stops name='width' from matching the 'width' inside
    'stroke-width', which used to win over the real root width.
    """
    pattern = _ATTR_CACHE.get(name)
    if pattern is None:
        pattern = re.compile(
            r"(?<![-\w])" + re.escape(name)
            + r"\s*=\s*(?:\"([^\"]*)\"|'([^']*)')",
            re.I,
        )
        _ATTR_CACHE[name] = pattern
    match = pattern.search(attrs)
    if not match:
        return None
    if match.group(1) is not None:
        return match.group(1)
    return match.group(2)


def _path_attrs(attrs: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Read d, fill, transform and fill-rule in one pass.

    Rendering used to run separate regex searches per path element; a single
    attribute scan keeps the same precedence (the first occurrence of each name
    wins) with one traversal of the tag text.

    ``fill-rule`` is read here because our tracer emits its printable ink band
    as ``<g fill=… fill-rule="evenodd"><path …/></g>``: both values live on the
    group, so a path-only scan loses the band entirely.
    """
    data = fill = transform = rule = None
    for match in _ATTR_PAIR_RE.finditer(attrs):
        name = match.group(1).lower()
        value = match.group(2) if match.group(2) is not None else match.group(3)
        if name == "d":
            data = value
        elif name == "fill":
            fill = value
        elif name == "transform":
            transform = value
        elif name == "fill-rule":
            rule = value
    return data, fill, transform, rule


def _length(value: str | None) -> float | None:
    """Parse a positive absolute length; percentages and junk yield None."""
    if value is None:
        return None
    match = re.match(r"\s*(" + _NUM + r")\s*(?:px)?\s*$", value, re.I)
    if not match:
        return None
    number = float(match.group(1))
    return number if number > 0 else None


def _svg_size(svg_text: str) -> tuple[float, float, float, float]:
    """Return (display width, display height, viewBox width, viewBox height)."""
    root = _SVG_ROOT_RE.search(svg_text)
    attrs = root.group(1) if root else ""
    vb_text = _attr(attrs, "viewBox")
    vb_nums = [float(x) for x in _NUM_RE.findall(vb_text)] if vb_text else []
    # A viewBox needs four numbers; anything shorter is ignored instead of
    # raising IndexError.
    vb_w = vb_nums[2] if len(vb_nums) >= 4 else 0.0
    vb_h = vb_nums[3] if len(vb_nums) >= 4 else 0.0
    out_w = _length(_attr(attrs, "width"))
    out_h = _length(_attr(attrs, "height"))
    if out_w is None and out_h is None:
        if vb_w > 0 and vb_h > 0:
            out_w, out_h = vb_w, vb_h
        else:
            out_w = out_h = _DEFAULT_SIZE
    elif out_w is None:
        out_w = out_h * vb_w / vb_h if vb_w > 0 and vb_h > 0 else out_h
    elif out_h is None:
        out_h = out_w * vb_h / vb_w if vb_w > 0 and vb_h > 0 else out_w
    if vb_w <= 0 or vb_h <= 0:
        vb_w, vb_h = out_w, out_h
    return out_w, out_h, vb_w, vb_h


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    key = value.strip()
    cached = _HEX_CACHE.get(key)
    if cached is not None:
        return cached
    v = key
    if not v.startswith("#"):
        rgb = (0, 0, 0)
    else:
        if len(v) == 4:
            v = "#" + "".join(c * 2 for c in v[1:])
        try:
            rgb = (int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16)) \
                if len(v) == 7 else (0, 0, 0)
        except ValueError:
            rgb = (0, 0, 0)
    _HEX_CACHE[key] = rgb
    return rgb


def _flatten_cubic(p0, p1, p2, p3, steps: int = _CURVE_STEPS):
    # Returning a list avoids one generator frame per segment; the caller only
    # ever consumes the points with list.extend(), never lazily.
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        u = 1.0 - t
        points.append((
            u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
            u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
        ))
    return points


def _flatten_quad(p0, p1, p2, steps: int = _CURVE_STEPS):
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        u = 1.0 - t
        points.append((
            u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
        ))
    return points


def _arc_points(p0, rx, ry, rotation, large_arc, sweep, p1,
                steps: int = _ARC_STEPS):
    """Flatten an SVG elliptical arc (endpoint parametrisation) into points."""
    rx, ry = abs(rx), abs(ry)
    if rx == 0.0 or ry == 0.0 or p0 == p1:
        yield p1
        return
    phi = math.radians(rotation % 360.0)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    dx2 = (p0[0] - p1[0]) / 2.0
    dy2 = (p0[1] - p1[1]) / 2.0
    x1 = cos_phi * dx2 + sin_phi * dy2
    y1 = -sin_phi * dx2 + cos_phi * dy2
    scale = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry)
    if scale > 1.0:
        scale = math.sqrt(scale)
        rx *= scale
        ry *= scale
    numerator = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
    denominator = rx * rx * y1 * y1 + ry * ry * x1 * x1
    factor = math.sqrt(max(0.0, numerator / denominator)) if denominator else 0.0
    if bool(large_arc) == bool(sweep):
        factor = -factor
    cx1 = factor * rx * y1 / ry
    cy1 = -factor * ry * x1 / rx
    cx = cos_phi * cx1 - sin_phi * cy1 + (p0[0] + p1[0]) / 2.0
    cy = sin_phi * cx1 + cos_phi * cy1 + (p0[1] + p1[1]) / 2.0

    def _angle(ux, uy, vx, vy):
        length = math.hypot(ux, uy) * math.hypot(vx, vy)
        if length == 0.0:
            return 0.0
        ratio = max(-1.0, min(1.0, (ux * vx + uy * vy) / length))
        result = math.acos(ratio)
        return -result if (ux * vy - uy * vx) < 0 else result

    ux = (x1 - cx1) / rx
    uy = (y1 - cy1) / ry
    vx = (-x1 - cx1) / rx
    vy = (-y1 - cy1) / ry
    start_angle = _angle(1.0, 0.0, ux, uy)
    delta = _angle(ux, uy, vx, vy)
    if not sweep and delta > 0:
        delta -= 2.0 * math.pi
    elif sweep and delta < 0:
        delta += 2.0 * math.pi
    for i in range(1, steps + 1):
        theta = start_angle + delta * i / steps
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        yield (
            cx + rx * cos_theta * cos_phi - ry * sin_theta * sin_phi,
            cy + rx * cos_theta * sin_phi + ry * sin_theta * cos_phi,
        )


def _segments(d: str):
    """Yield (command letter exactly as written, [numbers]) in order.

    The letter case is preserved because relative commands (m/l/h/v/c/s/q/t/a)
    must be resolved against the current point, not used as absolute ones.
    """
    cmd, numbers = None, []
    for match in _TOKEN_RE.finditer(d):
        token = match.group()
        if token[0].isalpha():
            if cmd is not None:
                yield cmd, numbers
            cmd, numbers = token, []
        else:
            numbers.append(float(token))
    if cmd is not None:
        yield cmd, numbers


def _absolute(x, y, base, relative):
    if relative:
        return (base[0] + x, base[1] + y)
    return (x, y)


def _append_lines(numbers, pos, current, relative):
    for k in range(0, len(numbers) - 1, 2):
        pos = _absolute(numbers[k], numbers[k + 1], pos, relative)
        current.append(pos)
    return pos, current


def parse_subpaths(d: str) -> list[list[tuple[float, float]]]:
    """Return one closed polygon per subpath of an SVG path 'd' attribute."""
    polys: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    pos = (0.0, 0.0)
    subpath_start = (0.0, 0.0)
    control: tuple[float, float] | None = None
    previous = ""

    def flush() -> None:
        nonlocal current
        if len(current) >= 3:
            polys.append(current)
        current = []

    for raw, numbers in _segments(d):
        relative = raw.islower()
        cmd = raw.upper()
        if cmd == "M":
            if len(numbers) >= 2:
                pos = _absolute(numbers[0], numbers[1], pos, relative)
                flush()
                subpath_start = pos
                current = [pos]
                # Extra coordinate pairs after a moveto are implicit linetos.
                if relative:
                    for k in range(2, len(numbers) - 1, 2):
                        pos = (pos[0] + numbers[k], pos[1] + numbers[k + 1])
                        current.append(pos)
                else:
                    for k in range(2, len(numbers) - 1, 2):
                        pos = (numbers[k], numbers[k + 1])
                        current.append(pos)
        elif cmd == "L":
            if relative:
                for k in range(0, len(numbers) - 1, 2):
                    pos = (pos[0] + numbers[k], pos[1] + numbers[k + 1])
                    current.append(pos)
            else:
                for k in range(0, len(numbers) - 1, 2):
                    pos = (numbers[k], numbers[k + 1])
                    current.append(pos)
        elif cmd == "H":
            for value in numbers:
                x = pos[0] + value if relative else value
                pos = (x, pos[1])
                current.append(pos)
        elif cmd == "V":
            for value in numbers:
                y = pos[1] + value if relative else value
                pos = (pos[0], y)
                current.append(pos)
        elif cmd == "C":
            for k in range(0, len(numbers) - 5, 6):
                c1 = _absolute(numbers[k], numbers[k + 1], pos, relative)
                c2 = _absolute(numbers[k + 2], numbers[k + 3], pos, relative)
                end = _absolute(numbers[k + 4], numbers[k + 5], pos, relative)
                current.extend(_flatten_cubic(pos, c1, c2, end))
                pos, control = end, c2
        elif cmd == "S":
            for k in range(0, len(numbers) - 3, 4):
                if previous in ("C", "S") and control is not None:
                    c1 = (2.0 * pos[0] - control[0], 2.0 * pos[1] - control[1])
                else:
                    c1 = pos
                c2 = _absolute(numbers[k], numbers[k + 1], pos, relative)
                end = _absolute(numbers[k + 2], numbers[k + 3], pos, relative)
                current.extend(_flatten_cubic(pos, c1, c2, end))
                pos, control = end, c2
        elif cmd == "Q":
            for k in range(0, len(numbers) - 3, 4):
                q = _absolute(numbers[k], numbers[k + 1], pos, relative)
                end = _absolute(numbers[k + 2], numbers[k + 3], pos, relative)
                current.extend(_flatten_quad(pos, q, end))
                pos, control = end, q
        elif cmd == "T":
            for k in range(0, len(numbers) - 1, 2):
                if previous in ("Q", "T") and control is not None:
                    q = (2.0 * pos[0] - control[0], 2.0 * pos[1] - control[1])
                else:
                    q = pos
                end = _absolute(numbers[k], numbers[k + 1], pos, relative)
                current.extend(_flatten_quad(pos, q, end))
                pos, control = end, q
        elif cmd == "A":
            for k in range(0, len(numbers) - 6, 7):
                end = _absolute(numbers[k + 5], numbers[k + 6], pos, relative)
                current.extend(_arc_points(
                    pos, numbers[k], numbers[k + 1], numbers[k + 2],
                    numbers[k + 3] != 0.0, numbers[k + 4] != 0.0, end,
                ))
                pos, control = end, None
        elif cmd == "Z":
            flush()
            pos, control = subpath_start, None
        previous = cmd
    flush()
    return polys


_TAG_RE = re.compile(
    r"<(?P<close>/?)g\b(?P<gattrs>[^>]*?)(?P<gself>/?)>"
    r"|<path\b(?P<pathattrs>[^>]*?)(?P<pself>/?)>",
    re.I)


def _iter_paintables(svg_text: str):
    """Yield (path_attrs, inherited_groups) in document order.

    Our tracer wraps the printable ink band in ``<g fill=… fill-rule="evenodd">``
    and leaves the inner ``<path>`` without any paint attributes.  Scanning only
    ``<path>`` tags therefore dropped that band from every preview PNG, which is
    why the preview showed no outlines at all.  This walks ``<g>`` / ``</g>``
    too and hands each path the stack of enclosing groups.
    """
    stack: list[dict[str, str | None]] = []
    for match in _TAG_RE.finditer(svg_text):
        group_attrs = match.group("gattrs")
        if group_attrs is not None:                            # a <g …> tag
            if match.group("close") == "/":
                if stack:
                    stack.pop()
            elif not match.group("gself"):                     # not self-closing
                _d, fill, transform, rule = _path_attrs(group_attrs)
                stack.append({"fill": fill, "transform": transform,
                              "fill-rule": rule})
            continue
        yield match.group("pathattrs") or "", stack


def _resolve_paint(attrs: str, stack):
    """Path attributes win; anything missing is inherited from the nearest group."""
    path_data, fill, transform, rule = _path_attrs(attrs)
    for group in reversed(stack):
        if fill is None and group.get("fill"):
            fill = group["fill"]
        if transform is None and group.get("transform"):
            transform = group["transform"]
        if rule is None and group.get("fill-rule"):
            rule = group["fill-rule"]
        if fill is not None and transform is not None and rule is not None:
            break
    return path_data, fill, transform, rule


def _fill_evenodd(img: Image.Image, polys, colour) -> None:
    """Fill a multi-subpath polygon using the even-odd rule.

    PIL's ``polygon`` fills every subpath independently, so a path that punches
    a hole (our ink band outlines the eyes/mouth as inner rings) would come out
    as a solid blob.  XOR-accumulating each ring in a 1-bit mask reproduces what
    the browser shows, and stays inside the path's own bbox so the cost does not
    scale with canvas size.
    """
    xs = [x for poly in polys for x, _ in poly]
    ys = [y for poly in polys for _, y in poly]
    x0 = max(0, int(math.floor(min(xs))) - 1)
    y0 = max(0, int(math.floor(min(ys))) - 1)
    x1 = min(img.width, int(math.ceil(max(xs))) + 2)
    y1 = min(img.height, int(math.ceil(max(ys))) + 2)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return
    acc = Image.new("1", (w, h), 0)
    for poly in polys:
        ring = Image.new("1", (w, h), 0)
        ImageDraw.Draw(ring).polygon([(x - x0, y - y0) for x, y in poly], fill=1)
        acc = ImageChops.logical_xor(acc, ring)
    img.paste(Image.new("RGB", (w, h), colour), (x0, y0), acc)


def render(svg_text: str, max_size: int | None = None) -> Image.Image:
    """Render SVG source text to an RGB PIL image."""
    out_w, out_h, vb_w, vb_h = _svg_size(svg_text)

    width = max(1, int(round(out_w)))
    height = max(1, int(round(out_h)))
    if max_size and max(width, height) > max_size:
        ratio = max_size / max(width, height)
        width = max(1, int(round(width * ratio)))
        height = max(1, int(round(height * ratio)))
    sx = width / vb_w if vb_w else 1.0
    sy = height / vb_h if vb_h else 1.0

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for attrs, inherited in _iter_paintables(svg_text):
        path_data, fill, transform, rule = _resolve_paint(attrs, inherited)
        if not path_data or not fill:
            continue
        colour = _hex_to_rgb(fill)
        move = _TRANSLATE_RE.search(transform or "")
        dx = float(move.group(1)) if move else 0.0
        dy = float(move.group(2) or 0.0) if move else 0.0
        ox, oy = (dx * sx, dy * sy) if (dx or dy) else (0.0, 0.0)
        polys = [
            [(x * sx + ox, y * sy + oy) for x, y in poly]
            for poly in parse_subpaths(path_data)
        ]
        if not polys:
            continue
        if rule == "evenodd":
            _fill_evenodd(img, polys, colour)
        else:
            for points in polys:
                draw.polygon(points, fill=colour, outline=colour)
    return img


def render_file(svg_path: str | Path,
                max_size: int | None = None) -> Image.Image:
    return render(Path(svg_path).read_text(encoding="utf-8"), max_size)
