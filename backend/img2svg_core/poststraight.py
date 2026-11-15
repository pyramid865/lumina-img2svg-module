"""Straighten near-straight bezier runs.

Curve fitting happily replaces a perfectly straight edge with a shallow cubic,
which reads as "slightly bent" next to a hand-drawn vector (Vectorizer.AI keeps
straight edges straight on purpose).  This pass measures how far each bezier
deviates from its own chord and rewrites it as a line when the deviation is
below the tolerance.

Both absolute (M/L/C/Q) and relative (m/l/c/q) commands are handled - the
lumina engine emits relative ones, and treating them as absolute turns every
path into a radiating mess.  Arc (A) commands are passed through untouched.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"([MmLlCcQqAaZzHhVv])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_PATH = re.compile(r'(<path\b[^>]*?d=")([^"]+)(")')


def _point_on_cubic(p0, p1, p2, p3, t):
    mt = 1.0 - t
    a, b = mt * mt * mt, 3 * mt * mt * t
    c, d = 3 * mt * t * t, t * t * t
    return (a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1])


def _dist_to_line(pt, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        return ((pt[0] - ax) ** 2 + (pt[1] - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / L2))
    return ((pt[0] - ax - t * dx) ** 2 + (pt[1] - ay - t * dy) ** 2) ** 0.5


def _curve_dev(p0, p1, p2, p3, samples=7) -> float:
    worst = 0.0
    for i in range(1, samples):
        worst = max(worst, _dist_to_line(_point_on_cubic(p0, p1, p2, p3, i / float(samples)), p0, p3))
    return worst


def straighten_path(d: str, tol: float = 0.6) -> tuple[str, int]:
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    out: list[str] = []
    changed = 0
    cmd = None
    buf: list[float] = []

    def flush(c: str, nums: list[float]) -> None:
        nonlocal cur, start, changed
        rel = c.islower()
        base = c.upper()
        if base == "M":
            x, y = nums[0], nums[1]
            if rel:
                x, y = cur[0] + x, cur[1] + y
            cur = start = (x, y)
            out.append("M%.2f,%.2f" % cur)
            if len(nums) > 2:                    # implicit lineto
                flush("l" if rel else "L", nums[2:])
        elif base == "L":
            x, y = nums[0], nums[1]
            if rel:
                x, y = cur[0] + x, cur[1] + y
            cur = (x, y)
            out.append("L%.2f,%.2f" % cur)
            if len(nums) > 2:
                flush(c, nums[2:])
        elif base == "H":
            x = nums[0] + (cur[0] if rel else 0.0)
            cur = (x, cur[1])
            out.append("L%.2f,%.2f" % cur)
            if len(nums) > 1:
                flush(c, nums[1:])
        elif base == "V":
            y = nums[0] + (cur[1] if rel else 0.0)
            cur = (cur[0], y)
            out.append("L%.2f,%.2f" % cur)
            if len(nums) > 1:
                flush(c, nums[1:])
        elif base == "C":
            p0 = cur
            if rel:
                p1 = (p0[0] + nums[0], p0[1] + nums[1])
                p2 = (p0[0] + nums[2], p0[1] + nums[3])
                p3 = (p0[0] + nums[4], p0[1] + nums[5])
            else:
                p1, p2, p3 = (nums[0], nums[1]), (nums[2], nums[3]), (nums[4], nums[5])
            if _curve_dev(p0, p1, p2, p3) <= tol:
                out.append("L%.2f,%.2f" % p3)
                changed += 1
            else:
                out.append("C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (
                    p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]))
            cur = p3
            if len(nums) > 6:
                flush(c, nums[6:])
        elif base == "Q":
            p0 = cur
            if rel:
                p1 = (p0[0] + nums[0], p0[1] + nums[1])
                p2 = (p0[0] + nums[2], p0[1] + nums[3])
            else:
                p1, p2 = (nums[0], nums[1]), (nums[2], nums[3])
            c1 = (p0[0] + 2 / 3 * (p1[0] - p0[0]), p0[1] + 2 / 3 * (p1[1] - p0[1]))
            c2 = (p2[0] + 2 / 3 * (p1[0] - p2[0]), p2[1] + 2 / 3 * (p1[1] - p2[1]))
            if _curve_dev(p0, c1, c2, p2) <= tol:
                out.append("L%.2f,%.2f" % p2)
                changed += 1
            else:
                out.append("Q%.2f,%.2f %.2f,%.2f" % (p1[0], p1[1], p2[0], p2[1]))
            cur = p2
            if len(nums) > 4:
                flush(c, nums[4:])
        elif base == "A":
            x, y = nums[5], nums[6]
            if rel:
                x, y = cur[0] + x, cur[1] + y
            out.append("A%.2f,%.2f %.2f %d %d %.2f,%.2f" % (
                nums[0], nums[1], nums[2], int(nums[3]), int(nums[4]), x, y))
            cur = (x, y)
            if len(nums) > 7:
                flush(c, nums[7:])
        elif base == "Z":
            out.append("Z")
            cur = start

    for m in _TOKEN.finditer(d):
        name, num = m.group(1), m.group(2)
        if name:
            if cmd is not None:
                if buf:
                    flush(cmd, buf)
                elif cmd in "Zz":
                    flush(cmd, [])
                buf = []
            cmd = name
        elif num is not None:
            buf.append(float(num))
    if cmd is not None:
        if buf:
            flush(cmd, buf)
        elif cmd in "Zz":
            flush(cmd, [])
    return "".join(out), changed


def straighten_svg(svg: str, tol: float = 0.6) -> tuple[str, dict]:
    total = 0
    paths = 0

    def repl(m):
        nonlocal total, paths
        new_d, changed = straighten_path(m.group(2), tol)
        if changed:
            total += changed
            paths += 1
        return m.group(1) + new_d + m.group(3)

    out = _PATH.sub(repl, svg)
    return out, dict(curves_straightened=total, paths=paths)
