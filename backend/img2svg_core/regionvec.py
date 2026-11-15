"""Region-based colour vectoriser (OpenCV contours -> smooth bezier).

Pipeline (matches how 家教纲吉.svg behaves):

    median flatten -> palette (median cut) -> per-colour mask
    -> cv2.findContours (CCOMP, holes included)
    -> approxPolyDP -> Chaikin smoothing -> Catmull-Rom cubic bezier
    -> one <path> per colour (evenodd) + same-colour seam strokes underneath

Every boundary is emitted once (no layer stacking), so edges stay hard; the
under-stroke pass fills the hairline cracks between adjacent colours.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter


def _chaikin(points: np.ndarray, iters: int = 1) -> np.ndarray:
    for _ in range(iters):
        q = 0.75 * points + 0.25 * np.roll(points, -1, axis=0)
        r = 0.25 * points + 0.75 * np.roll(points, -1, axis=0)
        out = np.empty((len(points) * 2, 2), float)
        out[0::2] = q
        out[1::2] = r
        points = out
    return points



def _circle_fit(pts: np.ndarray):
    """Kasa algebraic circle fit -> (cx, cy, r) or None."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(pts))]
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, float(np.sqrt(r2))


def _arc_path(pts: np.ndarray, tol: float, corner_deg: float = 28.0) -> str:
    """Emit L / A / Q: one arc per smooth run between detected corners.

    Linear in the number of points (one circle fit per run), so it stays fast
    even for contours with hundreds of vertices.
    """
    n = len(pts)
    if n < 3:
        return ""
    # turn angle at every vertex
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    v1 = pts - prev
    v2 = nxt - pts
    a1 = np.arctan2(v1[:, 1], v1[:, 0])
    a2 = np.arctan2(v2[:, 1], v2[:, 0])
    turn = np.abs((a2 - a1 + np.pi) % (2 * np.pi) - np.pi)
    corners = turn > np.radians(corner_deg)

    idx = list(np.nonzero(corners)[0])
    if not idx:                       # fully smooth loop: one arc if it fits
        idx = [0]
    idx = sorted(set(idx))

    out = ["M%.1f %.1f" % (pts[idx[0]][0], pts[idx[0]][1])]
    runs = []
    for a, b in zip(idx, idx[1:] + [idx[0] + n]):
        seg = [pts[i % n] for i in range(a, b + 1)]
        runs.append(np.array(seg, float))

    for run in runs:
        if len(run) == 2:
            out.append("L%.1f %.1f" % (run[-1][0], run[-1][1]))
            continue
        c = _circle_fit(run[1:-1] if len(run) > 3 else run)
        ok = False
        if c is not None:
            cx, cy, r = c
            if 3 < r < 5e4:
                dev = np.abs(np.hypot(run[:, 0] - cx, run[:, 1] - cy) - r).max()
                if dev <= tol:
                    p0, p1 = run[0], run[-1]
                    a0 = np.arctan2(p0[1] - cy, p0[0] - cx)
                    a1 = np.arctan2(p1[1] - cy, p1[0] - cx)
                    da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
                    large = 1 if abs(da) > np.pi else 0
                    sweep = 1 if da > 0 else 0
                    out.append("A%.1f %.1f 0 %d %d %.1f %.1f" %
                               (r, r, large, sweep, p1[0], p1[1]))
                    ok = True
        if not ok:
            # smooth fallback: quadratic through the middle of each pair
            for k in range(1, len(run) - 1):
                mid = (run[k] + run[k + 1]) / 2.0
                out.append("Q%.1f %.1f %.1f %.1f" %
                           (run[k][0], run[k][1], mid[0], mid[1]))
            out.append("L%.1f %.1f" % (run[-1][0], run[-1][1]))
    out.append("Z")
    return "".join(out)


def _bezier(points: np.ndarray) -> str:
    n = len(points)
    if n < 3:
        return ""
    parts = ["M%.1f %.1f" % (points[0][0], points[0][1])]
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        parts.append("C%.1f %.1f %.1f %.1f %.1f %.1f" % (
            c1[0], c1[1], c2[0], c2[1], p2[0], p2[1]))
    parts.append("Z")
    return "".join(parts)



def merge_regions(labels: np.ndarray, lab: np.ndarray, cent: np.ndarray,
                  min_area: int = 40, max_de: float = 0.035,
                  rounds: int = 2) -> np.ndarray:
    """Area + deltaE region merging.

    Small components are absorbed into the nearest neighbour *only if* the
    colour difference is small - high-contrast thin features (lashes, panel
    lines, catch-lights) have a large deltaE and therefore survive.
    """
    cur = labels.copy()
    h, w = cur.shape
    K = len(cent)
    for _ in range(rounds):
        changed = 0
        for k in range(K):
            mask = (cur == k).astype(np.uint8)
            if not mask.any():
                continue
            n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for ci in range(1, n):
                area = int(stats[ci, cv2.CC_STAT_AREA])
                if area >= min_area:
                    continue
                x, y = int(stats[ci, cv2.CC_STAT_LEFT]), int(stats[ci, cv2.CC_STAT_TOP])
                ww, hh = int(stats[ci, cv2.CC_STAT_WIDTH]), int(stats[ci, cv2.CC_STAT_HEIGHT])
                x0, y0 = max(0, x - 1), max(0, y - 1)
                x1, y1 = min(w, x + ww + 1), min(h, y + hh + 1)
                sub = cc[y0:y1, x0:x1] == ci
                ring = cv2.dilate(sub.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                ring &= ~sub
                if not ring.any():
                    continue
                neigh = cur[y0:y1, x0:x1][ring]
                neigh = neigh[neigh != k]
                if neigh.size == 0:
                    continue
                mean_lab = lab[y0:y1, x0:x1][sub].mean(0)
                # nearest neighbour colour that is close enough
                best, best_d = None, 1e9
                for nk in np.unique(neigh):
                    d = float(np.linalg.norm(mean_lab - cent[int(nk)]))
                    if d < best_d:
                        best, best_d = int(nk), d
                if best is None or best_d > max_de:
                    continue
                cur[y0:y1, x0:x1][sub] = best
                changed += 1
        if not changed:
            break
    return cur

def vectorise(img: Image.Image, colors: int = 128, grid: int = 1200,
              median: int = 3, eps: float = 1.2, min_area: float = 8.0,
              gap: float = 1.0, chaikin: int = 1, supersample: int = 2,
              sp: int = 0, sr: int = 22, merge: bool = False,
              merge_area: int = 40, merge_de: float = 0.035,
              arc_fit: bool = False, arc_tol: float = 0.55,
              min_hole: float = 24.0) -> tuple[str, dict]:
    im = img.convert("RGB")
    if im.width != grid:
        im = im.resize((grid, max(1, round(im.height * grid / im.width))), Image.LANCZOS)
    if sp > 0:
        arr = cv2.pyrMeanShiftFiltering(np.asarray(im), int(sp), int(sr), None, int(1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 5, 10.0))
        im = Image.fromarray(arr)
    if median >= 3:
        im = im.filter(ImageFilter.MedianFilter(median))

    # OKLab k-means palette keeps vivid small areas (eye highlight, cyan dial)
    from cel import flatten, inject_details, quantize as _oklab_quantize
    from color import palette_rgb, srgb_to_oklab

    lab = srgb_to_oklab(np.asarray(im, np.float32) / 255.0)
    flat, base = flatten(lab, 2)
    labels, cent = _oklab_quantize(flat, colors, power=0.5)
    labels = inject_details(labels, cent, lab, base, line_thr=0.04)
    if merge:
        labels = merge_regions(labels, lab, cent, min_area=merge_area, max_de=merge_de)
    pal = palette_rgb(cent)
    h, w = labels.shape

    # work at 2x so a 1px dilation only costs half a pixel at output scale
    S = max(1, int(supersample))
    if S > 1:
        labs = np.repeat(np.repeat(labels, S, axis=0), S, axis=1)
    else:
        labs = labels
    H, W = labs.shape

    counts = np.bincount(labels.ravel(), minlength=len(pal))
    order = [int(i) for i in np.argsort(-counts) if counts[i] > 0]

    fills, seams = [], []
    total = 0
    for k in order:
        mask = (labs == k).astype(np.uint8)
        if not mask.any():
            continue
        if gap > 0:
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        d = []
        h0 = hier[0] if hier is not None else []
        for ci, c in enumerate(cnts):
            area = abs(cv2.contourArea(c))
            is_hole = bool(h0[ci][3] != -1) if len(h0) > ci else False
            # tiny holes are quantisation dust - drop them (they render as white dots)
            if is_hole and area < (min_hole * S * S):
                continue
            if len(c) < 4 or area < min_area * S * S:
                continue
            pts = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
            if len(pts) < 3:
                continue
            pts /= float(S)
            if arc_fit:
                path = _arc_path(pts, arc_tol)
            else:
                if chaikin and len(pts) >= 10:
                    pts = _chaikin(pts, chaikin)
                path = _bezier(pts)
            if path:
                d.append(path)
                total += 1
        if not d:
            continue
        rgb = "#%02x%02x%02x" % tuple(int(v) for v in pal[k])
        body = "".join(d)
        fills.append('<path d="%s" fill="%s" fill-rule="evenodd"/>' % (body, rgb))

    bg = int(order[0]) if order else 0
    bgc = "#%02x%02x%02x" % tuple(int(v) for v in pal[bg])
    body = ('<rect x="0" y="0" width="%d" height="%d" fill="%s"/>\n' % (w, h, bgc))
    body += "<g>\n" + "\n".join(fills) + "\n</g>\n"
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
           'width="%d" height="%d" viewBox="0 0 %d %d">\n%s</svg>\n'
           % (w, h, w, h, body))
    return svg, dict(colors=len(fills), contours=total, grid=(w, h))
