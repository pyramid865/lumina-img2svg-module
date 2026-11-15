"""Colour vectoriser built on potrace (route A).

For every palette colour we build a 1-bit mask, run potrace on it and collect
the optimised bezier path.  potrace's curve optimisation is the same family of
algorithm the reference file was produced with: long straight runs, true arcs
and smooth beziers, with a tunable corner threshold (``alphamax``) and
tolerance (``opttolerance``).

All colour paths are emitted inside one potrace-compatible group
(``translate(0,H) scale(0.1,-0.1)``) so the coordinates stay in potrace units;
a same-colour stroke pass underneath fills the hairline seams.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageFilter

from color import palette_rgb, srgb_to_oklab

import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_POTRACE_CANDIDATES = (
    os.path.join(_HERE, "potrace.exe"),
    os.path.join(getattr(sys, "_MEIPASS", _HERE), "potrace.exe"),
    os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "potrace.exe"),
)


def find_potrace() -> str | None:
    for p in _POTRACE_CANDIDATES:
        if os.path.isfile(p):
            return p
    env = os.environ.get("POTRACE_EXE")
    if env and os.path.isfile(env):
        return env
    return None


def _mask_pbm(mask: np.ndarray, path: str) -> None:
    h, w = mask.shape
    bits = np.packbits(mask.astype(np.uint8), axis=1)
    with open(path, "wb") as fh:
        fh.write(b"P4\n%d %d\n" % (w, h))
        fh.write(bits.tobytes())


_PATH_RE = re.compile(r'<path\s+d="([^"]+)"', re.S)


def _potrace_path(exe: str, mask: np.ndarray, tmp: str, tag: int,
                  turd: int, alphamax: float, opttol: float) -> str | None:
    pbm = os.path.join(tmp, "m%d.pbm" % tag)
    out = os.path.join(tmp, "m%d.svg" % tag)
    _mask_pbm(mask, pbm)
    subprocess.run([exe, "-s", "--flat", "-t", str(turd), "-a", "%.2f" % alphamax,
                    "-O", "%.2f" % opttol, "-o", out, pbm],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   creationflags=0x08000000)  # CREATE_NO_WINDOW
    try:
        txt = open(out, encoding="utf-8").read()
    except OSError:
        return None
    m = _PATH_RE.search(txt)
    return m.group(1) if m else None


def vectorise(img: Image.Image, colors: int = 128, grid: int = 1200,
              median: int = 3, turd: int = 2, alphamax: float = 1.0,
              opttol: float = 0.2, gap: float = 0.0, grow: int = 0,
              smooth_r: int = 4, merge_area: int = 60, merge_de: float = 0.05,
              label_grid: int = 1200, ms_sp: int = 0, ms_sr: int = 0,
              dup: int = 1,
              workdir: str | None = None) -> tuple[str, dict]:
    exe = find_potrace()
    if not exe:
        raise RuntimeError("potrace.exe 未找到")
    im = img.convert("RGB")
    lg = int(label_grid or grid)
    if lg > grid:
        lg = grid
    if im.width != lg:
        im = im.resize((lg, max(1, round(im.height * lg / im.width))), Image.LANCZOS)
    if ms_sp > 0:                          # mean-shift: one flat region per blob
        arr = cv2.pyrMeanShiftFiltering(np.asarray(im), int(ms_sp), int(ms_sr),
                                        None, 1,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 5, 10.0))
        im = Image.fromarray(arr)
    if median >= 3:
        im = im.filter(ImageFilter.MedianFilter(median))

    from cel import flatten, inject_details, quantize as _q

    lab = srgb_to_oklab(np.asarray(im, np.float32) / 255.0)
    flat, base = flatten(lab, smooth_r)
    labels, cent = _q(flat, colors, power=0.5)
    labels = inject_details(labels, cent, lab, base, line_thr=0.04)
    if merge_area > 0:                      # kill mottled islands on skin/cloth
        from regionvec import merge_regions

        labels = merge_regions(labels, lab, cent, min_area=merge_area,
                               max_de=merge_de, rounds=2)
    pal = palette_rgb(cent)
    # palette colours must be the regions' real average colour: k-means
    # centres drift towards the dark side, which made every trace look washed out
    try:
        _src = np.asarray(im, np.float32).reshape(-1, 3)
        _flat_lab = labels.reshape(-1)
        for _k in range(len(pal)):
            _sel = _flat_lab == _k
            if _sel.any():
                pal[_k] = _src[_sel].mean(0).round().astype(np.uint8)
    except Exception:  # noqa: BLE001
        pass
    h, w = labels.shape
    S = max(1, int(round(grid / float(w))))
    if S > 1:                              # bigger regions, fewer speckles
        labels = np.repeat(np.repeat(labels, S, axis=0), S, axis=1)
        lab = np.repeat(np.repeat(lab, S, axis=0), S, axis=1)
        h, w = labels.shape

    counts = np.bincount(labels.ravel(), minlength=len(pal))
    order = [int(i) for i in np.argsort(-counts) if counts[i] > 0]

    fills, seams = [], []
    base_dir = workdir or os.environ.get("POTRACE_TMP") or os.getcwd()
    os.makedirs(base_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="ptr_", dir=base_dir)
    used = 0
    try:
        for n, k in enumerate(order):
            mask = (labels == k).astype(np.uint8)
            if not mask.any():
                continue
            if grow > 0:                       # overlap 1px: kills hairline gaps
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=grow)
            d = _potrace_path(exe, mask, tmp, n, turd, alphamax, opttol)
            if not d:
                continue
            rgb = "#%02x%02x%02x" % tuple(int(v) for v in pal[k])
            fills.append('<path d="%s" fill="%s"/>' % (d, rgb))
            if gap > 0:
                # coordinates are scaled by 0.1 inside the group -> width x10
                seams.append('<path d="%s" fill="none" stroke="%s" stroke-width="%.1f" '
                             'stroke-linejoin="round"/>' % (d, rgb, gap * 10.0))
            used += 1
    finally:
        # robust cleanup: shutil.rmtree handles subdirs and locked files that a
        # manual remove+rmdir loop silently leaves behind (the source of the
        # stale ptr_* dirs left in the project root).
        shutil.rmtree(tmp, ignore_errors=True)

    bgk = int(order[0]) if order else 0
    bgc = "#%02x%02x%02x" % tuple(int(v) for v in pal[bgk])
    body = '<rect x="0" y="0" width="%d" height="%d" fill="%s"/>\n' % (w, h, bgc)
    if seams and dup:
        body += '<g transform="translate(0,%d) scale(0.1,-0.1)">\n%s\n</g>\n' % (
            h, "\n".join(seams))
    body += '<g transform="translate(0,%d) scale(0.1,-0.1)">\n%s\n</g>\n' % (
        h, "\n".join(fills))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
           'width="%d" height="%d" viewBox="0 0 %d %d">\n%s</svg>\n'
           % (w, h, w, h, body))
    return svg, dict(colors=used, contours=used, grid=(w, h))
