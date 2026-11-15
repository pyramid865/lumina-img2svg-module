# -*- coding: utf-8 -*-
"""瓦片回填补丁（产品内置版）。

来源：`_dev/svg_tilefix.py`。修复「分块 vtracer」SVG 露出的瓦片底色卡：
这类 SVG 顶部有几百个规则 <rect>，每块取该瓦片平均色，相邻块略有色差 ->
平滑渐变区出现矩形方块边界 / 细线。直接删掉会露出全屏 bg 变成大片错色。

本模块只把每个瓦片 rect 的颜色换成**原图同位置的局部均值**：相邻块颜色
连续过渡，方块与细线消失；不删任何元素、不动任何 path，绝不露底。

对外暴露两个纯函数（字符串进 / 字符串出，不碰文件 IO），方便 GUI 与 CLI 复用：
    has_tiles(svg_text) -> bool          # 是否含大量小瓦片 rect
    fix(svg_text, image_path) -> text    # 回填瓦片颜色，返回新 SVG 文本
"""
from __future__ import annotations

import io
import re

import numpy as np
from PIL import Image

_FULL_BG_RATIO = 0.45  # 边长超过 viewBox 此比例的 rect 视为全屏背景，保留
_MIN_TILES = 20        # 少于这么多小瓦片就不算「瓦片风格」


def _viewbox(svg_text):
    m = re.search(r'viewBox="([^"]+)"', svg_text)
    if not m:
        return None
    v = [float(x) for x in m.group(1).replace(",", " ").split()]
    if len(v) < 4:
        return None
    return v[2], v[3]


def has_tiles(svg_text: str, min_count: int = _MIN_TILES) -> bool:
    """Return True if the SVG contains many small tile <rect> elements."""
    rects = re.findall(r"<rect\b[^>]*/>", svg_text)
    if len(rects) < min_count:
        return False
    vb = _viewbox(svg_text)
    if not vb:
        return False
    vbw, vbh = vb
    small = 0
    for r in rects:
        wm = re.search(r'\bwidth="([\d.]+)"', r)
        hm = re.search(r'\bheight="([\d.]+)"', r)
        if not wm or not hm:
            continue
        if float(wm.group(1)) < _FULL_BG_RATIO * vbw and \
           float(hm.group(1)) < _FULL_BG_RATIO * vbh:
            small += 1
    return small >= min_count


def fix(svg_text: str, image_path: str) -> str:
    """Replace each small tile <rect> fill with the local mean colour of *image_path*."""
    im = Image.open(image_path).convert("RGB")
    iw, ih = im.size
    vb = _viewbox(svg_text)
    if vb:
        vb_w, vb_h = vb
        sx, sy = vb_w / iw, vb_h / ih
    else:
        sx = sy = 1.0
        vb_w, vb_h = iw, ih

    n = 0

    def repl(m):
        nonlocal n
        tag = m.group(0)
        try:
            x = float(re.search(r'\bx="([\d.]+)"', tag).group(1))
            y = float(re.search(r'\by="([\d.]+)"', tag).group(1))
            w = float(re.search(r'\bwidth="([\d.]+)"', tag).group(1))
            h = float(re.search(r'\bheight="([\d.]+)"', tag).group(1))
        except AttributeError:
            return tag
        # 全屏背景 rect 保留原色
        if w >= _FULL_BG_RATIO * vb_w and h >= _FULL_BG_RATIO * vb_h:
            return tag
        cx, cy = (x + w / 2) / sx, (y + h / 2) / sy
        r = max(2.0, min(w / sx, h / sy) * 0.45)
        box = (max(0, int(cx - r)), max(0, int(cy - r)),
               min(iw, int(cx + r)), min(ih, int(cy + r)))
        if box[2] <= box[0] or box[3] <= box[1]:
            return tag
        rgb = tuple(int(v) for v in np.asarray(
            im.crop(box).resize((1, 1), Image.BOX), np.uint8).reshape(-1)[:3])
        n += 1
        return re.sub(r'fill="#[0-9a-fA-F]{6}"',
                      'fill="#%02x%02x%02x"' % rgb, tag)

    out = re.sub(r"<rect\b[^>]*(?:/>|></rect>)", repl, svg_text)
    return out, n


def fix_file(svg_path: str, image_path: str, dst: str | None = None) -> int:
    """Convenience: fix an on-disk SVG file. Returns number of tiles patched."""
    s = io.open(svg_path, encoding="utf-8", errors="ignore").read()
    out, n = fix(s, image_path)
    if dst is None:
        dst = svg_path
    io.open(dst, "w", encoding="utf-8", newline="\n").write(out)
    return n
