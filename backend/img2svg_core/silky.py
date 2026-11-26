"""顺滑优先引擎 (MODE_SILKY) — v7.18.0 完全自研。

设计目标（用户 2026-11-17 明确取舍）：
    为「线条顺滑 + 色块过渡顺滑 + 色块边缘顺滑」，主动牺牲一部分精细度，
    把图像里相邻的小碎色块合并掉。

与 fusion (flatfill.py) 的关系：
    fusion  = OKLab 量化 -> vtracer 描摹 -> 色板吸附 -> 补缝（保真，碎线多）
    silky   = OKLab 量化 -> 五官钉色 -> 近色加权归并(合并相邻小色块)
              -> 区域均值取色 -> vtracer 描摹 -> 色板吸附 -> 补缝（顺滑，细节略减）

关键：
  - 用 smartmerge.pin_features 钉住瞳孔/嘴角(小暗核)、眼高光/齿白(极小亮核)，
    归并时这些特征只许并同色 -> 五官不糊。
  - 用 smartmerge.merge_fragments 的「共享边长/(16+OKLab色距)」近色加权投票，
    把小于 area 的碎块并给最像的大邻居 -> 相邻小色块被吃掉、色块成整片。
  - 仍走 vtracer cutout/spline 描摹（不用会掰断细长特征的长弧重拟合），
    所以眼线/睫毛保持连贯，不会像 flatstyle 那样出锯齿断裂。

零复制搬运：全部为项目自研资产（color.py / smartmerge.py / flatfill.py / core.py）。

对外接口：
    trace(src, dst, colors=112, hd_scale=2.0, strength='medium', ...) -> dict
    strength: 'light'(area80) / 'medium'(area120) / 'strong'(area240)
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import time

import numpy as np
from PIL import Image, ImageFilter

import color
import core
import flatfill
import smartmerge

# 归并强度 -> min_area（在 2x 高清网格上；对应网页立绘档 area30 * 4）
_STRENGTH_AREA = {"light": 80, "medium": 120, "strong": 240}


def trace(src: str, dst: str, colors: int = 112, hd_scale: float = 2.0,
          grid: int = 1600, seam: float = 0.8, median: int = 3,
          strength: str = "medium", progress=None) -> dict:
    """把 *src* 顺滑化描摹成 *dst* SVG，返回统计信息 dict。"""
    t0 = time.time()
    w = flatfill.DEFAULT_W
    area = _STRENGTH_AREA.get(str(strength), 120)
    src_im = Image.open(src).convert("RGB")
    ow, oh = src_im.size

    # 高清化 + 统一网格 + 中值硬边（与 fusion 完全一致的预处理）
    scale = max(1.0, float(hd_scale))
    work = src_im.resize((int(round(ow * scale)), max(1, int(round(oh * scale)))),
                         Image.LANCZOS)
    g = max(256, int(grid))
    if work.width != g:
        work = work.resize((g, max(1, round(work.height * g / work.width))), Image.LANCZOS)
    if median >= 3:
        work = work.filter(ImageFilter.MedianFilter(median if median % 2 else median + 1))

    if progress:
        progress("OKLab 感知量化（%d 色 · 顺滑优先）…" % colors)
    pal, q = flatfill.quantize_palette(work, colors, w)

    # 量化图 -> 标签（每个像素归属到最近调色板色）
    # v7.17.3：改用分块版，避免 H×W×K×3 广播的 58GB 峰值；结果逐像素一致
    labels = flatfill.labels_of_quantized(pal, q, w)
    arr = np.asarray(work)

    if progress:
        progress("五官钉色 + 近色加权归并（合并相邻小色块）…")
    # ws2: 归并面积阈值按高清倍率的平方放大，保证不同倍率下碎块尺度一致
    ws2 = max(1, int(round(scale * scale)))
    lb, pb, guard = smartmerge.pin_features(labels, arr, pal, ws2=ws2)
    lm = smartmerge.merge_fragments(lb, pb, min_area=area * max(1, ws2 // 4),
                                    guard=guard, iters=2)
    # 归并后每个区域取原图真实均值色（取色更准）
    for kk in range(len(pb)):
        m = lm == kk
        if m.any():
            pb[kk] = arr[m].mean(0).round().astype(np.uint8)
    smooth_img = Image.fromarray(pb[lm].astype(np.uint8), "RGB")

    tmp = tempfile.mkdtemp(prefix="silky_")
    png = os.path.join(tmp, "trace_in.png")
    raw = os.path.join(tmp, "trace_out.svg")
    smooth_img.save(png, "PNG", compress_level=1)

    if progress:
        progress("曲线描摹（cutout 共边 + 连续线条）…")
    core._vtrace(png, raw, **core._ref_kwargs(core.Settings(mode=core.MODE_REF)))
    svg = io.open(raw, encoding="utf-8").read()

    if progress:
        progress("色板吸附 + 补缝描边…")
    svg = flatfill.snap_palette(svg, pb, w)
    svg = core._add_seam_strokes(svg, seam)
    disp_scale = ow / work.width if work.width else 1.0
    svg = core._set_root(svg, work.width, work.height, disp_scale,
                         "Silky traced artwork (%dx%d grid, %s)" % (work.width, work.height, strength))

    io.open(dst, "w", encoding="utf-8", newline="\n").write(svg)
    fills = set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg))
    return dict(mode="silky", src=src, dst=dst, colors=len(fills),
                paths=svg.count("<path"), bytes=os.path.getsize(dst),
                seconds=round(time.time() - t0, 1), hd="%.1fx" % scale,
                strength=strength, area=area)
