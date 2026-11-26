# -*- coding: utf-8 -*-
"""区域分割重绘引擎 (MODE_SEGREGION) — v7.17.5 引擎级重写原型转正。

设计（与旧 flatfill 根本区别）：
  旧：先把整图切成 170 色 → 事后合并碎块。平滑区渐变天生被切成色带，补救无效。
  新：先按源图做区域分割（felzenszwalb，区域边界自然落在图像结构上）
      → 每个区域直接用原图像素取主色 → 再描摹。平滑区天然是一整块，无色带。

关键实证（五条悟下巴区）：
  群友主色占比 59.9%，我们 fusion 24.2%；本引擎在 4× 画布上脸部明显成整块。
  4× 分辨率是必要条件：同算法 900px 上五官被吞（强边 31%），3600px 强边 69.6%。

定位：**手动可选档，默认 auto 不变**。色准（RMSE 16.58）不如保真线 fusion（11.79），
但平滑区干净度接近群友；适合"嫌脸上/背景碎块多、愿意牺牲一点色准"的图。
复杂图耗时长（约 2~3 分钟，4× 画布），已在 UI 文案写明。

零复制搬运：只用项目已有自研资产（core._vtrace / core._set_root / color），
区域分割用运行时自带 skimage.segmentation.felzenszwalb（BSD 许可，随环境安装）。
"""
from __future__ import annotations

import io
import os
import tempfile
import time

import numpy as np
from PIL import Image, ImageFilter
from skimage.segmentation import felzenszwalb

import core

SEG_SCALE = 60          # felzenszwalb 区域尺度（越小区域越多、细节越全）
SEG_MIN_SIZE = 40       # 最小区域像素数（4× 画布上）
CANVAS_SCALE = 4.0      # 内部画布 = 原图 4×（对标群友 viewBox 4×）


def region_colors(seg: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """向量化按区域取原图均值色（bincount，1800 万像素秒级）。"""
    flat = seg.ravel()
    n = int(seg.max()) + 1
    area = np.bincount(flat, minlength=n)
    f = arr.reshape(-1, 3).astype(np.float32)
    out = np.zeros((n, 3), np.float32)
    for ch in range(3):
        out[:, ch] = np.bincount(flat, weights=f[:, ch], minlength=n) / np.maximum(area, 1)
    return np.clip(out, 0, 255).astype(np.uint8)


def trace(src: str, dst: str, colors: int = 0, grid: int = 0,
          progress=None) -> dict:
    """区域分割重绘：src -> dst SVG，返回统计 dict。"""
    t0 = time.time()

    def step(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    step("读取图片…")
    src_im = Image.open(src).convert("RGB")
    ow, oh = src_im.size

    # 4× 内部画布（群友做法：viewBox 4×、显示 1×），不缩回去
    iw = max(256, int(round(ow * CANVAS_SCALE)))
    ih = max(1, int(round(oh * CANVAS_SCALE)))
    work = src_im.resize((iw, ih), Image.LANCZOS)

    step("区域分割（4× 画布 · 边界落在图像结构上）…")
    arr = np.asarray(work)
    seg = felzenszwalb(arr, scale=SEG_SCALE, sigma=1.0, min_size=SEG_MIN_SIZE)
    n_reg = int(seg.max()) + 1
    step("区域取色（每个区域用原图主色）…")
    pal = region_colors(seg, arr)

    # 所有区域都填自己的颜色（含小区域）——绝不留黑色洞（RMSE 43 的坑）
    img = pal[seg]
    work2 = Image.fromarray(img, "RGB").filter(ImageFilter.MedianFilter(3))

    tmp = tempfile.mkdtemp(prefix="segregion_")
    png = os.path.join(tmp, "in.png")
    raw = os.path.join(tmp, "out.svg")
    try:
        work2.save(png, "PNG", compress_level=1)
        step("曲线描摹（4× 画布）…")
        core._vtrace(png, raw, **core._ref_kwargs(core.Settings(mode=core.MODE_REF)))
        svg = io.open(raw, encoding="utf-8").read()
    finally:
        for p in (png, raw):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass

    svg = core._set_root(svg, work2.width, work2.height,
                         ow / work2.width, "Region-segmented artwork (4x canvas)")
    with io.open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)

    fills = set(__import__("re").findall(r'fill="(#[0-9a-fA-F]{6})"', svg))
    return dict(mode="segregion", src=src, dst=dst,
                colors=len(fills), paths=svg.count("<path"),
                regions=n_reg, bytes=os.path.getsize(dst),
                seconds=round(time.time() - t0, 1), hd="4.0x")
