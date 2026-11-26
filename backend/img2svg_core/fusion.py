# -*- coding: utf-8 -*-
"""双超分融合（v7.6.1）：aihd 洗色超分拿平滑几何 + 原图色板重映射拿正确颜色。

背景（实测，详见工作日志 2026-11-15）：
  * v7.5 旧超分 Real-ESRGAN 在洗淡颜色的同时把图"洗平滑" → vtracer 描摹碎片
    少、边缘顺（jagged 33.25），但颜色发灰（sat 0.366）。
  * v7.5.7 换保色超分后颜色对了（sat 0.439），但图变锐/碎 → 描摹碎片翻倍、
    像素阶梯暴露 = 边缘毛糙（jagged 35.80）。
  * 正解：几何走洗色超分（平滑），颜色用原图重映射补回 —— 二者兼得。

关键实现约束（都是踩过的坑）：
  1. 描摹后必须有 0.6px 同色 stroke 覆盖色块缝隙（core.lumina 已有）；裸几何
     jagged 39.2，加 stroke 立刻降到 33.4。
  2. 重映射**只换 fill、绝不换 stroke**。stroke 是平滑覆盖层，换色会把 jagged
     从 35.0 破坏到 37.9。
  3. 色板先 merge 到 ~256 色（大色块），再重映射，保证每个色覆盖区够大、零漏映射。
  4. 重映射必须 O(N) 查表：逐色全图 mask 是 256×500万≈13亿次比较，实测 8 分钟；
     查表法 1.8 秒。

本模块只负责"重映射"这一步（渲染依赖 resvg_py）；洗色超分复用 aihd.preprocess。
"""
from __future__ import annotations

import io
import re
from typing import Optional

import numpy as np
from PIL import Image

_FILL_RE = re.compile(r'fill="(#[0-9a-fA-F]{6})"')
_STROKE_RE = re.compile(r'stroke="(#[0-9a-fA-F]{6})"')


def available() -> bool:
    """resvg_py 渲染依赖是否可用（打包后必须带上 resvg_py.pyd）。"""
    try:
        import resvg_py  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _render(svg: str, w: int, h: int) -> np.ndarray:
    import resvg_py

    raw = resvg_py.svg_to_bytes(svg_string=svg, width=w, height=h,
                                background='#ffffff')
    return np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'), np.uint8)


def _root_size(svg: str) -> tuple[int, int]:
    m = re.search(r'<svg\b[^>]*?width="([\d.]+)"[^>]*?height="([\d.]+)"', svg)
    if m:
        return int(float(m.group(1))), int(float(m.group(2)))
    m = re.search(r'viewBox="\s*[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)"', svg)
    if m:
        return int(float(m.group(1))), int(float(m.group(2)))
    return 0, 0


def recolor_svg(svg: str, src_path: str, min_px: int = 8) -> tuple[str, dict]:
    """把 SVG 的 fill 色板重映射到原图对应区域的中位色。

    * 只替换 fill，stroke 原样保留（见模块文档坑 2）。
    * O(N) 查表：RGB→int24→LUT 一次归类→argsort 分组取中位色。
    * 大图限制到 600 万像素以内渲染，避免大画布内存/耗时暴涨。

    返回 (new_svg, info)；info 含 mapped/missed/size 便于日志与回归。
    """
    w, h = _root_size(svg)
    if not w or not h:
        raise ValueError("SVG 缺少可解析的 width/height 或 viewBox")
    # 渲染上限：超大画布等比缩小；fill 映射与分辨率无关，缩小不影响颜色统计。
    CAP = 6_000_000
    rw, rh = w, h
    if rw * rh > CAP:
        k = (CAP / float(rw * rh)) ** 0.5
        rw, rh = max(1, int(rw * k)), max(1, int(rh * k))

    fills = sorted(set(_FILL_RE.findall(svg)))
    if not fills:
        return svg, dict(mapped=0, missed=0, size="%dx%d" % (w, h))

    rendered = _render(svg, rw, rh)
    with Image.open(src_path) as im:
        orig = im.convert('RGB').resize((rw, rh), Image.LANCZOS)
    ob = np.asarray(orig, np.uint8)

    # RGB -> int24 打包
    fr = np.array([int(c[1:3], 16) for c in fills], np.int32)
    fg = np.array([int(c[3:5], 16) for c in fills], np.int32)
    fb = np.array([int(c[5:7], 16) for c in fills], np.int32)
    packed = (fr << 16) | (fg << 8) | fb
    flat = ((rendered[..., 0].astype(np.int32) << 16)
            | (rendered[..., 1].astype(np.int32) << 8)
            | rendered[..., 2].astype(np.int32))

    lut = np.full(1 << 24, -1, np.int32)
    lut[packed] = np.arange(len(fills), dtype=np.int32)
    idx = lut[flat]                       # (rh,rw) 每像素归属哪个 fill

    # 取"内部像素"（周围 5×5 都属同一 fill），避开抗锯齿边界混色对中位色的污染
    inner = np.ones_like(idx, bool)
    for dy in (-2, -1, 1, 2):
        rs = np.roll(idx, dy, axis=0)
        for dx in (-2, -1, 1, 2):
            inner &= (np.roll(rs, dx, axis=1) == idx)
    valid = (idx >= 0) & inner

    vi = idx[valid]
    obs = ob.reshape(-1, 3)[valid.ravel()]
    counts = np.bincount(vi, minlength=len(fills))
    order = np.argsort(vi, kind='stable')
    sorted_obs = obs[order]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    mp: dict[str, str] = {}
    missed = 0
    for i, c in enumerate(fills):
        n = int(counts[i])
        if n < min_px:                    # 覆盖区太小 → 保留原色（通常是极细线）
            missed += 1
            continue
        nc = np.median(sorted_obs[starts[i]:starts[i] + n], axis=0).astype(int)
        mp[c] = '#%02x%02x%02x' % tuple(int(v) for v in nc)

    new_svg = _FILL_RE.sub(lambda m: 'fill="%s"' % mp.get(m.group(1), m.group(1)), svg)
    return new_svg, dict(mapped=len(mp), missed=missed,
                         total=len(fills), size="%dx%d" % (rw, rh))
