"""保真高清化：把模糊处补足，但不改变原图已有的清晰度与颜色。

与旧的 ``aihd.preprocess`` 的区别（旧实现会先把半尺寸模糊图以 35% 比例混进
原图，等于主动糊掉整张图，再把整图交给网络重绘，因此线条粗细 / 颜色都可能
被"高清化"改掉）：

1. 不做全局 grain-kill 混合，原图直接送进超分网络；
2. 超分结果**只取高频残差**（``sr - blur(sr)``），低频颜色与明暗完全由
   原图的 LANCZOS 放大结果提供 —— 颜色不会被改变；
3. 按原图的**局部锐度**生成权重：本来就清晰的区域权重趋近 0（原样保留），
   模糊区域权重高（注入超分补出的细节）；
4. 高频残差限幅 + **下采样一致性回退**：把结果缩回原尺寸与原图比对，若误差
   超过阈值就自动收缩权重重算，保证"清晰度不发生转变"可量化、可兜底。

只依赖 numpy / Pillow / aihd(onnxruntime)。模型不可用时退化为纯 LANCZOS
放大（绝不会破坏原图）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter

import aihd


def _gray(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), np.float32)


def _sharp_map(im: Image.Image, radius: float = 2.0, box: int = 3) -> np.ndarray:
    """局部高频能量：越大表示这块本来就越清晰（不该被改动）。"""
    g = _gray(im)
    b = np.asarray(im.convert("L").filter(ImageFilter.GaussianBlur(radius)), np.float32)
    hi = np.clip(np.abs(g - b), 0, 255).astype(np.uint8)
    return np.asarray(
        Image.fromarray(hi, "L").filter(ImageFilter.BoxBlur(box)), np.float32)


def _blur_rgb(im: Image.Image, radius: float) -> np.ndarray:
    return np.asarray(im.filter(ImageFilter.GaussianBlur(radius)), np.float32)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))


def enhance(im: Image.Image, scale: float = 2.0, strength: float = 1.0,
            max_pixels: int = 16_000_000, tile: int = 192, overlap: int = 12,
            detail_clip: float = 24.0, consistency_limit: float = 3.0,
            progress=None) -> tuple[Image.Image, dict[str, Any]]:
    """返回 ``(高清图, 指标 dict)``。

    consistency_limit: 允许的下采样 RMSE（0-255）；超限时自动收缩注入强度。
    """
    src = im.convert("RGB")
    W, H = src.size
    s = float(scale)
    sW, sH = int(round(W * s)), int(round(H * s))
    if sW * sH > max_pixels:
        k = (max_pixels / float(sW * sH)) ** 0.5
        sW, sH = max(1, int(sW * k)), max(1, int(sH * k))
    real_scale = sW / float(W)

    def step(msg):
        if progress:
            progress(msg)

    base = src.resize((sW, sH), Image.LANCZOS)
    orig = np.asarray(src, np.float32)
    A = np.asarray(base, np.float32)
    meta: dict[str, Any] = dict(width=sW, height=sH, scale=round(real_scale, 3))

    if not aihd.available() or strength <= 0:
        meta.update(model="none (LANCZOS fallback)", down_rmse=0.0,
                    color_shift=0.0, strength=0.0)
        return base, meta

    step("AI 超分（REALEsr animevideov3 x4）…")
    sr4 = aihd.upscale(src, tile=int(tile), overlap=int(overlap))
    sr = sr4.resize((sW, sH), Image.LANCZOS)

    step("颜色低频锚定 + 清晰度加权融合…")
    # 低频半径随放大倍率增长，保证取到的是"细节"而不是整体色调
    r = max(1.5, 2.0 * real_scale)
    detail = np.asarray(sr, np.float32) - _blur_rgb(sr, r)
    detail = np.clip(detail, -float(detail_clip), float(detail_clip))

    sm = _sharp_map(src)
    lo, hi = np.percentile(sm, [10, 90])
    t = np.clip((sm - lo) / max(1e-3, hi - lo), 0.0, 1.0)
    # 清晰区 -> 0（原样保留）；模糊区 -> 1（补细节）；平方使过渡更陡
    wmap = (1.0 - t) ** 2
    wmap = np.asarray(
        Image.fromarray((wmap * 255).astype(np.uint8), "L").resize(
            (sW, sH), Image.BICUBIC), np.float32) / 255.0
    w = (wmap * float(strength))[..., None]

    # 分区统计用的掩膜（原图尺度）
    clear = t > 0.7
    blur = t < 0.3
    sb = float(sm[clear].mean()) if clear.any() else 0.0
    sb_blur = float(sm[blur].mean()) if blur.any() else 0.0

    def compose(k: float) -> Image.Image:
        out = np.clip(A + detail * w * k, 0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGB")

    k = 1.0
    out_im = compose(k)
    down = np.asarray(out_im.resize((W, H), Image.LANCZOS), np.float32)
    rmse = _rms(down, orig)
    # 一致性兜底：改动过大就收缩，最多 4 次
    for _ in range(4):
        if rmse <= consistency_limit:
            break
        k *= max(0.25, consistency_limit / max(rmse, 1e-3) * 0.85)
        out_im = compose(k)
        down = np.asarray(out_im.resize((W, H), Image.LANCZOS), np.float32)
        rmse = _rms(down, orig)

    # 低频颜色偏差（缩回原尺寸后比较高斯模糊层）
    color_shift = _rms(
        np.asarray(Image.fromarray(down.astype(np.uint8), "RGB").filter(
            ImageFilter.GaussianBlur(2.0)), np.float32),
        np.asarray(src.filter(ImageFilter.GaussianBlur(2.0)), np.float32))

    # 分区锐度：在缩回原尺寸的图上量，验证"清晰区不变、模糊区提升"
    dsharp = _sharp_map(Image.fromarray(down.astype(np.uint8), "RGB"))
    sa = float(dsharp[clear].mean()) if clear.any() else 0.0
    sa_blur = float(dsharp[blur].mean()) if blur.any() else 0.0

    meta.update(
        model="realesr-animevideov3 x4 + 保真融合",
        strength=round(k * float(strength), 3),
        down_rmse=round(rmse, 3),
        color_shift=round(color_shift, 3),
        sharp_clear_before=round(sb, 2), sharp_clear_after=round(sa, 2),
        sharp_clear_keep=round(100.0 * sa / max(sb, 1e-3), 1),
        sharp_blur_before=round(sb_blur, 2), sharp_blur_after=round(sa_blur, 2),
        sharp_blur_gain=round(100.0 * (sa_blur - sb_blur) / max(sb_blur, 1e-3), 1),
        detail_clip=float(detail_clip),
    )
    return out_im, meta


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        img = Image.open(sys.argv[1]).convert("RGB")
        out, m = enhance(img, scale=float(sys.argv[3]) if len(sys.argv) > 3 else 2.0,
                         progress=lambda x: print(" ", x))
        out.save(sys.argv[2])
        print(m)
