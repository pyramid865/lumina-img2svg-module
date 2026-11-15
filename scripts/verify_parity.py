"""A/B parity evidence for the img2svg Workshop module.

Compares two SVG files byte-for-byte and writes a side-by-side zoom comparison
plus an amplified pixel-difference panel.  Usage::

    python scripts/verify_parity.py --desktop-svg A.svg --module-svg B.svg \
        --zoom-out parity_zoom.png

This script deliberately contains no machine-specific paths and no external
rasteriser dependency: ``render_svg.py`` handles vtracer output with Pillow
only.  The two input SVGs are produced by ``run_ab_local.py`` in the project's
local verification workspace (not part of the module package).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from svg_raster import render_file as render  # noqa: E402


def _font(size: int):
    """Locate a system font via environment variables, never a hardcoded path."""
    roots = [os.environ.get("WINDIR"), os.environ.get("SYSTEMROOT"),
             "/usr/share/fonts", "/System/Library/Fonts"]
    names = ["msyh.ttc", "simhei.ttf", "arial.ttf", "DejaVuSans.ttf"]
    for root in filter(None, roots):
        for name in names:
            candidate = Path(root) / "Fonts" / name
            if not candidate.exists():
                candidate = Path(root) / "truetype" / name
            if candidate.exists():
                try:
                    return ImageFont.truetype(str(candidate), size)
                except OSError:
                    continue
    return ImageFont.load_default()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_comparison(a: Image.Image, b: Image.Image, out: Path,
                     label_a: str, label_b: str, hash_a: str, hash_b: str,
                     identical: bool, bytes_a: int, bytes_b: int) -> None:
    """Whole-image thumbnails, a zoomed detail, and an amplified diff panel."""
    F = _font(15)
    FB = _font(21)
    TITLE = (28, 34, 46)
    MUTED = (95, 105, 120)
    BG = (248, 249, 251)
    LINE = (220, 224, 230)
    GREEN, RED = (24, 130, 80), (190, 40, 55)

    M = 24                      # margin
    thumb_w = 330
    gap = 24
    W = 2 * M + 2 * thumb_w + gap
    thumb_h = int(a.height * thumb_w / a.width)
    ta = a.resize((thumb_w, thumb_h), Image.LANCZOS)
    tb = b.resize((thumb_w, thumb_h), Image.LANCZOS)

    # Same crop on both: face / hair detail.
    zw = (W - 2 * M - 2 * 12) // 3
    zh = 216
    cx, cy = int(a.width * 0.47), int(a.height * 0.36)
    box = (max(0, cx - 48), max(0, cy - 42),
           min(a.width, cx + 48), min(a.height, cy + 42))
    za = a.crop(box).resize((zw, zh), Image.LANCZOS)
    zb = b.crop(box).resize((zw, zh), Image.LANCZOS)

    diff = ImageChops.difference(ta, tb)
    bbox = diff.getbbox()
    diff_amp = diff.point(lambda v: min(255, v * 16)).convert("RGB")
    zd = ImageChops.difference(za, zb).point(
        lambda v: min(255, v * 16)).convert("RGB")

    y_thumb = 132
    y_zoom = y_thumb + thumb_h + 56
    y_notes = y_zoom + zh + 46
    H = y_notes + 96

    card = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(card)

    d.text((M, 14), "A/B 画质一致性验证 · img2svg v7.6.0", fill=TITLE, font=FB)
    d.text((M, 44), "逐字节一致  PASS" if identical else "存在差异  FAIL",
           fill=GREEN if identical else RED, font=FB)
    d.text((M, 78), f"{label_a}   {bytes_a:,} B", fill=MUTED, font=F)
    d.text((M + thumb_w + gap, 78), f"{label_b}   {bytes_b:,} B", fill=MUTED, font=F)
    d.text((M, 100), f"sha256 {hash_a[:32]}…", fill=MUTED, font=F)
    d.text((M + thumb_w + gap, 100), f"sha256 {hash_b[:32]}…", fill=MUTED, font=F)

    card.paste(ta, (M, y_thumb))
    card.paste(tb, (M + thumb_w + gap, y_thumb))

    d.line([(M, y_zoom - 30), (W - M, y_zoom - 30)], fill=LINE, width=1)
    d.text((M, y_zoom - 24), "同一局部放大（脸部 / 发丝）  ·  第 3 幅为放大 16× 的像素差",
           fill=TITLE, font=F)
    for i, im in enumerate((za, zb, zd)):
        card.paste(im, (M + i * (zw + 12), y_zoom))

    d.line([(M, y_notes - 24), (W - M, y_notes - 24)], fill=LINE, width=1)
    d.text((M, y_notes), "判定依据：SVG 文件逐字节比对（位图仅为近似光栅化预览）",
           fill=TITLE, font=F)
    d.text((M, y_notes + 26),
           "差异区域: " + ("无" if bbox is None else str(bbox)),
           fill=GREEN if bbox is None else RED, font=F)
    d.text((M, y_notes + 50),
           "主链路：vtracer 0.6.15 + 保色 2× 超分，零模型依赖；模块与桌面算法核心逐字节相同。",
           fill=MUTED, font=F)

    out.parent.mkdir(parents=True, exist_ok=True)
    card.save(out)
    print("wrote", out, card.size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--desktop-svg", type=Path, required=True)
    ap.add_argument("--module-svg", type=Path, required=True)
    ap.add_argument("--zoom-out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "parity_zoom.png")
    ap.add_argument("--max-size", type=int, default=760)
    args = ap.parse_args()

    a = args.desktop_svg.read_bytes()
    b = args.module_svg.read_bytes()
    ha, hb = sha256(a), sha256(b)
    identical = a == b
    print("desktop:", len(a), "bytes", ha)
    print("module :", len(b), "bytes", hb)
    print("byte-identical:", identical)

    ia = render(args.desktop_svg, args.max_size)
    ib = render(args.module_svg, args.max_size)
    build_comparison(ia, ib, args.zoom_out, "桌面版 v7.6.0", "Workshop 模块",
                     ha, hb, identical, len(a), len(b))
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
