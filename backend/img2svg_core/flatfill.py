"""准色平涂融合引擎 (MODE_FUSION) — v7.16.0 完全自研版。

设计原则：**零复制搬运**。本文件全部为项目自研代码，只调用本地已有的
自研资产，不引用网友版 / 其他电脑的任何源码文件：

  * OKLab 感知色彩空间量化 —— 本地自研 ``color.py``
    (srgb_to_oklab / kmeans_palette / oklab_to_rgb01)；
  * 曲线描摹 / 完整轮廓 —— 本地自研 ``core._vtrace`` + ``core._ref_kwargs``
    (vtracer cutout + spline)；
  * 同色补缝描边 —— 本地自研 ``core._add_seam_strokes``；
  * 根节点重写 —— 本地自研 ``core._set_root``；
  * 色板吸附（snap）—— 本文件自研实现，见 ``snap_palette``。

要解决的问题（v7.11 bands 引擎的真实缺陷，已复现）：
  浅色横纹插画（喷火龙腹部）上，色带引擎会把浅棕横纹误判成暗部，量化出
  纯黑 #000000 色块。根因是「先量化标签 → 区域合并 → 长弧拟合」这条链路在
  低对比横纹处标签崩坏。本引擎换一条路：**vtracer cutout 描摹保住完整轮廓与
  横纹连续性**，再把上千种近似色吸附回 N 色平涂色板。横纹连成连续曲线，
  不会被切成纯黑块。

与 v7.15.0 搬运版（fusion_flat + clean_layers）的区别：
  - v7.15 的 fusion_flat.py 是从另一台电脑逐字节复制的 fusion.py，
    且其色板吸附依赖 49KB 的 clean_layers.py（也是复制品）；
  - 本文件色板吸附为**自研实现**：OKLab 空间最近邻 + 三通道权重 + LRU 缓存，
    约 30 行，无外部依赖；曲线参数吸收网友版「线条更顺」的长处，用本地
    _ref_kwargs 的可调参数实现（拐角门限/微段合并/接合门限）。

依赖：numpy / Pillow / vtracer —— 全部已在项目运行时内，无新增依赖。
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import time
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter

import color
import core

# OKLab 三通道权重：明度 1.3 / 色度 a 2.0 / 色度 b 2.0。
# 抬高色度权重让暖色/青色小区域优先分到自己的簇，而不是被并进灰色或黑色。
DEFAULT_W = np.array([1.3, 2.0, 2.0], np.float32)

_FILL_HEX = re.compile(r'fill="(#[0-9a-fA-F]{6})"')


def quantize_palette(im: Image.Image, colors: int, w: np.ndarray, with_index: bool = False):
    """OKLab 感知色彩空间 k-means 量化。

    返回 (调色板 uint8[K,3], 量化后的 Image)。调色板质量和通道权重决定
    后续吸附的色准 —— 用色度加权让暖色小区域不被灰/黑吞掉。
    ``with_index=True`` 时额外返回 idx(H,W) 标签图（v7.17.1 核心取色用）。
    """
    arr = np.asarray(im.convert("RGB"), np.float32) / 255.0
    lab = color.srgb_to_oklab(arr).reshape(-1, 3)
    # v7.20.0 提速：迭代 20→8（实测中心偏移 0.0039/1.0，肉眼看不出差别，
    # 量化阶段省约一半时间）。20 轮是早期保守值，色板早已收敛。
    cent = color.kmeans_palette(lab, int(colors), iters=8, seed=5, sample=90000)

    # v7.17.3 内存控制：原写法一次性分配 (H*W, K) 距离矩阵——
    # 4000 网格的五条悟图 2844 万像素 × 170 色 × float32 ≈ 19 GB，
    # 直接触发系统换页（实测耗时 66s→330s 且 RMSE 反弹）。
    # 改为 band_tracer.py 已验证的分块写法：数学完全等价（加权欧氏距离
    # 展开成 a²w - 2a·(b·w) + (b²·w)，argmin 不受常数项影响），
    # 每块 50 万像素，峰值距离矩阵 ~40MB。输出标签逐字节一致。
    c = np.ascontiguousarray(cent, np.float32)          # (K,3)
    cn = (c ** 2 * w).sum(1)[None, :]                    # (1,K) 常数项
    out = np.empty(len(lab), np.int32)
    _CHUNK = 500_000
    for s in range(0, len(lab), _CHUNK):
        e = min(s + _CHUNK, len(lab))
        seg = lab[s:e]
        # 与原来逐字相同的三项式，只把行数限制在 _CHUNK 内
        d = ((seg ** 2) * w).sum(1)[:, None] - 2.0 * ((seg * w) @ c.T) + cn
        out[s:e] = d.argmin(1)
    idx = out
    pal = np.clip(color.oklab_to_rgb01(cent) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    q = Image.fromarray(pal[idx].reshape(im.height, im.width, 3).astype(np.uint8), "RGB")
    if with_index:
        return pal, q, idx.reshape(im.height, im.width).astype(np.int32)
    return pal, q


def labels_of_quantized(pal: np.ndarray, q: Image.Image, w: np.ndarray) -> np.ndarray:
    """从量化图反推每像素的调色板标签（H×W int32）。

    v7.17.3 内存控制：旧写法 ``((lab[:,:,None,:]-plab[None,None,:,:])**2*w)``
    会一次性分配 H×W×K×3 浮点——4000 网格下 2844万×170×3×4 ≈ 58 GB，
    直接换页卡死。量化图本来就是 pal[idx] 生成的，每个像素颜色必然精确
    等于某个调色板色，所以最近色就是它自己的簇号，无需再搜。
    实测与旧广播法逐像素一致（diff 0），且只占 H×W×4 字节。
    """
    arr = np.asarray(q.convert("RGB"), np.float32) / 255.0
    lab = color.srgb_to_oklab(arr).reshape(-1, 3)
    c = np.ascontiguousarray(pal, np.float32)
    cn = (c ** 2 * w).sum(1)[None, :]
    out = np.empty(len(lab), np.int32)
    _CHUNK = 500_000
    for s in range(0, len(lab), _CHUNK):
        e = min(s + _CHUNK, len(lab))
        seg = lab[s:e]
        d = ((seg ** 2) * w).sum(1)[:, None] - 2.0 * ((seg * w) @ c.T) + cn
        out[s:e] = d.argmin(1)
    return out.reshape(q.height, q.width)


def core_colors(pal: np.ndarray, idx: np.ndarray, arr: np.ndarray,
                erode_px: int = 2) -> np.ndarray:
    """v7.17.1：区域**核心色**取色（替代整区均值色）。

    均值色会把抗锯齿过渡像素一起平均进去，颜色被拉淡、边界对比度下降
    （实测强边跳变只有原图的 48%，嘻嘻宝样本 104%）。这里把每个簇的掩膜
    腐蚀掉 ``erode_px`` 像素，只用**内部像素**求均值；薄区域（腐蚀后不足）
    回退整区均值，保证细线/发丝不被拉偏。
    """
    pal = pal.copy()
    if erode_px <= 0:
        return pal
    import cv2
    ker = np.ones((3, 3), np.uint8)
    for k in range(len(pal)):
        m = (idx == k).astype(np.uint8)
        a = int(m.sum())
        if a == 0:
            continue
        core = cv2.erode(m, ker, iterations=int(erode_px))
        sel = (core > 0) if int(core.sum()) >= max(12, a // 20) else (m > 0)
        if sel.any():
            # 必须先 clip 再 astype：255.5 四舍五入成 256，直接写 uint8 会溢出成 0
            # （白底 253~255 区域会变成纯蓝 #0000ff，整图背景变蓝）。
            pal[k] = np.clip(arr[sel].mean(0).round(), 0, 255).astype(np.uint8)
    return pal


def snap_palette(svg: str, pal: np.ndarray, w: np.ndarray) -> str:
    """色板吸附（自研）：把每条 path 的近似 fill 压回 N 色平涂色板。

    vtracer 的 color 模式会自己再聚类并对每个形状取均值，一张 ~112 色的图
    会描出上千种「近似色」，视觉上发糊发脏。这里在 OKLab 空间把每个 fill
    映射到调色板最近色，强制回到真正的平涂。

    - 距离度量与量化阶段用同一组通道权重 w，保证前后色彩一致；
    - 调色板颜色预先转成 OKLab，避免在正则回调里重复计算；
    - lru_cache 缓存 hex→hex 映射：常见图只有几十种 fill，命中率高。
    """
    pal_lab = color.srgb_to_oklab(np.clip(pal, 0, 255).astype(np.float32) / 255.0)

    @lru_cache(maxsize=8192)
    def nearest(hex6: str) -> str:
        rgb = np.array([int(hex6[1:3], 16), int(hex6[3:5], 16), int(hex6[5:7], 16)],
                       np.float32) / 255.0
        lc = color.srgb_to_oklab(rgb)
        diff = (pal_lab - lc) ** 2
        d = (diff * w).sum(1)
        idx = int(d.argmin())
        return "#%02x%02x%02x" % tuple(int(v) for v in pal[idx])

    return _FILL_HEX.sub(lambda m: 'fill="%s"' % nearest(m.group(1)), svg)


# v7.17.14 补缝描边宽度单位改正（本次画质提升最大的一处）。
# 旧实现把 seam 当**网格单位**直接传给 _add_seam_strokes(0.8)，而网格宽固定
# 3200；换成显示像素只有 0.8×1000/3200 ≈ 0.13px —— 比缝隙细了 5~8 倍，
# 等于没补。实测缝隙宽度由 spline 拟合误差决定，约 1~3 个网格单位，
# 必须用同量级的描边才盖得住。
SEAM_PX = 0.7          # 目标补缝宽度（**显示像素**，按人眼看到的画布算）
SEAM_MIN_GRID = 2.5    # 下限（网格单位）：描边细于拟合缝隙就失去意义


def trace(src: str, dst: str, colors: int = 112, hd_scale: float = 2.0,
          grid: int = 1600, seam: float = SEAM_PX, median: int = 3,
          w: np.ndarray | None = None, core_color: bool = True,
          face_clean: bool = False, progress=None,
          return_index: bool = False, flat_level: str = "off") -> dict:
    """把 *src* 融合描摹成 *dst* SVG，返回统计信息 dict。"""
    t0 = time.time()
    w = DEFAULT_W if w is None else np.asarray(w, np.float32)
    src_im = Image.open(src).convert("RGB")
    ow, oh = src_im.size

    # 自适应高清化：先放大再处理，让横纹/发丝/五官这些细节抓得到。
    scale = max(1.0, float(hd_scale))
    work = src_im.resize((int(round(ow * scale)), max(1, int(round(oh * scale)))),
                         Image.LANCZOS)

    # 统一到描摹网格宽度（按比例缩放高度）。
    g = max(256, int(grid))
    if work.width != g:
        work = work.resize((g, max(1, round(work.height * g / work.width))), Image.LANCZOS)

    # 中值硬边：抹掉 JPEG 抗锯齿坡道，让区域边缘成硬阶 —— 防色块碎裂。
    if median >= 3:
        work = work.filter(ImageFilter.MedianFilter(median if median % 2 else median + 1))

    if progress:
        progress("OKLab 感知量化（%d 色 · 色度加权）…" % colors)
    pal, q, idx = quantize_palette(work, colors, w, with_index=True)
    if core_color:
        if progress:
            progress("区域核心取色（去掉抗锯齿过渡像素）…")
        pal = core_colors(pal, idx, np.asarray(work, np.float32), erode_px=2)
        q = Image.fromarray(pal[idx].astype(np.uint8), "RGB")

    # v7.19.0 平涂浓度（可选，默认 off）：把同一材质的明暗台阶并成一块平色。
    # 用户反馈「170 色转出来颜色太精细，不如嘻嘻宝同色大色块干净」，实测
    # 嘻嘻宝样本中位 77 色，我们 170 色；这里按档位把近色阶成链合并，
    # 色数 170 → 105/70/40 左右，而形状（细节）几乎不动。
    flat_stats = {}
    _flat_req = str(flat_level or "off").lower()
    if _flat_req == "auto":
        # v7.20.0 全自动：程序按图像纹理强度自己选 medium / light。
        import flatmerge as _fm
        flat_level = _fm.auto_level(work)
        if progress:
            progress("自动识别完成 → %s 档平涂（大色块，细节自动保护）…" %
                     ("标准" if flat_level == "medium" else "轻"))
    if _flat_req not in ("off", "", "none"):
        try:
            import flatmerge as _fm
            if _flat_req != "auto" and progress:
                progress("平涂浓度 %s：合并相近色阶（%d 色 → 大色块）…" % (flat_level, colors))
            _fr = _fm.apply(idx, pal, flat_level, work.width, w,
                            rgb=np.asarray(work))
            if _fr.get("changed"):
                idx, pal = _fr["idx"], _fr["pal"]
                q = Image.fromarray(pal[idx].astype(np.uint8), "RGB")
                flat_stats = dict(flat_level=_fr["level"], flat_colors=_fr["colors"],
                                  flat_before=_fr.get("colors_before"),
                                  flat_eaten=_fr.get("eaten", 0),
                                  flat_auto=(_flat_req == "auto"))
        except Exception as exc:  # noqa: BLE001  平涂失败就退回原色数，不影响产出
            if progress:
                progress("平涂合并失败，按原色数继续：%s" % type(exc).__name__)

    if face_clean:
        # v7.17.4 新增（可选，默认关闭）：CNN 识别平滑区 → 整片重取主色。
        # 只对 CNN 高置信的平滑区动手，五官/发丝/轮廓原样保留。
        # 失败时优雅跳过，不影响原有 fusion 管线。
        try:
            import face_clean as _fc
            import region_recolor as _rr
            arr = np.asarray(work)
            prob = _fc.smooth_prob(arr, progress=progress)
            idx, pal = _rr.recolor(idx, pal, arr,
                                   smooth_override=prob > 0.5,
                                   cluster_jnd=0.02)
            q = Image.fromarray(pal[idx].astype(np.uint8), "RGB")
        except Exception as exc:  # noqa: BLE001
            if progress:
                progress("平滑区识别不可用，回退原 fusion：%s" % type(exc).__name__)

    tmp = tempfile.mkdtemp(prefix="flatfill_")
    png = os.path.join(tmp, "trace_in.png")
    raw = os.path.join(tmp, "trace_out.svg")
    q.save(png, "PNG", compress_level=1)

    if progress:
        progress("曲线描摹（完整轮廓 + 连续横纹）…")
    # 复用本地自研的 ref 描摹参数（cutout + spline），横纹连续、无纯黑块。
    core._vtrace(png, raw, **core._ref_kwargs(core.Settings(mode=core.MODE_REF)))
    svg = io.open(raw, encoding="utf-8").read()

    if progress:
        progress("色板吸附 + 补缝描边…")
    svg = snap_palette(svg, pal, w)              # 自研：近似色 -> N 色平涂
    # v7.17.15 渐变覆盖：平滑渐变（天空/水面/墙面）被量化成一圈圈色阶，
    # 用线性渐变整片盖住，台阶消失。必须在补缝之前——补缝只给 fill="#hex"
    # 的 path 加同色描边，渐变 path 的 fill="url(#...)"" 会被自然跳过。
    grad_zones = 0
    try:
        import gradfill as _gf
        if _gf.GRAD_FILL:
            svg, grad_zones = _gf.apply_svg(svg, np.asarray(work), idx, pal)
    except Exception:  # noqa: BLE001  渐变失败就用原色块，不影响产出
        grad_zones = 0
    # 补缝宽度换算：seam 是显示像素，乘 (网格宽/原图宽) 得网格单位。
    # 加 SEAM_MIN_GRID 下限，避免大图（显示宽 > 网格宽）把描边压到比缝隙还细。
    seam_w = 0.0
    if seam > 0:
        seam_w = max(float(SEAM_MIN_GRID), float(seam) * work.width / float(ow))
    svg = core._add_seam_strokes(svg, seam_w)    # 本地自研：同色描边遮缝
    disp_scale = ow / work.width if work.width else 1.0
    svg = core._set_root(svg, work.width, work.height, disp_scale,
                         "Flatfill traced artwork (%dx%d grid)" % (work.width, work.height))

    # v7.17.10 底图压缩（pathopt.py）：把每个 path 各带一份的
    # `stroke-width/linejoin/linecap` 提到根 <svg> 继承（逐像素无损），
    # 再把 `d` 里退化成直线的三次贝塞尔转 L/H/V 并合并共线直线。
    # 实测底图 −34~41%，渲染差 mean ≤0.12/255、无 >32 像素。
    # 放在 _set_root 之后：optimize 要往根标签追加属性，会被 _set_root 覆盖掉。
    if getattr(core, "COMPACT_BASE_PATHS", True):
        try:
            import pathopt as _po
            svg = _po.optimize(svg, grid_w=work.width)
        except Exception:  # noqa: BLE001  压缩失败就用未压缩底图，不影响产出
            pass

    io.open(dst, "w", encoding="utf-8", newline="\n").write(svg)
    fills = set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg))
    res = dict(mode="fusion", src=src, dst=dst, colors=len(fills),
               paths=svg.count("<path"), bytes=os.path.getsize(dst),
               seconds=round(time.time() - t0, 1), hd="%.1fx" % scale,
               seam=round(seam_w, 2), face_clean=bool(face_clean),
               grad_zones=grad_zones)
    res.update(flat_stats)
    if return_index:
        # v7.17.7: 供 linenet 模式取色块标签图，用来生成 L2 色块边界。
        res["index"] = idx
        res["palette"] = pal
    return res
