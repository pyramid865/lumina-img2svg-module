# -*- coding: utf-8 -*-
"""先定线后填色引擎 (MODE_LINEFIRST) — v7.21.0 全新独立引擎。

思路（与 fusion / flatfill 根本不同，是对标别人工具三层结构的新引擎）：
  旧引擎：先把整图切成上千色块 → 事后合并 → 最后才把线贴上去。
          线是"事后补的"，色块一合并，网袜那种细线就被吞掉。
  本引擎：先用 CNN 把线稿"钉死"当成一道道墙（分水岭屏障），
          再让颜色从平滑区自然生长成一个个封闭盆地（区域生长），
          每个盆地只填一个代表色，最后把钉死的线压回最上层。
          线在最里面就是骨架，色块再怎么合并也越不过这道墙。

三层 SVG 结构（对标别人工具 region-boundaries / line-art / region-colours）：
  - 色块层 region-colours：分水岭盆地的平涂色块（vtracer spline 平滑描摹）
  - 线稿层 line-art-layer：CNN 提线（暗墨 + 彩线，按原图色分组）
  - 边界层 region-boundaries：色块之间的补缝描边，防止相邻块留白缝

依赖：numpy / opencv / scipy / onnxruntime / vtracer —— 全部已在运行时内，无新增依赖。
打包环境 (.venv312) 无 sklearn / skimage，本文件一律不使用。
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import time

import numpy as np
import cv2

import core

# ---- 参数（每个都配大白话：管什么 / 调大怎样 / 调小怎样 / 为什么定这里）----
# 内部工作画布宽度（px）：先用 1600 做分区。越大色块边界越精细、但越慢；
# 越小越快、但网袜细格会糊。1600 是"网袜能保住 + 3~4 秒出结果"的平衡点。
WORK_W = 1600

# 种子平滑尺度（高斯 σ，px）：决定从哪里开始"长"色块。越大→色块越大越整、
# 但小斑块被抹平；越小→色块越碎、细节越多。1.2 是让脸/皮肤成整块的经验值。
SEED_SIGMA = 1.2

# 种子山谷深度（局部极小值阈值，灰度级）：决定分水岭从哪里"长"色块。
# 这是做出规整纹理格的关键——用局部极小值当种子：
#   平滑区（脸/皮肤）起伏小 → 极少极小值 → 长成大色块；
#   纹理区（网袜/发丝）起伏密 → 很多极小值 → 每条纹理切成独立小格。
# 越大→只有更深的谷当种子→种子少→大块更整、网眼更大；越小→种子密→更碎。
#
# 【v7.21.2：10 → 35】
# 实测 10 对"浅色柔光插画"（白卫衣、紫发蜡烛）过敏感：把 JPEG 噪点、细微
# 明暗起伏都当成种子谷，紫图切出 10309 个碎块，头发高光被切成斑驳浅色碎点。
# 提高到 35 后：紫图降到 3230 块、头发变顺畅大块、五官线更清晰；金克丝网袜
# 网眼仍完好（网眼之间有强梯度线当墙，不会被合并），白卫衣更干净。
# 35 是"三张风格差异极大的图都能兼顾"的统一值（10/20 对柔光图仍偏碎、60+ 会丢五官）。
SEED_H = 35

# 极小盆地清理面积（px）：面积小于它的碎片盆地并入邻居，清掉纹理区的噪点碎块。
# 越大→背景越干净，但太大会连网眼一起吞掉（网眼通常几十~上百 px）；
# 越小→保留更多细节但噪点多。25 只清最小的噪点，网眼安全。0=关闭清理。
MIN_BASIN_PX = 25

# 线稿墙厚度（膨胀迭代次数）：把 1px 线加厚成墙，防止色块"渗"过线的断口。
# 越大→越不容易跨线合并、色块越贴线但可能把细处撑粗；越小→线细处可能漏色。
# 1 表示线稿膨胀 1 圈（约 3px 墙），实测网袜不断色。
LINE_DILATE = 1

# 线片段最小面积（px）：小于它的 CNN 线当噪点删掉。越大→线越干净但丢细发丝；
# 越小→细节多但噪点也多。12 是暗光霓虹图噪点与细节的平衡点。
MIN_LINE_PX = 12

# 线稿粗细（工作网格 px）：线画多粗。越大越清晰、但过粗会糊住细节；
# 越小越细越精致、但太细在缩略图里会看不见。2.4 让粉网在缩略图也看得清。
LINE_WIDTH = 2.4

# 彩色线分组数：线层按原图色聚成 N 组，每组一个 stroke 颜色。
# 越大越接近原图的丰富线色（网袜粉、发丝蓝各自分明）但文件更大；
# 越小线色被合并、网袜可能和暗线混在一起。48 让网袜粉线更可能独占一组。
LINE_COLOR_GROUPS = 48

# 彩色线饱和度增强倍数：把原图线色稍微提艳，让网袜粉线更明显。
# 越大线越鲜艳但可能过饱和发荧光；1.0=原色不动。1.5 让暗光下的粉线回到可见。
LINE_SAT_BOOST = 1.5

# CNN 线稿最短长度（px）：短于它的碎线丢掉。
#
# 【v7.21.2：20 → 8】
# v7.21.0/7.21.1 为了压掉"脸部黑色虚线"一度把它提到 20，但实测那批黑线真正
# 来源是"色块边界层"（v7.21.1 已整层关闭），线稿层本身是无辜的。提到 20 会
# 连带砍掉 82~86% 的正常线稿（金克丝 715→100 条、紫图 1626→293 条），画面
# 因此缺线条、五官发糊。边界层关掉后线稿层不会再产生黑虚线，故回落到 8：
# 恢复发丝/五官/衣物褶皱的线稿密度，画面重新有"线稿感"。太小的 6 会有噪点短线。
LINE_MIN_LEN = 8

# 每个大色区内部的明暗色阶数：皮肤/大腿不会是一个死板的平均色，而是按亮度
# 分成 N 级（亮面/中间调/暗面），每级取原图实际色——这就是别人工具那种
# "大色块里带柔和明暗过渡"的效果。越大层次越丰富但可能出条纹；越小越平。
# 3 是"皮肤有立体感又不脏"的最佳值（实测 2 偏平、4 偶有窄条纹）。
SHADE_LEVELS = 3

# 色阶细分的最小盆地面积（px）：只有面积大于它的盆地才按明暗分色阶。
# 太小的盆地本身只有几十像素，再切 3 级就变成针尖碎块（头发上的浅色碎斑）。
# 越大→更多小盆地保持单色、画面更干净，但过大会让该有明暗的地方变平；
# 越小→小盆地也被分层，碎斑多。600 是"大块有明暗过渡、小块不碎"的分界。
# （v7.21.1 及以前是 200，紫图 10309 块，碎斑明显。）
SHADE_MIN_AREA = 600

# 色阶平滑尺度（px）：分级前先把亮度小幅模糊，让明暗过渡边界是柔和波浪形
# 而非锯齿。越大越柔和但层次变糊；越小越锐利但可能碎。2 是柔和度与层次的平衡。
SHADE_SIGMA = 2.0

# 色阶最低亮度门槛（OKLab L，0~1）：只有平均亮度高于它的盆地才做明暗分层。
# 暗部（沙发/背景/阴影）亮度低、色阶挤在一起，分层只会产生锯齿黑碎块，
# 所以暗部保持单色平滑。越大越多区域不分色阶；越小暗部也会分层（易脏）。
# 0.28 是"皮肤/亮部有层次、暗部干净"的分界（实测 0.22 暗部出碎块）。
SHADE_MIN_L = 0.28

# 色块边界描边色差门限（OKLab ΔE，0=关）：色块之间的边界只有两侧颜色差
# 超过它才描深色轮廓线。越大→只描强对比主轮廓、柔和渐变不勾线（更干净）；
# 越小→更多块边界被描出、轮廓更全但可能出现等高线噪声；0=整层关闭。
#
# 【v7.21.1 修复：从 0.10 改为 0，彻底关闭这一层】
# 用户实测浅色动漫图（白卫衣、蜡烛少女）出现大量"异常黑线"——白卫衣上
# 密布垂直黑色虚线短划。定位确认：这层"边界层"会把每个色块（尤其明暗色阶
# 细分块）之间的柔和色差都勾成深色线，浅色柔光图被切成上千小盆地后，就变成
# 一片黑虚线。对照实验（关掉这层）证明：黑线全部消失，而人物轮廓（本就由
# CNN 线稿层负责）、网袜、色阶、颜色全都完好无损——这层是纯负作用。
# 它原本的职责（防止相邻色块留白缝）已由 vtracer cutout 描摹 + 屏障填色解决。
# 保留开关与代码，万一将来某类图确实需要补轮廓，可改回小值（如 0.12）。
BOUNDARY_DE = 0.0

# 边界线最短长度（px）：色块边界短于它的碎段直接丢掉。自适应种子把纹理区
# 切得很碎，会产生大量"虚线短划"看着脏。越大→只留连贯长轮廓、越干净，
# 但太大会丢掉小的真实轮廓；越小→碎线多。32 是"虚线消失、主轮廓还在"的点。
BOUNDARY_MIN_LEN = 32

_FILL_HEX = re.compile(r'fill="(#[0-9a-fA-F]{6})"')


def colored_line_svg(rgb: np.ndarray, mask: np.ndarray, stroke_width: float,
                     n_groups: int = 24, sat_boost: float = 1.25,
                     min_len: float = 8.0, eps: float = 0.7):
    """按原图实际颜色画 CNN 线稿（不压暗）——别人工具的 L1 彩线做法。

    别人工具的"线稿层"其实是用 fill 画、颜色取原图该处的色：
    网袜线在原图是粉色 → 画出来就是粉线；轮廓线原图是暗色 → 自然是黑线。
    本函数用 stroke 走中心线（一条线一趟），每条线采样沿线原图色，
    再聚成 n_groups 组避免每条线一个颜色导致文件爆炸。

    返回 (svg片段, 路径数, 分组数)。
    """
    import linenet_line as ln
    chains = ln.trace_centerlines(ln.thin(mask), min_len=min_len, eps=eps)
    if not chains:
        return "", 0, 0
    h, w = mask.shape[:2]
    # 取色关键：CNN 中心线常落在网线旁边的暗底上，直接沿线取会偏黑。
    # 对每个线点在 ±R 邻域内找"最亮"的像素色（网袜粉线比暗底亮很多），
    # 再沿线取较亮分位——这样才能采到粉线本色。
    R = 2
    padded = np.pad(rgb, ((R, R), (R, R), (0, 0)), mode="reflect")
    # 邻域最大亮度图（RGB 各通道 max，够用且快）
    hue_max = np.zeros((h, w, 3), np.uint8)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            patch = padded[R + dy:R + dy + h, R + dx:R + dx + w]
            np.maximum(hue_max, patch, out=hue_max)
    cols = []
    for p, _ in chains:
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
        pix = hue_max[yi, xi].astype(np.float32)
        cols.append(np.percentile(pix, 90, axis=0))
    cols = np.asarray(cols, np.float32)
    # 提艳 + 适度提亮（暗光下粉线会发闷，S 和 V 一起抬）
    hsv = cv2.cvtColor(np.clip(cols, 0, 255).astype(np.uint8).reshape(1, -1, 3),
                       cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 1] = np.clip(hsv[:, 1] * sat_boost, 0, 255)
    hsv[:, 2] = np.clip(hsv[:, 2] * 1.12, 0, 255)
    cols = cv2.cvtColor(hsv.astype(np.uint8).reshape(1, -1, 3),
                        cv2.COLOR_HSV2RGB).reshape(-1, 3).astype(np.float32)

    k = int(max(1, min(n_groups, len(chains))))
    if len(chains) > k:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _, labels, centers = cv2.kmeans(cols, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(-1)
        centers = centers.astype(np.float32)
    else:
        labels = np.arange(len(chains))
        centers = cols

    # 亮的先画、暗的最后压顶层（与 linenet 一致）
    order = np.argsort(-centers.max(axis=1))
    rank = np.empty(k, np.int32)
    for r, gi in enumerate(order):
        rank[gi] = r
    segs = []
    for gi in order:
        d = " ".join(ln._poly_d(p, 1) for p, l in zip((c[0] for c in chains), labels)
                     if int(l) == int(gi))
        if not d:
            continue
        rr, gg, bb = (int(np.clip(round(v), 0, 255)) for v in centers[gi])
        segs.append('<g fill="none" stroke="#%02x%02x%02x" stroke-width="%.2f" '
                    'stroke-linejoin="round" stroke-linecap="round"><path d="%s"/></g>\n'
                    % (rr, gg, bb, stroke_width, d))
    return "".join(segs), len(chains), len(segs)


def trace(src: str, dst: str, progress=None, colors: int = 0) -> dict:
    """src 图片 → dst SVG（三层结构），返回统计 dict。"""
    t0 = time.time()

    def step(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    # ---------- 1) 钉线：CNN 提线稿 ----------
    step("先定线后填色引擎 · 识别线稿…")
    import linenet_line as ln
    from PIL import Image as _I

    with _I.open(src) as _im:
        rgb = np.asarray(_im.convert("RGB"))
    oh, ow = rgb.shape[:2]
    sc = WORK_W / float(ow)
    work = cv2.resize(rgb, (WORK_W, max(1, int(round(oh * sc)))),
                      interpolation=cv2.INTER_AREA)
    h, w = work.shape[:2]

    if not ln.available():
        raise RuntimeError("线稿模型缺失，无法使用先定线后填色引擎")
    heat = ln.heatmap(work, progress=progress)
    line = ln.filter_small(ln.steger_lines(heat), MIN_LINE_PX)
    wall = cv2.dilate(line.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=LINE_DILATE).astype(bool)
    step("线稿已锁定（%.1f%% 像素），开始分区填色…" % (100 * line.mean()))

    # ---------- 2) 分水岭分区：自适应种子（纹理区切网眼，平滑区成大块）----------
    import color as clr
    from scipy import ndimage as _ndi
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sm = cv2.GaussianBlur(gray, (0, 0), SEED_SIGMA)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # 自适应种子：局部极小值（谷）当种子。平滑区谷少→大块；网袜谷密→网眼。
    # 用 7×7 形态学近似局部极小：比邻域低 SEED_H 灰度级才算一个种子谷。
    er = _ndi.grey_erosion(sm, size=7)
    dil = _ndi.grey_dilation(sm, size=7)
    seed = ((sm <= er + SEED_H) & (sm < dil - SEED_H) & ~wall).astype(np.uint8)
    nseed, markers = cv2.connectedComponents(seed, connectivity=4)
    terrain = np.clip(grad, 0, 255).astype(np.uint8)
    terrain[wall] = 255
    ws = cv2.watershed(cv2.cvtColor(terrain, cv2.COLOR_GRAY2BGR), markers)

    # 屏障像素填最近区域的色（避免黑色裂缝）
    from scipy import ndimage
    barrier = ws < 0
    if barrier.any():
        _, idx = ndimage.distance_transform_edt(
            barrier, return_distances=True, return_indices=True)
        ws[barrier] = ws[idx[0][barrier], idx[1][barrier]]

    # 清理极小盆地（纹理区的噪点碎块）：面积 < MIN_BASIN_PX 的盆地
    # 抹成 0，再用距离变换填最近的大盆地色。网眼面积远大于阈值不会被误伤。
    if MIN_BASIN_PX > 0:
        areas = np.bincount(ws.ravel())
        tiny = areas < MIN_BASIN_PX
        tiny[0] = False
        if tiny.any():
            ws_tiny = np.where(tiny[ws], 0, ws)
            _, idx2 = ndimage.distance_transform_edt(
                ws_tiny == 0, return_distances=True, return_indices=True)
            zero = ws_tiny == 0
            ws[zero] = ws_tiny[idx2[0][zero], idx2[1][zero]]

    # ---------- 3) 每区按明暗分色阶，取原图实际色 ----------
    # 关键（对标别人工具）：皮肤/大腿不是一个死板平均色，而是在每个盆地内部
    # 按亮度分成 SHADE_LEVELS 级（亮面/中间调/暗面），每级用原图 sRGB 平均色。
    # 这样色块带柔和明暗过渡，又不会跨盆地边界乱跑。
    flat = work.reshape(-1, 3).astype(np.float32) / 255.0
    lab = clr.srgb_to_oklab(flat).reshape(h, w, 3)
    L = lab[:, :, 0]                       # OKLab 明度通道
    Ls = cv2.GaussianBlur(L, (0, 0), SHADE_SIGMA)

    regs = np.unique(ws)
    regs = regs[regs >= 1]
    remap = np.zeros(int(ws.max()) + 2, np.int32)
    for j, r in enumerate(regs):
        remap[r] = j
    basin_id = remap[ws]                    # (h,w) 盆地编号
    nb = len(regs)

    # 每个盆地内按明度分位切成 SHADE_LEVELS 级；暗部跳过（避免锯齿黑碎块）
    lvl = np.zeros((h, w), np.int32)
    basin_bright = np.bincount(basin_id.ravel(), weights=L.ravel(),
                               minlength=nb) / np.maximum(
        np.bincount(basin_id.ravel(), minlength=nb), 1)
    for j in range(nb):
        m = basin_id == j
        if not m.any():
            continue
        lv = Ls[m]
        # 亮度够 + 面积够 才分层；暗部/小斑块保持单色
        if lv.size > SHADE_MIN_AREA and SHADE_LEVELS > 1 and basin_bright[j] >= SHADE_MIN_L:
            edges = np.quantile(lv, np.linspace(0, 1, SHADE_LEVELS + 1)[1:-1])
            lvl[m] = np.searchsorted(edges, lv)
    # 盆地×级别 组合标签
    combo = basin_id * 16 + lvl
    uc = np.unique(combo)
    cremap = np.zeros(int(combo.max()) + 1, np.int32)
    for j, c in enumerate(uc):
        cremap[c] = j
    fine = cremap[combo]                    # (h,w) 最终细分块 id
    nf = len(uc)

    # 每个细分块取原图实际 sRGB 平均色（不用 OKLab 反算，避免整体发灰）
    srgb = flat.reshape(-1, 3)
    rf = fine.ravel()
    cnt = np.bincount(rf, minlength=nf)
    fc = np.zeros((nf, 3), np.float32)
    for ch in range(3):
        fc[:, ch] = np.bincount(rf, weights=srgb[:, ch], minlength=nf) / np.maximum(cnt, 1)
    fill = (np.clip(fc, 0, 1) * 255).astype(np.uint8)[fine]
    step("已分成 %d 个色块（含明暗色阶），正在描摹…" % nf)

    # ---------- 4) 色块层：vtracer 平滑描摹填色图 ----------
    tmp = tempfile.mkdtemp(prefix="linefirst_")
    png = os.path.join(tmp, "fill.png")
    raw = os.path.join(tmp, "fill.svg")
    try:
        _I.fromarray(fill).save(png, "PNG", compress_level=1)
        # cutout 模式：互不重叠的平铺色块（每个分水岭盆地一块），
        # 与 segregion.py 相同的经过验证的组合。stacked 叠加模式会让满屏底色
        # 盖住细节，这里绝不能用。
        kw = core._ref_kwargs(core.Settings(mode=core.MODE_REF))
        core._vtrace(png, raw, **kw)
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

    # 根节点重写：viewBox = 工作网格，显示尺寸 = 原图
    svg = core._set_root(svg, w, h, ow / float(w), "Line-first flat artwork")
    # 给色块层加标识
    svg = svg.replace("<svg ", '<svg data-engine="linefirst" ', 1)

    # ---------- 5) 线稿层：CNN 线压顶层（暗墨 + 彩线，坐标=工作网格）----------
    step("压回锁定的线稿（原色，网袜粉线保留）…")
    try:
        # 别人工具做法：线取原图实际色——网袜线是粉色就画粉色，轮廓是暗色就画暗色。
        # 用 p75 沿线亮色避免被暗底拉黑，轻提艳让暗光下的粉线回到可见。
        seg, npath, ngroup = colored_line_svg(
            work, line, stroke_width=LINE_WIDTH, n_groups=LINE_COLOR_GROUPS,
            sat_boost=LINE_SAT_BOOST, min_len=LINE_MIN_LEN, eps=0.7)
    except Exception:
        try:
            seg, npath = ln.stroke_svg_center(
                line, stroke_width=LINE_WIDTH, color="#1a1520",
                min_len=LINE_MIN_LEN, eps=0.7)
            ngroup = 1
        except Exception:
            seg, npath, ngroup = "", 0, 0
    line_layer = (
        '<g id="line-art-layer" data-role="cnn-line-art-layer">\n%s</g>\n' % seg
    ) if seg else ""

    # ---------- 6) 色块边界层：主体轮廓线（脸/身体/衣物的清晰边界）----------
    # CNN 线稿主要抓纹理线（网袜/发丝），主体大轮廓靠盆地边界补。
    # 基于盆地（不是明暗细分块）勾线：明暗色阶之间不勾，避免等高线噪声。
    # ΔE 门控滤掉柔和渐变的假轮廓，只描真实结构边界。
    boundary_layer = ""
    nbound = 0
    if BOUNDARY_DE > 0:
        try:
            # 盆地平均 sRGB 色板（内部 strong_boundary 会转 OKLab）
            bpal = np.zeros((nb, 3), np.float32)
            for ch in range(3):
                bpal[:, ch] = np.bincount(
                    basin_id.ravel(), weights=srgb[:, ch], minlength=nb
                ) / np.maximum(np.bincount(basin_id.ravel(), minlength=nb), 1)
            pal_srgb = np.clip(bpal, 0, 1).astype(np.float32) * 255
            bmask = ln.strong_boundary(basin_id, pal_srgb.astype(np.uint8),
                                       de_min=BOUNDARY_DE, min_size=20)
            bseg, nbound = ln.stroke_svg_center(
                bmask, stroke_width=LINE_WIDTH, color="#1a1520",
                min_len=BOUNDARY_MIN_LEN, eps=0.8)
            if bseg:
                boundary_layer = (
                    '<g id="region-boundaries" data-role="line-constrained-boundary">\n'
                    '%s</g>\n' % bseg)
        except Exception:
            boundary_layer = ""

    cut = svg.rfind("</svg>")
    if cut < 0:
        cut = len(svg)
        svg = svg + "</svg>\n"
        cut = svg.rfind("</svg>")
    svg = svg[:cut] + boundary_layer + line_layer + svg[cut:]

    with io.open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)

    fills = set(_FILL_HEX.findall(svg))
    return dict(mode="linefirst", src=src, dst=dst,
                grid="%d (linefirst)" % w,
                colors=len(fills), paths=svg.count("<path"),
                regions=nf, line_paths=npath, line_groups=ngroup,
                boundary_paths=nbound,
                bytes=os.path.getsize(dst), hd="",
                seconds=round(time.time() - t0, 1))
