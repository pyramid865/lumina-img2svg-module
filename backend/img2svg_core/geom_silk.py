# -*- coding: utf-8 -*-
"""丝状几何化重绘（silk mosaic）—— 对标嘻嘻宝的曲线拟合层。

与 ``geom_shared`` 的区别（HANDOFF §25 解剖结论）：

===========================  ======================  ==========================
                             嘻嘻宝                  geom_shared (v6.8)
===========================  ======================  ==========================
path 组织                     一块一 path (90.9% 单M)  同色所有环合并 (8.6 M/path)
子路径命令数                  median 9 / p90 33       median 4 / p90 10
弦长                          median 2.5px             median 20px
A 弧                          97% 带旋转、半径 0.6px  全正圆、半径 20px
细丝 path                     624 条                   4 条
===========================  ======================  ==========================

所以「头发不顺滑」不是缝、也不是色数，而是**拟合层把每个环糊成几条大圆弧**，
细丝在被合并时丢掉。本模块做三件事：

1. **一块一 path**：用 ``skimage.measure.label`` 把硬标签按「同色 + 4 连通」
   切成独立色块，每块单独提轮廓、单独输出 ``<path>``（+ 成对 gap-filler）；
   细长丝状块按形状判据保留，孤立小噪点丢弃。
2. **强制细分**：``_fit_polyline_silk`` 限制单条命令的最大弦长 ``max_chord``，
   把一条平滑环拟合成 8~30 条连续短命令，而不是 4 条大弧。输入折线先做
   高斯平滑，所以密集采样得到的是平滑曲线链，不是像素阶梯。
3. **带旋转椭圆弧**：短转折优先输出 ``A rx ry φ large sweep x y``（φ 取起点
   切线角；rx=ry 时数学等价但格式与嘻嘻宝一致），其次 Q，再 L / C。

共边（marching-squares level=0.5，不做高斯/上采样）与几何外扩 ``expand``
沿用 ``geom_shared``，保证放大无亚像素缝；gap-filler 只作预览遮缝，
Lumina 导入时剥离。
"""
from __future__ import annotations

import io
import math
import os
import re
import time

import numpy as np
from PIL import Image

import geom_trace as g
import geom_shared as gs
import qq3d_like as qq


# --------------------------------------------------------------------------- #
# 拟合层
# --------------------------------------------------------------------------- #
def _rot_arc(cmd: str | None, P: np.ndarray) -> str | None:
    """把 ``_try_arc`` 输出的 ``A r r 0 <large> <sweep> x y`` 换成带旋转版本。

    rx == ry 时 rotation 对形状无影响，取起点切线角只为保证输出格式与
    嘻嘻宝一致（其 97.4% 的 A 带旋转），并让下游解析器兼容椭圆弧语法。
    """
    if cmd is None:
        return None
    m = re.match(r"A\s+(\S+)\s+(\S+)\s+0\s+([01])\s+([01])\s+(.*)$", cmd)
    if not m:
        return cmd
    if len(P) >= 2:
        v = P[1] - P[0]
        phi = math.degrees(math.atan2(v[1], v[0])) if np.hypot(*v) > 1e-9 else 0.0
    else:
        phi = 0.0
    return "A %s %s %.6f %s %s %s" % (m.group(1), m.group(2), phi,
                                      m.group(3), m.group(4), m.group(5))


def _fit_polyline_silk(P: np.ndarray, tol: float, max_chord: float,
                       depth: int = 0, max_depth: int = 12) -> list[str]:
    """把一段（已平滑的）折线拟合成密集的短 A / Q / L / C 命令链。

    与 ``geom_trace._fit_polyline`` 的根本差别：不看误差就先按弦长强制对半
    细分，直到单段弦长 <= ``max_chord``。这样每个环会产出 8~30 条连续命令，
    而不是一条包住整段的大圆弧。
    """
    if len(P) < 2:
        return []
    chord = float(np.linalg.norm(P[-1] - P[0]))
    if chord > max_chord and len(P) > 4 and depth < max_depth:
        j = len(P) // 2
        # 对半点必须同时属于两段，避免曲线在拼接处断开
        return (_fit_polyline_silk(P[:j + 1], tol, max_chord, depth + 1, max_depth)
                + _fit_polyline_silk(P[j:], tol, max_chord, depth + 1, max_depth))
    # 叶子：优先弧 / 二次，再直线，最后三次兜底
    arc = _rot_arc(g._try_arc(P, tol), P)
    if arc is not None:
        return [arc]
    quad = g._try_quad(P, tol)
    if quad is not None:
        return [quad]
    err, _ = g._line_error(P)
    if err <= tol:
        return ["L %s" % g._pt(P[-1])]
    return [g._fit_cubic(P)]


def _fit_closed_silk(poly: np.ndarray, tol: float, max_chord: float,
                     smooth: float, corner_deg: float = 45.0,
                     fit_mode: str = "smooth") -> str:
    """闭合轮廓 -> 一条完整的 ``M ... Z``，密集短基元。

    注意：绝不能中途截断命令链。截断会让 Z 从半路直线连回起点，
    留下一个巨大的三角形空洞（S0/S1 的纯白三角即此因）。
    嘻嘻宝最大子环有 1456 条命令，长环完整输出是可接受的。
    """
    P = np.asarray(poly, np.float64)
    if not np.allclose(P[0], P[-1], atol=1e-6):
        P = np.vstack([P, P[:1]])
    if len(P) < 8:
        return ""
    ring = g._resample_ring(P[:-1], 1.0)
    n = len(ring)
    if n < 8:
        return ""
    if smooth > 0:
        from scipy.ndimage import gaussian_filter1d
        ext = np.vstack([ring[-6:], ring, ring[:6]])
        sm = gaussian_filter1d(ext, smooth, axis=0, mode="nearest")[6:6 + n]
    else:
        sm = ring.copy()
    cs = g._corner_indices(sm, corner_deg)
    if not cs:
        d = np.hypot(*(sm - sm[0]).T)
        j = int(d.argmax())
        cs = sorted({0, j}) if j > 0 else [0, n // 2]
    parts: list[str] = []
    for a, b in zip(cs, cs[1:] + [cs[0] + n]):
        idx = np.arange(a, b + 1) % n
        run = sm[idx]
        if len(run) < 2:
            continue
        if fit_mode == "chord":
            parts += _fit_polyline_silk(run, tol, max_chord)
        else:
            parts += _emit_chain(_smooth_chain(_fit_run_adaptive(run, tol)))
    if not parts:
        return ""
    return "M %s %s Z" % (g._pt(sm[cs[0]]), " ".join(parts))


# --------------------------------------------------------------------------- #
# 自适应拟合层（误差驱动切分 + C1 连续，对标嘻嘻宝「少而长的基元」）
# --------------------------------------------------------------------------- #
def _quad_fit(P: np.ndarray) -> tuple[np.ndarray, float]:
    """最小二乘单控制点二次贝塞尔，返回 (ctrl, max_err)。"""
    t = g._arc_len_t(P)
    p0, p1 = P[0], P[-1]
    B = 2.0 * (1.0 - t) * t
    denom = float((B * B).sum())
    if denom < 1e-12:
        return P[len(P) // 2].copy(), 1e9
    rhs = P - ((1.0 - t) ** 2)[:, None] * p0 - (t ** 2)[:, None] * p1
    ctrl = (B[:, None] * rhs).sum(0) / denom
    fit = (((1.0 - t) ** 2)[:, None] * p0 + B[:, None] * ctrl
           + (t ** 2)[:, None] * p1)
    return ctrl, float(np.hypot(*(P - fit).T).max())


def _cubic_ctrls(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """最小二乘三次贝塞尔兜底，返回 (c1, c2) 控制点。"""
    t = g._arc_len_t(P)
    p0, p1 = P[0], P[-1]
    u = 1.0 - t
    B1, B2 = 3 * u * u * t, 3 * u * t * t
    rhs = P - (u ** 3)[:, None] * p0 - (t ** 3)[:, None] * p1
    A = np.stack([B1, B2], 1)
    try:
        ctrl = np.linalg.lstsq(A, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        d = (p1 - p0) / 3.0
        ctrl = np.array([p0 + d, p1 - d])
    return ctrl[0], ctrl[1]


def _quad_worst(P: np.ndarray) -> int:
    """单二次拟合的最差点索引（误差自适应切分点）。"""
    ctrl, _ = _quad_fit(P)
    t = g._arc_len_t(P)
    p0, p1 = P[0], P[-1]
    fit = ((1.0 - t) ** 2)[:, None] * p0 + (2 * (1.0 - t) * t)[:, None] * ctrl + (t ** 2)[:, None] * p1
    return int(np.hypot(*(P - fit).T).argmax())


def _fit_run_adaptive(P: np.ndarray, tol: float, depth: int = 0,
                      max_depth: int = 16) -> list[dict]:
    """误差自适应拟合：弧/二次不超 tol 就整段保留，否则在最差处切分。

    返回 seg 列表 {'k':A|Q|L|C, 'e':终点, ...}。不按弦长硬切，因此平滑段
    自然成为少数长基元（与嘻嘻宝一致），而非被强制剁成碎段。
    """
    if len(P) < 3:
        return [{'k': 'L', 'e': P[-1]}]
    arc = _rot_arc(g._try_arc(P, tol), P)
    if arc is not None:
        return [{'k': 'A', 'e': P[-1], 'a': arc}]
    ctrl, err = _quad_fit(P)
    if err <= tol:
        return [{'k': 'Q', 'c': ctrl, 'e': P[-1]}]
    if depth < max_depth and len(P) > 8:
        i = _quad_worst(P)
        if 1 < i < len(P) - 1:
            return (_fit_run_adaptive(P[:i + 1], tol, depth + 1, max_depth)
                    + _fit_run_adaptive(P[i:], tol, depth + 1, max_depth))
    c1, c2 = _cubic_ctrls(P)
    return [{'k': 'C', 'c': (c1, c2), 'e': P[-1]}]


def _smooth_chain(segs: list[dict]) -> list[dict]:
    """相邻 Q 在连接点处强制 C1 切线连续（控制点共线），消除放大后微折角。"""
    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        if a.get('k') == 'Q' and b.get('k') == 'Q' and a['c'] is not None and b['c'] is not None:
            pm = a['e']
            dl = a['c'] - pm
            dr = b['c'] - pm
            nl, nr = np.hypot(*dl.T), np.hypot(*dr.T)
            if nl < 1e-6 or nr < 1e-6:
                continue
            t = dl / nl + dr / nr
            n = np.hypot(*t.T)
            if n < 1e-6:
                continue
            t = t / n
            a['c'] = pm - t * nl
            b['c'] = pm + t * nr
    return segs


def _emit_chain(segs: list[dict]) -> list[str]:
    out = []
    for s in segs:
        if s['k'] == 'A':
            out.append(s['a'])
        elif s['k'] == 'Q':
            out.append("Q %s %s" % (g._pt(s['c']), g._pt(s['e'])))
        elif s['k'] == 'C':
            out.append("C %s %s %s" % (g._pt(s['c'][0]), g._pt(s['c'][1]), g._pt(s['e'])))
        else:
            out.append("L %s" % g._pt(s['e']))
    return out


# --------------------------------------------------------------------------- #
# 标签去噪（形状保护）
# --------------------------------------------------------------------------- #
def _merge_rare_colors(labels: np.ndarray, pal: np.ndarray,
                       min_frac: float) -> tuple[np.ndarray, np.ndarray, int]:
    """把「总像素占比 < min_frac」的稀有调色板色并入 OKLab 最近的常用色。

    k-means 常为少数离群像素分配一个极端高饱和色（画面上就是刺眼的亮蓝 /
    亮黄小点），这些色总面积很小。按「颜色稀有度」而不是「色块面积」合并，
    能去掉彩色碎斑，又完整保留渐变里的大量中等色块与细丝。
    """
    from qq3d_like import _srgb_to_oklab
    K = len(pal)
    cnt = np.bincount(labels.ravel(), minlength=K)
    thr = max(1.0, labels.size * min_frac)
    rare = np.flatnonzero((cnt > 0) & (cnt < thr))
    common = np.flatnonzero(cnt >= thr)
    if len(rare) == 0 or len(common) == 0:
        return labels, pal, 0
    ok = _srgb_to_oklab(pal[None].astype(np.float32) / 255.0)[0]
    mapping = np.arange(K, dtype=np.int32)
    for c in rare:
        d = ((ok[common] - ok[c]) ** 2).sum(1)
        mapping[c] = common[int(d.argmin())]
    out = mapping[labels]
    pal2 = pal.copy()
    pal2[rare] = pal[mapping[rare]]
    return out, pal2, len(rare)


def _despeckle(labels: np.ndarray, min_area: float,
               silk_ar: float, silk_len: float) -> tuple[np.ndarray, int]:
    """把孤立小色块并入最近邻色块，细长丝状块保留。

    直接「丢弃」小色块会露出 bg rect（白点）；「众数平滑」又会连细丝一起
    抹平。这里用距离变换找每个噪点像素最近的非噪点像素，把它的标签拷贝
    过来 —— 标签图仍是无缝拼图，既去斑又不破坏共边。
    """
    from scipy import ndimage as ndi
    from skimage import measure
    rid = measure.label(labels, connectivity=1)
    cnt = np.bincount(rid.ravel(), minlength=rid.max() + 1)
    slices = ndi.find_objects(rid)
    noise_ids = []
    for k, sl in enumerate(slices):
        if sl is None or cnt[k + 1] >= min_area or cnt[k + 1] == 0:
            continue
        bw = sl[1].stop - sl[1].start
        bh = sl[0].stop - sl[0].start
        ar = max(bw, bh) / max(1.0, min(bw, bh))
        if ar >= silk_ar and max(bw, bh) >= silk_len:
            continue                      # 细丝：保护
        noise_ids.append(k + 1)
    if not noise_ids:
        return labels, 0
    noise = np.isin(rid, noise_ids)
    _, inds = ndi.distance_transform_edt(noise, return_indices=True)
    out = labels.copy()
    out[noise] = labels[tuple(inds[:, noise])]
    return out, len(noise_ids)


def _despeckle_contrast(labels: np.ndarray, pal: np.ndarray, min_area: float,
                        silk_ar: float, silk_len: float,
                        de_thr: float) -> tuple[np.ndarray, int]:
    """局部色差感知去斑：只合并「小 + 非细丝 + 与邻域颜色突兀」的色块。

    与 ``_despeckle`` 的根本差别是多一道 OKLab ΔE 门槛：

    * 刺眼的亮蓝 / 亮黄小点：面积小、形状圆、与周围色差大 -> 合并；
    * 渐变里的中等色块：即使面积小，颜色与邻域连贯（ΔE 小）-> 保留；
    * 细长丝状块：形状判据保护 -> 保留。

    这正好避开两条已证伪的旧路：纯按面积去斑（D2/D3 把白发/枫叶抹成
    大色块平涂）、纯按颜色稀有度合并（E 组打不到常用色的孤立小块）。
    合并目标取「外环非噪点像素的众数色」，整块统一替换，标签图仍是
    无缝拼图，不露底、不破坏共边。
    """
    from scipy import ndimage as ndi
    from skimage import measure
    from qq3d_like import _srgb_to_oklab

    rid = measure.label(labels, connectivity=1)
    n = rid.max() + 1
    cnt = np.bincount(rid.ravel(), minlength=n)
    slices = ndi.find_objects(rid)
    ok = _srgb_to_oklab(pal[None].astype(np.float32) / 255.0)[0]
    K = len(pal)

    # 第一遍：按「小 + 非细丝 + ΔE 大」挑出噪点块
    cand: list[int] = []
    for k, sl in enumerate(slices):
        rk = k + 1
        if sl is None or cnt[rk] == 0 or cnt[rk] >= min_area:
            continue
        bw, bh = sl[1].stop - sl[1].start, sl[0].stop - sl[0].start
        ar = max(bw, bh) / max(1.0, min(bw, bh))
        if ar >= silk_ar and max(bw, bh) >= silk_len:
            continue                                   # 细丝保护
        # 外扩 2px 取邻域，按频次加权算邻域平均色（比单纯众数更稳）
        y0, y1 = max(0, sl[0].start - 2), min(labels.shape[0], sl[0].stop + 2)
        x0, x1 = max(0, sl[1].start - 2), min(labels.shape[1], sl[1].stop + 2)
        m = rid[y0:y1, x0:x1] == rk
        ring = ndi.binary_dilation(m, structure=np.ones((3, 3), np.uint8)) & ~m
        nb = labels[y0:y1, x0:x1][ring]
        me = labels[sl][rid[sl] == rk]
        if not len(nb) or not len(me):
            continue
        c = int(np.bincount(me, minlength=K).argmax())
        w = np.bincount(nb, minlength=K).astype(np.float64)
        if w.sum() <= 0:
            continue
        nb_ok = (ok * w[:, None]).sum(0) / w.sum()
        de = float(np.sqrt(((ok[c] - nb_ok) ** 2).sum()))
        if de >= de_thr:
            cand.append(rk)
    if not cand:
        return labels, 0

    noise = np.isin(rid, cand)
    out = labels.copy()
    handled = np.zeros_like(noise)
    # 第二遍：外环「排除其它噪点块」后取众数色，整块统一替换
    for rk in cand:
        sl = slices[rk - 1]
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - 3), min(labels.shape[0], sl[0].stop + 3)
        x0, x1 = max(0, sl[1].start - 3), min(labels.shape[1], sl[1].stop + 3)
        m = rid[y0:y1, x0:x1] == rk
        ring = (ndi.binary_dilation(m, structure=np.ones((3, 3), np.uint8))
                & ~m & ~noise[y0:y1, x0:x1])
        nb = labels[y0:y1, x0:x1][ring]
        if not len(nb):
            continue                                  # 交给 EDT 兜底
        target = int(np.bincount(nb, minlength=K).argmax())
        own = rid[sl] == rk
        out[sl][own] = target
        handled[sl][own] = True
    # 环内全是噪点的兜底：取最近的非噪点像素标签
    leftover = noise & ~handled
    if leftover.any():
        _, inds = ndi.distance_transform_edt(noise, return_indices=True)
        out[leftover] = labels[tuple(inds[:, leftover])]
    return out, len(cand)


def _coalesce_print(labels: np.ndarray, pal: np.ndarray, dot_area: float,
                    silk_ar: float, silk_len: float,
                    iters: int = 2) -> tuple[np.ndarray, int]:
    """打印可打印性合并：把小于一个挤出点的连通区域并入邻域。

    与 ``_despeckle_contrast`` 的区别：后者只打「颜色突兀」的小块（去刺眼
    彩色碎斑），这里打的是**物理尺寸**——面积小于一个挤出点的区域，无论
    颜色是否突兀，在切片时都会变成一个小圈 + 一次回抽，渐变里的微小等色
    块同样要并掉。判据完全由 print_mm / line_mm 推导，不写死像素。

    细长丝状块一律保留（上游「按整条线判断窄区」：真线删了会断画），
    迭代是因为两个相邻小块并到同一邻域后会连成稍大的块。
    """
    from scipy import ndimage as ndi
    from skimage import measure

    K = len(pal)
    out = labels.copy()
    total = 0
    for _ in range(max(1, iters)):
        rid = measure.label(out, connectivity=1)
        n = int(rid.max()) + 1
        cnt = np.bincount(rid.ravel(), minlength=n)
        slices = ndi.find_objects(rid)
        cand: list[int] = []
        for k, sl in enumerate(slices):
            rk = k + 1
            if sl is None or cnt[rk] == 0 or cnt[rk] >= dot_area:
                continue
            bw = sl[1].stop - sl[1].start
            bh = sl[0].stop - sl[0].start
            ar = max(bw, bh) / max(1.0, min(bw, bh))
            if ar >= silk_ar and max(bw, bh) >= silk_len:
                continue                                   # 细丝保护
            cand.append(rk)
        if not cand:
            break
        noise = np.isin(rid, cand)
        changed = 0
        for rk in cand:
            sl = slices[rk - 1]
            if sl is None:
                continue
            y0, y1 = max(0, sl[0].start - 3), min(out.shape[0], sl[0].stop + 3)
            x0, x1 = max(0, sl[1].start - 3), min(out.shape[1], sl[1].stop + 3)
            m = rid[y0:y1, x0:x1] == rk
            ring = (ndi.binary_dilation(m, structure=np.ones((3, 3), np.uint8))
                    & ~m & ~noise[y0:y1, x0:x1])
            nb = out[y0:y1, x0:x1][ring]
            if not len(nb):
                continue
            target = int(np.bincount(nb, minlength=K).argmax())
            own = rid[sl] == rk
            out[sl][own] = target
            changed += 1
        total += changed
        if changed == 0:
            break
    return out, total


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def trace(src: str, dst: str | None = None, *, colors: int = 116,
          tol: float = 0.9, max_chord: float = 3.5,
          smooth: float = 2.2, corner_deg: float = 45.0,
          label_smooth: float = 1.0,
          min_area: float = 0.0, rare_frac: float = 0.0,
          contrast_de: float = 0.0, contrast_area: float = 24.0,
          print_mm: float = 0.0, line_mm: float = 0.42,
          print_scale: float = 1.0, print_iters: int = 2,
          band_sigma: float = 0.0, band_contrast: float = 0.06,
          silk_ar: float = 2.2, silk_len: float = 7.0,
          expand: float = 0.9, seam: float = 0.8,
          fit_mode: str = "chord", max_side: int = 0, seed: int = 7,
          progress=None) -> dict:
    """丝状几何化重绘，返回与 geom_shared.trace 同结构 dict。"""
    t0 = time.time()

    def step(msg):
        if progress:
            progress(msg)

    step("读取图片…")
    im = Image.open(src).convert("RGB")
    if max_side and max(im.size) > max_side:
        r = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))),
                       Image.LANCZOS)
    w, h = im.size

    if band_sigma > 0:
        # 避线稿可变模糊（现成 banding.py，原先只挂在 core.py 的 cel 分支）：
        # 离线条越远模糊半径越大，把平滑渐变压成干净等色带，线条像素原样保留。
        try:
            import banding as _band
            step("渐变压平（避线稿可变模糊 σ≤%.1f）…" % band_sigma)
            im = _band.flatten(im.convert("RGB"), contrast=band_contrast,
                               max_sigma=band_sigma, near=1.0)
        except Exception as exc:  # noqa: BLE001
            step("渐变压平跳过（%s）" % exc)

    step("感知量化 %d 级…" % int(colors))
    q, labels, pal = qq.quantize_perceptual(im, int(colors), seed=seed,
                                            return_labels=True)
    if rare_frac > 0:
        step("合并稀有色（去彩色碎斑）…")
        labels, pal, nrare = _merge_rare_colors(labels, pal, rare_frac)
        q = Image.fromarray(pal[np.clip(labels, 0, len(pal) - 1)])
    else:
        nrare = 0
    if label_smooth > 0:
        step("标签平滑…")
        labels = qq._label_smooth(labels, len(pal), label_smooth)
    if min_area > 0:
        step("形状保护去斑（小色块并入邻域）…")
        labels, ndespeck = _despeckle(labels, min_area, silk_ar, silk_len)
    else:
        ndespeck = 0
    if contrast_de > 0:
        step("局部色差感知去斑（只并颜色突兀的小块）…")
        labels, ncontrast = _despeckle_contrast(
            labels, pal, contrast_area, silk_ar, silk_len, contrast_de)
    else:
        ncontrast = 0
    nprint = 0
    if print_mm > 0 and print_scale > 0:
        bead = float(line_mm) / float(print_mm) * max(w, h)
        dot_area = (bead * print_scale) ** 2
        step("打印可打印性合并（< %.1f px² 区域并入邻域）…" % dot_area)
        labels, nprint = _coalesce_print(labels, pal, dot_area,
                                         silk_ar, silk_len, print_iters)
        q = Image.fromarray(pal[np.clip(labels, 0, len(pal) - 1)])

    step("按同色连通块切分色块…")
    from scipy import ndimage as ndi
    from skimage import measure
    rid = measure.label(labels, connectivity=1)          # 同色 + 4 连通 = 一个色块
    slices = ndi.find_objects(rid)
    nblk = len(slices)

    parts: list[str] = []
    used: set[str] = set()
    npaths = 0
    kept_silk = 0
    ncmd = 0
    step("丝状拟合（一块一 path / 短 A·Q 链）…")
    for k, sl in enumerate(slices):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        bw, bh = x1 - x0, y1 - y0
        # bbox 左上角未必属于该块（斜细丝），必须从块内真实像素取色
        yy, xx = np.nonzero(rid[y0:y1, x0:x1] == (k + 1))
        if len(yy) == 0:
            continue
        c = int(labels[y0 + yy[0], x0 + xx[0]])
        # 细长丝状块单独标记：限制外扩/描边宽度，避免 1px 丝被撑粗
        ar = max(bw, bh) / max(1.0, min(bw, bh))
        is_silk = ar >= silk_ar and max(bw, bh) >= silk_len
        if is_silk:
            kept_silk += 1

        y0p, y1p = max(0, y0 - 2), min(h, y1 + 2)
        x0p, x1p = max(0, x0 - 2), min(w, x1 + 2)
        crop = (rid[y0p:y1p, x0p:x1p] == (k + 1)).astype(np.float32)
        p = np.pad(crop, 2, mode="constant", constant_values=0.0)
        subs: list[str] = []
        for cc in measure.find_contours(p, 0.5, fully_connected="low"):
            xy = cc[:, ::-1].copy()
            xy[:, 0] += x0p - 2
            xy[:, 1] += y0p - 2
            if len(xy) < 8:
                continue
            if expand:
                # 细丝按自身宽度限制外扩，避免 1px 丝被撑成 3px
                dl = min(expand, 0.35 * min(bw, bh)) if is_silk else expand
                xy = gs._expand_ring(xy, max(0.0, dl))
                d = _fit_closed_silk(xy, tol, max_chord, smooth, corner_deg, fit_mode)
            if d:
                subs.append(d)
        if not subs:
            continue
        col = "#%02x%02x%02x" % tuple(int(v) for v in pal[c])
        d = " ".join(subs)
        ncmd += sum(len(re.findall(r"[A-Za-z]", x)) for x in subs)
        if seam > 0:
            sw = min(seam, 0.3 * min(bw, bh)) if is_silk else seam
            parts.append(
                '<path class="qq3d-gap-filler" d="%s" fill="none" stroke="%s" '
                'stroke-width="%.2f" stroke-linejoin="round" '
                'stroke-linecap="round"/>' % (d, col, max(0.15, sw)))
        parts.append('<path d="%s" fill="%s"/>' % (d, col))
        used.add(col)
        npaths += 1
        if progress and (k + 1) % 2000 == 0:
            step("  已拟合 %d/%d 个色块…" % (k + 1, nblk))

    bg = qq._bg_color(q)
    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
           'width="%d" height="%d" viewBox="0 0 %d %d">\n'
           '<rect x="0" y="0" width="%d" height="%d" fill="%s"/>\n'
           % (w, h, w, h, w, h, bg)
           + "\n".join(parts) + "\n</svg>\n")
    if dst:
        io.open(dst, "w", encoding="utf-8", newline="\n").write(svg)
    step("完成 %d 个色块 / %d 细丝 / %d 命令 / %d 色（稀有色 %d / 面积去斑 %d / 色差去斑 %d / 打印合并 %d）"
         % (npaths, kept_silk, ncmd, len(used), nrare, ndespeck, ncontrast, nprint))
    return {"svg": svg, "paths": npaths, "subpaths": npaths,
            "colors": len(used), "width": w, "height": h,
            "commands": ncmd, "silk": kept_silk, "blocks": nblk,
            "contrast": ncontrast, "print_merged": nprint,
            "bytes": os.path.getsize(dst) if dst else len(svg.encode("utf-8")),
            "ms": int((time.time() - t0) * 1000)}
