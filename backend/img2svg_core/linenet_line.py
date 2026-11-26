# -*- coding: utf-8 -*-
"""LineNet v0.4 线稿识别 (v7.17.6 新增) — 离线 ONNX 推理，输出矢量线稿。

模型：linenet-v0.4.0.onnx（46.7 万参数轻量 U-Net，1.8 MB，仅 CPU 推理，
required provider = CPUExecutionProvider，不依赖 GPU / torch / scipy / skimage）。

流程（复刻 $Bug$ 路线：CNN 出线条热力图 → 传统算法细化成线）：
  1. 分块推理 + 羽化 → 线概率热力图（避免大图吃满内存，消除拼接缝）
  2. Steger 亚像素线提取（Hessian 法线过零点 + 双线性投票 + 法线峰约束）
     —— 用 OpenCV 解析高斯导数实现，与离线 scipy 版数学等价，但不引入 scipy
  3. 膨胀成细带 + cv2.findContours → SVG path（纯矢量，不依赖未打包的 potrace）

本模块只依赖 numpy / cv2 / onnxruntime / PIL，均为 exe 已有依赖。
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

MODEL_NAME = "linenet-v0.4.0.onnx"
_TILE, _STRIDE, _OV = 128, 96, 32
_SESSION = None

# 底图/线层开关（core._apply_linenet 读取），保留旧实现便于回滚对比：
#   USE_CENTERLINE —— 线层走骨架中线单条 path（旧实现描 1px 带的两侧轮廓）
#   COMPACT_BASE  —— 底图压缩（v7.17.10 起 = 属性提升 + d 简化）
#   PATH_EPS      —— d 简化里「C 段视为直线」的容差（网格单位 @3200 网格），
#                    0 表示只做 v7.17.9 的整数化。实测 0.30 → 体积 −38~41%，
#                    渲染差 mean 0.09/255 且无 >32 像素；0.50 可到 −45~51%。
USE_CENTERLINE = True
COMPACT_BASE = True
PATH_EPS = 0.30

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def available() -> bool:
    if cv2 is None:
        return False
    return os.path.exists(model_path())


def model_path() -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "models", MODEL_NAME)


def _get_session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        _SESSION = ort.InferenceSession(model_path(),
                                        providers=["CPUExecutionProvider"])
    return _SESSION


def _feather() -> np.ndarray:
    yy, xx = np.mgrid[0:_TILE, 0:_TILE].astype(np.float32)
    fw = np.minimum.reduce([yy, xx, _TILE - 1 - yy, _TILE - 1 - xx])
    return np.clip(fw / 16.0, 0, 1)


def _axis_pos(n: int) -> list[int]:
    ps = list(range(0, max(1, n - _TILE + 1), _STRIDE))
    if not ps or ps[-1] != max(0, n - _TILE):
        ps.append(max(0, n - _TILE))
    return ps


def heatmap(rgb: np.ndarray, progress=None) -> np.ndarray:
    """rgb: H×W×3 uint8 → H×W float32 线概率（0~1）。分块推理 + 羽化。"""
    h, w = rgb.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    x = np.pad(rgb.astype(np.float32) / 255.0,
               ((0, ph), (0, pw), (0, 0)), mode="reflect")
    H, W = x.shape[:2]
    sess = _get_session()
    acc = np.zeros((H, W), np.float32)
    wgt = np.zeros((H, W), np.float32)
    fw = _feather()
    ys, xs = _axis_pos(H), _axis_pos(W)
    total = len(ys) * len(xs)
    done = 0
    for y in ys:
        for x0 in xs:
            p = np.ascontiguousarray(
                x[y:y + _TILE, x0:x0 + _TILE].transpose(2, 0, 1)[None])
            logit = sess.run(None, {"input": p})[0][0, 0]
            prob = 1.0 / (1.0 + np.exp(-logit))
            acc[y:y + _TILE, x0:x0 + _TILE] += prob * fw
            wgt[y:y + _TILE, x0:x0 + _TILE] += fw
            done += 1
        if progress and total:
            progress("CNN 识别线稿 %d/%d 块…" % (done, total))
    return (acc / np.maximum(wgt, 1e-6))[:h, :w]


def _gauss_derivs(f: np.ndarray, sigma: float):
    """解析高斯一/二阶偏导，与 scipy.ndimage.gaussian_filter(order=..) 等价。

    用可分离卷积构造高斯核及其 1/2 阶解析导数核，避免 Sobel 差分带来的
    量级失配（Sobel 近似会让法线/过零判定漂移，线量偏多）。无 scipy 依赖。
    """
    rad = max(2, int(4 * sigma + 0.5))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    g = np.exp(-x * x / (2 * sigma * sigma))
    g /= g.sum()
    g1 = (-x / (sigma * sigma)) * g
    g2 = (x * x / (sigma ** 4) - 1.0 / (sigma * sigma)) * g
    g = g.astype(np.float32); g1 = g1.astype(np.float32); g2 = g2.astype(np.float32)

    def sep(kx, ky):
        return cv2.sepFilter2D(f, cv2.CV_32F, kx, ky,
                               borderType=cv2.BORDER_REFLECT)

    sm = sep(g, g)
    # 注意：scipy.gaussian_filter(order=1) 的导数核带负号约定，
    # 这里取负号与之对齐（离线验证 corr=-1.0），否则线心过零点漂移。
    fx = -sep(g1, g)   # x 方向一阶（axis=1）
    fy = -sep(g, g1)   # y 方向一阶（axis=0）
    fxx = sep(g2, g)
    fyy = sep(g, g2)
    fxy = sep(g1, g1)
    return sm, fx, fy, fxx, fxy, fyy


def steger_lines(heat: np.ndarray, sigma: float = 1.0,
                 high: float = 0.12, min_size: int = 6) -> np.ndarray:
    """Steger 亚像素线提取，返回 bool 线心图。cv2 版，无 scipy/skimage 依赖。"""
    f = heat.astype(np.float32)
    sm, fx, fy, fxx, fxy, fyy = _gauss_derivs(f, sigma)

    theta = 0.5 * np.arctan2(2 * fxy, fxx - fyy)
    nx = -np.sin(theta)
    ny = np.cos(theta)

    d2 = nx * nx * fxx + 2 * nx * ny * fxy + ny * ny * fyy
    d1 = fx * nx + fy * ny
    safe = np.where(d2 < -1e-4, d2, -1e-4)
    t = np.clip(-d1 / safe, -0.7071, 0.7071)

    h, w = f.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    px = xs + t * nx
    py = ys + t * ny

    def sampf(d):
        yy = np.clip(ys + d * ny, 0, h - 1)
        xx = np.clip(xs + d * nx, 0, w - 1)
        x0 = np.floor(xx).astype(int); y0 = np.floor(yy).astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1); y1 = np.clip(y0 + 1, 0, h - 1)
        wx = xx - x0; wy = yy - y0
        return (sm[y0, x0] * (1 - wx) * (1 - wy) + sm[y0, x1] * wx * (1 - wy) +
                sm[y1, x0] * (1 - wx) * wy + sm[y1, x1] * wx * wy)

    stronger = np.zeros_like(sm, bool)
    for d in (-2.0, -1.0, 1.0, 2.0):
        stronger |= sampf(d) > sm + 1e-4
    is_peak = ~stronger
    accept = (np.abs(t) <= 0.7071) & (d2 < -1e-3) & (sm > high) & is_peak

    x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int)
    x1 = x0 + 1; y1 = y0 + 1
    fx_ = px - x0; fy_ = py - y0
    votes = np.zeros((h, w), np.float32)
    for yy, xx, wt in [(y0, x0, (1 - fx_) * (1 - fy_)),
                       (y0, x1, fx_ * (1 - fy_)),
                       (y1, x0, (1 - fx_) * fy_),
                       (y1, x1, fx_ * fy_)]:
        m = accept & (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        np.add.at(votes, (yy[m], xx[m]), wt[m])
    line = votes > 0.35

    if min_size:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            line.astype(np.uint8), 8)
        keep = np.zeros_like(line)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_size:
                keep |= lab == i
        line = keep
    return line


def line_mask(rgb: np.ndarray, progress=None, sigma: float = 1.0,
              high: float = 0.12) -> np.ndarray:
    """原图 RGB → bool 线稿 mask。"""
    return steger_lines(heatmap(rgb, progress=progress), sigma=sigma, high=high)


_NB = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def thin(mask: np.ndarray, max_iter: int = 48) -> np.ndarray:
    """细化到 1 px 骨架。优先 opencv-contrib 的 ximgproc（C++ 快），否则
    Zhang-Suen 纯 numpy 向量化实现（无额外依赖，exe 里一定可用）。

    只做必要的一步：厚度 ≤1 px 的 mask（分割边界、Steger 线心）本来就不用
    细化，先探测一次避免白跑几十轮。
    """
    m = mask.astype(np.uint8)
    if not m.any():
        return mask.astype(bool)
    # 是否需要细化，用「完整 2×2 前景块」的覆盖率判：真·粗带里几乎每个像素都
    # 落在某个 2×2 块里（实测覆盖率 2.5~3.3），而已是 1 px 的线心/边界只有
    # 0.26~0.40。对后者跳过细化很重要——Zhang-Suen 会削掉端点、在交叉点附近
    # 把长链切成几段，白白多出一批短 path。
    er = cv2.erode(m, np.ones((2, 2), np.uint8))
    if 4.0 * float(er.sum()) < float(m.sum()):
        return m.astype(bool)

    try:
        return cv2.ximgproc.thinning(m * 255, cv2.THINNING_ZHANGSUEN) > 0
    except Exception:  # noqa: BLE001  contrib 未打包时走下面的自研实现
        pass

    # 只在 mask 外接框内迭代：图幅 3200×4000 时整幅跑一轮要 0.6s，而线只占
    # 中间一块，裁掉空白边通常能省一半以上。
    ys, xs = np.nonzero(m)
    y0, y1 = max(0, int(ys.min()) - 1), min(m.shape[0], int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min()) - 1), min(m.shape[1], int(xs.max()) + 2)
    p = np.pad(m[y0:y1, x0:x1], 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            c = p[1:-1, 1:-1]
            P2, P3, P4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            P8, P9 = p[1:-1, :-2], p[:-2, :-2]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            # A = 0→1 的环绕次数（8 邻域环上的跳变数）
            ring = np.stack([P2, P3, P4, P5, P6, P7, P8, P9, P2])
            A = (ring[:-1] == 0)
            A = (A & (ring[1:] == 1)).sum(0, dtype=np.uint8)
            if step == 0:
                cond = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            rem = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rem.any():
                c[rem] = 0
                changed = True
        if not changed:
            break
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = p[1:-1, 1:-1]
    return out.astype(bool)


def trace_centerlines(skel: np.ndarray, min_len: float = 6.0,
                      smooth: int = 1, eps: float = 0.7, prec: int = 1):
    """1 px 骨架 → 中心线折线列表（每条 open path 一段，不再绕双侧）。

    为什么不用 findContours：1 px 细线的轮廓 = 沿一侧走到头再沿另一侧折回，
    同一条线被画了两遍，点数也是两倍 —— 这是旧版「一条线画两侧、体积压不下」
    的直接原因。这里把骨架当图：度=1 的端点、度≥3 的交叉点作为链端点，沿
    未走过的邻边走出长链，度=2 的闭环单独抽取，每条链即一条中线。

    返回 [(H×W×2 float32 折线, 弧长)]。坐标是 mask 像素坐标。
    """
    h, w = skel.shape
    ys, xs = np.nonzero(skel)
    n = len(ys)
    if n == 0:
        return []

    lut = np.full(h * w + 1, -1, np.int64)          # 越界统一落到末尾的 -1
    lut[ys * w + xs] = np.arange(n)
    nbr = np.full((n, 8), -1, np.int64)
    deg = np.zeros(n, np.int8)
    for k, (dy, dx) in enumerate(_NB):
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        j = np.where(ok, lut[np.where(ok, yy * w + xx, 0)], -1)
        nbr[:, k] = j
        deg += (j >= 0)

    used = np.zeros((n, 8), bool)

    def walk(a, k, b):
        """从边 a -k-> b 出发一路走到底，返回链（点 id 列表）。"""
        chain = [a]
        used[a, k] = True
        used[b, (k + 4) % 8] = True
        cur = b
        while True:
            chain.append(cur)
            if deg[cur] != 2:
                break
            nb = nbr[cur]
            nxt = -1
            for kk in range(8):
                j = nb[kk]
                if j >= 0 and not used[cur, kk]:
                    nxt = kk
                    break
            if nxt < 0:
                break
            used[cur, nxt] = True
            used[nb[nxt], (nxt + 4) % 8] = True
            cur = nb[nxt]
        return chain

    chains = []
    for s in np.nonzero(deg != 2)[0]:                # 端点 + 交叉点先走
        for k in range(8):
            j = nbr[s, k]
            if j >= 0 and not used[s, k]:
                chains.append(walk(s, k, j))
    for s in np.nonzero(deg == 2)[0]:                # 剩下的都是闭环
        if not used[s].any():
            k = int(np.argmax(nbr[s] >= 0))
            chains.append(walk(s, k, nbr[s, k]))

    out = []
    for ch in chains:
        if len(ch) < 2:
            continue
        p = np.stack([xs[ch], ys[ch]], axis=1).astype(np.float32)
        if smooth and len(p) > 2:
            for _ in range(int(smooth)):             # 吃掉 1 px 台阶
                q = p.copy()
                q[1:-1] = (p[:-2] + 2.0 * p[1:-1] + p[2:]) * 0.25
                p = q
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        arc = float(seg.sum())
        if arc < min_len:
            continue
        ap = cv2.approxPolyDP(p.reshape(-1, 1, 2), eps, False).reshape(-1, 2)
        if len(ap) < 2:
            continue
        out.append((ap, arc))
    out.sort(key=lambda t: -t[1])
    return out


def _poly_d(pts: np.ndarray, prec: int = 1) -> str:
    f = "%." + str(prec) + "f,%." + str(prec) + "f"
    return "M" + " L".join(f % (float(x), float(y)) for x, y in pts)


def stroke_svg_center(mask: np.ndarray, stroke_width: float = 1.6,
                      color: str = "#141414", min_len: float = 8.0,
                      eps: float = 0.7, prec: int = 1) -> tuple:
    """bool mask → 中心线描边 SVG（每条线一条 open path，不再绕双侧）。

    与 stroke_svg 的区别：那里是「沿着 1 px 带的轮廓描一圈」，同一条线走两
    趟；这里先细化再用 graph 追踪，只出中线一条。path 数、点数都近似减半。
    """
    chains = trace_centerlines(thin(mask), min_len=min_len, eps=eps)
    if not chains:
        return "", 0
    d = " ".join(_poly_d(p, prec) for p, _ in chains)
    seg = ('<g fill="none" stroke="%s" stroke-width="%.2f" stroke-linejoin="round" '
           'stroke-linecap="round"><path d="%s"/></g>\n' % (color, stroke_width, d))
    return seg, len(chains)


def build_centerline_svg(rgb: np.ndarray, mask: np.ndarray,
                         stroke_width: float = 1.6, n_colors: int = 10,
                         min_len: float = 8.0, eps: float = 0.7,
                         prec: int = 1) -> tuple:
    """同 build_layered_svg，但线层走中心线（L0 暗墨 + L1 彩线，按色聚类）。

    每条链取沿线原图平均色 → _inkify → 聚成 ≤n_colors 组，每组一个 <g>。
    返回 (片段, 路径数, 组数)。
    """
    chains = trace_centerlines(thin(mask), min_len=min_len, eps=eps)
    if not chains:
        return "", 0, 0
    h, w = mask.shape[:2]
    cols = []
    for p, _ in chains:
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
        cols.append(rgb[yi, xi].mean(0, dtype=np.float64))
    cols = _inkify(np.asarray(cols, np.float32))

    k = int(max(1, min(n_colors, len(chains))))
    if len(chains) > k:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _, labels, centers = cv2.kmeans(cols.astype(np.float32), k, None, crit, 2,
                                        cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(-1)
        centers = centers.astype(np.float32)
    else:
        labels = np.arange(len(chains))
        centers = cols

    order = np.argsort(-centers.max(axis=1))          # 亮的先画、暗的最后压上
    segs = []
    for gi in order:
        d = " ".join(_poly_d(p, prec) for p, l in zip((c[0] for c in chains), labels)
                     if int(l) == int(gi))
        if not d:
            continue
        r, g, b = (int(np.clip(round(v), 0, 255)) for v in centers[gi])
        segs.append('<g fill="none" stroke="#%02x%02x%02x" stroke-width="%.2f" '
                    'stroke-linejoin="round" stroke-linecap="round"><path d="%s"/></g>\n'
                    % (r, g, b, stroke_width, d))
    return ''.join(segs), len(chains), len(segs)


def compact_coords(svg: str, decimals: int = 0) -> str:
    """把 SVG 里 path 的 d 坐标压到指定小数位（linenet 模式专用，体积杀手）。

    底图 viewBox 是 3200 网格，1 个网格单位在 720 CSS px 的显示宽度下只有
    0.225 px；vtracer 默认给 2 位小数（精度 1/32000），远超源图能承载的信息。
    压到整数后每张图实测省 32%，渲染像素差在 1/255 量级，肉眼不可见。

    只动 d="..." 属性内部，绝不碰 viewBox / width / stroke-width / transform。
    """
    pat = re.compile(r'd="([^"]*)"')
    num = re.compile(r'-?\d+\.\d+')

    def fix(m):
        s = m.group(1)

        def r(mm):
            fmt = "%." + str(decimals) + "f"
            out = fmt % float(mm.group(0))
            if out.startswith("-") and float(out) == 0.0:   # -0 → 0
                return "0"
            if "." in out:
                out = out.rstrip("0").rstrip(".")
            return out

        return 'd="%s"' % num.sub(r, s)

    return pat.sub(fix, svg)


def _comp_colors(rgb: np.ndarray, lab: np.ndarray, n: int) -> np.ndarray:
    """每个连通线段的原图平均色（N×3 float）。"""
    flat = lab.reshape(-1).astype(np.int64)
    cols = rgb.reshape(-1, 3).astype(np.float64)
    cnt = np.bincount(flat, minlength=n).astype(np.float64)
    sums = np.stack([np.bincount(flat, weights=cols[:, c], minlength=n)
                     for c in range(3)], axis=1)
    return sums / np.maximum(cnt, 1)[:, None]


def _inkify(cols: np.ndarray) -> np.ndarray:
    """按 $Bug$ 的 L0/L1 思路把「线段的原图色」整理成墨色。

    - 暗段（V<0.34）→ L0 暗墨：压到近黑并保留一点色相，避免死板纯黑
    - 彩色段（S≥0.12）→ L1 彩线：饱和度 +30%、亮度 ×0.82，保留原色相
      （皮肤线偏橙、发丝线偏蓝紫，正是参考图里 L1 的样子）
    - 中性浅段 → 轻微压暗，防止与浅底色一起消失
    """
    c8 = np.clip(cols, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    hsv = cv2.cvtColor(c8, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    s, v = hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    dark = v < 0.34
    chrom = (~dark) & (s >= 0.12)
    neutral = (~dark) & (~chrom)

    # L0：按目标亮度等比缩放 RGB
    tgt = 0.10 + 0.20 * v
    scale = np.where(dark, tgt / np.maximum(v, 1e-6), 1.0)
    # 必须 clip：v 接近 0（纯黑线）时 tgt/v 会放大到把 RGB 顶出 255，
    # 下游 "%02x" 会格式化出 16 字符的非法颜色（如 #1631d13f9a13f9a）。
    out = np.clip(cols.astype(np.float32) * scale[:, None], 0, 255)

    hsv2 = hsv.copy()
    hsv2[chrom, 1] = np.clip(hsv2[chrom, 1] * 1.30, 0, 255)
    hsv2[chrom, 2] = np.clip(hsv2[chrom, 2] * 0.82, 8, 255)
    # 中性浅段压暗并封顶：亮部发丝上的浅色线画出来像白划痕，必须避免
    hsv2[neutral, 2] = np.clip(hsv2[neutral, 2] * 0.92, 0, 140)
    conv = cv2.cvtColor(hsv2.astype(np.uint8).reshape(1, -1, 3),
                        cv2.COLOR_HSV2RGB).reshape(-1, 3)
    return np.where(dark[:, None], out, conv.astype(np.float32))


def build_layered_svg(rgb: np.ndarray, mask: np.ndarray, dilate: int = 0,
                      stroke_width: float = 1.6, n_colors: int = 10,
                      bezier: bool = False) -> tuple:
    """线稿 mask → 分层墨线 SVG 片段（L0 暗墨 + L1 彩线，按色聚类分组）。

    每条线段取原图平均色，经 _inkify 后聚成 ≤n_colors 组；每组一个 <g>，
    共用一个 stroke 颜色，路径不与颜色数成正比膨胀。返回 (片段, path 数, 组数)。
    """
    m = mask.astype(np.uint8)
    if dilate:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=dilate)
    h, w = m.shape[:2]
    n, lab = cv2.connectedComponents(m, 8)[:2]
    mean = _comp_colors(rgb, lab, n)

    contours, _ = cv2.findContours(m * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _ot = None
    if bezier:
        import own_tracer as _ot  # 复用其 Schneider 三次贝塞尔拟合
    paths, cols = [], []
    for c in contours:
        if len(c) < 2:
            continue
        x = int(np.clip(c.reshape(-1, 2)[0][0], 0, w - 1))
        y = int(np.clip(c.reshape(-1, 2)[0][1], 0, h - 1))
        lbl = int(lab[y, x])
        if lbl <= 0:
            continue
        if _ot is not None:
            # 折线会让细长轮廓留着像素台阶，放大看是一节节锯齿；贝塞尔更顺
            segs = _ot.fit_closed(c.reshape(-1, 2).astype(np.float64), tol=0.55)
            if not segs:
                continue
            d_ = ["M%.2f,%.2f" % tuple(segs[0][0])]
            for sg in segs:
                d_.append("C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (*sg[1], *sg[2], *sg[3]))
            paths.append("".join(d_) + "Z")
        else:
            ap = cv2.approxPolyDP(c, 0.3, True).reshape(-1, 2)
            if len(ap) < 2:
                continue
            paths.append('M' + ' L'.join('%d,%d' % (int(p[0]), int(p[1])) for p in ap) + ' Z')
        cols.append(mean[lbl])
    if not paths:
        return "", 0, 0

    cols = _inkify(np.asarray(cols, np.float32))
    k = int(max(1, min(n_colors, len(paths))))
    if len(paths) > k:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _, labels, centers = cv2.kmeans(cols.astype(np.float32), k, None, crit, 2,
                                        cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(-1)
        centers = centers.astype(np.float32)
    else:
        labels = np.arange(len(paths))
        centers = cols

    # 亮的先画、暗的最后压上，保证轮廓墨线在顶层
    order = np.argsort(-centers.max(axis=1))
    segs = []
    for gi in order:
        d = ' '.join(p for p, l in zip(paths, labels) if int(l) == int(gi))
        if not d:
            continue
        r, g, b = (int(np.clip(round(v), 0, 255)) for v in centers[gi])
        segs.append('<g fill="none" stroke="#%02x%02x%02x" stroke-width="%.2f" '
                    'stroke-linejoin="round" stroke-linecap="round"><path d="%s"/></g>\n'
                    % (r, g, b, stroke_width, d))
    return ''.join(segs), len(paths), len(segs)


def block_outline(svg: str, color: str = "#141414",
                  width: float = 2.2) -> tuple:
    """底图每个色块 path 的同色遮缝描边 → 深色细描边，得到 L2「色块边界」。

    不新增任何路径数据（复用色块路径本身），所以文件几乎不增大，而且
    描边天然与色块边缘完全重合，不会出现 CNN 线与色块错位的彩边。
    返回 (新 SVG, 改写的 path 数)。
    """
    pat = re.compile(r'stroke="#[0-9a-fA-F]{6}" stroke-width="[0-9.]+"')
    out, n = pat.subn('stroke="%s" stroke-width="%.2f"' % (color, width), svg)
    return out, n


def boundary_mask(index: np.ndarray) -> np.ndarray:
    """色块标签图 → 相邻异标签的边界 bool mask（1 px 宽）。"""
    b = np.zeros(index.shape, bool)
    b[:, 1:] |= index[:, 1:] != index[:, :-1]
    b[1:, :] |= index[1:, :] != index[:-1, :]
    return b


def filter_small(mask: np.ndarray, min_size: int = 8) -> np.ndarray:
    """去掉面积 < min_size 的线片段（去重后残留的碎点会显得脏）。"""
    if min_size <= 1:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_size:
            keep |= lab == i
    return keep


def strong_boundary(index: np.ndarray, pal: np.ndarray, de_min: float = 0.06,
                    min_size: int = 20, min_area: int = 0) -> np.ndarray:
    """只保留「相邻色块色差足够大」的边界（OKLab ΔE 门控）。

    柔和渐变里量化台阶之间的 ΔE 只有 JND 量级（≈0.02），勾出来就是一圈圈
    等高线噪声；真实轮廓（深色线 vs 皮肤）ΔE 常在 0.3 以上。用这个判据就能
    把「该画的轮廓」和「量化台阶」分开，而且完全不依赖 CNN 的稀疏响应。

    min_area：两侧色块面积都必须 ≥ 该值才描边 —— 高光细丝那种几像素的小片
    被勾出来会像白划痕，直接跳过。
    """
    import color as _color
    pal = np.asarray(pal, np.uint8)
    n = int(index.max()) + 1
    lab = _color.srgb_to_oklab(pal.astype(np.float32) / 255.0)
    if len(lab) < n:                      # 重取色后标签可能超过调色板长度
        lab = np.vstack([lab, np.repeat(lab[-1:], n - len(lab), axis=0)])
    D = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1).astype(np.float32)
    if min_area:
        area = np.bincount(index.ravel(), minlength=n).astype(np.int32)
        area = np.where(area >= int(min_area), 1, 0).astype(np.float32)
    else:
        area = None

    def ok(a, b):
        m = D[a, b] > de_min
        if area is not None:
            m &= (area[a] > 0) & (area[b] > 0)
        return m

    h, w = index.shape
    b = np.zeros((h, w), bool)
    step = 256
    for y in range(0, h, step):                      # 分块避免 12M 级中间数组
        sub = index[y:y + step]
        b[y:y + step, 1:] = ok(sub[:, :-1], sub[:, 1:])
    for y in range(0, h - 1, step):
        y2 = min(y + step, h - 1)
        b[y + 1:y2 + 1] = b[y + 1:y2 + 1] | ok(index[y:y2], index[y + 1:y2 + 1])
    return filter_small(b, min_size)


def heat_gate(heat: np.ndarray, shape_hw: tuple, thr: float = 0.30,
              min_size: int = 12) -> np.ndarray:
    """热力图软约束 → 目标网格尺寸下的 bool 掩码（$Bug$「热力图只出约束」）。

    热力图不必精确提线，它的职责只是回答「这里该不该有线」。连续概率阈值化
    后是连成片的带（不像 NMS 骨架那样稀疏），缩放到分割网格后能稳稳覆盖对应
    的色块边界，所以与 ΔE 门控**并集**使用、互补：

      - ΔE 抓强对比轮廓（黑线 vs 皮肤，ΔE>0.3）
      - heat 抓淡彩轮廓（ΔE 不到阈值但确实有线，如浅色衣服的褶皱）

    实测四张风格差异很大的图（扁平动漫/柔光霓虹/淡彩/浅色少女），热力图 p97
    都落在 0.30±0.01，故固定 0.30 即等价于自适应阈值，无需逐图调参。
    """
    h, w = int(shape_hw[0]), int(shape_hw[1])
    hz = heat.astype(np.float32)
    if hz.shape[:2] != (h, w):
        hz = cv2.resize(hz, (w, h), interpolation=cv2.INTER_LINEAR)
    return filter_small(hz > thr, min_size)


def stroke_svg(mask: np.ndarray, stroke_width: float = 1.8,
               color: str = "#141414", min_size: int = 12,
               eps: float = 0.5) -> tuple:
    """bool 线/边界 mask → 描边 SVG 片段（坐标即 mask 像素坐标）。"""
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_size:
            keep |= (lab == i)
    if not keep.any():
        return "", 0
    contours, _ = cv2.findContours(keep * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for c in contours:
        if len(c) < 2:
            continue
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) < 2:
            continue
        parts.append('M' + ' L'.join('%d,%d' % (int(p[0]), int(p[1])) for p in ap) + ' Z')
    if not parts:
        return "", 0
    seg = ('<g fill="none" stroke="%s" stroke-width="%.2f" stroke-linejoin="round" '
           'stroke-linecap="round"><path d="%s"/></g>\n'
           % (color, stroke_width, ' '.join(parts)))
    return seg, len(parts)


def stroke_svg_bezier(mask: np.ndarray, stroke_width: float = 1.8,
                      color: str = "#141414", min_size: int = 12,
                      tol: float = 1.2, corner_deg: float = 55.0,
                      eps: float = 1.0, prec: int = 1) -> tuple:
    """同 stroke_svg，但轮廓用三次贝塞尔拟合（复用 own_tracer.fit_closed）。

    $Bug$ 实测：热力图直接转 SVG（折线）得到 1781 段且毛刺明显，改贝塞尔后
    474 段、线条连贯顺滑，Bezier+Fill 达 22.4dB。折线在细长轮廓上会保留像素
    台阶（放大看是一节节的锯齿），贝塞尔拟合能把台阶吃掉，同时把每条路径的
    点数降一个量级，文件明显变小。

    eps>0 时先用 approxPolyDP 粗化再拟合：1 px 线的轮廓点数极多，粗化不改变
    形状（后续 fit_closed 会按 1 px 重采样），只为提速。
    """
    import own_tracer as _ot

    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_size:
            keep |= (lab == i)
    if not keep.any():
        return "", 0

    contours, _ = cv2.findContours(keep, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    parts = []
    for c in contours:
        if len(c) < 3:
            continue
        poly = c.reshape(-1, 2).astype(np.float32)
        if eps > 0:
            poly = cv2.approxPolyDP(poly.reshape(-1, 1, 2), eps, True)
            poly = poly.reshape(-1, 2)
        if len(poly) < 3:
            continue
        segs = _ot.fit_closed(poly.astype(np.float64), tol=tol,
                             corner_deg=corner_deg)
        if not segs:
            continue
        f = "%." + str(prec) + "f,%." + str(prec) + "f"
        d = ["M" + f % tuple(segs[0][0])]
        cf = "C" + " ".join([f] * 3)
        for sg in segs:
            d.append(cf % (*sg[1], *sg[2], *sg[3]))
        parts.append("".join(d) + "Z")
    if not parts:
        return "", 0
    seg = ('<g fill="none" stroke="%s" stroke-width="%.2f" stroke-linejoin="round" '
           'stroke-linecap="round"><path d="%s"/></g>\n'
           % (color, stroke_width, ' '.join(parts)))
    return seg, len(parts)


def build_boundary_svg(index: np.ndarray, support: np.ndarray | None = None,
                       stroke_width: float = 1.8, min_size: int = 20,
                       color: str = "#141414", support_dilate: int = 2):
    """色块边界 → 矢量描边层（L2），可选 CNN 支持门控。

    support（CNN 线概率/掩码）给出「这里该不该有线」的判断：
    只保留与 CNN 线邻近的色块边界，柔和渐变里那些一圈圈的等高线边界会被
    滤掉；平坦插画的真实轮廓因为 CNN 也认得，全部保留。

    返回 (SVG 片段, path 数, 保留边界像素占比)。
    """
    b = boundary_mask(index).astype(np.uint8)
    frac0 = float(b.mean())
    if support is not None:
        s = support
        if s.shape != b.shape:
            s = cv2.resize(s.astype(np.uint8), (b.shape[1], b.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        if support_dilate:
            s = cv2.dilate(s, np.ones((3, 3), np.uint8), iterations=support_dilate)
        b = b & (s > 0).astype(np.uint8)
    b = filter_small(b.astype(bool), min_size).astype(np.uint8)
    if not b.any():
        return "", 0, 0.0
    contours, _ = cv2.findContours(b * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for c in contours:
        if len(c) < 2:
            continue
        ap = cv2.approxPolyDP(c, 0.5, True).reshape(-1, 2)
        if len(ap) < 2:
            continue
        parts.append('M' + ' L'.join('%d,%d' % (int(p[0]), int(p[1])) for p in ap) + ' Z')
    if not parts:
        return "", 0, 0.0
    seg = ('<g fill="none" stroke="%s" stroke-width="%.2f" stroke-linejoin="round" '
           'stroke-linecap="round"><path d="%s"/></g>\n'
           % (color, stroke_width, ' '.join(parts)))
    return seg, len(parts), round(float(b.mean()) / max(frac0, 1e-9), 4)


def to_size(mask: np.ndarray, size_wh: tuple) -> np.ndarray:
    """bool mask 最近邻缩放到目标尺寸（用于把网格坐标的边界换算回原图坐标）。"""
    bw, bh = int(size_wh[0]), int(size_wh[1])
    m = mask.astype(np.uint8)
    if m.shape[1] != bw or m.shape[0] != bh:
        m = cv2.resize(m, (bw, bh), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)


def boundary_at(index: np.ndarray, size_wh: tuple, radius: int = 2) -> np.ndarray:
    """色块边界放大到目标尺寸并膨胀，用于剔除与 L2 重合的 CNN 线（防双线）。"""
    bw, bh = int(size_wh[0]), int(size_wh[1])
    b = boundary_mask(index).astype(np.uint8)
    if b.shape[1] != bw or b.shape[0] != bh:
        b = cv2.resize(b, (bw, bh), interpolation=cv2.INTER_NEAREST)
    if radius:
        b = cv2.dilate(b, np.ones((3, 3), np.uint8), iterations=radius)
    return b.astype(bool)


def build_line_svg(mask: np.ndarray, color: str = "#1a1a1a",
                   dilate: int = 0, stroke_width: float = 1.4) -> tuple:
    """线心 bool mask → (SVG path 片段, path 数)。纯矢量，无 potrace 依赖。

    用 fill="none" + stroke 描边，而非区域填充：线网密集区（眼睛、眼镜、发丝）
    相互连通时，区域填充会把整个闭合轮廓涂成实心黑块（整张脸糊掉）；描边只画
    轮廓线，闭合圈保持空心，线条干净。用 RETR_LIST 保留所有内外轮廓。

    注意：返回的坐标是 mask 像素坐标，调用方用 transform=scale 对齐底图 viewBox。
    stroke_width 以 mask 像素为单位，会随调用方的 scale 一起放大。
    """
    m = mask.astype(np.uint8) * 255
    if dilate:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=dilate)
    contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for c in contours:
        if len(c) < 2:
            continue
        ap = cv2.approxPolyDP(c, 0.3, True).reshape(-1, 2)
        if len(ap) < 2:
            continue
        parts.append('M' + ' L'.join('%d,%d' % (int(p[0]), int(p[1])) for p in ap) + ' Z')
    if not parts:
        return "", 0
    seg = ('<g fill="none" stroke="%s" stroke-width="%.2f" stroke-linejoin="round" '
           'stroke-linecap="round"><path d="%s"/></g>\n'
           % (color, stroke_width, ' '.join(parts)))
    return seg, len(parts)
