"""色块边界顺滑化后处理。

vtracer 的 spline 输出是大量**短三次贝塞尔**，逐段追随整数像素阶梯，
因此放大后边界呈锯齿/毛刺。qq3d 那类转换器的边界是「角点检测 + 长段
圆弧/二次贝塞尔」，所以顺滑。本模块把 vtracer 的路径重采样成折线 →
Douglas-Peucker 简化掉阶梯抖动 → 用 Catmull-Rom 重新拟合成连续长曲线，
同时保留真正的尖角（转角大于阈值的点断开平滑）。

只依赖 numpy / re，可直接作用于 core/own_tracer 产出的 SVG 字符串。
"""

from __future__ import annotations

import re
import numpy as np

_NUM = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")
_TOK = re.compile(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")
_PATH_RE = re.compile(r"<path\b([^>]*?)/>")


def _cubic(p0, p1, p2, p3, n=6):
    t = np.linspace(0.0, 1.0, n)[:, None]
    u = 1.0 - t
    return u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3


def _moving_average(P: np.ndarray, win: int) -> np.ndarray:
    """环形移动平均：只抹掉高频锯齿，保留低频轮廓位置。"""
    if win <= 1 or len(P) < win:
        return P
    k = max(1, int(win) // 2)
    # 偶数窗口也统一成 2k+1，否则 arange 与权重长度不一致会广播失败
    half = 2 * k + 1
    idx = (np.arange(-k, k + 1)[None, :] + np.arange(len(P))[:, None]) % len(P)
    w = np.ones(half, np.float32)
    w[k] = 2.0            # 中心加权，抑制整体收缩
    w /= w.sum()
    return (P[idx] * w[None, :, None]).sum(1)


def _rdp(points: np.ndarray, eps: float, bend_min_deg: float = 18.0,
         bend_wiggle: float = 1.8) -> np.ndarray:
    """迭代式 Douglas-Peucker + 缓弯保护。

    单纯加大 eps 会把阶梯合并成长直线（这是我们想要的），但也会把脸颊、
    衣褶这类**真实缓弯**一并削直，导致轮廓偏移、ΔE 上升。

    区分两者的判据是转角：锯齿是左右反复摆动（累计绝对转角 >> 净转角），
    长直段净转角≈0，而缓弯是单向渐变（累计绝对转角 ≈ 净转角）。
    因此对「净转角大且不摆」的段强制在中点继续细分，把曲率留下来。
    bend_min_deg<=0 时退化为标准 RDP。
    """
    n = len(points)
    if n < 3:
        return points
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    guard = bend_min_deg > 0
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = points[i], points[j]
        seg = b - a
        L2 = float(seg @ seg)
        mid = points[i + 1:j]
        if L2 < 1e-12:
            d = np.linalg.norm(mid - a, axis=1)
        else:
            t = np.clip(((mid - a) @ seg) / L2, 0.0, 1.0)
            proj = a + t[:, None] * seg
            d = np.linalg.norm(mid - proj, axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            idx = i + 1 + k
            keep[idx] = True
            stack.append((i, idx))
            stack.append((idx, j))
            continue
        if guard and j - i >= 4:
            v = np.diff(points[i:j + 1], axis=0)
            Lv = np.linalg.norm(v, axis=1)
            ok = Lv > 1e-9
            if ok.sum() >= 2:
                vv = v[ok]
                ang = np.arctan2(vv[:, 1], vv[:, 0])
                da = np.diff(ang)
                da = (da + np.pi) % (2 * np.pi) - np.pi
                net = abs(float(da.sum()))
                total = float(np.abs(da).sum())
                if net > np.radians(bend_min_deg) and total < net * bend_wiggle:
                    idx = (i + j) // 2
                    keep[idx] = True
                    stack.append((i, idx))
                    stack.append((idx, j))
    return points[keep]


def _corners(P: np.ndarray, deg: float = 55.0) -> np.ndarray:
    """标记折线上的尖角点：相邻线段夹角超过阈值。"""
    if len(P) < 3:
        return np.zeros(len(P), dtype=bool)
    v = np.diff(P, axis=0)
    L = np.linalg.norm(v, axis=1)
    good = L > 1e-9
    cos = np.ones(len(v) - 1)
    for k in range(len(v) - 1):
        if good[k] and good[k + 1]:
            cos[k] = float(v[k] @ v[k + 1]) / (L[k] * L[k + 1])
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    c = np.zeros(len(P), dtype=bool)
    c[1:-1] = ang > deg
    return c


def _fit_segments(P: np.ndarray, corner: np.ndarray, tension: float = 1.0) -> str:
    """把简化折线拟合成连续三次贝塞尔，尖角处断开成直线。"""
    n = len(P)
    if n < 2:
        return ""
    out = ["M%s,%s" % (_f(P[0][0]), _f(P[0][1]))]
    for i in range(n - 1):
        p0, p1 = P[i], P[i + 1]
        # 尖角端点：用直线，避免把衣领/发梢削圆
        if corner[i] or corner[i + 1]:
            out.append("L%s,%s" % (_f(p1[0]), _f(p1[1])))
            continue
        pm = P[i - 1] if i > 0 else p0 - (p1 - p0)
        pp = P[i + 2] if i + 2 < n else p1 + (p1 - p0)
        c1 = p0 + (p1 - pm) * (tension / 6.0)
        c2 = p1 - (pp - p0) * (tension / 6.0)
        out.append("C%s,%s %s,%s %s,%s" % (
            _f(c1[0]), _f(c1[1]), _f(c2[0]), _f(c2[1]), _f(p1[0]), _f(p1[1])))
    return " ".join(out) + " Z"


def _f(x: float) -> str:
    s = "%.2f" % float(x)
    return s.rstrip("0").rstrip(".") if "." in s else s


def _poly_area(P: np.ndarray) -> float:
    """Shoelace area of a closed polygon (abs)."""
    if len(P) < 3:
        return 0.0
    return 0.5 * abs(float(np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1])))


def _feature_width(subs: list) -> float:
    """Width proxy for a path's filled regions: median sqrt(area)/perimeter.

    细长条（发丝、细线）→ 值小；大色块 → 值大。这是 0.36.2 用 localWidth
    沿边界法向探测「较窄一侧色块物理宽度」的同源思路，但我们用闭合多边形自身
    的几何尺度估算，不依赖它的区域图，纯自研、零外部依赖。
    """
    scales = []
    for P in subs:
        if len(P) < 3:
            continue
        a = _poly_area(P)
        ring = np.vstack([P, P[0]])
        per = float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())
        if per < 1e-6 or a <= 0:
            continue
        scales.append((a ** 0.5) / per)
    return float(np.median(scales)) if scales else 0.0


def _smooth_one(attrs: str, W: float, H: float, opts: dict) -> str:
    """对单条 <path> 的 attrs 做边界重拟合，返回完整 <path .../> 字符串。

    与旧版 smooth_svg 内联 repl 的逐条逻辑完全一致；抽成纯函数是为了能
    安全地用 multiprocessing 在多条路径间并行（每条路径相互独立）。
    """
    dm = re.search(r'\bd="([^"]*)"', attrs)
    if not dm:
        return "<path%s/>" % attrs
    fill = re.search(r'fill="(#[0-9a-fA-F]{3,8})"', attrs)
    if not fill or fill.group(1).lower() == "none":
        return "<path%s/>" % attrs
    subs = _subpaths(dm.group(1))
    if W > 0 and H > 0 and subs:
        allpts = np.concatenate([p for p in subs if len(p)], axis=0)
        if len(allpts):
            lo, hi = allpts.min(0), allpts.max(0)
            if (hi[0] - lo[0]) >= 0.85 * W and (hi[1] - lo[1]) >= 0.85 * H:
                return "<path%s/>" % attrs
    eps = opts.get("eps", 0.6)
    corner_deg = opts.get("corner_deg", 120.0)
    tension = opts.get("tension", 1.0)
    min_points = opts.get("min_points", 5)
    smooth_win = opts.get("smooth_win", 5)
    bend_min_deg = opts.get("bend_min_deg", 18.0)
    iters = opts.get("iters", 2)
    # 微小区域（发丝尖端、细噪点）直接原样保留：不平滑、不简化，避免把锐利
    # 细节磨圆或拉变形（对应 0.36.2 的「小边保护」）。
    if opts.get("tiny_area", 0) > 0 and subs:
        tot = sum(_poly_area(p) for p in subs if len(p) >= 3)
        if tot < opts["tiny_area"]:
            return "<path%s/>" % attrs
    # 宽度自适应容差：以全图特征宽度的中位数为基准 —— 中位数区域保持原样、
    # 宽色块给大容差（更顺滑）、细线给小容差（更锐利）。这是 0.36.2 localWidth
    # 误差预算的同源思路，但用闭合多边形几何尺度估算，不复制其区域图代码。
    eps_local = eps
    if opts.get("adaptive"):
        fw = _feature_width(subs)
        ref = opts.get("ref_fw", 0.0)
        if fw > 0 and ref > 0:
            scale = fw / ref
        elif fw > 0:
            scale = fw / 6.0
        else:
            scale = 1.0
        eps_local = min(opts.get("eps_max", 2.0),
                        max(opts.get("eps_min", 0.12), eps * scale))
    new_parts = []
    for P in subs:
        if len(P) < min_points:
            new_parts.append("M%s,%s L%s,%s Z" % (
                _f(P[0][0]), _f(P[0][1]), _f(P[-1][0]), _f(P[-1][1])))
            continue
        if np.linalg.norm(P[0] - P[-1]) < 1e-6:
            P = P[:-1]
        cmask = _corners(P, corner_deg)
        S = P.copy()
        for _ in range(max(1, int(iters))):
            S = _moving_average(S, smooth_win)
        S = np.where(cmask[:, None], P, S)
        Q = _rdp(S, eps_local, bend_min_deg)
        if len(Q) < 3:
            Q = S
        c = _corners(Q, corner_deg)
        new_parts.append(_fit_segments(Q, c, tension))
    nd = " ".join(new_parts)
    return "<path%s/>" % attrs.replace('d="%s"' % dm.group(1), 'd="%s"' % nd)


def _smooth_chunk(payload):
    """多进程 worker：处理一块连续的 path，返回新 path 字符串列表（保序）。"""
    jobs, W, H, opts = payload
    return [_smooth_one(a, W, H, opts) for a in jobs]


def _subpaths(d: str) -> list[np.ndarray]:
    """从 path d 里抽取每个子路径的折线（取 C/L/M 的端点）。"""
    toks = _TOK.findall(d)
    out, cur, pts, i = [], np.array([0.0, 0.0]), [], 0
    cmd = None
    while i < len(toks):
        t = toks[i]
        if re.fullmatch(r"[A-Za-z]", t):
            cmd = t
            i += 1
            if cmd in ("Z", "z") and len(pts) >= 2:
                out.append(np.array(pts))
                pts = []
            continue
        if cmd is None:
            i += 1
            continue
        try:
            if cmd in ("M", "L"):
                if len(pts) >= 2 and cmd == "M":
                    out.append(np.array(pts)); pts = []
                p = np.array([float(toks[i]), float(toks[i + 1])]); i += 2
                cur = p; pts.append(p)
                if cmd == "M": cmd = "L"
            elif cmd == "C":
                p1 = np.array([float(toks[i]), float(toks[i + 1])])
                p2 = np.array([float(toks[i + 2]), float(toks[i + 3])])
                p3 = np.array([float(toks[i + 4]), float(toks[i + 5])]); i += 6
                # 密集采样，保留控制点里的曲率信息（只取端点会把曲线拉直）
                pts.extend(list(_cubic(cur, p1, p2, p3, 6)[1:]))
                cur = p3
            elif cmd == "Q":
                p1 = np.array([float(toks[i]), float(toks[i + 1])])
                p2 = np.array([float(toks[i + 2]), float(toks[i + 3])]); i += 4
                t = np.linspace(0.2, 1.0, 5)[:, None]; u = 1 - t
                pts.extend(list(u * u * cur + 2 * u * t * p1 + t * t * p2))
                cur = p2
            elif cmd == "A":
                p = np.array([float(toks[i + 5]), float(toks[i + 6])]); i += 7
                cur = p; pts.append(p)
            elif cmd in ("H", "V"):
                v = float(toks[i]); i += 1
                cur = np.array([v, cur[1]]) if cmd == "H" else np.array([cur[0], v])
                pts.append(cur)
            else:
                i += 1
        except (ValueError, IndexError):
            break
    if len(pts) >= 2:
        out.append(np.array(pts))
    return out


def smooth_svg(svg: str, eps: float = 0.6, corner_deg: float = 120.0,
               tension: float = 1.0, min_points: int = 5,
               smooth_win: int = 5, bend_min_deg: float = 18.0,
               iters: int = 2, parallel: bool = True,
               max_workers: int | None = None,
               adaptive: bool = True, eps_min: float = 0.12,
               eps_max: float = 6.0, tiny_area: float = 9.0) -> str:
    """把 SVG 里每条 fill path 的边界重拟合成顺滑曲线，返回新 SVG。

    平滑策略（实测参考 qq3d 目标图后重写）：
      1) 多轮局部移动平均：把逐像素的阶梯抖动松弛成光滑中心线。每轮只在
         「低频轮廓 / 真实曲线」上作用，高频锯齿被抹掉，因此只需很轻的后续
         简化，不会像大 eps 的 RDP 那样把脸颊 / 衣褶一并削直。
      2) 尖角钉回：真实尖角（夹角 > corner_deg）顶点在平滑后钉回原位，避免
         被磨圆；拟合时这些点仍走直线，保留锐利转角。
      3) 轻度 RDP：eps 很小，只丢掉近共线的残余节点以压缩点数，不做大尺度
         形状简化 —— 这是与旧版（一次弱平均 + 大 eps RDP）最关键的区别。
      4) Catmull-Rom 重拟合，得到流畅长曲线。

    iters: 移动平均轮数（越大越顺滑，但收敛后形状不再变）。
    smooth_win: 局部移动平均窗口（越大越顺滑）。
    eps: 后处理 RDP 容差（保持很小，仅去冗余节点）。
    bend_min_deg: 缓弯保护角度，>0 时保留真实曲线不被削直；设 0 关闭。
    adaptive: 宽度自适应容差（默认开）。宽色块给大容差更顺滑、细线给小容差
        更锐利，源自 0.36.2 的 localWidth 误差预算思路（用闭合多边形几何尺度
        估算，非复制其代码）。tiny_area>0 时微小区域原样保留（不平滑）。
    """
    # Canvas extent, so full-canvas base layers can be left untouched.  Those
    # paths are (near-)rectangular backdrops: smoothing them has no visual
    # benefit, and the ring moving-average can pull a corner into a long
    # diagonal, leaving an uncovered wedge that shows the background rect
    # (seen as a solid black triangle on framed / bordered artwork).
    W = H = 0.0
    root = re.search(r"<svg\b[^>]*>", svg)
    if root:
        vb = re.search(r'viewBox="([^"]*)"', root.group(0))
        if vb:
            try:
                v = [float(x) for x in vb.group(1).replace(",", " ").split()]
                W, H = v[2], v[3]
            except (ValueError, IndexError):
                pass
        if W <= 0 or H <= 0:
            wm = re.search(r'\bwidth="([\d.]+)"', root.group(0))
            hm = re.search(r'\bheight="([\d.]+)"', root.group(0))
            W = float(wm.group(1)) if wm else 0.0
            H = float(hm.group(1)) if hm else 0.0

    opts = dict(eps=eps, corner_deg=corner_deg, tension=tension,
                min_points=min_points, smooth_win=smooth_win,
                bend_min_deg=bend_min_deg, iters=iters,
                adaptive=adaptive, eps_min=eps_min, eps_max=eps_max,
                tiny_area=tiny_area, ref_fw=0.0)

    matches = list(_PATH_RE.finditer(svg))
    if not matches:
        return svg

    # 宽度自适应的基准：以全图各路径特征宽度的中位数为 1.0，避免绝对尺度偏差
    # 把所有区域一刀切地推到最小/最大容差。仅做一次轻量预扫描。
    if adaptive:
        fws = []
        for m in matches:
            dm = re.search(r'\bd="([^"]*)"', m.group(1))
            if not dm:
                continue
            fws.append(_feature_width(_subpaths(dm.group(1))))
        fws = [f for f in fws if f > 0]
        if fws:
            opts["ref_fw"] = float(np.median(fws))

    # 单线程 / 小图：与旧版逐字节一致
    if not parallel or len(matches) < 256:
        return _PATH_RE.sub(lambda m: _smooth_one(m.group(1), W, H, opts), svg)

    # 多核并行：每条 path 相互独立，切成连续块分发到各进程，按全局顺序回填。
    # 非 path 内容（defs / style / 注释）原样保留在切片里，确保输出逐字节不变。
    import os
    import multiprocessing as mp
    try:
        n = len(matches)
        cw = max(2, min(max_workers or (os.cpu_count() or 4), n))
        nw = min(cw, 32)
        chunk = (n + nw - 1) // nw
        chunks = [matches[i * chunk:(i + 1) * chunk] for i in range(nw)]
        chunks = [c for c in chunks if c]
        nw = len(chunks)
        jobs_chunks = [([m.group(1) for m in c], W, H, opts) for c in chunks]
        # chunk 划分不变，只按可用内存限制同时运行的进程数 → 输出逐字节一致。
        import mem_budget as _mb
        with mp.Pool(_mb.workers(nw)) as pool:
            out_lists = pool.map(_smooth_chunk, jobs_chunks)
        new_strings = [s for ol in out_lists for s in ol]
        out = []
        last = 0
        for m, ns in zip(matches, new_strings):
            out.append(svg[last:m.start()])
            out.append(ns)
            last = m.end()
        out.append(svg[last:])
        return "".join(out)
    except Exception:
        # 受限环境多核失败 → 退回单线程，保证可用
        return _PATH_RE.sub(lambda m: _smooth_one(m.group(1), W, H, opts), svg)
