# -*- coding: utf-8 -*-
"""底图 path 压缩（v7.17.10 起，v7.17.11 扩到相对坐标）——自研，无外部依赖。

v7.17.9 量过底图体积构成：flat 6.72MB 里 `d` 4.84MB（72%）、每个 path 的
重复属性约 1.0MB。本模块针对这两块做无损/亚像素级压缩：

1. **公共属性提升**：`_add_seam_strokes` 会给*每个* path 各写一份
   `stroke-width="0.8" stroke-linejoin="round" stroke-linecap="round"`。
   13000 个 path 就是 13000 份重复。提到根 `<svg>` 上由继承生效，渲染
   逐像素一致（实测 diff max=0），体积 −12~14%。

2. **d 路径简化**：vtracer 输出的 `d` 只有 M/C/Z，但实测 **38.8%** 的 C 段
   控制点完全共线——那是直线被写成了三次贝塞尔（如
   `C1056 0 2112 0 3200 0`）。判定共线后转 `L`，水平的转 `H`、垂直的转
   `V`，相邻共线的连续直线再合并成一段。
   容差 eps 以网格单位计（3200 网格 1 单位 ≈ 0.225 CSS px），默认 0.30
   即 0.068 CSS px——比 v7.17.9 的坐标整数化误差还小。

3. **折 translate + 相对坐标**（v7.17.11）：vtracer 还给每个 path 写一份
   `transform="translate(618,1457)"`，`d` 是相对该原点的局部值。整数位移
   折进首个 `M` 点即可删掉属性（两端都是整数才折，1429/13010 带小数的保留）。
   折完再把绝对坐标全部改相对：整数化后 delta 仍是精确整数，几何零误差，
   而 delta 的数字位宽只有绝对坐标的一半左右。

实测（v7.17.10 → v7.17.11）：体积再 −25~28%，flat 3.95→2.86MB。
flat 逐像素 0 差，其余图最大差 9/255、仅 0.07% 像素——那是渲染器"有/无
transform 时曲线细分点不同"的抗锯齿差，不是几何变化。

刻意**没有**做的两件事，都量过不划算：
  * 同色 path 合并成一个 `<g>`（按色重排）：只多省 6 个百分点，但改变了
    0.8 宽同色描边的覆盖顺序，导致 0.49% 的边缘像素换色（>32/255）。
  * 同色 path 合并成单条多子路径的 `d`：省下的 `<path>` 标签与 transform
    抵不过绝对坐标位数变长的开销，净亏。（**相对坐标把这个前提翻掉了**，
    但合并仍会破坏 z 序，故仍不做。）
  * CSS class 替掉 fill/stroke：还能再 −5~6%，但会和本步的 translate 折叠
    互斥（class 会吞掉 transform），且引入 `<style>` 依赖——Figma 等工具
    对 SVG 内联 CSS 支持不稳，收益不值得这个风险，暂不做。
"""
from __future__ import annotations

import math
import re

_TOK = re.compile(r"[A-Za-z]|-?(?:\d+\.?\d*|\.\d+)")
_ROOT_RE = re.compile(r"<svg\b([^>]*)>")
_PATH_RE = re.compile(r"<path\b([^>]*?)/>")
_D_RE = re.compile(r'\bd="([^"]*)"')
# vtracer 会给每个 path 写一份局部原点位移，坐标是相对该原点的局部值
_TR_RE = re.compile(r'\s*transform="translate\(\s*(-?[\d.]+)\s*[ ,]\s*(-?[\d.]+)\s*\)"')

# v7.17.11 两个开关，改 False 即回 v7.17.10 行为
RELATIVE = True              # d 绝对坐标 → 相对坐标（整数 delta 精确无损）
FOLD_TRANSLATE = True        # 整数 translate 折进首个 M 点，删掉 transform 属性

# 会被提到根节点上的公共渲染属性（同色描边遮缝那一套）
_SHARED_ATTRS = ("stroke-width", "stroke-linejoin", "stroke-linecap")

_DEFAULT_EPS = 0.30          # 网格单位；3200 网格下 1 单位 ≈ 0.225 CSS px
_EPS_GRID = 3200.0


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return "%d" % round(v)
    s = ("%.3f" % v).rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def compact_d(d: str, eps: float = _DEFAULT_EPS, snap_int: bool = True) -> str:
    """把退化 C 段转成 L/H/V 并合并共线直线。

    ``snap_int`` 与 v7.17.9 的 ``compact_coords(decimals=0)`` 等价：坐标先
    四舍五入到整数网格单位，再做共线判定（整数下叉积是精确的）。

    只支持绝对命令 M/L/H/V/C/Z；遇到其他命令（相对命令 a/c/l 等）原样返回，
    保证不破坏非 vtracer 来源的 SVG。
    """
    tk = _TOK.findall(d)
    n = len(tk)

    def R(t):
        v = float(t)
        return float(round(v)) if snap_int else v

    out: list[tuple] = []
    cur = (0.0, 0.0)
    sub = (0.0, 0.0)
    prev = None                     # 再前一个绘图点，用于合并连续直线
    last = None

    def push_line(x: float, y: float) -> None:
        nonlocal prev, last
        if out and out[-1][0] == "L" and prev is not None and last is not None:
            bx, by = prev
            px, py = last
            d1x, d1y = px - bx, py - by
            d2x, d2y = x - px, y - py
            l1 = math.hypot(d1x, d1y)
            l2 = math.hypot(d2x, d2y)
            if l1 > 1e-9 and l2 > 1e-9:
                cross = abs(d1x * d2y - d1y * d2x) / (l1 * l2)
                dot = (d1x * d2x + d1y * d2y) / (l1 * l2)
                if cross < 1e-6 and dot > 0:          # 同向共线 → 延长上一段
                    out[-1] = ("L", (x, y))
                    prev, last = last, (x, y)
                    return
        out.append(("L", (x, y)))
        prev, last = last, (x, y)

    i = 0
    while i < n:
        c = tk[i]
        if c == "M":
            i += 1
            first = True
            while i + 1 < n and not tk[i].isalpha():
                x, y = R(tk[i]), R(tk[i + 1])
                i += 2
                if first:
                    cur = sub = (x, y)
                    out.append(("M", (x, y)))
                    prev, last = None, (x, y)
                    first = False
                else:
                    push_line(x, y)
                    cur = (x, y)
            if first:                              # M 后面没数字：异常，放弃
                return d
        elif c == "Z":
            i += 1
            cur = sub
            out.append(("Z", ()))
            prev, last = None, sub
        elif c == "L":
            i += 1
            while i + 1 < n and not tk[i].isalpha():
                x, y = R(tk[i]), R(tk[i + 1])
                i += 2
                push_line(x, y)
                cur = (x, y)
        elif c == "H":
            i += 1
            while i < n and not tk[i].isalpha():
                x = R(tk[i])
                i += 1
                push_line(x, cur[1])
                cur = (x, cur[1])
        elif c == "V":
            i += 1
            while i < n and not tk[i].isalpha():
                y = R(tk[i])
                i += 1
                push_line(cur[0], y)
                cur = (cur[0], y)
        elif c == "C":
            i += 1
            while i + 5 < n and not tk[i].isalpha():
                x1, y1 = R(tk[i]), R(tk[i + 1])
                x2, y2 = R(tk[i + 2]), R(tk[i + 3])
                x3, y3 = R(tk[i + 4]), R(tk[i + 5])
                i += 6
                x0, y0 = cur
                dx, dy = x3 - x0, y3 - y0
                seg = math.hypot(dx, dy)
                straight = False
                if seg < 1e-9:
                    straight = (abs(x1 - x0) < eps and abs(y1 - y0) < eps
                                and abs(x2 - x0) < eps and abs(y2 - y0) < eps)
                else:
                    def off(qx, qy):
                        return abs((qx - x0) * dy - (qy - y0) * dx) / seg

                    def par(qx, qy):
                        return ((qx - x0) * dx + (qy - y0) * dy) / (seg * seg)

                    straight = (off(x1, y1) < eps and off(x2, y2) < eps
                                and -0.02 <= par(x1, y1) <= 1.02
                                and -0.02 <= par(x2, y2) <= 1.02)
                if straight:
                    push_line(x3, y3)
                else:
                    out.append(("C", (x1, y1, x2, y2, x3, y3)))
                    prev, last = last, (x3, y3)
                cur = (x3, y3)
            if i < n and not tk[i].isalpha() and i + 5 >= n:
                return d
        else:
            return d                               # 相对命令等：不处理

    parts = []
    cx = cy = sx = sy = 0.0
    for cmd, a in out:
        if cmd == "Z":
            parts.append("Z")
            cx, cy = sx, sy
        elif cmd == "M":
            cx, cy = a
            sx, sy = a
            parts.append("M%s %s" % (_fmt(cx), _fmt(cy)))
        elif cmd == "L":
            nx, ny = a
            if ny == cy:
                parts.append("H%s" % _fmt(nx))
            elif nx == cx:
                parts.append("V%s" % _fmt(ny))
            else:
                parts.append("L%s %s" % (_fmt(nx), _fmt(ny)))
            cx, cy = nx, ny
        elif cmd == "C":
            parts.append("C" + " ".join(_fmt(v) for v in a))
            cx, cy = a[-2], a[-1]
    return "".join(parts)


def _join_nums(vals) -> str:
    """最短分隔：负数可以直接贴在前一个数字后面（`5-3` 即 5,-3），省一个空格。"""
    s = ""
    for v in vals:
        t = _fmt(v)
        if s and not t.startswith("-"):
            s += " "
        s += t
    return s


def to_relative(d: str, tx: float = 0.0, ty: float = 0.0) -> str:
    """绝对坐标 d → 相对坐标 d，可顺带把 (tx, ty) 折进首个 M 点。

    只处理 ``compact_d`` 的产物（M/L/H/V/C/Z 绝对整数）。因为整数化后
    delta 仍是精确整数，这一步**几何上零误差**：相邻贝塞尔控制点只差几十个
    网格单位，写成 ``c-12 5-30 8 41 3`` 比 ``C1234 5678 ...`` 省一半以上。

    注意 cx/cy 始终用**未平移**的坐标记账，只有输出首个绝对 M 时才加
    (tx, ty)——这样后续 ``m`` 的 delta 天然与平移无关。
    遇到非整数或未知命令，原样返回（宁可不变，不做错）。
    """
    tk = _TOK.findall(d)
    n = len(tk)
    nums = [t for t in tk if not t.isalpha()]
    if any(abs(float(t) - round(float(t))) > 1e-9 for t in nums):
        return d

    out: list[str] = []
    cx = cy = sx = sy = 0.0
    i = 0
    while i < n:
        c = tk[i]
        if c == "M":
            i += 1
            first = True
            while i + 1 < n and not tk[i].isalpha():
                x, y = float(tk[i]), float(tk[i + 1])
                i += 2
                if first:
                    if not out:                      # 全串第一个绝对点
                        out.append("M" + _join_nums([x + tx, y + ty]))
                        cx, cy = x, y                # 记账用未平移值
                        sx, sy = cx, cy
                    else:                            # 后续子路径：相对跳转
                        out.append("m" + _join_nums([x - cx, y - cy]))
                        cx, cy = x, y
                        sx, sy = cx, cy
                    first = False
                else:                                # M 后的多余点对＝隐式 lineto
                    out.append("l" + _join_nums([x - cx, y - cy]))
                    cx, cy = x, y
            if first:
                return d
        elif c == "L":
            i += 1
            while i + 1 < n and not tk[i].isalpha():
                x, y = float(tk[i]), float(tk[i + 1])
                i += 2
                dx, dy = x - cx, y - cy
                if dy == 0:
                    out.append("h" + _join_nums([dx]))
                elif dx == 0:
                    out.append("v" + _join_nums([dy]))
                else:
                    out.append("l" + _join_nums([dx, dy]))
                cx, cy = x, y
        elif c == "H":
            i += 1
            while i < n and not tk[i].isalpha():
                x = float(tk[i])
                i += 1
                out.append("h" + _join_nums([x - cx]))
                cx = x
        elif c == "V":
            i += 1
            while i < n and not tk[i].isalpha():
                y = float(tk[i])
                i += 1
                out.append("v" + _join_nums([y - cy]))
                cy = y
        elif c == "C":
            i += 1
            while i + 5 < n and not tk[i].isalpha():
                x1, y1 = float(tk[i]), float(tk[i + 1])
                x2, y2 = float(tk[i + 2]), float(tk[i + 3])
                x3, y3 = float(tk[i + 4]), float(tk[i + 5])
                i += 6
                out.append("c" + _join_nums([x1 - cx, y1 - cy, x2 - cx, y2 - cy,
                                             x3 - cx, y3 - cy]))
                cx, cy = x3, y3
        elif c == "Z":
            i += 1
            out.append("Z")
            cx, cy = sx, sy
        else:
            return d
    return "".join(out)


def _split_base(svg: str):
    """切出 (根头, 底图 path 区, 其余)。

    底图 = 根 `<svg>` 之后、**第一个 `<g>` 之前**的全部 path。vtracer 把底图
    直接写在根下，线层/边界层才用 `<g>` 包。

    注意：flatfill 刚写出的底图 SVG 里**一个 `<g>` 都没有**（边界层/线层由
    `core._apply_linenet` 事后拼接），此时整个 path 区就是底图，``rest`` 为空。
    早先版本在这里直接 return 原串，导致端到端压了个寂寞（体积只降 0%）——
    只在"已拼好线层"的成品上测才会误判成功。
    """
    m = re.search(r"<g\b", svg)
    cut = m.start() if m else len(svg)
    head, rest = svg[:cut], svg[cut:]
    i = head.find("<path")
    if i < 0:
        return head, "", rest
    return head[:i], head[i:], rest


def optimize(svg: str, eps: float | None = None, grid_w: float | None = None,
             lift: bool = True, dpath: bool = True, snap_int: bool = True) -> str:
    """压缩底图 path：提升公共属性 + 简化 d。

    ``eps`` 缺省时按 viewBox 宽度折算，保证不同网格尺寸下视觉偏差一致
    （``_DEFAULT_EPS`` @3200 网格）。``snap_int=False`` 保留原始坐标精度，
    只做几何简化（差异更小、体积略大）。
    """
    pre, base, rest = _split_base(svg)
    if not base:
        return svg

    if eps is None:
        vbw = grid_w
        if vbw is None:
            m = re.search(r'viewBox="0\s+0\s+([\d.]+)', svg)
            vbw = float(m.group(1)) if m else _EPS_GRID
        eps = max(0.20, _DEFAULT_EPS * float(vbw) / _EPS_GRID)

    # 1) 公共属性提升
    if lift:
        # v7.17.15：由「全部同值才提升」改为「按众数提升」。
        # 补缝描边改成按色块大小分三档后，少数 path 的 stroke-width 与多数不同；
        # 旧逻辑会整体放弃提升，1.5 万条 path 各背一份属性（实测体积 ×3）。
        # 现在只提升与多数同值的那些，少数保留自己的属性，体积回到原水平。
        vals = {a: {} for a in _SHARED_ATTRS}
        npath = 0
        for m in _PATH_RE.finditer(base):
            npath += 1
            for a in _SHARED_ATTRS:
                mm = re.search(r'\b%s="([^"]*)"' % a, m.group(1))
                if mm:
                    v = mm.group(1)
                    vals[a][v] = vals[a].get(v, 0) + 1
        shared = {a: max(c, key=c.get) for a, c in vals.items()
                  if c and max(c.values()) > 1}
        if shared and npath:
            def strip(mm):
                attrs = mm.group(1)
                for a, v in shared.items():
                    attrs = re.sub(r'\s+%s="%s"' % (a, re.escape(v)), "", attrs)
                return "<path%s/>" % attrs

            base = _PATH_RE.sub(strip, base)
            # 只加根上没有的属性，避免重复
            root_attrs = "".join(
                ' %s="%s"' % (a, v) for a, v in shared.items()
                if not re.search(r'\b%s=' % a, pre))
            if root_attrs:
                pre = _ROOT_RE.sub(lambda mm: "<svg%s%s>" % (mm.group(1), root_attrs),
                                   pre, count=1)

    # 2) d 简化（转 L/H/V + 合并共线）→ 折 translate → 转相对坐标
    if dpath:
        def fix(mm):
            attrs = mm.group(1)
            dm = _D_RE.search(attrs)
            if not dm or not dm.group(1):
                return mm.group(0)
            d0 = dm.group(1)
            nd = compact_d(d0, eps, snap_int=snap_int)

            # 折掉整数位移：坐标本来就是相对该原点的局部值，折进首个 M 即可
            tx = ty = 0.0
            tr = None
            if FOLD_TRANSLATE:
                tm = _TR_RE.search(attrs)
                if tm:
                    fx, fy = float(tm.group(1)), float(tm.group(2))
                    if fx.is_integer() and fy.is_integer():
                        tx, ty, tr = fx, fy, tm.group(0)

            if RELATIVE:
                nd = to_relative(nd, tx, ty)
            elif tx or ty:
                m0 = re.match(r"M(-?[\d.]+) (-?[\d.]+)", nd)
                if m0:
                    nd = "M%s %s%s" % (_fmt(float(m0.group(1)) + tx),
                                       _fmt(float(m0.group(2)) + ty), nd[m0.end():])
                else:
                    tr = None

            if nd == d0 and not tr:
                return mm.group(0)
            new = attrs[:dm.start()] + 'd="%s"' % nd + attrs[dm.end():]
            if tr:
                new = new.replace(tr, "", 1)
            return "<path%s/>" % new

        base = _PATH_RE.sub(fix, base)

    return pre + base + rest
