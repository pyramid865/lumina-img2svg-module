# -*- coding: utf-8 -*-
"""大色块平涂化（flatmerge.py）— v7.19.0。

**要解决的问题**：用户在动漫插画上抱怨「颜色太精细、碎块太多，不如嘻嘻宝
那种同色大色块干净」。实测 463 个嘻嘻宝样本的统计中位值：

| 指标 | 嘻嘻宝中位 | 我们 fusion 170 色 | 差距 |
|---|---|---|---|
| 色数 | 77 | 170 | 2.2× |
| 路径密度 | 408 条/Mpx（复杂图 2300 条/Mpx） | ~7000 条/Mpx | 3~17× |
| 单个色块平均面积 | ~2500 px² | ~140 px² | 18× |

结论：不是「色数」的问题，是**碎块**的问题——量化把同一片皮肤的明暗台阶
切成几十条细带，每条带又各自成为一条 path，密到看着发毛。所以这里做两件事：

1. **阶梯合并**：相邻两簇颜色差 <= ``de_lim`` 就并成一簇（同一材质的明暗
   台阶），链式合并时用簇级均色二次校验，防止两端差很远的色被一路
   「接力」连起来。
2. **吞碎块**：按面积阈值逐级抬高，把小连通块并进「共享边界最长且颜色
   最近」的邻居，直到整图连通块数落到目标密度。这一步直接决定 path 条数。

取色规则（v7.19.0 修正）：合并后的颜色取**成员里面积最大的那块的原色**，
不取面积加权均值。均值会把「脸」和「脸影」平均成中间色，整张脸发灰发白
（v7.18 原型实测：合并 9ΔE 时皮肤发白、头发糊成一团）；取最大块原色则
永远落在原图真实存在的颜色上，不会凭空造出中间色。
"""
from __future__ import annotations

import numpy as np

import color

# 平涂强度 → (阶梯合并 ΔE 上限, 目标连通块密度 每百万像素)
# 参照嘻嘻宝样本：中位 408 块/Mpx，复杂插画 2300 块/Mpx。
FLAT_LEVELS = {
    "off":    (0.0, 0.0),     # 不处理
    "light":  (4.0, 40.0),    # 只并最接近的台阶 + 清 40px² 以下麻点
    "medium": (6.0, 70.0),    # 默认档：170 色 → 约 60~80 色
    "strong": (9.0, 110.0),   # 更平，色阶更少（细节会少一点）
}
FLAT_ABSORB_DE = 22.0      # 碎块并进邻居时允许的最大色差（OKLab ΔE×100）

# v7.20.0 自动档位：程序自己选，用户不用管。
# 判据只用一个稳健信号——「局部纹理强度」。实测四张样本的 9×9 局部标准差：
#   大色块插画 ~21（动漫） / 写实照 ~19~24（金克丝、照片）
#   细节爆炸图 ~42（五条悟：满屏头发丝+高光）
# 只有细节强度远超常规模板（>34）时才降到 light，避免把密集细线并糊；
# 其余一律 medium——它对动漫/插画/一般照片都是实测的最佳平衡点。
AUTO_TEX_STRONG = 34.0     # 局部纹理标准差超过这个值 → 自动降为 light（保护密集细线）


def auto_level(rgb) -> str:
    """根据图像内容自动选平涂档位，返回 medium / light。

    只在「纹理极度密集」（如满屏发丝高光的照片）时降级到 light，
    其余情况都用 medium（大色块平涂，实测最佳平衡点）。不返回 off/strong：
      - off 会让皮肤重新出现一条条细色带，正是用户要解决的问题；
      - strong 会糊掉低对比花纹，实测腿上图案消失。
    """
    import cv2
    from PIL import Image
    if isinstance(rgb, Image.Image):
        rgb = np.asarray(rgb.convert("RGB"))
    a = np.asarray(rgb, np.uint8)
    # 缩到 400 宽统计即可，快且与全分辨率结论一致
    h, w = a.shape[:2]
    if w > 400:
        a = cv2.resize(a, (400, max(1, int(h * 400 / w))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(g, (9, 9))
    var = cv2.blur(g * g, (9, 9)) - mean * mean
    tex = float(np.sqrt(np.maximum(var, 0)).mean())
    return "light" if tex > AUTO_TEX_STRONG else "medium"



def _adjacency(idx: np.ndarray, k: int):
    """返回相邻标签对 (pairs[N,2]) 与共享边界像素数 (cnt[N])。"""
    a = idx[:, :-1].ravel()
    b = idx[:, 1:].ravel()
    c = idx[:-1, :].ravel()
    d = idx[1:, :].ravel()
    lo = np.concatenate([np.minimum(a, b), np.minimum(c, d)])
    hi = np.concatenate([np.maximum(a, b), np.maximum(c, d)])
    m = lo != hi
    key = lo[m].astype(np.int64) * k + hi[m]
    u, cnt = np.unique(key, return_counts=True)
    return np.stack([u // k, u % k], 1).astype(np.int32), cnt


class _UF:
    __slots__ = ("p",)

    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int32)

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
        return ra


def texture_map(rgb: "np.ndarray | PIL.Image") -> np.ndarray:
    """细纹理掩膜（True = 细网格/格纹/发丝这类必须保住的细节）。

    只抓「小尺度的明暗反复横跳」——网袜格纹、蕾丝、细发丝都是这种；
    大面积的平滑渐变（皮肤、天空、墙面）不会被抓到。用梯度符号翻转
    再形态学开运算，去掉 JPEG 噪点这类孤立单点。

    返回 bool 图，尺寸与输入一致。
    """
    import cv2
    from PIL import Image
    if isinstance(rgb, Image.Image):
        rgb = np.asarray(rgb.convert("RGB"))
    g = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = np.diff(g, axis=1, prepend=g[:, :1])
    gy = np.diff(g, axis=0, prepend=g[:1, :])
    fx = (gx[:, 1:] * gx[:, :-1] < 0) & (np.abs(gx[:, 1:]) > 6)
    fy = (gy[1:, :] * gy[:-1, :] < 0) & (np.abs(gy[1:, :]) > 6)
    flip = np.zeros(g.shape, np.uint8)
    flip[:, 1:] = np.maximum(flip[:, 1:], fx.astype(np.uint8))
    flip[:, :-1] = np.maximum(flip[:, :-1], fx.astype(np.uint8))
    flip[1:, :] = np.maximum(flip[1:, :], fy.astype(np.uint8))
    flip[:-1, :] = np.maximum(flip[:-1, :], fy.astype(np.uint8))
    flip = cv2.morphologyEx(flip, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 膨胀一点，把线条周围的抗锯齿像素也算进保护区
    return cv2.dilate(flip, np.ones((5, 5), np.uint8)).astype(bool)


def merge_bands(idx: np.ndarray, pal: np.ndarray, de_lim: float,
                weights: np.ndarray | None = None, rounds: int = 3,
                protect: "np.ndarray | None" = None):
    """相邻近色簇成链合并。返回 (idx, pal, 合并次数)。

    ``protect`` 是纹理掩膜（True = 必须保住的细纹理像素）。落在保护区里的
    色簇不参与近色合并——网袜格纹、蕾丝、发丝靠它保住；平滑区照常并色阶。
    """
    if de_lim <= 0:
        return idx, pal, 0
    w = np.ones(3, np.float32) if weights is None else np.asarray(weights, np.float32)
    cur = idx
    merged_total = 0
    # 每个色簇的纹理像素占比（只算一次，后续轮次沿用同一份保护集合）
    locked = np.zeros(int(pal.max()) + 1 if False else max(int(idx.max()) + 1, len(pal)), bool)
    if protect is not None and protect.shape == idx.shape:
        k0 = int(idx.max()) + 1
        tot = np.bincount(idx.ravel(), minlength=k0).astype(np.float64)
        tex = np.bincount(idx.ravel()[protect.ravel()], minlength=k0).astype(np.float64)
        ratio = tex / np.maximum(tot, 1.0)
        # 该簇有 8% 以上像素落在细纹理里就锁住不合并
        locked[:k0] = ratio > 0.08
    for _ in range(max(1, int(rounds))):
        k = int(cur.max()) + 1
        pal = pal[:k] if len(pal) >= k else pal
        lab = color.srgb_to_oklab(pal.astype(np.float32) / 255.0)
        area = np.bincount(cur.ravel(), minlength=k).astype(np.float64)
        pairs, _cnt = _adjacency(cur, k)
        if len(pairs) == 0:
            break
        de = np.sqrt(((lab[pairs[:, 0]] - lab[pairs[:, 1]]) ** 2 * w).sum(1)) * 100.0
        uf = _UF(k)
        acc_lab = lab.copy()
        acc_area = area.copy()
        # 被锁的簇其所在并查集根也不许吸收别人
        is_locked = locked[:k].copy()
        changed = 0
        for i in np.argsort(de):
            if de[i] > de_lim:
                break
            ra, rb = uf.find(int(pairs[i, 0])), uf.find(int(pairs[i, 1]))
            if ra == rb:
                continue
            # 纹理保护：任一侧在保护区，跳过这一对
            if is_locked[ra] or is_locked[rb]:
                continue
            wa, wb = acc_area[ra], acc_area[rb]
            if wa + wb <= 0:
                continue
            # 二次校验：两簇当前均色再判一次，防链式漂移
            if np.sqrt(((acc_lab[ra] - acc_lab[rb]) ** 2 * w).sum()) * 100.0 > de_lim:
                continue
            root = uf.union(ra, rb)
            acc_lab[root] = (acc_lab[ra] * wa + acc_lab[rb] * wb) / (wa + wb)
            acc_area[root] = wa + wb
            is_locked[root] = False
            changed += 1
        if not changed:
            break
        merged_total += changed
        roots = np.array([uf.find(i) for i in range(k)], np.int32)
        _uniq, inv = np.unique(roots, return_inverse=True)
        # 颜色取「成员里面积最大的原簇色」
        best = np.zeros(int(inv.max()) + 1, np.int64)
        seen = np.zeros(int(inv.max()) + 1, bool)
        for old in np.argsort(-area):
            g = inv[old]
            if not seen[g]:
                seen[g] = True
                best[g] = old
        cur = inv.astype(np.int32)[cur]
        pal = pal[best]
        # 合并后重新映射锁定状态到新的簇号
        new_locked = np.zeros(len(pal), bool)
        for old in range(k):
            if is_locked[old]:
                new_locked[inv[old]] = True
        locked = new_locked
    return cur, pal, merged_total


def _label_regions(idx: np.ndarray) -> np.ndarray:
    """同色 + 4 邻接的连通块编号（0 = 背景/空）。用 skimage 一次搞定。"""
    try:
        from skimage import measure
        reg = measure.label(idx + 1, connectivity=1)
        return reg.astype(np.int32)
    except Exception:  # noqa: BLE001
        from scipy import ndimage as ndi
        struct = ndi.generate_binary_structure(2, 1)
        out = np.zeros(idx.shape, np.int32)
        n0 = 0
        for c in np.unique(idx):
            m = idx == c
            cc, n = ndi.label(m, structure=struct)
            out[m] = cc[m] + n0
            n0 += n
        return out


def _absorb_pass(idx: np.ndarray, pal: np.ndarray, min_area: float,
                 absorb_de: float, weights: np.ndarray | None):
    """一轮：把面积 < min_area 的连通块并进最佳邻居。返回 (idx, pal, 吞块数, 剩余块数)。"""
    w = np.ones(3, np.float32) if weights is None else np.asarray(weights, np.float32)
    k = int(idx.max()) + 1
    reg = _label_regions(idx)
    nreg = int(reg.max())
    if nreg == 0:
        return idx, pal, 0, 0
    area = np.bincount(reg.ravel(), minlength=nreg + 1).astype(np.float64)
    area[0] = 0.0
    # 每个连通块的颜色（块内同色，取第一个像素的标签即可）
    reg_color = np.zeros(nreg + 1, np.int32)
    flat = reg.ravel()
    reg_color[flat[::997]] = idx.ravel()[::997]      # 采样兜底
    for c in range(k):
        m = idx == c
        if m.any():
            ids = np.unique(reg[m])
            reg_color[ids] = c
    lab = color.srgb_to_oklab(pal.astype(np.float32) / 255.0)

    # 块邻接（共享边界像素数）
    H, W = idx.shape
    f = flat
    pa = np.concatenate([f[:-1], f[:-W]])
    pb = np.concatenate([f[1:], f[W:]])
    m = (pa != pb) & (pa > 0) & (pb > 0)
    pa, pb = pa[m], pb[m]
    lo = np.minimum(pa, pb).astype(np.int64)
    hi = np.maximum(pa, pb).astype(np.int64)
    key = lo * (nreg + 2) + hi
    u, cnt = np.unique(key, return_counts=True)
    ea = (u // (nreg + 2)).astype(np.int32)
    eb = (u % (nreg + 2)).astype(np.int32)
    dc = np.sqrt(((lab[reg_color[ea]] - lab[reg_color[eb]]) ** 2 * w).sum(1)) * 100.0
    # 邻居表：只保留能吞的（色差达标）
    nbr = [[] for _ in range(nreg + 1)]
    for i, j, n, d in zip(ea, eb, cnt, dc):
        if d <= absorb_de:
            nbr[i].append((int(n), int(j)))
            nbr[j].append((int(n), int(i)))

    uf = _UF(nreg + 1)
    eaten = 0
    for r in np.argsort(area):
        if r == 0 or area[r] <= 0 or area[r] >= min_area:
            continue
        if uf.find(int(r)) != int(r):
            continue
        for n, j in sorted(nbr[r], reverse=True):
            if uf.find(j) == int(r):
                continue
            uf.union(int(r), j)
            eaten += 1
            break

    roots = np.array([uf.find(i) for i in range(nreg + 1)], np.int32)
    _uniq, inv = np.unique(roots, return_inverse=True)
    # 每个根取面积最大成员的原色
    best = np.zeros(int(inv.max()) + 1, np.int64)
    seen = np.zeros(int(inv.max()) + 1, bool)
    for r in np.argsort(-area):
        if r == 0 or area[r] <= 0:
            continue
        g = inv[r]
        if not seen[g]:
            seen[g] = True
            best[g] = r
    new_color = reg_color[best]
    used, inv2 = np.unique(new_color, return_inverse=True)
    pal2 = pal[used]
    root2col = inv2.astype(np.int32)
    out = np.zeros(idx.shape, np.int32)
    nz = reg > 0
    out[nz] = root2col[inv[reg[nz]]]
    ncomp = int(np.unique(reg[nz]).size) if nz.any() else 0
    remain = ncomp - eaten
    return out, pal2, eaten, remain


def apply(idx: np.ndarray, pal: np.ndarray, level: str = "off",
          grid_w: int = 1600, weights: np.ndarray | None = None,
          max_passes: int = 1, rgb: "np.ndarray | None" = None) -> dict:
    """按平涂强度处理标签图。返回 {idx, pal, ...统计}。

    ``level`` 四档：off / light / medium / strong，见 ``FLAT_LEVELS``。
    主体是**阶梯合并**（同一材质的明暗台阶并成一块平色），碎块清理只清
    极小麻点（几十 px² 级别），绝不按面积大刀砍——v7.19.0 试过按面积砍到
    「嘻嘻宝密度」，结果脸和手直接糊成一坨，说明观感差的主因是色阶太多，
    不是块太大。
    """
    lv = (level or "off").lower()
    if lv not in FLAT_LEVELS:
        lv = "off"
    de_lim, area_px = FLAT_LEVELS[lv]
    c0 = int(len(pal))
    if de_lim <= 0 and area_px <= 0:
        return dict(idx=idx, pal=pal, level=lv, colors=c0, colors_before=c0,
                    eaten=0, merged=0, passes=0, min_area=0, changed=False)

    h, w = idx.shape
    # v7.20.0 纹理保护：从原图算出必须保住的细网格（网袜/蕾丝/发丝），
    # 合并色阶时跳过它们——否则平涂会把粉紫网袜线并进肤色，整条腿糊掉。
    # v7.20.0 实测：纹理保护在网袜这类"渐变过渡细线"上无效（翻转法只能抓
    # 硬边细网格），且多耗 0.5s；默认不启用，保留接口供后续引擎级重写用。
    protect = None
    cur, pal, merged = merge_bands(idx, pal, de_lim, weights, protect=protect)

    # 面积阈值按分辨率缩放（以短边 1200 为基准）
    ref = (min(h, w) / 1200.0) ** 2
    min_area = max(16.0, float(area_px) * ref)
    eaten_total = 0
    passes = 0
    for _ in range(max(1, int(max_passes))):
        cur, pal, eaten, remain = _absorb_pass(cur, pal, min_area,
                                               FLAT_ABSORB_DE, weights)
        eaten_total += eaten
        passes += 1
        if eaten == 0:
            break
        min_area *= 1.6

    used = np.unique(cur)
    remap = np.full(int(cur.max()) + 1, -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    cur = remap[cur]
    pal = pal[used]
    return dict(idx=cur, pal=pal, level=lv, colors=int(len(pal)),
                colors_before=c0, eaten=int(eaten_total), merged=int(merged),
                passes=passes, min_area=round(min_area, 0), changed=True)
