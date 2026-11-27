# -*- coding: utf-8 -*-
"""先定线后填色引擎 (MODE_LINEFIRST)。

v7.22.0 起改用「色阶量化 + 空间投票」路线（借鉴开源项目 Lumina-IMTVec 的 bands 思路）。
旧的分水岭分区实现保留在文件末尾 `trace_watershed`，可用于回退对照。

为什么换路线（实测 5 张图，PSNR 平均 +3dB）：
  旧：先找"谷"当种子 → 分水岭长色块。渐变区的梯度在小数值间抖动，
      会被切成碎片；而平滑区又切得太粗，细节丢失。两头不讨好。
  新：先把整图量化成一套细密调色板（每像素归到最接近的一档），
      再对低对比区域「先按离边缘距离自适应模糊、再分配、再做高斯投票」，
      色阶边界因此被拉成平滑等值线；高对比区域（线、眼高光、眉）保持原始锐利分配。
      最后合并碎块、合并几乎同色的相邻档。
      量化本身就决定了分块，不依赖梯度抖动，渐变区不会被切碎，细节也能留住。

打印版输出（v7.26.0）：
  - 色块层：56 档量化色阶块（vtracer 平滑描摹），全部闭合实心
  - 补缝层：每个色块复制一份描边、颜色=它自己的颜色（防亚像素白缝，不是黑线）
  - 实心线带层：CNN 真轮廓加粗成 2px 宽闭合深色色带，打印能挤出、找回发丝五官
  - 线稿细线层：默认关闭（1~2px 线打印挤不出料），改 EMIT_LINE_LAYER=1 可恢复，仅供当图片看
"""

from __future__ import annotations

import io
import math
import os
import re
import base64
import tempfile
import time

import numpy as np
import cv2

import core

# ==================== 参数 ====================
# 每个参数都按「管什么 / 调大怎样 / 调小怎样」写清楚，方便日后调。

# 色阶档数：整张图被量化成多少种颜色（就是"用几种颜色的塑料片去叠"）。
# 越大→色阶越细腻、越接近原图，但色片数量暴涨、文件变大、换料次数多；
# 越小→画面越平、越像海报，细节层次变少。
# v7.22.3：112 → 160（厚涂图暗纹不被压成阶梯）。
# v7.22.3：112 → 160（厚涂图暗纹不被压成阶梯）。
# 工坊模块按 **v7.24** 行为移植，所以停在 160（v7.26 的 56 档不采用）。
BANDS = 160

# 细节专用色数：眼高光、鼻点、嘴线这些小而深/亮的东西，颜色常常在主调色板里
# 找不到对应档，会被糊进周围肤色。这里给高对比像素单独留一批颜色。
# 越大→小细节颜色越准，但碎点变多；越小→小细节更容易丢；0=关闭。
# v7.26.0：48 → 18，跟主档 56 按 1:3 配套（实验 A 用的同一组设置）。
# 工坊模块按 v7.24 行为移植，保持 48（与 160 档配套）。
BANDS_DETAIL = 48

# 边缘判定阈值（OKLab 明度梯度）：超过它算「高对比」，用原始像素分配（保住锐利）。
# 越大→只有很硬的边才走锐利路线，更多区域被平滑（更干净，但淡线可能糊）；
# 越小→更多区域保留原始锐利分配（细节多，但边缘更硬、噪点多）。
# v7.22.3：0.40 → 0.18。背景墙纸蝙蝠、裤子暗纹这类"对比度低但形状清晰"的图案，
# 原来被判成平坦区、遭大半径模糊抹平；降门槛让它们也走锐利路线，形状得以保留。
EDGE_THR = 0.18

# 明度权重 / 色度权重：决定「亮暗差异」和「色相差异」哪个更容易被当成不同色块。
# 明度权重调大→更按亮暗分层；色度权重调大→更按颜色分层（更接近人眼感受）。
L_WEIGHT = 1.3
C_WEIGHT = 2.0

# 投票平滑强度（px）：色阶边界被高斯投票拉直的程度。
# 越大→边界越顺滑、越像手画的等值线，但小结构会被抹掉；
# 越小→边界更贴合原图，但容易出锯齿台阶。3.0 是「顺滑又留得住小结构」的点。
VOTE_SIGMA = 3.0

# 开阔区模糊半径（px）：离线条远的地方先模糊再分配，色带才会平滑不留阶梯。
# 越大→大面积渐变越顺、越少色带断层，但过度会糊掉柔和细节；0=关闭该特性。
# v7.22.3：12 → 4。12px 会把背景墙纸纹样、布料细纹整片抹平（用户反馈"花了"）；
# 4px 仍能顺开大面积渐变的色带，但不再连小图案一起糊掉。
BLUR_SIGMA = 4.0

# 贴边模糊半径（px）：离线条很近的地方只轻微模糊，避免把下巴阴影、轮廓阴影糊掉。
# 越大→贴边处也会被平滑（更干净，但可能丢贴边的小阴影）；越小→贴边细节保得越狠。
NEAR_SIGMA = 2.0

# 模糊半径从「贴边值」过渡到「开阔值」的距离区间（px）：
# 离边 4px 内用贴边值，24px 外完全用开阔值，中间线性过渡。
BLUR_RAMP = (4.0, 24.0)

# 碎块合并面积（px）：面积小于它的连通块并入最相似的邻居。
# 越大→画面越干净，但会吃掉小装饰（纽扣、小花瓣）；越小→细节多但噪点碎斑多。
SPECK_AREA = 48

# 碎块保留门限（OKLab 色差）：碎块与邻居色差超过它的，即使很小也保留。
# 这是「眼高光、鼻点」能被留住的原因——它们小但和周围差得多。
# 越大→越倾向保留小亮点（更保真但可能有噪点）；越小→更激进地清掉小点。
SPECK_KEEP_DE = 0.10

# 相邻色档合并门限（OKLab 色差）：颜色几乎一样的两档直接合并。
# 用来抹掉量化在渐变区留下的同心色带（阶梯感）。
# 越大→色带越少越平滑，但层次变少；越小→层次多但可能看出断层。0=不合并。
MERGE_DE = 0.040

# 一组合并后允许的最大颜色跨度（OKLab 色差，0=不限＝旧版无限串联）。
# v7.22.5 起从「整条渐变串成一坨」改成「局部合并」：合并后这一组的颜色
# 从最浅到最深如果超过 MERGE_SPAN，就拒绝这次合并。
# 越大→每组吃得越宽，色块更少文件更小，渐变层次也更少；
# 越小→层次保得越细，但色块和文件变大。0.025 实测：渐变图层次翻倍、
# 网点图不再被压平（网点图反而 +10dB）。
MERGE_SPAN = 0.025

# ---- 线稿层参数 ----
# 线片段最小面积（px）：小于它的 CNN 线当噪点删掉。越大→线越干净但丢细发丝。
MIN_LINE_PX = 12

# 线稿粗细（工作网格 px）：越大越清晰、但过粗会糊住细节。
LINE_WIDTH = 2.4

# 彩色线分组数：线层按原图色聚成 N 组。越大线色越接近原图，但文件更大。
LINE_COLOR_GROUPS = 48

# 彩色线饱和度增强倍数：1.0=原色不动。1.5 让暗光下的粉线回到可见。
LINE_SAT_BOOST = 1.5

# 线稿最短长度（px）：短于它的碎线丢掉。8 能保住细发丝又不留噪点短线。
LINE_MIN_LEN = 8

# ---- 高光噪点过滤（深色区域白点/白圈的根因）----
# 现象：CNN 会把头发上的高光纹理也当成"线"提出来，取色是浅色，
#       画到深色头发上就成了一堆白点、白色小圈圈。
# 判据：一条线自身比 LINE_HL_L 亮、而它周围底色比 LINE_HL_BG 暗，
#       就判定为"亮线落在暗区 = 高光误检"，整条线删掉。
# 越大/越低 → 触发删除的条件越苛刻，保留的线越多（白点可能残留）；
# 越低/越高 → 删得越狠（白点干净，但极少数贴着暗底的真浅线可能一起被删）。
LINE_HL_L = 0.52      # 线亮度门槛(0~1)：线自身超过它才算"浅色线"
LINE_HL_BG = 0.42     # 背景亮度门槛(0~1)：周围底色低于它才算"暗区"
LINE_HL_R = 6         # 采样底色时离线的距离(px)：越大看得越远、越不被线本身干扰
LINE_HL_SAT = 70      # 饱和度上限(0~255)：线饱和度低于它才算"灰白高光"。
                      # 加这一条是为了保住粉线/金线——它们也亮，但饱和度高，不是白点。
                      # 越大→越多亮线被当白点删（粉线风险升高）；越小→只删最灰白的。

# ---- 纹理线过滤（金属拉丝/布料排线被误当线稿，画面显脏显糊的根因）----
# 引擎先量化出色块再提线，真轮廓一定落在色块交界处。若一条线大部分躺在
# 同一色块内部，它就是材质纹理而非轮廓，留着会让画面脏。
# 【实测无效，默认关闭 0】两种判据都试过：
#   ①"线是否落在色块边界上"—— 色块有 160 档、边界带覆盖 33% 画面，
#      线稿本来就从梯度处提，几乎所有线 frac≈1.00，删不掉任何一条；
#   ②"线两侧色差"—— 金属拉丝/排线本身就有明暗色差（中位 0.72），
#      与真轮廓的色差分布重叠，阈值 0.16 只能删 3%，且会误删细节线。
# 结论：纹理线与轮廓线在像素统计上不可分，属于本引擎架构的已知边界。
LINE_ON_BOUNDARY = 0.0    # 0=关闭。只有找到新的可分特征时才值得重新开启。

# ---- 双模式自动切换（v7.23.0）----
# 本引擎有两种底色做法：
#   ① 纯矢量色块（老路线）：整个 SVG 都是数学图形，无限放大不糊；
#      但柔和渐变会被切成一圈圈硬色块，放大看出色带。
#   ② 嵌平滑底图（新路线）：底色用原图压成的小图垫在下面，线稿仍是矢量；
#      渐变完美平滑、文件还更小；但底色是位图，无限放大最终会糊。
# 程序按「线覆盖率」自动选：线条少（<阈值）说明是渐变/徽章类图，走②；
# 线条多说明是网点/厚涂/线稿类图，走①（这类图本来就没有渐变硬边问题，
# 且更需要放大不糊）。
# 越大→更多图走嵌底图（画质更平滑但更多图不是纯矢量）；
# 越小→更多图走纯矢量（更保"无限放大不糊"但渐变图会有硬边）。
SOFT_BG_RATIO = 1.0       # 线覆盖率阈值(%)。仅嵌底图模式用，打印模式不生效。
SOFT_BG_JPEG_Q = 90       # 嵌底图图片质量(1~100)。仅嵌底图模式用，打印模式不生效。
# v7.24.0：本产品用途是 3D 打印叠色——切片软件读不了 SVG 里的 <image> 位图，
# 渐变图一旦嵌底图，打印时那层颜色直接丢失。故默认关掉，全程纯矢量色块。
# 回滚：改成 1 可恢复老双模式（只适合当图片看，不适合打印）。
SOFT_BG_ENABLED = 0       # 3D 打印叠色必须为 0：绝不嵌位图，所有图都走纯色块。

# ---- 3D 打印叠色：输出层控制（v7.24.0）----
# 打印机一层塑料一条线堆料，没有"描边"概念：CNN 提取的彩色细线（1~2px）
# 打出来要么细到挤不出、要么翘成一根根塑料丝，费料还破坏画面 → 不输出。
# 色块层每块都是闭合实心 path，对应一枚"色片"，这才是打印要的；补缝层保留，
# 让相邻色块略微重叠、打印不留白缝。
# 回滚：改成 1 可恢复线稿层（只适合当图片看）；打印必须保持 0。
EMIT_LINE_LAYER = 0

# ---- 补缝层参数 ----
# 每个色块向外描边的宽度（工作网格 px），描边色=该块自身颜色。
# 只用来盖住相邻色块之间的亚像素白缝，颜色一样所以看不出描边。
# 越大→缝盖得越彻底，但色块会略微变胖；0=关闭。
# v7.22.2：重新开启，但改成"直接在色块 path 上加同色细描边"，
# 不再单独生成空心描边层；宽度很细 + 跳过近白块，所以不会在白底图勾出毛线团。
# 0.30 是实测平衡点：盖得住亚像素白缝，画质仅降 0.12dB（0.7 会降 0.63dB）。
GAP_W = 0.3

# 跳过描边的近白块亮度门槛(0~1)：比它亮的色块不描边。
# 白块之间漏的是白缝、本来就看不出来，描边只会在浅色图上勾出犬牙乱线。
# 越大→更多浅色块也描边（白底图乱线风险升高）；越小→更多浅块跳过。
GAP_SKIP_LIGHT = 0.90

# ---- 实心线带（v7.26.0，用户实验 B 选定的"加轮廓"方案）----
# 之前打印版把 CNN 细线整层关掉，因为 1~2px 的线打出来挤不出料。
# 这里换一种做法：把"真轮廓线"加粗成一条有宽度的实心色带，变成闭合的
# 深色色片，切片软件能直接打印，头发丝、五官轮廓因此能找回来。
#
# SOLID_BAND_PX = 线带宽度（工作网格 px，相当于"深色塑料线画多粗"）。
#   2px 是用户看对比图后定的：1px 太细打印容易断；3px 以上颜色覆盖太多、
#   画面变脏。调大→轮廓更粗更醒目、但深色料用得多；调小→更轻但可能打不出。
# SOLID_BAND_COLOR = 线带颜色。用实验对比图同款深棕 #2a2118，
#   相当于"多备一卷深色料"，轮廓才压得住底下 72 种色片。
# EMIT_SOLID_BAND = 总开关。1=输出（打印版默认）；0=回到纯色块（不推荐）。
EMIT_SOLID_BAND = 0
SOLID_BAND_PX = 2
SOLID_BAND_COLOR = "#2a2118"

# 打印体检（v7.25.0 新增）：转完统计颜色数/碎块/细线占比，写进 SVG 的 metadata。
# 工坊模块按 v7.24 行为移植，因此关闭：输出里不带体检段落。
EMIT_PRINTCHECK = 0

# 密集线保护（v7.26.0 实测必需）：眼睛、嘴巴、衣褶这类地方线挤在一起，
# 每根都加粗到 2px 就会连成黑块糊住五官。这里数 5×5 小格子里有几根线：
# 超过 SOLID_BAND_DENSE 根就保持原细线不膨胀，只有稀疏的真轮廓才加粗。
# 调大→越多地方加粗（轮廓醒目但可能糊脸）；调小→越多地方保持细（五官更安全）。
# 注意：一根直线穿过 5×5 小格子本身就有约 5 个点，所以门槛必须高于 5；
# 9 是实测平衡点：单根发丝/外轮廓照常加粗，交叉成团的五官细节不被糊住。
SOLID_BAND_DENSE = 9
SOLID_BAND_DENSE_K = 5

# 内部工作画布宽度（px）：越大色块边界越精细但越慢；越小越快但细节少。
# 1600 是「细节够 + 20~30 秒出结果」的平衡点。
# 注意：原图比它窄时会被放大到 1600。放大本身能让边界更精细，但**必须用
# 平滑插值**（LANCZOS），用 INTER_AREA 放大（它只适合缩小）会出块状锯齿。
WORK_W = 1600
# 工作网格的另外两道闸门。v0.1.6 只把宽度固定到 1600，窄长图（如 16×1600）
# 会被等比放大到 1600×160000——2.56 亿像素，RGB 原始数组就 768 MB，后续中间
# 数组继续放大。这里同时限制总像素与高度；普通图片（宽 1600、高 ≤2869）不受影响，
# 输出 SVG 的 width/height 仍是原图尺寸，只有内部工作网格会按需要收紧。
MAX_WORK_PIXELS = 5_000_000
MAX_WORK_H = 4096

_FILL_HEX = re.compile(r'fill="(#[0-9a-fA-F]{6})"')


def work_canvas_size(ow: int, oh: int) -> tuple[int, int]:
    """按原图宽高比算工作网格，同时满足宽/高/总像素三道闸门。

    普通图仍保持旧行为：宽度放大/缩小到 WORK_W，高度等比变化。
    窄长图会被高度上限或总像素上限收紧，避免 cv2.resize 申请超大数组。
    返回的只是内部工作网格；SVG 的 width/height 仍由原图尺寸决定。
    """
    ow, oh = max(1, int(ow)), max(1, int(oh))
    scale = min(
        WORK_W / float(ow),
        MAX_WORK_H / float(oh),
        (MAX_WORK_PIXELS / float(ow * oh)) ** 0.5,
    )
    w = max(1, min(WORK_W, int(round(ow * scale))))
    h = max(1, min(MAX_WORK_H, int(round(oh * scale))))
    # 四舍五入后仍可能略微越线，按总像素再收一次高度。
    if w * h > MAX_WORK_PIXELS:
        h = max(1, int(MAX_WORK_PIXELS // w))
    return w, h


def _force_root_px(svg: str, ow: int, oh: int) -> str:
    """把 SVG 显示尺寸钉回原图宽高，避免工作网格取整带来 1~2px 偏差。"""
    m = re.search(r'<svg\b[^>]*>', svg)
    if not m:
        return svg
    tag = m.group(0)
    tag = re.sub(r'\bwidth="[^"]*"', 'width="%d"' % ow, tag, count=1)
    tag = re.sub(r'\bheight="[^"]*"', 'height="%d"' % oh, tag, count=1)
    return svg[:m.start()] + tag + svg[m.end():]


def _load_rgb(src):
    """读成 RGB 三通道，并且把「透明」真正当透明处理。

    为什么必须这样做：有些 PNG 是「擦掉但没清干净」的——像素被标成
    alpha=0（看不见），底下的 RGB 却还留着深色（例：蓝徽章圆外藏着一圈
    看不见的深蓝）。直接 convert("RGB") 会把这种看不见的颜色当成真内容，
    描出来的 SVG 就会凭空多出原图没有的弧线/色块。
    正确做法：按 alpha 把图贴到白底上，看不见的地方就是白。
    同时做 EXIF 方向校正（手机拍的竖图不校正会躺倒）。
    """
    from PIL import Image as _I
    from PIL import ImageOps as _IO

    with _I.open(src) as im:
        im = _IO.exif_transpose(im)
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (
            im.mode == "P" and "transparency" in im.info)
        if has_alpha:
            im = im.convert("RGBA")
            base = _I.new("RGB", im.size, (255, 255, 255))
            base.paste(im, mask=im.getchannel("A"))
            return np.asarray(base)
        return np.asarray(im.convert("RGB"))


# ==================== 色阶量化核心 ====================

def _oklab_w(rgb_u8):
    """sRGB(0~255) → 加权 OKLab（明度×L_WEIGHT，色度×C_WEIGHT）。"""
    import color as clr
    flat = rgb_u8.reshape(-1, 3).astype(np.float32) / 255.0
    lab = clr.srgb_to_oklab(flat).astype(np.float32)
    lab[:, 0] *= L_WEIGHT
    lab[:, 1] *= C_WEIGHT
    lab[:, 2] *= C_WEIGHT
    return lab


def _fit_palette(lab_w, k, sample=200_000, mask=None):
    """在加权 OKLab 空间做 k-means，得到 k 档调色板。"""
    pts = lab_w[mask.ravel()] if mask is not None else lab_w
    if len(pts) < k * 4:
        return None
    if len(pts) > sample:
        idx = np.random.default_rng(7).choice(len(pts), sample, replace=False)
        pts = pts[idx]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    _, _, cen = cv2.kmeans(pts, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    return cen.astype(np.float32)


def _assign(lab_w, cen):
    """每个像素归到最近的调色板档。用矩阵乘代替 (N,k,3) 差值数组，省内存。"""
    c = np.ascontiguousarray(cen, np.float32)
    cn = (c * c).sum(1)
    out = np.empty(len(lab_w), np.int32)
    step = 1_000_000
    for s in range(0, len(lab_w), step):
        out[s:s + step] = (cn[None, :] - 2.0 * (lab_w[s:s + step] @ c.T)).argmin(1)
    return out


def _edge_strength(lab_w, hw):
    """OKLab 明度的局部梯度（膨胀一点，让边缘判定更稳）。"""
    L = lab_w[:, 0].reshape(hw).astype(np.float32)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.dilate(np.hypot(gx, gy), np.ones((3, 3), np.uint8))


def _masked_blur(rgb, low, sigma):
    """只用「非边缘」像素做归一化卷积，避免线条的深色渗进皮肤。"""
    m = cv2.dilate((~low).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
    mf = m.astype(np.float32)
    num = cv2.GaussianBlur(rgb.astype(np.float32) * mf[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(mf, (0, 0), sigma)[..., None]
    return np.clip(np.where(den > 1e-3, num / np.maximum(den, 1e-3),
                            rgb.astype(np.float32)), 0, 255).astype(np.uint8)


def _adaptive_blur(rgb, low, sigma, near_sigma=NEAR_SIGMA, ramp=BLUR_RAMP):
    """离边缘越远模糊越大：开阔区得到平滑色带，贴边处保住真实阴影。"""
    from scipy import ndimage as ndi
    d = ndi.distance_transform_edt(low)
    w = np.clip((d - ramp[0]) / (ramp[1] - ramp[0]), 0, 1)[..., None].astype(np.float32)
    near = _masked_blur(rgb, low, near_sigma).astype(np.float32)
    if sigma <= 0:
        return np.clip(near, 0, 255).astype(np.uint8)
    far = _masked_blur(rgb, low, sigma).astype(np.float32)
    return np.clip(near * (1 - w) + far * w, 0, 255).astype(np.uint8)


def _soft_vote(labels, sigma, mask=None):
    """高斯加权多数投票：每个色档做一张模糊图，逐像素取胜出者。
    这一步把锯齿状色阶边界拉成平滑等值线 —— 是「干净」的主要来源。"""
    if sigma <= 0:
        return labels
    from scipy import ndimage as ndi
    H, W = labels.shape
    pad = int(math.ceil(sigma * 3.5)) + 2
    boxes = ndi.find_objects(labels + 1)
    best = np.full(labels.shape, -1.0, np.float32)
    out = labels.copy()
    m_f = None if mask is None else mask.astype(np.float32)
    for c in np.unique(labels if mask is None else labels[mask]):
        c = int(c)
        if c < 0 or c >= len(boxes) or boxes[c] is None:
            continue
        sy, sx = boxes[c]
        y0, y1 = max(0, sy.start - pad), min(H, sy.stop + pad)
        x0, x1 = max(0, sx.start - pad), min(W, sx.stop + pad)
        ind = (labels[y0:y1, x0:x1] == c).astype(np.float32)
        if m_f is not None:
            ind *= m_f[y0:y1, x0:x1]
        p = cv2.GaussianBlur(ind, (0, 0), sigma)
        bsub = best[y0:y1, x0:x1]
        upd = p > bsub
        if mask is not None:
            upd &= mask[y0:y1, x0:x1]
        osub = out[y0:y1, x0:x1]
        osub[upd] = c
        bsub[upd] = p[upd]
        out[y0:y1, x0:x1] = osub
        best[y0:y1, x0:x1] = bsub
    return out


def _label_by_value(labels):
    """按「同值且 4 连通」打标签。
    OpenCV 的 connectedComponents 只看非零、不看值，直接把标签图喂进去会把整图当一个块，
    所以必须逐类做二值连通标记再拼接。"""
    out = np.zeros(labels.shape, np.int64)
    off = 0
    for c in np.unique(labels):
        m = (labels == c)
        n, lb = cv2.connectedComponents(m.astype(np.uint8), connectivity=4)
        sel = lb > 0
        out[sel] = lb[sel] + off
        off += int(n) - 1
    return out, off


def _merge_specks(labels, lab_w, max_area=SPECK_AREA, keep_de=SPECK_KEEP_DE, iters=2):
    """小碎块并入最相似邻居；色差超过 keep_de 的保留（眼高光、鼻点等小但重要的点）。"""
    n_reg = 0
    for _ in range(iters):
        reg, n_reg = _label_by_value(labels)
        sizes = np.bincount(reg.ravel(), minlength=n_reg + 1)
        small = sizes < max_area
        small[0] = False
        if not small[reg].any():
            break
        n_cls = int(labels.max()) + 1
        # 统计「碎块 → 邻居色类」的票数（纯 numpy 排序聚合）
        rr_l, oo_l = [], []
        for dy, dx in ((0, 1), (1, 0)):
            a = labels[:(-1 if dy else None), :(-1 if dx else None)].ravel()
            b = labels[(1 if dy else 0):, (1 if dx else 0):].ravel()
            ra = reg[:(-1 if dy else None), :(-1 if dx else None)].ravel()
            rb = reg[(1 if dy else 0):, (1 if dx else 0):].ravel()
            m1 = (a != b) & small[ra]
            if m1.any():
                rr_l.append(ra[m1]); oo_l.append(b[m1])
            m2 = (a != b) & small[rb]
            if m2.any():
                rr_l.append(rb[m2]); oo_l.append(a[m2])
        if not rr_l:
            break
        rr = np.concatenate(rr_l).astype(np.int64)
        oo = np.concatenate(oo_l).astype(np.int64)
        uq, cnt = np.unique(rr * (n_cls + 1) + oo, return_counts=True)
        r_of, nb_of = uq // (n_cls + 1), uq % (n_cls + 1)
        order = np.lexsort((cnt, r_of))
        r_s, nb_s = r_of[order], nb_of[order]
        cutp = np.flatnonzero(np.diff(r_s)) + 1
        ends = np.concatenate([cutp, [len(r_s)]]) - 1
        best_rid, best_nb = r_s[ends], nb_s[ends]

        cnt_all = np.bincount(labels.ravel(), minlength=n_cls).astype(np.float64)
        cen = np.stack([np.bincount(labels.ravel(), lab_w[:, i], minlength=n_cls)
                        for i in range(3)], 1) / np.maximum(cnt_all, 1)[:, None]
        # 每个连通块自己属于哪个色类（众数）
        own_key = reg.ravel().astype(np.int64) * (n_cls + 1) + labels.ravel()
        ouq, ocnt = np.unique(own_key, return_counts=True)
        o_r, o_c = ouq // (n_cls + 1), ouq % (n_cls + 1)
        o_ord = np.lexsort((ocnt, o_r))
        o_r, o_c = o_r[o_ord], o_c[o_ord]
        o_cut = np.flatnonzero(np.diff(o_r)) + 1
        o_ends = np.concatenate([o_cut, [len(o_r)]]) - 1
        own_map = np.zeros(n_reg + 1, np.int32)
        own_map[o_r[o_ends]] = o_c[o_ends]

        # 用查找表一次写回（逐块 labels[reg==rid] 是每块一次全图比较，会卡死）
        rep = np.zeros(n_reg + 1, np.int64)
        for rid, tgt in zip(best_rid.tolist(), best_nb.tolist()):
            own = int(own_map[rid])
            if tgt == own:
                continue
            if float(np.linalg.norm(cen[own] - cen[tgt])) <= keep_de:
                rep[rid] = tgt + 1
        mrep = rep[reg] > 0
        if not mrep.any():
            break
        labels[mrep] = rep[reg][mrep] - 1
    return labels


def _merge_close_classes(labels, lab_w, thresh=MERGE_DE, span=MERGE_SPAN):
    """把颜色几乎一样的相邻色档合并，抹掉量化在渐变区留下的同心色带。

    v7.22.5 关键改动：加「组内颜色跨度上限 span」。
    旧版一路 union-find 串下去，渐变 1-2-3-4…每对差都小于 thresh，于是整条
    渐变被串成一坨压平成 1 色（柔和渐变丢失的根因）；网点图也会被压平。
    现在合并前检查「合并后这一组最浅到最深差多少」，超过 span 就拒绝合并，
    于是渐变被切成若干段，每段约 span 宽，层次保住。
    span<=0 时退回旧的无限串联行为（仅作对照用）。
    """
    n = int(labels.max()) + 1
    if n < 2 or thresh <= 0:
        return labels
    cnt = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    cen = np.stack([np.bincount(labels.ravel(), lab_w[:, i], minlength=n)
                    for i in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    pl = []
    for dy, dx in ((0, 1), (1, 0)):
        a = labels[:(-1 if dy else None), :(-1 if dx else None)]
        b = labels[(1 if dy else 0):, (1 if dx else 0):]
        diff = a != b
        if diff.any():
            pa, pb = a[diff].astype(np.int64), b[diff].astype(np.int64)
            lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
            pl.append((lo << 32) | hi)
    if not pl:
        return labels
    e = np.unique(np.concatenate(pl))
    ea, eb = (e >> 32).astype(np.int32), (e & 0xFFFFFFFF).astype(np.int32)
    dE = np.linalg.norm(cen[ea] - cen[eb], axis=1)
    parent = np.arange(n, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    area = cnt.copy()
    # 每一组当前的颜色包围盒（下界 / 上界），用来算「组内跨度」
    lo_c = cen.copy()
    hi_c = cen.copy()
    for idx in np.argsort(dE):
        if dE[idx] >= thresh:
            break
        ra, rb = find(int(ea[idx])), find(int(eb[idx]))
        if ra != rb:
            if area[ra] < area[rb]:
                ra, rb = rb, ra
            if span > 0:
                nlo = np.minimum(lo_c[ra], lo_c[rb])
                nhi = np.maximum(hi_c[ra], hi_c[rb])
                if float(np.linalg.norm(nhi - nlo)) > span:
                    continue          # 合并后色跨度过大 → 拒绝，保住层次
            else:
                nlo, nhi = lo_c[ra], hi_c[ra]
            parent[rb] = ra
            area[ra] += area[rb]
            lo_c[ra], hi_c[ra] = nlo, nhi
    for i in range(n):
        if parent[i] != i:
            parent[i] = find(i)
    out = parent[labels]
    _, inv = np.unique(out, return_inverse=True)
    return inv.reshape(labels.shape).astype(np.int32)


def posterize(rgb, progress=None, return_maps=False):
    """把一张 sRGB 图量化成「色阶标签图」，返回 (标签图, 用到的档数)。

    return_maps=True 时额外返回 (labels, pal)：铺色用的色块索引图 + 每档颜色，
    供渐变填充模块（gradfit）按色块拟合渐变。
    """
    h, w = rgb.shape[:2]
    lab_w = _oklab_w(rgb)

    if progress:
        progress("分析画面边缘…")
    edges = _edge_strength(lab_w, (h, w)) > EDGE_THR

    if progress:
        progress("建立 %d 档调色板…" % BANDS)
    cen = _fit_palette(lab_w, BANDS)
    if cen is None:
        raise RuntimeError("颜色种类太少，无法量化")
    if BANDS_DETAIL and edges.any():
        cd = _fit_palette(lab_w, BANDS_DETAIL, mask=edges)
        if cd is not None:
            cen = np.concatenate([cen, cd], 0)

    if progress:
        progress("分配色阶…")
    raw = _assign(lab_w, cen).reshape(h, w)
    low = ~edges
    if BLUR_SIGMA > 0 and low.any():
        if progress:
            progress("平滑色带（%d 档投票）…" % len(cen))
        smooth_rgb = _adaptive_blur(rgb, low, BLUR_SIGMA)
        smooth = _assign(_oklab_w(smooth_rgb), cen).reshape(h, w)
        base = np.where(low, smooth, raw)
        labels = np.where(low, _soft_vote(base, VOTE_SIGMA, low), raw)
    else:
        labels = _soft_vote(raw, VOTE_SIGMA)

    if progress:
        progress("清理碎块…")
    if SPECK_AREA > 0:
        labels = _merge_specks(labels, lab_w, SPECK_AREA, SPECK_KEEP_DE)
    if MERGE_DE > 0:
        labels = _merge_close_classes(labels, lab_w, MERGE_DE, MERGE_SPAN)

    # 每档颜色取源图该档像素的中位数（不用调色板原色，避免整体偏色）
    n_lab = int(labels.max()) + 1
    flat_l, flat_s = labels.ravel(), rgb.reshape(-1, 3)
    order = np.argsort(flat_l, kind="stable")
    sl, ss = flat_l[order], flat_s[order]
    b = np.searchsorted(sl, np.arange(n_lab + 1))
    pal = np.zeros((n_lab, 3), np.uint8)
    for c in np.unique(labels):
        seg = ss[b[c]:b[c + 1]]
        if len(seg):
            pal[c] = np.median(seg, 0)
    if return_maps:
        return pal[labels], int(len(np.unique(labels))), labels, pal
    return pal[labels], int(len(np.unique(labels)))


# ==================== 线稿层 ====================

def drop_highlight_lines(chains, rgb,
                         line_l=LINE_HL_L, bg_l=LINE_HL_BG,
                         R=LINE_HL_R, sat_max=LINE_HL_SAT):
    """删掉"浅灰白的线 + 落在暗底上"的高光误检线（深色区域白点/白圈根因）。

    CNN 会把头发上的高光纹理也当成线提出来；这些线取色是浅灰白，画到深色块上
    就成了扎眼的白点、小白圈。判据三条同时满足才删：
      1) 线自身亮度 > line_l
      2) 线周围 R px 处的底色亮度 < bg_l
      3) 线自身饱和度 < sat_max（保住粉色/金色这类高饱和的真线）
    返回 (保留的 chains, 删除条数)。任一开关 <=0 视为关闭该项。
    """
    if not chains:
        return chains, 0
    if line_l <= 0 and bg_l <= 0:
        return chains, 0
    h, w = rgb.shape[:2]
    # 亮度与饱和度图
    mx = rgb.max(axis=2).astype(np.float32)
    mn = rgb.min(axis=2).astype(np.float32)
    lumi = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1]
            + 0.0722 * rgb[:, :, 2]) / 255.0
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1) * 255.0, 0)
    R = max(1, int(R))
    keep, ndrop = [], 0
    for p, length in chains:
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
        lf = float(np.percentile(lumi[yi, xi], 60))
        sf = float(np.percentile(sat[yi, xi], 60))
        # 周围底色：在上下左右各 R px 处采样取中位，避开线自身
        ys = [np.clip(yi + dy, 0, h - 1) for dy in (-R, R)]
        xs = [np.clip(xi + dx, 0, w - 1) for dx in (-R, R)]
        bgv = np.concatenate([lumi[y, xi] for y in ys] +
                             [lumi[yi, x] for x in xs])
        bf = float(np.median(bgv))
        is_white_line = (line_l > 0 and lf > line_l) or (sat_max > 0 and sf < sat_max)
        is_dark_bg = (bg_l > 0 and bf < bg_l)
        # 必须"浅(或灰) + 暗底"同时成立；避免误删浅底上的白线(如白发高光)。
        if is_white_line and (bg_l <= 0 or is_dark_bg) and lf > 0.42:
            ndrop += 1
            continue
        keep.append((p, length))
    return keep, ndrop


def drop_off_boundary(chains, boundary, min_frac=LINE_ON_BOUNDARY):
    """删掉"不在色块边界上"的线——材质纹理被误当线稿的根因。

    引擎是「先量化出色块、再提线」，所以真正的轮廓线必然落在色块与色块的交界处。
    如果一条线大部分落在同一色块内部（比如金属拉丝、布料排线、渐变肌理），
    它就不是轮廓，而是纹理；留着会让画面显得脏、糊、满是碎线。
    判据：沿线采样，落在边界带上的点占比 < min_frac 就整条删。
    """
    if boundary is None or not chains or min_frac <= 0:
        return chains, 0
    h, w = boundary.shape[:2]
    keep, ndrop = [], 0
    for p, length in chains:
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
        frac = float(boundary[yi, xi].mean())
        if frac < min_frac:
            ndrop += 1
            continue
        keep.append((p, length))
    return keep, ndrop


def colored_line_svg(rgb: np.ndarray, mask: np.ndarray, stroke_width: float,
                     n_groups: int = 24, sat_boost: float = 1.25,
                     min_len: float = 8.0, eps: float = 0.7,
                     boundary=None):
    """按原图实际颜色画 CNN 线稿（不压暗）。

    线稿的颜色取原图该处的色：网袜线在原图是粉色 → 画出来就是粉线；
    轮廓线原图是暗色 → 自然是黑线。用 stroke 走中心线，每条线采样沿线原图色，
    再聚成 n_groups 组，避免每条线一个颜色导致文件爆炸。

    返回 (svg片段, 路径数, 分组数)。
    """
    import linenet_line as ln
    chains = ln.trace_centerlines(ln.thin(mask), min_len=min_len, eps=eps)
    if not chains:
        return "", 0, 0
    # 高光误检线过滤（深色区域白点根因）
    chains, ndrop = drop_highlight_lines(chains, rgb)
    if not chains:
        return "", 0, 0
    # 纹理线过滤（不在色块边界上的线 = 材质纹理，不是轮廓）
    chains, ndrop2 = drop_off_boundary(chains, boundary)
    if not chains:
        return "", 0, 0
    h, w = mask.shape[:2]
    # 取色策略：
    #   max （v7.22.0 旧）：±R 邻域最亮 + 90 分位 + V×1.12。
    #       → 会把本身是暗色的墨线染成近白（44.8% 线色亮度>230），
    #         画到深色块上就成了小白点。
    #   self（新默认）：取线自身的颜色 + 中高分位，不强行提亮。
    #       墨线本身就暗、粉线本身就亮，天然分开，无需邻域 max。
    mode = os.environ.get("LF_LINECOLOR", "self")
    if mode == "max":
        R = 2
        padded = np.pad(rgb, ((R, R), (R, R), (0, 0)), mode="reflect")
        hue_max = np.zeros((h, w, 3), np.uint8)
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                patch = padded[R + dy:R + dy + h, R + dx:R + dx + w]
                np.maximum(hue_max, patch, out=hue_max)
        sample = hue_max
        q = 90.0
    else:
        sample = rgb
        q = float(os.environ.get("LF_LINE_Q", "60"))
    cols = []
    for p, _ in chains:
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
        pix = sample[yi, xi].astype(np.float32)
        cols.append(np.percentile(pix, q, axis=0))
    cols = np.asarray(cols, np.float32)
    hsv = cv2.cvtColor(np.clip(cols, 0, 255).astype(np.uint8).reshape(1, -1, 3),
                       cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 1] = np.clip(hsv[:, 1] * sat_boost, 0, 255)
    if mode == "max":
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

    # 亮的先画、暗的最后压顶层
    order = np.argsort(-centers.max(axis=1))
    segs = []
    for gi in order:
        d = " ".join(ln._poly_d(p, l) for p, l in zip((c[0] for c in chains), labels)
                     if int(l) == int(gi))
        if not d:
            continue
        rr, gg, bb = (int(np.clip(round(v), 0, 255)) for v in centers[gi])
        segs.append('<g fill="none" stroke="#%02x%02x%02x" stroke-width="%.2f" '
                    'stroke-linejoin="round" stroke-linecap="round"><path d="%s"/></g>\n'
                    % (rr, gg, bb, stroke_width, d))
    return "".join(segs), len(chains), len(segs)


# ==================== 补缝层 ====================

def add_gap_filler(svg, width=GAP_W, skip_light=GAP_SKIP_LIGHT):
    """直接在每个色块 path 上加 stroke=自身颜色，盖住相邻色块间的亚像素白缝。

    v7.22.2 改法（旧的"复制一份空心描边层"会在白底图勾出毛线团乱线，已弃用）：
      - 描边色 = 色块自身颜色，所以不会出现黑边；
      - 描边很细(width)，只把色块向外撑 0.35px 盖住缝，视觉上几乎不变；
      - 近白块(skip_light)跳过——白块之间漏的是白缝、本来看不出，描了反而勾乱线。
    返回 (新 svg, 描边块数)。
    """
    if width <= 0:
        return svg, 0
    n = [0]

    def _stroke(m):
        tag = m.group(0)
        fm = _FILL_HEX.search(tag)
        if not fm or "stroke=" in tag:
            return tag
        hx = fm.group(1)
        r, g, b = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        if lum > skip_light:
            return tag
        n[0] += 1
        return (tag[:-2] + ' stroke="%s" stroke-width="%.2f" '
                'stroke-linejoin="round"/>' % (hx, width))

    out = re.sub(r'<path\b[^>]*?/>', _stroke, svg)
    return out, n[0]


# ==================== 实心线带层（v7.26.0） ====================

def solid_band_svg(rgb, fill, width_px=SOLID_BAND_PX, color=SOLID_BAND_COLOR,
                   progress=None):
    """把 CNN 轮廓线加粗成「有宽度的闭合实心带」，供切片软件当色片打印。

    和老的线稿层区别：
      老：stroke 描边，只有一条细线的宽度 → 打印挤不出料、翘成塑料丝；
      新：先把线变成实心区域再描外轮廓 → 一个闭合 path = 一枚深色色片。

    只保留落在色块交界处的线（真轮廓），色块内部的材质纹理不输出，
    否则金属拉丝、布料排线会变成一堆脏点。提取失败时安静返回空，
    绝不影响主流程（打印版仍能出纯色块）。

    返回 (svg片段, 闭合轮廓数, 覆盖画面百分比)。
    """
    if not EMIT_SOLID_BAND or width_px <= 0:
        return "", 0, 0.0
    try:
        import linenet_line as ln
        heat = ln.heatmap(rgb, progress=progress)
        line = ln.filter_small(ln.steger_lines(heat), MIN_LINE_PX)
        if line is None or not np.any(line):
            return "", 0, 0.0

        # 色块边界带 = 形态学梯度再向外扩一圈，跟老线稿层用同一套判据。
        k = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(fill, k)
        ero = cv2.erode(fill, k)
        boundary = (np.abs(dil.astype(np.int16) - ero.astype(np.int16)).sum(2) > 0)
        boundary = cv2.dilate(boundary.astype(np.uint8), k).astype(bool)
        line_b = line & boundary
        if not np.any(line_b):
            return "", 0, 0.0

        # 加粗成实心带：椭圆结构元膨胀，2px 时正好是"两个像素宽"的一根带。
        bw = max(2, int(round(width_px)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bw, bw))
        grown = cv2.dilate(line_b.astype(np.uint8), ker).astype(np.uint8)
        # 线条挤在一起的地方（五官、衣褶）不膨胀，避免深色带连成黑块。
        kk = max(3, int(SOLID_BAND_DENSE_K))
        if kk % 2 == 0:
            kk += 1
        density = cv2.boxFilter(line_b.astype(np.float32), ddepth=-1,
                                ksize=(kk, kk), normalize=False)
        band = np.where(density > float(SOLID_BAND_DENSE),
                        line_b.astype(np.uint8), grown)

        # 每条连通带描成一个闭合 path（颜色统一，打印只需多备一卷深色料）。
        # 【v7.26.0 关键】必须用 RETR_CCOMP 把「带中间的小空洞」也描出来：
        # 眼睛/嘴巴的几根线加粗后会连成一片，只取最外轮廓(RETR_EXTERNAL)
        # 会把中间空隙一起填成黑块，五官直接糊掉。配 fill-rule="evenodd"
        # 后，洞会被正确挖空，只剩细细的深色轮廓。
        cnts, _ = cv2.findContours(band, cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_SIMPLE)
        parts = []
        for c in cnts:
            if cv2.contourArea(c) < 2:
                continue
            pts = c.reshape(-1, 2)
            d = "M" + " L".join("%d %d" % (x, y) for x, y in pts) + " Z"
            parts.append(d)
        if not parts:
            return "", 0, 0.0
        seg = ('<g id="solid-line-band" data-role="printable-ink-band" '
               'fill="%s" fill-rule="evenodd">\n<path d="%s"/>\n</g>\n'
               % (color, " ".join(parts)))
        return seg, len(parts), round(float(band.mean()) * 100.0, 2)
    except Exception:
        # 提线模型缺失/异常时退回纯色块，保证转换一定成功。
        return "", 0, 0.0


# ==================== 主流程 ====================

def _trace_softbg(work, ow, oh, heat, dst, step, t0):
    """嵌平滑底图模式：原图压成 JPEG 垫底 + 矢量线稿叠在上面。

    用于柔和渐变/徽章类图（线覆盖率低）。这类图用纯色块会出硬边色带，
    底图直接保留原图的连续渐变，硬边消失、文件还小；线稿仍是矢量。
    线稿层与纯矢量模式完全共用同一套 colored_line_svg，保证两种模式观感一致。
    """
    import linenet_line as ln
    h, w = work.shape[:2]

    # 1) 底图：原图直接压 JPEG（soft=0：实测任何模糊都会掉画质，网点图尤甚）
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(work, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, SOFT_BG_JPEG_Q])
    if not ok:
        raise RuntimeError("嵌底图压缩失败")
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    image_layer = (
        '<g id="smooth-base-layer" data-role="raster-base">'
        '<image x="0" y="0" width="%d" height="%d" preserveAspectRatio="none" '
        'href="data:image/jpeg;base64,%s"/></g>\n' % (w, h, b64))

    # 2) 线稿层（复用已算的 heat，与纯矢量模式同一套提线/着色逻辑）
    seg, npath, ngroup = "", 0, 0
    try:
        line = ln.filter_small(ln.steger_lines(heat), MIN_LINE_PX)
        if line is not None and np.any(line):
            seg, npath, ngroup = colored_line_svg(
                work, line, stroke_width=LINE_WIDTH, n_groups=LINE_COLOR_GROUPS,
                sat_boost=LINE_SAT_BOOST, min_len=LINE_MIN_LEN, eps=0.7)
    except Exception:
        seg, npath, ngroup = "", 0, 0
    line_layer = ('<g id="line-art-layer" data-role="cnn-line-art-layer">\n%s</g>\n'
                  % seg) if seg else ""

    # 3) 拼装：viewBox=工作网格，显示尺寸=原图（与纯矢量模式一致）
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
           'width="%d" height="%d" viewBox="0 0 %d %d" '
           'preserveAspectRatio="xMidYMid meet" data-engine="linefirst-softbg">\n'
           '<title>Line-first smooth-gradient artwork</title>\n'
           '%s%s</svg>\n' % (ow, oh, w, h, image_layer, line_layer))
    with io.open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)

    return dict(mode="linefirst-softbg", src=dst, dst=dst,
                grid="%d (softbg)" % w, colors=0, paths=seg.count("<path"),
                regions=0, line_paths=npath, line_groups=ngroup,
                boundary_paths=0, gap_paths=0, bands=0,
                bytes=os.path.getsize(dst), hd="",
                engine="softbg", seconds=round(time.time() - t0, 1))


def trace(src: str, dst: str, progress=None, colors: int = 0) -> dict:
    """图片 → SVG。返回统计 dict。"""
    t0 = time.time()

    def step(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    from PIL import Image as _I

    step("读入图片…")
    rgb = _load_rgb(src)
    oh, ow = rgb.shape[:2]
    w, h = work_canvas_size(ow, oh)
    if (w, h) == (ow, oh):
        work = rgb
    else:
        # 缩小用 INTER_AREA（多像素平均，抗锯齿最好）；放大用 LANCZOS4，
        # INTER_AREA 放大会出块状阶梯。
        interp = cv2.INTER_AREA if (w < ow or h < oh) else cv2.INTER_LANCZOS4
        work = cv2.resize(rgb, (w, h), interpolation=interp)

    # ---------- 0) 自动双模式分流（v7.24.0 打印模式默认关闭嵌底图）----------
    # 3D 打印叠色必须全程纯矢量色块：<image> 位图切片软件读不了，打印会丢色，
    # 所以 SOFT_BG_ENABLED=0 时整段跳过，所有图都走下面的纯色块流程。
    # 只有回滚开关设 1（当图片看）时才按线覆盖率分流到嵌底图模式。
    if SOFT_BG_ENABLED:
        try:
            import linenet_line as _ln
            _heat = _ln.heatmap(work)
            _ratio = float((_heat > 0.30).mean()) * 100.0
            if _ratio < SOFT_BG_RATIO:
                step("平滑渐变图（线覆盖 %.2f%%）→ 嵌底图模式…" % _ratio)
                return _trace_softbg(work, ow, oh, _heat, dst, step, t0)
            step("细节丰富图（线覆盖 %.2f%%）→ 纯矢量模式…" % _ratio)
        except Exception:
            pass  # 判据失败一律回退老流程，保证不崩

    # ---------- 1) 色阶量化 + 空间投票 ----------
    step("色阶量化（%d 档）…" % BANDS)
    fill, n_used = posterize(work, progress=step)
    step("已分出 %d 种色阶，正在描摹…" % n_used)

    # ---------- 1.5) 打印体检（v7.25.0 自研；模块按 v7.24 行为关闭）----------
    # 借鉴 Lumina-Vec「打印前检查」思路，但自研实现：直接从色块图统计，
    # 不用无头浏览器渲染，毫秒级。报告跟着 SVG 走（写进 metadata），
    # 同时经返回值透传给 GUI 显示。体检失败绝不影响转换主流程。
    _report = dict(level="none", colors=n_used, regions=0, tiny_regions=0,
                   thin_area_pct=0.0, verdict="", print_mm=0.0, line_mm=0.0)
    if EMIT_PRINTCHECK:
        import printcheck as _pc
        try:
            _report = _pc.check(fill)
        except Exception:
            _report = dict(level="none", colors=n_used, regions=0,
                           tiny_regions=0, thin_area_pct=0.0,
                           verdict="体检未完成", print_mm=_pc.PRINT_LONG_MM,
                           line_mm=_pc.LINE_MM)
        step("打印体检：" + _report.get("verdict", ""))

    # ---------- 2) vtracer 平滑描摹 ----------
    tmp = tempfile.mkdtemp(prefix="linefirst_")
    png = os.path.join(tmp, "fill.png")
    raw = os.path.join(tmp, "fill.svg")
    try:
        _I.fromarray(fill).save(png, "PNG", compress_level=1)
        # cutout 模式：互不重叠的平铺色块。stacked 叠加模式会让满屏底色盖住细节。
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
    _title = "3D-print color layers" if not EMIT_LINE_LAYER else "Line-first flat artwork"
    svg = core._set_root(svg, w, h, ow / float(w), _title)
    svg = _force_root_px(svg, ow, oh)
    _purpose = "3d-print" if not EMIT_LINE_LAYER else "view"
    svg = svg.replace("<svg ", '<svg data-engine="linefirst" data-purpose="%s" ' % _purpose, 1)

    # ---------- 3) 线稿层（3D 打印叠色默认不输出）----------
    # CNN 线稿是 1~2px 彩色细线，打印时要么挤不出、要么翘成塑料丝，费料又破坏
    # 画面，所以 EMIT_LINE_LAYER=0 时整层跳过（还能省掉两次模型推理、明显提速）。
    # 注意：色块分区靠的是 posterize() 里的像素边缘检测，跟这层线稿无关，
    # 关掉它不影响色块结构，只是不再往色块上面叠装饰细线。
    seg, npath, ngroup = "", 0, 0
    if EMIT_LINE_LAYER:
        step("压回线稿（原色，网袜粉线保留）…")
        import linenet_line as ln
        if ln.available():
            heat = ln.heatmap(work, progress=progress)
            line = ln.filter_small(ln.steger_lines(heat), MIN_LINE_PX)
            # 色块边界带：先腐蚀再膨胀，相邻色块间的交界就变成一条窄带。
            # 线稿只保留落在这条带上的（真轮廓），色块内部的纹理线被过滤掉。
            _k = np.ones((3, 3), np.uint8)
            _dil = cv2.dilate(fill, _k)
            _ero = cv2.erode(fill, _k)
            boundary = (np.abs(_dil.astype(np.int16) - _ero.astype(np.int16)).sum(2) > 0)
            boundary = cv2.dilate(boundary.astype(np.uint8), _k).astype(bool)
            try:
                seg, npath, ngroup = colored_line_svg(
                    work, line, stroke_width=LINE_WIDTH, n_groups=LINE_COLOR_GROUPS,
                    sat_boost=LINE_SAT_BOOST, min_len=LINE_MIN_LEN, eps=0.7,
                    boundary=boundary)
            except Exception:
                try:
                    seg, npath = ln.stroke_svg_center(
                        line, stroke_width=LINE_WIDTH, color="#1a1520",
                        min_len=LINE_MIN_LEN, eps=0.7)
                    ngroup = 1
                except Exception:
                    seg, npath, ngroup = "", 0, 0

    # ---------- 3.5) 实心线带（v7.26.0，打印版默认开）----------
    # 老的细线稿层不适合打印；这里把真轮廓加粗成 2px 宽的闭合实心色带，
    # 找回发丝/五官轮廓。提线失败就安静退回纯色块，不影响主流程。
    band_seg, nband, band_cov = "", 0, 0.0
    if EMIT_SOLID_BAND:
        step("把轮廓加粗成实心线带（%d px）…" % SOLID_BAND_PX)
        band_seg, nband, band_cov = solid_band_svg(
            work, fill, SOLID_BAND_PX, SOLID_BAND_COLOR, progress=progress)
        if band_seg:
            step("实心线带：%d 条色片，覆盖 %.2f%%" % (nband, band_cov))
            # 同步写进体检报告，GUI 右键和 SVG metadata 里都能看到这笔"深色料"。
            _report["band_px"] = SOLID_BAND_PX
            _report["band_paths"] = nband
            _report["band_coverage"] = band_cov

    # ---------- 4) 补缝 + 拼装 ----------
    step("补缝并拼装…")
    ngap = 0
    svg, ngap = add_gap_filler(svg, GAP_W)
    line_layer = ('<g id="line-art-layer" data-role="cnn-line-art-layer">\n%s</g>\n' % seg) if seg else ""
    cut = svg.rfind("</svg>")
    if cut < 0:
        svg += "</svg>\n"
        cut = svg.rfind("</svg>")
    # 实心带放最上层，才能压住底下的色块、看出轮廓。
    svg = svg[:cut] + line_layer + band_seg + svg[cut:]

    # 体检报告写进 SVG metadata：跟着文件走，重新打开也能看到，CLI/GUI 通用。
    # 放在 <svg> 根标签「内部」靠前的位置——GUI 只需读前 200KB 就能拿到，
    # 放末尾的话大文件（几 MB）读不到，会误判成"没有体检信息"。
    #
    # 【v7.25.1 关键修复】旧代码用 svg.find(">") 找根标签结尾，命中的却是
    # 第一行 <?xml version="1.0" ...?> 里的 >，结果 metadata 被插到了根元素
    # 「外面」。XML 只允许一个根，浏览器/切片软件读到第二个 <svg> 就报
    # "Unexpected token <"，整张图打不开。现在改成从 <svg 开始逐字符扫描，
    # 跳过引号内的 >，准确找到根标签真正的结尾。
    if EMIT_PRINTCHECK:
        try:
            import re as _re
            import xml.sax.saxutils as _sx
            _meta = ('<metadata id="printcheck" data-level="%s">%s</metadata>\n'
                     % (_report.get("level", "none"),
                        _sx.escape(_pc.report_text(_report))))
            _s = svg.find("<svg")
            if _s >= 0 and len(svg) > _s + 4 and svg[_s + 4] in " \t\r\n>":
                _q, _i = None, _s + 4
                while _i < len(svg):
                    _c = svg[_i]
                    if _q:
                        if _c == _q:
                            _q = None
                    elif _c in "\"'":
                        _q = _c
                    elif _c == ">":
                        break
                    _i += 1
                if _i < len(svg):
                    svg = svg[:_i + 1] + "\n" + _meta + svg[_i + 1:]
        except Exception:
            pass

    # 写盘前自检：宁可丢掉体检报告，也绝不能让用户拿到一张打不开的图。
    # v7.25.0 就是没有这一步，坏文件发到了用户手上才发现。
    try:
        import re as _re2
        import xml.etree.ElementTree as _ET
        try:
            _ET.fromstring(svg)
        except Exception:
            # 解析失败：先摘掉刚加的 metadata 再验一次；能过就少个体检报告，
            # 但图一定能正常打开。
            try:
                svg = _re2.sub(r'<metadata id="printcheck".*?</metadata>\s*',
                               '', svg, flags=_re2.S)
                _ET.fromstring(svg)
            except Exception:
                pass
    except Exception:
        pass

    with io.open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)

    fills = set(_FILL_HEX.findall(svg))
    return dict(mode="linefirst", src=src, dst=dst,
                grid="%d (linefirst)" % w,
                colors=len(fills), paths=svg.count("<path"),
                regions=n_used, line_paths=npath, line_groups=ngroup,
                boundary_paths=ngap, gap_paths=ngap, bands=n_used,
                band_paths=nband, band_coverage=band_cov, band_px=SOLID_BAND_PX,
                bytes=os.path.getsize(dst), hd="", printcheck=_report,
                seconds=round(time.time() - t0, 1))


# ==================== 旧实现（v7.21.x 分水岭路线，保留作回退） ====================

def trace_watershed(src: str, dst: str, progress=None, colors: int = 0) -> dict:
    """v7.21.x 的分水岭分区实现。仅用于回退对照，主流程已改用 trace()。

    思路：CNN 线稿当墙 → 灰度局部极小值当种子 → 分水岭长盆地 → 每盆地按明暗分 3 级。
    缺点：渐变区梯度抖动导致碎块，平滑区又切得太粗。
    """
    import linefirst_v7212_backup as old
    return old.trace(src, dst, progress=progress, colors=colors)
