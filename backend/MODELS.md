# 模型与依赖说明（v7.24 linefirst 现状）

**结论：`manifest.json` 不声明 `models[]`，这是正确且刻意的，不是遗漏。**

## 为什么不需要模型

桌面版 v7.5.7 起，默认 `lumina` 引擎的 2× 超分已从 Real-ESRGAN
（`aihd` / ONNX Runtime）换成 `enhance.py` 的保色算法（guided filter +
LANCZOS + 边缘调制）。`core.py` 里对应注释写明原因：Real-ESRGAN 实测会把
饱和度压掉约 20%、亮度压低约 5%，导致转出的 SVG 整体发淡；换成保色算法后
几乎零颜色损失，同时保留 2× 细节。

因此默认画质链路只依赖 `numpy` / `Pillow` / `vtracer 0.6.15`。
2026-11 起工坊版换成桌面 v7.24 的 `linefirst` 引擎（先定线后填色），
该引擎另外需要两个**纯计算**依赖：

| 依赖 | 用途 |
|---|---|
| `opencv-python-headless` 5.1.0.95 | 调色板 k-means、Sobel 边缘、高斯投票、形态学、颜色空间转换 |
| `scipy` 1.18.1 | `distance_transform_edt`（贴边自适应模糊）、`find_objects`（软投票包围盒） |

它们都是预编译 wheel，安装时不需要编译器，也不含任何模型权重。
`linefirst` 路径不会 import `skimage`（只有本模块从不调用的旧引擎才引用它）。

`aihd.py` / `hd_faithful.py` 仍随算法核心保留，但只在 `faithful_hd` 的
qq3d 分支里被引用，且 `hd_faithful.enhance()` 在模型不可用时明确
「退化为纯 LANCZOS 放大，绝不破坏原图」。缺失模型不会导致失败，更不会
触发后端兜底。

## 实测证据

在同一台 Windows 机器上，用**完全没有 `models/` 目录**的模块后端目录，
以 `numpy` / `Pillow` / `vtracer` 三个包（无 cv2 / scipy / onnxruntime / torch）
跑默认 anime 预设：

| 项目 | 桌面版 v7.6.0 | 模块后端 |
| --- | --- | --- |
| SVG 字节数 | 2,905,307 | 2,905,307 |
| 路径数 | 5,912 | 5,912 |
| sha256 | `2a91987d…a00a10` | `2a91987d…a00a10` |
| 逐字节一致 | — | **PASS** |

同输入重复运行两次输出完全相同（确定性通过）。放大对比图见交付材料
`画质一致性对比_v7.6.0.png`。

## 若将来要重新引入模型

按 `docs/backend-protocol.md`，权重不得打进模块包，须在
`manifest.json` 的 `backend.models[]` 逐项声明
`id` / `fileName` / `url` / `sha256` / `bytes` / `license`，由宿主按哈希
下载校验。`url` 必须是公开、长期不变、可校验的 HTTPS 地址；
`sha256` 必须是文件真实哈希，不得推测。当前不需要，故留空（字段省略）。
