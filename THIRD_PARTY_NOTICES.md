# 第三方组件与来源说明

本文件说明 `图片转 SVG` 使用的第三方组件、许可证，以及算法思想的来源，便于
分发时核对。本工具自身的代码除下列明确列出的组件外，均为本项目独立实现。

##  bundled / 运行时依赖（开源）

| 组件 | 用途 | 许可证 | 随 exe 分发 |
| --- | --- | --- | --- |
| [vtracer](https://github.com/visioncortex/vtracer) 0.6.15 | 默认栅格→矢量描摹 | MIT | 是 |
| [Potrace](http://potrace.sourceforge.net/)（potrace.exe） | 位图轮廓→贝塞尔曲线 | GPL-2.0（独立可执行文件，仅通过命令行调用，未链接/未改写其源码） | 是 |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) `realesr-animevideov3`（`models/anime_sr.onnx`） | 2× 超分（默认开启） | BSD-3-Clause | 是 |
| numpy / Pillow / OpenCV / scipy / scikit-image / onnxruntime | 通用数值与图像库 | 各自 BSD / Apache-2.0 / MIT 等宽松许可 | 是 |

> **v7.5.6 起，运行时只依赖上表组件**，不加载任何外部检测 / 线条模型权重；
> 所有线条层能力均由本项目自研 `line_layer.py`（形态学黑帽 + ridge + 迟滞）实现。

> `potrace.exe` 作为**独立的、未修改的命令行程序**随附，通过进程调用，不与本程序
> 链接成同一可执行文件；其 GPL-2.0 义务不因本程序而扩展。若需要，可从其官网单独获取。

##  已移除的依赖（历史问题，v7.5.3 起不再包含）

| 组件 | 问题 | 处理 |
| --- | --- | --- |
| `neroued_vectorizer` 0.2.0 | GPL-3.0-or-later。曾被 `lum.py` 作为可选引擎引用并被打进 exe；分发会触发对整个程序开源的要求 | **已完全移除**。该引擎改由本项目自研 potrace 管线（`ptr.py`：mean-shift + OKLab 量化 + potrace 曲线拟合）实现，功能保留、无 GPL 依赖 |
| Lumina-Vec `lumina_vec/lines.py`、`manga_net.py` | 来自第三方**私有**仓库，早期版本曾原样复制进 `_lv_lines/` 并打进 exe，且未附作者信息与致谢 | **已彻底删除整个目录**。「线条层」功能改由本项目自研 `line_layer.py`（多尺度黑帽 + ridge 检验 + 迟滞阈值 + potrace）实现；不依赖 torch、不下载任何模型权重、不含任何复制代码 |
| PiDiNet（`models/pidinet_table5.onnx` + `.data`） | 上游 LICENSE 首句：「It is just for research purpose, and commercial use should be contacted with authors first」——**仅授权研究用途**，不适合随对外分发的工具附带；旧说明含糊写成「以发布方许可为准」 | **v7.5.6 起完全移除**。`self_line` 功能位保留，改用项目自研 `line_layer.stroke_mask()` 形态学描线，无需任何模型 / 权重 |
| `vtracer10`（vtracer 1.0 + `simplify`） | 无任何 CLI / GUI 入口；A/B 实测同图 path 数 23414→106339、脸部出现细碎条纹，画质更差 | **v7.5.6 起移除**相关代码与打包项，默认描摹器固定为 vtracer 0.6 |
| `resvg_py` | 仅被已关闭的右侧大图预览与 SVG 缩略图分支使用，运行时从不调用 | **v7.5.6 起移除**，预览死代码与孤儿控件一并删除 |
| `models/anime.pth`（Real-ESRGAN torch 权重） | 仅用于构建期导出 `anime_sr.onnx`，运行时从不加载 | **v7.5.6 起不再打包**，只保留 `.onnx` + `.data` |

##  算法思想来源与致谢

本项目是 Lumina 叠色 3D 打印版画工作流中的**独立实现**。在方案设计阶段参考过
[Lumina-Vec](https://github.com/MESFET5198/Lumina-Vec) 公开的转换**思路**（分色带、
mean-shift 平滑、共享边界、线稿叠加、打印可打印性检查等），以及
`vtracer` / `Potrace` / `Real-ESRGAN` 的公开文档。这些属于算法思想与公开技术
方案，源码均为本项目独立编写，未复制该仓库代码。

代码注释中出现的 "Lumina-Vec" 字样仅用于说明思路来源与结果对标，不代表代码来源。

**分发说明**：
- 早期 v7.5 / v7.5.1 / v7.5.2 的 exe 包含上述两类第三方代码（私有仓库复制 + GPL 依赖），**禁止分发**。
- v7.5.3 / v7.5.4 / v7.5.5 已清除上述两类，但仍附带 PiDiNet 权重（仅研究许可）；
  **若用于商业分发，请改用 v7.5.6 及以后版本**。
- 当前可分发版本：**v7.5.6**（运行时零外部检测模型、零复制代码、零 GPL 依赖）。
