# community.img2svg — Lumina Workshop 模块

把桌面版「图片转 SVG」**v7.24** 的「先定线后填色」链路封装成 Workshop 后端模块
（manifest v2 / Workshop API 1.1.0）。模块里的算法取自桌面 v7.26 源码，
但档位常量按 **v7.24** 固定：160 档主色、48 档细节色、不输出实心线带、不嵌位图。

> **v0.1.6 换引擎（最大的一次变化）**：描摹从旧的 vtracer 路线换成桌面 7.21 起的
> 自研 `linefirst`（先定线、后填色）。输出全部是闭合的实心纯色块，`data-purpose="3d-print"`，
> 不嵌位图、不留细线稿层，切片软件能直接分层。实测三张素材：
> 2000×3586 从 62.4 秒降到 34.8 秒、体积 8.4MB→6.6MB、路径 19480→9513；
> 边缘/整体误差（MAE）两张更好、一张持平。
> 代价：首次安装要额外下载 OpenCV 与 SciPy 两个预编译包（合计约 84 MB）。

> **v0.1.4 修复**：宿主在提交任务前要求文件名只能是 ASCII 字母、数字与 `._ -`，
> 且不能以点或空格结尾；中文名（`微信图片_*.png`、`截图 1.png`、`未标题-1.png`）
> 会被整单驳回，界面显示「描摹失败：Workshop backend task input is invalid.」，
> 描摹根本不会开始。0.1.4 在提交前自动把文件名规范成 ASCII（如 `image.png`），
> 原文件名仍用于交接溯源，这类失败不会再出现。

> **v0.1.5 修复**：描摹阶段要几十秒到几分钟，进度条只显示 1/3，用户容易以为卡死。本版做了两件事：
> ① 提交后状态行提示实际尺寸，超过 2000 像素时建议先缩图；
> ② 进度行每秒走表（「描摹 1/3 · 已用 12 秒」），超过 30 秒追加「还在算，别关窗口」，
> 完成后改成「完成 3/3 · 用时 70 秒」。计时不依赖尺寸，所以小图但细节多、同样很慢的情况也能看出来。

## 下载 / 立即使用

不想装 Python、只想双击就能把图片转成 SVG？下载**内置 Python 的本地网页版**（全自动、离线）：

- **直接下载**：[img2svg-webui-portable-v7.6.1.zip（约 31.5 MB）](https://github.com/pyramid865/lumina-img2svg-module/releases/download/v0.1.2/img2svg-webui-portable-v7.6.1.zip)
- 下载后解压，**双击 `START.bat`** 即可，浏览器会自动打开。无需安装任何东西，全程离线。
- 所有版本和文件都在 [Releases 页面](https://github.com/pyramid865/lumina-img2svg-module/releases) → **v0.1.2** 下方的 **Assets** 区域。

其他下载项：
- Workshop 模块包（最新补丁版）：`community.img2svg-0.1.2.lumina-workshop`
  （[Release v0.1.2](https://github.com/pyramid865/lumina-img2svg-module/releases/tag/v0.1.2) 的 Assets 里）。
  - v0.1.1 修复了交接时 ArrayBuffer 被转移导致二次确认失败的问题，并补齐内联 SDK 的 GPL 声明与许可证正文。
  - v0.1.2 新增画质补丁：backport 桌面 v7.6.1 的高细节描摹参数（case7），高细节插画的发丝、镜框、暗部明显更清晰，零模型、零二进制。
  - 旧版 v0.1.0 / v0.1.1 的标签与资产保持不变。

**当前状态**：Registry 里登记的后端版仍是 v0.1.1（PR #9）。v0.1.6（linefirst 引擎）
已提交追加登记 PR，等官方合并 + 签名发布后，工坊里才会出现新版本。

## 结构

```
manifest.json           manifest v2，锁 SDK 1.1.0，后端入口 backend/main.py
package.json / pnpm-lock.yaml   锁定 SDK 1.1.0 的不可变 Release tarball
src/main.ts             前端；esbuild 打包成自包含 ui/index.html
scripts/package.mjs     把 JS、CSS、图标全部内联进 ui/index.html
scripts/verify_parity.py        A/B 逐字节比对 + 放大对比图
test/                   模块契约测试
backend/main.py         任务入口：适配宿主任务协议，输出 SVG / PNG / meta
backend/svg_raster.py   纯 Pillow 的 SVG 光栅化（PNG 兼容回退），按 width/height 出图
backend/img2svg_core/   50 个算法核心 .py，取自桌面 v7.26 源码（linefirst 档位固定为 v7.24）
backend/pylock.toml     PEP 751 依赖锁（6 个官方 wheel，全部真实哈希）
backend/MODELS.md       说明依赖构成与零模型依赖的原因
THIRD_PARTY_NOTICES.md  第三方组件与许可证来源说明
```

## 效果不变（已实测，不是纸面承诺）

- `backend/img2svg_core/` 与桌面 v7.6.0 的对应文件**逐字节相同**；`core.py`
  的 sha256 两侧一致。已移除桌面版 v7.5.6 起删除的 GPL 引擎、私有源码复制、
  原生 `.pyd` 与实验脚本（含 `lum.py`、`lines.py` 等 7 个文件）。
- 默认参数与桌面 `classify.preset_for("anime")` 逐项一致：
  `MODE_LUMINA` / 256 色 / 强边缘 / `faithful_hd=True` / `hd=True` / `hd_scale=2.0`。
- **A/B 结果**：同一张原图，桌面版与模块版各产出 2,905,307 字节 / 5,912 条路径
  的 SVG，sha256 完全相同，逐字节一致；同输入重跑确定性通过；在没有 `models/`
  目录、没有 cv2/scipy/onnxruntime 的最小环境里同样一致。
  证据图：`画质一致性对比_v7.6.0.png`。
- `png` 输出是**描摹结果 SVG 的光栅化**（不是原图），用于宿主兼容回退；
  SVG 本身不受影响，仍是权威输出。
- 桌面版的 `potrace.exe` 兜底在模块侧换成 vtracer 0.6 基础档，因为规范禁止
  `backend/` 放原生可执行文件；**只在主链路异常时触发**，正常路径不受影响。

## 零模型依赖

v7.5.7 起默认 2× 超分已改为 `enhance.py` 的保色算法（纯 numpy/Pillow），
不再加载 Real-ESRGAN ONNX 权重。因此 `manifest.json` **刻意不声明**
`backend.models[]`。详见 `backend/MODELS.md`。

## 许可证与发布者（已确定）

- **模块自身代码：MIT**（Copyright (c) 2026 pyramid17）。
- **内联的 `@lumina/workshop-sdk`：GPL-3.0-or-later**
  （Copyright (C) Lumina Layer Studio contributors）。UI 是自包含单文件，
  内联了 SDK 客户端；由于构建时 `legalComments: "none"` 去除了上游声明，
  根 `LICENSE` 现同时保留作者 MIT 声明、SDK 上游声明与**完整 GPL-3.0 正文**，
  以满足所分发 SDK 的许可义务。模块自身代码与第三方依赖的来源说明见
  `THIRD_PARTY_NOTICES.md`。
- **发布者：`pyramid17`**（`manifest.json` 的 `publisher`，与 `package.json` 的
  `author` 一致）。

其余原阻塞项已解决：模型公开 URL 不再需要；`pylock.toml` 已补全为完整的
最小依赖集（4 个官方 wheel，真实 URL + sha256）；`src/main.ts` 的
`handoff.image` / `image.pick` / `backend.waitFor` 字段已按 SDK 1.1.0
的 `.d.ts` 与官方测试逐字核对修正。

## 平台范围

`manifest.json.backend.platforms` 目前只声明 `windows-x86_64`，`pylock.toml`
也只锁 Windows + CPython 3.12 的 wheel —— 这是**实际跑过 A/B 的平台**。
若要支持 Linux / macOS，需先取对应平台的官方 wheel 补进锁、扩展
`platforms`，并在那个平台重跑 A/B 比对，不能只改声明。

## 验证

```bash
pnpm install --frozen-lockfile
pnpm test
python -m unittest discover -s backend/tests -v
python scripts/verify_parity.py --help
```

## 不能提交的内容

本机路径、凭据、生成压缩包、exe、`.pyd`、模型权重二进制、含本机目录字样的
交接文档 —— 均按 SDK 仓库 `CONTRIBUTING.md` 第 5 条排除。
