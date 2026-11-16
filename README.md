# community.img2svg — Lumina Workshop 模块

把桌面版「图片转 SVG」**v7.6.1** 的默认画质链路封装成 Workshop 后端模块
（manifest v2 / Workshop API 1.1.0）。v0.1.2 起将 v7.6.1 的高细节描摹参数
（case7：color_precision=8 / layer_difference=8 / corner_threshold=86）backport
到模块，实测高细节插画的发丝、镜框、暗部明显更清晰，且全程零模型、零二进制。

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

**当前状态**：模块包 v0.1.2 已发布到 Release v0.1.2；Lumina-Workshop-Registry
登记 PR（#9，v0.1.1）已合并，v0.1.2 作为新版本单独登记。

## 结构

```
manifest.json           manifest v2，锁 SDK 1.1.0，后端入口 backend/main.py
package.json / pnpm-lock.yaml   锁定 SDK 1.1.0 的不可变 Release tarball
src/main.ts             前端；esbuild 打包成自包含 ui/index.html
scripts/package.mjs     把 JS、CSS、图标全部内联进 ui/index.html
scripts/verify_parity.py        A/B 逐字节比对 + 放大对比图
test/                   模块契约测试
backend/main.py         任务入口：适配宿主任务协议，输出 SVG / PNG / meta
backend/svg_raster.py   纯 Pillow 的 vtracer SVG 光栅化（PNG 兼容回退）
backend/img2svg_core/   27 个算法核心 .py，从桌面版 v7.6.0 原样复制
backend/pylock.toml     PEP 751 依赖锁（4 个官方 wheel，全部真实哈希）
backend/MODELS.md       说明为何零模型依赖，附 A/B 实测数据
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
