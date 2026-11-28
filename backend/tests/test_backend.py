"""Backend contract tests that need no heavy tracing dependencies."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"


class BackendContractTests(unittest.TestCase):
    def test_tracing_core_matches_desktop_source(self) -> None:
        core = BACKEND / "img2svg_core" / "core.py"
        self.assertTrue(core.is_file())
        text = core.read_text(encoding="utf-8")
        # 算法源码取自桌面 v7.26.0；工坊模块只把 linefirst 的档位常量
        # 调回 v7.24（见下一条测试），核心代码本身不裁剪。
        self.assertIn('APP_VERSION = "v7.26.0"', text)
        self.assertIn("corner_threshold=86", text)
        self.assertTrue((BACKEND / "img2svg_core" / "linefirst.py").is_file())
        # Files removed by the desktop compliance rewrite must not reappear.
        for gone in ("lum.py", "lines.py", "detail_guided.py", "cleanup.py",
                     "postfilter.py", "printcheck.py", "shared_edge.py"):
            self.assertFalse((BACKEND / "img2svg_core" / gone).exists(),
                             f"已删除的旧文件又出现了: {gone}")

    def test_default_settings_use_linefirst(self) -> None:
        text = (BACKEND / "main.py").read_text(encoding="utf-8")
        for expected in ["mode=core.MODE_LINEFIRST", "faithful_hd=True",
                         "hd=True", "hd_scale=2.0"]:
            self.assertIn(expected, text, f"缺少默认参数: {expected}")
        self.assertNotIn("MODE_LUMINA", text,
                         "默认链路已换成 linefirst，不应再引用旧引擎")

    def test_linefirst_params_frozen_at_v724(self) -> None:
        """工坊版按桌面 v7.24 行为发布：160 档、48 细节色、无线带、无体检。"""
        text = (BACKEND / "img2svg_core" / "linefirst.py").read_text(encoding="utf-8")
        expected = {
            "BANDS": "160",
            "BANDS_DETAIL": "48",
            "EMIT_SOLID_BAND": "0",
            "SOFT_BG_ENABLED": "0",
            "EMIT_LINE_LAYER": "0",
            "EMIT_PRINTCHECK": "0",
        }
        for name, value in expected.items():
            self.assertIsNotNone(
                re.search(r"^%s = %s\b" % (name, value), text, re.M),
                f"{name} 应为 {value}（v7.24 行为）")

    def test_fallback_does_not_use_native_potrace(self) -> None:
        text = (BACKEND / "main.py").read_text(encoding="utf-8")
        for forbidden in ["find_potrace", "_potrace_path", "MODE_PTRACE", "import potrace"]:
            self.assertNotIn(forbidden, text)
        self.assertIn("_fallback_vtracer", text)
        self.assertIn("import vtracer", text)

    def test_outputs_are_svg_png_and_meta(self) -> None:
        text = (BACKEND / "main.py").read_text(encoding="utf-8")
        for name in ['"svg"', '"png"', '"meta"']:
            self.assertIn(name, text)
        # PNG must be the traced artwork, not the source photo.
        self.assertIn("from svg_raster import render", text)
        self.assertIn("preview = render(svg_text)", text)
        self.assertNotIn("rgba.save(png_out", text)

    def test_backend_has_no_native_binaries(self) -> None:
        allowed = {".py", ".toml", ".json", ".txt", ".md", ".cfg", ".ini", ".csv"}
        for p in BACKEND.rglob("*"):
            if "tests" in p.parts or "__pycache__" in p.parts or ".bak" in p.name:
                continue
            if p.is_file():
                self.assertIn(p.suffix, allowed, f"不允许的文件: {p}")
                self.assertFalse(p.name.startswith("."), f"隐藏文件: {p}")

    def test_backend_sources_have_no_local_paths(self) -> None:
        bad = re.compile(r"(?:[CDFcdf]):[\\/]|ASUS|DSH-Work|deepseek-harness")
        for p in BACKEND.rglob("*"):
            if "tests" in p.parts or "__pycache__" in p.parts or ".bak" in p.name:
                continue
            if p.is_file() and p.suffix in {".py", ".toml", ".md", ".json", ".txt"}:
                self.assertIsNone(bad.search(p.read_text(encoding="utf-8")),
                                  f"含本机路径: {p}")

    def test_manifest_declares_no_models_for_zero_model_build(self) -> None:
        m = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["backend"]["python"], "3.12")
        self.assertEqual(m["backend"]["lock"], "backend/pylock.toml")
        self.assertNotIn("models", m["backend"],
                         "linefirst 链路零模型依赖，不应声明 models[]")
        perms = {p["name"] for p in m["permissions"]}
        self.assertEqual(perms, {"image.pick", "color-library.read", "handoff.image"})

    def test_lock_entries_have_real_hashes(self) -> None:
        text = (BACKEND / "pylock.toml").read_text(encoding="utf-8")
        self.assertIn("93e9b3e9fb9bc99318bfa675b00a6aa7d7b5b0cefa2fe67a781bf60d65aaa972", text)
        self.assertIn("b0f08b66734e41872d4ac343ed6d08870b3235346def3e112e10b3b2443e619e", text)
        # SDK / vtracer / numpy / pillow / opencv-headless / scipy
        self.assertEqual(len(re.findall(r"\[\[packages\]\]", text)), 6)
        self.assertEqual(len(re.findall(r"\[\[packages\.wheels\]\]", text)), 6)
        self.assertIn('name = "opencv-python-headless"', text)
        self.assertIn('name = "scipy"', text)
        # size must not be invented when unknown
        self.assertNotIn("size = 0", text)

    def test_svg_raster_handles_vtracer_transforms(self) -> None:
        raster = (BACKEND / "svg_raster.py").read_text(encoding="utf-8")
        self.assertIn("_TRANSLATE_RE", raster)
        self.assertIn("translate(", raster)
        self.assertIn("parse_subpaths", raster)
        # linefirst 的 viewBox 是工作网格，输出尺寸必须取 width/height
        self.assertIn("sx = width / vb_w", raster)
        self.assertIn("sy = height / vb_h", raster)


if __name__ == "__main__":
    unittest.main()


class ManifestEncodingTests(unittest.TestCase):
    def test_manifest_is_lf_and_matches_packaged_bytes(self) -> None:
        """v0.1.6 教训：清单必须是 LF。

        宿主（2.0.1-nightly）会把 CRLF 的 manifest.json 判成
        「package is invalid」，而错误信息只给一句笼统文案，
        排查了很久。这里钉死行尾，防止再次被 Windows 编辑器改成 CRLF。
        """
        raw = (ROOT / "manifest.json").read_bytes()
        self.assertNotIn(b"\r\n", raw, "manifest.json 必须是 LF 行尾")
        self.assertEqual(raw.count(b"\n"), raw.count(b"\n"))


class WorkCanvasTests(unittest.TestCase):
    """v0.1.7：窄长图不能把工作网格放大到吃掉几 GB 内存。"""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(BACKEND))
        sys.path.insert(0, str(BACKEND / "img2svg_core"))
        import linefirst  # noqa: PLC0415
        cls.lf = linefirst

    def test_normal_images_keep_the_old_work_grid(self) -> None:
        """正常图仍按宽度 1600 等比换算，输出/画质不变。"""
        cases = [
            (683, 916, (1600, 2146)),
            (1254, 1881, (1600, 2400)),
            (2000, 3586, (1600, 2869)),
        ]
        for ow, oh, expected in cases:
            self.assertEqual(self.lf.work_canvas_size(ow, oh), expected)

    def test_extreme_aspect_ratio_is_capped(self) -> None:
        """16×1600 若按旧逻辑会变成 1600×160000（2.56 亿像素）。"""
        w, h = self.lf.work_canvas_size(16, 1600)
        self.assertLessEqual(w, self.lf.WORK_W)
        self.assertLessEqual(h, self.lf.MAX_WORK_H)
        self.assertLessEqual(w * h, self.lf.MAX_WORK_PIXELS)
        # 等比：宽高比偏差不超过 5%
        self.assertAlmostEqual(w / float(h), 16 / 1600.0, delta=0.0005)

    def test_trace_intercepts_before_huge_allocation(self) -> None:
        """真实 trace 入口在调用 cv2.resize 前就已被收口。"""
        from PIL import Image  # noqa: PLC0415

        seen: dict[str, tuple[int, int]] = {}

        def fake_resize(_array, size, *_args, **_kwargs):
            seen["size"] = tuple(size)
            raise RuntimeError("stop-before-allocation")

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "narrow.png"
            Image.new("RGB", (16, 1600), (180, 30, 30)).save(src)
            with mock.patch.object(self.lf.cv2, "resize", side_effect=fake_resize):
                with self.assertRaises(RuntimeError):
                    self.lf.trace(str(src), str(Path(td) / "out.svg"))

        self.assertIn("size", seen)
        w, h = seen["size"]
        self.assertLessEqual(w, self.lf.WORK_W)
        self.assertLessEqual(h, self.lf.MAX_WORK_H)
        self.assertLessEqual(w * h, self.lf.MAX_WORK_PIXELS)

    def test_root_size_is_pinned_to_source_pixels(self) -> None:
        """工作网格取整不能让 SVG 显示尺寸变成 1598 这类偏差。"""
        svg = '<svg width="41" height="1598" viewBox="0 0 41 4096"></svg>'
        out = self.lf._force_root_px(svg, 16, 1600)
        self.assertIn('width="16"', out)
        self.assertIn('height="1600"', out)

    def test_palette_is_reproducible(self) -> None:
        """cv2.kmeans 自带 RNG，必须显式钉死，否则同图两次结果不同。"""
        import numpy as np  # noqa: PLC0415

        lab = np.random.default_rng(1).random((1200, 3)).astype(np.float32)
        self.lf.cv2.setRNGSeed(111)
        first = self.lf._fit_palette(lab, 8)
        self.lf.cv2.setRNGSeed(999)
        second = self.lf._fit_palette(lab, 8)
        self.assertIsNotNone(first)
        self.assertTrue(np.array_equal(first, second))


class SvgRasterPaintTests(unittest.TestCase):
    """预览渲染器必须理解 linefirst 的输出结构。

    linefirst 把打印用的深色带写成
    ``<g fill="#2a2118" fill-rule="evenodd"><path d="…"/></g>``：
    ``fill`` 与 ``fill-rule`` 都在分组上，``path`` 自身没有任何绘制属性。
    只扫描 ``<path>`` 的渲染器会把这一层整条丢掉——预览图因此没有勾边，
    和浏览器打开的真实效果对不上（2026-11-28 实测：4577 条 path 里恰好
    这 1 条没有 fill，预览缺少全部轮廓线）。
    """

    def setUp(self) -> None:
        sys.path.insert(0, str(BACKEND))
        from backend import svg_raster  # noqa: PLC0415

        self.sr = svg_raster

    def test_group_fill_is_inherited_by_path(self) -> None:
        svg = (
            '<svg width="40" height="40" viewBox="0 0 40 40">'
            '<g fill="#2a2118"><path d="M0 0 L20 0 L20 20 L0 20 Z"/></g>'
            "</svg>"
        )
        img = self.sr.render(svg)
        self.assertEqual(img.getpixel((10, 10)), (0x2A, 0x21, 0x18))

    def test_path_fill_overrides_group(self) -> None:
        svg = (
            '<svg width="40" height="40" viewBox="0 0 40 40">'
            '<g fill="#000000"><path fill="#ff0000" d="M0 0 L20 0 L20 20 L0 20 Z"/></g>'
            "</svg>"
        )
        img = self.sr.render(svg)
        self.assertEqual(img.getpixel((10, 10)), (255, 0, 0))

    def test_nested_groups_inherit_nearest(self) -> None:
        svg = (
            '<svg width="40" height="40" viewBox="0 0 40 40">'
            '<g fill="#000000"><g fill="#00ff00">'
            '<path d="M0 0 L20 0 L20 20 L0 20 Z"/></g></g>'
            "</svg>"
        )
        img = self.sr.render(svg)
        self.assertEqual(img.getpixel((10, 10)), (0, 255, 0))

    def test_evenodd_punches_hole(self) -> None:
        """evenodd 的内圈必须挖空，否则深色带会把五官糊成黑块。"""
        svg = (
            '<svg width="40" height="40" viewBox="0 0 40 40">'
            '<g fill="#2a2118" fill-rule="evenodd">'
            '<path d="M0 0 L40 0 L40 40 L0 40 Z M10 10 L30 10 L30 30 L10 30 Z"/>'
            "</g></svg>"
        )
        img = self.sr.render(svg)
        self.assertEqual(img.getpixel((5, 5)), (0x2A, 0x21, 0x18))    # ring
        self.assertEqual(img.getpixel((20, 20)), (255, 255, 255))      # hole

    def test_nested_rings_alternate(self) -> None:
        """三层嵌套：最外层填、中间挖空、最内层再填。"""
        svg = (
            '<svg width="40" height="40" viewBox="0 0 40 40">'
            '<g fill="#2a2118" fill-rule="evenodd"><path d="'
            "M0 0 L40 0 L40 40 L0 40 Z "
            "M5 5 L35 5 L35 35 L5 35 Z "
            'M15 15 L25 15 L25 25 L15 25 Z"/></g></svg>'
        )
        img = self.sr.render(svg)
        self.assertEqual(img.getpixel((2, 2)), (0x2A, 0x21, 0x18))     # 外环
        self.assertEqual(img.getpixel((10, 10)), (255, 255, 255))      # 中环挖空
        self.assertEqual(img.getpixel((20, 20)), (0x2A, 0x21, 0x18))   # 最内

    def test_group_transform_is_inherited(self) -> None:
        svg = (
            '<svg width="40" height="40" viewBox="0 0 40 40">'
            '<g fill="#0000ff" transform="translate(10,10)">'
            '<path d="M0 0 L10 0 L10 10 L0 10 Z"/></g>'
            "</svg>"
        )
        img = self.sr.render(svg)
        self.assertEqual(img.getpixel((15, 15)), (0, 0, 255))
        self.assertEqual(img.getpixel((5, 5)), (255, 255, 255))
