"""Backend contract tests that need no heavy tracing dependencies."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

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
