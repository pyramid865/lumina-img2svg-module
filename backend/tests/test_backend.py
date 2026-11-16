"""Backend contract tests that need no heavy tracing dependencies."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"


class BackendContractTests(unittest.TestCase):
    def test_tracing_core_is_current_v761(self) -> None:
        core = BACKEND / "img2svg_core" / "core.py"
        self.assertTrue(core.is_file())
        text = core.read_text(encoding="utf-8")
        self.assertIn('APP_VERSION = "v7.6.1"', text)
        # v0.1.2 backports the v7.6.1 high-detail tracing parameters.
        self.assertIn("corner_threshold=86", text)
        # Files removed by the desktop compliance rewrite must not reappear.
        for gone in ("lum.py", "lines.py", "detail_guided.py", "cleanup.py",
                     "postfilter.py", "printcheck.py", "shared_edge.py"):
            self.assertFalse((BACKEND / "img2svg_core" / gone).exists(),
                             f"已删除的旧文件又出现了: {gone}")

    def test_default_settings_match_desktop_anime_preset(self) -> None:
        text = (BACKEND / "main.py").read_text(encoding="utf-8")
        for expected in ["mode=core.MODE_LUMINA", "colors=256",
                         'edge_smooth="strong"', "faithful_hd=True",
                         "hd=True", "hd_scale=2.0"]:
            self.assertIn(expected, text, f"缺少桌面版默认参数: {expected}")

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
            if "tests" in p.parts or "__pycache__" in p.parts:
                continue
            if p.is_file():
                self.assertIn(p.suffix, allowed, f"不允许的文件: {p}")
                self.assertFalse(p.name.startswith("."), f"隐藏文件: {p}")

    def test_backend_sources_have_no_local_paths(self) -> None:
        bad = re.compile(r"(?:[CDFcdf]):[\\/]|ASUS|DSH-Work|deepseek-harness")
        for p in BACKEND.rglob("*"):
            if "tests" in p.parts or "__pycache__" in p.parts:
                continue
            if p.is_file() and p.suffix in {".py", ".toml", ".md", ".json", ".txt"}:
                self.assertIsNone(bad.search(p.read_text(encoding="utf-8")),
                                  f"含本机路径: {p}")

    def test_manifest_declares_no_models_for_zero_model_build(self) -> None:
        m = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["backend"]["python"], "3.12")
        self.assertEqual(m["backend"]["lock"], "backend/pylock.toml")
        self.assertNotIn("models", m["backend"],
                         "v7.6.1 默认链路零模型依赖，不应声明 models[]")
        perms = {p["name"] for p in m["permissions"]}
        self.assertEqual(perms, {"image.pick", "color-library.read", "handoff.image"})

    def test_lock_entries_have_real_hashes(self) -> None:
        text = (BACKEND / "pylock.toml").read_text(encoding="utf-8")
        self.assertIn("93e9b3e9fb9bc99318bfa675b00a6aa7d7b5b0cefa2fe67a781bf60d65aaa972", text)
        self.assertIn("b0f08b66734e41872d4ac343ed6d08870b3235346def3e112e10b3b2443e619e", text)
        self.assertEqual(len(re.findall(r"\[\[packages\]\]", text)), 4)
        self.assertEqual(len(re.findall(r"\[\[packages\.wheels\]\]", text)), 4)
        # size must not be invented when unknown
        self.assertNotIn("size = 0", text)

    def test_svg_raster_handles_vtracer_transforms(self) -> None:
        raster = (BACKEND / "svg_raster.py").read_text(encoding="utf-8")
        self.assertIn("_TRANSLATE_RE", raster)
        self.assertIn("translate(", raster)
        self.assertIn("parse_subpaths", raster)


if __name__ == "__main__":
    unittest.main()
