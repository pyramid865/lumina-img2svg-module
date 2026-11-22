#!/usr/bin/env python3
"""Sandbox run of backend/main.py without Lumina and without the host SDK.

Builds a 96x64 two-colour PNG, stubs `lumina_workshop_backend` with the pieces
`backend/main.py` uses (Backend.operation, TaskContext, TaskFailure), then calls
the real `image_to_svg` so the actual tracing pipeline runs.

Usage: python scripts/sandbox_backend.py [--keep]
Exit code 0 + a JSON summary means the packaged backend still works.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"


class TaskFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _Input:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read_bytes(self) -> bytes:
        return self._data


class TaskContext:
    """Minimal stand-in for the host-provided task context."""

    def __init__(self, inputs: dict[str, bytes]) -> None:
        self.inputs = {name: _Input(data) for name, data in inputs.items()}
        self.outputs: dict[str, tuple[str, str, bytes]] = {}
        self.progress_log: list[tuple[int, int, str]] = []
        self.cancelled = False

    def progress(self, completed: int, total: int, phase: str) -> None:
        self.progress_log.append((completed, total, phase))

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise TaskFailure("cancelled", "Task cancelled.")

    def write_output(self, name: str, file_name: str, mime_type: str, data: bytes) -> None:
        self.outputs[name] = (file_name, mime_type, data)


class Backend:
    def __init__(self) -> None:
        self.operations: dict[str, object] = {}

    def operation(self, name: str):
        def register(fn):
            self.operations[name] = fn
            return fn

        return register

    def serve(self) -> None:  # pragma: no cover - never called in the sandbox
        raise SystemExit("sandbox stub does not serve")


def install_stub() -> None:
    module = types.ModuleType("lumina_workshop_backend")
    module.Backend = Backend
    module.TaskContext = TaskContext
    module.TaskFailure = TaskFailure
    sys.modules["lumina_workshop_backend"] = module


def build_sample() -> bytes:
    """96x64 red/blue flat artwork, the shape the host review used."""
    from PIL import Image

    image = Image.new("RGBA", (96, 64), (255, 255, 255, 255))
    pixels = image.load()
    for y in range(10, 54):
        for x in range(8, 48):
            pixels[x, y] = (196, 32, 38, 255)
        for x in range(48, 88):
            pixels[x, y] = (28, 62, 158, 255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    install_stub()
    sys.path.insert(0, str(BACKEND_DIR))
    sys.path.insert(0, str(BACKEND_DIR / "img2svg_core"))

    import main as module_main  # backend/main.py

    context = TaskContext({"image": build_sample()})
    module_main.image_to_svg(context)

    missing = {"svg", "png", "meta"} - set(context.outputs)
    if missing:
        raise SystemExit(f"missing outputs: {sorted(missing)}")

    svg_name, svg_mime, svg_bytes = context.outputs["svg"]
    png_name, png_mime, png_bytes = context.outputs["png"]
    meta_name, meta_mime, meta_bytes = context.outputs["meta"]
    meta = json.loads(meta_bytes)

    svg_text = svg_bytes.decode("utf-8", errors="replace")
    checks = {
        "svg_is_svg": svg_text.lstrip().startswith("<svg") or "<svg" in svg_text[:400],
        "svg_bytes": len(svg_bytes),
        "png_bytes": len(png_bytes),
        "meta": meta,
        "progress": context.progress_log,
        "outputs": {
            "svg": [svg_name, svg_mime],
            "png": [png_name, png_mime],
            "meta": [meta_name, meta_mime],
        },
    }

    if not checks["svg_is_svg"]:
        raise SystemExit("svg output does not look like SVG")
    if checks["svg_bytes"] <= 0 or checks["png_bytes"] <= 0:
        raise SystemExit("empty output")
    if meta.get("width", 0) <= 0 or meta.get("height", 0) <= 0:
        raise SystemExit(f"bad meta: {meta}")

    if "--keep" in sys.argv:
        out_dir = Path(tempfile.mkdtemp(prefix="img2svg-sandbox-"))
        (out_dir / "result.svg").write_bytes(svg_bytes)
        (out_dir / "preview.png").write_bytes(png_bytes)
        checks["kept_at"] = str(out_dir)

    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
