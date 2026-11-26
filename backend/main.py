"""Workshop backend for the img2svg linefirst tracer (desktop v7.24 behaviour).

The tracing code under ``img2svg_core`` is copied from the desktop project
(v7.26 source) with the linefirst constants pinned to v7.24: 160 colour bands,
48 detail colours, no solid outline band, no embedded bitmap, no print check.
That is the last desktop build before the 56-band / 2px-band experiment, and it
still emits pure closed colour blocks, which is what 3D print slicing needs.

This file only adapts file-based I/O to the host task protocol and replaces the
desktop's legacy native-binary fallback with a wheel-supplied vtracer 0.6 pass,
because native executables cannot live under backend/.

Outputs:
  * ``svg`` - the layered SVG produced by the desktop linefirst pipeline.
  * ``png`` - a rasterised preview of that SVG (NOT the source photo), used by
    the host as the compatible fallback when native SVG is unavailable.
  * ``meta`` - JSON with the SVG's pixel dimensions for handoff sizing.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

from lumina_workshop_backend import Backend, TaskContext, TaskFailure

_BACKEND_DIR = Path(__file__).resolve().parent
_CORE_DIR = _BACKEND_DIR / "img2svg_core"
for p in (str(_BACKEND_DIR), str(_CORE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

backend = Backend()

def _default_settings():
    import core

    # 桌面 v7.24 的默认链路：先定线、后填色（linefirst），纯色块、不嵌位图。
    # 色阶/细节色/线带开关都在 img2svg_core/linefirst.py 顶部常量里，
    # 与桌面同源，避免两处参数各写一份。
    return core.Settings(
        mode=core.MODE_LINEFIRST,
        faithful_hd=True,
        hd=True,
        hd_scale=2.0,
    ).normalised()


def _fallback_vtracer(src: str, dst: str) -> None:
    """Compliant replacement for the desktop fallback (no native binary).

    Only runs when the primary pipeline fails.  The normal path above is
    byte-for-byte identical to the desktop build, so quality is unaffected.
    """
    import vtracer

    vtracer.convert_image_to_svg_py(
        src, dst,
        colormode="color", hierarchical="stacked", mode="spline",
        filter_speckle=4, color_precision=6, layer_difference=16,
        corner_threshold=60, length_threshold=4.0, path_precision=3,
    )


@backend.operation("image-to-svg")
def image_to_svg(task: TaskContext) -> None:
    data = task.inputs["image"].read_bytes()
    if not data:
        raise TaskFailure("empty-input", "The selected image is empty.")

    task.progress(0, 3, "preparing")

    # vtracer 0.6's Rust loader panics on non-ASCII paths with JPEG input.
    # Normalise to a PNG in an ASCII-only temporary directory (the desktop
    # build has the same workaround and it must be preserved verbatim).
    ascii_root = Path(tempfile.gettempdir()) / "lumina_svg_ascii"
    ascii_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lsvg_", dir=ascii_root) as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "source.png"
        dst = tmp_path / "result.svg"
        png_out = tmp_path / "preview.png"

        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            rgba = im.convert("RGBA")
            rgba.save(src, format="PNG")

        task.progress(1, 3, "tracing")
        degraded = False
        try:
            import core

            core.convert(str(src), str(dst), _default_settings(), lambda *_: None)
        except Exception:
            degraded = True
            _fallback_vtracer(str(src), str(dst))

        if not dst.exists() or dst.stat().st_size == 0:
            raise TaskFailure("empty-output", "Tracing produced no SVG.")

        svg_bytes = dst.read_bytes()
        svg_text = svg_bytes.decode("utf-8", errors="replace")

        # Render the traced artwork for the PNG fallback, never the source.
        # Keep it at the SVG's own size so the host's PNG/pixel-size check
        # agrees with the dimensions reported in `meta`.  core.py already caps
        # traced bitmaps at 5 million pixels, so this stays far below 64 MiB.
        from svg_raster import render

        preview = render(svg_text)
        buf = io.BytesIO()
        preview.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        # vtracer's fallback output has width/height but no viewBox; in that
        # case the rasteriser cannot know the canvas size, so prefer the PNG's
        # own dimensions to keep meta consistent with what the host validates.
        w, h = preview.size

        task.check_cancelled()
        task.progress(2, 3, "fallback applied" if degraded else "writing outputs")
        task.write_output("svg", "result.svg", "image/svg+xml", svg_bytes)
        task.write_output("png", "preview.png", "image/png", png_bytes)
        task.write_output(
            "meta", "size.json", "application/json",
            json.dumps({"width": w, "height": h, "degraded": degraded}).encode("utf-8"),
        )
        task.progress(3, 3, "done")


if __name__ == "__main__":
    backend.serve()
