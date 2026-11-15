# -*- coding: utf-8 -*-
"""Ink line layer, entirely self-developed (no external models or weights).

v7.5.3 COMPLIANCE REWRITE
-------------------------
An earlier build vendored two files from a **private** third-party repository
(Lumina-Vec ``lumina_vec/lines.py`` / ``manga_net.py``) and downloaded ~370 MB
of third-party torch weights.  That code has been removed in full.  This module
now uses only the project's own classic image-processing pipeline:

    multi-scale black-hat -> ridge test -> hysteresis thresholding
    -> connected-component cleanup -> potrace curve fitting

``line_layer.py`` implements the first three steps in-house (its module
docstring has always stated "Line layer, self-developed (no AI model)");
``line_layer.build_from_mask`` renders the mask as a top-level vector ``<g>``
via the bundled potrace executable.  No torch, no downloaded weights, no
third-party network definitions, and no copied source code are involved.

Public API (unchanged so core.py needs no rework):
    available() -> bool
    ink_map(rgb, weights_dir=None) -> float32 [0,1]
    build_line_layer(rgb, svg_text, weights_dir=None) -> str (<g> snippet or "")
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def available() -> bool:
    """The in-house line layer only needs numpy + OpenCV (both bundled)."""
    return cv2 is not None


def ink_map(rgb: np.ndarray, weights_dir: str | None = None) -> np.ndarray:
    """0..1 ink probability field from the self-developed morphology pipeline.

    ``weights_dir`` is kept for API compatibility; it is intentionally ignored
    because this implementation uses no model weights at all.
    """
    from line_layer import stroke_field

    return stroke_field(np.asarray(rgb)).astype(np.float32)


def build_line_layer(rgb: np.ndarray, svg_text: str,
                     weights_dir: str | None = None) -> str:
    """Return a ``<g>`` line-layer snippet (or "") for the detected ink."""
    P = ink_map(rgb)
    mask = (P > 0.5).astype(np.uint8)
    if not mask.any():
        return ""
    # drop isolated specks so noise does not become a stroke
    if cv2 is not None:
        try:
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            keep = np.zeros_like(mask)
            for i in range(1, n):
                if int(stats[i, cv2.CC_STAT_AREA]) >= 12:
                    keep[lab == i] = 1
            mask = keep
        except Exception:
            pass
    if not mask.any():
        return ""

    from PIL import Image
    from line_layer import build_from_mask

    return build_from_mask(Image.fromarray(np.asarray(rgb)), mask)
