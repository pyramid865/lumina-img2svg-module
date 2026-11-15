#!/usr/bin/env python3
"""Build the reproducible single-file .lumina-workshop package.

The layout follows the Lumina Workshop Registry inspector (package.py):
  manifest.json, ui/index.html, assets/icon.png, README.md, LICENSE,
  backend/** (UTF-8 text sources only).
Prints the exact values the Registry source record needs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parent.parent
BACKEND_SUFFIXES = {".py", ".toml", ".json", ".txt", ".md", ".cfg", ".ini", ".csv"}
FIXED_TIME = (1980, 1, 1, 0, 0, 0)

manifest_bytes = (ROOT / "manifest.json").read_bytes()
manifest = json.loads(manifest_bytes)
asset = f'{manifest["id"]}-{manifest["version"]}.lumina-workshop'
out_dir = ROOT / "dist-package"
out = out_dir / asset

files: list[tuple[str, bytes]] = [
    ("manifest.json", manifest_bytes),
    ("ui/index.html", (ROOT / "ui/index.html").read_bytes()),
    ("assets/icon.png", (ROOT / "public/assets/icon.png").read_bytes()),
    ("README.md", (ROOT / "README.md").read_bytes()),
    ("LICENSE", (ROOT / "LICENSE").read_bytes()),
]

for path in sorted((ROOT / "backend").rglob("*")):
    if not path.is_file():
        continue
    rel_parts = path.relative_to(ROOT).parts
    if any(part.startswith(".") or part == "__pycache__" for part in rel_parts):
        continue
    if path.suffix.lower() not in BACKEND_SUFFIXES:
        continue
    files.append((path.relative_to(ROOT).as_posix(), path.read_bytes()))

def add(zf: ZipFile, name: str, data: bytes) -> None:
    info = ZipInfo(name, date_time=FIXED_TIME)
    info.create_system = 0  # FAT: inspector skips unix symlink/mode checks
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0
    zf.writestr(info, data)

out_dir.mkdir(exist_ok=True)
with ZipFile(out, "w", ZIP_DEFLATED, compresslevel=9) as zf:
    for name, data in files:
        add(zf, name, data)

result = {
    "asset": asset,
    "assetBytes": out.stat().st_size,
    "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "fileCount": len(files),
    "files": [name for name, _ in files],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
