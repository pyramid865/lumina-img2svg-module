import { test } from "node:test";
import assert from "node:assert/strict";
import {
  isHostSafeFileName,
  safeTaskFileName,
  safeTaskMimeType,
} from "../src/taskInput.ts";

// Mirrors the host validator in Lumina `core/workshop/backend/runtime.py`.
const HOST_FILE_NAME = /^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$/;

function assertHostAccepts(name) {
  assert.match(name, HOST_FILE_NAME, `宿主会拒绝: ${name}`);
  assert.ok(!/[. ]$/.test(name), `宿主会拒绝以点或空格结尾: ${name}`);
  assert.ok(name.length <= 128, `文件名过长: ${name}`);
}

test("ASCII 文件名原样提交，保持既有行为", () => {
  for (const name of ["witch.png", "photo 01.jpg", "a_b-c.webp", "IMG_0001.JPEG"]) {
    assert.equal(safeTaskFileName(name, "image/png"), name);
  }
});

test("中文文件名会被规范成宿主可接受的 ASCII 名", () => {
  const cases = [
    "微信图片_20261122193000.png",
    "截图 2026-11-22 193000.png",
    "未标题-1.jpg",
    "魔女の宅急便.webp",
    "测试.png",
    "图.png",
  ];
  for (const name of cases) {
    const safe = safeTaskFileName(name, "image/png");
    assertHostAccepts(safe);
    assert.ok(!/[^\x20-\x7e]/.test(safe), `仍含非 ASCII 字符: ${safe}`);
  }
});

test("特殊字符、结尾点和空格不会漏过宿主校验", () => {
  const cases = [
    "a#1.png",
    "a+b(2).png",
    "tab\tname.png",
    "trailingdot.png.",
    "trailingspace.png ",
    "  .png",
    "🎨art.png",
    "a".repeat(200) + ".png",
    "中文".repeat(80) + ".png",
  ];
  for (const name of cases) {
    const safe = safeTaskFileName(name, "image/png");
    assertHostAccepts(safe);
    assert.ok(safe.startsWith("image") || /^[A-Za-z0-9]/.test(safe));
  }
});

test("扩展名按 MIME 或原名保留，不丢格式线索", () => {
  assert.equal(safeTaskFileName("图片.png", "image/jpeg"), "image.jpg");
  assert.equal(safeTaskFileName("图片.webp", "image/webp"), "image.webp");
  assert.ok(safeTaskFileName("插画.jpg", "image/jpeg").endsWith(".jpg"));
  assert.ok(safeTaskFileName("插画", "image/png").endsWith(".png"));
});

test("空或超长 MIME 会按扩展名兜底", () => {
  assert.equal(safeTaskMimeType("", "a.png"), "image/png");
  assert.equal(safeTaskMimeType("   ", "a.jpg"), "image/jpeg");
  assert.equal(safeTaskMimeType("", "a.webp"), "image/webp");
  assert.equal(safeTaskMimeType("image/jpeg", "a.png"), "image/jpeg");
  assert.equal(safeTaskMimeType("x".repeat(200), "a.jpeg"), "image/jpeg");
});

test("宿主安全判定与校验规则一致", () => {
  assert.equal(isHostSafeFileName("witch.png"), true);
  assert.equal(isHostSafeFileName("图片.png"), false);
  assert.equal(isHostSafeFileName("a b.png"), true);
  assert.equal(isHostSafeFileName("a.png."), false);
  assert.equal(isHostSafeFileName("a.png "), false);
  assert.equal(isHostSafeFileName("_a.png"), false);
});

test("随机文件名（含中文/符号/emoji）规范后都能过宿主校验", () => {
  const pool = [
    ..."中文图片截图微信未标题魔女宅急便测试艺术",
    ..."abcdefXYZ0123456789",
    ..."._- ()#+&%[]{}'~!@$^,",
    "🎨", "🖼️", "　", "\t", "\n",
  ];
  let seed = 20261122;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const exts = ["", ".png", ".jpg", ".jpeg", ".webp", ".PNG", ".bmp", "."];
  const mimes = ["image/png", "image/jpeg", "image/webp", "", "  ", "x".repeat(140)];
  for (let i = 0; i < 500; i += 1) {
    const length = 1 + Math.floor(rand() * 40);
    let name = "";
    for (let c = 0; c < length; c += 1) {
      name += pool[Math.floor(rand() * pool.length)];
    }
    name += exts[Math.floor(rand() * exts.length)];
    const mime = mimes[Math.floor(rand() * mimes.length)];
    const safeName = safeTaskFileName(name, mime);
    assertHostAccepts(safeName);
    const safeMime = safeTaskMimeType(mime, name);
    assert.ok(safeMime.length > 0 && safeMime.length <= 127, `MIME 非法: ${safeMime}`);
    if (isHostSafeFileName(name)) assert.equal(safeName, name, "安全名不应被改写");
  }
});
