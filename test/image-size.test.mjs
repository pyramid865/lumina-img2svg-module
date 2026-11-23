import { test } from "node:test";
import assert from "node:assert/strict";
import {
  LARGE_IMAGE_EDGE,
  largeImageHint,
  readImageSize,
} from "../src/imageSize.ts";

function pngHeader(width, height) {
  const bytes = Buffer.alloc(33);
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(bytes, 0);
  bytes.writeUInt32BE(0x0d, 8); // IHDR length
  bytes.write("IHDR", 12, "latin1");
  bytes.writeUInt32BE(width, 16);
  bytes.writeUInt32BE(height, 20);
  return bytes;
}

function jpegHeader(width, height) {
  const bytes = Buffer.alloc(20);
  bytes.writeUInt16BE(0xffd8, 0); // SOI
  bytes.writeUInt16BE(0xffc0, 2); // SOF0
  bytes.writeUInt16BE(0x0011, 4); // segment length
  bytes.writeUInt8(8, 6); // precision
  bytes.writeUInt16BE(height, 7);
  bytes.writeUInt16BE(width, 9);
  return bytes;
}

function webpVp8xHeader(width, height) {
  const bytes = Buffer.alloc(30);
  bytes.write("RIFF", 0, "latin1");
  bytes.writeUInt32LE(22, 4);
  bytes.write("WEBP", 8, "latin1");
  bytes.write("VP8X", 12, "latin1");
  bytes.writeUInt32LE(10, 16);
  const w = width - 1;
  const h = height - 1;
  bytes.writeUInt8(w & 0xff, 24);
  bytes.writeUInt8((w >> 8) & 0xff, 25);
  bytes.writeUInt8((w >> 16) & 0xff, 26);
  bytes.writeUInt8(h & 0xff, 27);
  bytes.writeUInt8((h >> 8) & 0xff, 28);
  bytes.writeUInt8((h >> 16) & 0xff, 29);
  return bytes;
}

test("从文件头读出尺寸：PNG / JPEG / WebP", () => {
  assert.deepEqual(readImageSize(pngHeader(3200, 4800), "image/png"), { width: 3200, height: 4800 });
  assert.deepEqual(readImageSize(jpegHeader(2000, 3586), "image/jpeg"), { width: 2000, height: 3586 });
  assert.deepEqual(readImageSize(webpVp8xHeader(1280, 1864), "image/webp"), { width: 1280, height: 1864 });
});

test("不给 MIME 时按文件头自己认", () => {
  assert.deepEqual(readImageSize(pngHeader(96, 64)), { width: 96, height: 64 });
  assert.deepEqual(readImageSize(jpegHeader(640, 480)), { width: 640, height: 480 });
});

test("支持 Uint8Array 视图（带偏移）与 ArrayBuffer", () => {
  const header = pngHeader(1000, 1500);
  const padded = Buffer.concat([Buffer.alloc(7), header]);
  const view = new Uint8Array(padded.buffer, padded.byteOffset + 7, header.length);
  assert.deepEqual(readImageSize(view, "image/png"), { width: 1000, height: 1500 });
  const copy = new Uint8Array(header).buffer;
  assert.deepEqual(readImageSize(copy, "image/png"), { width: 1000, height: 1500 });
});

test("认不出来时返回 null，不影响主流程", () => {
  assert.equal(readImageSize(Buffer.from("not an image at all"), "image/png"), null);
  assert.equal(readImageSize(Buffer.alloc(0), "image/png"), null);
  assert.equal(readImageSize(pngHeader(10, 10), "image/jpeg"), null);
});

test("只有长边超过 2000 才提示大图", () => {
  assert.equal(LARGE_IMAGE_EDGE, 2000);
  assert.equal(largeImageHint(2000, 2000), null);
  assert.equal(largeImageHint(1800, 900), null);
  assert.equal(largeImageHint(2001, 900) === null, false);
  const hint = largeImageHint(3200, 4800);
  assert.ok(hint && hint.includes("3200×4800"));
  assert.ok(hint.includes("2000"), "提示里要给出建议的长边");
  assert.ok(hint.includes("别关窗口"));
});
