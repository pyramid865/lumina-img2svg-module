/**
 * Read the pixel size out of the bytes we were handed, without decoding them.
 *
 * The host only offers PNG / JPEG / WebP, so the three header parsers below are
 * enough. Knowing the size lets the UI warn about a large artwork *before* the
 * backend spends minutes tracing it — the progress bar alone cannot tell the
 * user whether it is working or stuck.
 */

export interface ImageSize {
  width: number;
  height: number;
}

/** Long edge above which tracing takes minutes rather than seconds. */
export const LARGE_IMAGE_EDGE = 2000;

type Bytes = ArrayBuffer | ArrayBufferView;

function viewOf(bytes: Bytes): DataView {
  if (bytes instanceof ArrayBuffer) return new DataView(bytes);
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function readPng(view: DataView): ImageSize | null {
  if (view.byteLength < 24) return null;
  if (view.getUint32(0) !== 0x89504e47) return null;
  if (view.getUint32(4) !== 0x0d0a1a0a) return null;
  return { width: view.getUint32(16), height: view.getUint32(20) };
}

function readJpeg(view: DataView): ImageSize | null {
  if (view.byteLength < 4) return null;
  if (view.getUint16(0) !== 0xffd8) return null;
  let offset = 2;
  while (offset + 9 < view.byteLength) {
    if (view.getUint8(offset) !== 0xff) {
      offset += 1;
      continue;
    }
    const marker = view.getUint8(offset + 1);
    // SOF0..SOF15 except DHT (C4), JPG (C8) and DAC (CC) carry the frame size.
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      return { height: view.getUint16(offset + 5), width: view.getUint16(offset + 7) };
    }
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      offset += 2;
      continue;
    }
    const segment = view.getUint16(offset + 2);
    if (segment < 2) return null;
    offset += 2 + segment;
  }
  return null;
}

function readWebp(view: DataView): ImageSize | null {
  if (view.byteLength < 30) return null;
  if (view.getUint32(0) !== 0x52494646) return null; // RIFF
  if (view.getUint32(8) !== 0x57454250) return null; // WEBP
  const chunk = view.getUint32(12);
  if (chunk === 0x56503858) { // VP8X
    const width = 1 + (view.getUint8(24) | (view.getUint8(25) << 8) | (view.getUint8(26) << 16));
    const height = 1 + (view.getUint8(27) | (view.getUint8(28) << 8) | (view.getUint8(29) << 16));
    return { width, height };
  }
  if (chunk === 0x5650384c) { // VP8L
    const bits = view.getUint32(21, true);
    return { width: 1 + (bits & 0x3fff), height: 1 + ((bits >> 14) & 0x3fff) };
  }
  if (chunk === 0x56503820) { // VP8
    if (view.byteLength < 30) return null;
    return {
      width: view.getUint16(26, true) & 0x3fff,
      height: view.getUint16(28, true) & 0x3fff,
    };
  }
  return null;
}

/** Returns null when the header is missing or not one of the supported types. */
export function readImageSize(bytes: Bytes, mimeType?: string): ImageSize | null {
  let view: DataView;
  try {
    view = viewOf(bytes);
  } catch {
    return null;
  }
  const byMime = (mimeType ?? "").toLowerCase();
  const readers = byMime === "image/png"
    ? [readPng]
    : byMime === "image/jpeg"
      ? [readJpeg]
      : byMime === "image/webp"
        ? [readWebp]
        : [readPng, readJpeg, readWebp];
  for (const reader of readers) {
    const size = reader(view);
    if (size !== null && size.width > 0 && size.height > 0) return size;
  }
  return null;
}

/**
 * Warning shown next to the task status for large artwork. Measured on the
 * real backend: 2000x3586 takes ~70 s, 1200x1900 takes 25-65 s depending on
 * how much detail the drawing has, so anything past 2000 px can be minutes.
 */
export function largeImageHint(width: number, height: number): string | null {
  const longEdge = Math.max(width, height);
  if (longEdge <= LARGE_IMAGE_EDGE) return null;
  return `大图 ${width}×${height}，描摹可能要几分钟，进度不动是在算，别关窗口；`
    + `想快可以先把长边缩到 ${LARGE_IMAGE_EDGE} 再转`;
}
