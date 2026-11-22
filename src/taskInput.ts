/**
 * Host-safe naming for backend task inputs.
 *
 * Before a task is queued the host validates every input
 * (Lumina `core/workshop/backend/runtime.py`, `submit()`):
 *
 *   - input name: `^[a-z0-9][a-z0-9._-]{0,63}$`
 *   - file name:  `^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$`, no trailing "." or " "
 *   - mime type:  non-empty, at most 127 characters
 *
 * The host passes the picked file name through verbatim
 * (`electron/workshop-file-picker.cjs`: `name: path.basename(filePath)`),
 * so a Chinese file name — `微信图片_20261122.png`, `截图 1.png`, `未标题-1.png`
 * — fails the whole submit with "Workshop backend task input is invalid."
 * before the tracer ever starts. The user only sees a failed conversion.
 *
 * We therefore submit an ASCII-safe name and keep the original name for the
 * handoff recipe, so nothing user-visible is lost and the failure cannot
 * happen again.
 */
const SAFE_FILE_NAME = /^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$/;
const MAX_FILE_NAME_LENGTH = 128;

const EXTENSION_BY_MIME: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/webp": "webp",
};

const KNOWN_EXTENSIONS = new Set(["png", "jpg", "jpeg", "webp"]);

function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  if (dot <= 0) return "";
  const ext = name.slice(dot + 1).toLowerCase();
  return KNOWN_EXTENSIONS.has(ext) ? ext : "";
}

/** True when the host would accept this file name as-is. */
export function isHostSafeFileName(name: string): boolean {
  return SAFE_FILE_NAME.test(name) && !/[. ]$/.test(name);
}

/**
 * File name the host accepts: ASCII letters/digits/`.`/`_`/space/`-`, starting
 * with a letter or digit, ending with the extension, at most 128 characters.
 * ASCII names pass through untouched so the common case stays byte-identical.
 */
export function safeTaskFileName(name: string, mimeType: string): string {
  if (isHostSafeFileName(name)) return name;

  const ext = EXTENSION_BY_MIME[mimeType] || extensionOf(name) || "png";
  const dot = name.lastIndexOf(".");
  const rawStem = dot > 0 ? name.slice(0, dot) : name;
  const budget = MAX_FILE_NAME_LENGTH - ext.length - 1;
  let stem = rawStem
    .replace(/[^A-Za-z0-9._ -]/g, "-")
    .replace(/^[^A-Za-z0-9]+/, "")
    .replace(/[. ]+$/, "");
  if (stem.length > budget) {
    stem = stem.slice(0, budget).replace(/[. ]+$/, "");
  }
  if (stem.length === 0) stem = "image";

  const candidate = `${stem}.${ext}`;
  return isHostSafeFileName(candidate) ? candidate : `image.${ext}`;
}

/**
 * MIME type the host accepts. The picker only offers PNG/JPEG/WebP, but an
 * empty or over-long value fails the same input check, so fall back to the
 * extension when the host leaves it blank.
 */
export function safeTaskMimeType(mimeType: string, name: string): string {
  const trimmed = (mimeType ?? "").trim();
  if (trimmed.length > 0 && trimmed.length <= 127) return trimmed;
  const ext = extensionOf(name);
  if (ext === "jpg" || ext === "jpeg") return "image/jpeg";
  if (ext === "webp") return "image/webp";
  return "image/png";
}
