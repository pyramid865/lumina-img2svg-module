import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

async function readJson(rel) {
  return JSON.parse(await readFile(path.join(root, rel), "utf8"));
}

test("manifest 使用 v2 且声明后端与 API 下限", async () => {
  const m = await readJson("manifest.json");
  assert.equal(m.manifestVersion, 2);
  assert.equal(m.workshopApi.min, "1.1.0");
  assert.equal(m.entrypoints.ui, "ui/index.html");
  assert.equal(typeof m.backend.entry, "string");
  assert.equal(typeof m.backend.lock, "string");
  assert.ok(m.backend.platforms.length > 0);
  assert.ok(m.permissions.length > 0);
});

test("manifest 无未知字段、无重复权限、版本号完整", async () => {
  const m = await readJson("manifest.json");
  const allowed = new Set([
    "manifestVersion", "id", "version", "name", "description", "publisher",
    "workshopApi", "luminaVersion", "entrypoints", "permissions", "backend",
  ]);
  for (const key of Object.keys(m)) assert.ok(allowed.has(key), `未知字段: ${key}`);
  const names = m.permissions.map((p) => p.name);
  assert.equal(new Set(names).size, names.length, "权限不得重复");
  for (const p of m.permissions) assert.ok(p.reason && p.reason.length > 0);
  assert.match(m.version, /^\d+\.\d+\.\d+$/);
  assert.match(m.workshopApi.min, /^\d+\.\d+\.\d+$/);
  assert.match(m.workshopApi.maxExclusive, /^\d+\.\d+\.\d+$/);
});

test("manifest 版本与 package.json 版本一致", async () => {
  const m = await readJson("manifest.json");
  const p = await readJson("package.json");
  assert.equal(m.version, p.version);
});

test("ui/index.html 自包含，不引用外部文件或 URL", async () => {
  const html = await readFile(path.join(root, "ui/index.html"), "utf8");
  assert.ok(html.includes("<script>"), "脚本必须内联");
  assert.ok(!/<script[^>]+src=/i.test(html), "不得引用外部脚本");
  assert.ok(!/<link[^>]+href=["']https?:/i.test(html), "不得引用外部样式或资源");
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
  assert.ok(scripts.length === 1 && scripts[0][1].length > 100, "必须内联一份完整脚本");
  assert.ok(!/https?:\/\//.test(scripts[0][1].replace(/\/\/[^\n]*/g, "")), "脚本内不得硬编码外部 URL");
});

test("backend 只含允许的文本源文件", async () => {
  const allowedExt = new Set([".py", ".toml", ".json", ".txt", ".md", ".cfg", ".ini", ".csv"]);
  async function walk(dir) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        // Runtime caches are generated locally and excluded by .gitignore;
        // they are not part of the committed backend package.
        if (entry.name === "tests" || entry.name === "__pycache__") continue;
        assert.ok(!entry.name.startsWith("."), `不得有隐藏目录: ${entry.name}`);
        await walk(full);
      } else {
        assert.ok(allowedExt.has(path.extname(entry.name)), `不允许的后端文件: ${entry.name}`);
        assert.ok(!entry.name.startsWith("."), `不得有隐藏文件: ${entry.name}`);
      }
    }
  }
  await walk(path.join(root, "backend"));
});

test("backend 源码不含本机路径或凭据字样", async () => {
  const bad = /(?:[CDFcdf]):[\\/]|ASUS|DSH-Work|deepseek-harness|ghp_|github_pat_/;
  async function walk(dir) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === "tests") continue;
        await walk(full);
      } else if (/\.(py|toml|json|md|txt)$/.test(entry.name)) {
        const text = await readFile(full, "utf8");
        assert.ok(!bad.test(text), `${path.relative(root, full)} 含本机路径/凭据字样`);
      }
    }
  }
  await walk(path.join(root, "backend"));
});

test("src/main.ts 使用 SDK 1.1.0 的真实字段", async () => {
  const src = await readFile(path.join(root, "src/main.ts"), "utf8");
  // WorkshopPickedImage uses `name`, not `fileName`.
  assert.ok(src.includes("picked.name"), "image.pick() 返回 name 字段");
  assert.ok(!src.includes("picked.fileName"), "不得使用不存在的 picked.fileName");
  // onProgress receives the whole task; progress lives at task.progress.
  assert.ok(src.includes("t.progress"), "waitFor 回调需读 task.progress");
  // WorkshopImageHandoff required fields.
  for (const field of ["moduleId", "moduleVersion", "projectId", "pngBytes",
                       "pixelWidth", "pixelHeight", "recommendedWidthMm",
                       "recommendedHeightMm", "preserveCanvasBounds",
                       "colorLibraryId", "recipeSource", "svgBytes"]) {
    assert.ok(src.includes(`${field}:`) || src.includes(`${field} `),
              `handoff.image 缺少字段: ${field}`);
  }
  assert.ok(!src.includes("widthPx"), "widthPx 不是 SDK 字段");
  // Read the traced SVG's real size from the backend meta output.
  assert.ok(src.includes('readOutput(finished.taskId, "meta")'), "需读取 meta 输出");
  // Avoid transferring away the only copy of the buffers.
  assert.ok(src.includes(".slice(0)"), "交接前需复制 ArrayBuffer 以免被转移");
});

test("pylock.toml 的每个包都有 HTTPS URL 与 sha256", async () => {
  const lock = await readFile(path.join(root, "backend/pylock.toml"), "utf8");
  assert.ok(lock.includes("lock-version"), "缺少 lock-version");
  const packages = lock.split(/\[\[packages\]\]/).slice(1);
  assert.ok(packages.length > 0, "锁内必须有包");
  for (const block of packages) {
    if (!block.includes("[[packages.wheels]]")) continue;
    assert.match(block, /url = "https:\/\//, "wheel 必须给出 HTTPS URL");
    assert.match(block, /sha256 = "[0-9a-f]{64}"/, "wheel 必须给出完整 sha256");
  }
});
