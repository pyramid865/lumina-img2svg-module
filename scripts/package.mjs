import { build } from "esbuild";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const result = await build({
  entryPoints: [path.join(root, "src/main.ts")],
  bundle: true,
  format: "iife",
  target: "es2022",
  write: false,
  minify: true,
  legalComments: "none",
});

const js = result.outputFiles[0].text;
const icon = await readFile(path.join(root, "public/assets/icon.png"));
const iconData = `data:image/png;base64,${icon.toString("base64")}`;

const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>图片转 SVG</title>
<link rel="icon" href="${iconData}">
<style>
:root{color-scheme:light dark}
body{font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:24px;background:#f7f8fa;color:#1c2430}
.card{max-width:560px;margin:0 auto;background:#fff;border:1px solid #dfe3e8;border-radius:12px;padding:24px}
h1{font-size:18px;font-weight:500;margin:0 0 6px}
p.sub{color:#5f6b7a;margin:0 0 20px;font-size:13px}
button{font:inherit;padding:8px 16px;border-radius:8px;border:1px solid #c8d0da;background:#fff;color:#1c2430;cursor:pointer;margin-right:8px}
button:disabled{opacity:.45;cursor:default}
button.primary{background:#185fa5;border-color:#185fa5;color:#fff}
#status{margin-top:18px;padding:12px;background:#f1f4f8;border-radius:8px;font-size:13px;min-height:20px}
#progress{margin-top:8px;color:#5f6b7a;font-size:12px}
</style>
</head>
<body>
<div class="card">
<h1>图片转 SVG</h1>
<p class="sub">把插画转成纯路径分层 SVG，交给 Lumina 叠色 3D 打印版画。</p>
<button id="convert" class="primary" disabled>选择图片并转换</button>
<button id="handoff" disabled>交给 Lumina</button>
<div id="status">正在连接宿主…</div>
<div id="progress"></div>
</div>
<script>${js.replace(/<\/script/gi, "<\\/script")}</script>
</body>
</html>
`;

await mkdir(path.join(root, "ui"), { recursive: true });
await writeFile(path.join(root, "ui/index.html"), html, "utf8");
console.log(`ui/index.html written (${html.length} bytes, self-contained)`);
