import { applyWorkshopUiState, connectWorkshop } from "@lumina/workshop-sdk";

const MODULE_ID = "community.img2svg";
const MODULE_VERSION = "0.1.0";
const PROJECT_SCHEMA = "community.img2svg/v1";
const RENDER_SCHEMA = "community.img2svg-render/v1";

// Conservative print size at 300 dpi; the host re-validates everything we send.
const MM_PER_PIXEL = 25.4 / 300;

const statusEl = document.getElementById("status") as HTMLElement;
const convertBtn = document.getElementById("convert") as HTMLButtonElement;
const handoffBtn = document.getElementById("handoff") as HTMLButtonElement;
const progressEl = document.getElementById("progress") as HTMLElement;

let lastPng: ArrayBuffer | null = null;
let lastSvg: ArrayBuffer | null = null;
let lastWidth = 0;
let lastHeight = 0;
let lastInputName = "source.png";

function setStatus(text: string): void {
  statusEl.textContent = text;
}

async function main(): Promise<void> {
  const client = await connectWorkshop({
    moduleId: MODULE_ID,
    moduleVersion: MODULE_VERSION,
    features: ["backend.tasks"],
  });

  const unsubscribeUiState = client.ui.subscribeState(applyWorkshopUiState);
  applyWorkshopUiState(await client.ui.getState());
  await client.lifecycle.ready();
  setStatus("已连接宿主，等待选择图片");
  convertBtn.disabled = false;

  convertBtn.addEventListener("click", async () => {
    try {
      convertBtn.disabled = true;
      setStatus("正在选择图片…");
      // WorkshopPickedImage uses `name`, not `fileName`.
      const picked = await client.image.pick();
      if (!picked) {
        setStatus("已取消选择");
        return;
      }
      lastInputName = picked.name;
      lastWidth = picked.raster.width;
      lastHeight = picked.raster.height;

      setStatus("已提交描摹任务…");
      const task = await client.backend.submit({
        operation: "image-to-svg",
        params: {},
        inputs: [{
          name: "image",
          fileName: picked.name,
          mimeType: picked.mimeType,
          bytes: picked.bytes,
        }],
        idempotencyKey: crypto.randomUUID(),
      });

      // onProgress receives the whole WorkshopBackendTask, not a progress object.
      const finished = await client.backend.waitFor(task.taskId, {
        onProgress: (t) => {
          const p = t.progress;
          progressEl.textContent = `${p?.phase ?? "描摹中"} ${p?.completed ?? 0}/${p?.total ?? 3}`;
        },
      });
      if (finished.status === "failed") {
        throw new Error(finished.error?.code ?? "task-failed");
      }

      const svgOutput = await client.backend.readOutput(finished.taskId, "svg");
      const pngOutput = await client.backend.readOutput(finished.taskId, "png");
      const metaOutput = await client.backend.readOutput(finished.taskId, "meta");
      // The traced SVG is 2x-upscaled; use its real size, not the input's.
      const meta = JSON.parse(new TextDecoder().decode(metaOutput.bytes)) as {
        width: number; height: number; degraded: boolean;
      };
      lastWidth = meta.width;
      lastHeight = meta.height;
      // `bytes` is an ArrayBuffer; handoff transfers it, so keep a copy.
      lastSvg = svgOutput.bytes.slice(0);
      lastPng = pngOutput.bytes.slice(0);
      handoffBtn.disabled = false;
      setStatus("描摹完成，可交给 Lumina 转换器");
    } catch (err) {
      setStatus("描摹失败：" + (err instanceof Error ? err.message : "unknown-error"));
    } finally {
      convertBtn.disabled = false;
    }
  });

  handoffBtn.addEventListener("click", async () => {
    if (!lastPng) return;
    try {
      const library = await client.colorLibrary.read();
      const result = await client.handoff.image({
        moduleId: MODULE_ID,
        moduleVersion: MODULE_VERSION,
        projectId: crypto.randomUUID(),
        pngBytes: lastPng,
        ...(lastSvg ? { svgBytes: lastSvg } : {}),
        pixelWidth: lastWidth,
        pixelHeight: lastHeight,
        recommendedWidthMm: Math.round(lastWidth * MM_PER_PIXEL * 100) / 100,
        recommendedHeightMm: Math.round(lastHeight * MM_PER_PIXEL * 100) / 100,
        preserveCanvasBounds: true,
        colorLibraryId: library?.id ?? null,
        recipeSource: {
          manifestSchemaVersion: 1,
          moduleId: MODULE_ID,
          moduleVersion: MODULE_VERSION,
          projectSchemaVersion: PROJECT_SCHEMA,
          renderSchemaVersion: RENDER_SCHEMA,
          payload: { source: lastInputName, engine: "lumina", colors: 256, hdScale: 2 },
        },
      });
      setStatus(result.status === "needs-confirmation"
        ? "转换器已有工作，请确认替换后再交接一次"
        : "已交接给 Lumina 转换器");
    } catch (err) {
      setStatus("交接失败：" + (err instanceof Error ? err.message : "unknown-error"));
    }
  });

  window.addEventListener("beforeunload", () => unsubscribeUiState());
}

void main();
