import { applyWorkshopUiState, connectWorkshop } from "@lumina/workshop-sdk";
import { buildHandoffRequest, newConversion, type ConversionResult } from "./handoff";

const MODULE_ID = "community.img2svg";
const MODULE_VERSION = "0.1.1";
const PROJECT_SCHEMA = "community.img2svg/v1";
const RENDER_SCHEMA = "community.img2svg-render/v1";

const statusEl = document.getElementById("status") as HTMLElement;
const convertBtn = document.getElementById("convert") as HTMLButtonElement;
const handoffBtn = document.getElementById("handoff") as HTMLButtonElement;
const progressEl = document.getElementById("progress") as HTMLElement;

// One entry per completed conversion. projectId lives inside the result and
// is stable across all handoffs of that result; a new conversion replaces it.
let lastResult: ConversionResult | null = null;

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
      // `bytes` is an ArrayBuffer; handoff transfers its own COPY at the send
      // boundary, so these stored originals stay intact for repeated handoffs.
      lastResult = newConversion({
        pngBytes: pngOutput.bytes.slice(0),
        svgBytes: svgOutput.bytes.slice(0),
        pixelWidth: meta.width,
        pixelHeight: meta.height,
        inputName: picked.name,
      });
      handoffBtn.disabled = false;
      setStatus("描摹完成，可交给 Lumina 转换器");
    } catch (err) {
      setStatus("描摹失败：" + (err instanceof Error ? err.message : "unknown-error"));
    } finally {
      convertBtn.disabled = false;
    }
  });

  handoffBtn.addEventListener("click", async () => {
    if (!lastResult) return;
    try {
      const library = await client.colorLibrary.read();
      const request = buildHandoffRequest(lastResult, {
        moduleId: MODULE_ID,
        moduleVersion: MODULE_VERSION,
        projectSchema: PROJECT_SCHEMA,
        renderSchema: RENDER_SCHEMA,
        colorLibraryId: library?.id ?? null,
      });
      const result = await client.handoff.image(request);
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
