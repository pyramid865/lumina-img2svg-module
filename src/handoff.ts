import type { WorkshopImageHandoff } from "@lumina/workshop-sdk";
import { MODULE_ID, MODULE_VERSION } from "./version.ts";

// Conservative print size at 300 dpi; the host re-validates everything we send.
const MM_PER_PIXEL = 25.4 / 300;

/**
 * One conversion result. The projectId is generated exactly once, when the
 * conversion finishes, and is reused for every handoff of this same result.
 * A brand-new conversion produces a brand-new ConversionResult (and id).
 */
export interface ConversionResult {
  projectId: string;
  pngBytes: ArrayBuffer;
  svgBytes: ArrayBuffer | null;
  pixelWidth: number;
  pixelHeight: number;
  inputName: string;
}

export function newConversion(input: {
  pngBytes: ArrayBuffer;
  svgBytes: ArrayBuffer | null;
  pixelWidth: number;
  pixelHeight: number;
  inputName: string;
}): ConversionResult {
  return {
    ...input,
    // The host identifies "confirm replace" by the same moduleId + projectId.
    // Generate it once per conversion, never per handoff click.
    projectId: crypto.randomUUID(),
  };
}

export interface HandoffContext {
  projectSchema: string;
  renderSchema: string;
  colorLibraryId: string | null;
}

/**
 * Build the SDK request payload for one handoff click.
 *
 * The SDK puts pngBytes/svgBytes into postMessage's transfer list, which
 * DETACHES them. Therefore we must copy from the stored result at the send
 * boundary, every single time, otherwise the second click transfers an
 * already-detached buffer. We also reuse result.projectId so a confirmation
 * retry is recognised by the host.
 */
export function buildHandoffRequest(
  result: ConversionResult,
  ctx: HandoffContext,
): WorkshopImageHandoff {
  const pngBytes = result.pngBytes.slice(0);
  const svgBytes = result.svgBytes ? result.svgBytes.slice(0) : undefined;
  return {
    moduleId: MODULE_ID,
    moduleVersion: MODULE_VERSION,
    projectId: result.projectId,
    pngBytes,
    ...(svgBytes ? { svgBytes } : {}),
    pixelWidth: result.pixelWidth,
    pixelHeight: result.pixelHeight,
    recommendedWidthMm: Math.round(result.pixelWidth * MM_PER_PIXEL * 100) / 100,
    recommendedHeightMm: Math.round(result.pixelHeight * MM_PER_PIXEL * 100) / 100,
    preserveCanvasBounds: true,
    colorLibraryId: ctx.colorLibraryId,
    recipeSource: {
      manifestSchemaVersion: 1,
      moduleId: MODULE_ID,
      moduleVersion: MODULE_VERSION,
      projectSchemaVersion: ctx.projectSchema,
      renderSchemaVersion: ctx.renderSchema,
      payload: { source: result.inputName, engine: "lumina", colors: 256, hdScale: 2 },
    },
  };
}
