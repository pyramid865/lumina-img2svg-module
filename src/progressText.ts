/**
 * Progress line text for the module card.
 *
 * Two facts shaped this:
 *
 *   1. The backend spends almost all of its time in the tracing step, and the
 *      host only reports coarse 3-phase progress, so the line can sit on "1/3"
 *      for a minute without changing.
 *   2. Cost is driven by detail as much as by pixel count: a 1192x1913 drawing
 *      took 64.5 s in our measurements while a 1280x1864 one took 27.2 s. So a
 *      size-only warning (see `largeImageHint`) misses the small-but-detailed
 *      case entirely.
 *
 * A ticking "已用 N 秒" costs nothing, cannot be wrong, and works for every
 * image: once the number keeps moving the user knows it is still working.
 */
const PHASE_TEXT: Record<string, string> = {
  preparing: "准备",
  tracing: "描摹",
  "fallback applied": "备用引擎",
  "writing outputs": "写文件",
  done: "完成",
};

/** After this many seconds the line adds "still working, do not close". */
export const SLOW_HINT_SECONDS = 30;

export function phaseLabel(phase: string | undefined): string {
  if (!phase) return "描摹中";
  return PHASE_TEXT[phase] ?? phase;
}

export function describeProgress(
  phase: string | undefined,
  completed: number | undefined,
  total: number | undefined,
  elapsedSeconds: number,
): string {
  const step = `${completed ?? 0}/${total ?? 3}`;
  const seconds = Math.floor(elapsedSeconds);
  // The finished state is not "still working": say how long it took instead of
  // asking the user to keep waiting.
  if (phase === "done") {
    return seconds >= 1 ? `完成 ${step} · 用时 ${seconds} 秒` : `完成 ${step}`;
  }
  const parts = [`${phaseLabel(phase)} ${step}`];
  if (elapsedSeconds >= 1) parts.push(`已用 ${seconds} 秒`);
  if (elapsedSeconds >= SLOW_HINT_SECONDS) parts.push("还在算，别关窗口");
  return parts.join(" · ");
}
