import { test } from "node:test";
import assert from "node:assert/strict";
import { MessageChannel } from "node:worker_threads";
import {
  WORKSHOP_RPC_VERSION,
  connectWorkshop,
} from "@lumina/workshop-sdk";
import { buildHandoffRequest, newConversion } from "../src/handoff.ts";

const MODULE_ID = "community.img2svg";
const MODULE_VERSION = "0.1.1";

function rgba(n) {
  const b = new ArrayBuffer(n);
  new Uint8Array(b).fill(7);
  return b;
}

/**
 * Spin up a real SDK client over a real MessageChannel with a scripted host.
 * The host answers `handoff.image` using `answer(status, payload)` and records
 * every payload it receives (including byteLength, to detect detached buffers).
 */
async function withClient(answer) {
  const { port1: modulePort, port2: hostPort } = new MessageChannel();
  const seen = [];

  // Minimal fake window: SDK connects via window.parent.postMessage(...);
  // we reply with a `lumina.workshop.connect` message carrying modulePort.
  const listeners = new Set();
  const parent = {
    postMessage(msg) {
      if (msg?.type === "lumina.workshop.ready") {
        for (const fn of listeners) {
          fn({
            source: parent,
            data: {
              type: "lumina.workshop.connect",
              sessionId: "test-session",
            },
            ports: [modulePort],
          });
        }
      }
    },
  };
  const fakeWindow = {
    parent,
    addEventListener(type, fn) {
      if (type === "message") listeners.add(fn);
    },
    removeEventListener(type, fn) {
      listeners.delete(fn);
    },
  };

  hostPort.onmessage = (event) => {
    const req = event.data;
    if (req?.kind !== "request" || req.method !== "handoff.image") return;
    seen.push({
      requestId: req.requestId,
      projectId: req.payload.projectId,
      pngSize: req.payload.pngBytes?.byteLength ?? -1,
      svgSize: req.payload.svgBytes?.byteLength ?? -1,
      payload: req.payload,
    });
    const out = answer(seen.length, req.payload);
    hostPort.postMessage({
      protocol: "lumina-workshop-rpc",
      version: WORKSHOP_RPC_VERSION,
      kind: "response",
      requestId: req.requestId,
      ok: true,
      result: out,
    });
  };
  hostPort.start();

  const client = await connectWorkshop({
    windowObject: fakeWindow,
    moduleId: MODULE_ID,
    moduleVersion: MODULE_VERSION,
  });
  return { client, seen, close: () => { modulePort.close(); hostPort.close(); } };
}

function ctx() {
  return {
    moduleId: MODULE_ID,
    moduleVersion: MODULE_VERSION,
    projectSchema: "community.img2svg/v1",
    renderSchema: "community.img2svg-render/v1",
    colorLibraryId: null,
  };
}

test("真实 MessagePort：首次交接→已有工作二次确认→重复交接均成功，buffer 未 detached，projectId 复用", async () => {
  const { client, seen, close } = await withClient((call) => ({
    status: call === 1 ? "needs-confirmation" : "completed",
  }));
  try {
    const result = newConversion({
      pngBytes: rgba(64),
      svgBytes: rgba(128),
      pixelWidth: 1000,
      pixelHeight: 1500,
      inputName: "sample.png",
    });

    // 三次点击：第一次需确认、第二次确认完成、第三次重复交接。
    for (let i = 0; i < 3; i++) {
      const req = buildHandoffRequest(result, ctx());
      const res = await client.handoff.image(req);
      assert.equal(i === 0 ? res.status : res.status, i === 0 ? "needs-confirmation" : "completed");
    }

    assert.equal(seen.length, 3, "三次交接都应真正到达宿主");
    for (const s of seen) {
      assert.equal(s.projectId, result.projectId, "确认重试必须复用同一 projectId");
      assert.equal(s.pngSize, 64, "到达宿主的 PNG 未 detached");
      assert.equal(s.svgSize, 128, "到达宿主的 SVG 未 detached");
    }
    // 原始 result 的 buffer 从未被转移走 —— 发送边界复制生效。
    assert.equal(result.pngBytes.byteLength, 64, "存储的原 PNG 未被 detach");
    assert.equal(result.svgBytes.byteLength, 128, "存储的原 SVG 未被 detach");
  } finally {
    close();
  }
});

test("真实 MessagePort：新转换结果换新的 projectId，旧结果仍可再次交接", async () => {
  const { client, seen, close } = await withClient(() => ({ status: "completed" }));
  try {
    const first = newConversion({
      pngBytes: rgba(16), svgBytes: rgba(32), pixelWidth: 10, pixelHeight: 20, inputName: "a.png",
    });
    const second = newConversion({
      pngBytes: rgba(16), svgBytes: rgba(32), pixelWidth: 30, pixelHeight: 40, inputName: "b.png",
    });
    assert.notEqual(first.projectId, second.projectId, "两次转换必须有不同 projectId");

    await client.handoff.image(buildHandoffRequest(first, ctx()));
    await client.handoff.image(buildHandoffRequest(second, ctx()));
    await client.handoff.image(buildHandoffRequest(first, ctx())); // 回头再交第一个结果

    assert.deepEqual(
      seen.map((s) => s.projectId),
      [first.projectId, second.projectId, first.projectId],
      "projectId 随转换结果走，不随点击次数变",
    );
    for (const s of seen) {
      assert.ok(s.pngSize > 0 && s.svgSize > 0, "所有到达宿主的 buffer 均有效");
    }
  } finally {
    close();
  }
});

test("真实 MessagePort：只有 PNG（无 SVG）也能重复交接", async () => {
  const { client, seen, close } = await withClient((call) => ({
    status: call === 1 ? "needs-confirmation" : "completed",
  }));
  try {
    const result = newConversion({
      pngBytes: rgba(48), svgBytes: null, pixelWidth: 8, pixelHeight: 9, inputName: "no-svg.png",
    });
    for (let i = 0; i < 2; i++) {
      const res = await client.handoff.image(buildHandoffRequest(result, ctx()));
      assert.equal(res.status, i === 0 ? "needs-confirmation" : "completed");
    }
    assert.equal(seen.length, 2);
    for (const s of seen) {
      assert.equal(s.pngSize, 48);
      assert.equal(s.svgSize, -1, "未发送 SVG 字段");
    }
    assert.equal(result.pngBytes.byteLength, 48);
  } finally {
    close();
  }
});
