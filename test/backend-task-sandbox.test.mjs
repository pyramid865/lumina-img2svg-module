import { test } from "node:test";
import assert from "node:assert/strict";
import { MessageChannel } from "node:worker_threads";
import { WORKSHOP_RPC_VERSION, connectWorkshop } from "@lumina/workshop-sdk";
import { safeTaskFileName, safeTaskMimeType } from "../src/taskInput.ts";
import { MODULE_ID, MODULE_VERSION } from "../src/version.ts";

// ---------------------------------------------------------------------------
// A virtual Lumina host: the same validation the real host applies before a
// backend task is queued (core/workshop/backend/runtime.py `submit()`), plus a
// fake tracer that returns fixed outputs. No Lumina install, no Python, no
// network — this is the sandbox used to prove the fix end to end.
// ---------------------------------------------------------------------------

const HOST_INPUT_NAME = /^[a-z0-9][a-z0-9._-]{0,63}$/;
const HOST_FILE_NAME = /^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$/;

/** Returns the host's rejection message, or null when the input is accepted. */
function hostValidateInputs(inputs) {
  const names = new Set();
  for (const input of inputs) {
    if (
      !HOST_INPUT_NAME.test(input.name)
      || names.has(input.name)
      || !HOST_FILE_NAME.test(input.fileName)
      || input.fileName.endsWith(".")
      || input.fileName.endsWith(" ")
      || !(input.mimeType.length > 0 && input.mimeType.length <= 127)
    ) {
      return "Workshop backend task input is invalid.";
    }
    names.add(input.name);
  }
  return null;
}

const PNG_BYTES = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]).buffer;
const CHINESE_PICK = {
  name: "微信图片_20261122193000.png",
  mimeType: "image/png",
  bytes: PNG_BYTES,
};

async function withHost(handler) {
  const { port1: modulePort, port2: hostPort } = new MessageChannel();
  const listeners = new Set();
  const parent = {
    postMessage(message) {
      if (message?.type !== "lumina.workshop.ready") return;
      for (const fn of listeners) {
        fn({
          source: parent,
          data: { type: "lumina.workshop.connect", sessionId: "sandbox" },
          ports: [modulePort],
        });
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

  const calls = [];
  hostPort.onmessage = (event) => {
    const request = event.data;
    if (request?.kind !== "request") return;
    calls.push({ method: request.method, payload: request.payload });
    const reply = (ok, value) => {
      hostPort.postMessage({
        protocol: "lumina-workshop-rpc",
        version: WORKSHOP_RPC_VERSION,
        kind: "response",
        requestId: request.requestId,
        ok,
        ...(ok ? { result: value } : { error: value }),
      });
    };
    const outcome = handler(request.method, request.payload, reply);
    if (outcome !== undefined) reply(true, outcome);
  };
  hostPort.start();

  const client = await connectWorkshop({
    windowObject: fakeWindow,
    moduleId: MODULE_ID,
    moduleVersion: MODULE_VERSION,
    features: ["backend.tasks"],
  });
  return { client, calls, close: () => hostPort.close() };
}

/** Host replies for a healthy session; `fileName` decides accept vs reject. */
function healthyHost(rejectReason) {
  return (method, payload, reply) => {
    switch (method) {
      case "ui.getState":
        return { locale: "zh-CN", theme: "light", tokens: {} };
      case "lifecycle.ready":
        return {};
      case "image.pick":
        return CHINESE_PICK;
      case "backend.submit": {
        const reason = hostValidateInputs(payload.inputs);
        if (reason !== null) {
          reply(false, {
            code: "task-input-invalid",
            message: reason,
            retryable: false,
          });
          return null;
        }
        return { taskId: "task-1", status: "queued" };
      }
      case "backend.get":
        return {
          taskId: "task-1",
          status: "succeeded",
          progress: { phase: "done", completed: 3, total: 3 },
        };
      case "backend.readOutput": {
        const map = {
          svg: new TextEncoder().encode("<svg/>").buffer,
          png: new Uint8Array([1, 2, 3]).buffer,
          meta: new TextEncoder().encode(
            JSON.stringify({ width: 96, height: 64, degraded: false }),
          ).buffer,
        };
        return { name: payload.name, bytes: map[payload.name] };
      }
      default:
        return null;
    }
  };
}

test("沙箱复现网友报错：原始中文文件名会被宿主整单驳回", async () => {
  const { client, close } = await withHost(healthyHost());
  try {
    const picked = await client.image.pick();
    assert.equal(picked.name, CHINESE_PICK.name);
    await assert.rejects(
      client.backend.submit({
        operation: "image-to-svg",
        params: {},
        inputs: [{
          name: "image",
          fileName: picked.name,
          mimeType: picked.mimeType,
          bytes: picked.bytes,
        }],
        idempotencyKey: crypto.randomUUID(),
      }),
      (error) => {
        assert.equal(error.message, "Workshop backend task input is invalid.");
        assert.equal(error.code, "task-input-invalid");
        return true;
      },
    );
  } finally {
    close();
  }
});

test("沙箱验证 0.1.4 行为：中文文件名被规范化后全流程跑通", async () => {
  const { client, calls, close } = await withHost(healthyHost());
  try {
    const picked = await client.image.pick();
    const mimeType = safeTaskMimeType(picked.mimeType, picked.name);
    const fileName = safeTaskFileName(picked.name, mimeType);
    assert.ok(fileName !== picked.name, "中文名应被改写");
    assert.match(fileName, HOST_FILE_NAME);

    const task = await client.backend.submit({
      operation: "image-to-svg",
      params: {},
      inputs: [{ name: "image", fileName, mimeType, bytes: picked.bytes }],
      idempotencyKey: crypto.randomUUID(),
    });
    assert.equal(task.taskId, "task-1");

    const finished = await client.backend.waitFor(task.taskId);
    assert.equal(finished.status, "succeeded");

    const svg = await client.backend.readOutput(task.taskId, "svg");
    const png = await client.backend.readOutput(task.taskId, "png");
    const meta = await client.backend.readOutput(task.taskId, "meta");
    assert.ok(svg.bytes.byteLength > 0);
    assert.ok(png.bytes.byteLength > 0);
    const size = JSON.parse(new TextDecoder().decode(meta.bytes));
    assert.equal(size.width, 96);
    assert.equal(size.height, 64);

    const submit = calls.find((call) => call.method === "backend.submit");
    assert.ok(submit, "宿主应收到 backend.submit");
    assert.equal(submit.payload.inputs[0].name, "image");
    assert.match(submit.payload.inputs[0].fileName, HOST_FILE_NAME);
    assert.equal(submit.payload.inputs[0].mimeType, "image/png");
  } finally {
    close();
  }
});

test("沙箱验证：ASCII 文件名一字不改地提交", async () => {
  const { client, calls, close } = await withHost((method, payload) => {
    if (method === "image.pick") {
      return { name: "witch 01.png", mimeType: "image/png", bytes: PNG_BYTES };
    }
    if (method === "backend.submit") return { taskId: "task-2" };
    return healthyHost()(method, payload, () => {});
  });
  try {
    const picked = await client.image.pick();
    const mimeType = safeTaskMimeType(picked.mimeType, picked.name);
    const fileName = safeTaskFileName(picked.name, mimeType);
    assert.equal(fileName, "witch 01.png");
    await client.backend.submit({
      operation: "image-to-svg",
      params: {},
      inputs: [{ name: "image", fileName, mimeType, bytes: picked.bytes }],
      idempotencyKey: crypto.randomUUID(),
    });
    const submit = calls.find((call) => call.method === "backend.submit");
    assert.equal(submit.payload.inputs[0].fileName, "witch 01.png");
  } finally {
    close();
  }
});
