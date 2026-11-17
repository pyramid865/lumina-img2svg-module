import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import vm from "node:vm";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const {
  WORKSHOP_API_VERSION,
  WORKSHOP_BACKEND_API_VERSION,
  WORKSHOP_RPC_EVENT_NAMES,
  WORKSHOP_FEATURE_NAMES,
  WORKSHOP_RPC_VERSION,
} = await import("@lumina/workshop-sdk");

const manifest = JSON.parse(await readFile(path.join(root, "manifest.json"), "utf8"));

/**
 * Load the FINAL built UI (ui/index.html), extract its single inline script,
 * and really execute it with a fake host window. Capture the `ready` message
 * the SDK posts, then validate it the same way a production host's
 * negotiateReady does: type / moduleId / moduleVersion / apiVersion / events.
 *
 * This closes the release-boundary gap where the package said 0.1.2 but the
 * bundled UI still announced 0.1.1 (the host rejects that handshake).
 */
async function launchBuiltUiAndCaptureReady() {
  const html = await readFile(path.join(root, "ui/index.html"), "utf8");
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
  assert.equal(scripts.length, 1, "UI 必须只有一份内联脚本");
  const js = scripts[0][1];

  const readyMessages = [];
  const listeners = new Map();

  // A fake MessagePort so connectWorkshop completes the handshake instead of
  // timing out; the module doesn't actually send anything during startup.
  const fakePort = {
    onmessage: null,
    onmessageerror: null,
    start() {},
    close() {},
    postMessage(envelope) {
      // Answer the few requests made during startup so no pending timer is
      // left dangling (ui.getState, lifecycle.ready, ...).
      if (envelope?.kind !== "request") return;
      const result = envelope.method === "ui.getState"
        ? { locale: "zh-CN", theme: "light", tokens: {} }
        : {};
      queueMicrotask(() => {
        this.onmessage?.({
          data: {
            protocol: "lumina-workshop-rpc",
            version: WORKSHOP_RPC_VERSION,
            kind: "response",
            requestId: envelope.requestId,
            ok: true,
            result,
          },
        });
      });
    },
  };

  const parent = {
    postMessage(msg) {
      readyMessages.push(msg);
      if (msg?.type === "lumina.workshop.ready") {
        const onMessage = listeners.get("message");
        if (onMessage) {
          onMessage({
            source: parent,
            data: { type: "lumina.workshop.connect", sessionId: "release-test" },
            ports: [fakePort],
          });
        }
      }
    },
  };

  const elements = new Map();
  const documentStub = {
    documentElement: { lang: "", dataset: {}, style: { setProperty() {} } },
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, {
          textContent: "",
          disabled: true,
          addEventListener() {},
          classList: { toggle() {} },
        });
      }
      return elements.get(id);
    },
  };
  const fakeWindow = {
    parent,
    document: documentStub,
    addEventListener(t, fn) { if (t === "message") listeners.set(t, fn); },
    removeEventListener(t, fn) { if (listeners.get(t) === fn) listeners.delete(t); },
  };

  const sandbox = {
    window: fakeWindow,
    document: documentStub,
    crypto: globalThis.crypto,
    TextDecoder: globalThis.TextDecoder,
    setTimeout, clearTimeout,
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(js, sandbox, { timeout: 5000 });
  // Let the async connectWorkshop handshake finish.
  await new Promise((resolve) => setTimeout(resolve, 50));
  return readyMessages;
}

/** Production-host-style validation of a ready message against the manifest. */
function negotiateReady(msg) {
  if (!msg || typeof msg !== "object") return null;
  if (msg.type !== "lumina.workshop.ready") return null;
  if (msg.moduleId !== manifest.id) return null;
  if (msg.moduleVersion !== manifest.version) return null;
  const expectedApi = manifest.workshopApi?.min === "1.1.0"
    ? WORKSHOP_BACKEND_API_VERSION
    : WORKSHOP_API_VERSION;
  if (msg.apiVersion !== expectedApi) return null;
  if (!Array.isArray(msg.events)) return null;
  for (const e of WORKSHOP_RPC_EVENT_NAMES) {
    if (!msg.events.includes(e)) return null;
  }
  return { moduleId: msg.moduleId, moduleVersion: msg.moduleVersion, apiVersion: msg.apiVersion };
}

test("最终包 UI 的真实 ready 消息通过清单身份校验（版本不得分叉）", async () => {
  const messages = await launchBuiltUiAndCaptureReady();
  assert.equal(messages.length, 1, "应发出恰好一条 ready 消息");
  const ready = messages[0];

  // 精确值：直接断言握手上报的版本等于清单版本。
  assert.equal(ready.moduleVersion, manifest.version,
    `握手版本 ${ready.moduleVersion} 必须等于清单版本 ${manifest.version}`);
  assert.equal(ready.moduleId, manifest.id, "握手 id 必须等于清单 id");

  const negotiated = negotiateReady(ready);
  assert.ok(negotiated, "生产 negotiateReady 校验必须通过");
  assert.equal(negotiated.moduleVersion, manifest.version);

  // backend.tasks 模块应上报 1.1.0。
  assert.ok((ready.features ?? []).includes("backend.tasks"),
    "后端模块必须请求 backend.tasks 能力");
  assert.equal(ready.apiVersion, WORKSHOP_BACKEND_API_VERSION);
});

test("manifest 是唯一版本源：UI、handoff、配方来源都不含过时的硬编码版本", async () => {
  const html = await readFile(path.join(root, "ui/index.html"), "utf8");
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  // 版本来自内联的 manifest 对象（version:"..."），而不是另一处硬编码常量。
  assert.ok(script.includes(`version:"${manifest.version}"`),
    "内联 manifest.version 必须等于清单版本");
  // 不应再有旧版本的字面量出现在脚本里。
  const staleVersions = ["0.1.0", "0.1.1", "0.1.2"].filter((v) => v !== manifest.version);
  for (const v of staleVersions) {
    assert.ok(!script.includes(`"${v}"`), `脚本里不应残留旧版本 ${v}`);
  }
});
