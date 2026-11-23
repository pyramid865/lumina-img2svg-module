import { test } from "node:test";
import assert from "node:assert/strict";
import {
  SLOW_HINT_SECONDS,
  describeProgress,
  phaseLabel,
} from "../src/progressText.ts";

test("阶段名翻成中文，未知阶段原样保留", () => {
  assert.equal(phaseLabel("preparing"), "准备");
  assert.equal(phaseLabel("tracing"), "描摹");
  assert.equal(phaseLabel("writing outputs"), "写文件");
  assert.equal(phaseLabel("done"), "完成");
  assert.equal(phaseLabel("something-new"), "something-new");
  assert.equal(phaseLabel(undefined), "描摹中");
});

test("不足 1 秒不显示计时，避免闪「已用 0 秒」", () => {
  assert.equal(describeProgress("tracing", 1, 3, 0), "描摹 1/3");
  assert.equal(describeProgress("tracing", 1, 3, 0.9), "描摹 1/3");
});

test("超过 1 秒开始走表，秒数向下取整", () => {
  assert.equal(describeProgress("tracing", 1, 3, 1), "描摹 1/3 · 已用 1 秒");
  assert.equal(describeProgress("tracing", 1, 3, 12.9), "描摹 1/3 · 已用 12 秒");
});

test("到 30 秒追加「还在算，别关窗口」，边界前后各测一次", () => {
  assert.equal(SLOW_HINT_SECONDS, 30);
  assert.equal(describeProgress("tracing", 1, 3, 29.9), "描摹 1/3 · 已用 29 秒");
  assert.equal(
    describeProgress("tracing", 1, 3, 30),
    "描摹 1/3 · 已用 30 秒 · 还在算，别关窗口",
  );
});

test("进度字段缺失时用兜底值，不乱显示 undefined", () => {
  assert.equal(describeProgress(undefined, undefined, undefined, 5), "描摹中 0/3 · 已用 5 秒");
});

test("完成后显示「用时 N 秒」，不再喊「还在算」", () => {
  assert.equal(describeProgress("done", 3, 3, 70.4), "完成 3/3 · 用时 70 秒");
  assert.equal(describeProgress("done", 3, 3, 0.2), "完成 3/3");
  assert.ok(!describeProgress("done", 3, 3, 70).includes("还在算"));
});
