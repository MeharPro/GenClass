import assert from "node:assert/strict";
import { test } from "node:test";
import { rescueTarget } from "../../src/core/controller.js";
import { DEFAULT_THRESHOLDS as T } from "../../src/core/types.js";

const snap = { elements: [
  { eid: "e01", role: "link", label: "Laptops" }, { eid: "e02", role: "link", label: "Tablets" },
  { eid: "e03", role: "button", label: "Add to cart (1 of 2)" }, { eid: "e04", role: "button", label: "Add to cart (2 of 2)" },
  { eid: "e05", role: "button", label: "Cart" },
] };
const resp = (choice, p) => ({ answers: { intent: { type: "choice", choice: "click" }, target: { type: "choice", choice, confidence: p, probabilities: new Map([["e01", choice === "e01" ? p : 0.01], ["e02", choice === "e02" ? p : 0.01], ["e03", choice === "e03" ? p : 0], ["e04", choice === "e04" ? p : 0], ["e05", 0], ["none", choice === "none" ? p : 0.05]]) } } });
const tail = (text) => ({ text });

test("verbatim label rescues a none/low target", () => {
  const r = resp("none", 0.68);
  assert.equal(rescueTarget(r, tail("click the laptops link"), snap, T).eid, "e01");
  assert.equal(r.answers.target.choice, "e01");
});
test("a confident model answer is kept", () => {
  const r = resp("e02", 0.97);
  assert.equal(rescueTarget(r, tail("click tablets"), snap, T), null);
  assert.equal(r.answers.target.choice, "e02");
});
test("ordinal picks among duplicates; longest label wins over 'cart'", () => {
  const r = resp("e04", 0.56);
  assert.equal(rescueTarget(r, tail("click the first add to cart button"), snap, T).eid, "e03");
});
test("ambiguous duplicates without an ordinal are not guessed", () => {
  const r = resp("none", 0.9);
  assert.equal(rescueTarget(r, tail("click add to cart"), snap, T), null);
});
test("no rescue for other intents", () => {
  const r = resp("none", 0.9);
  r.answers.intent.choice = "type_text";
  assert.equal(rescueTarget(r, tail("type laptops"), snap, T), null);
});
