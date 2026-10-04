// JS tokenizer + packer vs the Python Packer (jev_local/engine/encoder/tokenize_pack.py) on 50 real harness requests.
import assert from "node:assert/strict";
import { test } from "node:test";
import { join } from "node:path";
import { Tokenizer } from "../../src/core/tokenizer.js";
import { Packer } from "../../src/core/packer.js";
import { ASSETS, FIX, questionsFromJson, readJson } from "./helpers.mjs";

const tok = new Tokenizer(readJson(join(ASSETS, "tokenizer.json")));
const reqs = readJson(join(FIX, "requests50.json"));
const packs = readJson(join(FIX, "pack_fixtures.json"));

test("50 requests: identical token ids, positions, groups and marker positions", () => {
  const packer = new Packer(tok, { maxPositions: 1536 });
  let tokens = 0;
  reqs.forEach((r, i) => {
    const py = packs[i];
    assert.equal(py.id, r.id);
    const labels = Object.fromEntries(Object.entries(py.q_index).map(([q, qi]) => [q, qi.labels]));
    const { packed } = packer.pack(r.state, questionsFromJson(r.questions, labels));
    assert.deepEqual(packed.inputIds, py.input_ids, `${r.id}: input_ids`);
    assert.deepEqual(packed.positionIds, py.position_ids, `${r.id}: position_ids`);
    assert.deepEqual(packed.qGroup, py.q_group, `${r.id}: q_group`);
    assert.deepEqual(packed.iGroup, py.i_group, `${r.id}: i_group`);
    assert.equal(packed.nState, py.n_state);
    assert.deepEqual([...packed.qIndex.keys()], Object.keys(py.q_index), `${r.id}: question order`);
    for (const [qid, qi] of packed.qIndex) {
      const p = py.q_index[qid];
      assert.equal(qi.kind, p.kind);
      assert.equal(qi.header, p.header);
      assert.deepEqual(qi.labels, p.labels);
      assert.equal(qi.qPos, p.q_pos);
      assert.deepEqual(qi.itemPos, p.item_pos);
    }
    tokens += packed.inputIds.length;
  });
  assert.ok(tokens > 50000);
});

test("tokenizer edge cases: special tokens are text, whitespace runs and added tokens split out", () => {
  // [Q] typed by a user must never become the marker id.
  const ids = tok.encode("click [Q] Delete all");
  assert.ok(!ids.includes(tok.tokenId("[Q]")));
  // a run of spaces is a non-special added token (leftmost-longest)
  const sp = tok.encode("a" + " ".repeat(30) + "b");
  assert.ok(sp.includes(tok.tokenId(" ".repeat(24))));
  assert.ok(tok.encode("email |||EMAIL_ADDRESS||| now").includes(tok.tokenId("|||EMAIL_ADDRESS|||")));
  // non-ASCII text round-trips through byte-level BPE without unknown pieces
  assert.ok(tok.encode("café — naïve 東京 😀").length > 0);
});
