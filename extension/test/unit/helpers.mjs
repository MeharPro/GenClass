import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
export const FIX = join(ROOT, "test", "fixtures");
export const ASSETS = join(ROOT, "release-assets");
export const readJson = (p) => JSON.parse(readFileSync(p, "utf8"));

/**
 * Wire questions with criteria as Maps in the order Python used. JSON.parse reorders integer-like keys, so the
 * order comes from the Python packer's labels when available (pack fixtures), else from criteriaOrder pairs.
 */
export function questionsFromJson(qs, labelsByQid = null) {
  const out = {};
  for (const [qid, q] of Object.entries(qs)) {
    if (q.type === "choice") {
      const order = labelsByQid ? labelsByQid[qid] : Object.keys(q.criteria);
      out[qid] = { ...q, criteria: new Map(order.map((l) => [l, q.criteria[l] ?? null])) };
    } else out[qid] = q;
  }
  return out;
}
