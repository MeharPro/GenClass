// Canonical text serialization of state and questions (port of jev_local/serialize.py).
//
// Questions use the wire shape {type, instructions, criteria}. Choice criteria are a Map (label -> desc|null)
// so labels that look like integers keep their order (plain JS objects would reorder them).

export const MAX_ARRAY_SEGMENTS = 64;

/** Python repr() of a str. */
export function pyRepr(s) {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = q;
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (ch === "\\") out += "\\\\";
    else if (ch === q) out += "\\" + q;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (c < 0x20 || c === 0x7f) out += "\\x" + c.toString(16).padStart(2, "0");
    else out += ch;
  }
  return out + q;
}

/** json.dumps(v, ensure_ascii=False, separators=(", ", ": ")). */
export function pyJson(v) {
  if (v === null || v === undefined) return "null";
  if (v === true) return "true";
  if (v === false) return "false";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    let s = String(v);
    if (/e/.test(s)) s = s.replace(/e([+-])(\d)$/, "e$10$2");
    return s;
  }
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(pyJson).join(", ") + "]";
  if (v instanceof Map) return "{" + [...v].map(([k, x]) => JSON.stringify(String(k)) + ": " + pyJson(x)).join(", ") + "}";
  return "{" + Object.entries(v).map(([k, x]) => JSON.stringify(k) + ": " + pyJson(x)).join(", ") + "}";
}

const isPlainObj = (e) => e !== null && typeof e === "object" && !Array.isArray(e);
const entries = (e) => (e instanceof Map ? [...e] : Object.entries(e));

export function entryText(e) {
  if (e === null || e === undefined) return "";
  if (typeof e === "string") return e.trim();
  if (Array.isArray(e)) {
    if (e.every((x) => typeof x === "string")) return e.map(entryText).filter((p) => p).join("; ");
    return pyJson(e);
  }
  if (isPlainObj(e)) {
    const lines = [];
    for (const [k, v] of entries(e)) {
      const nested = isPlainObj(v) || (Array.isArray(v) && !v.every((x) => typeof x === "string"));
      lines.push(nested ? `${k}: ${pyJson(v)}` : `${k}: ${entryText(v)}`);
    }
    return lines.join(" | ");
  }
  if (typeof e === "boolean") return e ? "True" : "False";
  return String(e);
}

function valueText(v) {
  if (Array.isArray(v) && v.every((x) => typeof x === "string")) return v.join("\n");
  return entryText(v);
}

/** -> [{key, text}] */
export function stateSegments(state, maxItems = MAX_ARRAY_SEGMENTS) {
  if (typeof state === "string") return [{ key: "", text: state.trim() }];
  if (Array.isArray(state)) {
    const segs = state.slice(0, maxItems).map((x, i) => ({ key: `[${i}]`, text: entryText(x) }));
    if (state.length > maxItems) {
      segs.push({ key: `[${maxItems}:]`, text: state.slice(maxItems).map(entryText).join("\n") });
    }
    return segs;
  }
  if (isPlainObj(state)) return entries(state).map(([k, v]) => ({ key: String(k), text: valueText(v) }));
  return [{ key: "", text: String(state) }];
}

export const DEFAULT_NOUL_INSTR = "Is the statement true of the state?";
export const DEFAULT_CHOICE_INSTR = "Which option best fits the state?";
export const DEFAULT_SCORE_INSTR = "Which level best describes the state?";

function bareLabel(label, desc) {
  if (desc === null || desc === undefined) return true;
  if (typeof desc === "string") {
    const d = desc.trim();
    return d === "" || d === label.trim();
  }
  return false;
}

export function criteriaMap(c) {
  return c instanceof Map ? c : new Map(Object.entries(c || {}));
}

/** -> {qid, kind, header, items[], labels[]} */
export function questionBlock(qid, q) {
  if (q.type === "noul") {
    const t = q.criteria ? entryText(q.criteria.true) : "";
    const f = q.criteria ? entryText(q.criteria.false) : "";
    return { qid, kind: "noul", header: entryText(q.instructions) || DEFAULT_NOUL_INSTR, items: [t || "yes", f || "no"], labels: ["true", "false"] };
  }
  if (q.type === "choice") {
    const crit = criteriaMap(q.criteria);
    const labels = [...crit.keys()];
    const items = [...crit].map(([lab, desc]) => (bareLabel(lab, desc) ? lab : `${lab}: ${entryText(desc)}`));
    return { qid, kind: "choice", header: entryText(q.instructions) || DEFAULT_CHOICE_INSTR, items, labels };
  }
  if (q.type === "score") {
    const items = q.criteria.map(entryText);
    return { qid, kind: "score", header: entryText(q.instructions) || DEFAULT_SCORE_INSTR, items, labels: items.map((_, i) => String(i)) };
  }
  throw new TypeError(`unsupported question type ${q.type}`);
}
