// Small Python-semantics helpers used by the ports.

export function strip(s, chars) {
  if (chars === undefined) return s.trim();
  let a = 0;
  let b = s.length;
  while (a < b && chars.includes(s[a])) a++;
  while (b > a && chars.includes(s[b - 1])) b--;
  return s.slice(a, b);
}

export function rstrip(s, chars) {
  let b = s.length;
  while (b > 0 && chars.includes(s[b - 1])) b--;
  return s.slice(0, b);
}

/** str.split() with no arguments: split on whitespace runs, no empty strings. */
export function split(s) {
  const t = s.trim();
  return t ? t.split(/\s+/) : [];
}

/** str.split(None, 1) on an already-stripped string. */
export function split1(s) {
  const t = s.replace(/^\s+/, "");
  const m = /\s+/.exec(t);
  if (!m) return t ? [t] : [];
  return [t.slice(0, m.index), t.slice(m.index + m[0].length)];
}

export const isAlnum = (ch) => /[\p{L}\p{N}]/u.test(ch);

/** Python round(x, digits) (round-half-even on exact binary ties). */
export function pyRound(x, digits = 2) {
  const m = 10 ** digits;
  const y = x * m;
  const f = Math.floor(y);
  const d = y - f;
  let r;
  if (d === 0.5) r = f % 2 === 0 ? f : f + 1;
  else r = Math.round(y);
  return r / m + 0;
}

export const clip01 = (x) => (Number.isNaN(x) ? 0 : Math.min(1, Math.max(0, x)) + 0);
