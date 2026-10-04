// rapidfuzz.fuzz.partial_ratio + process.extract (no processor), enough for questions.rank_apps.
// partial_ratio = best Indel ratio of the shorter string against every same-length window of the longer one,
// including the shorter windows at both edges (rapidfuzz's short-needle algorithm finds this optimum).

function lcs(a, b) {
  const m = a.length;
  const n = b.length;
  if (!m || !n) return 0;
  let prev = new Uint16Array(n + 1);
  let cur = new Uint16Array(n + 1);
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      cur[j] = a[i - 1] === b[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    }
    [prev, cur] = [cur, prev];
  }
  return prev[n];
}

export function ratio(a, b) {
  const t = a.length + b.length;
  if (!t) return 100;
  return (200 * lcs(a, b)) / t;
}

export function partialRatio(s1, s2) {
  let a = [...s1];
  let b = [...s2];
  if (!a.length || !b.length) return 0;
  if (a.length > b.length) [a, b] = [b, a];
  const m = a.length;
  const n = b.length;
  let best = 0;
  const consider = (w) => {
    const r = ratio(a, w);
    if (r > best) best = r;
  };
  for (let i = 1; i < m; i++) consider(b.slice(0, i));
  for (let s = 0; s + m <= n; s++) {
    consider(b.slice(s, s + m));
    if (best === 100) return 100;
  }
  for (let i = 1; i < m; i++) consider(b.slice(n - i));
  if (s1.length === s2.length) {
    // rapidfuzz also tries the other direction when the lengths are equal
    const r = partialRatioOneWay([...s2], [...s1]);
    if (r > best) best = r;
  }
  return best;
}

function partialRatioOneWay(a, b) {
  const m = a.length;
  const n = b.length;
  let best = 0;
  for (let i = 1; i < m; i++) best = Math.max(best, ratio(a, b.slice(0, i)));
  for (let s = 0; s + m <= n; s++) best = Math.max(best, ratio(a, b.slice(s, s + m)));
  for (let i = 1; i < m; i++) best = Math.max(best, ratio(a, b.slice(n - i)));
  return best;
}

/** process.extract(query, choices, scorer=partial_ratio, limit) -> [[choice, score, index]] best first. */
export function extract(query, choices, limit) {
  const scored = choices.map((c, i) => [c, partialRatio(query, c), i]);
  scored.sort((x, y) => y[1] - x[1] || x[2] - y[2]);
  return scored.slice(0, limit);
}
