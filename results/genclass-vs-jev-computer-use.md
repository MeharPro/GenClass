# Jev (typesafe/jev-1.13) vs jev-local-fast — held-out test, same inputs

Examples scored: 1280 (1000 computer-use + 300 generic sampled, seed 7)

| group | n | Jev | ours | ours right / Jev wrong | Jev right / ours wrong |
|---|---|---|---|---|---|
| CU all questions | 9093 |  79.0% |  97.2% | 1796 | 136 |
| CU intent, complete | 336 |  91.4% |  92.0% | 21 | 19 |
| CU intent, mid-sentence prefix | 664 |  66.4% |  90.4% | 188 | 29 |
| CU target (real element) | 119 |  86.6% |  82.4% | 12 | 17 |
| CU text span (real payload) | 98 |  75.5% |  94.9% | 23 | 4 |
| GEN held-out task families | 451 |  94.7% |  80.5% | 9 | 73 |

| question | n | Jev | ours |
|---|---|---|---|
| cu:app | 1000 |  96.5% |  99.0% |
| cu:complete | 1000 |  72.7% |  93.3% |
| cu:destructive | 1000 |  94.1% |  99.4% |
| cu:folder | 1000 |  98.7% | 100.0% |
| cu:intent | 1000 |  74.8% |  90.9% |
| cu:is_command | 1000 |  82.4% |  96.1% |
| cu:key | 1000 |  89.4% |  98.7% |
| cu:scroll_amount | 54 |  59.3% |  98.1% |
| cu:target | 990 |  89.1% |  97.9% |
| cu:text_span | 1000 |  14.0% |  99.5% |
| cu:url_span | 49 |  81.6% | 100.0% |
| gen:author | 23 | 100.0% |  21.7% |
| gen:claim_1 | 33 | 100.0% |  97.0% |
| gen:claim_2 | 22 | 100.0% |  95.5% |
| gen:deadline_line | 20 | 100.0% | 100.0% |
| gen:priority | 39 |  53.8% |  33.3% |

Latency per request (all questions in one call): Jev via OpenRouter p50 177 ms, p95 311 ms (includes network) · ours on Azure EPYC CPU, 4 threads (M1 not measured: Mac was swapping) p50 187 ms, p95 204 ms
Jev usage: 2,498,287 input tokens, $0.1049

## How to read this (2026-10-01)

**Home-field advantage.** The test set was produced by our own synthetic generator. It uses held-out templates, apps and screens, but the conventions are ours, and Jev answers zero-shot. Some gaps are conventions, not intelligence:
- `text_span` should be `none` when nothing is being typed. Jev usually picks a span anyway, which is why it scores 14% overall. The harness only reads `text_span` when the intent is type or search, so the fair row is "text span (real payload)".
- `complete` and `wait` follow our generator's rules for "committed" prefixes.

**The fair rows:**

| Row | Jev | Ours | Verdict |
|---|---|---|---|
| intent, complete command | 91.4% | 92.0% | tie |
| intent, mid-sentence | 66.4% | 90.4% | ours |
| target, real element (n=119, ±7 pts) | 86.6% | 82.4% | Jev slightly ahead |
| text span, real payload | 75.5% | 94.9% | ours |
| GEN held-out task families | 94.7% | 80.5% | Jev |

**Excluded examples.** OpenRouter rejected 20 GEN examples with HTTP 400 because its validator accepts only string criteria and these used object criteria. They are excluded from both columns.

**Conformance check against 1,280 real Jev responses:**
- Choice confidence `(K·pmax−1)/(K−1)` matches on 6,239 of 6,244 answers within 0.02. The 5 exceptions are rounding at exactly 0.02.
- Score confidence (mode-centred) matches on 1,053 of 1,053 within 0.03.
- Score value Σi·pᵢ matches on 1,051 of 1,053 within 0.02.
