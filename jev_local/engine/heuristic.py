"""HeuristicEngine: a zero-shot, deterministic, model-free System One engine.

It exists so the server and the harness run end to end before any model is trained, and as the
last-resort fallback of the auto route. It is fast (well under 5 ms for a 60-element harness
request) and honest about ignorance: with no evidence it answers uniform / 0.5.

How it reads a request:
- The *reference text* for a question is the state segment its instructions point at with a
  backticked path (``Which action does `transcript` ask...`` -> the "transcript" segment), else a
  transcript-like key (transcript, text, message, ...), else the whole serialized state.
- Choice: each option (label words + description words) is scored by the reference words it
  matches (exact, fuzzy >= 80 via rapidfuzz, or a shared 5-letter stem). A match is weighted by
  how rare the word is among the options (IDF, so "button" in every element line counts little),
  by field (label > description) and mildly by position (earlier words matter more: "the first
  command"). Scores go through a softmax. A "none"-like option gets a fixed baseline so it wins
  when nothing matches. Options that are verbatim spans of the reference (span-selection
  questions) all match equally well, so for those the candidate order is used as the prior.
- Noul: with true/false criteria, the reference is compared to each side's phrases/examples
  (whole-string fuzzy ratio, so length matters: "open" is closer to the unfinished example
  "open" than to the complete example "open safari"; plus a command-verb shape cue). Without
  criteria, keyword cues: content words of the instructions found in the reference push above
  0.5, a negation before the word pushes below, none found gives 0.4; instructions with no
  content words give exactly 0.5.
- Score: the choice scorer over level texts with a high temperature, so it stays uniform-ish.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from rapidfuzz import fuzz, process

from jev_local.engine.base import EngineResult, RawDist
from jev_local.schema import ChoiceQuestion, Entry, NoulQuestion, Question, ScoreQuestion, SystemOneRequest
from jev_local.serialize import Segment, entry_text, question_block, state_segments

NAME = "jev-local-heuristic-0.1.0"

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_ALNUM_RE = re.compile(r"\w+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_EXAMPLE_SPLIT_RE = re.compile(r"\s*(?:[,;:]|\bfor example\b|\be\.g\.|\bsuch as\b)\s*", re.I)

STOPWORDS = frozenset(
    """a an the to of for and or in on at by with from as is are was were be been being it its this that
    these those my your our me you i we he she they them his her their what which who whom whose does do
    did doing would should could can will shall may might must if then than so such some something any
    anything otherwise else other into onto about please ask asks asked tell tells told say says said
    there here via per also just very too own same each every both either whether how why when where
    while has have had having s t whole current currently""".split()
)
# Words that phrase a yes/no question rather than name what it is about.
_QUESTION_WORDS = frozenset(
    "statement true false state question instruction instructions answer yes no mention mentions "
    "mentioned contain contains contained already hard".split()
)
NEGATIONS = frozenset({"not", "don't", "dont", "never", "no", "without", "isn't", "doesn't", "didn't", "won't"})
NONE_LABELS = frozenset({"none", "nothing", "other", "neither", "n/a", "abstain", "unknown"})  # not "no": keep yes/no symmetric
TRANSCRIPT_KEYS = ("transcript", "utterance", "text", "message", "query", "input", "request", "document", "content")

# Tuning constants (chosen on a handful of harness and doc examples; see tests).
# Choice logits are score * (1 + ln K) / CHOICE_TEMP: a clear match should dominate however many
# unmatched options there are (60 screen elements, not just 3 departments).
CHOICE_TEMP = 0.6
SCORE_TEMP = 0.6
NONE_BASELINE = 0.45
LABEL_WEIGHT = 1.0
DESC_WEIGHT = 0.7
POSITION_DECAY = 0.5  # "the first command": earlier reference words weigh more
FUZZY_CUTOFF = 80.0
STEM_LEN = 5
SPAN_ORDER_STEP = 0.8  # logit drop per candidate position for span-selection questions
SPAN_NONE_LOGIT = -1.2
NOUL_SLOPE = 8.0
NOUL_MISSING_SIDE = 0.45
NOUL_SHAPE_WEIGHT = 0.3
NEGATION_SCOPE = 4  # "I do not want a refund": the negation is 3 words before "refund"

# Imperative openers. Used only as a shape cue for noul criteria whose examples differ in whether
# they start with a command verb (requests to a computer vs. talk to a person).
COMMAND_VERBS = frozenset(
    """open close quit exit launch start click press tap select choose check uncheck type write enter
    search google find look go navigate visit scroll page undo redo copy paste cut delete remove send save
    reload refresh play pause stop show hide switch new cancel confirm yes yeah yep no nope submit move drag
    zoom mute unmute turn set create make reply forward print download upload share pull bring take put add
    lock sign log fill dictate hit""".split()
)
_LEADING_FILLERS = frozenset("um uh uhm erm hmm please okay ok so hey just now then and oh well".split())


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    m = max(logits)
    ex = [math.exp(v - m) for v in logits]
    s = sum(ex)
    return tuple(e / s for e in ex)


@dataclass
class _Ref:
    """Reference text for one question, with a per-word match cache."""

    text: str
    words: list[str]
    positions: dict[str, int]  # content word -> first content-word position
    long_words: list[str]  # content words eligible for fuzzy matching
    negated: set[str]  # content words preceded by a negation within NEGATION_SCOPE words
    cache: dict[str, tuple[float, int, str]] = field(default_factory=dict)

    @classmethod
    def of(cls, text: str) -> "_Ref":
        ws = words(text)
        positions: dict[str, int] = {}
        negated: set[str] = set()
        j = 0
        for i, w in enumerate(ws):
            if w in STOPWORDS:
                continue
            positions.setdefault(w, j)
            j += 1
            if any(x in NEGATIONS for x in ws[max(0, i - NEGATION_SCOPE) : i]):
                negated.add(w)
        long_words = [w for w in positions if len(w) >= 4]
        return cls(text, ws, positions, long_words, negated)

    def match(self, w: str) -> tuple[float, int, str]:
        """(strength in [0, 1], reference position, reference word) for an option word.

        (0, -1, "") when the word does not occur in the reference, even approximately.
        """
        hit = self.cache.get(w)
        if hit is not None:
            return hit
        res = (0.0, -1, "")
        pos = self.positions.get(w)
        if pos is not None:
            res = (1.0, pos, w)
        elif len(w) >= 4 and self.long_words:
            best = process.extractOne(w, self.long_words, scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF)
            if best is not None:
                res = (0.9 * best[1] / 100.0, self.positions[best[0]], best[0])
            elif len(w) >= STEM_LEN:
                stem = w[:STEM_LEN]
                for rw in self.long_words:
                    if len(rw) >= STEM_LEN and rw[:STEM_LEN] == stem:
                        res = (0.8, self.positions[rw], rw)
                        break
        self.cache[w] = res
        return res


def _is_none_option(label: str, desc: str) -> bool:
    if label.strip().lower() in NONE_LABELS:
        return True
    d = desc.strip().lower()
    return d.startswith(("no ", "none", "nothing "))


class HeuristicEngine:
    name = NAME
    max_tokens = 65536  # Jev's documented context; the heuristic itself has no hard limit

    def __init__(self) -> None:
        # (state, questions, count) of the last count_tokens call: the router counts, then
        # evaluate is called with the same objects, so the second count is free.
        self._last_count: tuple[object, object, int] | None = None

    def supports(self, req: SystemOneRequest) -> bool:
        return True

    def count_tokens(self, req: SystemOneRequest) -> int:
        n = self._count(req.state, req.questions)
        self._last_count = (req.state, req.questions, n)
        return n

    def evaluate(self, state: Entry, questions: Mapping[str, Question]) -> EngineResult:
        t0 = time.perf_counter()
        segs = state_segments(state)
        refs: dict[str, _Ref] = {}
        dists: dict[str, RawDist] = {}
        for qid, q in questions.items():
            header = entry_text(q.instructions) if q.instructions is not None else ""
            ref_text = reference_text(segs, header)
            ref = refs.get(ref_text)
            if ref is None:
                ref = refs[ref_text] = _Ref.of(ref_text)
            if isinstance(q, ChoiceQuestion):
                labels = tuple(q.criteria.keys())
                descs = [entry_text(d) for d in q.criteria.values()]
                dists[qid] = RawDist("choice", self._choice(ref, labels, descs), labels)
            elif isinstance(q, ScoreQuestion):
                texts = [entry_text(c) for c in q.criteria]
                probs = _softmax([s / SCORE_TEMP for s in _option_scores(ref, [""] * len(texts), texts)])
                dists[qid] = RawDist("score", probs, tuple(str(i) for i in range(len(texts))))
            elif isinstance(q, NoulQuestion):
                dists[qid] = RawDist("noul", (self._noul(ref, header, q),))
            else:
                raise TypeError(f"unsupported question: {type(q).__name__}")
        last = self._last_count
        n_in = last[2] if last and last[0] is state and last[1] is questions else self._count(state, questions)
        n_out = sum(len(question_block(qid, q).items) + 1 for qid, q in questions.items())
        total = (time.perf_counter() - t0) * 1000.0
        return EngineResult(dists, n_in, n_out, NAME, timings_ms={"total": total})

    # ------------------------------------------------------------ choice

    def _choice(self, ref: _Ref, labels: Sequence[str], descs: Sequence[str]) -> tuple[float, ...]:
        none_idx = [i for i, (lab, d) in enumerate(zip(labels, descs)) if _is_none_option(lab, d)]
        if _is_span_question(ref, labels, descs, none_idx):
            logits, k = [], 0
            for i in range(len(labels)):
                if i in none_idx:
                    logits.append(SPAN_NONE_LOGIT)
                else:
                    logits.append(-SPAN_ORDER_STEP * k)
                    k += 1
            return _softmax(logits)
        scores = _option_scores(ref, labels, descs)
        for i in none_idx:
            scores[i] = max(scores[i], NONE_BASELINE)
        scale = (1.0 + math.log(len(labels))) / CHOICE_TEMP
        return _softmax([s * scale for s in scores])

    # ------------------------------------------------------------ noul

    def _noul(self, ref: _Ref, header: str, q: NoulQuestion) -> float:
        t = entry_text(q.criteria.true) if q.criteria else ""
        f = entry_text(q.criteria.false) if q.criteria else ""
        if t or f:
            st = _example_sim(ref, t, skip_extended=False) if t else NOUL_MISSING_SIDE
            sf = _example_sim(ref, f, skip_extended=True) if f else NOUL_MISSING_SIDE
            return _sigmoid(NOUL_SLOPE * (st - sf))
        paths = {w for m in _BACKTICK_RE.finditer(header) for w in words(m.group(1))}
        cues = [w for w in dict.fromkeys(content_words(header)) if w not in paths and w not in _QUESTION_WORDS]
        if not cues:
            return 0.5
        score = 0.0
        for w in cues:
            strength, _, matched = ref.match(w)
            if strength > 0:
                score += -strength if matched in ref.negated else strength
        if score == 0.0:
            return 0.4  # the question names things the text never mentions: weak evidence for "no"
        return 0.5 + 0.4 * math.tanh(0.8 * score)

    # ------------------------------------------------------------ tokens

    @staticmethod
    def _count(state: Entry, questions: Mapping[str, Question]) -> int:
        """Approximate BPE tokens of the packed request: ~1.3 per word, one per punctuation mark,
        one separator per segment / header / item."""
        parts: list[str] = []
        for s in state_segments(state):
            parts += (s.key, s.text)
        for qid, q in questions.items():
            b = question_block(qid, q)
            parts.append(b.header)
            parts.extend(b.items)
        text = "\n".join(parts)
        return int(1.3 * len(_ALNUM_RE.findall(text)) + len(_PUNCT_RE.findall(text)) + len(parts))


def reference_text(segs: Sequence[Segment], header: str) -> str:
    """The state text a question is about (see module docstring)."""
    if not segs:
        return ""
    by_key = {s.key.lower(): s for s in segs if s.key}
    for m in _BACKTICK_RE.finditer(header):
        root = re.split(r"[.\[]", m.group(1).strip(), maxsplit=1)[0].lower()
        if root in by_key:
            return by_key[root].text
    for k in TRANSCRIPT_KEYS:
        if k in by_key:
            return by_key[k].text
    return "\n".join(f"{s.key}: {s.text}" if s.key else s.text for s in segs)


def _option_scores(ref: _Ref, labels: Sequence[str], descs: Sequence[str]) -> list[float]:
    """Σ over reference words of the best IDF- and field-weighted match inside each option."""
    opt_words: list[list[tuple[str, float]]] = []
    df: dict[str, int] = {}
    for lab, desc in zip(labels, descs):
        ws: dict[str, float] = {}
        for w in content_words(lab.replace("_", " ").replace("+", " ")):
            ws[w] = LABEL_WEIGHT
        for w in content_words(desc):
            ws.setdefault(w, DESC_WEIGHT)
        opt_words.append(list(ws.items()))
        for w in ws:
            df[w] = df.get(w, 0) + 1
    n = max(1, len(labels))
    norm = math.log(1.0 + n)
    scores: list[float] = []
    for ws in opt_words:
        best: dict[int, float] = {}
        for w, field_w in ws:
            strength, pos, _ = ref.match(w)
            if strength <= 0:
                continue
            idf = math.log(1.0 + n / df[w]) / norm
            v = strength * field_w * idf / (1.0 + POSITION_DECAY * pos)
            if v > best.get(pos, 0.0):
                best[pos] = v
        scores.append(sum(best.values()))
    return scores


def _is_span_question(ref: _Ref, labels: Sequence[str], descs: Sequence[str], none_idx: Sequence[int]) -> bool:
    """Options are (near-)verbatim pieces of the reference text: every one 'matches' fully."""
    cands = [(lab, d) for i, (lab, d) in enumerate(zip(labels, descs)) if i not in none_idx]
    if len(cands) < 2 or any(d for _, d in cands):
        return False
    present = set(ref.words)
    inside = sum(1 for lab, _ in cands if (ws := words(lab)) and all(w in present for w in ws))
    return inside >= 0.75 * len(cands)


def _phrases(text: str) -> list[str]:
    parts = [" ".join(words(p)) for p in _EXAMPLE_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def _verb_initial(ws: Sequence[str]) -> bool:
    for w in ws:
        if w not in _LEADING_FILLERS:
            return w in COMMAND_VERBS
    return False


def _example_sim(ref: _Ref, crit: str, skip_extended: bool) -> float:
    """Similarity between the reference and one side's phrases/examples.

    Best whole-string fuzzy ratio over the phrases, plus a shape cue: the share of phrases that
    agree with the reference on starting with a command verb (neutral when both sides are alike,
    as for "complete"; decisive for "a request for the computer" vs "talking to a person").
    With `skip_extended` (the "false"/unfinished side), a phrase that the reference strictly
    extends with a content word ("search for" -> "search for cats") no longer describes it.
    """
    x = " ".join(ref.words)
    if not x:
        return 0.0
    phrases = _phrases(crit)
    best = 0.0
    for p in phrases:
        pw = p.split()
        if skip_extended and len(ref.words) > len(pw) and ref.words[: len(pw)] == pw:
            if any(w not in STOPWORDS for w in ref.words[len(pw) :]):
                continue
        best = max(best, fuzz.ratio(x, p) / 100.0)
    if not phrases:
        return best
    ref_verb = _verb_initial(ref.words)
    agree = sum(1 for p in phrases if _verb_initial(p.split()) == ref_verb) / len(phrases)
    return best + NOUL_SHAPE_WEIGHT * agree
