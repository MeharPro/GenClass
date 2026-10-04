"""Verbatim span candidates from a transcript (select, don't generate).

Jev never writes free text: code over-generates candidate spans of the user's own words and a
Choice question picks one (docs/research/SPEC.md §1.6, §2.2). Every candidate returned here is an
exact substring of the input, so whatever is typed is always something the user said.

Used by both the runtime question builder and the synthetic-data generator.
"""

from __future__ import annotations

import re

from jev_local.harness.catalog import CHAIN_WORDS, FILLERS

_QUOTE_RE = re.compile(r"[\"“”']([^\"“”']{1,120})[\"“”']")

# Payload verbs, longest first. Group "rest" is everything after the verb phrase.
_PAYLOAD_VERB_RE = re.compile(
    r"\b(?:"
    r"search\s+(?:the\s+web|google|online|the\s+internet)\s+for|"
    r"search\s+(?P<site>\w+(?:\s+\w+)?)\s+for|"
    r"search\s+for|look\s+up|google|find\s+me|find|"
    r"type\s+(?:in|out)|type|enter|write\s+down|write|put\s+in|put|fill\s+in|dictate|say|search"
    r")\b[\s,:]+(?P<rest>.+)$",
    re.IGNORECASE,
)

_DEST_RE = re.compile(
    r"\s+(?:in|into|on|inside|to|in\s+to)\s+(?:the\s+|this\s+|that\s+|a\s+)?(?:[\w-]+\s+){0,3}?"
    r"(?:box|field|input|bar|search\s*bar|search\s*box|search\s*field|search|text\s*box|textbox|"
    r"document|doc|note|notes|window|message|email|chat|address\s*bar|url\s*bar|terminal)\s*$",
    re.IGNORECASE,
)

_CHAIN_RE = re.compile(r"\s+(?:" + "|".join(re.escape(w) for w in CHAIN_WORDS) + r")\s+", re.IGNORECASE)
# A sentence end inside the text ("hello world. Press enter."): whisper punctuates short pauses, so
# chained commands often arrive as separate sentences with no chain word. Needs whitespace after
# it, so the dots in "wikipedia.org" or "3.5" are not sentence ends.
# Payload candidates are cut there only when the next sentence starts like a command, so dictation
# spanning sentences ("type running late. see you soon") still has its whole-text candidate first
# and gains no truncated one.
_SENT_END_RE = re.compile(
    r"[.!?]+[\"')\]]*(?=\s+(?:(?:and|then|um|uh|ok|okay|so)[\s,]+)*(?:open|launch|start|switch|quit|close|click|tap|"
    r"press|hit|select|choose|type|write|enter|dictate|search|google|look|go|visit|navigate|scroll|page|undo|"
    r"new|copy|paste|save|refresh|reload|cancel|confirm)\b)",
    re.IGNORECASE,
)
_EDGE_PUNCT = " \t,.;:!?"
_TRAILING_POLITE_RE = re.compile(r"(?:[\s,]+(?:please|thanks|thank\s+you|for\s+me|now|right\s+now))+\s*[.!?]*$", re.I)
_LEADING_FILLER_RE = re.compile(r"^(?:(?:" + "|".join(sorted(FILLERS, key=len, reverse=True)) + r")[\s,]+)+", re.I)


_CLEAN_STRIP = _EDGE_PUNCT + "\"'“”‘’"


def _clean(s: str) -> str:
    s = s.strip(_CLEAN_STRIP)
    s = _TRAILING_POLITE_RE.sub("", s)
    s = _LEADING_FILLER_RE.sub("", s)
    return s.strip(_CLEAN_STRIP)


def _variants(rest: str) -> list[str]:
    """rest, plus rest cut at each chain word, each also without a trailing destination phrase.

    "salt and pepper in the search box and press enter" ->
      whole, "salt", "salt and pepper in the search box", "salt and pepper", ...
    """
    cuts = sorted({c.start() for c in _CHAIN_RE.finditer(rest)} | {m.start() for m in _SENT_END_RE.finditer(rest)})
    prefixes = [rest] + [rest[:c] for c in cuts]
    out: list[str] = []
    for p in prefixes:
        m = _DEST_RE.search(p)
        if m:
            out.append(p[: m.start()])  # destination-stripped first: usually the intended payload
        out.append(p)
    return out


def extract_text_candidates(tail: str, max_n: int = 8, max_chars: int = 120) -> list[str]:
    text = tail.strip()
    if not text:
        return []
    raw: list[str] = []
    raw += [m.group(1) for m in _QUOTE_RE.finditer(text)]
    m = _PAYLOAD_VERB_RE.search(text)
    if m:
        raw += _variants(m.group("rest"))
    idx = text.lower().find(" for ")
    if idx >= 0:
        raw += _variants(text[idx + 5 :])
    parts = text.split(None, 1)
    if len(parts) == 2:
        raw += _variants(parts[1])
    raw.append(text)

    seen: set[str] = set()
    out: list[str] = []
    for r in raw:
        c = _clean(r)
        key = c.lower()
        if not c or key in seen or len(c) > max_chars or key == "none":
            continue
        # keep only exact substrings of the transcript (cleaning never invents text, but be strict)
        if c not in text:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_n:
            break
    return out


# ---------------------------------------------------------------- URLs

_TLDS = (
    "com|org|net|io|ai|dev|app|edu|gov|co|uk|ca|de|fr|es|it|nl|me|tv|gg|xyz|info|so|sh|ly|us|news|"
    "fm|to|in|jp|au|ch|se|no|fi|be|at|cc|biz|page|site|online|tech|blog|wiki"
)
_URL_RE = re.compile(r"\b((?:https?://)?(?:[a-z0-9-]+\.)+(?:" + _TLDS + r")(?:/[^\s,]*)?)", re.I)
_GOTO_RE = re.compile(r"\b(?:go\s+to|goto|visit|navigate\s+to|open\s+up|open|head\s+to|pull\s+up|load)\s+(?:the\s+)?([a-z0-9][a-z0-9-]{1,40})(?:\s+(?:website|site|page|dot\s+com))?\b", re.I)


def normalize_spoken_url(text: str) -> str:
    t = re.sub(r"\s+dot\s+", ".", text, flags=re.I)
    t = re.sub(r"\s+slash\s+", "/", t, flags=re.I)
    t = re.sub(r"\s*colon\s*/\s*/\s*", "://", t, flags=re.I)
    t = re.sub(r"\bw\s*w\s*w\b\.?", "www.", t, flags=re.I)
    return t


def extract_url_candidates(tail: str, max_n: int = 6) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(u: str) -> None:
        u = u.strip(_EDGE_PUNCT).lower()
        if u and u not in seen:
            seen.add(u)
            out.append(u)

    for src in (tail, normalize_spoken_url(tail)):
        for m in _URL_RE.finditer(src):
            add(m.group(1))
    has_web_cue = re.search(r"\b(website|site|web\s*page|dot\s+com|\.com|go\s+to|visit|navigate)\b", tail, re.I)
    if has_web_cue:
        for m in _GOTO_RE.finditer(tail):
            w = m.group(1).lower()
            if w not in {"the", "a", "my", "new", "tab", "page", "website", "site"}:
                add(f"{w}.com")
    return out[:max_n]


# ---------------------------------------------------------------- numbered picks

_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "last": -1,
}
_CARDINALS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "to": 2, "too": 2, "for": 4,  # common ASR homophones, only honored in "number X" / "option X" form
}


def parse_candidate_pick(tail: str, n_candidates: int) -> int | None:
    """'the second one' / 'number 3' / 'option two' / '3' -> 0-based index, else None."""
    t = tail.lower().strip(_EDGE_PUNCT)
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b(?:\s+one)?", t)
    if m:
        k = _ORDINALS[m.group(1)]
        idx = n_candidates - 1 if k == -1 else k - 1
        return idx if 0 <= idx < n_candidates else None
    m = re.search(r"\b(?:number|option|choice|item|pick)\s+(\d+|\w+)\b", t)
    if m:
        v = m.group(1)
        k = int(v) if v.isdigit() else _CARDINALS.get(v)
        if k is not None and 1 <= k <= n_candidates:
            return k - 1
    if re.fullmatch(r"\d{1,2}", t):
        k = int(t)
        return k - 1 if 1 <= k <= n_candidates else None
    if t in {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}:
        k = _CARDINALS[t]
        return k - 1 if k <= n_candidates else None
    return None


# ---------------------------------------------------------------- consumption

def _words(text: str) -> list[str]:
    return text.split()


def strip_leading_chain(words: list[str]) -> int:
    """Number of leading words that are chain/filler words ('and', 'then', 'um', ...)."""
    i = 0
    lowered = [w.lower().strip(_EDGE_PUNCT) for w in words]
    chain_single = {w for w in CHAIN_WORDS if " " not in w}
    while i < len(lowered):
        two = " ".join(lowered[i : i + 2])
        if two in CHAIN_WORDS:
            i += 2
        elif lowered[i] in chain_single or lowered[i] in FILLERS:
            i += 1
        else:
            break
    return i


def consumed_for(tail_words: list[str] | tuple[str, ...], intent: str, span: str | None = None) -> int:
    """How many tail words the executed action accounts for.

    Payload intents consume through the end of their span; other intents consume up to the first
    chain word after the command ("open notes and ..."). A trailing chain word is consumed too so
    the next virtual utterance starts at the next command's verb.
    """
    words = list(tail_words)
    if not words:
        return 0
    lowered = [w.lower().strip(_EDGE_PUNCT) for w in words]
    end = len(words)
    start_search = 1
    if span:
        sw = [w.lower().strip(_EDGE_PUNCT) for w in span.split()]
        n = len(sw)
        for i in range(len(lowered) - n + 1):
            if lowered[i : i + n] == sw:
                end = i + n
                start_search = end
                break
        else:
            return len(words)
        # also consume a destination phrase ("... in the search box") and the chain word after it
        rest = lowered[end:]
        cut = len(rest)
        for j in range(len(rest)):
            if rest[j] in CHAIN_WORDS or " ".join(rest[j : j + 2]) in CHAIN_WORDS:
                cut = j
                break
        seg = " ".join(words[end : end + cut])
        if seg and _DEST_RE.match(" " + seg):
            end += cut
        return end + strip_leading_chain(words[end:])
    for i in range(len(lowered)):
        if i >= start_search:
            two = " ".join(lowered[i : i + 2])
            if two in CHAIN_WORDS:
                return i + 2
            if lowered[i] in CHAIN_WORDS:
                return i + 1
        # A sentence end closes the command ("Open notes. Scroll down."). A one-word sentence only
        # does so for commands that need no argument ("Undo. Scroll down."), because whisper also
        # splits a single command ("Open. Notes.").
        if i + 1 < len(words) and ends_sentence(words[i]) and (i >= 1 or intent in _NO_ARG_INTENTS):
            return i + 1 + strip_leading_chain(words[i + 1 :])
    return end


_NO_ARG_INTENTS = frozenset({"undo", "confirm", "cancel", "go_back", "new_tab", "close_tab", "scroll_down", "scroll_up"})


def ends_sentence(word: str) -> bool:
    """'notes.' / 'down!' / 'it?"' end a sentence; 'wikipedia.org' and '...' alone do not."""
    w = word.rstrip("\"')]")
    return len(w) > 1 and w[-1] in ".!?" and any(ch.isalnum() for ch in w)
