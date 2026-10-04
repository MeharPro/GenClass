"""ASR noise works on tagged tokens and never moves or destroys the words labels depend on."""

from __future__ import annotations

import random

from jev_local.data.asr_noise import (
    Tok, _split_tok, apply_noise, drop, filler, homophone, lower, repeat, spell_number, spell_numbers_in, spoken_url,
    strip_punct,
)


def _cmd() -> list[Tok]:
    # "Please open the Spotify app, thanks."  (commit + ident on the app name)
    return [
        Tok("Please", slot="lead", part=0, droppable=True),
        Tok("open"),
        Tok("the", arg="app", droppable=True),
        Tok("Spotify", arg="app", content=True, ident=True, commit=True),
        Tok("app,", arg="app", droppable=True),
        Tok("thanks.", slot="tail", part=0, droppable=True),
    ]


def _payload() -> list[Tok]:
    words = "remind me to buy flour for the weekend".split()
    toks = [Tok("type", content=True)] + [Tok(w, arg="payload", content=True) for w in words]
    toks[1].ident = toks[1].commit = True
    return toks


def test_spell_number():
    assert spell_number(0) == "zero"
    assert spell_number(42) == "forty two"
    assert spell_number(100) == "one hundred"
    assert spell_number(1250) == "one thousand two hundred fifty"
    assert spell_numbers_in("meet at 3 in room 214") == "meet at three in room two hundred fourteen"


def test_spoken_url():
    rng = random.Random(0)
    assert spoken_url("github.com/anthropics", rng) == "github dot com slash anthropics"
    assert spoken_url("docs.python.org", rng) == "docs dot python dot org"


def test_lower_and_strip_punct():
    toks = strip_punct(lower(_cmd()))
    assert [t.text for t in toks] == ["please", "open", "the", "spotify", "app", "thanks"]
    assert toks[3].commit and toks[3].ident  # tags survive


def test_split_tok_keeps_commit_with_ident():
    # regression: "spotify" misheard as "spot if i" must not mark the app as mentioned at "spot"
    # while the intent only commits at "i" (label said wait + app=Spotify)
    t = Tok("Spotify", arg="app", content=True, ident=True, commit=True)
    out = _split_tok(t, "Spot if i")
    assert [x.text for x in out] == ["Spot", "if", "i"]
    assert out[0].ident and out[0].commit
    assert not any(x.ident or x.commit for x in out[1:])


def test_split_tok_moves_resolve_to_last_word():
    t = Tok("mail", arg="target", content=True, resolve=True)
    out = _split_tok(t, "male")
    assert out[-1].resolve
    t = Tok("discord", arg="target", content=True, commit=True, resolve=True)
    out = _split_tok(t, "this cord")
    assert out[-1].resolve and out[-1].commit and not out[0].commit


def _tagged(toks: list[Tok]) -> list[tuple[str, str, bool, bool]]:
    return [(t.arg, t.slot, t.ident, t.commit) for t in toks if t.ident or t.commit or t.resolve]


def test_ops_never_destroy_label_tokens():
    for seed in range(300):
        rng = random.Random(seed)
        for base in (_cmd(), _payload()):
            n_flags = len(_tagged(base))
            for fn in (drop, repeat, filler):
                out, _ = fn(rng, [Tok(**vars(t)) for t in base])
                assert len(_tagged(out)) == n_flags, (fn.__name__, [t.text for t in out])
            out, _ = homophone(rng, [Tok(**vars(t)) for t in base])
            assert sum(t.commit for t in out) == sum(t.commit for t in base)
            assert sum(t.ident for t in out) == sum(t.ident for t in base)


def test_drop_keeps_payload_edges():
    for seed in range(200):
        out, ok = drop(random.Random(seed), _payload())
        pay = [t.text for t in out if t.arg == "payload"]
        assert pay[0] == "remind" and pay[-1] == "weekend"


def test_repeat_never_duplicates_payload_or_url():
    for seed in range(200):
        out, ok = repeat(random.Random(seed), _payload())
        assert [t.text for t in out if t.arg == "payload"] == "remind me to buy flour for the weekend".split()


def test_filler_never_splits_an_argument():
    for seed in range(200):
        out, ok = filler(random.Random(seed), _cmd())
        args = [t.arg for t in out]
        # "the Spotify app" stays contiguous
        i = args.index("app")
        assert args[i : i + 3] == ["app", "app", "app"]


def test_apply_noise_reports_ops_and_is_deterministic():
    a = apply_noise(random.Random(5), _cmd(), strength=1.5)
    b = apply_noise(random.Random(5), _cmd(), strength=1.5)
    assert [t.text for t in a[0]] == [t.text for t in b[0]] and a[1] == b[1]
    ops = set()
    for seed in range(400):
        ops |= set(apply_noise(random.Random(seed), _payload(), strength=1.5)[1])
    assert {"lower", "nopunct", "drop", "repeat", "filler", "homophone"} <= ops
    assert apply_noise(random.Random(0), _cmd(), strength=0.0)[1] == []
