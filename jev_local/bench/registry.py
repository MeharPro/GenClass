"""jevbench registry: benchmark datasets and the training-exclusion rules (v2 PLAN §1.1, §1.2, §1.6; benchmax
PLAN §2.1-§2.4).

This file is the single source of truth for decontamination stage 1 (CONTRACT hard rule 5). Every training
converter must drop a source when `is_excluded(name, track=...)` is true for its dataset id, its source / task /
original_task name, or any inner source column. Matching is deliberately over-inclusive: losing a small
training source costs little; leaking a benchmark sibling voids the zero-shot claim.

Pure stdlib; safe to import anywhere (Mac included). It never reads files at import time: the `public_jev`
tables below are a frozen copy generated from bench/public/targets.json by
`scripts/benchmax_build_targets.py --write-registry` (a test checks the copy is not stale).

Two tracks (benchmax PLAN §2.1):
  Z  zero-shot: every category below is excluded, from any split, mirror, sibling or derived set.
  S  supervised: the official train split of a benchmark whose targets.json `track.S` is "train-split" is
     allowed, plus the sources named in its `train_sources`. Everything else stays excluded, and
     `jev_labelled`, the clean dev sets (selection only) and the S deny-list (§7 traps) are never waived.
     The S waiver is name-level only: the converter still has to take the TRAIN split, and the evaluated split
     is protected by the item-level hash check (jev_local.data.v2.decontam).

What is excluded (`exclusion_reason` returns the first matching category; precedence = this order):
  test:<key>        a jevbench-test dataset, any split, any mirror
  dev:<key>         a clean jevbench-dev dataset (checkpoint selection only), any split, any mirror
  sibling:<key>     derived / same-text sources of a test or dev dataset (v2 PLAN §1.6 table)
  jev_labelled      any data carrying Jev outputs (TypeSafe MCA §2.3(b)); never waived on any track
  public_jev:<key>  a source of one of the 549 published Jev rows (bench/public/targets.json): every hf_id,
                    every benchmark key, the sibling patterns of suite-reproduction-specs.md §1.5 and
                    train-data-and-supervised-ceilings.md §7, and the v2-mix overlap list of benchmax PLAN §0.4
  reserved:<key>    external typed-decision suites we compare against (jevbench, DI kit, kev, ...)
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

REGISTRY_VERSION = "jevbench-v1.1"  # v1.1: clean dev set + public_jev family (benchmax PLAN §2.3.1-2); v1 dev numbers are not comparable
DEV_SET_VERSION = "clean-2026-10-03"
SEED = 20261001  # every sample / shuffle in the builder derives from this
AREAS = {
    "A": "Topic",
    "B": "Sentiment",
    "C": "Emotion",
    "D": "Intent",
    "E": "Inference",
    "F": "Fact & safety",
    "G": "Multi-question",
}
TRACKS = ("Z", "S")


@dataclass(frozen=True)
class BenchDataset:
    key: str  # short id, used in file names: bench/jevbench/<role>/<key>.jsonl
    role: str  # "test" | "dev" | "ref" (shown, not counted) | "retired_dev" (moved to public_jev)
    area: str  # AREAS key for test; dev uses the same letters (§1.2 grouping)
    title: str
    hf_id: str
    config: str | None
    splits: tuple[str, ...]  # HF splits read (concatenated in this order)
    licence: str
    license_use: str  # "commercial" | "research" | "unknown"
    n_used: int  # target number of scored items (states)
    sampling: str  # "all" | "strat" (class-stratified) | "random" | "strat_by_source"
    mapping: str  # question mapping, human readable
    primary_metric: str
    label_source: str  # "deusser" | "btzsc" | "programmatic" | "native" (+ combinations)
    num: int = 0  # 1..25 for test datasets (PLAN §1.1 table order)
    n_total: int | None = None  # size of the split(s) before sampling
    revision: str | None = None  # pinned HF sha; bench/jevbench/manifest.json is authoritative
    gated: bool = False
    jev_published: str = ""  # sanity reference only (never a comparison target)
    question_kinds: tuple[str, ...] = ()  # kinds inside one request
    shuffle_rotation: bool = False  # one seeded option-shuffle rotation (choice datasets, PLAN §1.1)
    bare_variant: bool = False  # bare-names robustness variant exists (described choice labels)
    notes: str = ""


# ---------------------------------------------------------------------------------------------------------
# jevbench-test: 25 datasets, 7 areas (PLAN §1.1). Revisions are pinned by the builder into the manifest.

TEST: tuple[BenchDataset, ...] = (
    BenchDataset("ag_news", "test", "A", "AG News", "fancyzhx/ag_news", None, ("test",), "unknown (academic)",
                 "unknown", 2000, "strat", "choice(4), described labels", "acc", "deusser",
                 num=1, n_total=7600, jev_published="0.885 (n=7,600)", question_kinds=("choice",),
                 shuffle_rotation=True, bare_variant=True),
    BenchDataset("yahoo_topics", "test", "A", "Yahoo Answers Topics", "community-datasets/yahoo_answers_topics", None,
                 ("test",), "unknown (Webscope)", "unknown", 2000, "strat", "choice(10), BTZSC hypotheses",
                 "acc", "btzsc", num=2, n_total=60000, question_kinds=("choice",), shuffle_rotation=True,
                 bare_variant=True),
    BenchDataset("fin_topic", "test", "A", "Twitter Financial News Topic", "zeroshot/twitter-financial-news-topic",
                 None, ("validation",), "MIT", "commercial", 4117, "all", "choice(20)", "acc", "programmatic",
                 num=3, n_total=4117, jev_published="0.670", question_kinds=("choice",), shuffle_rotation=True,
                 bare_variant=True),
    BenchDataset("sst2", "test", "B", "SST-2", "stanfordnlp/sst2", None, ("validation",), "unknown (SST)", "unknown",
                 872, "all", "noul 'is the sentiment positive?' (primary) + separate choice(2) request", "acc",
                 "deusser", num=4, n_total=872, jev_published="0.964", question_kinds=("noul",),
                 notes="The choice variant is a second request per item (Deußer template), scored as secondary."),
    BenchDataset("fin_phrasebank", "test", "B", "Financial PhraseBank", "atrost/financial_phrasebank", None, ("test",),
                 "CC BY-NC-SA 3.0", "research", 970, "all", "choice(3), investor-sentiment instruction", "acc",
                 "deusser", num=5, n_total=970, jev_published="0.730", question_kinds=("choice",),
                 shuffle_rotation=True, bare_variant=True),
    BenchDataset("sst5", "test", "B", "SST-5", "SetFit/sst5", None, ("test",), "unknown (SST)", "unknown", 2210, "all",
                 "score(5)", "acc_mode", "deusser", num=6, n_total=2210, jev_published="0.579 acc",
                 question_kinds=("score",)),
    BenchDataset("yelp5", "test", "B", "Yelp Review Full", "Yelp/yelp_review_full", None, ("test",),
                 "Yelp dataset terms (NC)", "research", 2000, "strat", "score(5) stars", "acc_mode", "programmatic",
                 num=7, n_total=50000, question_kinds=("score",)),
    BenchDataset("dair_emotion", "test", "C", "DAIR Emotion", "dair-ai/emotion", "split", ("test",), "other (research)",
                 "research", 2000, "all", "choice(6), BTZSC hypotheses", "acc", "deusser+btzsc", num=8,
                 n_total=2000, jev_published="0.585 / 0.500 F1", question_kinds=("choice",), shuffle_rotation=True,
                 bare_variant=True),
    BenchDataset("tweeteval_emotion", "test", "C", "TweetEval emotion", "cardiffnlp/tweet_eval", "emotion", ("test",),
                 "SemEval-18 terms", "research", 1421, "all", "choice(4)", "acc", "programmatic", num=9,
                 n_total=1421, jev_published="0.827 (n=1,000)", question_kinds=("choice",), shuffle_rotation=True,
                 bare_variant=True),
    BenchDataset("go_emotions", "test", "C", "GoEmotions", "google-research-datasets/go_emotions", "simplified",
                 ("test",), "Apache-2.0", "commercial", 2000, "random", "28 nouls per state", "macro_f1", "deusser",
                 num=10, n_total=5427, jev_published="0.243", question_kinds=("noul",)),
    BenchDataset("banking77", "test", "D", "Banking77", "mteb/banking77", None, ("test",), "CC BY 4.0", "commercial",
                 3076, "all", "choice(77), BTZSC hypotheses", "acc", "deusser+btzsc", num=11, n_total=3076,
                 jev_published="0.797 / 0.788 F1", question_kinds=("choice",), shuffle_rotation=True,
                 bare_variant=True),
    BenchDataset("clinc150", "test", "D", "CLINC150 plus OOS", "clinc/clinc_oos", "plus", ("test",), "CC BY 3.0",
                 "commercial", 5500, "all", "choice(151 incl. oos)", "acc", "deusser+programmatic", num=12,
                 n_total=5500, jev_published="0.895", question_kinds=("choice",), shuffle_rotation=True,
                 bare_variant=True),
    BenchDataset("massive", "test", "D", "MASSIVE en-US intent", "mteb/amazon_massive_intent", "en", ("test",),
                 "CC BY 4.0", "commercial", 2974, "all", "choice(60), BTZSC hypotheses", "acc", "btzsc", num=13,
                 n_total=2974, question_kinds=("choice",), shuffle_rotation=True, bare_variant=True),
    BenchDataset("boolq", "test", "E", "BoolQ", "google/boolq", None, ("validation",), "CC BY-SA 3.0", "commercial",
                 3270, "all", "noul (state = passage + question)", "acc", "deusser", num=14, n_total=3270,
                 jev_published="0.913", question_kinds=("noul",)),
    BenchDataset("rte", "test", "E", "RTE", "nyu-mll/glue", "rte", ("validation",), "GLUE terms", "unknown", 277, "all",
                 "noul (entailment)", "acc", "deusser", num=15, n_total=277, question_kinds=("noul",)),
    BenchDataset("anli", "test", "E", "ANLI r1-r3", "facebook/anli", None, ("test_r1", "test_r2", "test_r3"),
                 "CC BY-NC 4.0", "research", 3200, "all", "choice(3)", "acc", "deusser", num=16, n_total=3200,
                 jev_published="0.739 / 0.748 F1", question_kinds=("choice",), shuffle_rotation=True),
    BenchDataset("paws", "test", "E", "PAWS", "google-research-datasets/paws", "labeled_final", ("test",),
                 "PAWS terms", "unknown", 2000, "strat", "noul (same meaning?)", "acc", "deusser", num=17,
                 n_total=8000, jev_published="0.850", question_kinds=("noul",)),
    BenchDataset("stsb", "test", "E", "STS-B", "sentence-transformers/stsb", None, ("test",), "STS terms", "unknown",
                 1379, "all", "score(6 levels, 0-5)", "spearman", "deusser", num=18, n_total=1379,
                 jev_published="0.890", question_kinds=("score",)),
    BenchDataset("llm_aggrefact", "test", "F", "LLM-AggreFact", "lytang/LLM-AggreFact", None, ("test",),
                 "CC BY-ND 4.0 (eval)", "research", 5000, "strat_by_source", "noul (claim supported?)",
                 "balanced_acc", "deusser", num=19, n_total=29320, gated=True, jev_published="0.786 pooled",
                 question_kinds=("noul",)),
    BenchDataset("climate_fever", "test", "F", "Climate-FEVER", "tdiggelm/climate_fever", None, ("test",),
                 "check upstream", "unknown", 1535, "all", "choice(4), state = claim + 5 evidence sentences", "acc",
                 "programmatic", num=20, n_total=1535, question_kinds=("choice",), shuffle_rotation=True),
    BenchDataset("toxic_chat", "test", "F", "ToxicChat", "lmsys/toxic-chat", "toxicchat0124", ("test",),
                 "CC BY-NC 4.0", "research", 5083, "all", "noul toxic (+ jailbreak noul, secondary)", "f1_pos",
                 "deusser", num=21, n_total=5083, jev_published="0.786", question_kinds=("noul", "noul")),
    BenchDataset("openai_moderation", "test", "F", "OpenAI moderation eval", "mmathys/openai-moderation-api-evaluation",
                 None, ("train",), "MIT", "commercial", 1680, "all", "8 nouls per state", "mean_auprc", "deusser",
                 num=22, n_total=1680, jev_published="0.717", question_kinds=("noul",) * 8),
    BenchDataset("typed_decisions", "test", "G", "typed-decisions", "LocalLLaMA/typed-decisions", "all", ("test",),
                 "Apache-2.0", "commercial", 400, "all", "native, 5 questions per state, soft teacher gold",
                 "acc", "native", num=23, n_total=400, jev_published="0.727",
                 question_kinds=("choice", "noul", "score")),
    BenchDataset("unfair_tos", "test", "G", "UNFAIR-ToS", "coastalcph/lex_glue", "unfair_tos", ("test",), "CC BY 4.0",
                 "commercial", 1607, "all", "8 nouls per clause", "micro_f1", "deusser", num=24, n_total=1607,
                 jev_published="0.499", question_kinds=("noul",) * 8),
    BenchDataset("helpsteer2", "test", "G", "HelpSteer2", "nvidia/HelpSteer2", None, ("validation",), "CC BY 4.0",
                 "commercial", 1038, "all", "5 scores (0-4) per state", "mean_spearman", "deusser", num=25,
                 n_total=1038, jev_published="0.412", question_kinds=("score",) * 5),
)

# ---------------------------------------------------------------------------------------------------------
# jevbench-dev, CLEAN set (benchmax PLAN §2.3.1): datasets with no published Jev number (SNIPS is only a
# component of LangWatch's off-topic mix, a context row; declared). Checkpoint selection only; excluded from
# training on both tracks. Jev is never run on dev.

DEV: tuple[BenchDataset, ...] = (
    BenchDataset("newsgroups20", "dev", "A", "20 Newsgroups", "SetFit/20_newsgroups", None, ("test",), "unknown",
                 "unknown", 2000, "strat", "choice(20)", "acc", "programmatic", question_kinds=("choice",)),
    BenchDataset("tweeteval_sentiment", "dev", "B", "TweetEval sentiment", "cardiffnlp/tweet_eval", "sentiment",
                 ("test",), "SemEval-17 terms", "research", 2000, "strat", "choice(3)", "acc", "deusser",
                 question_kinds=("choice",)),
    BenchDataset("atis", "dev", "D", "ATIS intents", "tuetschek/atis", None, ("test",), "unknown (academic)",
                 "unknown", 893, "all", "choice(intents present in test)", "acc", "programmatic",
                 question_kinds=("choice",),
                 notes="New in v1.1 (PLAN §2.3.1). Multi-intent rows ('a#b') keep the joint label as one option; "
                       "no published Jev number (jev-published.md §1)."),
    BenchDataset("snips", "dev", "D", "SNIPS", "benayas/snips", None, ("test",), "unknown", "unknown", 2000, "random",
                 "choice(7)", "acc", "programmatic", question_kinds=("choice",),
                 notes="Declared: SNIPS is one component of LangWatch's off-topic mix (targets.json lw_offtopic, "
                       "context row). Kept as dev per PLAN §2.3.1."),
    BenchDataset("mrpc", "dev", "E", "MRPC", "nyu-mll/glue", "mrpc", ("validation",), "unknown", "unknown", 408, "all",
                 "noul (same meaning?)", "acc", "deusser", question_kinds=("noul",)),
    BenchDataset("scitail", "dev", "E", "SciTail", "allenai/scitail", "snli_format", ("test",), "Apache-2.0",
                 "commercial", 2000, "strat", "noul (entailment)", "acc", "deusser", question_kinds=("noul",)),
    BenchDataset("cb", "dev", "E", "CommitmentBank", "aps/super_glue", "cb", ("validation",), "unknown", "unknown", 56,
                 "all", "choice(3)", "acc", "deusser", question_kinds=("choice",)),
    BenchDataset("tweeteval_offensive", "dev", "F", "TweetEval offensive", "cardiffnlp/tweet_eval", "offensive",
                 ("test",), "OffensEval-19 terms", "research", 860, "all", "noul (offensive)", "acc", "programmatic",
                 question_kinds=("noul",)),
    BenchDataset("scifact", "dev", "F", "SciFact dev", "allenai/scifact", "claims", ("validation",), "CC BY-NC 2.0",
                 "research", 300, "all", "choice(3) support/contradict/NEI over cited abstract", "acc",
                 "programmatic", question_kinds=("choice",),
                 notes="Stays (PLAN §2.3.1). BEIR SciFact test queries (BeIR/scifact, T387) are public_jev."),
    BenchDataset("tasksource_heldout", "dev", "G", "held-out tasksource-jev sources",
                 "tasksource/tasksource-jev-typed-decisions", None, ("train",), "mixed (per source)", "unknown", 2000,
                 "random", "native Jev-format rows of the DEV_TASKSOURCE_SOURCES families", "soft_acc", "native",
                 question_kinds=("choice", "noul", "score"),
                 notes="Whole source families listed in DEV_TASKSOURCE_SOURCES; data-real must drop them."),
)
# Also dev (owned by other agents): the data-synth agent's held-out synthetic workflows / domains (split
# "dev_family"). NLU++ folds 18-19 are NO LONGER dev: NLU++ is a public Jev benchmark (T112) -> public_jev.

# Retired from dev on 2026-10-03 (benchmax PLAN §2.3.1): each is a public-Jev test set. Kept here so the old
# bench/jevbench/dev/<key>__*.jsonl files can be identified and quarantined. Still excluded from training
# (now as public_jev:<key>); never used for selection again.
RETIRED_DEV: tuple[BenchDataset, ...] = (
    BenchDataset("tweet_topic", "retired_dev", "A", "tweet_topic_single", "cardiffnlp/tweet_topic_single", None,
                 ("test_2021",), "other", "unknown", 1693, "all", "choice(6)", "acc", "programmatic",
                 question_kinds=("choice",), notes="public Jev T245 (elcronos), test_2021 n=1,693"),
    BenchDataset("app_reviews", "retired_dev", "B", "app_reviews stars", "sealuzh/app_reviews", None, ("train",),
                 "unknown", "unknown", 2000, "strat", "score(5) stars", "acc_mode", "programmatic",
                 question_kinds=("score",), notes="public Jev T380 (goodrahstar); single split"),
    BenchDataset("hwu64", "retired_dev", "D", "HWU64", "DeepPavlov/hwu64", None, ("test",), "CC BY-SA 3.0",
                 "commercial", 2000, "random", "choice(64)", "acc", "programmatic", question_kinds=("choice",),
                 notes="public Jev T256 (thisisandreeeee), same 1,076-item test"),
    BenchDataset("toxigen", "retired_dev", "F", "ToxiGen annotated", "toxigen/toxigen-data", "annotated", ("test",),
                 "other", "research", 940, "all", "noul toxic + score(5)", "acc", "deusser", gated=True,
                 question_kinds=("noul", "score"), notes="public Jev T027 (Deußer)"),
    BenchDataset("prompt_injections", "retired_dev", "F", "deepset prompt-injections", "deepset/prompt-injections",
                 None, ("test",), "Apache-2.0", "commercial", 116, "all", "noul (injection)", "acc", "deusser",
                 question_kinds=("noul",), notes="public Jev T030 (Deußer) + all-662 rows"),
)

# ---------------------------------------------------------------------------------------------------------
# Shown, not counted (knowledge reference rows, PLAN §1.1). All four are also public Jev rows -> public_jev.

REF: tuple[BenchDataset, ...] = (
    BenchDataset("hellaswag", "ref", "K", "HellaSwag", "Rowan/hellaswag", None, ("validation",), "MIT", "commercial",
                 10042, "all", "choice(4)", "acc", "deusser", question_kinds=("choice",)),
    BenchDataset("winogrande", "ref", "K", "WinoGrande", "allenai/winogrande", "winogrande_xl", ("validation",),
                 "CC BY", "commercial", 1267, "all", "choice(2)", "acc", "deusser", question_kinds=("choice",)),
    BenchDataset("mmlu_pro", "ref", "K", "MMLU-Pro sample", "TIGER-Lab/MMLU-Pro", None, ("test",), "MIT", "commercial",
                 2000, "random", "choice(<=10)", "acc", "deusser", question_kinds=("choice",)),
    BenchDataset("bbh", "ref", "K", "BBH sample", "lukaemon/bbh", None, ("test",), "MIT", "commercial", 1000, "random",
                 "choice over parsed options", "acc", "deusser", question_kinds=("choice",)),
)

ALL_DATASETS: tuple[BenchDataset, ...] = TEST + DEV + REF
BY_KEY: dict[str, BenchDataset] = {d.key: d for d in ALL_DATASETS}
assert len(BY_KEY) == len(ALL_DATASETS), "duplicate dataset key"
assert not set(BY_KEY) & {d.key for d in RETIRED_DEV}, "retired dev key still active"
assert [d.num for d in TEST] == list(range(1, 26)), "test datasets must be numbered 1..25 in order"

# Whole tasksource-jev source families held out as dev (PLAN §1.2 "10 whole sources"). Picked with
# random.Random(SEED) from English families of 2k-12k train rows that no other rule excluded at the time
# (`pick_dev_families`). v1.1: "strategy-qa" was removed because StrategyQA is a public Jev benchmark
# (jev-bench T178/T179) and now falls under public_jev; the other nine are kept as they were (no re-pick, to
# keep the dev_family split stable).
DEV_TASKSOURCE_SOURCES: tuple[str, ...] = (
    "ade_corpus_v2", "conj_nli", "equate", "github-issue-similarity", "help-nli",
    "numer_sense", "proofwriter", "qasc", "resnli",
)
RETIRED_DEV_TASKSOURCE_SOURCES: tuple[str, ...] = ("strategy-qa",)  # -> public_jev:strategyqa_closed

# Families another bucket uses (PLAN §2.1), multilingual, or aggregators: never picked as dev families.
DEV_FAMILY_INELIGIBLE = frozenset({
    "glue", "super_glue", "snli", "WANLI", "lingnli", "vitaminc", "doc-nli", "nli_fever", "chaos-mnli-ambiguity",
    "dbpedia_14", "trec", "lex_glue", "conll2003", "civil_comments", "emo", "UltraFeedback-paired",
    "procedural-typed-decisions", "multilingual", "IntentGrasp",
})


def pick_dev_families(train_source_rows: dict[str, int], k: int = 10) -> list[str]:
    """The DEV_TASKSOURCE_SOURCES procedure: families (name before the first '/') that are eligible, not
    excluded by any other rule, with 2k-12k train rows; then random.Random(SEED).sample(sorted, k)."""
    import random

    fam: dict[str, list[tuple[str, int]]] = {}
    for s, n in train_source_rows.items():
        fam.setdefault(s.split("/")[0], []).append((s, n))
    elig = []
    for f, members in sorted(fam.items()):
        if f in DEV_FAMILY_INELIGIBLE or any(_rule_reason(normalize_source(s)) for s, _ in members):
            continue
        if 2000 <= sum(n for _, n in members) <= 12000:
            elig.append(f)
    return sorted(random.Random(SEED).sample(elig, k))


# ---------------------------------------------------------------------------------------------------------
# Exclusion rules (PLAN §1.6 stage 1). Patterns run on `normalize_source(name)` with re.search.

# Boundaries: (?<![a-z]) / (?![a-z]) keep short tokens from matching inside words ("rte" in "sorted",
# "anli" in "wanli"). Digits and separators count as boundaries, so "sst2", "glue/rte", "anli_r1" match.
_L = r"(?<![a-z])"
_R = r"(?![a-z])"

_TEST_RULES: dict[str, str] = {
    "ag_news": rf"{_L}ag[-_ ]?news",
    "yahoo_topics": r"yahoo",
    "fin_topic": r"twitter[-_ ]?financial[-_ ]?news",
    "sst2": rf"{_L}sst{_R}|stanford[-_ ]?sentiment[-_ ]?treebank",
    "fin_phrasebank": r"financial[-_ ]?phrase[-_ ]?bank|phrasebank|fingpt[-_]sentiment",
    "sst5": rf"{_L}sst{_R}",
    "yelp5": r"yelp",
    "dair_emotion": r"^(dair-ai/|mteb/|setfit/|tasksource/)?emotion(/(split|unsplit|default))?$|(?<![a-z])carer(?![a-z])",
    "tweeteval_emotion": r"tweet[-_ ]?eval|sem[-_ ]?eval[-_ ]?2018[-_ ]?task[-_ ]?1(?![0-9])|semeval[-_ ]?2018|affect[-_ ]in[-_ ]tweets",
    "go_emotions": r"go[-_ ]?emotions?",
    "banking77": r"banking",
    "clinc150": r"clinc",
    "massive": rf"{_L}massive{_R}|slurp",
    "boolq": r"boolq",
    "rte": rf"{_L}rte{_R}|recogni[sz]ing[-_ ]textual[-_ ]entailment",
    "anli": rf"{_L}anli{_R}|adversarial[-_ ]?nli",
    "paws": r"paws",
    "stsb": rf"stsb|{_L}sts[-_ ]b{_R}|sts[-_ ]?benchmark|{_L}sts[-_ ]?(?:20)?1[2-6](?![0-9])|sts[-_]companion",
    "llm_aggrefact": rf"aggre[-_ ]?fact|ragtruth|tofu[-_ ]?eval|{_L}wice{_R}|{_L}reveal{_R}|claim[-_ ]?verify"
                     rf"|factcheck[-_ ]?gpt|expert[-_ ]?qa|{_L}lfqa{_R}",
    "climate_fever": r"climate[-_ ]?fever",
    "toxic_chat": r"toxic[-_ ]?chat",
    "openai_moderation": r"openai[-_ ]?moderation|moderation[-_ ]?api",
    "typed_decisions": r"(?<![a-z0-9_-])typed[-_ ]decisions",  # not procedural-/synthetic-typed-decisions
    "unfair_tos": r"unfair|claudette",
    "helpsteer2": r"help[-_ ]?steer",
}

# Clean dev (v1.1). The TweetEval rules are config-specific so that the S track can still train on other
# tweet_eval configs by their qualified names ("tweet_eval/emotion"); the bare repo id is excluded on Z by the
# test rule above anyway.
_DEV_RULES: dict[str, str] = {
    "newsgroups20": r"20[-_ ]?news[-_ ]?groups|newsgroups",
    "tweeteval_sentiment": r"tweet[-_ ]?eval[/:._-]sentiment|tweet[-_]sentiment[-_]multilingual|sem[-_ ]?eval[-_ ]?2017[-_ ]?task[-_ ]?4",
    "atis": rf"{_L}atis{_R}",
    "snips": r"snips",
    "mrpc": rf"mrpc|msr[-_ ]?paraphrase",
    "scitail": r"scitail",
    "cb": r"(^|/)cb$|super[-_ ]?glue[/:._-]cb(?![a-z])|commitment[-_ ]?bank",
    "tweeteval_offensive": r"tweet[-_ ]?eval[/:._-]offensive|offenseval[-_ ]?2019|(?<![a-z])olid(?![a-z])",
    "scifact": r"scifact",
}

# Derived or same-text sources not caught by the name rules above (PLAN §1.6, classifier-benchmarks §4.2).
_SIBLING_RULES: dict[str, str] = {
    "btzsc": r"btzsc",  # BTZSC re-releases AG News, Yahoo, FPB, emotion, Banking77, MASSIVE, Yelp, RT, app reviews
    "rotten_tomatoes": r"rotten",  # same Pang & Lee review sentences as SST
    "laurer_train_nli": r"dataset[-_]train[-_]nli",  # Laurer's NLI recast of 28 classifier train splits
    "multilingual_nli_26lang": r"multilingual[-_ ]?nli[-_ ]?26lang",  # contains machine-translated ANLI
    "sst_mirrors": r"gpt3mix/sst|sentiment[-_ ]?treebank",
}

_JEV_LABELLED_RULES: dict[str, str] = {
    "jev_labelled": r"jev[-_ ]?distill|yuri[-_ ]?v3|bekko|jev[-_ ]?benchmarking|runs/compare|(?<![0-9])23039006(?![0-9])"
                    r"|autotrust/jev|open[-_ ]?jev[-_ ]?deberta",
}

# External typed-decision suites we compare against (never trained on, never waived; S track: none of them
# ships a usable train split). The knowledge reference rows that used to live here (hellaswag, winogrande,
# mmlu, bbh, daily_dialog, vast, isarcasm, nli4ct, hover) are public Jev rows and moved to public_jev.
_RESERVED_RULES: dict[str, str] = {
    "external_typed_suites": r"classifier[-_]benchmark|jevbench|decision[-_]index|kev[-_]suites"
                             r"|fast[-_]decisions|laya[-_]evals|decision[-_]?bench|decide[-_]?bench|workflow[-_]?evals"
                             r"|evalsafe",
}  # Praveenrajus/jev-bench is NOT here: it ships train splits (public_jev, S-allowed train configs only)

# --------------------------------------------------------------------------------------------------------
# public_jev, hand-written part: sibling / derived / mirror patterns per benchmax PLAN §2.3.2.
#   value = (pattern, targets.json benchmark_key or None). The key is what the S track looks up.
# Sources: suite-reproduction-specs.md §1.5 (new patterns), train-data-and-supervised-ceilings.md §7 (traps),
# benchmax PLAN §0.4 (v2-mix overlap list), the retired dev sets, and the former `reserved` knowledge rows.
PUBLIC_JEV_RULES: dict[str, tuple[str, str | None]] = {
    # the one public Jev suite that ships train splits (train-data §11.1); S: train configs only (S_DENY_RULES)
    "jev_bench": (r"praveenrajus|jev[-_ ]bench", "jevbench_hf_macro"),
    # retired dev sets (PLAN §2.3.1)
    "tweet_topic": (r"tweet[-_ ]?topic", "tweet_topic"),
    "app_reviews": (r"app[-_ ]?reviews", "app_reviews"),
    "hwu64": (rf"{_L}hwu", "hwu64"),
    "toxigen": (r"toxigen", "toxigen"),
    "prompt_injections": (r"prompt[-_ ]?injection|prompt[-_ ]?shield", "prompt_injections"),
    "nlupp": (r"nlu[-_ ]?(\+\+|pp|plus[-_ ]?plus)", "nlupp"),
    # former reserved knowledge rows
    "hellaswag": (r"hella[-_ ]?swag", "hellaswag"),
    "winogrande": (r"wino[-_ ]?grande", "winogrande"),
    "mmlu": (rf"{_L}mmlu(?![-_ ]?(?:pro|cf))", "mmlu"),  # cais/mmlu, tasksource/mmlu (aux_train is the S source)
    "mmlu_pro": (r"mmlu[-_ ]?pro", "mmlu_pro"),
    "mmlu_cf": (r"mmlu[-_ ]?cf", "mmlu_cf"),
    "tmmlu": (r"tmmlu", "tmmlu"),
    "bbh": (rf"{_L}bbh{_R}|big[-_ ]?bench", "bbh"),
    "daily_dialog": (r"daily[-_ ]?dialog|silicone[/:._-]dyda", "daily_dialog"),
    "vast": (rf"{_L}vast{_R}|zero[-_ ]?shot[-_ ]?stance", "vast"),
    "isarcasm": (r"isarcasm", "isarcasm"),
    "nli4ct": (r"nli4ct|semeval[-_ ]?2024[-_ ]?task[-_ ]?2", "nli4ct"),
    "hover": (rf"{_L}hover{_R}", "hover"),
    # suite-reproduction-specs.md §1.5
    "arc": (r"ai2[-_ ]?arc|arc[-_ ]?(easy|challenge)", "arc"),
    "csqa": (r"commonsense[-_ ]?qa|(?<![a-z])csqa(?![a-z])", "csqa"),
    "alphanli": (r"alpha[-_ ]?nli|abductive[-_ ]?nli", "alphanli"),  # 'art' itself: PUBLIC_JEV_REPOS segment match
    "pubmedqa": (r"pubmed[-_ ]?qa|pqa[-_ ]?(labeled|artificial|unlabeled)", "pubmedqa"),
    "sib200": (r"sib[-_ ]?200", "sib200"),
    "belebele": (r"belebele", "belebele"),
    "xnli": (r"afri[-_ ]?xnli|(?<![a-z])xnli", "xnli"),
    "ceval": (rf"{_L}c[-_ ]?eval{_R}", "ceval"),
    "language_id": (r"language[-_ ]?identification|papluca", "language_id"),
    "agb_de": (rf"{_L}agb{_R}", "agb_de"),
    "summeval": (r"summ[-_ ]?eval", "summeval"),
    "sms_spam": (r"sms[-_ ]?spam", "sms_spam"),
    "contractnli": (r"contract[-_ ]?nli", "contractnli"),
    "cladder": (r"cladder", "cladder"),
    "esci": (rf"{_L}esci{_R}", "esci"),
    "humicroedit": (r"humicroedit|funlines", "humicroedit"),
    "enron": (rf"enron[-_ ]?spam|{_L}enron{_R}", "enron"),
    "phishnchips": (r"phish[-_ ]?n[-_ ]?chips", "phishnchips"),
    "finentity": (r"fin[-_ ]?entity", "finentity"),
    "acos": (rf"{_L}acos{_R}", "acos"),
    "ragtruth": (r"rag[-_ ]?truth", "ragtruth"),
    "when2call": (r"when[-_ ]?2[-_ ]?call", "when2call"),
    "newyorker": (r"new[-_ ]?yorker", "newyorker"),
    "chaosnli": (r"chaos[-_ ]?nli|chaos[-_ ]?mnli", "chaosnli"),
    # benchmax PLAN §0.4 v2-mix overlap list (train-data §12.2)
    "imdb": (rf"{_L}imdb", "imdb"),
    "civil_comments": (r"civil[-_ ]?comments", "civil_comments"),
    "mhs": (r"measuring[-_ ]?hate[-_ ]?speech|(?<![a-z])mhs(?![a-z])", "mhs"),
    "aegis": (r"aegis", "aegis2_prompt"),
    "ledgar": (r"ledgar", "ledgar"),
    "mnli": (rf"{_L}mnli|multi[-_ ]?nli", "mnli"),
    "fever": (rf"{_L}fever{_R}", "fever"),
    "wanli": (r"wanli", "wanli"),
    "trec": (rf"{_L}trec{_R}", "trec_fine"),
    "conll": (r"conll", "conll_typing"),
    "squad": (r"squad", "squad_select"),
    "mtop": (rf"{_L}mtop", "label_pressure"),
    "dbpedia": (r"dbpedia", "label_pressure"),
    "amazon_polarity": (r"amazon[-_ ]?polarity", "amazon_polarity"),
    "amazon_counterfactual": (r"amazon[-_ ]?counterfactual", "amazon_counterfactual"),
    "casehold": (r"case[-_ ]?hold", "casehold"),
    "arxiv": (r"arxiv", "arxiv_2026"),  # judgment call: same source/task as T253 (flagged in targets.json)
    "strategyqa": (r"strategy[-_ ]?qa", "strategyqa_closed"),
    # train-data-and-supervised-ceilings.md §7 traps and parents/children (Z: siblings; S: see S_DENY_RULES)
    "text_moderation": (r"text[-_ ]?moderation|koalaai", "openai_mod"),
    "fpb_rereleases": (r"nickmuchi|phrasebank[-_ ]and[-_ ]sentfin|auditor[-_ ]?(review|sentiment)", "fpb"),
    "sata": (r"sata[-_ ]?bench", "sata"),
    "theoremqa_scibench": (r"theorem[-_ ]?qa|sci[-_ ]?bench", "mmlu_pro"),
    "gpqa": (r"gpqa", "gpqa"),
    "bfcl": (rf"{_L}bfcl|berkeley[-_ ]?function[-_ ]?calling|gorilla", "bfcl"),
    "api_bank": (r"api[-_ ]?bank", "api_bank"),
    "toolret": (r"tool[-_ ]?ret", "toolret"),
    "chatbot_arena": (r"chatbot[-_ ]?arena|arena[-_ ]?human[-_ ]?preference|lmarena", "ppe"),
    "prm800k": (r"prm[-_ ]?800k", "prmbench"),
    "aggregators": (rf"{_L}flan{_R}|tasksource[-_ ]?instruct", None),  # fold benchmark validation into train
    # the other benchmark families of targets.json (name variants the key tokens and hf ids miss)
    "lex_glue": (r"lex[-_ ]?glue", "lexglue_mean"),
    "ecthr": (r"ecthr", "ecthr_a"),
    "scotus": (r"scotus", "scotus"),
    "eurlex": (r"eur[-_ ]?lex", "eurlex"),
    "cuad": (rf"{_L}cuad", "cuad"),
    "medhallu": (r"med[-_ ]?hallu", "medhallu"),
    "medqa": (rf"{_L}med[-_ ]?qa|meta[-_ ]?med[-_ ]?qa", "metamedqa"),  # MetaMedQA extends MedQA-USMLE test
    "diagnosisarena": (r"diagnosis[-_ ]?arena", "diagnosisarena"),
    "wildguard": (r"wild[-_ ]?guard", "wildguard"),
    "harmbench": (r"harm[-_ ]?bench", "harmbench_resp"),
    "hatecheck": (r"hate[-_ ]?check", "hatecheck"),
    "xstest": (r"xs[-_ ]?test", "xstest"),
    "injection_corpora": (r"jailbreak[-_ ]?classification|(?<![a-z])spml|jailbreaks[-_ ]?over[-_ ]?time|not[-_ ]?inject"
                          r"|safe[-_ ]?guard[-_ ]?prompt|wa[-_ ]?inject", "injection_combined"),
    "rewardbench": (r"reward[-_ ]?bench", "rewardbench1"),
    "rmbench": (r"rm[-_ ]?bench", "rmbench_pair"),
    "ppe": (r"ppe[-_ ]?human[-_ ]?preference", "ppe"),
    "processbench": (r"process[-_ ]?bench", "processbench"),
    "prmbench": (r"prm[-_ ]?bench", "prmbench"),
    "rubricbench": (r"rubric[-_ ]?bench", "rubricbench"),
    "halueval": (r"halu[-_ ]?eval", "halueval_qa"),
    "nevir": (r"nevir", "nevir"),
    "beir": (rf"{_L}beir{_R}|nf[-_ ]?corpus|{_L}fiqa|code[-_ ]?search[-_ ]?net", "rerank_mean"),
    "msmarco": (r"ms[-_ ]?marco", "msmarco"),  # conflicts with the "MS MARCO (Z-eligible)" note in train_sources: PLAN §2.1 wins
    "nanorteb": (r"nano[-_ ]?rteb|hakari", "nanorteb"),
    "xquad": (r"xquad", "xquad_rerank"),
    "sgd": (rf"schema[-_ ]?guided|{_L}sgd[-_ ]?x?{_R}|dstc8", "sgd"),
    "routerbench": (r"router[-_ ]?bench", "routerbench"),
    "chessbench": (r"chess[-_ ]?bench|searchless[-_ ]?chess|lichess", "chessbench"),
    "pop909": (r"pop909", "pop909"),
    "cfcolor": (r"cfcolor", "cfcolor"),
    "bpomp": (r"bpomp", "bpomp"),
    "habermas": (r"habermas", "habermas"),
    "hle": (rf"{_L}hle{_R}|humanity.?s[-_ ]last[-_ ]exam", "hle"),
    "musr": (rf"{_L}musr{_R}", "musr"),
    "cruxeval": (r"crux[-_ ]?eval", "cruxeval"),
    "forecastbench": (r"forecast[-_ ]?bench", "forecastbench"),
    "gsm8k": (r"gsm[-_ ]?8k", "gsm8k_mc"),
    "obqa": (r"open[-_ ]?book[-_ ]?qa|(?<![a-z])obqa", "obqa"),
    "race": (rf"{_L}race(?:[-_ ]?(?:c|h|m|high|middle))?{_R}", "race_h"),
    "logiqa": (r"logiqa", "logiqa"),
    "truthfulqa": (r"truthful[-_ ]?qa", "truthfulqa"),
    "triviaqa": (r"trivia[-_ ]?qa", "triviaqa4"),
    "popqa": (r"pop[-_ ]?qa", "popqa4"),
    "simpleqa": (r"simple[-_ ]?qa", "simpleqa4"),
    "daily_oracle": (r"daily[-_ ]?oracle", "daily_oracle"),
    "tabfact": (r"tab[-_ ]?fact", "tabfact"),
    "multihop": (r"musique|hotpot[-_ ]?qa", "multihop_retrieval"),
    "mind2web": (r"mind2web", "mind2web"),
    "alfworld": (r"alfworld", "alfworld"),
    "agent_safety": (r"mcp[-_ ]?hunt|(?<![a-z])r[-_ ]judge|atbench|tracesafe|agent[-_ ]?harm|injec[-_ ]?agent",
                     "mcphunt"),
    "cyberseceval": (r"cyber[-_ ]?sec[-_ ]?eval|purple[-_ ]?llama", "cyberseceval_cwe"),
    "rexerr": (r"rexerr|rad[-_ ]?eval", "rexerr"),
    "ricechem": (r"ricechem", "ricechem"),
    "dimabsa": (r"dimabsa|semeval[-_ ]?2026[-_ ]?task[-_ ]?3", "dimabsa_va"),
    "cesnet": (r"cesnet|quicext", "cesnet"),
    "nslkdd": (r"nsl[-_ ]?kdd", "nslkdd"),
    "cfpb": (r"consumer[-_ ]?(finance[-_ ]?)?complaints|cfpb", "cfpb"),
    "upworthy": (r"upworthy", "upworthy"),
    "gaokao": (r"gaokao", "gaokao"),
    "jmedqa": (r"jmedqa|igaku[-_ ]?qa", "jmedqa"),
    "exams": (r"thai[-_ ]?exam|jmmlu|(?<![a-z])enem(?![a-z])|jfinqa", "exams_misc"),
    "kobest": (r"kobest", "kobest"),
    "kobbq": (r"kobbq", "kobbq"),
    "ajgt": (r"ajgt", "ajgt"),
    "darija": (r"darija", "darija"),
    "cjbench": (r"cj[-_ ]?bench", "cjbench"),
    "email_spam": (r"ling[-_ ]?spam|spam[-_ ]?assassin|email[-_ ]?spam", "email_spam"),
    "phishing_email": (r"phishing[-_ ]?email", "phishing_small"),
    "fake_jobs": (r"fake[-_ ]?job", "fake_jobs"),
    "whowhen": (r"who[-_&]?when", "whowhen"),
    "aita": (rf"{_L}aita{_R}", "aita"),
    "commonlit": (r"commonlit", "commonlit"),
    "dblp_acm": (r"dblp[-_ ]?acm", "dblp_acm"),
    "nejm": (rf"{_L}nejm", "nejm"),
    "healthbench": (r"health[-_ ]?bench", "healthbench"),
    "judgebench": (r"judge[-_ ]?bench", "judgebench"),
    "sys1cal": (r"sys1cal", "sys1cal"),
    "xdviolence": (r"xd[-_ ]?violence", "xdviolence"),
    "har": (r"pamap2|(?<![a-z])wisdm|uci[-_ ]?har", "har"),
    "video_quality": (r"vqdb|jevqa", "video_quality"),
    "jevout": (r"tom[-_ ]?bench|super[-_ ]?gpqa|lar[-_ ]?echr", "jevout"),
    "kiki_agent": (r"canitrustyou", "kiki_agent"),
    "amazonqa": (r"amazon[-_ ]?qa", "amazonqa"),
    "p4g": (rf"persuasion[-_ ]?for[-_ ]?good|{_L}p4g|{_L}persuasion{_R}", "p4g"),
    "craigslist": (r"craigslist", "craigslist"),
    "semif": (rf"{_L}semif{_R}", "nimble"),
    "recsys": (r"amazon[-_ ]?reviews[-_ ]?2023", "recsys"),
    "pii": (r"gretel[-_ ]?pii|(?<![a-z])privy(?![a-z])|nemotron[-_ ]?pii", "lw_pii"),
    "cohen": (r"cohen[-_ ]?2006|asreview", "cohen_adhd"),
    "css_ziems": (r"convokit|conversations[-_ ]?gone[-_ ]?awry|wiki[-_ ]?politeness|wikipedia[-_ ]?politeness"
                  r"|implicit[-_ ]?hate|media[-_ ]?ideology|reddit[-_ ]?humor|tempowic|(?<![a-z])raop(?![a-z])"
                  r"|(?<![a-z])flute(?![a-z])|indian[-_ ]?english[-_ ]?dialect|(?<![a-z])ibc(?![a-z])"
                  r"|(?<![a-z])mrf(?![a-z])|semeval[-_ ]?stance|talklife|(?<![a-z])tropes(?![a-z])", "css_big"),
}

# Hand-written exact ids (normalised). Checked before any pattern, so a specific category wins
# (e.g. BeIR/scifact is public_jev while allenai/scifact stays dev).
EXCLUDED_DATASET_IDS: dict[str, str] = {
    # test datasets and mirrors
    "fancyzhx/ag_news": "test:ag_news", "sh0416/ag_news": "test:ag_news",
    "community-datasets/yahoo_answers_topics": "test:yahoo_topics",
    "zeroshot/twitter-financial-news-topic": "test:fin_topic",
    "zeroshot/twitter-financial-news-sentiment": "sibling:fin_topic",
    "stanfordnlp/sst2": "test:sst2", "stanfordnlp/sst": "test:sst2", "setfit/sst2": "test:sst2",
    "setfit/sst5": "test:sst5",
    "atrost/financial_phrasebank": "test:fin_phrasebank", "takala/financial_phrasebank": "test:fin_phrasebank",
    "nickmuchi/financial-classification": "sibling:fin_phrasebank",
    "fingpt/fingpt-sentiment-train": "sibling:fin_phrasebank",
    "yelp/yelp_review_full": "test:yelp5", "fancyzhx/yelp_polarity": "sibling:yelp5",
    "dair-ai/emotion": "test:dair_emotion", "mteb/emotion": "test:dair_emotion", "setfit/emotion": "test:dair_emotion",
    "cardiffnlp/tweet_eval": "test:tweeteval_emotion",
    "semevalworkshop/sem_eval_2018_task_1": "sibling:tweeteval_emotion",
    "google-research-datasets/go_emotions": "test:go_emotions",
    "mteb/banking77": "test:banking77", "polyai/banking77": "test:banking77",
    "clinc/clinc_oos": "test:clinc150", "deeppavlov/clinc150": "test:clinc150",
    "mteb/amazon_massive_intent": "test:massive", "mteb/amazon_massive_scenario": "test:massive",
    "amazonscience/massive": "test:massive", "qanastek/massive": "test:massive",
    "google/boolq": "test:boolq",
    "nyu-mll/glue/rte": "test:rte",
    "facebook/anli": "test:anli",
    "google-research-datasets/paws": "test:paws", "google-research-datasets/paws-x": "sibling:paws",
    "sentence-transformers/stsb": "test:stsb", "mteb/stsbenchmark-sts": "test:stsb",
    "lytang/llm-aggrefact": "test:llm_aggrefact",
    "tdiggelm/climate_fever": "test:climate_fever",
    "lmsys/toxic-chat": "test:toxic_chat",
    "mmathys/openai-moderation-api-evaluation": "test:openai_moderation",
    "localllama/typed-decisions": "test:typed_decisions",
    "coastalcph/lex_glue/unfair_tos": "test:unfair_tos",
    "nvidia/helpsteer": "test:helpsteer2", "nvidia/helpsteer2": "test:helpsteer2", "nvidia/helpsteer3": "test:helpsteer2",
    # clean dev
    "setfit/20_newsgroups": "dev:newsgroups20", "cardiffnlp/tweet_eval/sentiment": "dev:tweeteval_sentiment",
    "cardiffnlp/tweet_eval/offensive": "dev:tweeteval_offensive", "tuetschek/atis": "dev:atis",
    "benayas/snips": "dev:snips", "nyu-mll/glue/mrpc": "dev:mrpc", "allenai/scitail": "dev:scitail",
    "aps/super_glue/cb": "dev:cb", "allenai/scifact": "dev:scifact",
    # retired dev -> public_jev
    "cardiffnlp/tweet_topic_single": "public_jev:tweet_topic", "cardiffnlp/tweet_topic_multi": "public_jev:tweet_topic",
    "sealuzh/app_reviews": "public_jev:app_reviews", "deeppavlov/hwu64": "public_jev:hwu64",
    "toxigen/toxigen-data": "public_jev:toxigen", "deepset/prompt-injections": "public_jev:prompt_injections",
    "beir/scifact": "public_jev:scifact",
    # siblings and Jev-labelled data
    "moritzlaurer/dataset_train_nli": "sibling:laurer_train_nli",
    "moritzlaurer/multilingual-nli-26lang-2mil7": "sibling:multilingual_nli_26lang",
    "btzsc/btzsc": "sibling:btzsc",
    "cornell-movie-review-data/rotten_tomatoes": "sibling:rotten_tomatoes",
    "sargedev/jev-distill-corpus-v3": "jev_labelled",
    "hotchpotch/bekko-system-one-dataset-v0": "jev_labelled",
}

# Field-level exclusions: these datasets may be used, but never these columns (they hold Jev outputs).
JEV_LABELLED_FIELDS: dict[str, tuple[str, ...]] = {
    "tasksource/synthetic-typed-decisions": ("answers",),
}

# S-track deny-list (train-data-and-supervised-ceilings.md §7; targets.json EXCLUDE/NEVER clauses). These are
# never waived on S even when the dataset itself is S-eligible: they contain the evaluated split, or were
# trained on it, or fold benchmark validation into training.
S_DENY_RULES: dict[str, str] = {
    "fpb_rereleases": r"nickmuchi|phrasebank[-_ ]and[-_ ]sentfin|auditor[-_ ]?(review|sentiment)|fingpt",
    "emotion_unsplit": r"emotion[/:._-]unsplit|(?<![a-z])unsplit(?![a-z])",
    "text_moderation": r"text[-_ ]?moderation|koalaai",
    "cladder_release": r"cladder",  # the released set IS the DI sample (S+generator only)
    "sata_bench": r"sata[-_ ]?bench",
    "finentity": r"fin[-_ ]?entity",  # single split = the eval
    "pqa_labeled": r"pqa[-_ ]?labeled",  # Deußer / Jevals eval; MedHallu test
    "sms_spam": r"sms[-_ ]?spam",  # single split = the eval (S-cv only)
    "mnli_validation_matched": r"validation[-_ ]?matched",  # ChaosNLI-M and jev-bench MNLI test
    "mmlu_pro_parents": r"theorem[-_ ]?qa|sci[-_ ]?bench",
    "gpqa_main_extended": r"gpqa",
    "chatbot_arena": r"chatbot[-_ ]?arena|arena[-_ ]?human[-_ ]?preference|lmarena",  # PPE
    "aggregators": rf"dataset[-_]train[-_]nli|{_L}flan{_R}|tasksource[-_ ]?instruct|multilingual[-_ ]?nli[-_ ]?26lang"
                   r"|intentgrasp|tasksource[-_ ]?dpo",
    "eval_rereleases": r"btzsc|jevals|jevbench|decision[-_ ]?index",
    "jev_bench_test_configs": r"jev[-_ ]bench[/:._-](test|validation)",  # Praveenrajus/jev-bench: train configs only
    "chaosnli_release": r"chaos[-_ ]?nli|chaos[-_ ]?mnli",  # the eval itself; S trains on SNLI + MNLI train
    "mhs_single_split": r"measuring[-_ ]?hate[-_ ]?speech|(?<![a-z])mhs(?![a-z])",  # pool holds the jev-bench test; use jev-bench train
    "beir_scifact_queries": r"beir[/:._-]scifact",  # BEIR test queries; S trains on allenai/scifact train claims
}

# S allow-list entries that targets.json names only in prose (no owner/name id to extract): pattern -> key.
S_ALLOW_RULES: dict[str, tuple[str, str]] = {
    "msmarco": (r"ms[-_ ]?marco", "rerank_mean"),  # "MS MARCO (Z-eligible)" in scifact/nfcorpus/rerank_mean train_sources
    "medqa": (rf"{_L}med[-_ ]?qa[-_ ]?usmle|medmcqa", "metamedqa"),  # "MedQA-USMLE, MedMCQA (dedupe)" (related data)
}

# jevbench keys -> targets.json benchmark_key (for the S track lookup; None = no public Jev row, never waived).
JEVBENCH_TO_TARGET: dict[str, str | None] = {
    "ag_news": "ag_news", "yahoo_topics": None, "fin_topic": "fin_topic", "sst2": "sst2", "fin_phrasebank": "fpb",
    "sst5": "sst5", "yelp5": "yelp5", "dair_emotion": "dair_emotion", "tweeteval_emotion": "tweeteval_emotion",
    "go_emotions": "goemotions", "banking77": "banking77", "clinc150": "clinc150", "massive": "massive_en",
    "boolq": "boolq", "rte": None, "anli": "anli", "paws": "paws", "stsb": "stsb", "llm_aggrefact": "llm_aggrefact",
    "climate_fever": None, "toxic_chat": "toxicchat", "openai_moderation": "openai_mod",
    "typed_decisions": "typed_decisions", "unfair_tos": "unfair_tos", "helpsteer2": "helpsteer2",
    # dev (only SciFact has a public Jev row: BEIR SciFact rerank, train claims allowed in S)
    "scifact": "scifact", "snips": None, "newsgroups20": None, "tweeteval_sentiment": None, "atis": None,
    "mrpc": None, "scitail": None, "cb": None, "tweeteval_offensive": None, "tasksource_heldout": None,
    # siblings
    "rotten_tomatoes": "rotten_tomatoes", "fin_topic_sibling": None, "btzsc": None, "laurer_train_nli": None,
    "multilingual_nli_26lang": None, "sst_mirrors": None,
}

# Kept on purpose, although training-data.md §5.3 listed them (they map to no jevbench test or dev set).
# v1.1: imdb, counterfactually-augmented-imdb, amazon_polarity, amazon_counterfactual, sms_spam, mtop and
# civil_comments left this list: they are public Jev rows now (public_jev). Stages 2-3 still dedup the rest.
KEPT_DESPITE_EARLIER_LIST: tuple[str, ...] = (
    "multilingual/amazon_reviews_multi", "hate_speech_offensive", "google_wellformed_query", "toxic_conversations",
    "jigsaw_toxicity", "SBIC_Disagreement",
)

# >>> GENERATED public_jev (scripts/benchmax_build_targets.py --write-registry; do not edit by hand) >>>
PUBLIC_JEV_TARGETS_SHA256 = "9e1552f61a92b4bab30019c2bddfe1b2a11e49e34819a135d9c22a3a47d63a91"
PUBLIC_JEV_IDS: dict[str, str] = {
    "agentic-learning-ai-lab/daily-oracle": "daily_oracle",
    "ai-safety-institute/agentharm": "agent_trace",
    "ai-systems/task-2-semeval-2024": "nli4ct",
    "akariasai/popqa": "popqa4",
    "alibabaresearch/damo-convai": "api_bank",
    "alisawuffles/wanli": "wanli",
    "allenai/ai2_arc": "arc",
    "allenai/art": "alphanli",
    "allenai/openbookqa": "obqa",
    "allenai/reward-bench": "rewardbench1",
    "allenai/reward-bench-2": "rewardbench2",
    "allenai/scifact": "scifact_claims",
    "allenai/wildguardmix": "wildguard",
    "allenai/winogrande": "winogrande",
    "amazon-science/esci-data": "esci",
    "amazonscience/massive": "massive_scenario51",
    "andeytait/jevforge-mind2web": "lw_webagent",
    "andyweasley2004/pop909-cl-dataset": "pop909",
    "apolinario/decision-index": "di_index",
    "arelit/phishnchips": "phishnchips",
    "atmaneayoub/jev-ar-bench": "gulf_arabic",
    "atrost/financial_phrasebank": "fpb",
    "bdsaglam/musique": "multihop_retrieval",
    "bee-spoke-data/consumer-finance-complaints": "lw_complaint",
    "beir/fiqa": "fiqa_rerank",
    "beir/nfcorpus": "nfcorpus",
    "beir/scifact": "scifact",
    "beki/privy": "lw_pii",
    "benayas/snips": "lw_offtopic",
    "btzsc/btzsc": "btzsc_ag",
    "cais/hle": "hle",
    "cais/mmlu": "mmlu",
    "cardiffnlp/tweet_eval": "tweeteval_emotion",
    "cardiffnlp/tweet_topic_single": "tweet_topic",
    "causalnlp/cladder": "cladder",
    "ceval/ceval-exam": "ceval",
    "choyiny/decidebench": "decidebench",
    "clinc/clinc_oos": "clinc150",
    "clinc/oos-eval": "clinc150",
    "coastalcph/lex_glue": "unfair_tos",
    "cogcomp/trec": "trec_fine",
    "cornell-movie-review-data/rotten_tomatoes": "rotten_tomatoes",
    "d4br4/agb-de": "agb_de",
    "dair-ai/emotion": "dair_emotion",
    "davlan/sib200": "sib200",
    "deepset/prompt-injections": "prompt_injections",
    "determined-ai/consumer_complaints_medium": "cfpb",
    "djapp18/jailbreaksovertime": "lw_injection",
    "earino/chaosnli": "diag",
    "ehovy/race": "race_h",
    "emilyallaway/zero-shot-stance": "vast",
    "facebook/anli": "anli",
    "facebook/belebele": "belebele",
    "facebook/xnli": "xnli",
    "facebookresearch/cruxeval": "cruxeval",
    "fancyzhx/ag_news": "ag_news",
    "fancyzhx/amazon_polarity": "amazon_polarity",
    "forecastingresearch/forecastbench-datasets": "forecastbench",
    "fstandhartinger/jevbench": "jevbench_composite",
    "google-deepmind/habermas_machine": "habermas",
    "google-deepmind/searchless_chess": "chessbench",
    "google-research-datasets/dstc8-schema-guided-dialogue": "sgd",
    "google-research-datasets/go_emotions": "goemotions",
    "google-research-datasets/paws": "paws",
    "google/boolq": "boolq",
    "google/civil_comments": "civil_comments",
    "google/simpleqa-verified": "simpleqa4",
    "google/xquad": "xquad_rerank",
    "gorilla-llm/berkeley-function-calling-leaderboard": "lw_toolrouting",
    "gorilla-llm/gorilla": "bfcl",
    "gretelai/gretel-pii-masking-en-v1": "lw_pii",
    "hanno-labs/decision-bench": "decisionbench",
    "hitsmy/prmbench_preview": "prmbench",
    "hotpotqa/hotpot_qa": "multihop_retrieval",
    "hover-nlp/hover": "hover",
    "iabufarha/isarcasmeval": "isarcasm",
    "idavidrein/gpqa": "gpqa",
    "ikala/tmmluplus": "tmmlu",
    "jackhhao/jailbreak-classification": "injection_combined",
    "jaredpalmer/kev": "kev",
    "jmhessel/newyorker_caption_contest": "newyorker",
    "kiddothe2b/contract-nli": "contractnli",
    "kikinlp/canitrustyou-jev": "kiki_agent",
    "komari6/ajgt_twitter_ar": "ajgt",
    "legacy-datasets/banking77": "banking77",
    "leolee99/notinject": "injection_combined",
    "li2017dailydialog/daily_dialog": "daily_dialog",
    "little-g-ai/sys1cal-v1": "sys1cal",
    "lmarena-ai/ppe-human-preference-v1": "ppe",
    "lmsys/toxic-chat": "toxicchat",
    "localllama/typed-decisions": "typed_decisions",
    "lucasmccabe/logiqa": "logiqa",
    "lytang/llm-aggrefact": "llm_aggrefact",
    "m-a-p/supergpqa": "jevout",
    "mandarjoshi/trivia_qa": "triviaqa4",
    "mangopy/toolret-queries": "toolret",
    "mangopy/toolret-tools": "toolret",
    "masakhane/afrixnli": "afrixnli",
    "maximegmd/metamedqa": "metamedqa",
    "mcauley-lab/amazon-reviews-2023": "recsys",
    "meta-llama/purplellama": "cyberseceval_cwe",
    "microsoft/mmlu-cf": "mmlu_cf",
    "microsoft/ms_marco": "msmarco",
    "mmathys/openai-moderation-api-evaluation": "openai_mod",
    "mteb/amazon_massive_intent": "lw_offtopic",
    "mteb/banking77": "banking77",
    "mteb/mtop_intent": "label_pressure",
    "mteb/summeval": "summeval",
    "naver-ai/kobbq": "kobbq",
    "nlp-waseda/jmmlu": "exams_misc",
    "nustm/acos": "acos",
    "nvidia/aegis-ai-content-safety-dataset-1.0": "aegis1_prompt",
    "nvidia/aegis-ai-content-safety-dataset-2.0": "aegis2_prompt",
    "nvidia/helpsteer2": "helpsteer2",
    "nvidia/nemotron-pii": "lw_pii",
    "nvidia/when2call": "when2call",
    "nyu-mll/glue": "mnli",
    "openai/gsm8k": "gsm8k_mc",
    "openai/healthbench": "healthbench",
    "openlmlab/gaokao-bench": "gaokao",
    "openrl/daily_dialog": "daily_dialog",
    "orionweller/nevir": "nevir",
    "osunlp/mind2web": "mind2web",
    "papluca/language-identification": "language_id",
    "particlemedia/ragtruth": "ragtruth",
    "paul/hatecheck": "hatecheck",
    "pminervini/halueval": "halueval_qa",
    "polyai-ldn/task-specific-datasets": "banking77",
    "polyai/banking77": "banking77",
    "praveenrajus/jev-bench": "chaosnli",
    "qiaojin/pubmedqa": "pubmedqa",
    "qwen/processbench": "processbench",
    "rajpurkar/squad": "squad_select",
    "reshabhs/spml": "lw_injection",
    "rowan/hellaswag": "hellaswag",
    "sata-bench/sata-bench": "sata",
    "scalerlab/judgebench": "judgebench",
    "scb10x/thai_exam": "exams_misc",
    "scienthoon/jev-ood-calibration": "nimble",
    "sealuzh/app_reviews": "app_reviews",
    "sentence-transformers/stsb": "stsb",
    "setfit/amazon_counterfactual": "amazon_counterfactual",
    "setfit/enron_spam": "enron",
    "setfit/sst5": "sst5",
    "setfit/trec-qc": "trec_coarse",
    "skt/kobest_v1": "kobest",
    "stanfordnlp/contract-nli": "contractnli",
    "stanfordnlp/imdb": "imdb",
    "stanfordnlp/sst2": "sst2",
    "suzgunmirac/big-bench-hard": "bbh",
    "tanaos/synthetic-intent-classifier-dataset-v1": "tanaos",
    "tasksource/bigbench": "bigbench",
    "tasksource/esci": "lw_search",
    "tasksource/mmlu": "mmlu",
    "tau/commonsense_qa": "csqa",
    "taur-lab/musr": "jevout",
    "theatticusproject/cuad": "cuad",
    "theoleecj/semif": "nimble",
    "thu-keg/rm-bench": "rmbench_pair",
    "tiger-lab/mmlu-pro": "mmlu_pro",
    "tner/conll2003": "conll_typing",
    "toxigen/toxigen-data": "toxigen",
    "truthfulqa/truthful_qa": "truthfulqa",
    "ucirvine/sms_spam": "sms_spam",
    "utaustin-aihealth/medhallu": "medhallu",
    "wenhu/tab_fact": "tabfact",
    "withmartian/routerbench": "routerbench",
    "xlangai/bright": "bright",
    "xtram1/safe-guard-prompt-injection": "injection_combined",
    "yelp/yelp_review_full": "yelp5",
    "yixuantt/finentity": "finentity",
    "zayne-sprague/musr": "musr",
    "zefang-liu/phishing-email-dataset": "phishing_small",
    "zeroshot/twitter-financial-news-topic": "fin_topic",
}
PUBLIC_JEV_REPOS: dict[str, str] = {
    "acos": "acos",
    "aegis-ai-content-safety-dataset-1.0": "aegis1_prompt",
    "aegis-ai-content-safety-dataset-2.0": "aegis2_prompt",
    "afrixnli": "afrixnli",
    "ag_news": "ag_news",
    "agb-de": "agb_de",
    "agentharm": "agent_trace",
    "ai2_arc": "arc",
    "ajgt_twitter_ar": "ajgt",
    "amazon-reviews-2023": "recsys",
    "amazon_counterfactual": "amazon_counterfactual",
    "amazon_massive_intent": "lw_offtopic",
    "amazon_polarity": "amazon_polarity",
    "anli": "anli",
    "app_reviews": "app_reviews",
    "art": "alphanli",
    "banking77": "banking77",
    "belebele": "belebele",
    "berkeley-function-calling-leaderboard": "lw_toolrouting",
    "big-bench-hard": "bbh",
    "bigbench": "bigbench",
    "boolq": "boolq",
    "bright": "bright",
    "btzsc": "btzsc_ag",
    "canitrustyou-jev": "kiki_agent",
    "ceval-exam": "ceval",
    "chaosnli": "diag",
    "civil_comments": "civil_comments",
    "cladder": "cladder",
    "clinc_oos": "clinc150",
    "commonsense_qa": "csqa",
    "conll2003": "conll_typing",
    "consumer-finance-complaints": "lw_complaint",
    "consumer_complaints_medium": "cfpb",
    "contract-nli": "contractnli",
    "cruxeval": "cruxeval",
    "cuad": "cuad",
    "daily-oracle": "daily_oracle",
    "daily_dialog": "daily_dialog",
    "damo-convai": "api_bank",
    "decidebench": "decidebench",
    "decision-bench": "decisionbench",
    "decision-index": "di_index",
    "dstc8-schema-guided-dialogue": "sgd",
    "emotion": "dair_emotion",
    "enron_spam": "enron",
    "esci": "lw_search",
    "esci-data": "esci",
    "financial_phrasebank": "fpb",
    "finentity": "finentity",
    "fiqa": "fiqa_rerank",
    "forecastbench-datasets": "forecastbench",
    "gaokao-bench": "gaokao",
    "glue": "mnli",
    "go_emotions": "goemotions",
    "gorilla": "bfcl",
    "gpqa": "gpqa",
    "gretel-pii-masking-en-v1": "lw_pii",
    "gsm8k": "gsm8k_mc",
    "habermas_machine": "habermas",
    "halueval": "halueval_qa",
    "hatecheck": "hatecheck",
    "healthbench": "healthbench",
    "hellaswag": "hellaswag",
    "helpsteer2": "helpsteer2",
    "hle": "hle",
    "hotpot_qa": "multihop_retrieval",
    "hover": "hover",
    "imdb": "imdb",
    "isarcasmeval": "isarcasm",
    "jailbreak-classification": "injection_combined",
    "jailbreaksovertime": "lw_injection",
    "jev-ar-bench": "gulf_arabic",
    "jev-bench": "chaosnli",
    "jev-ood-calibration": "nimble",
    "jevbench": "jevbench_composite",
    "jevforge-mind2web": "lw_webagent",
    "jmmlu": "exams_misc",
    "judgebench": "judgebench",
    "kev": "kev",
    "kobbq": "kobbq",
    "kobest_v1": "kobest",
    "language-identification": "language_id",
    "lex_glue": "unfair_tos",
    "llm-aggrefact": "llm_aggrefact",
    "logiqa": "logiqa",
    "massive": "massive_scenario51",
    "medhallu": "medhallu",
    "metamedqa": "metamedqa",
    "mind2web": "mind2web",
    "mmlu": "mmlu",
    "mmlu-cf": "mmlu_cf",
    "mmlu-pro": "mmlu_pro",
    "ms_marco": "msmarco",
    "mtop_intent": "label_pressure",
    "musique": "multihop_retrieval",
    "musr": "musr",
    "nemotron-pii": "lw_pii",
    "nevir": "nevir",
    "newyorker_caption_contest": "newyorker",
    "nfcorpus": "nfcorpus",
    "notinject": "injection_combined",
    "oos-eval": "clinc150",
    "openai-moderation-api-evaluation": "openai_mod",
    "openbookqa": "obqa",
    "paws": "paws",
    "phishing-email-dataset": "phishing_small",
    "phishnchips": "phishnchips",
    "pop909-cl-dataset": "pop909",
    "popqa": "popqa4",
    "ppe-human-preference-v1": "ppe",
    "privy": "lw_pii",
    "prmbench_preview": "prmbench",
    "processbench": "processbench",
    "prompt-injections": "prompt_injections",
    "pubmedqa": "pubmedqa",
    "purplellama": "cyberseceval_cwe",
    "race": "race_h",
    "ragtruth": "ragtruth",
    "reward-bench": "rewardbench1",
    "reward-bench-2": "rewardbench2",
    "rm-bench": "rmbench_pair",
    "rotten_tomatoes": "rotten_tomatoes",
    "routerbench": "routerbench",
    "safe-guard-prompt-injection": "injection_combined",
    "sata-bench": "sata",
    "scifact": "scifact",
    "searchless_chess": "chessbench",
    "semif": "nimble",
    "sib200": "sib200",
    "simpleqa-verified": "simpleqa4",
    "sms_spam": "sms_spam",
    "snips": "lw_offtopic",
    "spml": "lw_injection",
    "squad": "squad_select",
    "sst2": "sst2",
    "sst5": "sst5",
    "stsb": "stsb",
    "summeval": "summeval",
    "supergpqa": "jevout",
    "synthetic-intent-classifier-dataset-v1": "tanaos",
    "sys1cal-v1": "sys1cal",
    "tab_fact": "tabfact",
    "task-2-semeval-2024": "nli4ct",
    "task-specific-datasets": "banking77",
    "thai_exam": "exams_misc",
    "tmmluplus": "tmmlu",
    "toolret-queries": "toolret",
    "toolret-tools": "toolret",
    "toxic-chat": "toxicchat",
    "toxigen-data": "toxigen",
    "trec": "trec_fine",
    "trec-qc": "trec_coarse",
    "trivia_qa": "triviaqa4",
    "truthful_qa": "truthfulqa",
    "tweet_eval": "tweeteval_emotion",
    "tweet_topic_single": "tweet_topic",
    "twitter-financial-news-topic": "fin_topic",
    "typed-decisions": "typed_decisions",
    "wanli": "wanli",
    "when2call": "when2call",
    "wildguardmix": "wildguard",
    "winogrande": "winogrande",
    "xnli": "xnli",
    "xquad": "xquad_rerank",
    "yelp_review_full": "yelp5",
    "zero-shot-stance": "vast",
}
PUBLIC_JEV_KEYS: dict[str, str] = {
    "acos": "acos",
    "aegis1_prompt": "aegis1_prompt",
    "aegis2_prompt": "aegis2_prompt",
    "aegis2_response": "aegis2_response",
    "afrixnli": "afrixnli",
    "ag_news": "ag_news",
    "agb_de": "agb_de",
    "agent_trace": "agent_trace",
    "aita": "aita",
    "ajgt": "ajgt",
    "alfworld": "alfworld",
    "alphanli": "alphanli",
    "amazon_counterfactual": "amazon_counterfactual",
    "amazon_polarity": "amazon_polarity",
    "amazonqa": "amazonqa",
    "anli": "anli",
    "api_bank": "api_bank",
    "app_reviews": "app_reviews",
    "arc": "arc",
    "arc_challenge": "arc_challenge",
    "arc_easy": "arc_easy",
    "arize_spam": "arize_spam",
    "arxiv_2026": "arxiv_2026",
    "banking77": "banking77",
    "bbh": "bbh",
    "belebele": "belebele",
    "bfcl": "bfcl",
    "bigbench": "bigbench",
    "boolq": "boolq",
    "boolq_negation": "boolq_negation",
    "bpomp": "bpomp",
    "bright": "bright",
    "btzsc_ag": "btzsc_ag",
    "btzsc_b77": "btzsc_b77",
    "btzsc_emotion": "btzsc_emotion",
    "casehold": "casehold",
    "cesnet": "cesnet",
    "ceval": "ceval",
    "cfcolor": "cfcolor",
    "cfpb": "cfpb",
    "chaosnli": "chaosnli",
    "chessbench": "chessbench",
    "civil_comments": "civil_comments",
    "cjbench": "cjbench",
    "cladder": "cladder",
    "clinc150": "clinc150",
    "clinc_certify": "clinc_certify",
    "cohen_adhd": "cohen_adhd",
    "cohen_screening": "cohen_screening",
    "commonlit": "commonlit",
    "conll_typing": "conll_typing",
    "contractnli": "contractnli",
    "craigslist": "craigslist",
    "crt": "crt",
    "cruxeval": "cruxeval",
    "csqa": "csqa",
    "css_badbaseline": "css_badbaseline",
    "css_big": "css_big",
    "css_jevahead": "css_jevahead",
    "css_lowjev": "css_lowjev",
    "css_narrow": "css_narrow",
    "css_talklife": "css_talklife",
    "cuad": "cuad",
    "cyberseceval_cwe": "cyberseceval_cwe",
    "daily_dialog": "daily_dialog",
    "daily_oracle": "daily_oracle",
    "dair_emotion": "dair_emotion",
    "darija": "darija",
    "dblp_acm": "dblp_acm",
    "decidebench": "decidebench",
    "decisionbench": "decisionbench",
    "diagnosisarena": "diagnosisarena",
    "dialog_quality": "dialog_quality",
    "dimabsa_triplet": "dimabsa_triplet",
    "dimabsa_va": "dimabsa_va",
    "eclipse_severity": "eclipse_severity",
    "ecthr_a": "ecthr_a",
    "ecthr_b": "ecthr_b",
    "ellipse": "ellipse",
    "email_spam": "email_spam",
    "enron": "enron",
    "esci": "esci",
    "eurlex": "eurlex",
    "fake_jobs": "fake_jobs",
    "fever": "fever",
    "fin_topic": "fin_topic",
    "finentity": "finentity",
    "fiqa_rerank": "fiqa_rerank",
    "forecastbench": "forecastbench",
    "fpb": "fpb",
    "gaokao": "gaokao",
    "goemotions": "goemotions",
    "goemotions_single": "goemotions_single",
    "gpqa": "gpqa",
    "gsm8k_mc": "gsm8k_mc",
    "gulf_arabic": "gulf_arabic",
    "habermas": "habermas",
    "halueval_qa": "halueval_qa",
    "har": "har",
    "harmbench_prompt": "harmbench_prompt",
    "harmbench_resp": "harmbench_resp",
    "hatecheck": "hatecheck",
    "healthbench": "healthbench",
    "hellaswag": "hellaswag",
    "hellaswag_vercel": "hellaswag_vercel",
    "helpsteer2": "helpsteer2",
    "hle": "hle",
    "hover": "hover",
    "hs2_help_acc": "hs2_help_acc",
    "hs2_verb_acc": "hs2_verb_acc",
    "humicroedit": "humicroedit",
    "hwu64": "hwu64",
    "imdb": "imdb",
    "injecagent": "injecagent",
    "injection_combined": "injection_combined",
    "injection_deepset_all": "injection_deepset_all",
    "isarcasm": "isarcasm",
    "jevals_helpsteer": "jevals_helpsteer",
    "jevals_pubmedqa": "jevals_pubmedqa",
    "jevbench": "jevbench",
    "jevbench_calib": "jevbench_calib",
    "jevbench_composite": "jevbench_composite",
    "jevbench_public": "jevbench_public",
    "jevout": "jevout",
    "jmedqa": "jmedqa",
    "judgebench": "judgebench",
    "kev": "kev",
    "kiki_agent": "kiki_agent",
    "kobbq": "kobbq",
    "kobest": "kobest",
    "language_id": "language_id",
    "ledgar": "ledgar",
    "lexglue_unfair": "lexglue_unfair",
    "llm_aggrefact": "llm_aggrefact",
    "logiqa": "logiqa",
    "lw_commit": "lw_commit",
    "lw_community": "lw_community",
    "lw_complaint": "lw_complaint",
    "lw_injection": "lw_injection",
    "lw_moderation": "lw_moderation",
    "lw_offtopic": "lw_offtopic",
    "lw_pii": "lw_pii",
    "lw_rag": "lw_rag",
    "lw_routing20": "lw_routing20",
    "lw_search": "lw_search",
    "lw_toolrouting": "lw_toolrouting",
    "lw_webagent": "lw_webagent",
    "massive_en": "massive_en",
    "massive_scenario51": "massive_scenario51",
    "massive_sv": "massive_sv",
    "mcphunt": "mcphunt",
    "medhallu": "medhallu",
    "metamedqa": "metamedqa",
    "mhs": "mhs",
    "mind2web": "mind2web",
    "mmlu": "mmlu",
    "mmlu_cf": "mmlu_cf",
    "mmlu_pro": "mmlu_pro",
    "mnli": "mnli",
    "msmarco": "msmarco",
    "multihop_retrieval": "multihop_retrieval",
    "musr": "musr",
    "nanorteb": "nanorteb",
    "nejm": "nejm",
    "nevir": "nevir",
    "newyorker": "newyorker",
    "nfcorpus": "nfcorpus",
    "nimble": "nimble",
    "nli4ct": "nli4ct",
    "nlupp": "nlupp",
    "nslkdd": "nslkdd",
    "obqa": "obqa",
    "openai_mod": "openai_mod",
    "p4g": "p4g",
    "paws": "paws",
    "phishing_small": "phishing_small",
    "phishnchips": "phishnchips",
    "pop909": "pop909",
    "popqa4": "popqa4",
    "ppe": "ppe",
    "prmbench": "prmbench",
    "processbench": "processbench",
    "prompt_injections": "prompt_injections",
    "pubmedqa": "pubmedqa",
    "race_h": "race_h",
    "radiology": "radiology",
    "ragtruth": "ragtruth",
    "recsys": "recsys",
    "redhat": "redhat",
    "rewardbench1": "rewardbench1",
    "rewardbench2": "rewardbench2",
    "rexerr": "rexerr",
    "ricechem": "ricechem",
    "rmbench_pair": "rmbench_pair",
    "rmbench_point": "rmbench_point",
    "rotten_tomatoes": "rotten_tomatoes",
    "routerbench": "routerbench",
    "rubricbench": "rubricbench",
    "sata": "sata",
    "scifact": "scifact",
    "scifact_claims": "scifact_claims",
    "scotus": "scotus",
    "sgd": "sgd",
    "sib200": "sib200",
    "simpleqa4": "simpleqa4",
    "sms_spam": "sms_spam",
    "squad_select": "squad_select",
    "sst2": "sst2",
    "sst2_injection": "sst2_injection",
    "sst5": "sst5",
    "strategyqa_closed": "strategyqa_closed",
    "strategyqa_grounded": "strategyqa_grounded",
    "stsb": "stsb",
    "stsb_6level": "stsb_6level",
    "summeval": "summeval",
    "sys1cal": "sys1cal",
    "tabfact": "tabfact",
    "tanaos": "tanaos",
    "tmmlu": "tmmlu",
    "toolret": "toolret",
    "topic_reworded": "topic_reworded",
    "toxicchat": "toxicchat",
    "toxicchat_jailbreak": "toxicchat_jailbreak",
    "toxigen": "toxigen",
    "trec_coarse": "trec_coarse",
    "trec_fine": "trec_fine",
    "triviaqa4": "triviaqa4",
    "truthfulqa": "truthfulqa",
    "tweet_topic": "tweet_topic",
    "tweeteval_emotion": "tweeteval_emotion",
    "typed_decisions": "typed_decisions",
    "typesafe_workflow": "typesafe_workflow",
    "unfair_tos": "unfair_tos",
    "upworthy": "upworthy",
    "vast": "vast",
    "wainject": "wainject",
    "wanli": "wanli",
    "when2call": "when2call",
    "whowhen": "whowhen",
    "wildguard": "wildguard",
    "winogrande": "winogrande",
    "xdviolence": "xdviolence",
    "xnli": "xnli",
    "xquad_rerank": "xquad_rerank",
    "xstest": "xstest",
    "yelp5": "yelp5",
}
S_TRAIN_SPLIT_KEYS: frozenset[str] = frozenset({
    "acos",
    "aegis1_prompt",
    "aegis2_prompt",
    "aegis2_response",
    "ag_news",
    "agb_de",
    "alphanli",
    "amazon_polarity",
    "anli",
    "api_bank",
    "arc",
    "arc_challenge",
    "arc_easy",
    "arxiv_2026",
    "banking77",
    "bigbench",
    "boolq",
    "btzsc_ag",
    "btzsc_b77",
    "btzsc_emotion",
    "casehold",
    "cesnet",
    "ceval",
    "cfcolor",
    "cfpb",
    "chaosnli",
    "chessbench",
    "civil_comments",
    "clinc150",
    "clinc_certify",
    "contractnli",
    "csqa",
    "daily_dialog",
    "dair_emotion",
    "dimabsa_triplet",
    "dimabsa_va",
    "ecthr_a",
    "ecthr_b",
    "enron",
    "esci",
    "eurlex",
    "fever",
    "fin_topic",
    "forecastbench",
    "fpb",
    "general_decisions",
    "goemotions",
    "goemotions_single",
    "gsm8k_mc",
    "habermas",
    "hellaswag",
    "hellaswag_vercel",
    "helpsteer2",
    "hfblog_macro",
    "hover",
    "hs2_help_acc",
    "hs2_verb_acc",
    "humicroedit",
    "hwu64",
    "imdb",
    "isarcasm",
    "jevals_helpsteer",
    "jevbench_hf_macro",
    "language_id",
    "ledgar",
    "lexglue_mean",
    "lexglue_unfair",
    "massive_en",
    "massive_scenario51",
    "medhallu",
    "mhs",
    "mmlu",
    "mnli",
    "nevir",
    "newyorker",
    "nfcorpus",
    "nli4ct",
    "nlupp",
    "nslkdd",
    "obqa",
    "paws",
    "pop909",
    "prompt_injections",
    "ragtruth",
    "rerank_mean",
    "rotten_tomatoes",
    "routerbench",
    "scifact",
    "scotus",
    "sgd",
    "sib200",
    "sst2",
    "sst5",
    "strategyqa_closed",
    "strategyqa_grounded",
    "stsb",
    "stsb_6level",
    "toolret",
    "toxicchat",
    "toxicchat_jailbreak",
    "toxigen",
    "trec_fine",
    "tweet_topic",
    "tweeteval_emotion",
    "typed_decisions",
    "unfair_tos",
    "upworthy",
    "vast",
    "when2call",
    "wildguard",
    "winogrande",
    "yelp5",
})
S_ALLOWED_IDS: dict[str, str] = {
    "ai-systems/task-2-semeval-2024": "nli4ct",
    "alibabaresearch/damo-convai": "api_bank",
    "allenai/ai2_arc": "arc",
    "allenai/art": "alphanli",
    "allenai/openbookqa": "obqa",
    "allenai/scifact": "scifact",
    "allenai/wildguardmix": "wildguard",
    "allenai/winogrande": "winogrande",
    "amazon-science/esci-data": "esci",
    "amazonscience/massive": "massive_en",
    "andyweasley2004/pop909-cl-dataset": "pop909",
    "atrost/financial_phrasebank": "fpb",
    "beir/nfcorpus": "nfcorpus",
    "beir/scifact": "scifact",
    "btzsc/btzsc": "btzsc_ag",
    "cais/mmlu": "mmlu",
    "cardiffnlp/tweet_eval": "tweeteval_emotion",
    "cardiffnlp/tweet_topic_single": "tweet_topic",
    "ceval/ceval-exam": "ceval",
    "clinc/clinc_oos": "clinc150",
    "clinc/oos-eval": "clinc150",
    "coastalcph/lex_glue": "unfair_tos",
    "cogcomp/trec": "trec_fine",
    "copenlu/fever_gold_evidence": "fever",
    "cornell-movie-review-data/rotten_tomatoes": "rotten_tomatoes",
    "d4br4/agb-de": "agb_de",
    "dair-ai/emotion": "dair_emotion",
    "davlan/sib200": "sib200",
    "deeppavlov/hwu64": "hwu64",
    "deepset/prompt-injections": "prompt_injections",
    "determined-ai/consumer_complaints_medium": "cfpb",
    "emilyallaway/zero-shot-stance": "vast",
    "facebook/anli": "anli",
    "facebook/xnli": "afrixnli",
    "fancyzhx/ag_news": "ag_news",
    "fancyzhx/amazon_polarity": "amazon_polarity",
    "forecastingresearch/forecastbench-datasets": "forecastbench",
    "google-deepmind/habermas_machine": "habermas",
    "google-deepmind/searchless_chess": "chessbench",
    "google-research-datasets/dstc8-schema-guided-dialogue": "sgd",
    "google-research-datasets/go_emotions": "goemotions",
    "google-research-datasets/paws": "paws",
    "google/boolq": "boolq",
    "google/civil_comments": "civil_comments",
    "hover-nlp/hover": "hover",
    "iabufarha/isarcasmeval": "isarcasm",
    "jmhessel/newyorker_caption_contest": "newyorker",
    "kiddothe2b/contract-nli": "contractnli",
    "legacy-datasets/banking77": "banking77",
    "li2017dailydialog/daily_dialog": "daily_dialog",
    "liminghao1630/api-bank": "api_bank",
    "lmsys/toxic-chat": "toxicchat",
    "localllama/typed-decisions": "typed_decisions",
    "mangopy/toolret-queries": "toolret",
    "mangopy/toolret-tools": "toolret",
    "mangopy/toolret-training-20w": "toolret",
    "masakhane/afrixnli": "afrixnli",
    "mteb/banking77": "banking77",
    "neudm/acos": "acos",
    "nustm/acos": "acos",
    "nvidia/aegis-ai-content-safety-dataset-1.0": "aegis1_prompt",
    "nvidia/aegis-ai-content-safety-dataset-2.0": "aegis2_prompt",
    "nvidia/helpsteer2": "helpsteer2",
    "nvidia/when2call": "when2call",
    "nyu-mll/glue": "mnli",
    "openai/gsm8k": "gsm8k_mc",
    "openrl/daily_dialog": "daily_dialog",
    "orionweller/nevir": "nevir",
    "papluca/language-identification": "language_id",
    "particlemedia/ragtruth": "ragtruth",
    "polyai-ldn/task-specific-datasets": "banking77",
    "polyai/banking77": "banking77",
    "praveenrajus/jev-bench": "arc",
    "qiaojin/pubmedqa": "pubmedqa",
    "reasonir/reasonir-data": "bright",
    "rowan/hellaswag": "hellaswag",
    "sentence-transformers/stsb": "stsb",
    "setfit/enron_spam": "enron",
    "setfit/sst5": "sst5",
    "stanfordnlp/contract-nli": "contractnli",
    "stanfordnlp/imdb": "imdb",
    "stanfordnlp/sst2": "sst2",
    "tasksource/bigbench": "bigbench",
    "tasksource/esci": "esci",
    "tasksource/mmlu": "mmlu",
    "tasksource/nli4ct": "nli4ct",
    "tau/commonsense_qa": "csqa",
    "toxigen/toxigen-data": "toxigen",
    "utaustin-aihealth/medhallu": "medhallu",
    "wandb/ragtruth-processed": "ragtruth",
    "withmartian/routerbench": "routerbench",
    "yelp/yelp_review_full": "yelp5",
    "zeroshot/twitter-financial-news-topic": "fin_topic",
}
S_MIXED_ELIGIBILITY_IDS: dict[str, tuple[str, ...]] = {
    "amazonscience/massive": ('massive_en', 'massive_scenario51', 'massive_sv'),
    "ceval/ceval-exam": ('ceval', 'memorization_probe'),
    "clinc/clinc_oos": ('clinc150', 'clinc_certify', 'lw_offtopic'),
    "cogcomp/trec": ('trec_coarse', 'trec_fine'),
    "dair-ai/emotion": ('css_lowjev', 'dair_emotion', 'hfblog_macro'),
    "deepset/prompt-injections": ('injection_combined', 'injection_deepset_all', 'lw_injection', 'prompt_injections'),
    "fancyzhx/ag_news": ('ag_news', 'hfblog_macro', 'topic_reworded'),
    "google/boolq": ('boolq', 'boolq_negation', 'general_decisions'),
    "legacy-datasets/banking77": ('banking77', 'lw_routing20'),
    "nvidia/aegis-ai-content-safety-dataset-2.0": ('aegis2_prompt', 'aegis2_response', 'lw_moderation'),
    "nyu-mll/glue": ('mnli', 'sst2', 'sst2_injection', 'stsb'),
    "praveenrajus/jev-bench": ('arc', 'banking77', 'boolq', 'chaosnli', 'civil_comments', 'clinc150', 'fever', 'goemotions_single', 'hs2_help_acc', 'hs2_verb_acc', 'jevbench_hf_macro', 'ledgar', 'lw_community', 'massive_en', 'mhs', 'mmlu', 'mnli', 'paws', 'sms_spam', 'sst5', 'strategyqa_closed', 'strategyqa_grounded', 'stsb_6level', 'yelp5'),
    "tasksource/mmlu": ('memorization_probe', 'mmlu'),
}
# <<< GENERATED <<<

# Benchmark keys of targets.json that are not dataset names (roles, composites, synthetic DI rows): they are
# not turned into token patterns.
PUBLIC_JEV_KEY_STOP = frozenset({
    "diag", "di_index", "di_agentic", "memorization_probe", "general_decisions", "hfblog_macro", "exams_misc",
    "halueval_misc", "label_pressure", "email_spam_di", "support_tickets", "phishing_gradient", "home_appliance",
    "speech_rescoring", "video_quality", "tabular", "rerank_mean", "lexglue_mean", "jevbench_hf_macro",
})


# =========================================================================================================
# Generation from targets.json (pure functions; the frozen copies above are their output)

_ID_RE = re.compile(r"(?<![\w.\-/])([A-Za-z0-9][\w.-]*)/([A-Za-z0-9][\w.-]*)(?![\w.\-/])")  # exactly one slash
_FILE_EXT = (".py", ".json", ".jsonl", ".csv", ".tsv", ".pkl", ".txt", ".parquet", ".zip", ".gz")
_REPO_STOP = frozenset({"val", "test", "train", "dev", "suite", "build", "data", "default"})
# Free-text "a/b" fragments of targets.json that are not dataset ids (prose like "Enron/Ling-Spam", "C2D/D2C").
NOT_HF_IDS = frozenset({
    "c2d/d2c", "german/multilingual", "helpsteer/helpsteer3", "mnli/fever-nli", "mednli/radnli-style",
    "r-judge/atbench-style", "enron/ling-spam", "research/benchmarks", "beir/mteb", "magellan/deepmatcher",
    "angular/angular", "vitejs/vite", "api-bank/bfcl", "test/val",
})


def extract_hf_ids(text: str | None) -> list[str]:
    """`owner/name` tokens in a free-text hf_id / train_sources string, normalised. Skips file paths, generated
    items, deeper paths ("a/b/c"), version-like fragments and NOT_HF_IDS; keeps the order of appearance."""
    if not text or text.lower().startswith("generated"):
        return []
    out: list[str] = []
    for owner, repo in _ID_RE.findall(text):
        if repo.lower().endswith(_FILE_EXT) or "." in owner or repo.lower() in _REPO_STOP:
            continue
        if len(repo) <= 2 or not re.search(r"[a-z]", repo, re.I) or not re.search(r"[a-z]", owner, re.I):
            continue
        norm = normalize_source(f"{owner}/{repo}")
        if norm in NOT_HF_IDS or norm in out:
            continue
        out.append(norm)
    return out


def _s_eligible(track_s: str | None) -> bool:
    """The benchmark's own train split may be used in S (targets.json track.S "train-split...")."""
    return bool(track_s) and track_s.startswith("train-split")


def _s_related(track_s: str | None) -> bool:
    """No usable train split, but the named related public data may be used in S (PLAN §2.1)."""
    return bool(track_s) and track_s.startswith("related-data only")


def generate_public_jev(targets: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the public_jev tables from a parsed bench/public/targets.json (schema benchmax-targets/1)."""
    rows = targets["targets"]
    keys: dict[str, str] = {}
    s_keys: set[str] = set()
    s_ids: dict[str, str] = {}
    cand: dict[str, list[tuple[tuple[int, int, int], str]]] = {}  # id -> [(priority, key)]
    id_elig: dict[str, set[bool]] = {}
    role_rank = {"headline": 0, "s_bar": 1, "secondary": 1}
    for i, r in enumerate(rows):
        k = r["benchmark_key"]
        keys.setdefault(k, k)
        elig = _s_eligible((r.get("track") or {}).get("S"))
        if elig:
            s_keys.add(k)
        # an id shared by several benchmarks is attributed to its counted headline row, else an S-eligible
        # row, else the first row that names it (S looks the key up in s_train_split_keys)
        prio = (role_rank.get(r.get("role"), 2) if r.get("counted") or r.get("role") != "headline" else 1,
                0 if elig else 1, i)
        for hid in extract_hf_ids(r.get("hf_id")):
            cand.setdefault(hid, []).append((prio, k))
            id_elig.setdefault(hid, set()).add(elig)
            if elig:
                s_ids.setdefault(hid, k)
        if elig or _s_related((r.get("track") or {}).get("S")):
            for ts in r.get("train_sources") or ():
                # "EXCLUDE a, b (reason)" / "NEVER config 'unsplit'" clauses name deny-list items, not sources
                head = re.split(r"\b(?:EXCLUDE|NEVER)\b", ts)[0]
                for hid in extract_hf_ids(head):
                    s_ids.setdefault(hid, k)
    ids = {hid: min(c)[1] for hid, c in cand.items()}
    repos: dict[str, list[tuple[tuple[int, int, int], str]]] = {}
    for hid, c in cand.items():
        repos.setdefault(hid.split("/", 1)[1], []).extend(c)
    repos_best = {repo: min(c)[1] for repo, c in repos.items()}
    mixed = {hid: tuple(sorted({k for _, k in cand[hid]})) for hid in sorted(cand) if len(id_elig[hid]) > 1}
    return {
        "targets_sha256": targets.get("source", {}).get("sha256", ""),
        "ids": dict(sorted(ids.items())),
        "repos": dict(sorted(repos_best.items())),
        "keys": dict(sorted((k, v) for k, v in keys.items() if k not in PUBLIC_JEV_KEY_STOP)),
        "s_train_split_keys": sorted(s_keys),
        "s_allowed_ids": dict(sorted(s_ids.items())),
        "s_mixed_eligibility_ids": mixed,
    }


def public_jev_tables() -> dict[str, Any]:
    """The frozen tables in the same shape as `generate_public_jev` (for the staleness test and the manifest)."""
    return {
        "targets_sha256": PUBLIC_JEV_TARGETS_SHA256,
        "ids": dict(PUBLIC_JEV_IDS),
        "repos": dict(PUBLIC_JEV_REPOS),
        "keys": dict(PUBLIC_JEV_KEYS),
        "s_train_split_keys": sorted(S_TRAIN_SPLIT_KEYS),
        "s_allowed_ids": dict(S_ALLOWED_IDS),
        "s_mixed_eligibility_ids": {k: tuple(v) for k, v in S_MIXED_ELIGIBILITY_IDS.items()},
    }


# =========================================================================================================
# Matching

def normalize_source(name: str) -> str:
    """NFKC, lowercase, trim, collapse whitespace. Keeps '/', '-', '_' and ':' (they carry structure)."""
    s = unicodedata.normalize("NFKC", str(name)).strip().lower()
    return re.sub(r"\s+", " ", s)


def _compile(rules: Mapping[str, str], prefix: str) -> list[tuple[str, re.Pattern[str]]]:
    return [(f"{prefix}{k}" if prefix else k, re.compile(p)) for k, p in rules.items()]


# A hand rule's reason carries the targets.json benchmark key when it has one (what the S track and the
# decontamination report group by), else the rule's own name.
_PUBLIC_JEV_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (f"public_jev:{t or k}", re.compile(p)) for k, (p, t) in PUBLIC_JEV_RULES.items()
]

_PRIMARY_COMPILED: list[tuple[str, re.Pattern[str]]] = (
    _compile(_TEST_RULES, "test:")
    + _compile(_DEV_RULES, "dev:")
    + _compile(_SIBLING_RULES, "sibling:")
    + _compile(_JEV_LABELLED_RULES, "")
)
_RESERVED_COMPILED = _compile(_RESERVED_RULES, "reserved:")
_COMPILED: list[tuple[str, re.Pattern[str]]] = _PRIMARY_COMPILED + _PUBLIC_JEV_COMPILED + _RESERVED_COMPILED
_DEV_COMPILED = _compile(_DEV_RULES, "dev:")
_S_DENY_COMPILED = _compile(S_DENY_RULES, "")
_KEY_TOKEN_RE: re.Pattern[str] | None = None


def _key_token_re() -> re.Pattern[str] | None:
    global _KEY_TOKEN_RE
    if _KEY_TOKEN_RE is None and PUBLIC_JEV_KEYS:
        alts = "|".join(re.escape(k) for k in sorted(PUBLIC_JEV_KEYS, key=len, reverse=True))
        _KEY_TOKEN_RE = re.compile(rf"(?<![a-z0-9])({alts})(?![a-z0-9])")
    return _KEY_TOKEN_RE


def _segments(alt: str) -> list[str]:
    return [s for s in re.split(r"[/:]", alt) if s]


# Our own typed-decision sources whose names contain a public key as a token ("procedural_typed_decisions"):
# never a table hit (the LocalLLaMA suite itself is caught by the test rule / its exact id).
_TABLE_FALSE_FRIENDS = re.compile(r"(procedural|synthetic|tasksource[-_ ]jev)[-_ ]typed[-_ ]decisions")


def _public_jev_table_reason(norm: str, alt: str) -> str | None:
    """Generated-table matches: exact hf id (with optional config suffix), repo-name segment, key token."""
    if _TABLE_FALSE_FRIENDS.search(norm):
        return None
    if alt in PUBLIC_JEV_IDS:
        return f"public_jev:{PUBLIC_JEV_IDS[alt]}"
    segs = _segments(alt)
    for i in range(2, len(segs) + 1):
        pref = "/".join(segs[:i])
        if pref in PUBLIC_JEV_IDS:
            return f"public_jev:{PUBLIC_JEV_IDS[pref]}"
    for s in segs:
        if s in PUBLIC_JEV_REPOS:
            return f"public_jev:{PUBLIC_JEV_REPOS[s]}"
    rx = _key_token_re()
    if rx is not None:
        m = rx.search(norm)
        if m:
            return f"public_jev:{PUBLIC_JEV_KEYS[m.group(1)]}"
    return None


def _dev_family_reason(norm: str) -> str | None:
    for fam in DEV_TASKSOURCE_SOURCES:
        f = normalize_source(fam)
        if norm == f or norm.startswith(f + "/") or norm.endswith("/" + f) or ("/" + f + "/") in norm:
            return "dev:tasksource_heldout"
    return None


def _first(compiled: list[tuple[str, re.Pattern[str]]], norm: str) -> str | None:
    for reason, pat in compiled:
        if pat.search(norm):
            return reason
    return None


def _rule_reason(norm: str) -> str | None:
    """First matching rule in precedence order (test, dev, sibling, jev_labelled, public_jev, reserved)."""
    return _first(_COMPILED, norm)


def _z_reason(name: str) -> str | None:
    """Precedence: exact ids; test/dev/sibling/jev_labelled rules; public_jev hand rules; public_jev tables
    (hf ids, repo segments, key tokens); reserved; held-out tasksource dev families."""
    norm = normalize_source(name)
    if norm in EXCLUDED_DATASET_IDS:
        return EXCLUDED_DATASET_IDS[norm]
    # "owner/ds:config" and "owner/ds/config" forms both appear in the wild.
    alt = norm.replace(":", "/")
    if alt in EXCLUDED_DATASET_IDS:
        return EXCLUDED_DATASET_IDS[alt]
    return (_first(_PRIMARY_COMPILED, norm) or _first(_PUBLIC_JEV_COMPILED, norm)
            or _public_jev_table_reason(norm, alt) or _first(_RESERVED_COMPILED, norm) or _dev_family_reason(alt))


def _target_key_of(reason: str) -> str | None:
    """targets.json benchmark key behind an exclusion reason (None when the row has no public Jev benchmark)."""
    cat, _, key = reason.partition(":")
    if cat == "public_jev":
        return key
    if cat in ("test", "dev", "sibling"):
        return JEVBENCH_TO_TARGET.get(key)
    return None


def s_allow_reason(name: str | None) -> str | None:
    """Why `name` is on the S allow-list (a train-split benchmark or a train_sources id), else None.
    Independent of the exclusion verdict; `exclusion_reason(name, track="S")` combines the two."""
    if not name:
        return None
    norm = normalize_source(name)
    alt = norm.replace(":", "/")
    for d in (alt, *("/".join(_segments(alt)[:i]) for i in range(2, len(_segments(alt)) + 1))):
        if d in S_ALLOWED_IDS:
            return f"s_allow:id:{d}->{S_ALLOWED_IDS[d]}"
    for rule, (pat, k) in S_ALLOW_RULES.items():
        if re.search(pat, norm):
            return f"s_allow:rule:{rule}->{k}"
    r = _z_reason(name)
    if r:
        k = _target_key_of(r)
        if k and k in S_TRAIN_SPLIT_KEYS:
            return f"s_allow:key:{k}"
    return None


def _s_deny_reason(norm: str) -> str | None:
    for rule, pat in _S_DENY_COMPILED:
        if pat.search(norm):
            return f"s_deny:{rule}"
    return None


def exclusion_reason(name: str | None, track: str = "Z") -> str | None:
    """Why `name` (a dataset id, tasksource source/task name, or inner original_task) is excluded, else None.

    track="Z": every category. track="S": the S allow-list (targets.json train-split benchmarks and their
    train_sources ids) waives test/sibling/public_jev matches; jev_labelled, the clean dev sets, S_DENY_RULES
    and `reserved` are never waived. Pass config-qualified names too ("tweet_eval/sentiment"): the dev check
    runs on every name."""
    if not name:
        return None
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {track!r}")
    r = _z_reason(name)
    if r is None or track == "Z":
        return r
    norm = normalize_source(name)
    if r == "jev_labelled" or r.startswith("reserved:"):
        return r
    deny = _s_deny_reason(norm)
    if deny:
        return f"{deny} ({r})"
    # a clean dev set is never trained on, unless the very same dataset is itself a public train-split target
    for reason, pat in _DEV_COMPILED:
        if pat.search(norm) and (JEVBENCH_TO_TARGET.get(reason.split(":", 1)[1]) not in S_TRAIN_SPLIT_KEYS):
            return reason
    if r == "dev:tasksource_heldout":
        return r
    if s_allow_reason(name):
        return None
    return r


def is_excluded(name: str | None, track: str = "Z") -> bool:
    """True if a training source with this name must be dropped (stage 1). Apply it to every name a row
    carries: HF dataset id, config, tasksource `source` / `task`, and aggregator `original_task`."""
    return exclusion_reason(name, track) is not None


def is_excluded_any(*names: str | None, track: str = "Z") -> bool:
    """True if any of the names is excluded on `track` ("Z" default, or "S" for the supervised allow-list)."""
    return any(is_excluded(n, track) for n in names)


def exclusion_reasons(*names: str | None, track: str = "Z") -> dict[str, str]:
    """{name: reason} for the names that are excluded on `track` (audit helper)."""
    out: dict[str, str] = {}
    for n in names:
        r = exclusion_reason(n, track)
        if r:
            out[str(n)] = r
    return out


def test_keys() -> list[str]:
    return [d.key for d in TEST]


def dev_keys() -> list[str]:
    return [d.key for d in DEV]


def retired_dev_keys() -> list[str]:
    return [d.key for d in RETIRED_DEV]


# =========================================================================================================
# Machine-readable manifest (bench/public/exclusions.json)

def _ds_entry(d: BenchDataset) -> dict[str, Any]:
    return {"key": d.key, "title": d.title, "hf_id": d.hf_id, "config": d.config, "splits": list(d.splits),
            "n_used": d.n_used, "notes": d.notes}


def exclusion_manifest() -> dict[str, Any]:
    """Everything a converter or auditor needs to reproduce stage 1 without importing this module."""
    return {
        "schema": "benchmax-exclusions/1",
        "registry_version": REGISTRY_VERSION,
        "dev_set_version": DEV_SET_VERSION,
        "generated_from": {"targets": "bench/public/targets.json", "sha256": PUBLIC_JEV_TARGETS_SHA256},
        "precedence": ["exact_ids", "rules.test", "rules.dev", "rules.sibling", "rules.jev_labelled",
                       "rules.public_jev (reason = benchmark_key or rule name)", "public_jev.hf_ids (exact, or as a prefix)",
                       "public_jev.repo_segments (path segment == repo name)", "public_jev.benchmark_keys (whole token)",
                       "rules.reserved", "jevbench.dev_tasksource_families"],
        "tracks": {
            "Z": "every category excluded, any split, mirror, sibling or derived set",
            "S": "test/sibling/public_jev waived for s_track.train_split_keys and s_track.allowed_ids; "
                 "jev_labelled, dev, reserved and s_track.deny_rules never waived; converters take TRAIN splits only",
        },
        "jevbench": {
            "test": [_ds_entry(d) for d in TEST],
            "dev": [_ds_entry(d) for d in DEV],
            "ref": [_ds_entry(d) for d in REF],
            "retired_dev": [_ds_entry(d) for d in RETIRED_DEV],
            "dev_tasksource_families": list(DEV_TASKSOURCE_SOURCES),
            "retired_dev_tasksource_families": list(RETIRED_DEV_TASKSOURCE_SOURCES),
        },
        "exact_ids": dict(sorted(EXCLUDED_DATASET_IDS.items())),
        "rules": {
            "test": dict(_TEST_RULES), "dev": dict(_DEV_RULES), "sibling": dict(_SIBLING_RULES),
            "jev_labelled": dict(_JEV_LABELLED_RULES), "reserved": dict(_RESERVED_RULES),
            "public_jev": {k: {"pattern": p, "benchmark_key": t} for k, (p, t) in PUBLIC_JEV_RULES.items()},
        },
        "public_jev": {
            "hf_ids": dict(PUBLIC_JEV_IDS),
            "repo_segments": dict(PUBLIC_JEV_REPOS),
            "benchmark_keys": sorted(PUBLIC_JEV_KEYS),
            "key_stoplist": sorted(PUBLIC_JEV_KEY_STOP),
        },
        "s_track": {
            "train_split_keys": sorted(S_TRAIN_SPLIT_KEYS),
            "allowed_ids": dict(S_ALLOWED_IDS),
            "mixed_eligibility_ids": {k: list(v) for k, v in S_MIXED_ELIGIBILITY_IDS.items()},
            "deny_rules": dict(S_DENY_RULES),
            "jevbench_to_target": dict(JEVBENCH_TO_TARGET),
        },
        "jev_labelled_fields": {k: list(v) for k, v in JEV_LABELLED_FIELDS.items()},
        "kept_despite_earlier_list": list(KEPT_DESPITE_EARLIER_LIST),
    }


def manifest_bytes() -> bytes:
    return (json.dumps(exclusion_manifest(), indent=1, ensure_ascii=False, sort_keys=False) + "\n").encode()


def manifest_sha256() -> str:
    return hashlib.sha256(manifest_bytes()).hexdigest()


def write_manifest(path: str) -> str:
    """Write bench/public/exclusions.json; returns its sha256 (log it: PLAN §2.2.3 pre-registration)."""
    data = manifest_bytes()
    with open(path, "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="jevbench registry: write the exclusion manifest or audit names")
    ap.add_argument("--manifest", metavar="PATH", help="write the exclusions manifest here and print its sha256")
    ap.add_argument("--track", default="Z", choices=TRACKS)
    ap.add_argument("names", nargs="*", help="source names to audit (prints the exclusion reason per name)")
    a = ap.parse_args()
    if a.manifest:
        print(f"{write_manifest(a.manifest)}  {a.manifest}")
    for n in a.names:
        print(f"{n}\t{exclusion_reason(n, a.track) or '-'}\t{s_allow_reason(n) or ''}")
    if not a.manifest and not a.names:
        ap.print_help(sys.stderr)
