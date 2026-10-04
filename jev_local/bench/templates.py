"""jevbench v1 request templates: dataset row -> {state, questions} + gold (pure Python, testable anywhere).

Label text provenance (PLAN §1.1; CONTRACT rule 4: no LLM-written text):
  deusser       Deußer, Sparrenberg & Sifa, `jev-benchmarking` (MIT) at DEUSSER_COMMIT, copied verbatim from
                jev_benchmarking/tasks/*.py (raw files kept in bench/sources/deusser/). Two adaptations, both
                forced by OpenRouter's string-only schema: object instructions (OpenAI moderation, UNFAIR-ToS)
                are rendered into one string, and a noul never carries only one of true/false criteria.
  btzsc         BTZSC hypotheses (Laurer's label hypotheses as released in btzsc/btzsc at BTZSC_REVISION),
                read at build time from the rows with labels == 1; passed in through `Ctx.btzsc`.
  programmatic  a fixed sentence frame around the humanised label name (`humanize`), or the label name
                itself; written once here, never generated.
  native        typed-decisions ships Jev-format requests; used byte for byte.

Gold encoding, one dict per question id:
  choice {"type": "choice", "label": key}                       (+ "dist": {key: p} for soft gold)
  noul   {"type": "noul", "label": bool}                        (+ "p": float for soft gold)
  score  {"type": "score", "label": level_index, "value": float} (+ "dist": [p0..] for soft gold)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

TEMPLATE_VERSION = "jevbench-v1-t1"
DEUSSER_REPO = "github.com/AppliedMachineLearning-Lab/jev-benchmarking"
DEUSSER_COMMIT = "6bbdeb33474849b6de2f0cccc9f5e19756abd67e"
BTZSC_ID = "btzsc/btzsc"
BTZSC_REVISION = "fef2a2ac62b69c58670047dddf045c53d7c3cb5e"


@dataclass
class Item:
    item_id: str  # unique within the dataset; also the bootstrap cluster
    state: Any
    questions: dict[str, dict]
    gold: dict[str, dict]
    stratum: str  # target stratum for sampling and the stratified bootstrap
    template: str  # template id, e.g. "deusser:AGNews" or "programmatic:fin_topic"
    meta: dict = field(default_factory=dict)
    extra: dict[str, tuple[dict, dict]] = field(default_factory=dict)  # variant -> (questions, gold)


@dataclass
class Ctx:
    """Dataset-level facts a mapper needs (label names from the HF features, BTZSC hypotheses)."""

    names: list[str] = field(default_factory=list)  # class names in dataset label order
    btzsc: dict[str, str] = field(default_factory=dict)  # normalised label name -> hypothesis
    btzsc_missing: list[str] = field(default_factory=list)  # filled by mappers: labels with no hypothesis


# ---------------------------------------------------------------------------------------------- helpers


def choice(instructions: str, criteria: dict[str, str | None]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    """OpenRouter rejects a noul with only one of true/false (HTTP 400 in runs/compare): both or neither."""
    q: dict = {"type": "noul", "instructions": instructions}
    if true or false:
        if not (true and false):
            raise ValueError("noul criteria need both true and false text")
        q["criteria"] = {"true": true, "false": false}
    return q


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def g_choice(label: str) -> dict:
    return {"type": "choice", "label": label}


def g_noul(label: bool) -> dict:
    return {"type": "noul", "label": bool(label)}


def g_score(level: int, value: float | None = None) -> dict:
    return {"type": "score", "label": int(level), "value": float(level if value is None else value)}


def humanize(name: str) -> str:
    """'card_arrival' -> 'card arrival'; 'AddToPlaylist' -> 'add to playlist'; 'arts_&_culture' -> 'arts & culture'."""
    s = name
    if not re.search(r"[_\s-]", s):  # CamelCase -> words
        s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"\s+", " ", s.replace("_", " ").replace("-", " ")).strip()
    return s if s.isupper() else s.lower()


def norm_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def described(ctx: Ctx, names: list[str], frame: str) -> dict[str, str]:
    """BTZSC hypothesis where one exists, else the programmatic frame around the humanised name."""
    out = {}
    for n in names:
        h = ctx.btzsc.get(norm_label(n))
        if h is None:
            ctx.btzsc_missing.append(n)
            h = frame.format(humanize(n))
        out[n] = h
    return out


def _cap_question(q: str) -> str:
    q = q.strip()
    return q[0].upper() + q[1:] + ("" if q.endswith("?") else "?") if q else q


# ---------------------------------------------------------------------- Deußer text (verbatim, MIT)

SENTIMENT_2 = {
    "negative": "The text expresses a negative opinion.",
    "positive": "The text expresses a positive opinion.",
}
AG_CRITERIA = {
    "World": "International news, politics, conflicts and world affairs.",
    "Sports": "Sports events, teams and athletes.",
    "Business": "Companies, markets, the economy and finance.",
    "Sci/Tech": "Science, technology, computing, the internet and space.",
}
FPB_CRITERIA = {
    "negative": "The news is likely to have a negative effect on the company's stock price.",
    "neutral": "The news is unlikely to affect the company's stock price.",
    "positive": "The news is likely to have a positive effect on the company's stock price.",
}
NLI_LABELS = ("entailment", "neutral", "contradiction")
NLI_CRITERIA = {
    "entailment": "`hypothesis` is definitely true given `premise`.",
    "neutral": "`hypothesis` may or may not be true; `premise` does not settle it.",
    "contradiction": "`hypothesis` is definitely false given `premise`.",
}
PAWS_TRUE = "They are paraphrases: the same meaning, possibly with different wording or word order."
PAWS_FALSE = "Their meanings differ, even if they share most of their words."
GO_EMOTIONS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
)  # fmt: skip
MODERATION = {
    "S": ("sexual", "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness)."),
    "H": ("hate", "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste."),
    "V": ("violence", "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others."),
    "HR": ("harassment", "Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur."),
    "SH": ("self-harm", "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders."),
    "S3": ("sexual/minors", "Sexual content that includes an individual who is under 18 years old."),
    "H2": ("hate/threatening", "Hateful content that also includes violence or serious harm towards the targeted group."),
    "V2": ("violence/graphic", "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail."),
}  # fmt: skip
UNFAIR = {
    "Limitation of liability": "The provider limits or excludes its liability for damages or losses.",
    "Unilateral termination": "The provider can suspend or terminate the service or the contract at its discretion.",
    "Unilateral change": "The provider can change the terms or the service unilaterally.",
    "Content removal": "The provider can remove or modify the user's content at its discretion.",
    "Contract by using": "The user is bound by the terms simply by using the service.",
    "Choice of law": "The clause specifies which country's or state's law governs the contract.",
    "Jurisdiction": "The clause specifies which courts have jurisdiction over disputes.",
    "Arbitration": "Disputes must be resolved by arbitration rather than in court.",
}
STS_LEVELS = [
    "The two sentences are completely dissimilar.",
    "The two sentences are not equivalent, but are on the same topic.",
    "The two sentences are not equivalent, but share some details.",
    "The two sentences are roughly equivalent, but some important information differs or is missing.",
    "The two sentences are mostly equivalent, but some unimportant details differ.",
    "The two sentences are completely equivalent, as they mean the same thing.",
]
SST5_LEVELS = ["Very negative.", "Negative.", "Neutral.", "Positive.", "Very positive."]
HELPSTEER = {
    "helpfulness": ("Overall helpfulness of `response` to `prompt`.", ["Not helpful at all.", "Slightly helpful.", "Moderately helpful.", "Very helpful.", "Extremely helpful."]),
    "correctness": ("Inclusion of all pertinent facts in `response`, without errors.", ["Mostly incorrect or missing key facts.", "Several errors or omissions.", "Some errors or omissions.", "Minor errors or omissions.", "Fully correct and complete."]),
    "coherence": ("Consistency and clarity of expression in `response`.", ["Incoherent.", "Often unclear or inconsistent.", "Somewhat clear.", "Mostly clear and consistent.", "Perfectly clear and consistent."]),
    "complexity": ("Intellectual depth required to write `response`.", ["Basic: anyone who speaks the language could write it.", "Simple: needs only everyday knowledge.", "Intermediate: needs some education in the topic.", "Advanced: needs expertise in the field.", "Expert: needs deep domain expertise."]),
    "verbosity": ("Amount of detail in `response`, relative to what `prompt` asks for.", ["Very succinct.", "Succinct.", "Moderate length.", "Verbose.", "Very verbose."]),
}  # fmt: skip
TOXIGEN_LEVELS = [
    "Benign: not offensive to anyone.",
    "Slightly offensive or insensitive.",
    "Moderately offensive.",
    "Very offensive.",
    "Extremely offensive: hateful or dehumanizing.",
]
MC_INSTRUCTION = "Which option correctly answers `question`?"
CLINC_OOS = "Out of scope: the request matches none of the other intents."

# ---------------------------------------------------------------------- programmatic text (fixed frames)

FIN_TOPIC_NAMES = [  # zeroshot/twitter-financial-news-topic README (label map), revision acbc8af2
    "Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt", "Dividend",
    "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion", "Gold | Metals | Materials",
    "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets", "Politics", "Personnel Change",
    "Stock Commentary", "Stock Movement",
]  # fmt: skip
CLIMATE_FEVER = {  # label names and definitions of the Climate-FEVER dataset card
    "SUPPORTS": "The evidence supports the claim.",
    "REFUTES": "The evidence refutes the claim.",
    "NOT_ENOUGH_INFO": "The evidence neither supports nor refutes the claim.",
    "DISPUTED": "Some of the evidence supports the claim and some refutes it.",
}
STAR_LEVELS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
SCIFACT = {  # SciFact label names (Wadden et al. 2020; the data's SUPPORT / CONTRADICT plus the task's NOT_ENOUGH_INFO)
    "SUPPORT": "The cited abstracts contain evidence that supports the claim.",
    "CONTRADICT": "The cited abstracts contain evidence that contradicts the claim.",
    "NOT_ENOUGH_INFO": "The cited abstracts contain no evidence for or against the claim.",
}
ATIS_FRAME = "The user's request is about {}."


def _alpha(i: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]


def mc_criteria(options: list[str]) -> dict[str, str]:
    return {_alpha(i): o for i, o in enumerate(options)}


# ---------------------------------------------------------------------- mappers: test


def m_ag_news(row, i, ctx):
    lab = ctx.names[row["label"]]
    return Item(i, {"article": row["text"]}, {"answer": choice("What is the topic of the news `article`?", dict(AG_CRITERIA))},
                {"answer": g_choice(lab)}, lab, "deusser:AGNews")


def m_yahoo(row, i, ctx):
    names = ctx.names
    crit = described(ctx, names, "This question is about {}.")
    lab = names[row["topic"]]
    state = {"title": row["question_title"].strip(), "question": (row["question_content"] or "").strip(),
             "best_answer": (row["best_answer"] or "").strip()}
    state = {k: v for k, v in state.items() if v}  # many questions have no body; never send empty fields
    q = choice("Which topic category of the Yahoo Answers forum does `title` belong to?", crit)
    return Item(i, state, {"answer": q}, {"answer": g_choice(lab)}, lab, "btzsc:yahootopics")


def m_fin_topic(row, i, ctx):
    crit = {n: f"Financial news about: {n.replace(' | ', ', ')}." for n in FIN_TOPIC_NAMES}
    lab = FIN_TOPIC_NAMES[row["label"]]
    q = choice("What is the topic of the financial news `tweet`?", crit)
    return Item(i, {"tweet": row["text"]}, {"answer": q}, {"answer": g_choice(lab)}, lab, "programmatic:fin_topic")


def m_sst2(row, i, ctx):
    lab = ctx.names[row["label"]]  # negative / positive
    state = {"review": row["sentence"].strip()}
    q_noul = noul("Is the overall sentiment of `review` toward the movie positive?",
                  true=SENTIMENT_2["positive"], false=SENTIMENT_2["negative"])
    q_choice = choice("What is the overall sentiment of `review` toward the movie?", dict(SENTIMENT_2))
    return Item(i, state, {"answer": q_noul}, {"answer": g_noul(lab == "positive")}, lab,
                "deusser:SST2 (noul from SENTIMENT_2)", extra={"choice": ({"answer": q_choice}, {"answer": g_choice(lab)})})


def m_fin_phrasebank(row, i, ctx):
    lab = ctx.names[row["label"]]
    q = choice("From an investor's point of view, what is the sentiment of the financial news `sentence`?", dict(FPB_CRITERIA))
    return Item(i, {"sentence": row["sentence"]}, {"answer": q}, {"answer": g_choice(lab)}, lab, "deusser:FinancialPhraseBank")


def m_sst5(row, i, ctx):
    lv = int(row["label"])
    return Item(i, {"sentence": row["text"]}, {"answer": score("What is the sentiment of `sentence`?", SST5_LEVELS)},
                {"answer": g_score(lv)}, str(lv), "deusser:SST5")


def m_yelp5(row, i, ctx):
    lv = int(row["label"])
    q = score("How many stars did the author of `review` give the business?", STAR_LEVELS)
    return Item(i, {"review": row["text"]}, {"answer": q}, {"answer": g_score(lv)}, str(lv), "programmatic:yelp5")


def m_dair_emotion(row, i, ctx):
    names = ctx.names
    crit = described(ctx, names, "This example tweet expresses the emotion: {}")
    lab = names[row["label"]]
    q = choice("Which emotion does the author of `text` express most strongly?", crit)
    return Item(i, {"text": row["text"]}, {"answer": q}, {"answer": g_choice(lab)}, lab, "deusser:Emotion+btzsc:emotiondair")


def m_tweeteval_emotion(row, i, ctx):
    names = ctx.names
    crit = {n: f"The tweet expresses {n}." for n in names}
    lab = names[row["label"]]
    q = choice("Which emotion does the author of `tweet` express most strongly?", crit)
    return Item(i, {"tweet": row["text"]}, {"answer": q}, {"answer": g_choice(lab)}, lab, "programmatic:tweeteval_emotion")


def m_go_emotions(row, i, ctx):
    qs = {l: noul(f"Does the author of `comment` express {l}?") for l in GO_EMOTIONS if l != "neutral"}
    qs["neutral"] = noul("Is `comment` emotionally neutral, expressing no particular emotion?")
    labs = set(row["labels"])
    gold = {l: g_noul(k in labs) for k, l in enumerate(GO_EMOTIONS)}
    return Item(i, {"comment": row["text"]}, qs, gold, f"n{min(len(labs), 2)}", "deusser:GoEmotions", meta={"id": row.get("id")})


def m_banking77(row, i, ctx):
    names = ctx.names  # sorted(set(label_text)), as Deußer
    crit = described(ctx, names, "This banking customer example message is about {}.")
    q = choice("Which intent best describes the bank customer's `query`?", crit)
    return Item(i, {"query": row["text"]}, {"answer": q}, {"answer": g_choice(row["label_text"])}, row["label_text"],
                "deusser:Banking77+btzsc:banking77")


def m_clinc150(row, i, ctx):
    names = ctx.names
    crit = {n: f"The user's request is about {humanize(n)}." for n in names if n != "oos"}
    crit["oos"] = CLINC_OOS
    lab = names[row["intent"]]
    q = choice("Which intent does the user's `utterance` to a virtual assistant express?", crit)
    return Item(i, {"utterance": row["text"]}, {"answer": q}, {"answer": g_choice(lab)}, lab,
                "deusser:CLINC150+programmatic")


def m_massive(row, i, ctx):
    names = ctx.names
    crit = described(ctx, names, "The intent of the example utterance is {}.")
    q = choice("Which intent does the user's `utterance` to a voice assistant express?", crit)
    return Item(i, {"utterance": row["text"]}, {"answer": q}, {"answer": g_choice(row["label"])}, row["label"],
                "btzsc:massive")


def m_boolq(row, i, ctx):
    q = noul("According to `passage`, is the answer to `question` yes?")
    lab = bool(row["answer"])
    return Item(i, {"passage": row["passage"], "question": _cap_question(row["question"])}, {"answer": q},
                {"answer": g_noul(lab)}, str(lab), "deusser:BoolQ")


def _entail_noul() -> dict:
    return noul("Is `hypothesis` definitely true given `premise`?", true=NLI_CRITERIA["entailment"],
                false="`hypothesis` is not definitely true given `premise`: it may be false, or `premise` may not settle it.")


def m_rte(row, i, ctx):
    lab = ctx.names[row["label"]] == "entailment"
    return Item(i, {"premise": row["sentence1"], "hypothesis": row["sentence2"]}, {"answer": _entail_noul()},
                {"answer": g_noul(lab)}, str(lab), "deusser:NLI_CRITERIA (noul)")


def m_anli(row, i, ctx):
    lab = NLI_LABELS[row["label"]]
    q = choice("What is the relationship between `premise` and `hypothesis`?", dict(NLI_CRITERIA))
    return Item(i, {"premise": row["premise"], "hypothesis": row["hypothesis"]}, {"answer": q}, {"answer": g_choice(lab)},
                lab, "deusser:ANLI", meta={"uid": row["uid"]})


def _paws_noul() -> dict:
    return noul("Do `sentence_1` and `sentence_2` mean the same thing?", true=PAWS_TRUE, false=PAWS_FALSE)


def m_paws(row, i, ctx):
    lab = int(row["label"]) == 1
    return Item(i, {"sentence_1": row["sentence1"], "sentence_2": row["sentence2"]}, {"answer": _paws_noul()},
                {"answer": g_noul(lab)}, str(lab), "deusser:PAWS")


def m_stsb(row, i, ctx):
    v = float(row["score"]) * 5  # this copy stores the 0-5 score divided by 5
    q = score("How similar in meaning are `sentence_1` and `sentence_2`?", STS_LEVELS)
    return Item(i, {"sentence_1": row["sentence1"], "sentence_2": row["sentence2"]}, {"answer": q},
                {"answer": g_score(int(round(v)), v)}, str(int(round(v))), "deusser:STSB")


def m_llm_aggrefact(row, i, ctx):
    q = noul("Is every part of `claim` supported by `document`?",
             true="Everything the claim states is supported by the document.",
             false="Some part of the claim is unsupported by, or contradicts, the document.")
    lab = int(row["label"]) == 1
    return Item(i, {"document": row["doc"], "claim": row["claim"]}, {"answer": q}, {"answer": g_noul(lab)},
                f"{row['dataset']}|{lab}", "deusser:LLMAggreFact", meta={"source": row["dataset"]})


def m_climate_fever(row, i, ctx):
    lab = ctx.names[row["claim_label"]]
    ev = [e["evidence"].strip() for e in row["evidences"]]
    state = {"claim": row["claim"], "evidence": "\n".join(f"{k + 1}. {e}" for k, e in enumerate(ev))}
    q = choice("Taken together, what does `evidence` say about `claim`?", dict(CLIMATE_FEVER))
    return Item(i, state, {"answer": q}, {"answer": g_choice(lab)}, lab, "programmatic:climate_fever (card labels)")


def m_toxic_chat(row, i, ctx):
    qs = {
        "toxic": noul("Is the user's `prompt` to an AI chatbot toxic: harmful, offensive or sexual, or trying to get harmful output?"),
        "jailbreak": noul("Is the user's `prompt` a jailbreak attempt, trying to trick the AI chatbot into ignoring its safety rules?"),
    }
    gold = {"toxic": g_noul(row["toxicity"] == 1), "jailbreak": g_noul(row["jailbreaking"] == 1)}
    return Item(i, {"prompt": row["user_input"]}, qs, gold, str(row["toxicity"] == 1), "deusser:ToxicChat")


def _category_noul(subject: str, kind: str, name: str, definition: str) -> dict:
    # Deußer passes {"<kind>": {"name", "definition"}, "question"} as an object; OpenRouter needs a string.
    return noul(f'Does `{subject}` contain content that falls under the {kind} "{name}"? {name}: {definition}') \
        if kind == "category" else noul(f'Is `{subject}` a clause of type "{name}"? {name}: {definition}')


def m_openai_moderation(row, i, ctx):
    qs = {c: _category_noul("text", "category", n, d) for c, (n, d) in MODERATION.items()}
    gold = {c: g_noul(row[c] == 1) for c in MODERATION if row.get(c) is not None}
    return Item(i, {"text": row["prompt"]}, qs, gold, "pos" if any(g["label"] for g in gold.values()) else "neg",
                "deusser:OpenAIModeration (stringified)")


def m_typed_decisions(row, i, ctx):
    state = json.loads(row["state"])
    qs = json.loads(row["questions"])
    gold_raw = json.loads(row["gold"])
    gold: dict[str, dict] = {}
    for qid, g in gold_raw.items():
        t = g["type"]
        if t == "choice":
            gold[qid] = {"type": "choice", "label": g["label"], "dist": g["probabilities"]}
        elif t == "noul":
            gold[qid] = {"type": "noul", "label": g["label"] == "true", "p": float(g["noul"])}
        else:
            k = len(qs[qid]["criteria"])
            dist = [float(g["probabilities"].get(str(j), 0.0)) for j in range(k)]
            gold[qid] = {"type": "score", "label": int(g["label"]), "value": float(g["score"]), "dist": dist}
    return Item(i, state, qs, gold, row["workflow"], "native:typed-decisions", meta={"id": row["id"]})


def m_unfair_tos(row, i, ctx):
    qs, gold = {}, {}
    labs = set(row["labels"])
    for k, (label, definition) in enumerate(UNFAIR.items()):
        hid = re.sub(r"\W+", "_", label.lower())
        qs[hid] = _category_noul("clause", "clause", label, definition)
        gold[hid] = g_noul(k in labs)
    return Item(i, {"clause": row["text"].strip()}, qs, gold, "pos" if labs else "neg", "deusser:UnfairToS (stringified)")


def m_helpsteer2(row, i, ctx):
    qs = {a: score(instr, levels) for a, (instr, levels) in HELPSTEER.items()}
    gold = {a: g_score(int(row[a])) for a in HELPSTEER}
    return Item(i, {"prompt": row["prompt"], "response": row["response"]}, qs, gold, str(row["helpfulness"]),
                "deusser:HelpSteer2")


# ---------------------------------------------------------------------- mappers: dev


def m_newsgroups20(row, i, ctx):
    crit = {n: f"A post to the newsgroup {n}." for n in ctx.names}
    q = choice("Which newsgroup was `post` posted to?", crit)
    return Item(i, {"post": row["text"]}, {"answer": q}, {"answer": g_choice(row["label_text"])}, row["label_text"],
                "programmatic:20ng")


def m_tweet_topic(row, i, ctx):
    crit = {n: f"The tweet is about {humanize(n)}." for n in ctx.names}
    lab = ctx.names[row["label"]]
    return Item(i, {"tweet": row["text"]}, {"answer": choice("What is the topic of `tweet`?", crit)},
                {"answer": g_choice(lab)}, lab, "programmatic:tweet_topic")


def m_tweeteval_sentiment(row, i, ctx):
    crit = {"negative": SENTIMENT_2["negative"], "neutral": "The text expresses no clear positive or negative opinion.",
            "positive": SENTIMENT_2["positive"]}
    lab = ctx.names[row["label"]]
    return Item(i, {"tweet": row["text"]}, {"answer": choice("What is the sentiment of `tweet`?", crit)},
                {"answer": g_choice(lab)}, lab, "deusser:SENTIMENT_2+programmatic")


def m_app_reviews(row, i, ctx):
    lv = int(row["star"]) - 1
    q = score("How many stars did the author of `review` give the app?", STAR_LEVELS)
    return Item(i, {"review": row["review"]}, {"answer": q}, {"answer": g_score(lv)}, str(lv), "programmatic:app_reviews")


def m_hwu64(row, i, ctx):
    crit = {n: f"The user's request is about {humanize(n)}." for n in ctx.names}
    lab = ctx.names[row["label"]]
    q = choice("Which intent does the user's `utterance` to a home assistant express?", crit)
    return Item(i, {"utterance": row["utterance"]}, {"answer": q}, {"answer": g_choice(lab)}, lab, "programmatic:hwu64")


def m_snips(row, i, ctx):
    crit = {n: f"The user wants to {humanize(n)}." for n in ctx.names}
    q = choice("Which intent does the user's `utterance` to a voice assistant express?", crit)
    return Item(i, {"utterance": row["text"]}, {"answer": q}, {"answer": g_choice(row["category"])}, row["category"],
                "programmatic:snips")


def m_mrpc(row, i, ctx):
    lab = int(row["label"]) == 1
    return Item(i, {"sentence_1": row["sentence1"], "sentence_2": row["sentence2"]}, {"answer": _paws_noul()},
                {"answer": g_noul(lab)}, str(lab), "deusser:PAWS (on MRPC)")


def m_scitail(row, i, ctx):
    lab = row["gold_label"] == "entailment"
    return Item(i, {"premise": row["sentence1"], "hypothesis": row["sentence2"]}, {"answer": _entail_noul()},
                {"answer": g_noul(lab)}, str(lab), "deusser:NLI_CRITERIA (noul)")


def m_cb(row, i, ctx):
    lab = ctx.names[row["label"]]
    q = choice("What is the relationship between `premise` and `hypothesis`?", dict(NLI_CRITERIA))
    return Item(i, {"premise": row["premise"], "hypothesis": row["hypothesis"]}, {"answer": q}, {"answer": g_choice(lab)},
                lab, "deusser:NLI_CRITERIA")


def atis_desc(label: str) -> str:
    """ATIS_FRAME around the humanised intent; a joint intent ('flight+airfare') names every part."""
    parts = [humanize(p) for p in label.split("+")]
    if len(parts) == 1:
        return ATIS_FRAME.format(parts[0])
    if len(parts) == 2:
        return ATIS_FRAME.format(f"both {parts[0]} and {parts[1]}")
    return ATIS_FRAME.format(f"all of {', '.join(parts[:-1])} and {parts[-1]}")


def m_atis(row, i, ctx):
    # ctx.names = sorted(set(intent)) of the split (registry: "intents present in test"); joint intents such as
    # "flight+airfare" are their own options, so the option set equals the test label distribution.
    crit = {n: atis_desc(n) for n in ctx.names}
    q = choice("Which intent does the user's `utterance` to an air travel information system express?", crit)
    return Item(i, {"utterance": row["text"].strip()}, {"answer": q}, {"answer": g_choice(row["intent"])}, row["intent"],
                "programmatic:atis")


def scifact_claim_rows(claim_rows: list[dict], corpus: dict[str, dict]) -> list[dict]:
    """allenai/scifact `claims` is one row per (claim, evidence abstract, rationale); regroup to one row per claim
    (first-appearance order): {"id", "claim", "cited_doc_ids", "evidence": [[doc_id, label], ...], "docs": [...]}.
    `docs` are the cited abstracts (plus any evidence abstract missing from `cited_doc_ids`), in that order, where
    `corpus` (doc_id -> {"doc_id", "title", "abstract": [sentences]}) has them."""
    by: dict[Any, dict] = {}
    for r in claim_rows:
        g = by.setdefault(r["id"], {"id": r["id"], "claim": r["claim"], "cited_doc_ids": [str(x) for x in r["cited_doc_ids"]],
                                    "evidence": []})
        if r["evidence_label"]:
            doc = str(r["evidence_doc_id"])
            if doc not in g["cited_doc_ids"]:
                g["cited_doc_ids"].append(doc)
            if [doc, r["evidence_label"]] not in g["evidence"]:
                g["evidence"].append([doc, r["evidence_label"]])
    for g in by.values():
        g["docs"] = [corpus[d] for d in g["cited_doc_ids"] if d in corpus]
    return list(by.values())


def m_scifact(row, i, ctx):
    """One item per claim (rows from `scifact_claim_rows`): gold is the evidence abstracts' label (SciFact labels agree
    across a claim's abstracts; a claim with conflicting labels is dropped), NOT_ENOUGH_INFO when no abstract has evidence."""
    labs = {lab for _, lab in row["evidence"]}
    if len(labs) > 1 or not row["docs"]:
        return None
    lab = next(iter(labs), "NOT_ENOUGH_INFO")
    abstracts = "\n\n".join(f"{k + 1}. {d['title'].strip()}\n{' '.join(s.strip() for s in d['abstract'])}"
                            for k, d in enumerate(row["docs"]))
    q = choice("Taken together, what do the cited `abstracts` say about `claim`?", dict(SCIFACT))
    return Item(i, {"claim": row["claim"].strip(), "abstracts": abstracts}, {"answer": q}, {"answer": g_choice(lab)}, lab,
                "programmatic:scifact (per claim)",
                meta={"claim_id": row["id"], "doc_ids": [str(d["doc_id"]) for d in row["docs"]],
                      "evidence_doc_ids": [doc for doc, _ in row["evidence"]]})


def m_toxigen(row, i, ctx):
    qs = {"toxic": noul("Is `text` toxic, offensive or hateful toward a group of people?"),
          "toxicity": score("How offensive is `text` toward the group of people it mentions?", TOXIGEN_LEVELS)}
    tox = row["toxicity_ai"] + row["toxicity_human"] > 5.5  # lm-evaluation-harness binarisation (Deußer)
    lv = int(round(row["toxicity_human"])) - 1
    gold = {"toxic": g_noul(tox), "toxicity": g_score(min(4, max(0, lv)), row["toxicity_human"] - 1)}
    return Item(i, {"text": row["text"].strip()}, qs, gold, str(tox), "deusser:ToxiGen")


def m_tweeteval_offensive(row, i, ctx):
    lab = ctx.names[row["label"]] == "offensive"
    q = noul("Is `tweet` offensive: does it contain insults, threats, profanity or other offensive language?")
    return Item(i, {"tweet": row["text"]}, {"answer": q}, {"answer": g_noul(lab)}, str(lab), "programmatic:offensive")


def m_prompt_injections(row, i, ctx):
    q = noul("Is `text` a prompt injection: an attempt to override, ignore or hijack the instructions of an AI system?")
    lab = int(row["label"]) == 1
    return Item(i, {"text": row["text"]}, {"answer": q}, {"answer": g_noul(lab)}, str(lab), "deusser:PromptInjections")


# ---------------------------------------------------------------------- mappers: shown, not counted


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\.\.+", ".", text.replace("  ", " ")).strip()


def m_hellaswag(row, i, ctx):
    context = _hellaswag_clean(f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}")
    q = choice("Which ending is the most plausible continuation of `context`?", mc_criteria([_hellaswag_clean(e) for e in row["endings"]]))
    lab = _alpha(int(row["label"]))
    return Item(i, {"context": context}, {"answer": q}, {"answer": g_choice(lab)}, lab, "deusser:HellaSwag")


def m_winogrande(row, i, ctx):
    q = choice("Which option correctly fills the blank `_` in `sentence`?", mc_criteria([row["option1"], row["option2"]]))
    lab = _alpha(int(row["answer"]) - 1)
    return Item(i, {"sentence": row["sentence"]}, {"answer": q}, {"answer": g_choice(lab)}, lab, "deusser:WinoGrande")


def m_mmlu_pro(row, i, ctx):
    q = choice(MC_INSTRUCTION, mc_criteria(list(row["options"])))
    lab = _alpha(int(row["answer_index"]))
    return Item(i, {"subject": row["category"], "question": row["question"]}, {"answer": q}, {"answer": g_choice(lab)},
                row["category"], "deusser:MMLU (MC_INSTRUCTION)")


MAPPERS: dict[str, Callable[[dict, str, Ctx], Item | None]] = {
    "ag_news": m_ag_news, "yahoo_topics": m_yahoo, "fin_topic": m_fin_topic, "sst2": m_sst2,
    "fin_phrasebank": m_fin_phrasebank, "sst5": m_sst5, "yelp5": m_yelp5, "dair_emotion": m_dair_emotion,
    "tweeteval_emotion": m_tweeteval_emotion, "go_emotions": m_go_emotions, "banking77": m_banking77,
    "clinc150": m_clinc150, "massive": m_massive, "boolq": m_boolq, "rte": m_rte, "anli": m_anli, "paws": m_paws,
    "stsb": m_stsb, "llm_aggrefact": m_llm_aggrefact, "climate_fever": m_climate_fever, "toxic_chat": m_toxic_chat,
    "openai_moderation": m_openai_moderation, "typed_decisions": m_typed_decisions, "unfair_tos": m_unfair_tos,
    "helpsteer2": m_helpsteer2,
    "newsgroups20": m_newsgroups20, "tweet_topic": m_tweet_topic, "tweeteval_sentiment": m_tweeteval_sentiment,
    "app_reviews": m_app_reviews, "hwu64": m_hwu64, "snips": m_snips, "mrpc": m_mrpc, "scitail": m_scitail, "cb": m_cb,
    "atis": m_atis, "scifact": m_scifact,
    "toxigen": m_toxigen, "tweeteval_offensive": m_tweeteval_offensive, "prompt_injections": m_prompt_injections,
    "hellaswag": m_hellaswag, "winogrande": m_winogrande, "mmlu_pro": m_mmlu_pro,
}  # fmt: skip

# BTZSC config holding each dataset's hypotheses (None: not in BTZSC).
BTZSC_CONFIG = {"yahoo_topics": "yahootopics", "dair_emotion": "emotiondair", "banking77": "banking77", "massive": "massive"}


# ---------------------------------------------------------------------- variants and validation


def bare(questions: dict[str, dict]) -> dict[str, dict]:
    """Bare-names robustness variant: every choice criterion loses its description (value null)."""
    out = {}
    for qid, q in questions.items():
        out[qid] = {**q, "criteria": {k: None for k in q["criteria"]}} if q["type"] == "choice" else q
    return out


def shuffled(questions: dict[str, dict], rng) -> dict[str, dict]:
    """Option-shuffle rotation: every choice question's criteria in a seeded non-identity order."""
    out = {}
    for qid, q in questions.items():
        if q["type"] != "choice" or len(q["criteria"]) < 2:
            out[qid] = q
            continue
        keys = list(q["criteria"])
        perm = keys[:]
        for _ in range(10):
            rng.shuffle(perm)
            if perm != keys:
                break
        else:  # tiny K can keep landing on identity: rotate by one
            perm = keys[1:] + keys[:1]
        out[qid] = {**q, "criteria": {k: q["criteria"][k] for k in perm}}
    return out


def validate_request(req: dict) -> list[str]:
    """OpenRouter's schema, as observed: string instructions, string-or-null choice criteria, noul criteria
    both-or-neither strings, score levels strings (2..10), choice 2..255 options."""
    errs = []
    if not isinstance(req.get("questions"), dict) or not req["questions"]:
        return ["no questions"]
    if req.get("state") in (None, "", {}):
        errs.append("empty state")
    for qid, q in req["questions"].items():
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            errs.append(f"{qid}: instructions not a non-empty string")
        t = q.get("type")
        c = q.get("criteria")
        if t == "choice":
            if not isinstance(c, dict) or not 2 <= len(c) <= 255:
                errs.append(f"{qid}: choice needs 2..255 options")
            elif any(not (v is None or (isinstance(v, str) and v.strip())) for v in c.values()):
                errs.append(f"{qid}: choice criteria must be strings or null")
            elif any(k.strip().lower() in ("true", "false") for k in c):
                errs.append(f"{qid}: bare true/false choice key")
        elif t == "noul":
            if c is not None and (set(c) != {"true", "false"} or not all(isinstance(v, str) and v for v in c.values())):
                errs.append(f"{qid}: noul criteria must have both true and false strings")
        elif t == "score":
            if not isinstance(c, list) or not 2 <= len(c) <= 10 or not all(isinstance(v, str) and v for v in c):
                errs.append(f"{qid}: score needs 2..10 string levels")
        else:
            errs.append(f"{qid}: unknown type {t!r}")
    return errs


def chance_k(question: dict) -> int:
    t = question["type"]
    return 2 if t == "noul" else len(question["criteria"])
