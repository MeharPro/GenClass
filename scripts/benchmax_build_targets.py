"""Build bench/public/targets.json from bench/public/jev_published.json + the benchmax notes, then derive the
registry's `public_jev` exclusion family from it (benchmax PLAN §2.3.2) and write bench/public/exclusions.json.

One entry per published Jev row (549). Verdicts for the 131 counted benchmarks come from
docs/research/benchmax/feasibility-targets.md §3 (parsed); the rest are PLAN estimates (verdict_source says which).
Stdlib only (plus jev_local.bench.registry, itself stdlib); run on the Mac (tiny).

  python scripts/benchmax_build_targets.py                  # targets.json + exclusions.json; warn if the registry is stale
  python scripts/benchmax_build_targets.py --write-registry # also rewrite the GENERATED block of jev_local/bench/registry.py
  python scripts/benchmax_build_targets.py --check          # exit 1 if targets.json / registry / exclusions.json are stale
"""
import hashlib, importlib, json, math, os, re, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{ROOT}/bench/public/jev_published.json"
FEAS = f"{ROOT}/docs/research/benchmax/feasibility-targets.md"
OUT = f"{ROOT}/bench/public/targets.json"
REGISTRY_PY = f"{ROOT}/jev_local/bench/registry.py"
EXCL = f"{ROOT}/bench/public/exclusions.json"
SIZES = ("68m", "150m", "400m", "1b")
ARGS = set(sys.argv[1:])
CHECK = "--check" in ARGS

raw = open(SRC, "rb").read()
SHA = hashlib.sha256(raw).hexdigest()
rows = json.loads(raw)
assert len(rows) == 549, len(rows)

# ------------------------------------------------------------------ parse feasibility §3
t = open(FEAS).read()
sec = t[t.index("## 3. Verdict table"):t.index("## 4. Tallies")]
FEASROWS = {}
grp = None
for line in sec.splitlines():
    m = re.match(r"#### (\S+)\.", line)
    if m:
        grp = m.group(1)
    if line.startswith("|") and not line.startswith("|---") and not line.startswith("| Benchmark"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 9:
            name = c[0].strip("*").strip()
            FEASROWS[name] = dict(group=grp, jev=c[4], bar=c[5], S=c[6], Z=c[7], evidence=c[8])


def codes(s):
    s = s.replace("*", "")
    m = re.match(r"\s*([WTL]) ([WTL]) ([WTL]) ([WTL])", s)
    return dict(zip(SIZES, m.groups())) if m else None


def bars(s):
    s = s.replace("*", "").replace("≈", "")
    z = re.search(r"(?:^|Z\s*)(\.\d+)", s)
    sm = re.search(r"S\s*(?:≈)?\s*\.?(\d?\.\d+)(?:–(\.\d+))?", s)
    zb = float(z.group(1)) if z else None
    sb = None
    if sm:
        sb = float(sm.group(2)) if sm.group(2) else float(sm.group(1) if sm.group(1).startswith(".") else sm.group(1))
    return zb, sb


# ------------------------------------------------------------------ key registry
# feas: name of the feasibility §3 row; v: "SSSS/ZZZZ" PLAN estimate when no feasibility row.
# s: S-track eligibility; train: S-track training sources (train splits / related public data);
# zf: Z-track contamination flags to clear before any Z claim.
V2MIX = "v2 mix trains on this source (mix-v2.0-no-b6) -> Z claim needs the clean retrain"
DEVSET = "currently a jevbench-dev set (registry DEV) -> move dev elsewhere or declare dev-selection"
K = {}


def key(k, feas=None, v=None, s="train-split", train=(), zf=(), cat="", spec=None):
    K[k] = dict(feas=feas, v=v, s=s, train=list(train), zf=list(zf), cat=cat, spec=spec)


# --- Deußer core
key("ag_news", "AG News", train=["fancyzhx/ag_news train (120,000)"], cat="topic")
key("imdb", "IMDB", train=["stanfordnlp/imdb train (25,000)"], zf=[V2MIX + " (tsj imdb + counterfactual IMDB)"], cat="sentiment")
key("rotten_tomatoes", "Rotten Tomatoes", train=["rotten_tomatoes train (8,530)", "dedupe SST train vs RT test (Pang&Lee pool)"], zf=["registry sibling:rotten_tomatoes already excluded"], cat="sentiment")
key("sst2", "SST-2", train=["stanfordnlp/sst2 train (67,349 phrases); dev carved from train (eval = validation)"], cat="sentiment")
key("dair_emotion", "DAIR Emotion", train=["dair-ai/emotion split train (16,000); NEVER config 'unsplit' (contains test)"], cat="emotion")
key("fpb", "Financial PhraseBank", train=["atrost/financial_phrasebank train (3,100) / val (776)", "EXCLUDE nickmuchi/financial-classification, phrasebank_and_sentfin (contain FPB)"], cat="sentiment")
key("sms_spam", "SMS Spam", s="S-cv only (single split) or related corpora", train=["SetFit/enron_spam, phishing corpora (related)", "jev-bench sms_spam train 4,000 (only for the jev-bench row)"], zf=[V2MIX + " (sms_spam in KEPT_DESPITE_EARLIER_LIST): v2-derived number is VOID"], cat="spam")
key("language_id", "papluca language-id (20)", train=["papluca/language-identification train (70,000)"], cat="multilingual")
key("goemotions", "GoEmotions (28 nouls)", train=["go_emotions simplified train (43,410)"], cat="emotion")
key("banking77", "Banking77", train=["PolyAI banking77 train (10,003; 2 rows duplicate test -> drop)"], zf=["DMB pilot items come from train.csv: an S model is contaminated for that row"], cat="intent")
key("clinc150", "CLINC150 plus (151)", train=["clinc_oos plus train (15,250) + val (3,100) for OOS threshold"], cat="intent")
key("sib200", "SIB-200 (205 langs)", train=["Davlan/sib200 train (143,705)"], cat="multilingual")
key("anli", "ANLI r1-r3", train=["facebook/anli train r1-r3 (162,865) + MNLI/FEVER-NLI/WANLI"], cat="nli")
key("afrixnli", "AfriXNLI (18 langs)", s="related-data only (validation split only)", train=["masakhane/afrixnli validation (8,100) for selection only", "facebook/xnli MT train"], cat="multilingual")
key("paws", "PAWS", train=["paws labeled_final train (49,401)"], cat="pair")
key("llm_aggrefact", "LLM-AggreFact (11 sets)", s="Z only (card forbids training on dev/test)", train=["related: MiniCheck C2D/D2C, ANLI, DocNLI, VitaminC (dedupe vs AggreFact test, incl. RAGTruth test)"], cat="grounding")
key("boolq", "BoolQ", train=["google/boolq train (9,427)"], cat="qa")
key("belebele", "Belebele (122 langs)", s="related-data only (no train split)", train=["RACE, SciQ, MultiRC, MCTest, ReClor (Belebele recipe)"], cat="multilingual")
key("pubmedqa", "PubMedQA pqa_labeled", s="related-data only (pqa_labeled IS the eval)", train=["qiaojin/PubMedQA pqa_artificial (211,269) only"], cat="knowledge")
key("mmlu", "MMLU", train=["cais/mmlu auxiliary_train (99,842)"], cat="knowledge")
key("ceval", "C-Eval (Chinese)", train=["ceval-exam val (1,346) / dev (260)"], cat="knowledge")
key("bigbench", "BIG-bench MC (93 tasks)", train=["tasksource/bigbench train of the same 93 tasks (BBH items must be removed)"], cat="knowledge")
key("hellaswag", "HellaSwag", train=["Rowan/hellaswag train (39,905)"], cat="knowledge")
key("winogrande", "WinoGrande", train=["winogrande_xl train (40,398)"], cat="knowledge")
key("arc", "ARC E+C", train=["ai2_arc train (3,370)"], cat="knowledge")
key("csqa", "CommonsenseQA", train=["tau/commonsense_qa train (9,741)"], zf=[V2MIX + " (tasksource commonsense_qa)"], cat="knowledge")
key("alphanli", "alphaNLI (ART)", train=["allenai/art train (169,654)"], zf=[V2MIX + " (tasksource art)"], cat="nli")
key("toxigen", "ToxiGen annotated", train=["toxigen annotated train (8,960) + machine-generated (capped)"], zf=[DEVSET], cat="safety")
key("openai_mod", "OpenAI moderation (8 nouls)", s="related-data only (eval-only set)", train=["Aegis 1.0/2.0, BeaverTails, civil_comments (related)", "EXCLUDE KoalaAI/Text-Moderation* (trained on this eval)"], cat="safety")
key("toxicchat", "ToxicChat 0124", train=["lmsys/toxic-chat 0124 train (5,082)"], cat="safety")
key("prompt_injections", "deepset prompt-injections", train=["deepset/prompt-injections train (546)"], zf=[DEVSET, "training on deepset train voids the all-662 rows (ASEVlad, Gaurav-Gosain, switchboard, kiwi0719)"], cat="safety")
key("agb_de", "AGB-DE (German)", train=["d4br4/agb-de train (3,004); German/multilingual backbone needed"], cat="legal")
key("unfair_tos", "UNFAIR-ToS (8 nouls, positives only)", train=["lex_glue unfair_tos train (5,532) / val (2,275)"], cat="legal")
key("stsb", "STS-B", train=["stsb train (5,749)"], cat="pair")
key("sst5", "SST-5", train=["SetFit/sst5 train (8,544)"], cat="sentiment")
key("summeval", "SummEval (4 scores)", s="Z only (test-only set)", train=["none usable"], cat="judge")
key("helpsteer2", "HelpSteer2 (5 scores)", train=["nvidia/HelpSteer2 train (20,324); HelpSteer/HelpSteer3 deduped vs validation prompts"], cat="judge")
key("toxicchat_jailbreak", v="TTTT/LLLL", train=["lmsys/toxic-chat train jailbreaking label"], cat="safety")
key("memorization_probe", v="----/----", s="n/a (diagnostic)", cat="diagnostic")
# --- Decision Index
DI_RULES = "DI rules: one rendering, no per-benchmark calibration, no truncation (refuse)"
key("bfcl", "BFCL", s="related-data only", train=["ToolACE, glaive-function-calling-v2, xlam-60k (gated)"], cat="tools")
key("toolret", "ToolRet", train=["mangopy/ToolRet-Training-20w (208,826; dedupe vs API-Bank/BFCL/When2Call)"], cat="tools")
key("api_bank", "API-Bank", train=["API-Bank training data (liminghao1630/API-Bank)"], cat="tools")
key("home_appliance", "Home appliance sim.", s="Z only (generator)", cat="tools")
key("when2call", "When2Call", train=["nvidia/When2Call train_sft (15,000) / train_pref (9,000); dedupe vs BFCL"], cat="tools")
key("contractnli", "ContractNLI", train=["ContractNLI train 423 docs / dev 61"], zf=[V2MIX + " (tsj contract-nli)"], cat="legal")
key("bpomp", "BPoMP (limericks)", s="related-data only (self-made perturbations)", cat="arts")
key("humicroedit", "Humicroedit", train=["SemEval-2020 T7 subtask-2 train (9,381) + FunLines"], zf=[V2MIX + " (tsj humicroedit)"], cat="arts")
key("pop909", "POP909-CL (129 chords)", train=["POP909 songs not sampled by DI + other symbolic chord corpora"], cat="arts")
key("cfcolor", "cfcolor", train=["cfcolor own train ratings (train_vec)"], cat="arts")
key("gpqa", "GPQA Diamond", s="none (GPQA main/extended contain Diamond)", cat="knowledge")
key("gsm8k_mc", "GSM8K (MC)", train=["openai/gsm8k train (7,473)"], zf=["DI issue #32 gold-rank shortcut: run question-blind check"], cat="knowledge")
key("chessbench", "ChessBench", train=["searchless_chess train (action-values) / Lichess"], zf=["14.7% of test boards also in train (inherent; disclose)"], cat="games")
key("musr", "MuSR", s="S+generator only", cat="knowledge")
key("sata", "SATA-Bench", s="none", cat="knowledge")
key("cruxeval", "CRUXEval", s="none", cat="knowledge")
key("cladder", "CLadder", s="S+generator only (DI samples the whole released set)", train=["CLadder generator, new seeds"], zf=[V2MIX + " (tsj cladder; item overlap possible)"], cat="reasoning")
key("hle", "HLE", s="none", cat="knowledge")
key("mmlu_pro", "MMLU-Pro", s="none (validation 70 = CoT shots)", train=["MMLU aux_train; EXCLUDE MMLU test, TheoremQA, SciBench"], cat="knowledge")
key("bbh", "BBH", s="none (BIG-bench canary)", train=["EXCLUDE BIG-bench parent-task train splits"], cat="knowledge")
key("bright", "BRIGHT", s="related-data only", train=["reasonir/reasonir-data (CC BY-NC)"], cat="retrieval")
key("esci", "Amazon ESCI", train=["tasksource/esci train (2,027,874; cap 300k)"], zf=[V2MIX + " (tsj esci; item overlap possible)"], cat="retrieval")
key("acos", "ACOS", train=["NEUDM/acos train (4,464) / val (497)"], cat="sentiment")
key("finentity", "FinEntity", s="related-data only (DI scores all 979 docs)", train=["twitter-financial-news-sentiment, FiQA sentiment (related)"], cat="finance")
key("isarcasm", "iSarcasmEval", train=["iSarcasmEval task A En train (3,468) + SemEval-18 T3, SARC"], cat="sentiment")
key("vast", "VAST", train=["VAST train (13,477) / dev (2,062)"], cat="stance")
key("nli4ct", "NLI4CT", train=["tasksource/nli4ct train (1,700) / val (200)"], cat="nli")
key("ragtruth", "RAGTruth", train=["wandb/RAGTruth-processed train (15,090)"], cat="grounding")
key("hover", "HoVer", train=["HoVer train (18,171)"], cat="grounding")
key("phishnchips", "PhishNChips", s="related-data only", train=["phishing-email corpora (zefang-liu, seven-phishing, cybersectony)"], cat="spam")
key("forecastbench", "ForecastBench", train=["questions resolved before 2026-07-01"], cat="knowledge")
key("habermas", "Habermas Machine", train=["Habermas TRAIN cohorts"], cat="arts")
key("newyorker", "New Yorker", train=["matching train (9,792)"], cat="arts")
key("routerbench", "RouterBench (non-index)", train=["RouterBench (cap 50k, outside DI's sample)"], cat="routing")
key("sgd", "SGD/SGD-X (non-index)", train=["schema_guided_dstc8 train (16k dialogues)"], cat="intent")
key("arc_easy", "ARC-Easy (non-index)", train=["ai2_arc train"], cat="knowledge")
key("arc_challenge", "ARC-Challenge (non-index)", train=["ai2_arc train"], cat="knowledge")
key("obqa", "OpenBookQA (non-index)", train=["openbookqa train (4,957)"], cat="knowledge")
key("support_tickets", "Support-ticket calibration (non-index)", s="Z only (synthetic, no train)", cat="intent")
key("phishing_gradient", "Phishing difficulty gradient (non-index)", s="related-data only", cat="spam")
key("email_spam_di", "Email spam (non-index)", s="related-data only", train=["Enron/Ling-Spam/Kaggle spam corpora"], cat="spam")
key("di_index", v="LLLL/LLLL", s="DI rules (one rendering)", cat="composite")
key("di_agentic", v="----/----", s="n/a (multi-step environments)", cat="diagnostic")
# --- other suites (group C)
key("nlupp", "NLU++ (folds 18-19)", train=["NLU++ folds 0-17 (folds 16-17 = DMB validation)"], zf=["NLU++ folds 18-19 are currently our dev set -> move dev"], cat="intent")
key("tweet_topic", "tweet_topic_single", train=["tweet_topic_single train_2021 (1,516) / train_all (4,374)"], zf=[DEVSET], cat="topic")
key("fin_topic", "Twitter Fin. News Topic (20)", train=["zeroshot/twitter-financial-news-topic train (16,990)"], zf=["shares tweet pool with twitter-financial-news-sentiment"], cat="topic")
key("daily_dialog", "daily_dialog emotion (7)", train=["DailyDialog train (11,118 dialogues)"], cat="emotion")
key("tweeteval_emotion", "TweetEval emotion (4)", train=["tweet_eval emotion train (3,257)"], cat="emotion")
key("arxiv_2026", "arXiv 2026-09 (8 cats)", train=["pre-2026-09-17 arXiv abstracts (dedupe by id)"], zf=[V2MIX + " (arxiv source)"], cat="topic")
key("typed_decisions", "typed-decisions", train=["typed-decisions train (1,200 cases) -> 'fitted' table only"], zf=["v2 b5 open_jev workflow-controls share the 4 workflow names"], cat="typed")
# --- group D
key("jevals_pubmedqa", "Jevals PubMedQA (yes/no)", s="related-data only", train=["pqa_artificial"], cat="knowledge")
key("jevals_helpsteer", "Jevals HelpSteer2 helpfulness", train=["HelpSteer2 train"], cat="judge")
key("btzsc_emotion", "BTZSC pilot emotion", train=["dair-ai/emotion train"], zf=["registry sibling:btzsc"], cat="emotion")
key("btzsc_b77", "BTZSC pilot Banking77 (72)", train=["banking77 train"], zf=["registry sibling:btzsc"], cat="intent")
key("btzsc_ag", "BTZSC pilot AG News", train=["ag_news train"], zf=["registry sibling:btzsc"], cat="topic")
key("hfblog_macro", "SST-2/AG/Emotion/B77 macro", train=["component train splits"], cat="composite")
key("clinc_certify", "CLINC in-scope top-1", train=["clinc train"], cat="intent")
key("enron", "Enron spam", train=["SetFit/enron_spam train (31,716)"], cat="spam")
key("arize_spam", "Email spam 18,514", s="S-cv only (all rows evaluated)", cat="spam")
key("hellaswag_vercel", "HellaSwag (Vercel harness)", train=["hellaswag train"], cat="knowledge")
key("general_decisions", "'General decisions' (B77/BoolQ/Yelp/ChaosNLI)", cat="composite")
key("jevbench", "JevBench v1.5.5", s="none (public items: disclosed, gap-penalised)", cat="typed")
key("jabr", "jabr classifier-benchmark v2", s="none (README: do not train)", cat="typed")
key("decisionbench", "DecisionBench", s="none (eval-only)", cat="typed")
key("decidebench", "DecideBench v1.1", s="none", cat="typed")
# --- E-A
key("hwu64", "HWU64 (64 intents)", train=["DeepPavlov/hwu64 train (8,954)"], zf=[DEVSET, "MASSIVE/SLURP lineage: dedupe MASSIVE train vs HWU64 test"], cat="intent")
LEX = "lex_glue {} train"
key("ecthr_a", "LexGLUE ECtHR A", train=[LEX.format("ecthr_a") + " (9,000)"], cat="legal")
key("ecthr_b", "LexGLUE ECtHR B", train=[LEX.format("ecthr_b") + " (9,000)"], cat="legal")
key("scotus", "LexGLUE SCOTUS (13 areas)", train=[LEX.format("scotus") + " (5,000)"], cat="legal")
key("eurlex", "LexGLUE EUR-LEX (100 EuroVoc)", train=[LEX.format("eurlex") + " (55,000)"], cat="legal")
key("ledgar", "LexGLUE LEDGAR (100 provisions)", train=[LEX.format("ledgar") + " (60,000)"], zf=[V2MIX + " (ledgar)"], cat="legal")
key("lexglue_unfair", "LexGLUE UNFAIR-ToS ('none' convention)", train=[LEX.format("unfair_tos") + " (5,532)"], cat="legal")
key("casehold", "LexGLUE CaseHOLD (5-way)", train=[LEX.format("case_hold") + " (45,000)"], zf=[V2MIX + " (tasksource case_hold)"], cat="legal")
key("lexglue_mean", "LexGLUE 7-task mean (shown, not counted)", train=["all 7 LexGLUE train splits"], cat="composite")
key("medhallu", "MedHallu", train=["MedHallu pqa_artificial-derived train (9,000)"], zf=["MedHallu test is built from PubMedQA pqa_labeled (= Deußer PubMedQA eval)"], cat="grounding")
key("aegis1_prompt", "Aegis 1.0 prompt", train=["Aegis 1.0 train (10,798)"], cat="safety")
key("aegis2_prompt", "Aegis 2.0 prompt", train=["Aegis 2.0 train (30,007)"], zf=[V2MIX + " (aegis2)"], cat="safety")
key("aegis2_response", "Aegis 2.0 response", train=["Aegis 2.0 train (30,007)"], zf=[V2MIX + " (aegis2)"], cat="safety")
key("wildguard", "WildGuardTest prompt", train=["allenai/wildguardmix train (86,759; gated, AI2 licence - user decision)"], cat="safety")
key("hatecheck", "HateCheck", s="Z only (functional test suite)", cat="safety")
key("harmbench_resp", "HarmBench response", s="related-data only", train=["Aegis 2.0, WildGuardMix, BeaverTails"], cat="safety")
key("harmbench_prompt", "HarmBench prompt", s="related-data only", cat="safety")
key("injection_combined", "Combined injection corpus (4 HF sets, deduped)", s="Z only (eval pool contains the train splits)", cat="safety")
for k2, f in [("rewardbench1", "RewardBench v1"), ("rewardbench2", "RewardBench 2"), ("rmbench_pair", "RM-Bench pairwise"),
              ("rmbench_point", "RM-Bench pointwise"), ("rubricbench", "RubricBench (rubric given)"),
              ("ppe", "PPE Human Preference V1"), ("processbench", "ProcessBench"), ("prmbench", "PRMBench Preview")]:
    key(k2, f, s="related-data only (eval-only sets)", train=["HelpSteer3, UltraFeedback, Skywork preference, PRM800K, Math-Shepherd (deduped; EXCLUDE arena-human-preference for PPE)"], cat="reward")
key("tmmlu", "TMMLU+ (zh-TW, 66 subjects)", s="none", cat="knowledge")
key("gaokao", "GAOKAO-Bench objective", s="none", cat="knowledge")
key("jmedqa", "JMedQA (JMLE 2018–26)", s="none", cat="knowledge")
key("upworthy", "Upworthy headline A/B", train=["Upworthy archive exploratory + holdout (eval = confirmatory); licence to confirm"], cat="arts")
key("scifact", "BEIR SciFact rerank", train=["allenai/scifact train claims (809); MS MARCO (Z-eligible)"], zf=["SciFact validation is a jevbench-dev set (claims config); keep BEIR test queries out"], cat="retrieval")
key("nfcorpus", "BEIR NFCorpus rerank", train=["NFCorpus train queries; MS MARCO (Z-eligible)"], cat="retrieval")
key("rerank_mean", "Rerank mean (BEIR + BRIGHT + CodeSearchNet, 8 sets)", train=["MS MARCO; component train splits"], cat="retrieval")
key("nevir", "NevIR (negation pairs)", train=["orionweller/NevIR train (948)"], cat="retrieval")
# --- E-B
key("yelp5", "Yelp-5", train=["jev-bench yelp5 train (8,000) / Yelp full train (650,000)"], cat="sentiment")
key("civil_comments", "civil_comments (noul)", train=["jev-bench civil_comments train (8,000; from HF validation)"], zf=[V2MIX + " (civil_comments)"], cat="safety")
key("mhs", "measuring_hate_speech (score)", train=["jev-bench mhs train (comment-level; single-split source)"], zf=[V2MIX + " (mhs) and possible TEST leakage (single-split pool)"], cat="safety")
key("mnli", "MNLI", train=["MNLI train (392,702)"], zf=[V2MIX + " (mnli)"], cat="nli")
key("chaosnli", "ChaosNLI (MNLI part)", train=["SNLI + MNLI train; EXCLUDE MNLI validation_matched"], zf=[V2MIX + " (mnli)"], cat="nli")
key("fever", "FEVER (gold evidence, noul)", train=["copenlu/fever_gold_evidence train (228,277)"], zf=[V2MIX + " (fever_nli)"], cat="grounding")
key("strategyqa_closed", "StrategyQA closed-book", train=["StrategyQA train (1,603)"], cat="knowledge")
key("strategyqa_grounded", "StrategyQA grounded", train=["StrategyQA train (1,603)"], cat="knowledge")
key("massive_en", "MASSIVE en-US (60 intents)", train=["MASSIVE en-US train (11,514)"], zf=["MASSIVE is jevbench test #13 (already excluded); S use voids jevbench #13 zero-shot"], cat="intent")
key("goemotions_single", "GoEmotions single-label \"primary\"", train=["jev-bench go_emotions train (8,000)"], cat="emotion")
key("hs2_help_acc", "HelpSteer2 helpfulness (5 levels)", train=["jev-bench helpsteer2 train (8,000)"], cat="judge")
key("hs2_verb_acc", "HelpSteer2 verbosity (5 levels)", train=["jev-bench helpsteer2 train (8,000)"], cat="judge")
key("stsb_6level", "STS-B 6-level", train=["stsb train (5,749)"], cat="pair")
key("jevbench_hf_macro", "jev-bench macro, 22 configs (shown, not counted)", train=["jev-bench train configs"], cat="composite")
# --- G: new tier-A counted rows (resume pass 2026-10-03), PLAN estimates
key("xquad_rerank", v="LLLL/LLLL", s="Z only", cat="retrieval")
key("nanorteb", v="LLLT/LLLT", s="related-data only", train=["MS MARCO + component train splits"], cat="retrieval")
key("metamedqa", v="LLLL/LLLL", s="related-data only", train=["MedQA-USMLE, MedMCQA (dedupe)"], cat="knowledge")
key("diagnosisarena", v="LLLL/LLLL", s="none", cat="knowledge")
key("mcphunt", v="LLTT/LLTT", s="related-data only", train=["R-Judge/ATBench-style agent-trace safety data (dedupe)"], cat="agent-safety")
key("ricechem", v="LLLT/LLLT", s="S-cv only", cat="judge")
key("halueval_qa", v="LLLT/LLLT", s="Z only (no train split)", train=["related: RAGTruth train, MiniCheck synthetic"], cat="grounding")
key("cyberseceval_cwe", v="LLTT/LLLL", s="related-data only", train=["public CWE-labelled code corpora (dedupe vs CyberSecEval)"], cat="code")
key("trec_fine", v="WWWW/LTTT", train=["TREC train (5,452) fine labels"], zf=[V2MIX + " (trec)"], cat="intent")
for k2 in ("triviaqa4", "popqa4", "simpleqa4", "mmlu_cf", "daily_oracle"):
    key(k2, v="LLLL/LLLL", s="none", cat="knowledge")
key("wainject", v="LTTT/LLTT", s="related-data only", train=["deepset, jackhhao, SPML injection train splits (dedupe)"], cat="safety")
key("rexerr", v="LTTT/LTTT", s="related-data only (credentialed source)", train=["MedNLI/RadNLI-style data if licensable"], cat="grounding")
key("dimabsa_va", v="LLLT/----", s="train-split (Jev number is itself train-fitted)", train=["SemEval-2026 T3 DimABSA official train"], cat="sentiment")
key("dimabsa_triplet", v="LLLL/----", s="train-split (needs span extraction; no head for it)", train=["SemEval-2026 T3 official train"], cat="extraction")
key("cesnet", v="WWWW/TTTT", train=["CESNET-QUICEXT-25 training weeks (serialized packet features)"], cat="non-text")
key("nslkdd", v="TTTT/LLLL", train=["NSL-KDD KDDTrain+ (125,973)"], cat="non-text")
key("amazon_polarity", v="LTWW/LLLL", train=["amazon_polarity train (3.6M; cap)"], zf=[V2MIX + " (amazon_polarity in KEPT list)"], cat="sentiment")
key("cfpb", v="WWWW/LTTT", train=["CFPB complaints outside earino's eval rows (carve by complaint id)"], cat="intent")
key("massive_scenario51", v="LLLL/LLLL", train=["MASSIVE train, 51 locales (needs multilingual backbone)"], cat="multilingual")
# --- context keys (PLAN estimates)
ctx = {
    "lw_injection": ("TTTT/LLTT", "safety"), "lw_moderation": ("TTTT/LLTT", "safety"), "lw_pii": ("TTTT/LLLL", "safety"),
    "lw_rag": ("TTTT/LLLT", "grounding"), "lw_offtopic": ("TTWW/LLTT", "intent"), "lw_routing20": ("TWWW/LLLL", "intent"),
    "lw_toolrouting": ("LLLL/LLLL", "tools"), "lw_complaint": ("TWWW/LLLL", "intent"), "lw_commit": ("TTTT/LLLL", "code"),
    "lw_search": ("TTTW/LLLL", "retrieval"), "lw_webagent": ("LLLL/LLLL", "agent"), "lw_community": ("LLLL/LLLL", "composite"),
    "jevbench_public": ("LLLT/LLLL", "typed"), "jevbench_composite": ("LTTT/LTTT", "typed"), "jevbench_calib": ("TTTT/TTTT", "typed"),
    "typesafe_workflow": ("LLTT/LLTT", "typed"), "ajgt": ("LLLL/LLLL", "multilingual"), "sst2_injection": ("TTTT/LLTT", "safety"),
    "logiqa": ("LLLL/LLLL", "knowledge"), "msmarco": ("TTTT/LLTT", "retrieval"), "boolq_negation": ("LLTT/LLLL", "qa"),
    "conll_typing": ("WWWW/TTTT", "extraction"), "topic_reworded": ("WWWW/LLLL", "topic"), "squad_select": ("TWWW/LLTT", "qa"),
    "trec_coarse": ("WWWW/LTTT", "intent"), "truthfulqa": ("LLLL/LLLL", "knowledge"), "xnli": ("LLLL/LLLL", "multilingual"),
    "css_big": ("TWWW/LLLL", "css"), "css_lowjev": ("TWWW/LLTT", "css"), "css_narrow": ("LLTT/LLLL", "css"),
    "css_jevahead": ("LLLL/LLLL", "css"), "css_badbaseline": ("LLLT/LLLL", "css"), "css_talklife": ("LLLL/LTTT", "css"),
    "amazon_counterfactual": ("WWWW/LTTT", "sentiment"), "xstest": ("TTTT/TTTT", "safety"), "kev": ("LLTT/LLLL", "typed"),
    "wanli": ("TTTT/LLLT", "nli"), "webjev_misc": ("LLLL/LLLL", "typed"), "injection_deepset_all": ("LLTT/LLTT", "safety"),
    "redhat": ("----/----", "safety"), "cohen_screening": ("----/----", "diagnostic"), "cohen_adhd": ("TTTT/LLTT", "screening"),
    "exams_misc": ("LLLL/LLLL", "knowledge"), "kobest": ("LLLL/LLLL", "multilingual"), "email_spam": ("TTTT/TTTT", "spam"),
    "eclipse_severity": ("WWWW/LLTT", "code"), "massive_sv": ("LLLL/LLLL", "multilingual"), "tanaos": ("WWWW/LLTT", "intent"),
    "darija": ("LLLL/LLLL", "multilingual"), "tabfact": ("LLLL/LLLL", "grounding"), "fake_jobs": ("WWWW/TTTT", "spam"),
    "app_reviews": ("TWWW/LLTT", "sentiment"), "cuad": ("LLTT/LLLT", "legal"), "whowhen": ("LLLL/LLLL", "agent"),
    "mind2web": ("LLLL/LLLL", "agent"), "aita": ("TTTT/LLTT", "judge"), "commonlit": ("LTWW/LLLL", "score"),
    "fiqa_rerank": ("LTTT/LTTT", "retrieval"), "label_pressure": ("LLLL/LLLL", "intent"), "tabular": ("TTTT/TTTT", "non-text"),
    "dblp_acm": ("TTTT/LLTT", "pair"), "scifact_claims": ("TTTT/LLTT", "grounding"), "phishing_small": ("TTTT/LLTT", "spam"),
    "nejm": ("LLLL/LLLL", "knowledge"), "agent_trace": ("LLTT/LLLT", "agent-safety"), "healthbench": ("LLLL/LLLL", "judge"),
    "ellipse": ("WWWW/TTTT", "score"), "dialog_quality": ("TTTT/LLTT", "judge"), "judgebench": ("LLLL/LLLL", "judge"),
    "halueval_misc": ("LLTT/LLTT", "grounding"), "cjbench": ("LLLL/LLLL", "multilingual"), "radiology": ("LLLL/LLLL", "judge"),
    "speech_rescoring": ("LLLL/----", "non-text"), "recsys": ("LLTT/LLLL", "retrieval"), "har": ("TTTT/LLLL", "non-text"),
    "video_quality": ("LLTT/LLLL", "non-text"), "injecagent": ("----/----", "agent-safety"), "jevout": ("LLLL/LLLL", "knowledge"),
    "sys1cal": ("TTTT/TTTT", "calibration"), "crt": ("LLLL/LLLL", "knowledge"), "alfworld": ("----/----", "agent"),
    "xdviolence": ("LLLL/LLLL", "non-text"), "kobbq": ("LLLL/LLLL", "multilingual"), "multihop_retrieval": ("LLTT/LLLL", "retrieval"),
    "kiki_agent": ("LLTT/LLLL", "agent"), "gulf_arabic": ("LLLL/LLLL", "multilingual"), "race_h": ("LLLL/LLLL", "knowledge"),
    "amazonqa": ("TWWW/TWWW", "qa"), "p4g": ("TTTT/LLTT", "dialogue"), "craigslist": ("LLTT/LLLL", "dialogue"),
    "nimble": ("LLLL/LLLL", "typed"), "diag": ("----/----", "diagnostic"),
}
for k2, (v, cat) in ctx.items():
    if k2 not in K:
        key(k2, v=v, s="see source study", cat=cat)
K["xstest"]["s"] = "Z only"
K["injection_deepset_all"]["s"] = "Z only (eval includes deepset train)"
K["arize_spam"]["s"] = "S-cv only"
K["typesafe_workflow"].update(s="Z only (no train split; labels are closed-model consensus)", zf=["v2 b5 open_jev workflow-controls share the 4 workflow names; typed-decisions-trained S checkpoints barred"])
K["css_talklife"]["s"] = "none (TalkLife data not public)"
for k2 in ("css_big", "css_lowjev", "css_narrow", "css_jevahead", "css_badbaseline"):
    K[k2]["train"] = ["original CSS dataset train splits (Ziems et al. baselines/ scripts)"]

# ------------------------------------------------------------------ row -> (key, role)
# roles: H headline (counted benchmark), S Jev-given-train-data bar, 2 secondary (same benchmark, other
# harness/metric, full split), C context (sample / small n / one-off), M composite, X diagnostic (not a target).
M = {}


def m(idx, k, role):
    for i in (idx if isinstance(idx, (list, tuple, range)) else [idx]):
        assert i not in M, i
        M[i] = (k, role)


deu = ["ag_news", "imdb", "rotten_tomatoes", "sst2", "dair_emotion", "fpb", "banking77", "clinc150", "sib200",
       "language_id", "sms_spam", "goemotions", "anli", "afrixnli", "paws", "llm_aggrefact", "boolq", "belebele",
       "pubmedqa", "mmlu", "ceval", "bigbench", "hellaswag", "winogrande", "arc", "csqa", "alphanli", "toxigen",
       "openai_mod", "toxicchat", "prompt_injections", "agb_de", "unfair_tos", "stsb", "sst5", "summeval", "helpsteer2"]
for i, k2 in enumerate(deu):
    m(i, k2, "H")
m(37, "banking77", "2"); m(38, "dair_emotion", "2"); m([39, 40], "clinc150", "2"); m(41, "llm_aggrefact", "2")
m(42, "sst5", "2"); m(43, "toxicchat_jailbreak", "2"); m(44, "unfair_tos", "S"); m(45, "goemotions", "S"); m(46, "toxicchat", "S")
m(range(47, 54), "memorization_probe", "X")
di = {54: "bfcl", 55: "toolret", 56: "api_bank", 59: "routerbench", 61: "home_appliance", 62: "sgd", 63: "contractnli",
      70: "bpomp", 71: "humicroedit", 72: "pop909", 73: "cfcolor", 75: "gpqa", 76: "arc_easy", 77: "arc_challenge",
      80: "gsm8k_mc", 81: "chessbench", 82: "musr", 83: "sata", 84: "bright", 85: "esci", 86: "acos", 87: "finentity",
      88: "isarcasm", 89: "vast", 90: "nli4ct", 91: "cruxeval", 92: "cladder", 93: "hle", 94: "forecastbench",
      95: "habermas", 96: "obqa", 98: "support_tickets", 99: "phishing_gradient", 100: "email_spam_di",
      101: "phishnchips", 102: "mmlu_pro", 103: "bbh", 104: "ragtruth", 105: "hover", 106: "when2call", 107: "newyorker"}
for i, k2 in di.items():
    m(i, k2, "H")
m(57, "banking77", "2"); m(58, "clinc150", "2"); m(64, "anli", "2"); m(74, "mmlu", "2"); m(78, "winogrande", "2")
m(79, "hellaswag", "2"); m(97, "csqa", "2"); m([60, 65, 66, 67, 68, 69], "di_agentic", "X"); m(108, "di_index", "M")
m([109], "banking77", "2"); m([110, 111], "clinc150", "2"); m(112, "nlupp", "H"); m([113, 115, 116, 118], "banking77", "C")
m([114, 117], "sms_spam", "C"); m(119, "diag", "X")
m([120, 121], "banking77", "C"); m([122, 123], "jevals_helpsteer", "C"); m([124, 125], "jevals_pubmedqa", "C")
m(126, "typed_decisions", "H"); m([127, 128, 129, 130], "typed_decisions", "2")
m(131, "typesafe_workflow", "M"); m([132, 133, 134, 135], "typesafe_workflow", "C")
m(136, "jevbench_composite", "C"); m(137, "jevbench", "C"); m(138, "jevbench_calib", "C")
m(range(139, 144), "decisionbench", "C"); m(144, "decidebench", "C")
lw = ["lw_injection", "lw_moderation", "lw_pii", "lw_rag", "lw_offtopic", "lw_routing20", "banking77", "lw_toolrouting",
      "lw_complaint", "lw_commit", "lw_search", "typed_decisions", "lw_webagent", "lw_community", "jevbench_public"]
for j, k2 in enumerate(lw):
    m(145 + j, k2, "C")
jb = {160: ("arc", "C"), 161: ("banking77", "C"), 162: ("boolq", "C"), 163: ("chaosnli", "H"), 164: ("civil_comments", "H"),
      165: ("clinc150", "C"), 166: ("fever", "H"), 167: ("goemotions_single", "H"), 168: ("hs2_help_acc", "H"),
      169: ("hs2_verb_acc", "H"), 170: ("ledgar", "C"), 171: ("massive_en", "H"), 172: ("mhs", "H"), 173: ("mmlu", "C"),
      174: ("mnli", "H"), 175: ("paws", "C"), 176: ("sms_spam", "C"), 177: ("sst5", "C"), 178: ("strategyqa_closed", "H"),
      179: ("strategyqa_grounded", "H"), 180: ("stsb_6level", "H"), 181: ("yelp5", "H"), 182: ("jevbench_hf_macro", "M")}
for i, (k2, r) in jb.items():
    m(i, k2, r)
jdb = ["ag_news", "anli", "ajgt", "arc", "banking77", "boolq", "csqa", "dair_emotion", "helpsteer2", "gsm8k_mc", "hellaswag",
       "sst2_injection", "sst2_injection", "language_id", "logiqa", "mmlu", "mnli", "msmarco", "boolq_negation",
       "conll_typing", "nfcorpus", "topic_reworded", "paws", "yelp5", "sms_spam", "squad_select", "sst2", "stsb",
       "civil_comments", "trec_coarse", "truthfulqa", "squad_select", "winogrande", "xnli", "xnli", "xnli", "xnli", "xnli", "xnli"]
for j, k2 in enumerate(jdb):
    m(183 + j, k2, "C")
css = {222: "css_big", 223: "css_jevahead", 224: "css_lowjev", 225: "css_big", 226: "css_narrow", 227: "css_lowjev",
       228: "css_badbaseline", 229: "css_big", 230: "css_narrow", 231: "css_badbaseline", 232: "css_jevahead",
       233: "css_big", 234: "css_badbaseline", 235: "css_talklife", 236: "css_jevahead", 237: "css_jevahead",
       238: "css_big", 239: "css_big"}
for i, k2 in css.items():
    m(i, k2, "C")
m(240, "btzsc_ag", "C"); m(241, "btzsc_b77", "C"); m(242, "btzsc_emotion", "C")
m([243, 244], "dair_emotion", "2"); m(245, "tweet_topic", "H"); m(246, "fin_topic", "H"); m(247, "daily_dialog", "H")
m(248, "ag_news", "C"); m(249, "sst2", "2"); m(250, "banking77", "C"); m(251, "tweeteval_emotion", "H"); m(252, "paws", "C")
m(253, "arxiv_2026", "H")
m(254, "banking77", "2"); m(255, "clinc150", "2"); m(256, "hwu64", "H"); m(257, "sst2", "2"); m(258, "stsb", "2")
m(259, "banking77", "2"); m([260, 261], "banking77", "S")
m(262, "banking77", "2"); m(263, "prompt_injections", "2"); m(264, "sms_spam", "C"); m(265, "dair_emotion", "C")
m(266, "amazon_counterfactual", "C"); m(267, "massive_en", "C"); m(268, "xstest", "C")
for i, k2 in zip(range(269, 276), ["ecthr_a", "ecthr_b", "scotus", "eurlex", "ledgar", "lexglue_unfair", "casehold"]):
    m(i, k2, "H")
m(276, "lexglue_mean", "M"); m(277, "lexglue_mean", "S"); m(278, "banking77", "2"); m([279, 280], "clinc150", "2")
m(range(281, 286), "kev", "C"); m(286, "mmlu", "C"); m(287, "mmlu_pro", "C"); m(288, "wanli", "C")
m(289, "jevbench_public", "C"); m(290, "mmlu_pro", "C"); m(291, "typed_decisions", "C"); m([292, 293], "kev", "C")
m([294, 295, 296], "nimble", "C")
for i, k2 in zip(range(297, 305), ["hatecheck", "toxicchat", "aegis1_prompt", "aegis2_prompt", "aegis2_response",
                                     "wildguard", "harmbench_prompt", "harmbench_resp"]):
    m(i, k2, "2" if k2 == "toxicchat" else "H")
m([305, 306], "injection_deepset_all", "2"); m(307, "injection_deepset_all", "C"); m(308, "injection_combined", "H")
m([309, 310], "redhat", "C")
for i, k2 in zip(range(311, 319), ["rewardbench1", "rewardbench2", "rmbench_pair", "rmbench_point", "rubricbench", "ppe",
                                     "processbench", "prmbench"]):
    m(i, k2, "H")
m(319, "medhallu", "H"); m(320, "tmmlu", "H"); m(321, "gaokao", "H"); m(322, "jmedqa", "H")
m(323, "cohen_screening", "X"); m(324, "cohen_adhd", "C"); m(range(325, 330), "exams_misc", "C"); m(range(330, 335), "kobest", "C")
ctxrows = {335: "clinc150", 336: "banking77", 337: "sst5", 338: "imdb", 339: "hfblog_macro", 340: "ag_news", 341: "banking77",
           342: "sst2", 343: "enron", 344: "sst2", 345: "ag_news", 346: "banking77", 347: "imdb", 348: "banking77",
           349: "ag_news", 350: "sms_spam", 351: "hfblog_macro", 353: "clinc150", 354: "eclipse_severity",
           355: "banking77", 356: "clinc150", 357: "banking77", 358: "banking77", 359: "banking77", 360: "banking77",
           361: "banking77", 362: "banking77", 363: "clinc_certify", 364: "ag_news", 365: "massive_sv", 366: "tanaos",
           367: "darija", 368: "email_spam", 369: "arize_spam"}
for i, k2 in ctxrows.items():
    m(i, k2, "C")
m(352, "sst2", "2")
m(370, "obqa", "2"); m(371, "csqa", "2"); m(372, "hellaswag_vercel", "C"); m(373, "chaosnli", "X"); m(374, "phishnchips", "2")
m(375, "general_decisions", "C"); m(376, "llm_aggrefact", "C"); m(377, "tabfact", "C"); m(378, "nslkdd", "C")
m(379, "fake_jobs", "C"); m(380, "app_reviews", "C"); m(381, "cuad", "C"); m(382, "whowhen", "C"); m(383, "mind2web", "C")
m(384, "upworthy", "H"); m(385, "aita", "C"); m(386, "commonlit", "C"); m(387, "scifact", "H"); m(388, "nfcorpus", "H")
m(389, "scifact", "2"); m(390, "nfcorpus", "2"); m(391, "fiqa_rerank", "C"); m(392, "rerank_mean", "H"); m(393, "nevir", "H")
m(394, "label_pressure", "C"); m(395, "diag", "X"); m(396, "typed_decisions", "C")
m(397, "ag_news", "C"); m(398, "banking77", "C"); m(399, "sms_spam", "C"); m(400, "imdb", "C"); m(401, "banking77", "C")
m(range(402, 406), "tabular", "C"); m([406, 407], "cuad", "C"); m([408, 409], "injection_deepset_all", "2")
m([410, 411], "dblp_acm", "C"); m(412, "scifact_claims", "C"); m([413, 414], "scifact", "2"); m(415, "xquad_rerank", "G")
m(416, "scifact", "C"); m(417, "nfcorpus", "C"); m(418, "nanorteb", "G"); m(419, "mind2web", "C"); m(420, "phishing_small", "C")
m([421, 422], "banking77", "C"); m(423, "metamedqa", "G"); m(424, "pubmedqa", "2"); m(425, "diagnosisarena", "G")
m(426, "nejm", "C"); m([427, 428, 429, 431], "agent_trace", "C"); m(430, "mcphunt", "G"); m(432, "ricechem", "G")
m(433, "healthbench", "C"); m(434, "ellipse", "C"); m([435, 436, 438, 439, 440], "dialog_quality", "C")
m(437, "helpsteer2", "C"); m(441, "rewardbench1", "C"); m(442, "judgebench", "C"); m(443, "halueval_qa", "G")
m([444, 445], "halueval_misc", "C"); m([446, 447], "rmbench_pair", "2"); m(448, "rewardbench2", "C")
m(449, "cyberseceval_cwe", "G"); m(450, "anli", "2"); m(451, "diag", "X"); m(452, "typed_decisions", "C")
m(453, "clinc150", "C"); m(454, "css_big", "C"); m(455, "css_big", "C"); m(456, "css_lowjev", "C"); m(457, "css_big", "C")
m(458, "trec_fine", "G"); m(459, "clinc150", "C"); m(460, "massive_en", "C"); m(461, "contractnli", "2")
m(462, "banking77", "C"); m(463, "banking77", "S"); m(464, "clinc150", "C"); m(465, "clinc150", "S")
for i, k2 in zip(range(466, 471), ["triviaqa4", "popqa4", "simpleqa4", "mmlu_cf", "daily_oracle"]):
    m(i, k2, "G")
m(471, "truthfulqa", "C"); m(472, "cjbench", "C"); m(473, "wainject", "G"); m([474, 475], "agent_trace", "C")
m([476, 477], "diag", "X"); m([478, 479], "radiology", "C"); m(480, "rexerr", "G"); m(481, "dimabsa_va", "G")
m(482, "dimabsa_triplet", "G"); m(483, "speech_rescoring", "C"); m([484, 485, 486], "recsys", "C"); m([487, 488, 489], "har", "C")
m(490, "cesnet", "G"); m(491, "cesnet", "S"); m(492, "nslkdd", "G"); m(493, "diag", "X"); m(494, "video_quality", "C")
m(495, "injecagent", "C"); m(496, "mmlu", "2"); m(497, "jevbench_public", "C"); m(498, "diag", "X")
m(range(499, 506), "jevout", "C"); m(506, "sys1cal", "C"); m(507, "crt", "C"); m(508, "alfworld", "C"); m(509, "xdviolence", "C")
m(510, "diag", "X"); m(511, "sst2", "2"); m(512, "amazon_polarity", "G"); m(513, "kobbq", "C"); m([514, 515], "multihop_retrieval", "C")
m(516, "sst2", "C"); m(517, "trec_coarse", "C"); m(518, "trec_coarse", "S"); m([519, 520], "tabular", "C")
m(521, "cfpb", "G"); m(522, "cfpb", "2"); m(523, "ag_news", "2"); m(524, "dair_emotion", "2"); m(525, "banking77", "2")
m(526, "massive_scenario51", "G"); m(527, "typed_decisions", "C"); m(528, "kiki_agent", "C"); m(529, "gulf_arabic", "C")
m(530, "wanli", "C"); m(531, "csqa", "C"); m(532, "mmlu_cf", "C"); m(533, "race_h", "C"); m(534, "amazonqa", "C")
m([535, 536], "p4g", "C"); m(537, "craigslist", "C"); m([538, 539, 541], "banking77", "C"); m(540, "injection_deepset_all", "C")
m([542, 543], "trec_coarse", "C"); m([544, 545], "ag_news", "C"); m(546, "clinc150", "C"); m(547, "massive_en", "C")
m(548, "fin_topic", "C")
missing = [i for i in range(549) if i not in M]
assert not missing, missing
for i, (k2, _) in M.items():
    assert k2 in K, (i, k2)

# ------------------------------------------------------------------ specs per suite
SPECS = [
    (r"^Deusser", "deusser_exact@6bbdeb33", "spec-written (suite-reproduction-specs.md §2)"),
    (r"^Decision Index", "decision_index_0.2.1@87d4650b", "spec-written (§3); DI kit pipeline --engine http"),
    (r"^JevBench v1\.5\.5", "jevbench_v1.5.5", "maintainer run only; offline proxy = public-231 (§4)"),
    (r"^AbdelStark", "abdelstark_btzsc_pilot_v1@0d610cc5", "spec-written (§5); manifest sha256 ec064c52…"),
    (r"^Jevals", "jevals_0.1.0@21bb47b7", "spec-written (§6); reimplement, pin bytes by state_sha256"),
    (r"^DMB", "dmb_expanded@eabd88b0", "spec-written (§7); reimplement (no licence)"),
    (r"^typed-decisions", "typed_decisions_card@d0e2f0c4", "spec-written (§8); scorer pinned by Prior/Uniform rows"),
    (r"^TypeSafe workflow", "typesafe_workflowevals@0ac3b8ad", "spec-written (§9); patched model allowlist, disclosed"),
    (r"^chepyle", "chepyle_lexglue_systemone_v1", "adapter-todo (feasibility §9.7)"),
    (r"^Kev suites", "kev_suites", "pointer-only"),
    (r"^mbburabak", "mbburabak_safety", "adapter-todo (feasibility §9.7)"),
    (r"^goya4140", "goya_rm_eval", "adapter-todo (only if PPE pursued)"),
    (r"^jev-bench \(Praveenrajus", "jevbench_hf_praveenrajus_v0.1.1", "adapter-todo; exact items + train splits on HF"),
    (r"^thisisandreeeee", "thisisandreeeee", "adapter-todo"),
    (r"^(denser|hev/|anessbelbati)", "rerank_scripts", "adapter-todo"),
    (r"^ASEVlad", "asevlad_injection", "adapter-todo"),
    (r"^stperic", "stperic_medhallu", "adapter-todo"),
    (r"^elcronos", "elcronos_plain", "reimplement from README (no licence); suite-reproduction-specs §11"),
    (r"^zhuyansen", "zhuyansen_batch20", "reimplement (MIT); §11"),
    (r"^onlyoneaman", "onlyoneaman_cases", "exact (published case files); §11"),
    (r"^anisselbd", "decision_index_0.2.1@87d4650b", "use DI route (same items/questions)"),
    (r"^simonmesmith", "simonmesmith_b77", "S-bar reference (Jev given train data)"),
]


def spec_for(suite):
    for pat, sid, st in SPECS:
        if re.search(pat, suite):
            return sid, st
    slug = re.sub(r"[^a-z0-9]+", "_", suite.lower()).strip("_")[:48]
    return f"study:{slug}", "pointer-only (protocol in jev-published.md row notes)"


LOWER = re.compile(r"lower|brier|rmse|error rate|false-alarm|false-negative|\bkl\b|\bece\b|word error|attack success|\|p\(x\)", re.I)


def tier_of(r):
    mm = re.search(r"tier=(\w)", r["notes"] or "")
    return mm.group(1) if mm else None


def proto_of(r):
    mm = re.search(r"protocol=([\w-]+)", r["notes"] or "")
    p = mm.group(1) if mm else None
    return "uses-train-data" if p and p.startswith("uses-train") else p


def binom_bar(p, n, higher):
    if p is None or n in (None, 0) or not (0 < p < 1):
        return None
    mgn = 1.96 * math.sqrt(2 * p * (1 - p) / n)
    return round(p + mgn if higher else p - mgn, 4)


def verdicts(k):
    meta = K[k]
    if meta["feas"]:
        fr = FEASROWS[meta["feas"]]
        return codes(fr["S"]), codes(fr["Z"]), f"feasibility-targets.md §3 group {fr['group']}"
    if meta["v"]:
        s, z = meta["v"].split("/")
        cv = lambda x: None if x == "----" else dict(zip(SIZES, x))
        return cv(s), cv(z), "PLAN estimate [E] (benchmax/PLAN.md §7)"
    return None, None, "unassessed"


_miss = [v["feas"] for v in K.values() if v["feas"] and v["feas"] not in FEASROWS]; assert not _miss, _miss
GROUP_OF = {k: (FEASROWS[v["feas"]]["group"] if v["feas"] else None) for k, v in K.items()}

# S bars (Jev given train data) per key, from S rows
sbar_rows = defaultdict(list)
for i, (k2, role) in M.items():
    if role == "S":
        sbar_rows[k2].append(i)

targets = []
for i, r in enumerate(rows):
    k2, role = M[i]
    meta = K[k2]
    tier = tier_of(r)
    higher = not bool(LOWER.search(r["metric"] or ""))
    S, Z, vsrc = verdicts(k2)
    group = GROUP_OF.get(k2)
    if role == "G":
        group = "G"
    counted = role in ("H", "G") and not (meta["feas"] or "").endswith("(shown, not counted)") and \
        group not in ("D",)
    bz = bs = None
    bar_src = None
    note = None
    if role == "H" and meta["feas"]:
        bz, bs = bars(FEASROWS[meta["feas"]]["bar"])
        bar_src = "feasibility §3 (unpaired bar, §1.2)"
        fj = re.search(r"(\.\d+)", FEASROWS[meta["feas"]]["jev"].replace("*", ""))
        fj = float(fj.group(1)) if fj else None
        if k2 == "forecastbench":  # feasibility bar is in skill units; row metric is Brier
            bz = bs = round(0.25 * (1 - 0.318), 4)
            bar_src = "feasibility §3 skill bar .318 converted to Brier = 0.25*(1-skill)"
        elif fj is not None and bz is not None and r["jev_score"] is not None and abs(fj - r["jev_score"]) > 0.0015 \
                and r["jev_score"] < 1:
            base = max(fj, r["jev_score"]) if higher else min(fj, r["jev_score"])
            note = f"Jev value conflict: feasibility {fj} vs this row {r['jev_score']} (different DI files); bar = stricter value + same margin"
            d_z, d_s = bz - fj, (bs - fj) if bs is not None else None
            bz = round(base + d_z, 4)
            bs = round(base + d_s, 4) if d_s is not None else None
    if bz is None:
        bz = binom_bar(r["jev_score"], r["n"], higher)
        bar_src = "binomial 1.96*sqrt(2p(1-p)/n)" if bz is not None else "none (metric not a proportion / n unknown)"
    if bz is None and r["jev_score"] is not None and role != "S":
        bz = r["jev_score"]
        bar_src = "point estimate (no variance model for this metric/n); beat it, CI reported separately"
    if higher and bz is not None and r["jev_score"] is not None and r["jev_score"] <= 1 and bz > 1:
        bz = 1.0
        note = "ceiling: significance bar exceeds 1.0; only a perfect score clears it"
        bar_src = "capped at 1.0 (ceiling)"
    if bs is None:
        bs = bz
    sref = [j for j in sbar_rows.get(k2, []) if j != i]
    if role in ("H", "G") and bs is not None and not (meta["feas"] and "S" in FEASROWS[meta["feas"]]["bar"]):
        for j in sref:
            rj = rows[j]
            if (rj["metric"] or "").lower() == (r["metric"] or "").lower() and rj["n"] and r["n"] and rj["n"] >= 0.5 * r["n"]:
                bj = binom_bar(rj["jev_score"], rj["n"], higher)
                if bj is not None and (bj > bs if higher else bj < bs):
                    bs = bj
                    note = (note + "; " if note else "") + f"S bar raised to Jev-given-train-data row T{j:03d}"
    if role == "S":
        bs = binom_bar(r["jev_score"], r["n"], higher) if r["n"] else None
        bz = None
    raw_use = "none"
    if r.get("raw_outputs_url"):
        raw_use = "do-not-use (Jev Responses License §3.2)" if "zenodo.23039006" in r["raw_outputs_url"] else \
            "evaluation-only (never training/selection/calibration)"
    sid, sst = spec_for(r["suite"])
    zelig = "Z only" if meta["s"].startswith("Z only") else "eligible"
    if role == "S" or proto_of(r) == "uses-train-data":
        zelig = "n/a (Jev number uses train data -> compare S only)"
    targets.append(dict(
        id=f"T{i:03d}", row=i, suite=r["suite"], dataset=r["dataset"], hf_id=r["hf_id"], config=r["config"],
        split=r["split"], n=r["n"], metric=r["metric"], higher_is_better=higher, jev_score=r["jev_score"],
        jev_model=r["jev_model"], date=r["date"], source_url=r["source_url"], raw_outputs_url=r["raw_outputs_url"],
        raw_outputs_use=raw_use, tier=tier, jev_protocol=proto_of(r),
        benchmark_key=k2, category=meta["cat"], group=group,
        role={"H": "headline", "G": "headline", "S": "s_bar", "2": "secondary", "C": "context", "M": "composite",
              "X": "diagnostic"}[role],
        counted=bool(counted),
        bar_z=bz, bar_s=bs, bar_source=bar_src, s_bar_rows=[f"T{j:03d}" for j in sref],
        verdict={"S": S, "Z": Z, "source": vsrc},
        track={"Z": zelig, "S": meta["s"]},
        train_sources=meta["train"], z_contamination_flags=meta["zf"],
        spec_id=sid, spec_status=sst, note=note,
    ))

# ------------------------------------------------------------------ tallies
counted = [t for t in targets if t["counted"]]
assert len({t["benchmark_key"] for t in counted}) == len(counted), Counter(t["benchmark_key"] for t in counted).most_common(5)
tal = {}
for track in ("S", "Z"):
    for sz in SIZES:
        c = Counter((t["verdict"][track] or {}).get(sz, "n/a") for t in counted)
        tal[f"{track}-{sz}"] = dict(c)
grp = Counter(t["group"] for t in counted)
out = dict(
    schema="benchmax-targets/1",
    generated="2026-10-03",
    source=dict(path="bench/public/jev_published.json", sha256=SHA, rows=len(rows)),
    plan="docs/research/benchmax/PLAN.md",
    legend=dict(
        role="headline = counted benchmark (one per benchmark_key); s_bar = Jev given train data (S-track bar); "
             "secondary = same benchmark, other harness/metric; context = sample/small-n/one-off; composite; diagnostic = not a target",
        verdict="W likely win (clears bar), T toss-up, L unlikely; sizes = Ettin 68m/150m/400m/1b; [E] estimates",
        bar="score to beat for a significant win (unpaired); lower-is-better metrics have bar below Jev",
    ),
    counts=dict(rows=len(targets), counted=len(counted), by_group=dict(grp),
                by_role=dict(Counter(t["role"] for t in targets)), tallies_counted=tal),
    targets=targets,
)
new_targets = json.dumps(out, indent=1, ensure_ascii=False)
stale = []
if CHECK:
    if not os.path.exists(OUT) or open(OUT, encoding="utf-8").read() != new_targets:
        stale.append("bench/public/targets.json")
else:
    open(OUT, "w", encoding="utf-8").write(new_targets)
print(json.dumps(out["counts"], indent=1))

# ------------------------------------------------------------------ public_jev family -> registry + manifest
sys.path.insert(0, ROOT)
registry = importlib.import_module("jev_local.bench.registry")
gen = registry.generate_public_jev(out)
BEGIN, END = "# >>> GENERATED public_jev", "# <<< GENERATED <<<"


def py_dict(name, d, ann):
    items = [f"    {json.dumps(k)}: {json.dumps(v) if not isinstance(v, tuple) else repr(v)}," for k, v in d.items()]
    return f"{name}: {ann} = {{\n" + "\n".join(items) + "\n}" if items else f"{name}: {ann} = {{}}"


block = "\n".join([
    f"{BEGIN} (scripts/benchmax_build_targets.py --write-registry; do not edit by hand) >>>",
    f"PUBLIC_JEV_TARGETS_SHA256 = {json.dumps(gen['targets_sha256'])}",
    py_dict("PUBLIC_JEV_IDS", gen["ids"], "dict[str, str]"),
    py_dict("PUBLIC_JEV_REPOS", gen["repos"], "dict[str, str]"),
    py_dict("PUBLIC_JEV_KEYS", gen["keys"], "dict[str, str]"),
    "S_TRAIN_SPLIT_KEYS: frozenset[str] = frozenset({\n" + "\n".join(f"    {json.dumps(k)}," for k in gen["s_train_split_keys"]) + "\n})",
    py_dict("S_ALLOWED_IDS", gen["s_allowed_ids"], "dict[str, str]"),
    py_dict("S_MIXED_ELIGIBILITY_IDS", gen["s_mixed_eligibility_ids"], "dict[str, tuple[str, ...]]"),
    END,
])
src = open(REGISTRY_PY, encoding="utf-8").read()
i, j = src.index(BEGIN), src.index(END) + len(END)
new_src = src[:i] + block + src[j:]
registry_stale = registry.public_jev_tables() != {k: (v if k != "s_mixed_eligibility_ids" else {a: tuple(b) for a, b in v.items()})
                                                 for k, v in gen.items()}
if "--write-registry" in ARGS and new_src != src:
    open(REGISTRY_PY, "w", encoding="utf-8").write(new_src)
    print(f"registry: rewrote GENERATED block ({len(gen['ids'])} hf ids, {len(gen['repos'])} repo segments, "
          f"{len(gen['keys'])} keys, {len(gen['s_train_split_keys'])} S train-split keys, {len(gen['s_allowed_ids'])} S ids)")
    registry = importlib.reload(registry)
    registry_stale = False
elif registry_stale:
    stale.append("jev_local/bench/registry.py (run --write-registry)")
    print("registry: GENERATED block is STALE vs targets.json; run with --write-registry", file=sys.stderr)
else:
    print(f"registry: GENERATED block up to date ({len(gen['ids'])} hf ids, {len(gen['keys'])} keys)")

if not registry_stale:
    data = registry.manifest_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if CHECK:
        if not os.path.exists(EXCL) or open(EXCL, "rb").read() != data:
            stale.append("bench/public/exclusions.json")
    else:
        open(EXCL, "wb").write(data)
    print(f"exclusions: {EXCL} sha256 {sha} ({len(data)} bytes)")
if stale:
    print("STALE: " + ", ".join(stale), file=sys.stderr)
    sys.exit(1)
