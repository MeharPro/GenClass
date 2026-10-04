"""jevbench registry v1.1: dataset table invariants, the clean dev set, stage-1 exclusion rules on both tracks, the
generated public_jev family (vs bench/public/targets.json) and the exclusion manifest. Pure Python."""

import hashlib
import json
import os

import pytest

from jev_local.bench import registry as r

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = os.path.join(ROOT, "bench", "public", "targets.json")
EXCL = os.path.join(ROOT, "bench", "public", "exclusions.json")


# ------------------------------------------------------------------------------------------- tables

def test_counts_and_areas():
    assert len(r.TEST) == 25
    assert {d.area for d in r.TEST} == set(r.AREAS)
    assert sum(d.n_used for d in r.TEST) == 57609  # PLAN §1.1 total
    assert all(d.role == "dev" for d in r.DEV) and all(d.role == "ref" for d in r.REF)
    assert all(d.role == "retired_dev" for d in r.RETIRED_DEV)
    assert len(r.DEV_TASKSOURCE_SOURCES) == 9 and "strategy-qa" in r.RETIRED_DEV_TASKSOURCE_SOURCES
    assert r.REGISTRY_VERSION == "jevbench-v1.1"


def test_clean_dev_set_is_exactly_plan_2_3_1():
    # benchmax PLAN §2.3.1: 20 Newsgroups, MRPC, SciTail, CommitmentBank, TweetEval sentiment / offensive, ATIS, SNIPS;
    # SciFact claims validation stays; plus the held-out tasksource families.
    assert set(r.dev_keys()) == {"newsgroups20", "mrpc", "scitail", "cb", "tweeteval_sentiment", "tweeteval_offensive",
                                 "atis", "snips", "scifact", "tasksource_heldout"}
    assert set(r.retired_dev_keys()) == {"tweet_topic", "app_reviews", "hwu64", "toxigen", "prompt_injections"}
    assert r.BY_KEY["atis"].hf_id == "tuetschek/atis" and r.BY_KEY["scifact"].config == "claims"
    assert not set(r.dev_keys()) & set(r.retired_dev_keys())


@pytest.mark.skipif(not os.path.exists(TARGETS), reason="bench/public/targets.json not synced here")
def test_no_dev_set_is_a_public_jev_row():
    """PLAN §2.4(c): no dev set overlaps a Jev row (SNIPS is declared: a component of one LangWatch context mix)."""
    t = json.load(open(TARGETS))
    hf_ids = {h for row in t["targets"] for h in r.extract_hf_ids(row["hf_id"])}
    for d in r.DEV:
        if d.key in ("snips", "tasksource_heldout"):
            continue
        nid = r.normalize_source(d.hf_id)
        hits = [h for h in hf_ids if h == nid]
        if d.key == "scifact":
            assert hits == ["allenai/scifact"]  # scifact_claims context row (CompleteDotTech), BEIR test is separate
        elif d.key in ("tweeteval_sentiment", "tweeteval_offensive", "mrpc", "cb"):
            pass  # shared parent repos (tweet_eval, glue, super_glue); the configs themselves carry no Jev row
        else:
            assert not hits, (d.key, hits)
    # the retired dev sets are all public Jev rows
    for d in r.RETIRED_DEV:
        assert r.exclusion_reason(d.hf_id).startswith("public_jev:"), d.key
    for fam in r.DEV_TASKSOURCE_SOURCES:
        assert r.exclusion_reason(fam) == "dev:tasksource_heldout", fam
    assert r.exclusion_reason("strategy-qa") == "public_jev:strategyqa_closed"


# ------------------------------------------------------------------------------------------- Z track

@pytest.mark.parametrize("name", [
    # jevbench test
    "fancyzhx/ag_news", "ag_news", "sh0416/ag_news", "glue/sst2", "SetFit/sst5", "stanfordnlp/sst",
    "rotten_tomatoes", "cornell-movie-review-data/rotten_tomatoes", "yelp_polarity", "Yelp/yelp_review_full",
    "financial_phrasebank/sentences_allagree", "takala/financial_phrasebank", "zeroshot/twitter-financial-news-sentiment",
    "dair-ai/emotion", "emotion", "mteb/emotion", "SetFit/emotion", "tweet_eval/emotion", "cardiffnlp/tweet_eval",
    "SemEvalWorkshop/sem_eval_2018_task_1", "go_emotions/simplified", "banking77", "PolyAI/banking77",
    "clinc_oos/plus", "multilingual/massive", "mteb/amazon_massive_scenario", "AmazonScience/massive", "slurp",
    "super_glue/boolq", "boolq-natural-perturbations", "glue/rte", "super_glue/rte", "anli/a1", "facebook/anli",
    "paws/labeled_final", "multilingual/paws-x/de", "glue/stsb", "mteb/sts12-sts", "sts-companion",
    "lytang/LLM-AggreFact", "wice", "ragtruth", "climate_fever", "toxic-chat/toxicchat0124/toxicity",
    "mmathys/openai-moderation-api-evaluation", "LocalLLaMA/typed-decisions", "lex_glue/unfair_tos", "claudette",
    "HelpSteer2/helpfulness", "nvidia/HelpSteer3",
    # clean dev
    "SetFit/20_newsgroups", "glue/mrpc", "scitail/snli_format", "super_glue/cb", "tweet_eval/sentiment",
    "tweet_eval/offensive", "tuetschek/atis", "atis", "snips_built_in_intents", "benayas/snips", "scifact_entailment",
    "allenai/scifact", "qasc", "allenai/qasc", "proofwriter",
    # retired dev -> public_jev
    "cardiffnlp/tweet_topic_single", "app_reviews", "DeepPavlov/hwu64", "toxigen-data/annotated",
    "deepset/prompt-injections", "nlu++", "nlupp/folds_18_19",
    # siblings, Jev-labelled, external suites
    "dataset_train_nli", "MoritzLaurer/multilingual-NLI-26lang-2mil7", "btzsc/btzsc", "SargeDev/jev-distill-corpus-v3",
    "yuri_v3", "hotchpotch/bekko-system-one-dataset-v0", "autotrust/JEV-labels", "fstandhartinger/jevbench",
    "apolinario/decision-index", "Hanno-Labs/decision-bench", "jaredpalmer/kev-suites",
    # public_jev: former reserved rows, §1.5 patterns, §7 traps, PLAN §0.4 v2-overlap list
    "hellaswag", "winogrande/winogrande_xl", "cais/mmlu", "TIGER-Lab/MMLU-Pro", "lukaemon/bbh", "tasksource/bigbench",
    "daily_dialog", "silicone/dyda_da", "vast", "isarcasm", "nli4ct_semeval2024", "hover", "vincentkoc/hover-parquet",
    "ai2_arc/ARC-Easy/challenge", "allenai/ai2_arc", "commonsense_qa", "tau/commonsense_qa", "art", "allenai/art",
    "qiaojin/PubMedQA", "pubmed_qa", "Davlan/sib200", "facebook/belebele", "multilingual/xnli", "masakhane/afrixnli",
    "ceval/ceval-exam", "c-eval", "papluca/language-identification", "d4br4/agb-de", "mteb/summeval", "sms_spam",
    "ucirvine/sms_spam", "contract-nli/contractnli_a/seg", "cladder", "esci", "tasksource/esci", "humicroedit/subtask-2",
    "SetFit/enron_spam", "AreLit/PhishNChips", "yixuantt/FinEntity", "NEUDM/acos", "ParticleMedia/RAGTruth",
    "nvidia/When2Call", "jmhessel/newyorker_caption_contest", "earino/chaosnli", "chaos-mnli-ambiguity/votes",
    "imdb", "stanfordnlp/imdb", "counterfactually-augmented-imdb", "civil_comments/toxicity_share", "google/civil_comments",
    "ucberkeley-dlab/measuring-hate-speech", "mhs", "aegis2", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
    "ledgar", "lex_glue/ledgar", "lex_glue/case_hold", "lex_glue/scotus", "coastalcph/lex_glue", "mnli", "glue/mnli",
    "nyu-mll/multi_nli", "fever_nli", "pietrolesci/nli_fever", "copenlu/fever_gold_evidence", "WANLI", "alisawuffles/WANLI",
    "trec", "SetFit/TREC-QC", "CogComp/trec", "conll2003", "tner/conll2003", "squad_v2", "rajpurkar/squad",
    "multilingual/mtop", "mteb/mtop_intent", "dbpedia_14", "fancyzhx/dbpedia_14", "DeveloperOats/DBPedia_Classes",
    "amazon_polarity", "fancyzhx/amazon_polarity", "amazon_counterfactual/en", "SetFit/amazon_counterfactual",
    "ccdv/arxiv-classification", "strategy-qa", "KoalaAI/Text-Moderation", "nickmuchi/financial-classification",
    "sata-bench/sata-bench", "idavidrein/gpqa", "TheoremQA", "chatbot_arena_conversations",
    "lmarena-ai/PPE-Human-Preference-V1", "prm800k_dpo/step", "flan", "tasksource-instruct",
    # public_jev: generated hf ids, repo segments and key tokens
    "Praveenrajus/jev-bench", "jev-bench/train", "allenai/wildguardmix", "wildguardmix-cleaned/prompt_harm",
    "Paul/hatecheck", "allenai/reward-bench", "THU-KEG/RM-Bench", "Qwen/ProcessBench", "hitsmy/PRMBench_Preview",
    "UTAustin-AIHealth/MedHallu", "MedQA-USMLE-4-options-hf", "pminervini/HaluEval", "orionweller/NevIR", "BeIR/scifact",
    "BeIR/nfcorpus", "microsoft/ms_marco", "ms_marco", "google/xquad", "mangopy/ToolRet-Queries",
    "gorilla-llm/Berkeley-Function-Calling-Leaderboard", "withmartian/routerbench", "google-deepmind/searchless_chess",
    "openai/gsm8k", "allenai/openbookqa", "openbookqa", "ehovy/race", "race/high", "race-c", "lucasmccabe/logiqa",
    "logiqa", "truthfulqa/truthful_qa", "wenhu/tab_fact", "theatticusproject/cuad", "osunlp/Mind2Web",
    "hotpotqa/hotpot_qa", "determined-ai/consumer_complaints_medium", "cfpb", "FLUTE", "implicit-hate-stg1",
    "persuasion", "glue/qqp", "glue/cola", "cesnet", "nslkdd", "upworthy", "tmmluplus", "ikala/tmmluplus",
    "xstest", "jailbreak-classification", "gretelai/gretel-pii-masking-en-v1", "CyberSecEval", "ToMBench",
])
def test_excluded_z(name):
    assert r.is_excluded(name), name


@pytest.mark.parametrize("name", [
    "tasksource/tasksource-jev-typed-decisions", "tasksource/procedural-typed-decisions", "procedural_typed_decisions",
    "procedural-typed-decisions/taxonomy_routing", "tasksource/synthetic-typed-decisions", "ZefanCai/Open-Jev",
    "MoritzLaurer/synthetic_zeroshot_mixtral_v0.1", "tasksource/zero-shot-label-nli", "lingnli", "vitaminc",
    "tals/vitaminc", "doc-nli", "tasksource/doc-nli", "snli", "stanfordnlp/snli", "sick/relatedness", "emo/emo2019",
    "SemEvalWorkshop/emo", "crowdflower/text_emotion", "piqa", "ybisk/piqa", "allenai/social_i_qa", "allenai/sciq",
    "swag/regular", "super_glue/copa", "super_glue/wic", "super_glue/multirc", "multilingual/wili_2018",
    "multilingual/offenseval_2020/ar", "mteb/amazon_reviews_multi", "multilingual/amazon_reviews_multi/all_languages",
    "jigsaw_toxicity", "toxic_conversations", "hate_speech_offensive", "hate_speech18", "dynahate", "ethos/multilabel",
    "openbmb/UltraFeedback", "glaiveai/glaive-function-calling-v2", "Team-ACE/ToolACE", "ucirvine/reuters21578",
    "github:Helsinki-NLP/XED", "Estwld/empathetic_dialogues_llm", "medmcqa", "wikimedqa/medwiki", "hh-rlhf/harmless-base",
    "oasst2_pairwise_rlhf_reward", "lsat-rc", "arct", "winowhy", "sem_eval_2010_task_8", "babi_nli/path-finding",
    "sorted", "assistant", "converter", "dynasent/r1_votes", "solid_state", "satisfaction", "state_of_the_art",
    "feverous", "programmatic:jev_local.data.v2.extractive", "cu", "records", "s0_route_catalog_large",
    "pragmeval/persuasiveness-strength", "seahorse_summarization_evaluation", "code_x_glue_cc_defect_detection",
])
def test_kept_z(name):
    assert not r.is_excluded(name), (name, r.exclusion_reason(name))


def test_reasons_are_categorised():
    assert r.exclusion_reason("glue/rte") == "test:rte"
    assert r.exclusion_reason("WANLI") == "public_jev:wanli"  # v2-overlap list; "anli" inside "wanli" still no test hit
    assert r.exclusion_reason("scitail/snli_format") == "dev:scitail"
    assert r.exclusion_reason("tuetschek/atis") == "dev:atis"
    assert r.exclusion_reason("dataset_train_nli") == "sibling:laurer_train_nli"
    assert r.exclusion_reason("SargeDev/jev-distill-corpus-v3") == "jev_labelled"
    assert r.exclusion_reason("qasc") == "dev:tasksource_heldout"
    assert r.exclusion_reason("fstandhartinger/jevbench").startswith("public_jev:jevbench")  # id table before reserved
    assert r.exclusion_reason("TIGER-Lab/MMLU-Pro") == "public_jev:mmlu_pro" and r.exclusion_reason("tmmluplus") == "public_jev:tmmlu"
    assert r.exclusion_reason("mmlu/abstract_algebra") == "public_jev:mmlu"
    assert r.exclusion_reason("classifier-benchmark").startswith("reserved:")
    # former dev / reserved rows now carry their targets.json benchmark key
    assert r.exclusion_reason("DeepPavlov/hwu64") == "public_jev:hwu64"
    assert r.exclusion_reason("hellaswag") == "public_jev:hellaswag"
    assert r.exclusion_reason("allenai/scifact") == "dev:scifact" and r.exclusion_reason("BeIR/scifact") == "public_jev:scifact"
    assert r.exclusion_reason("nvidia/Aegis-AI-Content-Safety-Dataset-2.0") == "public_jev:aegis2_prompt"
    assert r.exclusion_reason("coastalcph/lex_glue") == "public_jev:lexglue_mean"
    assert r.exclusion_reason("lex_glue/case_hold") == "public_jev:casehold"
    assert r.exclusion_reason("KoalaAI/Text-Moderation") == "public_jev:openai_mod"
    assert r.exclusion_reason("Praveenrajus/jev-bench") == "public_jev:jevbench_hf_macro"
    assert r.exclusion_reason("race/high") == "public_jev:race_h"
    assert r.exclusion_reason("owner/ds:config") is None
    assert r.is_excluded_any("tasksource/tasksource-jev-typed-decisions", "glue/sst2")
    assert not r.is_excluded(None) and not r.is_excluded("")
    with pytest.raises(ValueError):
        r.exclusion_reason("ag_news", track="X")


# ------------------------------------------------------------------------------------------- S track

@pytest.mark.parametrize("name, allow", [
    ("fancyzhx/ag_news", "s_allow:id:fancyzhx/ag_news->ag_news"), ("ag_news", "s_allow:key:ag_news"),
    ("glue/sst2", "s_allow:key:sst2"), ("SetFit/sst5", "s_allow:id:setfit/sst5->sst5"), ("rotten_tomatoes", "s_allow:key:rotten_tomatoes"),
    ("microsoft/ms_marco", "s_allow:rule:msmarco->rerank_mean"), ("MedQA-USMLE-4-options-hf", "s_allow:rule:medqa->metamedqa"),
    ("mmlu/abstract_algebra", "s_allow:key:mmlu"),
    ("PolyAI/banking77", "s_allow:id:polyai/banking77->banking77"), ("clinc_oos/plus", "s_allow:key:clinc150"),
    ("facebook/anli", "s_allow:id:facebook/anli->anli"), ("lex_glue/ledgar", "s_allow:key:ledgar"),
    ("coastalcph/lex_glue", "s_allow:id:coastalcph/lex_glue->unfair_tos"), ("tweet_eval/emotion", "s_allow:key:tweeteval_emotion"),
    ("cardiffnlp/tweet_eval/emotion", "s_allow:id:cardiffnlp/tweet_eval->tweeteval_emotion"),
    ("LocalLLaMA/typed-decisions", "s_allow:id:localllama/typed-decisions->typed_decisions"),
    ("allenai/scifact", "s_allow:id:allenai/scifact->scifact"), ("scifact_entailment", "s_allow:key:scifact"),
    ("DeepPavlov/hwu64", "s_allow:id:deeppavlov/hwu64->hwu64"),
    ("toxigen/toxigen-data", "s_allow:id:toxigen/toxigen-data->toxigen"), ("deepset/prompt-injections", "s_allow:id:deepset/prompt-injections->prompt_injections"),
    ("imdb", "s_allow:key:imdb"), ("google/civil_comments", "s_allow:id:google/civil_comments->civil_comments"),
    ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "s_allow:id:nvidia/aegis-ai-content-safety-dataset-2.0->aegis2_prompt"),
    ("mnli", "s_allow:key:mnli"), ("nyu-mll/multi_nli", "s_allow:key:mnli"), ("copenlu/fever_gold_evidence", "s_allow:id:copenlu/fever_gold_evidence->fever"),
    ("CogComp/trec", "s_allow:id:cogcomp/trec->trec_fine"), ("tau/commonsense_qa", "s_allow:id:tau/commonsense_qa->csqa"),
    ("allenai/art", "s_allow:id:allenai/art->alphanli"), ("Rowan/hellaswag", "s_allow:id:rowan/hellaswag->hellaswag"),
    ("cais/mmlu", "s_allow:id:cais/mmlu->mmlu"), ("tasksource/esci", "s_allow:id:tasksource/esci->esci"),
    ("NEUDM/acos", "s_allow:id:neudm/acos->acos"), ("nvidia/When2Call", "s_allow:id:nvidia/when2call->when2call"),
    ("jev-bench/train", "s_allow:key:jevbench_hf_macro"), ("qiaojin/PubMedQA/pqa_artificial", "s_allow:id:qiaojin/pubmedqa->pubmedqa"),
    ("facebook/xnli", "s_allow:id:facebook/xnli->afrixnli"), ("strategy-qa", "s_allow:key:strategyqa_closed"),
    ("nlupp/folds_0_17", "s_allow:key:nlupp"), ("cardiffnlp/tweet_topic_single", "s_allow:id:cardiffnlp/tweet_topic_single->tweet_topic"),
])
def test_s_track_waives_train_split_benchmarks(name, allow):
    assert r.is_excluded(name, "Z"), name
    assert not r.is_excluded(name, track="S"), (name, r.exclusion_reason(name, "S"))
    assert r.s_allow_reason(name) == allow


@pytest.mark.parametrize("name, prefix", [
    # never waived
    ("SargeDev/jev-distill-corpus-v3", "jev_labelled"), ("fstandhartinger/jevbench", "s_deny:eval_rereleases (public_jev:jevbench"),
    ("classifier-benchmark", "reserved:"), ("apolinario/decision-index", "s_deny:eval_rereleases (public_jev:di_index"),
    ("BeIR/scifact", "s_deny:beir_scifact_queries"),
    # clean dev sets (selection only), including the config-qualified tweet_eval names under an S-allowed parent
    ("glue/mrpc", "dev:mrpc"), ("scitail/snli_format", "dev:scitail"), ("super_glue/cb", "dev:cb"),
    ("tweet_eval/sentiment", "dev:tweeteval_sentiment"), ("tweet_eval/offensive", "dev:tweeteval_offensive"),
    ("tuetschek/atis", "dev:atis"), ("benayas/snips", "dev:snips"), ("SetFit/20_newsgroups", "dev:newsgroups20"),
    ("qasc", "dev:tasksource_heldout"),
    # S deny-list (§7 traps, EXCLUDE/NEVER clauses)
    ("nickmuchi/financial-classification", "s_deny:fpb_rereleases"), ("dair-ai/emotion/unsplit", "s_deny:emotion_unsplit"),
    ("KoalaAI/Text-Moderation", "s_deny:text_moderation"), ("cladder", "s_deny:cladder_release"),
    ("sata-bench/sata-bench", "s_deny:sata_bench"), ("yixuantt/FinEntity", "s_deny:finentity"),
    ("qiaojin/PubMedQA/pqa_labeled", "s_deny:pqa_labeled"), ("sms_spam", "s_deny:sms_spam"),
    ("TheoremQA", "s_deny:mmlu_pro_parents"), ("idavidrein/gpqa", "s_deny:gpqa_main_extended"),
    ("chatbot_arena_conversations", "s_deny:chatbot_arena"), ("dataset_train_nli", "s_deny:aggregators"),
    ("flan", "s_deny:aggregators"), ("btzsc/btzsc", "s_deny:eval_rereleases"),
    ("Praveenrajus/jev-bench/test", "s_deny:jev_bench_test_configs"), ("earino/chaosnli", "s_deny:chaosnli_release"),
    ("ucberkeley-dlab/measuring-hate-speech", "s_deny:mhs_single_split"),
    # not on the allow-list: eval-only, related-data-only, context-only or "none" benchmarks
    ("lytang/LLM-AggreFact", "test:llm_aggrefact"), ("Paul/hatecheck", "public_jev:hatecheck"),
    ("microsoft/MMLU-CF", "public_jev:mmlu_cf"), ("tmmluplus", "public_jev:tmmlu"),
    ("mmathys/openai-moderation-api-evaluation", "test:openai_moderation"), ("mteb/summeval", "public_jev:summeval"),
    ("allenai/reward-bench", "public_jev:rewardbench1"), ("sealuzh/app_reviews", "public_jev:app_reviews"),
    ("ehovy/race", "public_jev:race_h"), ("logiqa", "public_jev:logiqa"), ("multilingual/xnli", "public_jev:xnli"),
    ("conll2003", "public_jev:conll_typing"), ("squad_v2", "public_jev:squad_select"), ("WANLI", "public_jev:wanli"),
    ("community-datasets/yahoo_answers_topics", "test:yahoo_topics"),
    ("tdiggelm/climate_fever", "test:climate_fever"), ("glue/rte", "test:rte"), ("TIGER-Lab/MMLU-Pro", "public_jev:mmlu_pro"),
])
def test_s_track_keeps_excluded(name, prefix):
    s = r.exclusion_reason(name, track="S")
    assert s and s.startswith(prefix), (name, s)
    assert r.is_excluded_any("something_clean", name, track="S")


def test_s_track_does_not_exclude_clean_sources():
    for n in ("tasksource/zero-shot-label-nli", "lingnli", "ZefanCai/Open-Jev", "piqa", "something_clean"):
        assert not r.is_excluded(n, "S") and not r.is_excluded(n, "Z"), n
    assert r.exclusion_reasons("piqa", "imdb", "glue/mrpc", track="S") == {"glue/mrpc": "dev:mrpc"}
    assert r.exclusion_reasons("piqa", "imdb", "glue/mrpc") == {"imdb": "public_jev:imdb", "glue/mrpc": "dev:mrpc"}


# ------------------------------------------------------------------------------------------- generated tables

def test_extract_hf_ids():
    assert r.extract_hf_ids("clinc/clinc_oos + mteb/amazon_massive_intent + benayas/snips") == \
        ["clinc/clinc_oos", "mteb/amazon_massive_intent", "benayas/snips"]
    assert r.extract_hf_ids("clinc/clinc_oos @155b9c7") == ["clinc/clinc_oos"]
    assert r.extract_hf_ids("PolyAI-LDN/task-specific-datasets banking_data/test.csv (= PolyAI/banking77)") == \
        ["polyai-ldn/task-specific-datasets", "polyai/banking77"]
    assert r.extract_hf_ids("generated (decision_index/suite/build/home_appliance.py, seed 1)") == []
    assert r.extract_hf_ids("public email corpora (Enron/Ling-Spam/phishing)") == []
    assert r.extract_hf_ids("related: MiniCheck C2D/D2C, ANLI") == []
    assert r.extract_hf_ids("UCI / scikit-learn dataset") == [] and r.extract_hf_ids(None) == []
    assert r.extract_hf_ids("mixed public test/val") == []


@pytest.mark.skipif(not os.path.exists(TARGETS), reason="bench/public/targets.json not synced here")
def test_generated_block_matches_targets_json():
    """The frozen public_jev tables in registry.py are exactly what targets.json generates (else run
    scripts/benchmax_build_targets.py --write-registry)."""
    t = json.load(open(TARGETS))
    assert t["schema"] == "benchmax-targets/1" and len(t["targets"]) == 549
    gen = r.generate_public_jev(t)
    frozen = r.public_jev_tables()
    assert gen["targets_sha256"] == frozen["targets_sha256"] == t["source"]["sha256"]
    for k in gen:
        assert gen[k] == frozen[k], k
    # every hf id and every benchmark key of targets.json is excluded on Z
    for row in t["targets"]:
        for hid in r.extract_hf_ids(row["hf_id"]):
            assert r.is_excluded(hid), (row["id"], hid)
        k = row["benchmark_key"]
        if k not in r.PUBLIC_JEV_KEY_STOP:
            assert r.is_excluded(k), (row["id"], k)
    # S allow-list = train-split benchmarks + their train_sources ids; S-eligible rows are waivable by key
    for row in t["targets"]:
        if (row["track"]["S"] or "").startswith("train-split"):
            assert row["benchmark_key"] in r.S_TRAIN_SPLIT_KEYS
    assert "llm_aggrefact" not in r.S_TRAIN_SPLIT_KEYS and "hatecheck" not in r.S_TRAIN_SPLIT_KEYS
    assert "fancyzhx/ag_news" in r.S_ALLOWED_IDS and "lytang/llm-aggrefact" not in r.S_ALLOWED_IDS


def test_generate_public_jev_attribution_priority():
    rows = [
        {"benchmark_key": "ctx_a", "role": "context", "counted": False, "hf_id": "o/shared", "track": {"S": "see source study"}},
        {"benchmark_key": "head_b", "role": "headline", "counted": True, "hf_id": "o/shared", "track": {"S": "train-split"},
         "train_sources": ["o/shared train (10)", "EXCLUDE x/y (contains test)"]},
        {"benchmark_key": "diag", "role": "diagnostic", "counted": False, "hf_id": "o/other", "track": {"S": "n/a"}},
    ]
    gen = r.generate_public_jev({"targets": rows, "source": {"sha256": "abc"}})
    assert gen["ids"] == {"o/other": "diag", "o/shared": "head_b"} and gen["repos"]["shared"] == "head_b"
    assert gen["s_train_split_keys"] == ["head_b"] and gen["s_allowed_ids"] == {"o/shared": "head_b"}
    assert gen["s_mixed_eligibility_ids"] == {"o/shared": ("ctx_a", "head_b")}
    assert "diag" not in gen["keys"] and gen["keys"] == {"ctx_a": "ctx_a", "head_b": "head_b"}


# ------------------------------------------------------------------------------------------- manifest

def test_manifest_shape_and_hash():
    m = r.exclusion_manifest()
    assert m["schema"] == "benchmax-exclusions/1" and m["registry_version"] == r.REGISTRY_VERSION
    assert m["generated_from"]["sha256"] == r.PUBLIC_JEV_TARGETS_SHA256
    assert {d["key"] for d in m["jevbench"]["dev"]} == set(r.dev_keys())
    assert {d["key"] for d in m["jevbench"]["retired_dev"]} == set(r.retired_dev_keys())
    assert set(m["rules"]) == {"test", "dev", "sibling", "jev_labelled", "reserved", "public_jev"}
    assert m["public_jev"]["hf_ids"] == r.PUBLIC_JEV_IDS and m["s_track"]["deny_rules"] == r.S_DENY_RULES
    data = r.manifest_bytes()
    assert json.loads(data) == m
    assert r.manifest_sha256() == hashlib.sha256(data).hexdigest()


@pytest.mark.skipif(not os.path.exists(EXCL), reason="bench/public/exclusions.json not synced here")
def test_manifest_file_is_current():
    assert open(EXCL, "rb").read() == r.manifest_bytes(), "run scripts/benchmax_build_targets.py"


def test_dev_family_pick_is_reproducible():
    # A toy audit: the procedure must be deterministic and respect the eligibility filters.
    rows = {f"fam{i}": 3000 + i for i in range(30)} | {"glue/mnli": 5000, "ag_news": 5000, "tiny": 10}
    a, b = r.pick_dev_families(rows), r.pick_dev_families(dict(reversed(list(rows.items()))))
    assert a == b and len(a) == 10
    assert not {"glue", "ag_news", "tiny"} & set(a)
