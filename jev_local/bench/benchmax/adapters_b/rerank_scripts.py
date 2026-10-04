"""The three rerank studies (targets T387-T393): denser-org/rerank-bench-jev, hev/reranker, anessbelbati/jev-rerank-bench.

Each repo's own candidate lists and nDCG@10 code are reproduced; results are reported PER REPO (SciFact moves
.751-.793 with the candidate set alone).

denser (Apache-2.0, @41fe2570): BM25 top-100 lists shipped in `cache/candidates_{ds}_test_k100.json` (sha256
  pinned below); one request per query, state {"query": q}, questions `p{i}` = noul with OBJECT instructions
  {"question": "Is this passage relevant to the query?", "passage": "<title> <text>"} and the generic
  true/false criteria (S2_packed). Score = P(true); stable sort descending (ties keep BM25 order); nDCG@10 with
  gains 2^rel-1 over the full qrels (metrics.py); all 300 / 323 queries.
hev (Apache-2.0, @1eb47266): BM25 top-30 shortlists built with rank_bm25 "stopworded" (NOT shipped: rebuilt here
  with a BM25Okapi re-implementation, so hev rows are INDICATIVE, flagged `candidates_rebuilt`); state
  {"query": q, "documents": {"D00": {"title", "text"}, ...}}, one noul per document from prompt.yaml; `pair` mode
  (T389, one document per request) and `batch` mode (30 per request); nDCG@10 via ir_measures (linear gains).
anessbelbati (MIT, @fecba75a): BM25 top-30 lists shipped in `candidates/{ds}.jsonl` (sha256 pinned); passages cut
  to 2,000 characters; `jev-score-batch` (T392): state {"query", "passages": {"p01"..}}, one 4-level score question
  per passage, score = expected level / (K-1) read from legend + probabilities; nDCG@10 with LINEAR gains over
  queries whose top-30 holds a relevant passage; headline = mean of the 8 English dataset means (also the
  per-question mean); NevIR (T393): paired accuracy, strict `>` (a tie is wrong), same score-batch question.
Corpora: BEIR via HF `BeIR/<ds>` (+ `-qrels`), MTEB BrightRetrieval / CodeSearchNetRetrieval, `orionweller/NevIR`
(revisions recorded at prepare). Text joins: denser `f"{title} {text}".strip()`; anessbelbati
`f"{title}\\n{text}".strip()` then 2,000 chars; hev {"title", "text"} fields.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from jev_local.bench.benchmax.adapters_b.common import (VENDOR, AdapterB, Item, answers_of, by_task, coverage,
                                                        hf_download, hf_list_files, hf_resolve_revision, mrr_at_k,
                                                        ndcg_at_k, noul_p, parquet_rows, read_jsonl, repos_dir,
                                                        sha256_file, take)

SPEC_ID = "rerank_scripts"
TARGETS = ("T387", "T388", "T389", "T390", "T391", "T392", "T393")
UPSTREAM = {
    "denser": {"repo": "denser-org/rerank-bench-jev", "commit": "41fe2570d3d0391c65895431dc2c3e40f329054c", "licence": "Apache-2.0"},
    "hev": {"repo": "hev/reranker", "commit": "1eb47266270b32b2a3667f9fb89646378ca9c9d6", "licence": "Apache-2.0"},
    "anessbelbati": {"repo": "anessbelbati/jev-rerank-bench", "commit": "fecba75a443c580a8979ce2587e2d452d4d8e511", "licence": "MIT"},
}
DENSER_CANDIDATES = {"scifact": "862978ce8d72959cd1c2a1750177216b49bf44604265138c9e257a4212c95b29",
                     "nfcorpus": "67fb727b52863d0b1a41f90542df59512e2ef20ba9f06671d3b61713e01ae5a3"}
ANESS_CANDIDATES = {
    "scifact": "bab1021b55f84fd1efd37028d7e4e860986a2b44b2fad574ece8faf734576ca6",
    "fiqa": "4a08e0a3e68555711860ead7609b04a40825c960227b7ebec09fa1b2d08c6275",
    "nq": "678883703cc228659a93b78e05204af51669bb96e2b315c2c056428ff1a68f56",
    "nfcorpus": "91b830981406032ce9b06ffd1b1d00a36109f916f7c984b62a68b2055dad0f7b",
    "trec-covid": "439588cf76d4b9662d038e56be3c4725f1c12826a1ab5ac5101f1741a7a77d82",
    "bright-biology": "31000c7a23e5a1a44f281524a7c9837f512c48b364e03fe2820f7ab3f4a4e622",
    "bright-economics": "01a024a9f2c82eac392ae2ed2b4a95531a3a100bbbae1f77695518426e61c88a",
    "csn-python": "0ea910e26a0d3289eb3c6e1ffde0d210dc42eaf59712bf379c6c9c37c544fdd8",
    "nevir": "0c3acd0deda80e2cd9c31ed3b50e9fcef7380caa83f5b6cff313d7e441b92e07",
}
ANESS_ENGLISH = ("scifact", "fiqa", "nq", "nfcorpus", "trec-covid", "bright-biology", "bright-economics", "csn-python")
ANESS_MAX_CHARS = 2000
ANESS_RUBRIC = ["The passage is off-topic for the query.",
                "The passage is on a related topic but does not supply what the query asks for.",
                "The passage partly supplies the information needed to answer or verify the query.",
                "The passage fully supplies the information needed to answer or verify the query."]
DENSER_RUBRIC = {"type": "noul", "instructions": "Is this passage relevant to the query?",
                 "criteria": {"true": "The passage contains information that helps answer the query.",
                              "false": "The passage does not help answer the query."}}
HEV_TOP_K, HEV_MAX_DOCS = 30, 30
MTEB = {"bright-biology": ("mteb/BrightRetrieval", "biology"), "bright-economics": ("mteb/BrightRetrieval", "economics"),
        "csn-python": ("mteb/CodeSearchNetRetrieval", "python")}
BEIR_SETS = ("scifact", "nfcorpus", "fiqa", "nq", "trec-covid")
TASKS = ("denser_scifact", "denser_nfcorpus", "hev_scifact_pair", "hev_scifact_batch", "hev_nfcorpus_pair",
         "hev_nfcorpus_batch", "hev_fiqa_pair", "hev_fiqa_batch") + tuple(f"aness_{d}" for d in ANESS_ENGLISH) + ("aness_nevir",)
TARGET_OF_TASK = {"denser_scifact": "T387", "denser_nfcorpus": "T388", "hev_scifact_pair": "T389", "hev_nfcorpus_pair": "T390",
                  "hev_fiqa_pair": "T391", "aness_nevir": "T393"}
LUCENE_STOP = set("a an and are as at be but by for if in into is it no not of on or such that the their then there "
                  "these they this to was will with".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def _hev_prompt() -> dict[str, Any]:
    return json.loads((VENDOR / "hev" / "prompt.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------- request builders


def denser_request(query: str, passages: Sequence[str]) -> dict[str, Any]:
    questions = {}
    for i, passage in enumerate(passages):
        q = dict(DENSER_RUBRIC)
        q["instructions"] = {"question": DENSER_RUBRIC["instructions"], "passage": passage}
        questions[f"p{i}"] = q
    return {"state": {"query": query}, "questions": questions}


def denser_passage(doc: dict[str, Any]) -> str:
    return f"{doc.get('title', '') or ''} {doc.get('text', '') or ''}".strip()


def hev_request(query: str, docs: Sequence[dict[str, Any]], offset: int = 0) -> dict[str, Any]:
    p = _hev_prompt()
    ids = [f"D{offset + j:02d}" for j in range(len(docs))]
    state = {"query": query, "documents": {k: {kk: d[kk] for kk in ("title", "text") if kk in d} for k, d in zip(ids, docs)}}
    questions = {k: {"type": "noul", "instructions": p["question"].format(id=k), "criteria": p["criteria"]} for k in ids}
    return {"state": state, "questions": questions}


def aness_doc(row: dict[str, Any]) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title}\n{text}".strip() if title else text


def aness_truncate(text: str) -> str:
    return text if len(text) <= ANESS_MAX_CHARS else text[:ANESS_MAX_CHARS]


def _pid(i: int) -> str:
    return f"p{i + 1:02d}"


def aness_score_batch_request(query: str, docs: Sequence[str]) -> dict[str, Any]:
    ids = [_pid(i) for i in range(len(docs))]
    state = {"query": query, "passages": dict(zip(ids, docs))}
    questions = {pid: {"type": "score", "instructions": f"How well does passage {pid} supply the information needed to answer or verify the query?",
                       "criteria": ANESS_RUBRIC} for pid in ids}
    return {"state": state, "questions": questions}


def aness_expected_level(answer: dict[str, Any] | None) -> float | None:
    """anessbelbati `_expected_level`: legend keys ordered by RUBRIC index, Σ p·idx / (K−1); else `score`."""
    if not answer:
        return None
    legend = answer.get("legend") or {}
    probs = answer.get("probabilities") or {}
    if legend and probs:
        keys = sorted(legend, key=lambda k: ANESS_RUBRIC.index(legend[k]) if legend[k] in ANESS_RUBRIC else 0)
        return sum(float(probs.get(k, 0.0)) * idx for idx, k in enumerate(keys)) / (len(keys) - 1)
    sc = answer.get("score")
    return float(sc) if isinstance(sc, (int, float)) else None


def order_by_score(dids: Sequence[str], scores: Sequence[float]) -> list[str]:
    idx = sorted(range(len(dids)), key=lambda i: -scores[i])  # stable: ties keep incoming order
    return [dids[i] for i in idx]


# ---------------------------------------------------------------------------------------------- BM25 (hev rebuild)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in LUCENE_STOP and len(t) > 1]


class BM25Okapi:
    """rank_bm25.BM25Okapi defaults (k1=1.5, b=0.75, epsilon=0.25), pure Python."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_len) / len(corpus_tokens) if corpus_tokens else 0.0
        self.tf = [Counter(d) for d in corpus_tokens]
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(corpus_tokens)
        idf = {w: math.log(n - f + 0.5) - math.log(f + 0.5) for w, f in df.items()}
        neg = [v for v in idf.values() if v < 0]
        avg = sum(idf.values()) / len(idf) if idf else 0.0
        self.idf = {w: (epsilon * avg if v < 0 else v) for w, v in idf.items()}

    def get_scores(self, query: Sequence[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.doc_len):
            s = 0.0
            for q in query:
                f = tf.get(q)
                if not f:
                    continue
                s += self.idf.get(q, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


# ---------------------------------------------------------------------------------------------- adapter


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test")  # validation = BEIR train/dev queries for self-checks (SciFact train, NevIR validation)
    TASKS = TASKS

    # --- BEIR / MTEB loading (HF parquet)
    def _rev(self, repo: str) -> str:
        cache = self.raw_dir("rerank") / "revisions.json"
        revs = json.loads(cache.read_text()) if cache.exists() else {}
        if repo not in revs:
            revs[repo] = hf_resolve_revision(repo)
            cache.write_text(json.dumps(revs, indent=1))
        return revs[repo]

    def _folder_rows(self, repo: str, folder: str, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
        rev = self._rev(repo)
        rows: list[dict[str, Any]] = []
        for f in hf_list_files(repo, rev):
            if f.startswith(folder + "/") and f.endswith(".parquet"):
                rows.extend(parquet_rows(hf_download(repo, f, rev, work=self.work), columns))
        return rows

    def _beir(self, name: str, split: str = "test") -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, int]]]:
        corpus = {str(r["_id"]): {"title": r.get("title") or "", "text": r.get("text") or ""}
                  for r in self._folder_rows(f"BeIR/{name}", "corpus", ["_id", "title", "text"])}
        queries_all = {str(r["_id"]): r["text"] for r in self._folder_rows(f"BeIR/{name}", "queries", ["_id", "text"])}
        qrels: dict[str, dict[str, int]] = {}
        qrel_repo = f"BeIR/{name}-qrels"
        rev = self._rev(qrel_repo)
        fname = f"{split}.tsv"
        if fname not in hf_list_files(qrel_repo, rev):
            raise FileNotFoundError(f"{qrel_repo} has no {fname}")
        with hf_download(qrel_repo, fname, rev, work=self.work).open(encoding="utf-8") as f:
            header = next(f)
            if not header.lower().startswith("query-id"):
                raise ValueError(f"unexpected qrels header {header!r}")
            for line in f:
                qid, did, score = line.rstrip("\n").split("\t")
                if int(float(score)) > 0:
                    qrels.setdefault(qid, {})[did] = int(float(score))
        queries = {q: queries_all[q] for q in qrels if q in queries_all}
        return corpus, queries, qrels

    def _mteb(self, name: str) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
        repo, sub = MTEB[name]
        corpus = {str(r.get("_id", r.get("id"))): aness_doc(r) for r in self._folder_rows(repo, f"{sub}-corpus")}
        queries = {str(r.get("_id", r.get("id"))): r["text"] for r in self._folder_rows(repo, f"{sub}-queries")}
        qrels: dict[str, dict[str, int]] = {}
        for r in self._folder_rows(repo, f"{sub}-qrels"):
            g = int(float(r["score"]))
            if g > 0:
                qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = g
        return corpus, queries, qrels

    def _docs_for(self, dataset: str, dids: Iterable[str]) -> dict[str, str]:
        """anessbelbati passage texts (title\\ntext, cut to 2,000 chars) for the candidate ids."""
        wanted = set(dids)
        cache = self.raw_dir("rerank") / f"aness_docs_{dataset}.json"
        if cache.exists():
            docs = json.loads(cache.read_text())
            if wanted <= set(docs):
                return docs
        if dataset == "nevir":
            rev = self._rev("orionweller/NevIR")
            docs = {}
            for split in ("test", "validation", "train"):
                try:
                    p = hf_download("orionweller/NevIR", f"{split}.jsonl", rev, work=self.work)
                except Exception:
                    continue
                for r in read_jsonl(p):
                    docs[f"{r['id']}-d1"] = aness_truncate(r["doc1"])
                    docs[f"{r['id']}-d2"] = aness_truncate(r["doc2"])
        elif dataset in MTEB:
            corpus, _, _ = self._mteb(dataset)
            docs = {d: aness_truncate(t) for d, t in corpus.items() if d in wanted}
        else:
            corpus, _, _ = self._beir(dataset)
            docs = {d: aness_truncate(aness_doc(corpus[d])) for d in wanted if d in corpus}
        cache.write_text(json.dumps(docs))
        missing = wanted - set(docs)
        if missing:
            raise KeyError(f"{dataset}: {len(missing)} candidate ids missing from the corpus (e.g. {sorted(missing)[:3]})")
        return docs

    # --- candidate files from the pinned checkouts
    def _denser_candidates(self, dataset: str) -> dict[str, list[str]]:
        p = repos_dir() / "denser" / "cache" / f"candidates_{dataset}_test_k100.json"
        if sha256_file(p) != DENSER_CANDIDATES[dataset]:
            raise ValueError(f"{p} does not match the pinned sha256")
        return json.loads(p.read_text(encoding="utf-8"))

    def _aness_candidates(self, dataset: str) -> list[dict[str, Any]]:
        p = repos_dir() / "anessbelbati" / "candidates" / f"{dataset}.jsonl"
        if sha256_file(p) != ANESS_CANDIDATES[dataset]:
            raise ValueError(f"{p} does not match the pinned sha256")
        return read_jsonl(p)

    def prepare(self, split: str) -> dict[str, Any]:
        man = {"spec": SPEC_ID, "upstream": UPSTREAM, "denser_candidates_sha256": DENSER_CANDIDATES,
               "aness_candidates_sha256": ANESS_CANDIDATES, "revisions": {}}
        for ds in ("scifact", "nfcorpus"):
            man["revisions"][f"BeIR/{ds}"] = self._rev(f"BeIR/{ds}")
            man["revisions"][f"BeIR/{ds}-qrels"] = self._rev(f"BeIR/{ds}-qrels")
        man["revisions"]["orionweller/NevIR"] = self._rev("orionweller/NevIR")
        man["note"] = "hev candidate lists are rebuilt (not shipped upstream): hev rows are indicative"
        return man

    def expected_counts(self, split: str) -> dict[str, int]:
        if split != "test":
            return {}
        return {"denser_scifact": 300, "denser_nfcorpus": 323, "hev_scifact_batch": 300, "hev_nfcorpus_batch": 323,
                "hev_fiqa_batch": 300, "aness_scifact": 300, "aness_nfcorpus": 323, "aness_fiqa": 648, "aness_nq": 500,
                "aness_trec-covid": 50, "aness_bright-biology": 103, "aness_bright-economics": 103, "aness_csn-python": 300,
                "aness_nevir": 2766}

    # --- items
    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        out: list[Item] = []
        for task in self._select_tasks(tasks):
            repo, _, rest = task.partition("_")
            if repo == "denser":
                out.extend(self._denser_items(rest, split, limit))
            elif repo == "hev":
                ds, mode = rest.rsplit("_", 1)
                out.extend(self._hev_items(ds, mode, split, limit))
            elif repo == "aness":
                out.extend(self._aness_items(rest, split, limit))
        return out

    def _denser_items(self, dataset: str, split: str, limit: int | None) -> list[Item]:
        corpus, queries, qrels = self._beir(dataset, "test" if split == "test" else "train")
        if split == "test":
            cands = self._denser_candidates(dataset)
        else:  # self-check on train queries: same BM25 (k1=0.9, b=0.4, Lucene stopwords), depth 100
            cands = self._bm25_candidates(corpus, queries, 100, k1=0.9, b=0.4, limit=limit)
        qids = sorted(cands)
        items = []
        for qid in take(qids, limit):
            dids = cands[qid][:100]
            req = denser_request(queries[qid], [denser_passage(corpus[d]) for d in dids])
            items.append(Item(f"denser_{dataset}/{qid}", f"denser_{dataset}", req, qrels.get(qid, {}), meta={"dids": dids}))
        return items

    def _bm25_candidates(self, corpus: dict[str, dict[str, Any]], queries: dict[str, str], depth: int, k1: float, b: float,
                         limit: int | None = None) -> dict[str, list[str]]:
        dids = list(corpus)
        bm = BM25Okapi([tokenize(denser_passage(corpus[d])) for d in dids], k1=k1, b=b)
        out = {}
        for qid in take(sorted(queries), limit):
            sc = bm.get_scores(tokenize(queries[qid]))
            top = sorted(range(len(sc)), key=lambda i: -sc[i])[:depth]
            out[qid] = [dids[i] for i in top]
        return out

    def _hev_items(self, dataset: str, mode: str, split: str, limit: int | None) -> list[Item]:
        corpus, queries, qrels = self._beir(dataset, "test" if split == "test" else "train")
        cands = self._bm25_candidates(corpus, queries, HEV_TOP_K, k1=1.5, b=0.75, limit=limit)  # rank_bm25 defaults
        items = []
        for qid in take(sorted(cands), limit):
            dids = cands[qid]
            docs = [corpus[d] for d in dids]
            task = f"hev_{dataset}_{mode}"
            if mode == "batch":
                req = hev_request(queries[qid], docs)
                items.append(Item(f"{task}/{qid}", task, req, qrels.get(qid, {}), group=qid,
                                  meta={"dids": dids, "candidates_rebuilt": True}))
            else:
                for j, d in enumerate(docs):
                    req = hev_request(queries[qid], [d])
                    items.append(Item(f"{task}/{qid}/{j:02d}", task, req, qrels.get(qid, {}), group=qid,
                                      meta={"did": dids[j], "pos": j, "candidates_rebuilt": True}))
        return items

    def _aness_items(self, dataset: str, split: str, limit: int | None) -> list[Item]:
        task = f"aness_{dataset}"
        if split != "test" and dataset != "nevir":
            raise KeyError(f"{task}: only the test lists are published; self-check with aness_nevir or --limit on test items")
        rows = self._aness_candidates(dataset)
        rows = [r for r in rows if r["present"]]
        if dataset == "nevir" and split != "test":
            rows = self._nevir_validation_rows()
        docs = self._docs_for(dataset, (c["did"] for r in take(rows, limit) for c in r["present"]))
        items = []
        for r in take(rows, limit):
            dids = [c["did"] for c in r["present"]]
            req = aness_score_batch_request(r["query"], [docs[d] for d in dids])
            items.append(Item(f"{task}/{r['qid']}", task, req, r["relevant"], group=r.get("pair"),
                              meta={"dids": dids, "n_rel_top30": r.get("n_rel_top30", 0)}))
        return items

    def _nevir_validation_rows(self) -> list[dict[str, Any]]:
        rev = self._rev("orionweller/NevIR")
        p = hf_download("orionweller/NevIR", "validation.jsonl", rev, work=self.work)
        rows = []
        for r in read_jsonl(p):
            d1, d2 = f"{r['id']}-d1", f"{r['id']}-d2"
            for qn, (q, rel) in enumerate(((r["q1"], d1), (r["q2"], d2)), start=1):
                rows.append({"qid": f"{r['id']}-q{qn}", "pair": r["id"], "query": q, "relevant": {rel: 1},
                             "present": [{"did": d1}, {"did": d2}], "n_rel_top30": 1})
        return rows

    # --- scoring
    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        res: dict[str, Any] = {"spec": SPEC_ID, "split": split, "tasks": {}}
        aness_means: dict[str, float] = {}
        pooled: list[float] = []
        for task, its in by_task(items).items():
            repo = task.split("_", 1)[0]
            if repo == "denser":
                res["tasks"][task] = self._score_ranking(its, answers, self._denser_scores, exponential=True, all_queries=True)
            elif repo == "hev":
                mode = task.rsplit("_", 1)[1]
                fn = self._hev_batch_scores if mode == "batch" else None
                res["tasks"][task] = (self._score_ranking(its, answers, fn, exponential=False, all_queries=True) if fn
                                      else self._score_hev_pair(its, answers))
                res["tasks"][task]["candidates_rebuilt"] = True
            elif task == "aness_nevir":
                res["tasks"][task] = self._score_nevir(its, answers)
            else:
                r = self._score_ranking(its, answers, self._aness_scores, exponential=False, all_queries=False)
                res["tasks"][task] = r
                if r.get("ndcg10") is not None:
                    aness_means[task] = r["ndcg10"]
                    pooled.extend(r["per_query_ndcg"])
            res["tasks"][task]["target"] = TARGET_OF_TASK.get(task)
        if aness_means:
            english = {t: v for t, v in aness_means.items() if t.split("_", 1)[1] in ANESS_ENGLISH}
            res["aness_headline"] = {"ndcg10_each_dataset_once": sum(english.values()) / len(english) if english else None,
                                     "ndcg10_each_question_once": sum(pooled) / len(pooled) if pooled else None,
                                     "datasets": sorted(english), "n_questions": len(pooled), "target": "T392",
                                     "complete": len(english) == len(ANESS_ENGLISH)}
        for t in list(res["tasks"]):
            res["tasks"][t].pop("per_query_ndcg", None)
        return res

    @staticmethod
    def _denser_scores(it: Item, a: dict[str, Any]) -> list[float] | None:
        out = []
        for i in range(len(it.meta["dids"])):
            p = noul_p(a.get(f"p{i}"))
            if p is None:
                return None
            out.append(p)
        return out

    @staticmethod
    def _hev_batch_scores(it: Item, a: dict[str, Any]) -> list[float] | None:
        out = []
        for j in range(len(it.meta["dids"])):
            p = noul_p(a.get(f"D{j:02d}"))
            if p is None:
                return None
            out.append(p)
        return out

    @staticmethod
    def _aness_scores(it: Item, a: dict[str, Any]) -> list[float] | None:
        out = []
        for i in range(len(it.meta["dids"])):
            v = aness_expected_level(a.get(_pid(i)))
            if v is None:
                return None
            out.append(v)
        return out

    def _score_ranking(self, its: Sequence[Item], answers, score_fn, exponential: bool, all_queries: bool) -> dict[str, Any]:
        nd, mrr, failed, skipped = [], [], 0, 0
        for it in its:
            if not all_queries and not it.meta.get("n_rel_top30"):
                skipped += 1
                continue
            a = answers_of(answers.get(it.id))
            sc = score_fn(it, a) if a else None
            if sc is None:
                failed += 1
                continue
            ranked = order_by_score(it.meta["dids"], sc)
            nd.append(ndcg_at_k(ranked, it.gold, 10, exponential=exponential))
            mrr.append(mrr_at_k(ranked, it.gold, 10))
        return {**coverage(its, answers), "n_scored": len(nd), "failed": failed, "skipped_no_relevant_in_list": skipped,
                "ndcg10": sum(nd) / len(nd) if nd else None, "mrr10": sum(mrr) / len(mrr) if mrr else None,
                "gains": "2^rel-1" if exponential else "linear", "per_query_ndcg": nd}

    def _score_hev_pair(self, its: Sequence[Item], answers) -> dict[str, Any]:
        groups: dict[str, list[Item]] = {}
        for it in its:
            groups.setdefault(it.group, []).append(it)
        nd, mrr, failed = [], [], 0
        for qid, group in groups.items():
            group.sort(key=lambda x: x.meta["pos"])
            scores = []
            for it in group:
                a = answers_of(answers.get(it.id))
                p = noul_p(a.get("D00")) if a else None
                if p is None:
                    scores = None
                    break
                scores.append(p)
            if scores is None:
                failed += 1
                continue
            ranked = order_by_score([it.meta["did"] for it in group], scores)
            nd.append(ndcg_at_k(ranked, group[0].gold, 10, exponential=False))
            mrr.append(mrr_at_k(ranked, group[0].gold, 10))
        return {**coverage(its, answers), "n_queries": len(groups), "n_scored": len(nd), "failed_queries": failed,
                "ndcg10": sum(nd) / len(nd) if nd else None, "mrr10": sum(mrr) / len(mrr) if mrr else None, "gains": "linear"}

    def _score_nevir(self, its: Sequence[Item], answers) -> dict[str, Any]:
        pairs: dict[str, dict[str, Item]] = {}
        for it in its:
            pairs.setdefault(it.group, {})[it.id[-2:]] = it
        ok_pairs, per_q, same_pick, failed = [], [], [], 0
        for pid, qs in pairs.items():
            if set(qs) != {"q1", "q2"}:
                failed += 1
                continue
            rights, picks = [], {}
            for k in ("q1", "q2"):
                it = qs[k]
                a = answers_of(answers.get(it.id))
                sc = self._aness_scores(it, a) if a else None
                if sc is None:
                    rights = None
                    break
                s = dict(zip(it.meta["dids"], sc))
                rel = next(iter(it.gold))
                other = [d for d in it.meta["dids"] if d != rel][0]
                rights.append(s[rel] > s[other])
                picks[k] = max(it.meta["dids"], key=lambda d: (s[d], -it.meta["dids"].index(d)))
            if rights is None:
                failed += 1
                continue
            per_q.extend(rights)
            ok_pairs.append(all(rights))
            same_pick.append(picks["q1"] == picks["q2"])
        return {**coverage(its, answers), "pairs": len(ok_pairs), "pairs_failed": failed,
                "paired_accuracy": sum(ok_pairs) / len(ok_pairs) if ok_pairs else None,
                "question_accuracy": sum(per_q) / len(per_q) if per_q else None,
                "same_top_pick_share": sum(same_pick) / len(same_pick) if same_pick else None, "target": "T393"}
