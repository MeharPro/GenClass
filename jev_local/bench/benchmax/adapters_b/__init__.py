"""benchmax adapters, group B (bm-adapters-b): LexGLUE (chepyle), safety (mbburabak), DMB expanded, the rerank
scripts (denser / hev / anessbelbati), the single-author study adapters and goya's reward-model suites.

Every adapter module exposes the same module-level contract (see `jev_local.bench.benchmax.specs_b`):

    SPEC_ID, TARGETS, SPLITS, TASKS
    prepare(work, split) -> manifest dict          (fetch + pin + verify upstream files; VM only)
    items(work, split, limit=None, tasks=None) -> list[Item]   (request bytes exactly as the publisher sends them)
    score(items, answers, split, thresholds=None) -> dict      (the publisher's metrics)
    fit(items, answers) -> thresholds dict          (optional; validation items only)

Pure-Python at import time: numpy / pyarrow / huggingface_hub are imported inside the functions that need them,
so the package imports on the Mac (CONTRACT hard rule 1).
"""
