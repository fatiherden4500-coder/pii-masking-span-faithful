"""Data preparation with DOCUMENT-LEVEL splits and gold PII spans.

ai4privacy/pii-masking-300k stores long documents as several chunks whose ids
share a numeric root (e.g. 40768A, 40768B, 40768C). Random record-level splits
put sibling chunks of the same document in train and test, which inflates
results. Here every document root is assigned to exactly one pool.
"""

from __future__ import annotations

import os
import random
import re
from collections import Counter, defaultdict

from .common import PII_TYPE_SET, TAG_RE, Paths, base_type, get_logger, mark_done, is_done, write_jsonl


def _doc_root(rec_id: str) -> str:
    return re.sub(r"[A-Za-z]+$", "", rec_id)


def _validate(rec, max_chars):
    src, tgt, pm = rec["source_text"], rec["target_text"], rec["privacy_mask"]
    if not src or not tgt:
        return None, "empty"
    if len(src) > max_chars:
        return None, "too_long"
    spans = []
    for sp in sorted(pm, key=lambda x: x["start"]):
        if src[sp["start"]:sp["end"]] != sp["value"]:
            return None, "span_offset_mismatch"
        bt = base_type(sp["label"])
        if bt not in PII_TYPE_SET:
            return None, "unknown_span_label"
        if spans and sp["start"] < spans[-1]["end"]:
            return None, "overlapping_spans"
        spans.append({"start": sp["start"], "end": sp["end"], "label": bt,
                      "raw_label": sp["label"], "value": sp["value"]})
    tags = TAG_RE.findall(tgt)
    if any(base_type(t) not in PII_TYPE_SET for t in tags):
        return None, "unknown_target_tag"
    if len(tags) != len(spans):
        return None, "tag_span_count_mismatch"
    # strict consistency: target must equal source with spans replaced by [raw_label]
    rebuilt, prev = [], 0
    for sp in spans:
        rebuilt.append(src[prev:sp["start"]])
        rebuilt.append(f"[{sp['raw_label']}]")
        prev = sp["end"]
    rebuilt.append(src[prev:])
    if "".join(rebuilt) != tgt:
        return None, "target_not_reconstructible"
    return spans, None


def prepare_data(cfg):
    paths = Paths(cfg)
    log = get_logger(paths, "prepare_data")
    if is_done(paths, "prepare_data"):
        log.info("prepare_data already done")
        return
    from datasets import load_dataset

    dc = cfg["data"]
    ds = load_dataset(dc["dataset"])
    docs = defaultdict(list)
    reasons = Counter()
    for split in ds:
        cols = ["source_text", "target_text", "privacy_mask", "id", "language"]
        for i, rec in enumerate(ds[split].select_columns(cols)):
            if rec["language"] != dc["language"]:
                continue
            spans, why = _validate(rec, dc["max_source_chars"])
            if why:
                reasons[why] += 1
                continue
            reasons["kept"] += 1
            docs[(split, _doc_root(rec["id"]))].append({
                "id": rec["id"], "doc_id": f"{split}:{_doc_root(rec['id'])}", "hf_split": split,
                "hf_index": i, "source_text": rec["source_text"], "target_text": rec["target_text"],
                "spans": spans,
            })
    log.info(f"filter stats: {dict(reasons)}")

    keys = sorted(docs)
    rng = random.Random(dc["split_seed"])
    rng.shuffle(keys)
    need = {"test": dc["n_test"], "val": dc["n_val"], "rl_pool": dc["n_rl_pool"], "train": dc["n_train"]}
    pools = {k: [] for k in need}
    order = ["test", "val", "rl_pool", "train"]
    ki = 0
    for name in order:
        # take whole documents until the pool has enough records
        while len(pools[name]) < need[name] and ki < len(keys):
            pools[name].extend(docs[keys[ki]])
            ki += 1
    for name in order:
        rows = pools[name]
        rng.shuffle(rows)
        pools[name] = rows[:need[name]]
        if len(pools[name]) < need[name]:
            log.info(f"WARNING: pool {name} has only {len(pools[name])} records")
        write_jsonl(paths.split(name), pools[name])

    # audit: no document overlap
    doc_sets = {n: {r["doc_id"] for r in pools[n]} for n in order}
    for a in order:
        for b in order:
            if a < b:
                assert not (doc_sets[a] & doc_sets[b]), f"doc overlap {a}/{b}"
    stats = {n: {"records": len(pools[n]), "docs": len(doc_sets[n]),
                 "spans": sum(len(r["spans"]) for r in pools[n]),
                 "type_dist": dict(Counter(s["label"] for r in pools[n] for s in r["spans"]).most_common())}
             for n in order}
    stats["filter"] = dict(reasons)
    mark_done(paths, "prepare_data", stats)
    log.info({n: stats[n]["records"] for n in order})
