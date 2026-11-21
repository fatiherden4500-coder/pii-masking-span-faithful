"""Span-level evaluation for generative placeholder anonymization.

For each gold PII span we align source and prediction tokens (difflib) and
assign one status:
  LEAKED_FULL / LEAKED_PARTIAL : all / some alphanumeric tokens copied verbatim
  MASKED_CORRECT               : replaced by a placeholder of the correct base type
  MASKED_WRONG                 : replaced by placeholder(s) of another type only
  REMOVED                      : replaced by free text or deleted (no placeholder)
  DROPPED                      : lost because the output stops early (truncation)
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, List

from .common import PII_TYPE_SET, TAG_RE, TIER_OF, TIER_WEIGHT, base_type

TOK_RE = re.compile(r"\[[A-Z_]+\d*\]|\w+|[^\w\s]")
WORD_RE = re.compile(r"\w")


def tokenize(text):
    return [(m.group(0), m.start(), m.end()) for m in TOK_RE.finditer(text)]


def _wildcard_align(s_str, span_bounds, p_str, is_tag):
    """LCS between source and prediction tokens where a whole gold span (token range
    [b, e)) may alternatively be matched to ONE placeholder token. Returns
    ({span_k: placeholder_label}, last matched source index)."""
    n, m = len(s_str), len(p_str)
    start_of = {b: (k, e) for k, (b, e) in span_bounds.items()}
    f = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        fi, fn, si = f[i], f[i + 1], s_str[i]
        st = start_of.get(i)
        fe = f[st[1]] if st else None
        for j in range(m - 1, -1, -1):
            a, b = fn[j], fi[j + 1]
            best = a if a > b else b
            if si == p_str[j]:
                v = fn[j + 1] + 1
                if v > best:
                    best = v
            if fe is not None and is_tag[j]:
                v = fe[j + 1] + 1
                if v > best:
                    best = v
            fi[j] = best
    span_tag, last_src, i, j = {}, -1, 0, 0
    while i < n and j < m:
        st = start_of.get(i)
        if st is not None and is_tag[j] and f[i][j] == f[st[1]][j + 1] + 1:
            span_tag[st[0]] = TAG_RE.fullmatch(p_str[j]).group(1)
            last_src = st[1] - 1
            i, j = st[1], j + 1
        elif s_str[i] == p_str[j] and f[i][j] == f[i + 1][j + 1] + 1:
            last_src = i
            i += 1; j += 1
        elif f[i + 1][j] >= f[i][j + 1]:
            i += 1
        else:
            j += 1
    return span_tag, last_src


def score_sample(source: str, prediction: str, spans: List[Dict], target: str | None = None) -> Dict:
    s_toks = tokenize(source)
    p_toks = tokenize(prediction)
    s_str = [t[0] for t in s_toks]
    p_str = [t[0] for t in p_toks]
    # (1) verbatim-copy alignment -> leakage & non-PII preservation
    sm = SequenceMatcher(None, s_str, p_str, autojunk=False)
    ops = sm.get_opcodes()
    s_op = [None] * len(s_toks)
    for tag, i1, i2, j1, j2 in ops:
        for i in range(i1, i2):
            s_op[i] = tag

    tok_span = [None] * len(s_toks)
    span_tok = []
    for k, sp in enumerate(spans):
        idx = [i for i, (_, a, b) in enumerate(s_toks) if a < sp["end"] and b > sp["start"]]
        for i in idx:
            if tok_span[i] is None:
                tok_span[i] = k
        span_tok.append(idx)
    in_span = [ts is not None for ts in tok_span]

    # (2) wildcard alignment -> which placeholder replaced which span; truncation
    bounds = {}
    for k, idx in enumerate(span_tok):
        own = [i for i in idx if tok_span[i] == k]
        if own and own == list(range(own[0], own[-1] + 1)):
            bounds[k] = (own[0], own[-1] + 1)
    is_tag = [bool(TAG_RE.fullmatch(t)) for t in p_str]
    if len(s_str) * len(p_str) <= 600_000:
        span_tag, last_src = _wildcard_align(s_str, bounds, p_str, is_tag)
    else:  # pathological length: fall back to copy alignment only
        span_tag, last_src = {}, max((i for i, o in enumerate(s_op) if o == "equal"), default=-1)

    span_res = []
    for k, sp in enumerate(spans):
        label = base_type(sp["label"])
        alnum = [i for i in span_tok[k] if WORD_RE.search(s_toks[i][0])]
        if not alnum:
            continue
        copied = [i for i in alnum if s_op[i] == "equal"]
        if len(copied) == len(alnum):
            status = "LEAKED_FULL"
        elif copied:
            status = "LEAKED_PARTIAL"
        elif k in span_tag:
            status = "MASKED_CORRECT" if base_type(span_tag[k]) == label else "MASKED_WRONG"
        elif span_tok[k][0] > last_src:
            status = "DROPPED"
        else:
            status = "REMOVED"
        span_res.append((label, status))

    non_pii = [i for i in range(len(s_toks)) if not in_span[i] and WORD_RE.search(s_toks[i][0])]
    preserved = sum(1 for i in non_pii if s_op[i] == "equal")
    fp_tags = max(sum(is_tag) - len(span_tag), 0)
    pred_tags = [TAG_RE.fullmatch(t[0]).group(1) for t in p_toks if TAG_RE.fullmatch(t[0])]
    invalid = sum(1 for t in pred_tags if base_type(t) not in PII_TYPE_SET)
    src_vocab = set(t[0] for t in s_toks)
    halluc = sum(1 for t in p_toks if WORD_RE.search(t[0]) and not TAG_RE.fullmatch(t[0])
                 and t[0] not in src_vocab)
    t_len = len(tokenize(target)) if target is not None else None
    return {
        "spans": span_res,
        "non_pii_total": len(non_pii),
        "non_pii_preserved": preserved,
        "fp_tags": fp_tags,
        "n_pred_tags": len(pred_tags),
        "invalid_tags": invalid,
        "halluc_tokens": halluc,
        "n_src_tokens": len(s_toks),
        "len_ratio": (len(p_toks) / max(t_len, 1)) if t_len is not None else None,
    }


# ───────────────────────── per-sample scalars ─────────────────────────
def sample_scalars(r: Dict) -> Dict[str, float]:
    spans = r["spans"]
    n = len(spans)
    if n:
        w = [TIER_WEIGHT[TIER_OF.get(l, "LOW")] for l, _ in spans]
        leak_w = sum(wi * (1.0 if s == "LEAKED_FULL" else 0.5 if s == "LEAKED_PARTIAL" else 0.0)
                     for wi, (_, s) in zip(w, spans))
        protect = 1.0 - leak_w / sum(w)
        typed = sum(1 for _, s in spans if s == "MASKED_CORRECT") / n
    else:
        protect, typed = 1.0, 1.0
    preserve = r["non_pii_preserved"] / r["non_pii_total"] if r["non_pii_total"] else 1.0
    invalid = r["invalid_tags"] / r["n_pred_tags"] if r["n_pred_tags"] else 0.0
    halluc = min(r["halluc_tokens"] / max(r["n_src_tokens"], 1), 1.0)
    return {"protect": protect, "typed": typed, "preserve": preserve, "invalid": invalid, "halluc": halluc}


def composite_reward(source: str, prediction: str, spans: List[Dict], w: Dict[str, float]) -> float:
    s = sample_scalars(score_sample(source, prediction, spans))
    return (w["w_protect"] * s["protect"] + w["w_typed"] * s["typed"] + w["w_preserve"] * s["preserve"]
            - w["w_invalid"] * s["invalid"] - w["w_halluc"] * s["halluc"])


# ───────────────────────── aggregation ─────────────────────────
def bag_of_tags(pred, tgt):
    pc, gc = Counter(TAG_RE.findall(pred)), Counter(TAG_RE.findall(tgt))
    tp = sum((pc & gc).values())
    return tp, sum(pc.values()) - tp, sum(gc.values()) - tp


def aggregate(scored: List[Dict]) -> Dict:
    """scored: list of {'r': score_sample(...), 'prediction', 'target'}"""
    st, per_type, per_tier = Counter(), defaultdict(Counter), defaultdict(Counter)
    leak_samples = npt = npp = fp = ptags = hal = inv = srcs = trunc = em = 0
    tp = fpb = fnb = 0
    for s in scored:
        r = s["r"]
        any_leak = False
        for label, status in r["spans"]:
            st[status] += 1
            per_type[label][status] += 1
            per_tier[TIER_OF.get(label, "LOW")][status] += 1
            any_leak |= status.startswith("LEAKED")
        leak_samples += any_leak
        npt += r["non_pii_total"]; npp += r["non_pii_preserved"]
        fp += r["fp_tags"]; ptags += r["n_pred_tags"]; hal += r["halluc_tokens"]
        inv += r["invalid_tags"]; srcs += r["n_src_tokens"]
        trunc += (r["len_ratio"] is not None and r["len_ratio"] < 0.8)
        em += s["prediction"].strip() == s["target"].strip()
        a, b, c = bag_of_tags(s["prediction"], s["target"])
        tp += a; fpb += b; fnb += c
    n = len(scored)
    n_sp = sum(st.values()) or 1
    leaked = st["LEAKED_FULL"] + st["LEAKED_PARTIAL"]
    bp, br = tp / max(tp + fpb, 1), tp / max(tp + fnb, 1)

    def leak_of(c):
        tot = sum(c.values())
        return round((c["LEAKED_FULL"] + c["LEAKED_PARTIAL"]) / tot, 4) if tot else None

    return {
        "n_samples": n,
        "n_gold_spans": sum(st.values()),
        "span_leak_rate": round(leaked / n_sp, 4),
        "typed_span_recall": round(st["MASKED_CORRECT"] / n_sp, 4),
        "untyped_mask_recall": round((st["MASKED_CORRECT"] + st["MASKED_WRONG"]) / n_sp, 4),
        "sample_leak_rate": round(leak_samples / max(n, 1), 4),
        "mask_precision": round(1 - fp / ptags, 4) if ptags else None,
        "non_pii_preservation": round(npp / max(npt, 1), 4),
        "invalid_tag_rate": round(inv / ptags, 4) if ptags else 0.0,
        "hallucination_rate": round(hal / max(srcs, 1), 4),
        "truncation_rate": round(trunc / max(n, 1), 4),
        "exact_match": round(em / max(n, 1), 4),
        "span_status": {k: round(v / n_sp, 4) for k, v in sorted(st.items())},
        "legacy_bag_of_tags": {"precision": round(bp, 4), "recall": round(br, 4),
                               "f1": round(2 * bp * br / max(bp + br, 1e-9), 4)},
        "per_tier_leak_rate": {k: leak_of(v) for k, v in sorted(per_tier.items())},
        "per_type": {k: {"support": sum(v.values()), "leak_rate": leak_of(v),
                         "typed_recall": round(v["MASKED_CORRECT"] / sum(v.values()), 4)}
                     for k, v in sorted(per_type.items(), key=lambda x: -sum(x[1].values()))},
    }


# ───────────────────────── bootstrap ─────────────────────────
BOOT_KEYS = ["span_leak_rate", "typed_span_recall", "sample_leak_rate", "non_pii_preservation"]


def sample_row(r):
    n = len(r["spans"])
    lk = sum(1 for _, s in r["spans"] if s.startswith("LEAKED"))
    mc = sum(1 for _, s in r["spans"] if s == "MASKED_CORRECT")
    return (n, lk, mc, r["non_pii_total"], r["non_pii_preserved"], 1 if lk else 0)


def _stats(rows_by_seed, idx):
    n = lk = mc = nt = npres = sl = cnt = 0
    for rows in rows_by_seed:
        for i in idx:
            a = rows[i]
            n += a[0]; lk += a[1]; mc += a[2]; nt += a[3]; npres += a[4]; sl += a[5]; cnt += 1
    return {"span_leak_rate": lk / max(n, 1), "typed_span_recall": mc / max(n, 1),
            "sample_leak_rate": sl / max(cnt, 1), "non_pii_preservation": npres / max(nt, 1)}


def paired_bootstrap(method_rows: Dict[str, List[List[tuple]]], ref: str, B: int = 2000, seed: int = 0,
                     clusters: List[str] | None = None):
    """method_rows[m] = list over seeds of per-sample rows (aligned by test index).
    Cluster bootstrap: resamples whole DOCUMENTS (test chunks of one document are correlated);
    all chunks and all seeds of a resampled document are included. clusters=None -> per-sample."""
    rng = random.Random(seed)
    N = len(next(iter(method_rows.values()))[0])
    groups = defaultdict(list)
    for i in range(N):
        groups[clusters[i] if clusters else i].append(i)
    glist = list(groups.values())
    draws = []
    for _ in range(B):
        d = []
        for _ in range(len(glist)):
            d.extend(glist[rng.randrange(len(glist))])
        draws.append(d)
    q = lambda v: [round(sorted(v)[int(0.025 * B)], 4), round(sorted(v)[min(int(0.975 * B), B - 1)], 4)]
    ci, diffs = {}, {}
    cache = {m: [_stats(rows, d) for d in draws] for m, rows in method_rows.items()}
    for m in method_rows:
        ci[m] = {k: q([s[k] for s in cache[m]]) for k in BOOT_KEYS}
    if ref in method_rows:
        for m in method_rows:
            if m == ref:
                continue
            diffs[m] = {}
            for k in BOOT_KEYS:
                dv = [b[k] - a[k] for a, b in zip(cache[ref], cache[m])]
                p = 2 * min(sum(x <= 0 for x in dv), sum(x >= 0 for x in dv)) / B
                diffs[m][k] = {"delta_ci95": q(dv), "p": round(min(p, 1.0), 4)}
    return ci, diffs
