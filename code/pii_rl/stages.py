"""Prediction, Best-of-N selection and final evaluation."""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from .common import Paths, get_logger, read_jsonl, write_jsonl
from .metrics import aggregate, composite_reward, paired_bootstrap, sample_row, score_sample


# ───────────────────────── generative prediction ─────────────────────────
def predict(cfg, method: str, seed: int):
    """method: zeroshot | sft | dpo | grpo | rloo | <any trained name>."""
    paths = Paths(cfg)
    name = f"{method}__s{seed}"
    log = get_logger(paths, f"{name}_predict")
    if paths.pred(name).with_suffix(".complete").exists():
        log.info("already done"); return
    from .generation import generate_to_file, load_policy
    if method == "zeroshot":
        base, adapter = cfg["zeroshot"].get("model") or cfg["model"]["name"], None
    elif method == "sft" or method.startswith("sft"):
        base, adapter = str(paths.model(f"{method}_s{seed}") / "merged"), None
    else:
        base = str(paths.model(f"sft_s{seed}") / "merged")
        adapter = str(paths.model(f"{method}_s{seed}") / "adapter")
    model, tok = load_policy(cfg, base, adapter)
    g = cfg["generation"]
    generate_to_file(model, tok, read_jsonl(paths.split("test")), paths.pred(name), n=1,
                     temperature=g["eval_temperature"], max_new_tokens=g["max_new_tokens"],
                     batch_size=g["batch_size"], log=log, extra={"method": method, "seed": seed})


# ───────────────────────── Best-of-N ─────────────────────────
def bon(cfg, seed: int):
    """Sample N candidates from SFT_s{seed}; select with
       - oracle  : gold-span reward (UPPER BOUND, not deployable)
       - verifier: same reward but with spans predicted by the DeBERTa detector (deployable)"""
    paths = Paths(cfg)
    log = get_logger(paths, f"bon_s{seed}")
    out_o, out_v = paths.pred(f"bon_oracle__s{seed}"), paths.pred(f"bon_verifier__s{seed}")
    if out_o.exists() and out_v.exists():
        log.info("already done"); return
    from .generation import generate_to_file, load_policy
    test = read_jsonl(paths.split("test"))
    bc = cfg["bon"]
    cand_path = paths.cands / f"bon_test_s{seed}.jsonl"
    if not cand_path.with_suffix(".complete").exists():
        model, tok = load_policy(cfg, str(paths.model(f"sft_s{seed}") / "merged"))
        generate_to_file(model, tok, test, cand_path, n=bc["n"], temperature=bc["temperature"],
                         top_p=bc["top_p"], max_new_tokens=cfg["generation"]["max_new_tokens"],
                         batch_size=cfg["generation"]["batch_size"], log=log)
        del model
    cands = read_jsonl(cand_path)
    # greedy SFT output is always a candidate as well
    greedy = {r["idx"]: r["prediction"] for r in read_jsonl(paths.pred(f"sft__s{seed}"))}
    det_path = paths.model("tokclf_s1") / "final"
    from .tokclf import SpanDetector
    det = SpanDetector(str(det_path), cfg["tokclf"]["max_length"])
    det_spans = det.predict([r["source_text"] for r in test])
    w = cfg["reward"]
    rows_o, rows_v = [], []
    for i, (rec, c) in enumerate(zip(test, cands)):
        pool = list(dict.fromkeys(([greedy[i]] if i in greedy else []) + c["candidates"]))
        o = max(pool, key=lambda t: composite_reward(rec["source_text"], t, rec["spans"], w))
        v = max(pool, key=lambda t: composite_reward(rec["source_text"], t, det_spans[i], w))
        base = {"idx": i, "id": rec["id"], "source_text": rec["source_text"], "target_text": rec["target_text"],
                "seed": seed}
        rows_o.append({**base, "prediction": o, "method": "bon_oracle"})
        rows_v.append({**base, "prediction": v, "method": "bon_verifier"})
    write_jsonl(out_o, rows_o)
    write_jsonl(out_v, rows_v)


# ───────────────────────── evaluation ─────────────────────────
MAIN_COLS = [("span_leak_rate", "Span leak ↓"), ("sample_leak_rate", "Sample leak ↓"),
             ("typed_span_recall", "Typed recall ↑"), ("mask_precision", "Mask prec. ↑"),
             ("non_pii_preservation", "Non-PII pres. ↑"), ("hallucination_rate", "Halluc. ↓"),
             ("truncation_rate", "Trunc. ↓"), ("exact_match", "EM ↑")]


def evaluate_all(cfg):
    paths = Paths(cfg)
    log = get_logger(paths, "evaluate")
    test = read_jsonl(paths.split("test"))
    files = sorted(p for p in paths.preds.glob("*__s*.jsonl") if not p.name.endswith(".partial.jsonl"))
    per_run, rows_by_method = {}, defaultdict(list)
    # reference systems
    refs = {"identity__s0": [r["source_text"] for r in test], "oracle_target__s0": [r["target_text"] for r in test]}
    runs = {k: v for k, v in refs.items()}
    for f in files:
        preds = read_jsonl(f)
        if len(preds) != len(test):
            log.info(f"skip incomplete {f.name} ({len(preds)}/{len(test)})"); continue
        runs[f.stem] = [p["prediction"] for p in sorted(preds, key=lambda x: x["idx"])]
    for run, preds in runs.items():
        scored = [{"r": score_sample(r["source_text"], p, r["spans"], r["target_text"]),
                   "prediction": p, "target": r["target_text"]} for r, p in zip(test, preds)]
        per_run[run] = aggregate(scored)
        method = run.split("__s")[0]
        if not run.endswith("__s0") or method == "presidio":
            rows_by_method[method].append([sample_row(s["r"]) for s in scored])
    # mean ± sd across seeds
    by_method = defaultdict(list)
    for run, m in per_run.items():
        by_method[run.split("__s")[0]].append(m)
    summary = {}
    for method, ms in by_method.items():
        summary[method] = {"n_seeds": len(ms)}
        for k, _ in MAIN_COLS + [("untyped_mask_recall", "")]:
            vals = [m[k] for m in ms if m[k] is not None]
            if vals:
                summary[method][k] = {"mean": round(statistics.mean(vals), 4),
                                      "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0}
        lf = [m["legacy_bag_of_tags"]["f1"] for m in ms]
        summary[method]["legacy_f1"] = {"mean": round(statistics.mean(lf), 4),
                                        "sd": round(statistics.stdev(lf), 4) if len(lf) > 1 else 0.0}
    ref = cfg["eval"]["reference_method"]
    ci, diffs = paired_bootstrap(dict(rows_by_method), ref, B=cfg["eval"]["bootstrap"],
                                  clusters=[r.get("doc_id", r["id"]) for r in test]) if rows_by_method else ({}, {})
    out = {"summary": summary, "bootstrap_ci95": ci, f"paired_vs_{ref}": diffs, "per_run": per_run}
    (paths.results / "evaluation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    md = _markdown(summary, ci, diffs, ref)
    (paths.results / "main_table.md").write_text(md)
    print(md)
    log.info(f"wrote {paths.results / 'evaluation.json'}")


ORDER = ["identity", "oracle_target", "presidio", "zeroshot", "tokclf", "sft", "sft_plus", "bon_verifier", "bon_oracle",
         "dpo", "rloo", "grpo"]


def _markdown(summary, ci, diffs, ref):
    methods = sorted(summary, key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m))
    head = "| Method | seeds | " + " | ".join(h for _, h in MAIN_COLS) + " | legacy F1 |"
    lines = [head, "|" + "---|" * (len(MAIN_COLS) + 3)]
    for m in methods:
        s = summary[m]
        cells = []
        for k, _ in MAIN_COLS:
            v = s.get(k)
            cells.append("–" if v is None else (f"{v['mean']:.3f}" + (f" ± {v['sd']:.3f}" if s["n_seeds"] > 1 else "")))
        lines.append(f"| {m} | {s['n_seeds']} | " + " | ".join(cells) + f" | {s['legacy_f1']['mean']:.3f} |")
    lines += ["", f"Paired document-level (cluster) bootstrap vs {ref} (Δ = method − {ref}, 95% CI, p):", ""]
    for m in methods:
        if m in diffs:
            d = diffs[m]
            lines.append(f"- **{m}**: " + "; ".join(
                f"{k}: Δ∈[{v['delta_ci95'][0]:+.3f}, {v['delta_ci95'][1]:+.3f}] p={v['p']:.3f}" for k, v in d.items()))
    return "\n".join(lines) + "\n"
