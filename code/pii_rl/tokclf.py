"""Encoder baseline: DeBERTa-v3 token classifier (BIO over base PII types).

Used (1) as a strong non-generative baseline and (2) as a reference-free
verifier for Best-of-N selection (detector spans replace gold spans).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .common import PII_TYPES, Paths, get_logger, is_done, last_checkpoint, mark_done, mixed_precision_flags, \
    read_jsonl, set_seed, write_jsonl, env_report

LABELS = ["O"] + [f"{p}-{t}" for t in PII_TYPES for p in ("B", "I")]
L2I = {l: i for i, l in enumerate(LABELS)}


def _encode(tok, rec, max_length):
    enc = tok(rec["source_text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
    labels = []
    spans = rec.get("spans", [])
    for (a, b) in enc["offset_mapping"]:
        if a == b:
            labels.append(-100)
            continue
        lab = "O"
        for sp in spans:
            if a < sp["end"] and b > sp["start"]:
                lab = ("B-" if a <= sp["start"] else "I-") + sp["label"]
                break
        labels.append(L2I[lab])
    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc


def train_tokclf(cfg, seed: int):
    paths = Paths(cfg)
    stage = f"tokclf_s{seed}_train"
    log = get_logger(paths, stage)
    if is_done(paths, stage):
        log.info("already done"); return
    set_seed(seed)
    from datasets import Dataset
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              Trainer, TrainingArguments)
    tc = cfg["tokclf"]
    tok = AutoTokenizer.from_pretrained(tc["model"])
    tr = Dataset.from_list([_encode(tok, r, tc["max_length"]) for r in read_jsonl(paths.split("train"))])
    va = Dataset.from_list([_encode(tok, r, tc["max_length"]) for r in read_jsonl(paths.split("val"))])
    # fp32 master weights: transformers 5 loads the fp16 DeBERTa-v3 checkpoint as fp16 by default,
    # and AdamW on fp16 weights (eps underflow) turns them NaN after the first step.
    import torch
    model = AutoModelForTokenClassification.from_pretrained(
        tc["model"], num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I, dtype=torch.float32)
    mdir = paths.model(f"tokclf_s{seed}")
    args = TrainingArguments(
        output_dir=str(mdir / "ckpt"), seed=seed, num_train_epochs=tc["epochs"], learning_rate=tc["lr"],
        per_device_train_batch_size=tc["batch_size"], per_device_eval_batch_size=tc["batch_size"],
        warmup_steps=0.06, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=20, report_to="none", **mixed_precision_flags(cfg))
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va,
                      data_collator=DataCollatorForTokenClassification(tok))
    trainer.train(resume_from_checkpoint=last_checkpoint(mdir / "ckpt"))
    tmp = mdir / "final.tmp"
    trainer.save_model(str(tmp)); tok.save_pretrained(str(tmp)); tmp.rename(mdir / "final")
    mark_done(paths, stage, {"env": env_report()})


class SpanDetector:
    def __init__(self, path, max_length=512, batch_size=32):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        from .generation import device_name
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(device_name()).eval()
        self.max_length, self.bs, self.torch = max_length, batch_size, torch

    def predict(self, texts: List[str]) -> List[List[Dict]]:
        out = []
        for i in range(0, len(texts), self.bs):
            batch = texts[i:i + self.bs]
            enc = self.tok(batch, truncation=True, max_length=self.max_length, padding=True,
                           return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping").tolist()
            with self.torch.no_grad():
                logits = self.model(**{k: v.to(self.model.device) for k, v in enc.items()}).logits
            preds = logits.argmax(-1).tolist()
            for text, offs, pr in zip(batch, offsets, preds):
                out.append(self._decode(text, offs, pr))
        return out

    @staticmethod
    def _decode(text, offs, pr):
        spans, cur = [], None
        for (a, b), p in zip(offs, pr):
            if a == b:
                continue
            lab = LABELS[p]
            if lab == "O":
                cur = None
                continue
            bio, typ = lab.split("-", 1)
            if cur is not None and cur["label"] == typ and (bio == "I" or a <= cur["end"] + 1):
                cur["end"] = b
            else:
                cur = {"start": a, "end": b, "label": typ}
                spans.append(cur)
        for sp in spans:  # strip whitespace captured by sentencepiece offsets
            while sp["start"] < sp["end"] and text[sp["start"]].isspace():
                sp["start"] += 1
            sp["value"] = text[sp["start"]:sp["end"]]
        return [s for s in spans if s["end"] > s["start"]]


def anonymize_with_spans(text: str, spans: List[Dict]) -> str:
    parts, prev = [], 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        if sp["start"] < prev:
            continue
        parts.append(text[prev:sp["start"]]); parts.append(f"[{sp['label']}]"); prev = sp["end"]
    parts.append(text[prev:])
    return "".join(parts)


def predict_tokclf(cfg, seed: int):
    paths = Paths(cfg)
    name = f"tokclf__s{seed}"
    log = get_logger(paths, f"{name}_predict")
    if paths.pred(name).exists():
        log.info("already done"); return
    test = read_jsonl(paths.split("test"))
    det = SpanDetector(str(paths.model(f"tokclf_s{seed}") / "final"), cfg["tokclf"]["max_length"])
    spans = det.predict([r["source_text"] for r in test])
    rows = [{"idx": i, "id": r["id"], "source_text": r["source_text"], "target_text": r["target_text"],
             "prediction": anonymize_with_spans(r["source_text"], sp), "pred_spans": sp}
            for i, (r, sp) in enumerate(zip(test, spans))]
    write_jsonl(paths.pred(name), rows)


def predict_presidio(cfg):
    """Optional off-the-shelf baseline (pip install presidio-analyzer + spaCy en_core_web_lg)."""
    paths = Paths(cfg)
    name = "presidio__s0"
    log = get_logger(paths, "presidio")
    if paths.pred(name).exists():
        return
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        log.info("presidio not installed -> skipped"); return
    MAP = {"PERSON": "GIVENNAME", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "TEL", "IP_ADDRESS": "IP",
           "LOCATION": "CITY", "DATE_TIME": "DATE", "US_SSN": "SOCIALNUMBER", "US_PASSPORT": "PASSPORT",
           "US_DRIVER_LICENSE": "DRIVERLICENSE", "NRP": "COUNTRY", "URL": "USERNAME"}
    eng = AnalyzerEngine()
    test = read_jsonl(paths.split("test"))
    rows = []
    for i, r in enumerate(test):
        res = eng.analyze(text=r["source_text"], language="en")
        sp = [{"start": x.start, "end": x.end, "label": MAP.get(x.entity_type, x.entity_type)} for x in res]
        rows.append({"idx": i, "id": r["id"], "source_text": r["source_text"], "target_text": r["target_text"],
                     "prediction": anonymize_with_spans(r["source_text"], sp)})
    write_jsonl(paths.pred(name), rows)
