"""Shared utilities: config, paths, IO, seeding, prompts, stage markers."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

V2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = V2_ROOT / "configs" / "default.yaml"

# Base PII types present in ai4privacy/pii-masking-300k (English)
PII_TYPES = [
    "BOD", "BUILDING", "CITY", "COUNTRY", "DATE", "DRIVERLICENSE", "EMAIL",
    "GEOCOORD", "GIVENNAME", "IDCARD", "IP", "LASTNAME", "PASS", "PASSPORT",
    "POSTCODE", "SECADDRESS", "SEX", "SOCIALNUMBER", "STATE", "STREET", "TEL",
    "TIME", "TITLE", "USERNAME",
]
PII_TYPE_SET = set(PII_TYPES)

RISK_TIERS = {
    "CRITICAL": ["PASS", "SOCIALNUMBER", "IDCARD", "DRIVERLICENSE", "PASSPORT"],
    "HIGH": ["EMAIL", "TEL", "IP", "USERNAME", "STREET", "BUILDING", "POSTCODE",
             "SECADDRESS", "GEOCOORD"],
    "MEDIUM": ["GIVENNAME", "LASTNAME", "BOD", "DATE", "STATE"],
    "LOW": ["CITY", "COUNTRY", "SEX", "TITLE", "TIME"],
}
TIER_OF = {t: k for k, v in RISK_TIERS.items() for t in v}
TIER_WEIGHT = {"CRITICAL": 4.0, "HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}

TAG_RE = re.compile(r"\[([A-Z_]+\d*)\]")

SYSTEM_PROMPT = (
    "You are a privacy protection assistant. Rewrite the user's text so that every piece of "
    "personally identifiable information (PII) is replaced by a placeholder of the form [TYPE], "
    "and everything else is copied exactly, character for character.\n"
    "Allowed TYPE values: " + ", ".join(PII_TYPES) + ".\n"
    "Repeated entities of the same type may be numbered, e.g. [GIVENNAME1], [GIVENNAME2]. "
    "Output only the anonymized text."
)
USER_TEMPLATE = "Anonymize the following text:\n\n{source}"


def base_type(label: str) -> str:
    return label.rstrip("0123456789")


def build_messages(source: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(source=source)},
    ]


# ───────────────────────── config ─────────────────────────
def _set_dotted(cfg: Dict, dotted: str, value: Any):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | None = None, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or ():
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k, yaml.safe_load(v))
    cfg["_run_dir"] = str(Path(cfg["run_dir"]).expanduser().resolve())
    if cfg.get("force_cpu"):
        os.environ["PII_RL_FORCE_CPU"] = "1"
    return cfg


def deep_copy(cfg):
    return copy.deepcopy(cfg)


# ───────────────────────── paths ─────────────────────────
class Paths:
    def __init__(self, cfg):
        self.root = Path(cfg["_run_dir"])
        self.data = self.root / "data"
        self.models = self.root / "models"
        self.preds = self.root / "predictions"
        self.cands = self.root / "candidates"
        self.results = self.root / "results"
        self.logs = self.root / "logs"
        self.markers = self.root / "markers"
        for p in [self.data, self.models, self.preds, self.cands, self.results, self.logs, self.markers]:
            p.mkdir(parents=True, exist_ok=True)

    def split(self, name):
        return self.data / f"{name}.jsonl"

    def model(self, name):
        return self.models / name

    def pred(self, name):
        return self.preds / f"{name}.jsonl"


# ───────────────────────── markers ─────────────────────────
def is_done(paths: Paths, stage: str) -> bool:
    return (paths.markers / f"{stage}.done").exists()


def mark_done(paths: Paths, stage: str, info: Dict | None = None):
    (paths.markers / f"{stage}.done").write_text(
        json.dumps({"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})}, indent=2))


# ───────────────────────── io ─────────────────────────
def read_jsonl(p) -> List[Dict]:
    if not Path(p).exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # partially written last line after a crash
                    pass
    return out


def write_jsonl(p, rows: Iterable[Dict]):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def append_jsonl(p, rows: Iterable[Dict]):
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ───────────────────────── misc ─────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(paths: Paths | None, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if paths is not None:
        fh = logging.FileHandler(paths.logs / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def torch_dtype(cfg):
    import torch
    want = cfg.get("precision", "auto")
    if cfg.get("force_cpu"):
        return torch.float32
    if want == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[want]


def mixed_precision_flags(cfg):
    """bf16/fp16 flags for HF TrainingArguments derived from the chosen dtype."""
    import torch
    if cfg.get("force_cpu"):
        return {"bf16": False, "fp16": False, "use_cpu": True}
    dt = torch_dtype(cfg)
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    return {"bf16": dt == torch.bfloat16, "fp16": dt == torch.float16}


def last_checkpoint(output_dir: Path):
    if not output_dir.exists():
        return None
    cks = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(cks[-1]) if cks else None


def env_report() -> Dict[str, Any]:
    info = {"python": sys.version.split()[0]}
    for mod in ["torch", "transformers", "trl", "peft", "datasets", "accelerate"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info
