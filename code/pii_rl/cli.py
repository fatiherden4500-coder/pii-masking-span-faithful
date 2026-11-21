"""Command-line entry point.

Examples
  python -m pii_rl.cli prepare_data
  python -m pii_rl.cli sft --seed 1
  python -m pii_rl.cli predict --method sft --seed 1
  python -m pii_rl.cli grpo --seed 1 --name grpo_noleak --set reward.w_protect=0
  python -m pii_rl.cli evaluate
Global options: --config path.yaml  --set key=value (repeatable)
"""

import argparse
import json

from .common import Paths, load_config, env_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare_data", "env", "zeroshot", "sft", "sft_plus", "predict", "tokclf", "presidio",
                                      "bon", "dpo_pairs", "dpo", "grpo", "rloo", "evaluate"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--method", default=None)
    ap.add_argument("--name", default=None, help="run name (for ablations)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    Paths(cfg)

    if a.stage == "env":
        print(json.dumps(env_report(), indent=2))
    elif a.stage == "prepare_data":
        from .data import prepare_data
        prepare_data(cfg)
    elif a.stage == "zeroshot":
        from .stages import predict
        predict(cfg, "zeroshot", 0)
    elif a.stage == "sft":
        from .train import train_sft
        train_sft(cfg, a.seed)
    elif a.stage == "sft_plus":
        from .train import train_sft
        train_sft(cfg, a.seed, "sft_plus", ("rl_pool",))
    elif a.stage == "predict":
        from .stages import predict
        predict(cfg, a.method, a.seed)
    elif a.stage == "tokclf":
        from .tokclf import predict_tokclf, train_tokclf
        train_tokclf(cfg, a.seed)
        predict_tokclf(cfg, a.seed)
    elif a.stage == "presidio":
        from .tokclf import predict_presidio
        predict_presidio(cfg)
    elif a.stage == "bon":
        from .stages import bon
        bon(cfg, a.seed)
    elif a.stage == "dpo_pairs":
        from .train import build_dpo_pairs
        build_dpo_pairs(cfg, a.seed)
    elif a.stage == "dpo":
        from .train import train_dpo
        train_dpo(cfg, a.seed, a.name or "dpo")
    elif a.stage in ("grpo", "rloo"):
        from .train import train_rl
        train_rl(cfg, a.seed, a.stage, a.name)
    elif a.stage == "evaluate":
        from .stages import evaluate_all
        evaluate_all(cfg)


if __name__ == "__main__":
    main()
