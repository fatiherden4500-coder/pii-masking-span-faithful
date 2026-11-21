"""Training stages: SFT, DPO (pairs + training), GRPO / RLOO."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .common import (Paths, build_messages, get_logger, is_done, last_checkpoint, mark_done,
                     mixed_precision_flags, read_jsonl, set_seed, torch_dtype, write_jsonl, env_report)
from .metrics import composite_reward


def _lora(cfg):
    from peft import LoraConfig
    m = cfg["model"]
    return LoraConfig(r=m["lora_r"], lora_alpha=m["lora_alpha"], lora_dropout=m["lora_dropout"],
                      target_modules=m["lora_targets"], task_type="CAUSAL_LM")


def _load_lm(cfg, path):
    """Load the policy ourselves (on CPU; Trainer moves it) instead of letting TRL do it.
    Avoids a crash of threaded/direct-to-device loading seen on some platforms."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch_dtype(cfg))
    m.config.use_cache = False
    return m


def _prompt_completion(rows):
    from datasets import Dataset
    return Dataset.from_list([
        {"prompt": build_messages(r["source_text"]),
         "completion": [{"role": "assistant", "content": r["target_text"]}]}
        for r in rows])


def _finish_adapter(trainer, out: Path, tok):
    tmp = out.with_name(out.name + ".tmp")
    trainer.model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    tmp.rename(out)


# ═════════════════════════ SFT ═════════════════════════
def train_sft(cfg, seed: int, name: str = "sft", extra_splits=()):
    """name='sft_plus', extra_splits=('rl_pool',): data-matched control. RL/DPO methods use the
    gold spans of rl_pool for their reward, so SFT on train+rl_pool sees the same labelled data."""
    paths = Paths(cfg)
    stage = f"{name}_s{seed}_train"
    log = get_logger(paths, stage)
    if is_done(paths, stage):
        log.info("already done"); return
    set_seed(seed)
    from trl import SFTConfig, SFTTrainer
    from .generation import load_tokenizer, merge_and_save

    sc = cfg["sft"]
    mdir = paths.model(f"{name}_s{seed}")
    tok = load_tokenizer(cfg["model"]["name"], padding_side="right")
    args = SFTConfig(
        output_dir=str(mdir / "ckpt"), seed=seed, num_train_epochs=sc["epochs"],
        learning_rate=sc["lr"], per_device_train_batch_size=sc["batch_size"],
        per_device_eval_batch_size=sc["batch_size"], gradient_accumulation_steps=sc["grad_accum"],
        warmup_steps=sc["warmup_ratio"], lr_scheduler_type="cosine", max_length=sc["max_length"],
        completion_only_loss=True, eval_strategy="steps", eval_steps=sc["save_steps"],
        save_strategy="steps", save_steps=sc["save_steps"], save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=sc.get("logging_steps", 10), report_to="none", gradient_checkpointing=False,
        **mixed_precision_flags(cfg),
    )
    trainer = SFTTrainer(model=_load_lm(cfg, cfg["model"]["name"]), args=args,
                         train_dataset=_prompt_completion(
                             [r for sp in ("train", *extra_splits) for r in read_jsonl(paths.split(sp))]),
                         eval_dataset=_prompt_completion(read_jsonl(paths.split("val"))),
                         processing_class=tok, peft_config=_lora(cfg))
    trainer.train(resume_from_checkpoint=last_checkpoint(mdir / "ckpt"))
    _finish_adapter(trainer, mdir / "adapter", tok)
    json.dump(trainer.state.log_history, open(mdir / "log_history.json", "w"), indent=1)
    del trainer
    merge_and_save(cfg, cfg["model"]["name"], str(mdir / "adapter"), mdir / "merged")
    mark_done(paths, stage, {"env": env_report()})


# ═════════════════════════ DPO ═════════════════════════
def build_dpo_pairs(cfg, seed: int):
    """On-policy pairs from the SFT policy, ranked by the gold-span reward (train-side labels only)."""
    paths = Paths(cfg)
    stage = f"dpo_pairs_s{seed}"
    log = get_logger(paths, stage)
    if is_done(paths, stage):
        log.info("already done"); return
    set_seed(seed)
    from .generation import generate_to_file, load_policy
    dc = cfg["dpo"]
    pool = read_jsonl(paths.split("rl_pool"))
    random.Random(seed).shuffle(pool)
    pool = pool[:dc["n_prompts"]]
    model, tok = load_policy(cfg, str(paths.model(f"sft_s{seed}") / "merged"))
    rows = generate_to_file(model, tok, pool, paths.cands / f"dpo_pool_s{seed}.jsonl", n=dc["n_samples"],
                            temperature=dc["temperature"], top_p=0.95,
                            max_new_tokens=cfg["generation"]["max_new_tokens"],
                            batch_size=cfg["generation"]["batch_size"], log=log)
    pairs = []
    for rec, row in zip(pool, rows):
        scored = sorted(((composite_reward(rec["source_text"], c, rec["spans"], cfg["reward"]), c)
                         for c in set(row["candidates"])), reverse=True)
        if len(scored) >= 2 and scored[0][0] - scored[-1][0] >= dc["min_margin"]:
            pairs.append({"source_text": rec["source_text"], "chosen": scored[0][1],
                          "rejected": scored[-1][1], "r_chosen": scored[0][0], "r_rejected": scored[-1][0]})
    write_jsonl(paths.data / f"dpo_pairs_s{seed}.jsonl", pairs)
    log.info(f"{len(pairs)} pairs from {len(pool)} prompts")
    mark_done(paths, stage, {"n_pairs": len(pairs)})


def train_dpo(cfg, seed: int, name: str = "dpo"):
    paths = Paths(cfg)
    stage = f"{name}_s{seed}_train"
    log = get_logger(paths, stage)
    if is_done(paths, stage):
        log.info("already done"); return
    set_seed(seed)
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    from .generation import load_tokenizer

    dc = cfg["dpo"]
    base = str(paths.model(f"sft_s{seed}") / "merged")
    pairs = read_jsonl(paths.data / f"dpo_pairs_s{seed}.jsonl")
    ds = Dataset.from_list([{"prompt": build_messages(p["source_text"]),
                             "chosen": [{"role": "assistant", "content": p["chosen"]}],
                             "rejected": [{"role": "assistant", "content": p["rejected"]}]} for p in pairs])
    mdir = paths.model(f"{name}_s{seed}")
    tok = load_tokenizer(base, padding_side="right")
    args = DPOConfig(
        output_dir=str(mdir / "ckpt"), seed=seed, beta=dc["beta"], learning_rate=dc["lr"],
        num_train_epochs=dc["epochs"], per_device_train_batch_size=dc["batch_size"],
        gradient_accumulation_steps=dc["grad_accum"], max_length=dc["max_length"],
        lr_scheduler_type="cosine", warmup_steps=0.05, save_strategy="steps",
        save_steps=dc["save_steps"], save_total_limit=2, logging_steps=10, report_to="none",
        gradient_checkpointing=False,
        **mixed_precision_flags(cfg),
    )
    # ref model = policy with LoRA disabled = merged SFT
    trainer = DPOTrainer(model=_load_lm(cfg, base), args=args, train_dataset=ds, processing_class=tok, peft_config=_lora(cfg))
    trainer.train(resume_from_checkpoint=last_checkpoint(mdir / "ckpt"))
    _finish_adapter(trainer, mdir / "adapter", tok)
    json.dump(trainer.state.log_history, open(mdir / "log_history.json", "w"), indent=1)
    mark_done(paths, stage, {"n_pairs": len(pairs), "env": env_report()})


# ═════════════════════════ GRPO / RLOO ═════════════════════════
def make_reward_fn(weights):
    """weights['control'] (ablation controls for the 'spurious reward' effect reported for Qwen2.5):
       'random' -> reward ~ U(0,1) independent of the output; 'format' -> only -invalid-halluc terms."""
    control = weights.get("control")
    if control == "random":
        import random as _r
        rng = _r.Random(12345)

        def random_reward(prompts, completions, **kwargs):
            return [rng.random() for _ in completions]
        return random_reward
    if control == "format":
        weights = {**weights, "w_protect": 0.0, "w_typed": 0.0, "w_preserve": 0.0}

    def span_reward(prompts, completions, source_text, spans_json, **kwargs):
        out = []
        for comp, src, sj in zip(completions, source_text, spans_json):
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            out.append(float(composite_reward(src, text, json.loads(sj), weights)))
        return out
    span_reward.__name__ = "span_reward"
    return span_reward


def train_rl(cfg, seed: int, algo: str = "grpo", name: str | None = None):
    """algo in {grpo, rloo}. Starts from the merged SFT model; KL reference = merged SFT."""
    name = name or algo
    paths = Paths(cfg)
    stage = f"{name}_s{seed}_train"
    log = get_logger(paths, stage)
    if is_done(paths, stage):
        log.info("already done"); return
    set_seed(seed)
    from datasets import Dataset
    from .generation import load_tokenizer

    rc = cfg[algo]
    base = str(paths.model(f"sft_s{seed}") / "merged")
    pool = read_jsonl(paths.split("rl_pool"))
    ds = Dataset.from_list([{"prompt": build_messages(r["source_text"]), "source_text": r["source_text"],
                             "spans_json": json.dumps(r["spans"])} for r in pool])
    mdir = paths.model(f"{name}_s{seed}")
    tok = load_tokenizer(base, padding_side="left")
    common = dict(
        output_dir=str(mdir / "ckpt"), seed=seed, max_steps=rc["max_steps"], learning_rate=rc["lr"],
        per_device_train_batch_size=rc["per_device_batch"], gradient_accumulation_steps=rc["grad_accum"],
        num_generations=rc["num_generations"], beta=rc["beta"], temperature=rc["temperature"],
        max_completion_length=rc["max_completion_length"], lr_scheduler_type="constant_with_warmup",
        warmup_steps=10, save_strategy="steps", save_steps=rc["save_steps"], save_total_limit=2,
        logging_steps=rc["logging_steps"], report_to="none", gradient_checkpointing=False,
        log_completions=False, **mixed_precision_flags(cfg),
    )
    if algo == "grpo":
        from trl import GRPOConfig as C, GRPOTrainer as T
        common["loss_type"] = rc.get("loss_type", "dapo")
    else:
        from trl import RLOOConfig as C, RLOOTrainer as T
    args = C(**common)
    trainer = T(model=_load_lm(cfg, base), reward_funcs=[make_reward_fn(cfg["reward"])], args=args, train_dataset=ds,
                processing_class=tok, peft_config=_lora(cfg))
    trainer.train(resume_from_checkpoint=last_checkpoint(mdir / "ckpt"))
    _finish_adapter(trainer, mdir / "adapter", tok)
    json.dump(trainer.state.log_history, open(mdir / "log_history.json", "w"), indent=1)
    json.dump({"reward": cfg["reward"], algo: rc}, open(mdir / "run_config.json", "w"), indent=1)
    mark_done(paths, stage, {"env": env_report()})
