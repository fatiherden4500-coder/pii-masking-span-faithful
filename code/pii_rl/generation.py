"""Model loading and resumable batched generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .common import append_jsonl, build_messages, read_jsonl, torch_dtype, write_jsonl


def device_name():
    import os
    import torch
    if os.environ.get("PII_RL_FORCE_CPU") == "1":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(name_or_path, padding_side="left"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy(cfg, base: str, adapter: Optional[str] = None, merge: bool = True):
    """Load base (HF id or merged dir) + optional LoRA adapter (merged for fast inference)."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch_dtype(cfg))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    model.to(device_name())
    model.eval()
    tok = load_tokenizer(adapter or base)
    return model, tok


def merge_and_save(cfg, base: str, adapter: str, out_dir: Path):
    """Write a merged full model (used as the starting point / reference for DPO, GRPO, RLOO)."""
    if (out_dir / "config.json").exists():
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch_dtype(cfg))
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    model.save_pretrained(tmp)
    load_tokenizer(adapter).save_pretrained(tmp)
    tmp.rename(out_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prompt_text(tok, source):
    return tok.apply_chat_template(build_messages(source), tokenize=False, add_generation_prompt=True)


def generate_to_file(model, tok, records: List[Dict], out_path: Path, *, n: int = 1,
                     temperature: float = 0.0, top_p: float = 1.0, max_new_tokens: int = 448,
                     batch_size: int = 32, log=None, extra: Optional[Dict] = None):
    """Generate for every record; resumable (append-only partial file, then finalized)."""
    import torch
    out_path = Path(out_path)
    final_marker = out_path.with_suffix(".complete")
    if final_marker.exists():
        return read_jsonl(out_path)
    partial = out_path.with_suffix(".partial.jsonl")
    done = {r["idx"] for r in read_jsonl(partial)}
    todo = [(i, r) for i, r in enumerate(records) if i not in done]
    # long prompts first -> OOM shows up early; similar lengths batch together
    todo.sort(key=lambda x: -len(x[1]["source_text"]))
    if log:
        log.info(f"generate -> {out_path.name}: {len(done)} done, {len(todo)} to go (n={n}, T={temperature})")
    dev = model.device
    bs = batch_size
    k = 0
    while k < len(todo):
        chunk = todo[k:k + max(1, bs // max(n, 1))]
        prompts = [prompt_text(tok, r["source_text"]) for _, r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                          num_return_sequences=n)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        try:
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            if log:
                log.info(f"OOM -> batch_size {bs}")
            continue
        plen = enc["input_ids"].shape[1]
        texts = tok.batch_decode(out[:, plen:], skip_special_tokens=True)
        hit_limit = [(seq[plen:] != tok.pad_token_id).sum().item() >= max_new_tokens for seq in out]
        rows = []
        for j, (i, r) in enumerate(chunk):
            cands = [t.strip() for t in texts[j * n:(j + 1) * n]]
            row = {"idx": i, "id": r.get("id")}
            if n == 1:
                row["prediction"] = cands[0]
                row["hit_max_tokens"] = bool(hit_limit[j])
            else:
                row["candidates"] = cands
            rows.append(row)
        append_jsonl(partial, rows)
        k += len(chunk)
        if log and (k // len(chunk)) % 10 == 0:
            log.info(f"  {k}/{len(todo)}")

    rows = {r["idx"]: r for r in read_jsonl(partial)}
    final = []
    for i, r in enumerate(records):
        row = dict(rows[i])
        row.update({"source_text": r["source_text"], "target_text": r["target_text"]})
        if extra:
            row.update(extra)
        final.append(row)
    write_jsonl(out_path, final)
    final_marker.write_text("ok")
    partial.unlink(missing_ok=True)
    return final
