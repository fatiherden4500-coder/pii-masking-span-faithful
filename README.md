# Span-faithful evaluation of SFT, Best-of-N, DPO, GRPO and RLOO for generative PII masking

Code, data splits, model predictions and evaluation outputs accompanying the article
*"Does Preference and Reinforcement Learning Optimisation Help Generative PII Masking? A Controlled Comparison under Span-Faithful Evaluation"*.

| Folder | Content |
|---|---|
| `code/` | Full pipeline: data preparation (document-level splits), SFT / BoN / DPO / GRPO / RLOO training, DeBERTa baseline, span-faithful metrics (`code/pii_rl/metrics.py`), document-cluster bootstrap. See `code/README.md`. |
| `splits/` | Document-disjoint splits of the English portion of `ai4privacy/pii-masking-300k` with gold spans: `test` (1,000), `val` (500), `rl_pool` (3,000), `train` (4,000). |
| `predictions/` | Test-set outputs of every system and seed reported in the article (`<method>__s<seed>.jsonl`; fields `idx`, `id`, `prediction`; source and reference texts are in `splits/test.jsonl`, aligned by `idx`). |
| `results/` | Evaluation outputs (`evaluation_vs_sft*.json`) and the main table. |
| `training_logs/` | Training curves (Hugging Face `log_history`) of all trained models. |

## Re-scoring the predictions (CPU only)
```bash
cd code && pip install -r requirements.txt
python -m pytest tests -q
mkdir -p runs/main && cp -r ../splits runs/main/data && cp -r ../predictions runs/main/predictions
python -m pii_rl.cli evaluate        # writes runs/main/results/main_table.md
```

## Reproducing training
`bash code/run_all.sh code/configs/default.yaml` (one 32 GB GPU; resumable). Data are synthetic and come from `ai4privacy/pii-masking-300k` (see its licence on the Hugging Face Hub).

## Licence
Code: MIT. Splits and predictions are derived from ai4privacy/pii-masking-300k and inherit its licence.
