#!/usr/bin/env bash
# Full experiment driver. Safe to re-run: finished stages are skipped, interrupted
# training resumes from the last checkpoint, interrupted generation resumes per sample.
#
#   bash run_all.sh [config.yaml] [phase]
#   phase: all | baselines | sft | bon | dpo | grpo | rloo | ablations | eval | scaling
set -euo pipefail
cd "$(dirname "$0")"
CFG="${1:-configs/default.yaml}"
PHASE="${2:-all}"
PY="${PYTHON:-python}"
SEEDS=$($PY -c "import yaml,sys;print(' '.join(map(str,yaml.safe_load(open('$CFG'))['seeds'])))")
run() { echo -e "\n>>> $*"; $PY -m pii_rl.cli "$@" --config "$CFG"; }
want() { [[ "$PHASE" == "all" || "$PHASE" == "$1" ]]; }

$PY -m pii_rl.cli env --config "$CFG"
run prepare_data

if want baselines; then
  run zeroshot
  run tokclf --seed 1
  run presidio || true
fi
if want sft; then
  for s in $SEEDS; do run sft --seed $s; run predict --method sft --seed $s; done
  # data-matched control (train + rl_pool labels, same data RL methods see)
  for s in $SEEDS; do run sft_plus --seed $s; run predict --method sft_plus --seed $s; done
fi
if want bon; then
  run tokclf --seed 1          # verifier
  for s in $SEEDS; do run bon --seed $s; done
fi
if want dpo; then
  for s in $SEEDS; do run dpo_pairs --seed $s; run dpo --seed $s; run predict --method dpo --seed $s; done
fi
if want grpo; then
  for s in $SEEDS; do run grpo --seed $s; run predict --method grpo --seed $s; done
fi
if want rloo; then
  for s in $SEEDS; do run rloo --seed $s; run predict --method rloo --seed $s; done
fi
if want ablations; then
  S1=$(echo $SEEDS | awk '{print $1}')
  run grpo --seed $S1 --name grpo_noleak --set reward.w_protect=0
  run predict --method grpo_noleak --seed $S1
  run grpo --seed $S1 --name grpo_nopres --set reward.w_preserve=0
  run predict --method grpo_nopres --seed $S1
  run grpo --seed $S1 --name grpo_nokl --set grpo.beta=0.0
  run predict --method grpo_nokl --seed $S1
  # spurious-reward controls (Qwen2.5 can improve under random/format-only rewards)
  run grpo --seed $S1 --name grpo_random --set reward.control=random
  run predict --method grpo_random --seed $S1
  run grpo --seed $S1 --name grpo_format --set reward.control=format
  run predict --method grpo_format --seed $S1
fi
if [[ "$PHASE" == "scaling" ]]; then
  # SFT data-scaling curve on the SAME test set: 1k and 16k here; 4k = runs/main, ~33k = configs/full_data.yaml
  for N in 1000 16000; do
    O="--set run_dir=./runs/scale_$N --set data.n_train=$N"
    $PY -m pii_rl.cli prepare_data --config "$CFG" $O
    $PY -m pii_rl.cli sft --seed 1 --config "$CFG" $O
    $PY -m pii_rl.cli predict --method sft --seed 1 --config "$CFG" $O
    $PY -m pii_rl.cli evaluate --config "$CFG" $O
  done
  exit 0
fi
run evaluate
echo "DONE. Results: $(grep run_dir "$CFG" | awk '{print $2}')/results/main_table.md"
