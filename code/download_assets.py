"""Pre-download models and the dataset (run once on a node WITH internet).
Afterwards set HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 for jobs on offline compute nodes.

    python download_assets.py [--extra Qwen/Qwen2.5-1.5B-Instruct]
"""
import argparse

from datasets import load_dataset
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--extra", nargs="*", default=[])
a = ap.parse_args()
for repo in ["Qwen/Qwen2.5-0.5B-Instruct", "microsoft/deberta-v3-base", *a.extra]:
    print("downloading", repo)
    snapshot_download(repo)
print("downloading dataset")
load_dataset("ai4privacy/pii-masking-300k")
print("ok")
