# v2 — Span-Faithful Comparison of SFT / BoN / DPO / GRPO / RLOO for Generative PII Masking

Self-contained pipeline. Copy this `v2/` folder to the GPU machine; nothing else from the old project is needed.

## 1. Kurulum (GPU sunucu)

```bash
python -m venv ~/venvs/pii && source ~/venvs/pii/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA sürümünüze göre
pip install -r requirements.txt
python download_assets.py            # internetli bir düğümde bir kez
python -m pii_rl.cli env             # GPU ve sürümleri doğrula
python -m pytest tests -q            # metrik birim testleri
bash run_all.sh configs/smoke.yaml   # ~10-15 dk uçtan uca duman testi
```

## 2. Ana deneyler

```bash
bash run_all.sh configs/default.yaml            # her şey (sırayla)
# veya aşama aşama (önerilir; her biri kısa iş):
bash run_all.sh configs/default.yaml baselines
bash run_all.sh configs/default.yaml sft
bash run_all.sh configs/default.yaml bon
bash run_all.sh configs/default.yaml dpo
bash run_all.sh configs/default.yaml grpo
bash run_all.sh configs/default.yaml rloo
bash run_all.sh configs/default.yaml ablations
bash run_all.sh configs/default.yaml eval
```
SLURM: `sbatch slurm_job.sh configs/default.yaml sft` (her aşama ayrı iş).

### Veri ölçeği
| Config | Train | Ne için |
|---|---|---|
| `configs/default.yaml` (`runs/main`) | 4 000 | Ana yöntem karşılaştırması (tüm yöntemler × 3 seed) |
| `configs/full_data.yaml` (`runs/full`) | ~32 900 (kalan **tüm** belgeler) | Tam veri: `baselines`, `sft`, `grpo`, `dpo` aşamaları |
| `bash run_all.sh configs/default.yaml scaling` | 1 000 / 16 000 | SFT veri-ölçekleme eğrisi |

Tüm config'lerde **test (1000), val (500) ve RL havuzu (3000) birebir aynıdır**; 4k eğitim kümesi tam kümenin alt kümesidir → sonuçlar doğrudan karşılaştırılabilir.

**Kesinti olursa aynı komutu tekrar çalıştırın.** Biten aşamalar `runs/<run>/markers/*.done` ile atlanır,
eğitim son checkpoint'ten, üretim kaldığı örnekten devam eder.

`nohup bash run_all.sh configs/default.yaml sft > sft.log 2>&1 &` ile SSH kopsa da sürer (veya `tmux`).

## 3. Geri getirilecek çıktılar

Sadece şu klasörler (model ağırlıkları gerekmez):
```
runs/main/results/        # main_table.md, evaluation.json
runs/main/predictions/    # tüm tahminler (yeniden analiz için)
runs/main/models/*/log_history.json   # eğitim eğrileri
runs/main/markers/        # ortam/sürüm bilgisi, veri istatistikleri
runs/main/logs/
```
`tar czf results_main.tgz runs/main/{results,predictions,markers,logs} runs/main/models/*/log_history.json`

## 4. Tasarım kararları (makale için)

| Konu | Karar |
|---|---|
| Bölme | Belge-düzeyi (ai4privacy chunk id kökü); train 4000 / val 500 / test 1000 / RL havuzu 3000, belgeler ayrık |
| Değerlendirme | Altın span hizalaması: leaked (full/partial), masked-correct, masked-wrong, removed, dropped; + PII-dışı korunum, halüsinasyon, kesilme |
| Ödül | `2·protect(risk-ağırlıklı) + 1·typed + 1.5·preserve − 0.5·invalid − 0.5·halluc` (sadece train/RL havuzu etiketleri) |
| BoN | `bon_oracle` = altın span'lı seçim (**üst sınır**); `bon_verifier` = DeBERTa dedektör span'larıyla seçim (uygulanabilir) |
| DPO | SFT politikasından on-policy örnekler, aynı ödülle sıralama; referans = SFT |
| GRPO / RLOO | TRL; başlangıç ve KL referansı = birleştirilmiş SFT; LoRA |
| Ablation | GRPO: protect=0 / preserve=0 / β=0 |
| İstatistik | 3 seed, ort ± ss; test örnekleri üzerinde eşleştirilmiş bootstrap (tüm seed'ler dahil) |

## 5. Tahmini süre (tek A100 40GB; T4'te ~3–4×)

| Aşama | Süre |
|---|---|
| baselines | ~30 dk |
| sft (×3) | ~1 sa |
| bon (×3) | ~1 sa |
| dpo (×3) | ~2 sa |
| grpo (×3) | ~6–9 sa |
| rloo (×3) | ~4–6 sa |
| ablations | ~6–8 sa |
