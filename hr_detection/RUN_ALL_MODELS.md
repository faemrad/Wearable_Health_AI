# Run all four HR models

Two ECG models on PTB-XL, two PPG models on PPG-Sport. Details and flags: `README.md` (ECG) and `README_PPG.md` (PPG).

| # | Model | Data | Train | Eval |
| - | ----- | ---- | ----- | ---- |
| 1 | ECGFounder | PTB-XL ECG | `train_ecgfounder` | `eval_founder` |
| 2 | HuBERT-ECG | PTB-XL ECG | `train_hubert` | `eval_hubert` |
| 3 | ECGFounder | PPG-Sport PPG | `train_ppgfounder` | `eval_ppgfounder` |
| 4 | Pulse-PPG | PPG-Sport PPG | `train_pulseppg` | `eval_pulseppg` |

---

## 0. Setup (once)

```bash
conda activate FIT4701
cd /Users/mayankbajpai/Dev/Other/FYP/Wearable_Health_AI

git submodule update --init --recursive
pip install transformers safetensors loguru biosppy
pip install -e models/HuBERT_ECG

python -m hr_detection.scripts.download_weights
python -m hr_detection.scripts.download_weights --pulseppg
```

Data must already be in place:

- `data/ptb-xl/ptbxl_database.csv` and `data/ptb-xl/records100/`
- `data/PPG-Sport/dataset/*_ppg.npy` and `data/PPG-Sport/cleaned_gt/*_hr.npy`

---

## 1. Build the manifests (once)

```bash
python -m hr_detection.scripts.prepare_ptbxl
python -m hr_detection.scripts.prepare_ppg
```

Both no-op if the manifest exists. Add `--force` to rebuild.

---

## 2. ECG models (PTB-XL)

```bash
python -m hr_detection.scripts.train_ecgfounder
python -m hr_detection.scripts.eval_founder

python -m hr_detection.scripts.train_hubert
python -m hr_detection.scripts.eval_hubert
```

HuBERT out of memory: add `--batch_size 2` to both HuBERT commands.

---

## 3. PPG models (PPG-Sport)

```bash
python -m hr_detection.scripts.train_ppgfounder
python -m hr_detection.scripts.eval_ppgfounder --smooth 5 --baselines

python -m hr_detection.scripts.train_pulseppg
python -m hr_detection.scripts.eval_pulseppg --smooth 5 --baselines
```

Pulse-PPG frozen-encoder baseline: `train_pulseppg --linear_prob`.

---

## Checkpoints written

| Model | Checkpoint |
| ----- | ---------- |
| ECGFounder (ECG) | `hr_detection/checkpoints/ecgfounder_hr.pth` |
| HuBERT-ECG | `hr_detection/checkpoints/hubert_hr/` |
| ECGFounder (PPG) | `hr_detection/checkpoints/ppg_ecgfounder_hr.pth` |
| Pulse-PPG | `hr_detection/checkpoints/ppg_pulseppg_hr.pth` |

Each eval prints MAE, RMSE, MAPE, and the share of held-out windows within ±5 and ±10 bpm. ECG results are scored against NeuroKit HR; PPG results against ECG-reference HR.
