# Heart-rate PoC — PPG-Sport (ECGFounder 1-lead)

Fine-tune the **same 1-lead ECGFounder** as the PTB-XL HR PoC, but on **wrist PPG** from PPG-Sport. Task is still **regress bpm**.

**PPG foundation model:** `train_pulseppg` / `eval_pulseppg` — [Pulse-PPG](https://github.com/maxxu05/pulseppg) ResNet, pretrained on field wrist PPG, resampled to 50 Hz. See **§5**.

- Input: 8 s PPG, upsampled 64 Hz → 500 Hz, ECGFounder 1-lead filter + z-score, shape `(1, 4000)`.
- Labels: `cleaned_gt` ECG-reference HR (not NeuroKit on the PPG). Interpolated to the 10 s window centre.
- Split: **patient-grouped** 80/10/10 (left + right wrist of one subject stay in one split).
- Default train: **full fine-tune** (`--linear_prob` exists but is usually worse — frozen ECG filters).

Do **not** mix with PTB-XL files. This path uses `data/hr_detection_ppg/` and `ppg_ecgfounder_hr.pth`.

**You run these commands.** Conda env `FIT4701`. **cwd = repo root** `Wearable_Health_AI`.

---

## Folder layout

```text
hr_detection/
  README.md                 ← ECG / PTB-XL PoC
  README_PPG.md             ← this file
  scripts/
    prepare_ppg.py
    train_ppgfounder.py
    eval_ppgfounder.py      ← % within ±N bpm after PPG train
    train_pulseppg.py       ← Pulse-PPG fine-tune (50 Hz)
    eval_pulseppg.py        ← same % metrics for Pulse-PPG
```

| Path | Role |
| ---- | ---- |
| `data/PPG-Sport/dataset/*_ppg.npy` | 64 Hz wrist PPG, 8 min |
| `data/PPG-Sport/cleaned_gt/*_hr.npy` | ECG-derived bpm (8 s / hop 2 s) |
| `data/hr_detection_ppg/cache/*.npy` | 10 s windows at 500 Hz (written by prepare) |
| `data/hr_detection_ppg/manifest.csv` | stem, npy, `hr_bpm`, `patient_id`, `hand`, `activity`, `split` |
| `models/PulsePPG/` | Pulse-PPG submodule |
| `models/checkpoints/1_lead_ECGFounder.pth` | Hugging Face **pretrain** backbone (same as ECG) |
| `models/checkpoints/pulseppg/checkpoint_best.pkl` | Pulse-PPG **pretrain** encoder |
| `hr_detection/checkpoints/ppg_ecgfounder_hr.pth` | **your** ECGFounder PPG HR weights |
| `hr_detection/checkpoints/ppg_pulseppg_hr.pth` | **your** Pulse-PPG HR weights |

Always:

```bash
python -m hr_detection.scripts.<name>
```

---

## 0. One-time env

Same as `README.md` (ECG). Need the ECGFounder pretrain weights:

```bash
conda activate FIT4701
cd /Users/mayankbajpai/Dev/Other/FYP/Wearable_Health_AI

python -m hr_detection.scripts.download_weights
```

Expect: `models/checkpoints/1_lead_ECGFounder.pth`.

---

## 1. Confirm PPG-Sport

```bash
ls data/PPG-Sport/dataset | head
ls data/PPG-Sport/cleaned_gt | head
find data/PPG-Sport/dataset -name '*_ppg.npy' | wc -l
```

Need `*_ppg.npy` under `dataset/` and `Sxx_<activity>_hr.npy` under `cleaned_gt/`.

---

## 2. Build cache + manifest

```bash
python -m hr_detection.scripts.prepare_ppg
```

Walks every `*_ppg.npy`, resamples to 500 Hz, cuts **8 s** windows with hop **2 s** (PPG-Sport `cleaned_gt` protocol), labels = GT bin, patient-grouped 80/10/10 **and** 6 subject folds → `data/hr_detection_ppg/manifest.csv`.

Later runs **no-op** if the manifest exists. Rebuild (wipes cache):

```bash
python -m hr_detection.scripts.prepare_ppg --force
```

Old 10 s / hop 5 s protocol:

```bash
python -m hr_detection.scripts.prepare_ppg --force --win_sec 10 --hop_sec 5
```

Success: split counts + `Wrote .../hr_detection_ppg/manifest.csv n=<N> mean_hr=<bpm>`.

---

## 3. Train

```bash
python -m hr_detection.scripts.train_ppgfounder
```

Defaults: `--epochs 8 --batch_size 32 --lr 1e-4` **full fine-tune**.

Subject 6-fold CV (~2–3 h on MPS; writes `ppg_ecgfounder_hr_fold{0-5}.pth`):

```bash
python -m hr_detection.scripts.train_ppgfounder --cv 6
```

Defaults: `--epochs 8 --batch_size 32 --lr 1e-4` **full fine-tune**. Linear probe only:

```bash
python -m hr_detection.scripts.train_ppgfounder --linear_prob
```

End of train prints **MAE/RMSE only**, e.g. `TEST ppg founder MAE=... RMSE=...`.

Writes:

- `hr_detection/checkpoints/ppg_ecgfounder_hr.pth`
- `hr_detection/checkpoints/ppg_ecgfounder_hr.safetensors`

Does **not** overwrite `ecgfounder_hr.pth` from the ECG run.

---

## 4. Eval = the **%** numbers

Train does **not** print percent-correct. `eval_ppgfounder` scores the **saved PPG checkpoint** on the PPG manifest (default **test**).

**“Accuracy” = share of windows whose |pred − ECG-reference bpm| is within ±5 bpm (and ±10).** Not clinical accuracy. Held-out **subjects**.

```bash
python -m hr_detection.scripts.eval_ppgfounder --smooth 5 --baselines
```

Prints raw + smoothed overall metrics, **per-activity MAE**, constant-mean and FFT-peak baselines.

CV fold (after `--cv 6` train):

```bash
python -m hr_detection.scripts.eval_ppgfounder --fold 0 --ckpt hr_detection/checkpoints/ppg_ecgfounder_hr_fold0.pth --smooth 5 --baselines
```

Prints something like:

```text
device mps
ckpt .../hr_detection/checkpoints/ppg_ecgfounder_hr.pth
split test
n=...
MAE=... bpm
RMSE=... bpm
MAPE=...%
within ±5 bpm = ...%
within ±10 bpm = ...%
```

Optional:

```bash
python -m hr_detection.scripts.eval_ppgfounder --split val
python -m hr_detection.scripts.eval_ppgfounder --within 5 10 15
python -m hr_detection.scripts.eval_ppgfounder --ckpt hr_detection/checkpoints/ppg_ecgfounder_hr.pth
```

Report wording: *“X% of held-out PPG-Sport windows within ±5 bpm of ECG-reference HR.”*

Needs `data/hr_detection_ppg/manifest.csv` **and** `ppg_ecgfounder_hr.pth`. Do **not** pass `ecgfounder_hr.pth` (PTB-XL) or the Hugging Face backbone as `--ckpt`.

Do **not** run `eval_founder` on this run — that script still loads the **ECG** manifest.

---

## 5. Pulse-PPG (PPG foundation model)

Same manifest and subject split as ECGFounder. Windows stay the 500 Hz cache; the loader resamples each window to **50 Hz** (Pulse-PPG pretrain rate). Shape `(1, 400)` for an 8 s window. The ResNet's own InstanceNorm is the normaliser.

Pretrain weights download on the first train (Zenodo, ~300 MB) into `models/checkpoints/pulseppg/checkpoint_best.pkl`.

```bash
python -m hr_detection.scripts.download_weights --pulseppg
python -m hr_detection.scripts.train_pulseppg
```

Defaults: `--epochs 8 --batch_size 32 --lr 1e-4` **full fine-tune**. Linear probe (frozen encoder):

```bash
python -m hr_detection.scripts.train_pulseppg --linear_prob
```

Subject 6-fold CV:

```bash
python -m hr_detection.scripts.train_pulseppg --cv 6
```

Writes `hr_detection/checkpoints/ppg_pulseppg_hr.pth`. End of train prints test MAE / smooth5 / ±5%.

Eval uses the same wording as Founder PPG:

```bash
python -m hr_detection.scripts.eval_pulseppg --smooth 5 --baselines
```

Quote: *“X% of held-out PPG-Sport windows within ±5 bpm of ECG-reference HR”* after Pulse-PPG.

CV fold:

```bash
python -m hr_detection.scripts.eval_pulseppg --fold 0 --ckpt hr_detection/checkpoints/ppg_pulseppg_hr_fold0.pth --smooth 5 --baselines
```

Needs `data/hr_detection_ppg/manifest.csv` (from `prepare_ppg`) and the Pulse-PPG submodule (`git submodule update --init models/PulsePPG`).

---

## Command map

| Step | Command |
| ---- | ------- |
| Pretrain weights | `python -m hr_detection.scripts.download_weights` |
| Pulse-PPG weights | `python -m hr_detection.scripts.download_weights --pulseppg` |
| Prepare | `python -m hr_detection.scripts.prepare_ppg` |
| Train Founder PPG | `python -m hr_detection.scripts.train_ppgfounder` |
| **% eval Founder** | `python -m hr_detection.scripts.eval_ppgfounder` |
| Train Pulse-PPG | `python -m hr_detection.scripts.train_pulseppg` |
| **% eval Pulse-PPG** | `python -m hr_detection.scripts.eval_pulseppg --smooth 5 --baselines` |

---

## Common failures

| Symptom | Fix |
| ------- | --- |
| `Missing .../data/PPG-Sport/dataset` | Put PPG-Sport there (`dataset/` + `cleaned_gt/`) |
| `No *_ppg.npy` | Processed `.npy` dump, not only `eval/` CSVs |
| `Exists .../hr_detection_ppg/manifest.csv` | Expected; `--force` to rebuild |
| `Empty train split` / `No labelled PPG windows` | Matching `cleaned_gt` HR files |
| `FileNotFoundError` on `ppg_ecgfounder_hr.pth` | Finish `train_ppgfounder` first |
| `FileNotFoundError` on `ppg_pulseppg_hr.pth` | Finish `train_pulseppg` first |
| `FileNotFoundError` on `1_lead_ECGFounder.pth` | `download_weights` |
| Pulse-PPG weight mismatch | `download_weights --pulseppg` (stale or partial zip) |
| MPS OOM | `--batch_size 8` |

---

## Not in this PoC

- HuBERT-ECG (12-lead). Skip.
- IMU fusion. This path stays PPG-only.
- `program/` CSV ingest. These CLIs read PPG-Sport `.npy` only.
