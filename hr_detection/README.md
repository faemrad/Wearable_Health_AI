# Heart-rate PoC (ECGFounder 1-lead + HuBERT-ECG small)

Fine-tune two ECG models to **regress heart rate (bpm)** from PTB-XL 10-second windows.

- **ECGFounder** — 1-lead (Lead II), linear probe by default.
- **HuBERT-ECG small** — 12-lead, last 2 transformer blocks unfrozen by default.

Labels come from NeuroKit2 R-peaks on Lead II (not the PTB-XL spreadsheet). Records that fail peak detection are dropped.

**You run these commands.** Conda env `FIT4701`. **cwd = repo root** `Wearable_Health_AI`.

---

## Folder layout

```text
hr_detection/
  README.md                 ← this file
  core/                     ← library (imported, not run)
    paths.py device.py preprocess.py datasets.py models.py metrics.py
  scripts/                  ← CLIs
    prepare_ptbxl.py
    download_weights.py
    train_ecgfounder.py
    eval_founder.py         ← % within ±N bpm after Founder train
    train_hubert.py
    eval_hubert.py          ← same % metrics for HuBERT
  checkpoints/              ← your trained HR weights (written by train)
```

Data and vendor code live **outside** this folder:


| Path                                       | Role                                               |
| ------------------------------------------ | -------------------------------------------------- |
| `data/ptb-xl/`                             | PTB-XL dump (`ptbxl_database.csv` + `records100/`) |
| `data/hr_detection/cache/*.npy`                  | 12-lead windows at 500 Hz (written by prepare)     |
| `data/hr_detection/manifest.csv`                 | stem, npy path, `hr_bpm`, `patient_id`, `split`    |
| `models/ECGFounder/`                       | vendor `net1d.py`                                  |
| `models/HuBERT_ECG/`                       | vendor package (`pip install -e`)                  |
| `models/checkpoints/1_lead_ECGFounder.pth` | Hugging Face **pretrain** backbone                 |
| `hr_detection/checkpoints/ecgfounder_hr.pth`     | **your** HR head after train                       |


Always:

```bash
python -m hr_detection.scripts.<name>
```

Do **not** `cd hr_detection` and run files from there.

---



## 0. One-time env

```bash
conda activate FIT4701
cd /Users/mayankbajpai/Dev/Other/FYP/Wearable_Health_AI

pip install transformers safetensors loguru biosppy
pip install -e models/HuBERT_ECG
```

HuBERT train fails with `ModuleNotFoundError: hubert_ecg` until the editable install succeeds.

Optional if Hugging Face rate-limits you:

```bash
huggingface-cli login
```

---



## 1. Confirm PTB-XL

Prepare reads **only** `records100` (100 Hz `_lr` files). A record is used only if **both** `.hea` and `.dat` exist.

```bash
ls data/ptb-xl/ptbxl_database.csv
ls data/ptb-xl/records100
find data/ptb-xl/records100 -name '*_lr.dat' | wc -l
```

If `ptbxl_database.csv` is missing, copy it from PhysioNet PTB-XL 1.0.3 into `data/ptb-xl/`.

---



## 2. Build cache + manifest

```bash
python -m hr_detection.scripts.prepare_ptbxl
```

Walks `records100`, upsamples to 500 Hz, estimates bpm, writes `data/hr_detection/cache/<stem>.npy`, patient-grouped 80/10/10 split → `data/hr_detection/manifest.csv`.

Later runs **no-op** if the manifest exists. Rebuild:

```bash
python -m hr_detection.scripts.prepare_ptbxl --force
```

Success: split counts + `Wrote .../manifest.csv n=<N> mean_hr=<bpm>`.

---



## 3. Download ECGFounder **pretrain** weights

This is **not** your HR model. It is the published 1-lead backbone.

```bash
python -m hr_detection.scripts.download_weights
```

Expect: `models/checkpoints/1_lead_ECGFounder.pth`. No-op if already there.

---



## 4. Train ECGFounder HR head

```bash
python -m hr_detection.scripts.train_ecgfounder
```

Defaults: `--epochs 8 --batch_size 32 --lr 1e-3` (linear probe). Full fine-tune:

```bash
python -m hr_detection.scripts.train_ecgfounder --full_ft --lr 1e-4
```

Device: MPS → CUDA → CPU.

End of train prints **MAE/RMSE only** (bpm vs NeuroKit), e.g. `TEST founder MAE=4.49 RMSE=7.25`.

Writes:

- `hr_detection/checkpoints/ecgfounder_hr.pth`
- `hr_detection/checkpoints/ecgfounder_hr.safetensors`

---



## 5. Eval = the **%** numbers (run this after train)

Train does **not** print percent-correct. HR is regression. `eval_founder` scores the **saved** checkpoint on a manifest split (default **test**).

**“Accuracy” here = share of windows whose |pred − NeuroKit bpm| is within ±5 bpm (and ±10).** Not clinical accuracy. Same PTB-XL patients the split already held out.

```bash
python -m hr_detection.scripts.eval_founder
```

Takes about one test pass (minutes). Prints something like:

```text
device mps
ckpt .../hr_detection/checkpoints/ecgfounder_hr.pth
split test
n=2153
MAE=4.49 bpm
RMSE=7.25 bpm
MAPE=6.0%
within ±5 bpm = 72.0%
within ±10 bpm = 91.0%
```

(`MAE`/`RMSE` should match train’s `TEST founder` line; the `%` lines are extra.)

Optional flags:

```bash
python -m hr_detection.scripts.eval_founder --split val
python -m hr_detection.scripts.eval_founder --within 5 10 15
python -m hr_detection.scripts.eval_founder --ckpt hr_detection/checkpoints/ecgfounder_hr.pth
```


| Metric         | Meaning                                  |
| -------------- | ---------------------------------------- |
| MAE / RMSE     | bpm error vs NeuroKit                    |
| MAPE           | mean |err| / true_hr × 100               |
| within ±5 bpm  | **the % to quote** if you need a percent |
| within ±10 bpm | looser band                              |


Report wording: *“X% of held-out PTB-XL windows within ±5 bpm of NeuroKit HR.”*

Needs `data/hr_detection/manifest.csv` **and** `hr_detection/checkpoints/ecgfounder_hr.pth`. Do **not** pass the Hugging Face backbone as `--ckpt` (that has no HR head).

---



## 6. Train HuBERT-ECG small (optional, heavier)

Needs step 2 and `pip install -e models/HuBERT_ECG`. First run downloads `Edoardo-Coppola/hubert-ecg-small`.

```bash
python -m hr_detection.scripts.train_hubert
```

OOM:

```bash
python -m hr_detection.scripts.train_hubert --batch_size 2 --epochs 3
```

Checkpoint: `hr_detection/checkpoints/hubert_hr/` (a **folder**, not a `.pth`).

**% eval** (same metrics as Founder). OOM → `--batch_size 2`.

```bash
python -m hr_detection.scripts.eval_hubert
python -m hr_detection.scripts.eval_hubert --batch_size 2
```

Do **not** pass the Hugging Face `hubert-ecg-small` id as `--ckpt`. That is the pretrain, not your HR head.

---



## Command map


| Step               | Command                                     |
| ------------------ | ------------------------------------------- |
| Prepare            | `python -m hr_detection.scripts.prepare_ptbxl`    |
| Pretrain weights   | `python -m hr_detection.scripts.download_weights` |
| Train Founder      | `python -m hr_detection.scripts.train_ecgfounder` |
| **% eval Founder** | `python -m hr_detection.scripts.eval_founder`     |
| Train HuBERT       | `python -m hr_detection.scripts.train_hubert`     |
| **% eval HuBERT**  | `python -m hr_detection.scripts.eval_hubert`      |


---



## Common failures


| Symptom                                      | Fix                                             |
| -------------------------------------------- | ----------------------------------------------- |
| `Missing .../data/ptb-xl/records100`         | Put 100 Hz records there                        |
| `FileNotFoundError` on `ptbxl_database.csv`  | Copy CSV into `data/ptb-xl/`                    |
| `Exists .../manifest.csv`                    | Expected; `--force` to rebuild                  |
| `No labelled windows`                        | Matching `.dat` + peakable Lead II              |
| `FileNotFoundError` on `ecgfounder_hr.pth`   | Finish train first                              |
| `ModuleNotFoundError: hubert_ecg`            | `pip install -e models/HuBERT_ECG`              |
| `No module named 'net1d'`                    | `models/ECGFounder/net1d.py` must exist         |
| Hugging Face 401 / gated                     | `huggingface-cli login`                         |
| MPS OOM on HuBERT                            | `--batch_size 2`                                |
| Old path `python -m poc_hr.scripts...` | Use `python -m hr_detection.scripts...` |


