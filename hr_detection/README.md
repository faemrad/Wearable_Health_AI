# Heart rate detection

Regress heart rate (bpm) from ECG and wrist PPG using pretrained foundation models. Four models across two datasets:

| Model | Data | Signal | Details |
| ----- | ---- | ------ | ------- |
| ECGFounder 1-lead | PTB-XL | ECG (Lead II) | `README_ECG.md` |
| HuBERT-ECG small | PTB-XL | ECG (12-lead) | `README_ECG.md` |
| ECGFounder 1-lead | PPG-Sport | wrist PPG | `README_PPG.md` |
| Pulse-PPG | PPG-Sport | wrist PPG | `README_PPG.md` |



Full setup and run order for all four: `RUN_EVERYTHING.md`. I recommend you refer to this for testing the repository.



Run everything from the repo root as `python -m hr_detection.scripts.<name>`.

## Layout

```text
hr_detection/
  core/          shared library: datasets, models, preprocessing, metrics, paths, device
  scripts/       CLIs: prepare_*, download_weights, train_*, eval_*, eval_sweep
  checkpoints/   trained weights (ECG: ecgfounder_hr*, hubert_hr*; PPG: ppg_*_hr)
```

## Scripts

- `prepare_ptbxl`, `prepare_ppg`: build the manifests under `data/hr_detection/` and `data/hr_detection_ppg/`.
- `download_weights`: fetch the ECGFounder backbone (`--pulseppg` for the Pulse-PPG encoder).
- `train_*` / `eval_*`: one pair per model (see the per-signal READMEs).
- `eval_sweep`: evaluate every checkpoint in `checkpoints/` and write `eval_results.txt` / `.html` to repo-root `output/`.

## Commands

| Command                                                                 | What it does                                                            |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `pip install -e models/HuBERT_ECG`                                      | Install HuBERT-ECG (needed for the HuBERT scripts)                      |
| `python -m hr_detection.scripts.download_weights`                       | Download the ECGFounder 1-lead backbone                                 |
| `python -m hr_detection.scripts.download_weights --pulseppg`            | Download the Pulse-PPG encoder                                          |
| `python -m hr_detection.scripts.prepare_ptbxl`                          | Build the PTB-XL manifest (`--force` rebuilds)                          |
| `python -m hr_detection.scripts.prepare_ppg`                            | Build the PPG-Sport manifest (`--force` rebuilds)                       |
| `python -m hr_detection.scripts.prepare_test_hr`                        | Write a few ECG and PPG windows to `data/test_data_hr/` for smoke tests |
| `python -m hr_detection.scripts.train_ecgfounder`                       | Train ECGFounder on PTB-XL (`--full_ft --lr 1e-4` for full fine-tune)   |
| `python -m hr_detection.scripts.eval_founder`                           | Evaluate ECGFounder on PTB-XL                                           |
| `python -m hr_detection.scripts.train_hubert`                           | Train HuBERT-ECG on PTB-XL (`--batch_size 2` if out of memory)          |
| `python -m hr_detection.scripts.eval_hubert`                            | Evaluate HuBERT-ECG on PTB-XL                                           |
| `python -m hr_detection.scripts.train_ppgfounder`                       | Train ECGFounder on wrist PPG (`--linear_prob`, `--cv 6`)               |
| `python -m hr_detection.scripts.eval_ppgfounder --smooth 5 --baselines` | Evaluate ECGFounder on PPG                                              |
| `python -m hr_detection.scripts.train_pulseppg`                         | Train Pulse-PPG (`--linear_prob` freezes the encoder)                   |
| `python -m hr_detection.scripts.eval_pulseppg --smooth 5 --baselines`   | Evaluate Pulse-PPG                                                      |
| `python -m hr_detection.scripts.eval_sweep`                             | Evaluate every checkpoint in `checkpoints/`                             |

Eval scripts also take `--split`, `--within`, `--ckpt`, and `--dataset`. PPG eval adds `--fold`.

## Notes

- Keep ECG and PPG runs separate: different data folders, different checkpoints. Do not cross-evaluate.
- Training prints MAE/RMSE only. Run the matching eval script for within ±5 / ±10 bpm percentages.
