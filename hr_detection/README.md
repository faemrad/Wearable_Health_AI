# Heart rate detection

Regress heart rate (bpm) from ECG and wrist PPG using pretrained foundation models. Four models across two datasets:

| Model | Data | Signal | Details |
| ----- | ---- | ------ | ------- |
| ECGFounder 1-lead | PTB-XL | ECG (Lead II) | `README_ECG.md` |
| HuBERT-ECG small | PTB-XL | ECG (12-lead) | `README_ECG.md` |
| ECGFounder 1-lead | PPG-Sport | wrist PPG | `README_PPG.md` |
| Pulse-PPG | PPG-Sport | wrist PPG | `README_PPG.md` |

Full setup and run order for all four: `RUN_EVERYTHING.md`.

Run everything from the repo root as `python -m hr_detection.scripts.<name>`.

## Layout

```text
hr_detection/
  core/          shared library: datasets, models, preprocessing, metrics, paths, device
  scripts/       CLIs: prepare_*, download_weights, train_*, eval_*, eval_sweep
  checkpoints/   trained weights (ECG: ecgfounder_hr, hubert_hr; PPG: ppg_*_hr; sweep/)
```

## Scripts

- `prepare_ptbxl`, `prepare_ppg`: build the manifests under `data/hr_detection/` and `data/hr_detection_ppg/`.
- `download_weights`: fetch the ECGFounder backbone (`--pulseppg` for the Pulse-PPG encoder).
- `train_*` / `eval_*`: one pair per model (see the per-signal READMEs).
- `eval_sweep`: evaluate every checkpoint in `checkpoints/sweep/` and write `eval_results.txt` / `.html` there.

## Notes

- Keep ECG and PPG runs separate: different data folders, different checkpoints. Do not cross-evaluate.
- Training prints MAE/RMSE only. Run the matching eval script for within ±5 / ±10 bpm percentages.
