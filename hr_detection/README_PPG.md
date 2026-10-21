# Heart rate from PPG (PPG-Sport)

Regress bpm from wrist PPG with two models:

- **ECGFounder 1-lead**: the same backbone as the ECG run. PPG is upsampled from 64 Hz to 500 Hz, shape `(1, 4000)`.
- **[Pulse-PPG](https://github.com/maxxu05/pulseppg)**: a ResNet pretrained on wrist PPG, with input resampled to 50 Hz.

Labels come from the `cleaned_gt` ECG-reference HR. Windows are 8 s with a 2 s hop. The split is grouped by patient (80/10/10, plus 6 subject folds). Full run order: `RUN_EVERYTHING.md`.

Run everything from the repo root as `python -m hr_detection.scripts.<name>`.

## Commands

```bash
python -m hr_detection.scripts.prepare_ppg                  # data/hr_detection_ppg/manifest.csv
python -m hr_detection.scripts.download_weights             # ECGFounder backbone
python -m hr_detection.scripts.download_weights --pulseppg  # Pulse-PPG encoder (~300 MB)

python -m hr_detection.scripts.train_ppgfounder
python -m hr_detection.scripts.eval_ppgfounder --smooth 5 --baselines

python -m hr_detection.scripts.train_pulseppg
python -m hr_detection.scripts.eval_pulseppg --smooth 5 --baselines
```

Useful flags:

- Train: `--linear_prob` (frozen backbone, usually worse) and `--cv 6` (subject folds).
- Eval: `--fold <k> --ckpt hr_detection/checkpoints/<name>_fold<k>.pth`, `--split val`, `--within 5 10 15`.
- Prepare: `--force` rebuilds the cache and manifest. `--win_sec 10 --hop_sec 5` uses the old protocol.

## Notes

- Keep the PPG and PTB-XL runs separate. PPG uses `data/hr_detection_ppg/` and the `ppg_*_hr.pth` checkpoints.
- Do not pass `ecgfounder_hr.pth` (the PTB-XL run) or the Hugging Face backbone as `--ckpt`.
- Pulse-PPG needs the `models/PulsePPG` submodule (`git submodule update --init models/PulsePPG`).
