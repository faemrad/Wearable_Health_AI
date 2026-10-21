# Heart rate from ECG (PTB-XL)

Regress bpm from 10 s PTB-XL windows with two models:

- **ECGFounder**: 1-lead (Lead II), linear probe by default.
- **HuBERT-ECG small**: 12-lead, last 2 transformer blocks unfrozen by default.

Labels are NeuroKit2 HR from Lead II R-peaks. Records where peak detection fails are dropped. Full run order: `RUN_EVERYTHING.md`.

Run everything from the repo root as `python -m hr_detection.scripts.<name>`.

## Commands

```bash
python -m hr_detection.scripts.prepare_ptbxl      # data/hr_detection/manifest.csv
python -m hr_detection.scripts.download_weights   # models/checkpoints/1_lead_ECGFounder.pth

python -m hr_detection.scripts.train_ecgfounder   # --full_ft --lr 1e-4 for full fine-tune
python -m hr_detection.scripts.eval_founder

python -m hr_detection.scripts.train_hubert       # out of memory: --batch_size 2
python -m hr_detection.scripts.eval_hubert
```

Useful eval flags: `--split val`, `--within 5 10 15`, `--ckpt <path>`.

## Notes

- Prepare reads only `data/ptb-xl/records100/`. A record needs both `.hea` and `.dat`.
- HuBERT needs `pip install -e models/HuBERT_ECG`. Without it you get `ModuleNotFoundError: hubert_ecg`.
- Training prints MAE/RMSE only. Run eval to get the within ±5 / ±10 bpm percentages.
- Do not run `eval_founder` on PPG checkpoints.
