"""Evaluate trained HuBERT-ECG HR weights on a manifest split.

Prints MAE/RMSE (bpm), MAPE, and % of windows within ±5 / ±10 bpm of NeuroKit labels.

Run from repo root after train:

    python -m hr_detection.scripts.eval_hubert
    python -m hr_detection.scripts.eval_hubert --dataset data/test_data_hr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hr_detection.core.datasets import HRWindows
from hr_detection.core.device import get_device
from hr_detection.core.metrics import format_report, regression_report
from hr_detection.core.models import load_hubert_trained
from hr_detection.core.paths import CKPT_DIR, MANIFEST_PATH, resolve_dataset


@torch.no_grad()
def collect_preds(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        out = model(x.to(device), attention_mask=None, return_dict=True)
        preds.append(out.logits.detach().cpu().numpy().reshape(-1))
        ys.append(y.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(ys)


def main():
    parser = argparse.ArgumentParser(
        description="HuBERT HR eval: % within ±N bpm vs NeuroKit labels (not clinical GT)."
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Manifest CSV, or a folder with ecg_manifest.csv. Default: data/hr_detection/manifest.csv",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(CKPT_DIR / "hubert_hr"),
        help="Directory from train_hubert.py save_pretrained",
    )
    parser.add_argument(
        "--within",
        type=float,
        nargs="+",
        default=[5.0, 10.0],
        help="bpm bands counted as 'correct' for the %% metric",
    )
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Missing HuBERT checkpoint dir {ckpt} (finish train_hubert first)")

    manifest_path = resolve_dataset(args.dataset, "ecg", MANIFEST_PATH)
    device = get_device()
    print("device", device)
    print("ckpt", ckpt)
    print("dataset", manifest_path)
    print("split", args.split)

    model = load_hubert_trained(str(ckpt), device)
    manifest = pd.read_csv(manifest_path)
    part = manifest[manifest.split == args.split]
    if part.empty:
        raise RuntimeError(f"No rows for split={args.split} in {manifest_path}")
    loader = DataLoader(
        HRWindows(part, "hubert", random_crop=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    pred, y = collect_preds(model, loader, device)
    stats = regression_report(pred, y, tolerances_bpm=tuple(args.within))
    print(format_report(stats))


if __name__ == "__main__":
    main()
