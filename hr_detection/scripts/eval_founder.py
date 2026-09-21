"""Evaluate trained ECGFounder HR weights on a manifest split.

Prints MAE/RMSE (bpm), MAPE, and % of windows within ±5 / ±10 bpm of NeuroKit labels.

Run from repo root after train:

    python -m hr_detection.scripts.eval_founder
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hr_detection.core.datasets import HRWindows
from hr_detection.core.device import get_device
from hr_detection.core.metrics import format_report, regression_report
from hr_detection.core.models import load_founder_hr
from hr_detection.core.paths import CKPT_DIR, MANIFEST_PATH, WEIGHTS_DIR


@torch.no_grad()
def collect_preds(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.detach().cpu().numpy().reshape(-1))
        ys.append(y.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(ys)


def main():
    parser = argparse.ArgumentParser(
        description="HR regression eval: % within ±N bpm vs NeuroKit labels (not clinical GT)."
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(CKPT_DIR / "ecgfounder_hr.pth"),
        help="Trained HR checkpoint from train_ecgfounder.py",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=str(WEIGHTS_DIR / "1_lead_ECGFounder.pth"),
        help="Original ECGFounder 1-lead weights (architecture + frozen backbone)",
    )
    parser.add_argument(
        "--within",
        type=float,
        nargs="+",
        default=[5.0, 10.0],
        help="bpm bands counted as 'correct' for the %% metric",
    )
    args = parser.parse_args()

    device = get_device()
    print("device", device)
    print("ckpt", args.ckpt)
    print("split", args.split)

    model = load_founder_hr(args.backbone, args.ckpt, device, linear_prob=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    part = manifest[manifest.split == args.split]
    if part.empty:
        raise RuntimeError(f"No rows for split={args.split} in {MANIFEST_PATH}")
    loader = DataLoader(
        HRWindows(part, "founder"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    pred, y = collect_preds(model, loader, device)
    stats = regression_report(pred, y, tolerances_bpm=tuple(args.within))
    print(format_report(stats))


if __name__ == "__main__":
    main()
