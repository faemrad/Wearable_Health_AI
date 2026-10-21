"""Evaluate Pulse-PPG HR weights on a PPG-Sport manifest split.

Prints overall MAE/RMSE/MAPE/%-within, per-activity MAE, optional temporal
smoothing, and constant / FFT-peak baselines. Same report as eval_ppgfounder.

    python -m hr_detection.scripts.eval_pulseppg --smooth 5 --baselines
    python -m hr_detection.scripts.eval_pulseppg --dataset data/test_data_hr
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, detrend, filtfilt
from torch.utils.data import DataLoader

from hr_detection.core.datasets import HRWindows
from hr_detection.core.device import get_device
from hr_detection.core.metrics import (
    activity_table,
    format_report,
    regression_report,
    smooth_by_recording,
)
from hr_detection.core.models import load_pulseppg_hr
from hr_detection.core.paths import (
    CKPT_DIR,
    PPG_MANIFEST_PATH,
    PULSEPPG_WEIGHTS,
    REPO_ROOT,
    TARGET_FS,
    resolve_dataset,
)


@torch.no_grad()
def collect_preds(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.detach().cpu().numpy().reshape(-1))
        ys.append(y.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(ys)


def fft_peak_bpm(npy_rel: str) -> float:
    x = np.load(REPO_ROOT / npy_rel).reshape(-1).astype(np.float64)
    b, a = butter(4, [0.5, 8], btype="bandpass", fs=TARGET_FS)
    xf = filtfilt(b, a, detrend(x))
    n = 1 << 15
    pwr = np.abs(np.fft.rfft(xf * np.hanning(len(xf)), n)) ** 2
    freq = np.fft.rfftfreq(n, 1 / TARGET_FS)
    sel = (freq >= 0.6) & (freq <= 3.6)
    return float(freq[sel][np.argmax(pwr[sel])] * 60.0)


def print_block(title: str, pred: np.ndarray, y: np.ndarray, frame: pd.DataFrame, within: tuple[float, ...]):
    print(title)
    print(format_report(regression_report(pred, y, tolerances_bpm=within)))
    print(activity_table(pred, frame, y))


def main():
    parser = argparse.ArgumentParser(
        description="Pulse-PPG HR eval: MAE + per-activity + optional smooth / baselines."
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Manifest CSV, or a folder with ppg_manifest.csv. Default: data/hr_detection_ppg/manifest.csv",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "ppg_pulseppg_hr.pth"))
    parser.add_argument("--backbone", type=str, default=str(PULSEPPG_WEIGHTS))
    parser.add_argument("--within", type=float, nargs="+", default=[5.0, 10.0])
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Centered moving-median window (odd). 0 or 1 disables.",
    )
    parser.add_argument("--baselines", action="store_true", help="Print constant-mean and FFT-peak rows")
    parser.add_argument("--fold", type=int, default=None, help="If set, eval this CV fold instead of --split")
    args = parser.parse_args()
    within = tuple(args.within)

    device = get_device()
    print("device", device)
    print("ckpt", args.ckpt)

    manifest_path = resolve_dataset(args.dataset, "ppg", PPG_MANIFEST_PATH)
    print("dataset", manifest_path)
    model = load_pulseppg_hr(args.backbone, args.ckpt, device, linear_prob=False)
    manifest = pd.read_csv(manifest_path)
    if args.fold is not None:
        part = manifest[manifest.fold == args.fold].reset_index(drop=True)
        train_part = manifest[manifest.fold != args.fold]
        print("fold", args.fold)
    else:
        part = manifest[manifest.split == args.split].reset_index(drop=True)
        train_part = manifest[manifest.split == "train"]
        print("split", args.split)
    if part.empty:
        raise RuntimeError(f"No rows for this split/fold in {manifest_path}")

    loader = DataLoader(HRWindows(part, "pulseppg"), batch_size=args.batch_size, shuffle=False, num_workers=0)
    pred, y = collect_preds(model, loader, device)
    print_block("=== model (raw) ===", pred, y, part, within)

    if args.smooth > 1:
        sm = smooth_by_recording(pred, part, args.smooth)
        print_block(f"=== model (smooth={args.smooth}) ===", sm, y, part, within)

    if args.baselines:
        c = float(train_part["hr_bpm"].mean())
        const = np.full_like(y, c, dtype=np.float64)
        print_block(f"=== baseline constant (train mean {c:.1f}) ===", const, y, part, within)
        fft = np.array([fft_peak_bpm(p) for p in part["npy"]], dtype=np.float64)
        print_block("=== baseline FFT-peak ===", fft, y, part, within)


if __name__ == "__main__":
    main()
