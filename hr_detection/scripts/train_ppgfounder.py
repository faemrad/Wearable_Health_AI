"""Fine-tune ECGFounder 1-lead as PPG HR regressor (PPG-Sport windows).

Default: single 80/10/10 split.
Subject 6-fold CV: --cv 6  (writes ppg_ecgfounder_hr_fold{k}.pth)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from hr_detection.core.datasets import HRWindows
from hr_detection.core.device import get_device
from hr_detection.core.metrics import mae_rmse, regression_report, smooth_by_recording
from hr_detection.core.models import load_founder_1lead
from hr_detection.core.paths import CKPT_DIR, PPG_CV_FOLDS, PPG_MANIFEST_PATH, WEIGHTS_DIR


@torch.no_grad()
def eval_loader(model, loader, device) -> tuple[float, float]:
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        preds.append(out.detach().cpu().numpy())
        ys.append(y.numpy())
    return mae_rmse(np.concatenate(preds), np.concatenate(ys))


@torch.no_grad()
def collect_preds(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.detach().cpu().numpy().reshape(-1))
        ys.append(y.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(ys)


def _loaders(train_df, val_df, test_df, batch_size):
    return (
        DataLoader(HRWindows(train_df, "founder_ppg"), batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(HRWindows(val_df, "founder_ppg"), batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(HRWindows(test_df, "founder_ppg"), batch_size=batch_size, shuffle=False, num_workers=0),
    )


def train_one(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    device,
    epochs: int,
    batch_size: int,
    lr: float,
    linear_prob: bool,
    weights: str,
    ckpt_path: Path,
    tag: str,
) -> dict:
    train_dl, val_dl, test_dl = _loaders(train_df, val_df, test_df, batch_size)
    model = load_founder_1lead(weights, device, linear_prob=linear_prob)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    crit = nn.MSELoss()
    best_mae = 1e9
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in tqdm(train_dl, desc=f"{tag} epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_mae, val_rmse = eval_loader(model, val_dl, device)
        print(
            f"{tag} epoch {epoch+1} train_mse={np.mean(losses):.2f} "
            f"val_mae={val_mae:.2f} val_rmse={val_rmse:.2f}"
        )
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mae": val_mae,
                    "epoch": epoch + 1,
                    "n_classes": 1,
                    "in_channels": 1,
                },
                ckpt_path,
            )
            print("saved", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    pred, y = collect_preds(model, test_dl, device)
    raw = regression_report(pred, y)
    sm = smooth_by_recording(pred, test_df.reset_index(drop=True), 5)
    smo = regression_report(sm, y)
    print(
        f"{tag} TEST raw MAE={raw['mae']:.2f}  smooth5 MAE={smo['mae']:.2f}  "
        f"±5={smo['within_5_bpm_pct']:.1f}%"
    )
    return {"raw": raw, "smooth": smo, "path": str(ckpt_path)}


def _cv_frames(manifest: pd.DataFrame, fold: int):
    test_df = manifest[manifest.fold == fold]
    rest = manifest[manifest.fold != fold]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42 + fold)
    tr_idx, va_idx = next(gss.split(rest, groups=rest["patient_id"]))
    return rest.iloc[tr_idx], rest.iloc[va_idx], test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--linear_prob", action="store_true")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_DIR / "1_lead_ECGFounder.pth"))
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        metavar="K",
        help=f"Subject GroupKFold with K folds (use {PPG_CV_FOLDS}). 0 = single split.",
    )
    args = parser.parse_args()
    linear_prob = args.linear_prob
    if linear_prob and args.lr == 1e-4:
        args.lr = 1e-3

    device = get_device()
    print("device", device)
    print("linear_prob", linear_prob, "lr", args.lr)
    manifest = pd.read_csv(PPG_MANIFEST_PATH)

    if args.cv:
        if "fold" not in manifest.columns:
            raise RuntimeError("Manifest has no fold column. Re-run prepare_ppg --force")
        summaries = []
        for k in range(args.cv):
            tr, va, te = _cv_frames(manifest, k)
            print(f"fold {k}: train n={len(tr)} val n={len(va)} test n={len(te)} "
                  f"subj_test={sorted(te.patient_id.unique())}")
            out = train_one(
                tr, va, te,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                linear_prob=linear_prob,
                weights=args.weights,
                ckpt_path=CKPT_DIR / f"ppg_ecgfounder_hr_fold{k}.pth",
                tag=f"fold{k}",
            )
            summaries.append(out)
        raw = np.array([s["raw"]["mae"] for s in summaries])
        sm = np.array([s["smooth"]["mae"] for s in summaries])
        print(
            f"CV{args.cv} raw MAE={raw.mean():.2f}±{raw.std():.2f}  "
            f"smooth5 MAE={sm.mean():.2f}±{sm.std():.2f}"
        )
        return

    train_ds = manifest[manifest.split == "train"]
    val_ds = manifest[manifest.split == "val"]
    test_ds = manifest[manifest.split == "test"]
    if len(train_ds) == 0:
        raise RuntimeError(f"Empty train split in {PPG_MANIFEST_PATH}. Run prepare_ppg first.")
    train_one(
        train_ds, val_ds, test_ds,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        linear_prob=linear_prob,
        weights=args.weights,
        ckpt_path=CKPT_DIR / "ppg_ecgfounder_hr.pth",
        tag="ppg founder",
    )
    from safetensors.torch import save_file

    ckpt = torch.load(CKPT_DIR / "ppg_ecgfounder_hr.pth", map_location="cpu", weights_only=False)
    model = load_founder_1lead(args.weights, device, linear_prob=linear_prob)
    model.load_state_dict(ckpt["state_dict"])
    save_file(model.state_dict(), str(CKPT_DIR / "ppg_ecgfounder_hr.safetensors"))
    print("wrote", CKPT_DIR / "ppg_ecgfounder_hr.safetensors")


if __name__ == "__main__":
    main()
