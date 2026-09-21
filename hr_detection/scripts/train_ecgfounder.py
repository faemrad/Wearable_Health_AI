from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hr_detection.core.datasets import HRWindows
from hr_detection.core.device import get_device
from hr_detection.core.metrics import mae_rmse
from hr_detection.core.models import load_founder_1lead
from hr_detection.core.paths import CKPT_DIR, MANIFEST_PATH, WEIGHTS_DIR


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--linear_prob", action="store_true", default=True)
    parser.add_argument("--full_ft", action="store_true")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_DIR / "1_lead_ECGFounder.pth"))
    args = parser.parse_args()
    linear_prob = not args.full_ft

    device = get_device()
    print("device", device)
    manifest = pd.read_csv(MANIFEST_PATH)
    train_ds = HRWindows(manifest[manifest.split == "train"], "founder")
    val_ds = HRWindows(manifest[manifest.split == "val"], "founder")
    test_ds = HRWindows(manifest[manifest.split == "test"], "founder")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = load_founder_1lead(args.weights, device, linear_prob=linear_prob)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    crit = nn.MSELoss()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_mae = 1e9
    best_path = CKPT_DIR / "ecgfounder_hr.pth"

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in tqdm(train_dl, desc=f"founder epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_mae, val_rmse = eval_loader(model, val_dl, device)
        print(
            f"epoch {epoch+1} train_mse={np.mean(losses):.2f} "
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
                best_path,
            )
            print("saved", best_path)

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_mae, test_rmse = eval_loader(model, test_dl, device)
    print(f"TEST founder MAE={test_mae:.2f} RMSE={test_rmse:.2f}")
    from safetensors.torch import save_file

    save_file(model.state_dict(), str(CKPT_DIR / "ecgfounder_hr.safetensors"))
    print("wrote", CKPT_DIR / "ecgfounder_hr.safetensors")


if __name__ == "__main__":
    main()
