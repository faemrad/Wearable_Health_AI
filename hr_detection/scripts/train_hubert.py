from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hr_detection.core.datasets import HRWindows
from hr_detection.core.device import get_device
from hr_detection.core.metrics import mae_rmse
from hr_detection.core.models import load_hubert_hr
from hr_detection.core.paths import CKPT_DIR, MANIFEST_PATH


@torch.no_grad()
def eval_loader(model, loader, device) -> tuple[float, float]:
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x, attention_mask=None, return_dict=True)
        preds.append(out.logits.detach().cpu().numpy())
        ys.append(y.cpu().numpy())
    return mae_rmse(np.concatenate(preds), np.concatenate(ys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained", type=str, default="Edoardo-Coppola/hubert-ecg-small")
    parser.add_argument("--unfreeze_last", type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    print("device", device)
    manifest = pd.read_csv(MANIFEST_PATH)
    train_ds = HRWindows(manifest[manifest.split == "train"], "hubert", random_crop=True)
    val_ds = HRWindows(manifest[manifest.split == "val"], "hubert", random_crop=False)
    test_ds = HRWindows(manifest[manifest.split == "test"], "hubert", random_crop=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = load_hubert_hr(args.pretrained, device, unfreeze_last=args.unfreeze_last)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_mae = 1e9
    best_dir = CKPT_DIR / "hubert_hr"

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in tqdm(train_dl, desc=f"hubert epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x, attention_mask=None, labels=y.squeeze(-1), return_dict=True)
            loss = out.loss
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
            model.save_pretrained(best_dir, safe_serialization=True)
            print("saved", best_dir)

    from hr_detection.core.models import load_hubert_trained

    model = load_hubert_trained(str(best_dir), device)
    test_mae, test_rmse = eval_loader(model, test_dl, device)
    print(f"TEST hubert MAE={test_mae:.2f} RMSE={test_rmse:.2f}")


if __name__ == "__main__":
    main()
