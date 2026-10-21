"""Evaluate every checkpoint in hr_detection/checkpoints/.

Writes:
  output/eval_results.txt
  output/eval_results.html

Run from the repo root:

    python -m hr_detection.scripts.eval_sweep
    python -m hr_detection.scripts.eval_sweep --dataset data/test_data_hr
"""

from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hr_detection.core.datasets import HRWindows
from hr_detection.core.device import get_device
from hr_detection.core.metrics import regression_report, smooth_by_recording
from hr_detection.core.models import load_founder_hr, load_hubert_trained, load_pulseppg_hr
from hr_detection.core.paths import (
    CKPT_DIR,
    MANIFEST_PATH,
    PPG_MANIFEST_PATH,
    PULSEPPG_WEIGHTS,
    REPO_ROOT,
    WEIGHTS_DIR,
    resolve_dataset,
)
from hr_detection.scripts.eval_founder import collect_preds as collect_founder
from hr_detection.scripts.eval_hubert import collect_preds as collect_hubert
from hr_detection.scripts.eval_ppgfounder import collect_preds as collect_ppg
from hr_detection.scripts.eval_ppgfounder import fft_peak_bpm

WITHIN = (5.0, 10.0)
SMOOTH = 5

JOBS = [
    {
        "dataset": "PTB-XL",
        "model": "ECGFounder",
        "mode": "Linear probe",
        "ckpt": CKPT_DIR / "ecgfounder_hr_probe.pth",
        "kind": "founder",
    },
    {
        "dataset": "PTB-XL",
        "model": "ECGFounder",
        "mode": "Full fine-tune",
        "ckpt": CKPT_DIR / "ecgfounder_hr_full.pth",
        "kind": "founder",
    },
    {
        "dataset": "PTB-XL",
        "model": "HuBERT-ECG",
        "mode": "Linear probe",
        "ckpt": CKPT_DIR / "hubert_hr_probe",
        "kind": "hubert",
    },
    {
        "dataset": "PTB-XL",
        "model": "HuBERT-ECG",
        "mode": "Partial (last 2 blocks)",
        "ckpt": CKPT_DIR / "hubert_hr_partial",
        "kind": "hubert",
    },
    {
        "dataset": "PTB-XL",
        "model": "HuBERT-ECG",
        "mode": "All 8 transformer blocks",
        "ckpt": CKPT_DIR / "hubert_hr_unfreeze8",
        "kind": "hubert",
    },
    {
        "dataset": "PPG-Sport",
        "model": "ECGFounder",
        "mode": "Linear probe",
        "ckpt": CKPT_DIR / "ppg_ecgfounder_hr_probe.pth",
        "kind": "founder_ppg",
    },
    {
        "dataset": "PPG-Sport",
        "model": "ECGFounder",
        "mode": "Full fine-tune",
        "ckpt": CKPT_DIR / "ppg_ecgfounder_hr_full.pth",
        "kind": "founder_ppg",
    },
    {
        "dataset": "PPG-Sport",
        "model": "Pulse-PPG",
        "mode": "Linear probe",
        "ckpt": CKPT_DIR / "ppg_pulseppg_hr_probe.pth",
        "kind": "pulseppg",
    },
    {
        "dataset": "PPG-Sport",
        "model": "Pulse-PPG",
        "mode": "Full fine-tune",
        "ckpt": CKPT_DIR / "ppg_pulseppg_hr_full.pth",
        "kind": "pulseppg",
    },
]


def _release(model) -> None:
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _activity_rows(pred: np.ndarray, frame: pd.DataFrame, y: np.ndarray) -> list[dict]:
    rows = []
    pred = np.asarray(pred).reshape(-1)
    y = np.asarray(y).reshape(-1)
    for act, g in frame.groupby("activity"):
        ii = g.index.to_numpy()
        err = np.abs(pred[ii] - y[ii])
        rows.append(
            {
                "activity": str(act),
                "n": int(len(ii)),
                "mae": float(err.mean()),
                "within_5": float((err <= 5).mean() * 100.0),
                "within_10": float((err <= 10).mean() * 100.0),
            }
        )
    rows.sort(key=lambda r: r["activity"])
    return rows


def _eval_job(job: dict, device, manifests: dict) -> dict:
    ckpt = job["ckpt"]
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    kind = job["kind"]
    print(f"eval {job['dataset']} | {job['model']} | {job['mode']}", flush=True)

    if kind == "founder":
        part = manifests["ecg"]
        model = load_founder_hr(
            str(WEIGHTS_DIR / "1_lead_ECGFounder.pth"), str(ckpt), device, linear_prob=True
        )
        loader = DataLoader(HRWindows(part, "founder"), batch_size=32, shuffle=False, num_workers=0)
        pred, y = collect_founder(model, loader, device)
        _release(model)
        return {"raw": regression_report(pred, y, WITHIN)}

    if kind == "hubert":
        part = manifests["ecg"]
        model = load_hubert_trained(str(ckpt), device)
        loader = DataLoader(
            HRWindows(part, "hubert", random_crop=False), batch_size=8, shuffle=False, num_workers=0
        )
        pred, y = collect_hubert(model, loader, device)
        _release(model)
        return {"raw": regression_report(pred, y, WITHIN)}

    part = manifests["ppg"]
    if kind == "founder_ppg":
        model = load_founder_hr(
            str(WEIGHTS_DIR / "1_lead_ECGFounder.pth"), str(ckpt), device, linear_prob=True
        )
        loader = DataLoader(HRWindows(part, "founder_ppg"), batch_size=32, shuffle=False, num_workers=0)
        pred, y = collect_ppg(model, loader, device)
    else:
        model = load_pulseppg_hr(str(PULSEPPG_WEIGHTS), str(ckpt), device, linear_prob=False)
        loader = DataLoader(HRWindows(part, "pulseppg"), batch_size=32, shuffle=False, num_workers=0)
        pred, y = collect_ppg(model, loader, device)
    _release(model)
    sm = smooth_by_recording(pred, part, SMOOTH)
    return {
        "raw": regression_report(pred, y, WITHIN),
        "smooth": regression_report(sm, y, WITHIN),
        "activity": _activity_rows(pred, part, y),
        "activity_smooth": _activity_rows(sm, part, y),
    }


def _baselines(part: pd.DataFrame, train_part: pd.DataFrame) -> dict:
    y = part["hr_bpm"].to_numpy(dtype=np.float64)
    c = float(train_part["hr_bpm"].mean())
    const = np.full_like(y, c)
    print("eval PPG baselines (constant mean, FFT peak)", flush=True)
    fft = np.array([fft_peak_bpm(p) for p in part["npy"]], dtype=np.float64)
    return {
        "train_mean": c,
        "constant": regression_report(const, y, WITHIN),
        "fft": regression_report(fft, y, WITHIN),
        "fft_activity": _activity_rows(fft, part, y),
    }


def _fmt(stats: dict, key: str, digits: int = 2) -> str:
    val = stats[key]
    if key == "n":
        return str(int(val))
    if key == "mape_pct" or key.startswith("within_"):
        return f"{val:.1f}"
    return f"{val:.{digits}f}"


def _txt(rows: list[dict], baselines: dict, when: str, device: str) -> str:
    lines = [
        f"Sweep eval  {when}",
        f"device {device}",
        "split test",
        "PTB-XL reference: NeuroKit2 lead II. PPG-Sport reference: cleaned_gt chest ECG.",
        "PPG smooth is a centered 5-window median within each recording. Labels are not smoothed.",
        "",
        "dataset\tmodel\tmode\tview\tn\tMAE\tRMSE\tMAPE%\twithin5%\twithin10%",
    ]
    for row in rows:
        for view in ("raw", "smooth"):
            if view not in row:
                continue
            s = row[view]
            lines.append(
                "\t".join(
                    [
                        row["dataset"],
                        row["model"],
                        row["mode"],
                        view,
                        _fmt(s, "n"),
                        _fmt(s, "mae"),
                        _fmt(s, "rmse"),
                        _fmt(s, "mape_pct"),
                        _fmt(s, "within_5_bpm_pct"),
                        _fmt(s, "within_10_bpm_pct"),
                    ]
                )
            )
    lines.append("")
    lines.append(f"PPG baseline constant (train mean {baselines['train_mean']:.1f} bpm)")
    for name, key in (("constant", "constant"), ("fft_peak", "fft")):
        s = baselines[key]
        lines.append(
            "\t".join(
                [
                    "PPG-Sport",
                    name,
                    "baseline",
                    "raw",
                    _fmt(s, "n"),
                    _fmt(s, "mae"),
                    _fmt(s, "rmse"),
                    _fmt(s, "mape_pct"),
                    _fmt(s, "within_5_bpm_pct"),
                    _fmt(s, "within_10_bpm_pct"),
                ]
            )
        )
    lines.append("")
    lines.append("PPG per activity (raw predictions)")
    lines.append("model\tmode\tactivity\tn\tMAE\twithin5%\twithin10%")
    for row in rows:
        if "activity" not in row:
            continue
        for act in row["activity"]:
            lines.append(
                f"{row['model']}\t{row['mode']}\t{act['activity']}\t{act['n']}\t"
                f"{act['mae']:.2f}\t{act['within_5']:.1f}\t{act['within_10']:.1f}"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def _table(headers: list[str], body: list[list[str]], best_mae_col: int | None = None) -> str:
    best = None
    if best_mae_col is not None:
        nums = []
        for cells in body:
            try:
                nums.append(float(cells[best_mae_col]))
            except ValueError:
                nums.append(None)
        present = [n for n in nums if n is not None]
        best = min(present) if present else None
    out = ["<table>", "<thead><tr>"]
    out.extend(f"<th>{html.escape(h)}</th>" for h in headers)
    out.append("</tr></thead><tbody>")
    for cells in body:
        mark = ""
        if best is not None:
            try:
                mark = " class='best'" if float(cells[best_mae_col]) == best else ""
            except ValueError:
                mark = ""
        out.append(f"<tr{mark}>")
        out.extend(f"<td>{html.escape(c)}</td>" for c in cells)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _html(rows: list[dict], baselines: dict, when: str, device: str) -> str:
    ecg = [r for r in rows if r["dataset"] == "PTB-XL"]
    ppg = [r for r in rows if r["dataset"] == "PPG-Sport"]
    ecg_body = []
    for r in ecg:
        s = r["raw"]
        ecg_body.append(
            [
                r["model"],
                r["mode"],
                _fmt(s, "n"),
                _fmt(s, "mae"),
                _fmt(s, "rmse"),
                _fmt(s, "mape_pct"),
                _fmt(s, "within_5_bpm_pct"),
                _fmt(s, "within_10_bpm_pct"),
            ]
        )
    ppg_body = []
    for r in ppg:
        s, sm = r["raw"], r["smooth"]
        ppg_body.append(
            [
                r["model"],
                r["mode"],
                _fmt(s, "n"),
                _fmt(s, "mae"),
                _fmt(s, "rmse"),
                _fmt(s, "mape_pct"),
                _fmt(s, "within_5_bpm_pct"),
                _fmt(s, "within_10_bpm_pct"),
                _fmt(sm, "mae"),
                _fmt(sm, "within_5_bpm_pct"),
                _fmt(sm, "within_10_bpm_pct"),
            ]
        )
    base_body = []
    for label, key in (
        (f"Constant train mean ({baselines['train_mean']:.1f} bpm)", "constant"),
        ("FFT peak (36–216 bpm)", "fft"),
    ):
        s = baselines[key]
        base_body.append(
            [
                label,
                _fmt(s, "n"),
                _fmt(s, "mae"),
                _fmt(s, "rmse"),
                _fmt(s, "mape_pct"),
                _fmt(s, "within_5_bpm_pct"),
                _fmt(s, "within_10_bpm_pct"),
            ]
        )
    act_body = []
    for r in ppg:
        for act in r["activity"]:
            act_body.append(
                [
                    r["model"],
                    r["mode"],
                    act["activity"],
                    str(act["n"]),
                    f"{act['mae']:.2f}",
                    f"{act['within_5']:.1f}",
                    f"{act['within_10']:.1f}",
                ]
            )
    headers_main = ["Model", "Training", "n", "MAE", "RMSE", "MAPE %", "±5 %", "±10 %"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HR sweep eval</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 980px; margin: 2rem auto; color: #1c1c1c; padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
  p {{ line-height: 1.45; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.6rem 0 1.2rem; font-variant-numeric: tabular-nums; }}
  th, td {{ border-bottom: 1px solid #ccc; text-align: left; padding: 0.35rem 0.5rem; font-size: 0.92rem; }}
  th {{ border-bottom: 2px solid #222; }}
  tr.best td {{ background: #eef6ee; }}
  .note {{ color: #444; font-size: 0.92rem; }}
</style>
</head>
<body>
<h1>Heart-rate sweep, test split</h1>
<p class="note">{html.escape(when)} · device {html.escape(device)}. Highlighted row in each model table has the lowest raw MAE. ±5 % is the share of windows within 5 bpm of the reference.</p>
<h2>PTB-XL</h2>
<p class="note">Reference is the NeuroKit2 heart rate on lead II. One 10-second window per recording.</p>
{_table(headers_main, ecg_body, best_mae_col=3)}
<h2>PPG-Sport</h2>
<p class="note">Reference is the chest-ECG <code>cleaned_gt</code> rate. Smooth is a centered 5-window median of the predictions inside each recording. The labels are not smoothed.</p>
{_table(headers_main + ["Smooth MAE", "Smooth ±5 %", "Smooth ±10 %"], ppg_body, best_mae_col=3)}
<h2>PPG baselines</h2>
<p class="note">Same test windows. These do not depend on a checkpoint.</p>
{_table(["Baseline", "n", "MAE", "RMSE", "MAPE %", "±5 %", "±10 %"], base_body, best_mae_col=2)}
<h2>PPG per activity, raw predictions</h2>
{_table(["Model", "Training", "Activity", "n", "MAE", "±5 %", "±10 %"], act_body)}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every checkpoint in hr_detection/checkpoints/."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Folder with ecg_manifest.csv and ppg_manifest.csv. "
            "Default: the full prepared manifests. "
            "With this set, reports go in the folder, not output/."
        ),
    )
    args = parser.parse_args()
    if args.dataset is not None and not Path(args.dataset).is_dir():
        raise SystemExit(
            "--dataset for eval_sweep must be a folder with ecg_manifest.csv and ppg_manifest.csv"
        )

    device = get_device()
    ecg_path = resolve_dataset(args.dataset, "ecg", MANIFEST_PATH)
    ppg_path = resolve_dataset(args.dataset, "ppg", PPG_MANIFEST_PATH)
    print("device", device, flush=True)
    print("dataset ecg", ecg_path, flush=True)
    print("dataset ppg", ppg_path, flush=True)
    ecg = pd.read_csv(ecg_path)
    ppg = pd.read_csv(ppg_path)
    manifests = {
        "ecg": ecg[ecg.split == "test"].reset_index(drop=True),
        "ppg": ppg[ppg.split == "test"].reset_index(drop=True),
    }
    if manifests["ecg"].empty or manifests["ppg"].empty:
        raise RuntimeError(f"Test split is empty in {ecg_path} or {ppg_path}.")

    rows = []
    for job in JOBS:
        stats = _eval_job(job, device, manifests)
        rows.append({**job, **stats})
        s = stats["raw"]
        print(
            f"  MAE={s['mae']:.2f}  ±5={s['within_5_bpm_pct']:.1f}%",
            flush=True,
        )

    train_ppg = ppg[ppg.split == "train"]
    if train_ppg.empty:
        print("no PPG train rows; constant baseline uses the eval rows", flush=True)
        train_ppg = manifests["ppg"]
    baselines = _baselines(manifests["ppg"], train_ppg)
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    if args.dataset is None:
        out_dir = REPO_ROOT / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / "eval_results.txt"
        html_path = out_dir / "eval_results.html"
    else:
        out_dir = Path(args.dataset)
        txt_path = out_dir / "sweep_eval_results.txt"
        html_path = out_dir / "sweep_eval_results.html"
    txt_path.write_text(_txt(rows, baselines, when, str(device)), encoding="utf-8")
    html_path.write_text(_html(rows, baselines, when, str(device)), encoding="utf-8")
    print("wrote", txt_path, flush=True)
    print("wrote", html_path, flush=True)


if __name__ == "__main__":
    main()
