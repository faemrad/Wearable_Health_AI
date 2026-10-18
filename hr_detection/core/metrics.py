from __future__ import annotations

import numpy as np


def mae_rmse(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    err = pred.reshape(-1) - y.reshape(-1)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def regression_report(
    pred: np.ndarray,
    y: np.ndarray,
    tolerances_bpm: tuple[float, ...] = (5.0, 10.0),
) -> dict[str, float]:
    pred = pred.reshape(-1)
    y = y.reshape(-1)
    abs_err = np.abs(pred - y)
    mae, rmse = mae_rmse(pred, y)
    mape = float(np.mean(abs_err / np.clip(y, 1e-6, None)) * 100.0)
    out: dict[str, float] = {
        "n": float(len(y)),
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
    }
    for t in tolerances_bpm:
        out[f"within_{t:g}_bpm_pct"] = float((abs_err <= t).mean() * 100.0)
    return out


def smooth_by_recording(
    pred: np.ndarray,
    frame: pd.DataFrame,
    window: int,
) -> np.ndarray:
    """Centered moving median over time within (patient, hand, activity)."""
    import pandas as pd

    if window <= 1:
        return pred
    if window % 2 == 0:
        window += 1
    out = np.asarray(pred, dtype=np.float64).copy()
    tmp = frame.reset_index(drop=True).copy()
    tmp["_pred"] = out
    tmp["_ord"] = np.arange(len(tmp))
    pieces = []
    for _, g in tmp.groupby(["patient_id", "hand", "activity"], sort=False):
        g = g.sort_values("start_sample_64")
        g["_pred"] = g["_pred"].rolling(window, center=True, min_periods=1).median()
        pieces.append(g)
    merged = pd.concat(pieces, axis=0).sort_values("_ord")
    return merged["_pred"].to_numpy(dtype=np.float64)


def activity_table(pred: np.ndarray, frame: pd.DataFrame, y: np.ndarray | None = None) -> str:
    import pandas as pd

    frame = frame.reset_index(drop=True)
    y = frame["hr_bpm"].to_numpy() if y is None else np.asarray(y).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    rows = []
    for act, g in frame.groupby("activity"):
        ii = g.index.to_numpy()
        err = np.abs(pred[ii] - y[ii])
        rows.append((str(act), len(ii), float(err.mean()), float((err <= 5).mean() * 100)))
    rows.sort(key=lambda r: r[0])
    lines = ["activity          n    MAE   ±5%"]
    for act, n, mae, w5 in rows:
        lines.append(f"{act:<14} {n:5d}  {mae:5.2f}  {w5:5.1f}")
    return "\n".join(lines)


def format_report(stats: dict[str, float]) -> str:
    lines = [
        f"n={int(stats['n'])}",
        f"MAE={stats['mae']:.2f} bpm",
        f"RMSE={stats['rmse']:.2f} bpm",
        f"MAPE={stats['mape_pct']:.1f}%",
    ]
    for key, val in stats.items():
        if key.startswith("within_"):
            tol = key.removeprefix("within_").removesuffix("_bpm_pct")
            lines.append(f"within ±{tol} bpm = {val:.1f}%")
    return "\n".join(lines)
