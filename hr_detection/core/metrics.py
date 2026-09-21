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
