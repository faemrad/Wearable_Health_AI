"""Write a few PTB-XL and PPG-Sport windows for eval smoke tests.

Outputs under data/test_data_hr/:

    ecg/<stem>.npy          12-lead, 500 Hz, 10 s
    ppg/<stem>.npy          1-lead PPG, 500 Hz, 8 s
    ecg_manifest.csv
    ppg_manifest.csv

    python -m hr_detection.scripts.prepare_test_hr
    python -m hr_detection.scripts.eval_founder --dataset data/test_data_hr
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample

from hr_detection.core.paths import (
    PPG_FS,
    PPG_HOP_SEC,
    PPG_SPORT_ROOT,
    PPG_WIN_SEC,
    PTBXL_ROOT,
    REPO_ROOT,
    TARGET_FS,
    TEST_HR_DIR,
)
from hr_detection.core.preprocess import upsample_to_500
from hr_detection.scripts.prepare_ppg import PPG_NAME, _hr_for_window
from hr_detection.scripts.prepare_ptbxl import _complete_records, _heart_rate_bpm, _reorder_leads

N_ECG = 4
ECG_SCAN = 40
ECG_SPLITS = ("train", "train", "test", "test")


def _ecg_rows() -> list[dict]:
    db = pd.read_csv(PTBXL_ROOT / "ptbxl_database.csv")
    db["stem"] = db["filename_lr"].map(lambda p: Path(p).name.replace("_lr", ""))
    out_dir = TEST_HR_DIR / "ecg"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n_seen, rec_path in enumerate(_complete_records()):
        if len(rows) >= N_ECG or n_seen >= ECG_SCAN:
            break
        rec = wfdb.rdrecord(str(rec_path))
        fs = int(rec.fs)
        sig = np.asarray(rec.p_signal, dtype=np.float64)
        ecg12 = _reorder_leads(sig, [str(n) for n in rec.sig_name])
        ecg500 = upsample_to_500(ecg12, fs)
        bpm = _heart_rate_bpm(ecg500[1], 500)
        if bpm is None:
            continue
        stem = rec_path.name.replace("_lr", "")
        npy_path = out_dir / f"{stem}.npy"
        np.save(npy_path, ecg500.astype(np.float32))
        meta = db.loc[db["stem"] == stem]
        patient_id = int(meta["patient_id"].iloc[0]) if len(meta) else -1
        strat_fold = int(meta["strat_fold"].iloc[0]) if len(meta) else 1
        rows.append(
            {
                "stem": stem,
                "npy": str(npy_path.relative_to(REPO_ROOT)),
                "hr_bpm": bpm,
                "patient_id": patient_id,
                "strat_fold": strat_fold,
                "src": str(rec_path.relative_to(PTBXL_ROOT)),
                "split": ECG_SPLITS[len(rows)],
            }
        )

    if len(rows) < N_ECG:
        raise RuntimeError(f"Only labelled {len(rows)} PTB-XL records in the first {ECG_SCAN}.")
    return rows


def _ppg_rows(filename: str, plan: list[tuple[str, int]]) -> list[dict]:
    path = PPG_SPORT_ROOT / "dataset" / filename
    match = PPG_NAME.match(path.stem)
    if match is None:
        raise ValueError(f"Unexpected PPG name {path.name}")
    subject, hand, activity = match.group(1), match.group(2), match.group(3)
    gt_path = PPG_SPORT_ROOT / "cleaned_gt" / f"{subject}_{activity}_hr.npy"
    gt = np.load(gt_path).astype(np.float64, copy=False).reshape(-1)

    total = sum(n for _, n in plan)
    win_64 = int(round(PPG_WIN_SEC * PPG_FS))
    hop_64 = int(round(PPG_HOP_SEC * PPG_FS))
    need = win_64 + hop_64 * (total - 1)
    ppg = np.load(path).astype(np.float64, copy=False).reshape(-1)
    if ppg.size < need:
        raise RuntimeError(f"{path.name} has {ppg.size} samples, need {need}")
    ppg = ppg[:need]
    n_out = int(round(ppg.size * TARGET_FS / PPG_FS))
    full_500 = resample(ppg[None, :], n_out, axis=1).astype(np.float32)
    win_500 = int(round(PPG_WIN_SEC * TARGET_FS))

    out_dir = TEST_HR_DIR / "ppg"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cursor = 0
    for split, count in plan:
        for _ in range(count):
            start_64 = cursor
            cursor += hop_64
            bpm = _hr_for_window(gt, start_64, win_64, hop_64)
            if bpm is None:
                raise RuntimeError(f"No HR label for {filename} at sample {start_64}")
            start_500 = int(round(start_64 * TARGET_FS / PPG_FS))
            window = full_500[:, start_500 : start_500 + win_500]
            if window.shape[1] < win_500:
                raise RuntimeError(f"Short PPG window {filename} at sample {start_64}")
            stem = f"{subject}{hand}_{activity}_{start_64:05d}"
            npy_path = out_dir / f"{stem}.npy"
            np.save(npy_path, window)
            rows.append(
                {
                    "stem": stem,
                    "npy": str(npy_path.relative_to(REPO_ROOT)),
                    "hr_bpm": bpm,
                    "patient_id": int(subject[1:]),
                    "hand": hand,
                    "activity": activity,
                    "start_sample_64": start_64,
                    "gt_index": start_64 // hop_64,
                    "src": str(path.relative_to(PPG_SPORT_ROOT)),
                    "split": split,
                    "fold": 0,
                }
            )
    return rows


def prepare() -> None:
    TEST_HR_DIR.mkdir(parents=True, exist_ok=True)
    ecg = pd.DataFrame(_ecg_rows())
    ppg = pd.DataFrame(
        _ppg_rows("S01L_running_ppg.npy", [("test", 4), ("train", 2)])
        + _ppg_rows("S01L_stationary_ppg.npy", [("test", 2), ("train", 2)])
    )
    ecg_path = TEST_HR_DIR / "ecg_manifest.csv"
    ppg_path = TEST_HR_DIR / "ppg_manifest.csv"
    ecg.to_csv(ecg_path, index=False)
    ppg.to_csv(ppg_path, index=False)
    print(f"Wrote {ecg_path} n={len(ecg)}")
    print(ecg.groupby("split").size().to_string())
    print(f"Wrote {ppg_path} n={len(ppg)}")
    print(ppg.groupby(["split", "activity"]).size().to_string())


def main() -> None:
    prepare()


if __name__ == "__main__":
    main()
