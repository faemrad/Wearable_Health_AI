"""Build PPG windows at 500 Hz + ECG-reference HR labels from PPG-Sport.

Default matches PPG-Sport cleaned_gt: 8 s windows, hop 2 s (WIN=512, HOP=128 @ 64 Hz).
Label is the GT bin, not interpolated.

Writes:
  data/hr_detection_ppg/cache/<stem>.npy
  data/hr_detection_ppg/manifest.csv
"""

from __future__ import annotations

import argparse
import re
import shutil

import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from tqdm import tqdm

from hr_detection.core.paths import (
    PPG_CACHE_DIR,
    PPG_CV_FOLDS,
    PPG_FS,
    PPG_HOP_SEC,
    PPG_HR_DATA,
    PPG_MANIFEST_PATH,
    PPG_SPORT_ROOT,
    PPG_WIN_SEC,
    REPO_ROOT,
    TARGET_FS,
)

PPG_NAME = re.compile(r"^(S\d{2})([LR])_([a-z]+)_ppg$")
GT_WIN_SAMP = 512
GT_HOP_SAMP = 128
MIN_BPM, MAX_BPM = 30.0, 220.0


def _hr_for_window(gt: np.ndarray, start_64: int, win_64: int, hop_64: int) -> float | None:
    aligned = win_64 == GT_WIN_SAMP and hop_64 == GT_HOP_SAMP and start_64 % GT_HOP_SAMP == 0
    if aligned:
        w = start_64 // GT_HOP_SAMP
        if w < 0 or w >= len(gt):
            return None
        bpm = float(gt[w])
    else:
        if gt.size < 2:
            return None
        t_gt = (np.arange(len(gt), dtype=np.float64) * GT_HOP_SAMP + GT_WIN_SAMP / 2.0) / PPG_FS
        t_c = (start_64 + win_64 / 2.0) / PPG_FS
        if t_c < t_gt[0] or t_c > t_gt[-1]:
            return None
        bpm = float(np.interp(t_c, t_gt, gt.astype(np.float64)))
    if not (MIN_BPM <= bpm <= MAX_BPM) or not np.isfinite(bpm):
        return None
    return bpm


def _assign_splits(frame: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, rest_idx = next(gss.split(frame, groups=frame["patient_id"]))
    rest = frame.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_rel, test_rel = next(gss2.split(rest, groups=rest["patient_id"]))
    frame["split"] = "train"
    frame.iloc[rest_idx[val_rel], frame.columns.get_loc("split")] = "val"
    frame.iloc[rest_idx[test_rel], frame.columns.get_loc("split")] = "test"

    subjects = np.sort(frame["patient_id"].unique())
    subj_fold = {}
    dummy = np.zeros(len(subjects))
    for k, (_, te) in enumerate(GroupKFold(n_splits=n_folds).split(dummy, groups=subjects)):
        for s in subjects[te]:
            subj_fold[int(s)] = k
    frame["fold"] = frame["patient_id"].map(subj_fold)
    return frame


def prepare(win_sec: float, hop_sec: float, n_folds: int) -> pd.DataFrame:
    dataset = PPG_SPORT_ROOT / "dataset"
    gt_root = PPG_SPORT_ROOT / "cleaned_gt"
    if not dataset.is_dir():
        raise FileNotFoundError(f"Missing {dataset}")
    if not gt_root.is_dir():
        raise FileNotFoundError(f"Missing {gt_root}")

    if PPG_CACHE_DIR.exists():
        shutil.rmtree(PPG_CACHE_DIR)
    PPG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    win_64 = int(round(win_sec * PPG_FS))
    hop_64 = int(round(hop_sec * PPG_FS))
    if hop_64 < 1:
        raise ValueError(f"hop_sec={hop_sec} too small at {PPG_FS} Hz")

    ppg_files = sorted(dataset.glob("*_ppg.npy"))
    if not ppg_files:
        raise FileNotFoundError(f"No *_ppg.npy under {dataset}")

    rows = []
    skipped = 0
    for path in tqdm(ppg_files, desc="PPG-Sport windows"):
        m = PPG_NAME.match(path.stem)
        if m is None:
            skipped += 1
            continue
        subject, hand, activity = m.group(1), m.group(2), m.group(3)
        gt_path = gt_root / f"{subject}_{activity}_hr.npy"
        if not gt_path.exists():
            skipped += 1
            continue
        ppg = np.load(path).astype(np.float64, copy=False).reshape(-1)
        gt = np.load(gt_path).astype(np.float64, copy=False).reshape(-1)
        if ppg.size < win_64:
            skipped += 1
            continue
        ppg_ch = ppg[None, :]
        n_out = int(round(ppg.size * TARGET_FS / PPG_FS))
        full_500 = resample(ppg_ch, n_out, axis=1).astype(np.float32)
        win_500 = int(round(win_sec * TARGET_FS))

        for start_64 in range(0, ppg.size - win_64 + 1, hop_64):
            bpm = _hr_for_window(gt, start_64, win_64, hop_64)
            if bpm is None:
                skipped += 1
                continue
            start_500 = int(round(start_64 * TARGET_FS / PPG_FS))
            sl = full_500[:, start_500 : start_500 + win_500]
            if sl.shape[1] < win_500:
                skipped += 1
                continue
            stem = f"{subject}{hand}_{activity}_{start_64:05d}"
            npy_path = PPG_CACHE_DIR / f"{stem}.npy"
            np.save(npy_path, sl)
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
                }
            )

    if not rows:
        raise RuntimeError("No labelled PPG windows. Check dataset/*.npy and cleaned_gt.")

    frame = _assign_splits(pd.DataFrame(rows), n_folds=n_folds)
    PPG_HR_DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PPG_MANIFEST_PATH, index=False)
    print(frame["split"].value_counts().to_string())
    print("folds", frame.groupby("fold")["patient_id"].nunique().to_string())
    print(f"win_sec={win_sec} hop_sec={hop_sec} skipped={skipped}")
    print(f"Wrote {PPG_MANIFEST_PATH} n={len(frame)} mean_hr={frame['hr_bpm'].mean():.1f}")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--win_sec", type=float, default=PPG_WIN_SEC)
    parser.add_argument("--hop_sec", type=float, default=PPG_HOP_SEC)
    parser.add_argument("--cv_folds", type=int, default=PPG_CV_FOLDS)
    args = parser.parse_args()
    if PPG_MANIFEST_PATH.exists() and not args.force:
        print(f"Exists {PPG_MANIFEST_PATH} (use --force to rebuild)")
        return
    prepare(win_sec=args.win_sec, hop_sec=args.hop_sec, n_folds=args.cv_folds)


if __name__ == "__main__":
    main()
