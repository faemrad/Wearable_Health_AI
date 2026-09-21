from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from hr_detection.core.paths import CACHE_DIR, HR_DATA, LEAD_ORDER, MANIFEST_PATH, PTBXL_ROOT, REPO_ROOT
from hr_detection.core.preprocess import upsample_to_500


def _reorder_leads(sig: np.ndarray, names: list[str]) -> np.ndarray:
    names_u = [n.upper() for n in names]
    alias = {"AVR": "AVR", "AVL": "AVL", "AVF": "AVF", "RA": "AVR"}
    lookup = {}
    for i, n in enumerate(names_u):
        lookup[n] = i
        lookup[alias.get(n, n)] = i
    out = np.zeros((12, sig.shape[0]), dtype=np.float32)
    for j, lead in enumerate(LEAD_ORDER):
        idx = lookup.get(lead)
        if idx is None:
            continue
        out[j] = sig[:, idx]
    return out


def _heart_rate_bpm(lead_ii: np.ndarray, fs: int) -> float | None:
    import neurokit2 as nk

    try:
        cleaned = nk.ecg_clean(lead_ii, sampling_rate=fs)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        if len(peaks) < 3:
            return None
        rr = np.diff(peaks) / float(fs)
        rr = rr[(rr > 0.3) & (rr < 2.0)]
        if len(rr) < 2:
            return None
        bpm = float(60.0 / np.median(rr))
        if not (30.0 <= bpm <= 220.0):
            return None
        return bpm
    except Exception:
        return None


def _complete_records() -> list[Path]:
    records100 = PTBXL_ROOT / "records100"
    if not records100.is_dir():
        raise FileNotFoundError(f"Missing {records100}")
    out = []
    for hea in sorted(records100.rglob("*.hea")):
        if hea.with_suffix(".dat").exists():
            out.append(hea.with_suffix(""))
    return out


def prepare() -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    db = pd.read_csv(PTBXL_ROOT / "ptbxl_database.csv")
    db["stem"] = db["filename_lr"].map(lambda p: Path(p).name.replace("_lr", ""))

    rows = []
    for rec_path in tqdm(_complete_records(), desc="PTB-XL windows"):
        rec = wfdb.rdrecord(str(rec_path))
        fs = int(rec.fs)
        sig = np.asarray(rec.p_signal, dtype=np.float64)
        names = [str(n) for n in rec.sig_name]
        ecg12 = _reorder_leads(sig, names)
        ecg500 = upsample_to_500(ecg12, fs)
        bpm = _heart_rate_bpm(ecg500[1], 500)
        if bpm is None:
            continue
        stem = rec_path.name.replace("_lr", "")
        npy_path = CACHE_DIR / f"{stem}.npy"
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
            }
        )

    if not rows:
        raise RuntimeError("No labelled windows. Check records100/*.dat and NeuroKit peaks.")

    frame = pd.DataFrame(rows)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, rest_idx = next(gss.split(frame, groups=frame["patient_id"]))
    rest = frame.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_rel, test_rel = next(gss2.split(rest, groups=rest["patient_id"]))
    frame["split"] = "train"
    frame.iloc[rest_idx[val_rel], frame.columns.get_loc("split")] = "val"
    frame.iloc[rest_idx[test_rel], frame.columns.get_loc("split")] = "test"
    HR_DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MANIFEST_PATH, index=False)
    print(frame["split"].value_counts().to_string())
    print(f"Wrote {MANIFEST_PATH} n={len(frame)} mean_hr={frame['hr_bpm'].mean():.1f}")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if MANIFEST_PATH.exists() and not args.force:
        print(f"Exists {MANIFEST_PATH} (use --force to rebuild)")
        return
    prepare()


if __name__ == "__main__":
    main()
