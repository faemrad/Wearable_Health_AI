from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from hr_detection.core.paths import PULSEPPG_WEIGHTS, WEIGHTS_DIR

PULSEPPG_URL = (
    "https://zenodo.org/records/17345536/files/pulseppg_model_weights.zip?download=1"
)


def download_founder_1lead() -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / "1_lead_ECGFounder.pth"
    if dest.exists():
        return dest
    path = hf_hub_download(
        repo_id="PKUDigitalHealth/ECGFounder",
        filename="1_lead_ECGFounder.pth",
        local_dir=str(WEIGHTS_DIR),
    )
    return Path(path)


def _pulse_member(names: list[str]) -> str:
    candidates = [name for name in names if name.endswith("checkpoint_best.pkl")]
    preferred = [
        name
        for name in candidates
        if name.replace("\\", "/").endswith("out/pulseppg/checkpoint_best.pkl")
    ]
    if preferred:
        return sorted(preferred, key=len)[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        "pulseppg_model_weights.zip has no out/pulseppg/checkpoint_best.pkl. "
        f"Found: {candidates[:12]}"
    )


def download_pulseppg() -> Path:
    """Zenodo pretrained Pulse-PPG encoder → models/checkpoints/pulseppg/checkpoint_best.pkl."""
    dest = PULSEPPG_WEIGHTS
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / "pulseppg_model_weights.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 300_000_000:
        import urllib.request

        print("downloading", PULSEPPG_URL)
        urllib.request.urlretrieve(PULSEPPG_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        member = _pulse_member(archive.namelist())
        print("extract", member)
        with archive.open(member) as src, dest.open("wb") as out:
            out.write(src.read())
    zip_path.unlink(missing_ok=True)
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pulseppg", action="store_true", help="Download Pulse-PPG encoder weights")
    parser.add_argument("--founder", action="store_true", help="Download ECGFounder 1-lead weights")
    args = parser.parse_args()
    if not args.pulseppg and not args.founder:
        args.founder = True
    if args.founder:
        print(download_founder_1lead())
    if args.pulseppg:
        print(download_pulseppg())


if __name__ == "__main__":
    main()
