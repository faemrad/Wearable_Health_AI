from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from hr_detection.core.paths import WEIGHTS_DIR


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


def main():
    p = download_founder_1lead()
    print(p)


if __name__ == "__main__":
    main()
