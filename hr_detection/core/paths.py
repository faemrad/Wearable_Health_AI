from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PTBXL_ROOT = REPO_ROOT / "data" / "ptb-xl"
HR_DATA = REPO_ROOT / "data" / "hr_detection"
CACHE_DIR = HR_DATA / "cache"
MANIFEST_PATH = HR_DATA / "manifest.csv"
WEIGHTS_DIR = REPO_ROOT / "models" / "checkpoints"
FOUNDER_DIR = REPO_ROOT / "models" / "ECGFounder"
HUBERT_DIR = REPO_ROOT / "models" / "HuBERT_ECG"
CKPT_DIR = REPO_ROOT / "hr_detection" / "checkpoints"

LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_FS = 500
WINDOW_SEC = 10
TARGET_LEN = TARGET_FS * WINDOW_SEC
HUBERT_CROP = 2500
HUBERT_DOWNSAMPLE = 5
