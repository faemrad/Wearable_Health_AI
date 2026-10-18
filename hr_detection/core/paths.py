from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PTBXL_ROOT = REPO_ROOT / "data" / "ptb-xl"
HR_DATA = REPO_ROOT / "data" / "hr_detection"
CACHE_DIR = HR_DATA / "cache"
MANIFEST_PATH = HR_DATA / "manifest.csv"
PPG_SPORT_ROOT = REPO_ROOT / "data" / "PPG-Sport"
PPG_HR_DATA = REPO_ROOT / "data" / "hr_detection_ppg"
PPG_CACHE_DIR = PPG_HR_DATA / "cache"
PPG_MANIFEST_PATH = PPG_HR_DATA / "manifest.csv"
PPG_FS = 64
PPG_WIN_SEC = 8.0
PPG_HOP_SEC = 2.0
PPG_CV_FOLDS = 6
WEIGHTS_DIR = REPO_ROOT / "models" / "checkpoints"
FOUNDER_DIR = REPO_ROOT / "models" / "ECGFounder"
HUBERT_DIR = REPO_ROOT / "models" / "HuBERT_ECG"
PULSEPPG_DIR = REPO_ROOT / "models" / "PulsePPG"
PULSEPPG_FS = 50
PULSEPPG_WEIGHTS = WEIGHTS_DIR / "pulseppg" / "checkpoint_best.pkl"
CKPT_DIR = REPO_ROOT / "hr_detection" / "checkpoints"

LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_FS = 500
WINDOW_SEC = 10
TARGET_LEN = TARGET_FS * WINDOW_SEC
HUBERT_CROP = 2500
HUBERT_DOWNSAMPLE = 5
