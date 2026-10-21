# Wearable Health AI

Processes wearable sensor data (ECG & PPG) and extracts key features:

- **Heart rate estimation (HRE):** regress bpm from ECG (PTB-XL) or wrist PPG (PPG-Sport).
- **Human activity recognition (HAR):** classify activity from ECG / PPG windows from various datasets.

Both tasks fine-tune pretrained foundation models (ECGFounder, HuBERT-ECG & Pulse-PPG).

## Repository layout

```text
Wearable_Health_AI/
  hr_detection/           heart-rate estimation (see hr_detection/RUN_EVERYTHING.md)
  activity_recognition/   activity recognition (see activity_recognition/README.md)
  models/                 pretrained model code / weights (ECGFounder, HuBERT_ECG, PulsePPG)
  data/                   datasets (not committed, see below)
  requirements.txt
```

## Datasets

Place all datasets under `./data/`.

| Dataset (folder name)     | Used for      | Download                                                                                                               | Path              |
| ------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------- |
| PPG-Sport                 | HR (PPG), HAR | [GitHub](https://github.com/LaserHu/PPG-Sport)                                                                         | `data/PPG-Sport/` |
| PTB-XL                    | HR (ECG)      | [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)                                                               | `data/ptb-xl/`    |
| PPG + Dalia               | HAR           | [Ghttps://archive.ics.uci.edu/dataset/495/ppg+dalia](https://github.com/netd-tud/human-activity-Recognition)           | `data/`           |
| Wrist PPG During Exercise | HAR           | [https://physionet.org/content/wrist/1.0.0/](https://physionet.org/content/wrist/1.0.0/)                               | `data/`           |
| Pulse Transit Time PPG    | HAR           | [Ghttps://physionet.org/content/pulse-transit-time-ppg/1.1.0/](https://github.com/netd-tud/human-activity-Recognition) | `data/`           |

Extra HAR datasets (PPG-DaLiA, pulse-transit-time-ppg, wrist-ppg-during-exercise) are listed in `activity_recognition/README.md`. These can be added manually if required.

## Setup

Run all commands from the repo root (`Wearable_Health_AI/`).

```bash
git submodule update --init --recursive # Download the foundation models we are using
pip install -r requirements.txt
```

Convert PPG-Sport `.npy` files to CSV (writes to `data/PPG-Sport/converted_data/`):

```python
from activity_recognition.helpers.ppg_sport_preparation import PPGSportPreparation

PPGSportPreparation(add_time=True, first_time=True)
```

## Running

Refer to the READMEs/documentation in `hr_detection/` and `activity_recognition/` folders for information on how to run those two things separately.

**Summary:**

- Heart rate:
  - `hr_detection/RUN_EVERYTHING.md` (all four models end to end, plus pre-trained weights).
  - `hr_detection/README_ECG.md` (ECG)
  - `hr_detection/README_PPG.md` (PPG)
- Activity recognition
  - `activity_recognition/README.md`.

Scripts run as modules
