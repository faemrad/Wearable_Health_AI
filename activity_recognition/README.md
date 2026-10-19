# Human Activity Recognition for ECG and PPG

To be able to run all scripts the following datasets must be downloaded. Files should be place in a top level ./data folder
- PPG-Sport: https://github.com/LaserHu/PPG-Sport
- ppg + dalia: https://archive.ics.uci.edu/dataset/495/ppg+dalia
- pulse-transit-time-ppg: https://physionet.org/content/pulse-transit-time-ppg/1.1.0/
- wrist-ppg-during-exercise: https://physionet.org/content/wrist/1.0.0/

If files are not downloaded and to only test model download and HAR skip to running eval

1. Ensure requirements have been installed for the entire HAR + HRE package | pip install -r requirements.txt
2. Run python -m activity_recognition.helpers.ppg_sport_preperation

Once these steps have been completed and assuming data has been stored as requested all of the scripts should be able to run via.
python -m activity_recognition.scripts.{SPECIFIC_SCRIPT}

## Script config file explained

All settings live in `activity_recognition/scripts/config.py`. They are shared by the training, linear probe, and evaluation scripts, so a change here affects all of them.

## Model parameters

These control how raw recordings are turned into model inputs. They affect training and the linear probe, and are saved into the model's `config.json` so evaluation uses the same preprocessing.

| Parameter | Type | Description |
|---|---|---|
| `WINDOW_SIZE` | int | Length of each window in seconds. |
| `OVERLAP` | int | Overlap between consecutive windows in seconds. Must be smaller than `WINDOW_SIZE`. |
| `FREQUENCY` | int | Target sampling rate in Hz. Recordings at any other rate are resampled to this before windowing. |
| `NORMALISE` | bool | Apply robust z-score normalisation (median/MAD, clipped to ±10). Flat segments become NaN and are dropped. |
| `RESCALE` | bool | Min-max rescale the signal to the range [-1, 1]. |
| `WINDOW_FIRST` | bool | `True`: split into windows, then normalise/rescale each window. `False`: normalise/rescale the whole recording, then split into windows. |
| `FREEZE_BACKBONE` | bool | `True`: freeze the pretrained backbone and train only the classification head. `False`: fine-tune everything. |
| `DATATYPE` | str | Signal to use: `"ecg"` or `"ppg"`. Used as the default by the evaluation script. |

### Training parameters

| Parameter | Type | Description |
|---|---|---|
| `N_EPOCHS` | int | Number of epochs to train before choosing a model. |
| `N_SPLITS` | int | Number of cross-validation splits. Splits are grouped by subject, so no subject appears in both train and test. |
| `BATCH_SIZE` | int | Batch size for training. |

### Hyperparameters

| Parameter | Type | Description |
|---|---|---|
| `BACKBONE_LEARNING_RATE` | float | Learning rate for the pretrained backbone. Ignored when `FREEZE_BACKBONE = True`. |
| `HEAD_LEARNING_RATE` | float | Learning rate for the classification head. |

### Final model

| Parameter | Type | Description |
|---|---|---|
| `SAVE_PATH` | str | Where the final trained model and its config are written. |

### Hugging Face repositories

| Parameter | Description |
|---|---|
| `HUBERT_REPO` | Hub repo holding the fine-tuned HuBERT-ECG weights and config. |
| `ECGFOUNDER_REPO` | Hub repo holding the fine-tuned ECGFounder weights and config. |
| `ECG_MODEL_NAME` | Weights file for the ECG model (`.safetensors`). |
| `PPG_MODEL_NAME` | Weights file for the PPG model (`.safetensors`). |
| `CONFIG_NAME` | Config file downloaded alongside the weights. |

### Saved `config.json`

Training writes a `config.json` next to the weights. Evaluation reads these keys, so the names must match exactly:

| Key | Comes from |
|---|---|
| `model_name` | `"HuBERT"` or `"ECGFounder"` |
| `n_classes` | Number of activity classes |
| `class_names` | Ordered list of activity labels |
| `dropout` | Classifier dropout (default 0.3 if missing) |
| `window` | `WINDOW_SIZE` |
| `overlap` | `OVERLAP` |
| `sampling_rate` | `FREQUENCY` |
| `normalise` | `NORMALISE` |
| `rescale` | `RESCALE` |
| `window_first` | `WINDOW_FIRST` |

If a key is missing, evaluation falls back to a default instead of failing, which can silently cause a train/eval mismatch.

## Running Evaluation

Once data has been obtained for activity classification the evaluations can occur. To do this the data first will need to be transformed to a numpy array.
There is a number of prediction options in eval_har_ECGFounder and eval_har_HuBERT for more information refer to their documentation.

To test the models are loading and predicting as expected you can run python -m activity_recognition.scriptes.test_load_and_eval. This should output something like this:
Running HuBERT evaluation on running ECG data:
{'cycling': 0.0672345935, 'running': 0.8721865471, 'stationary': 0.0066826368, 'walking': 0.0538962136}

Running HuBERT evaluation on stationary PPG data:
{'cycling': 0.152689104, 'running': 0.0537876987, 'stationary': 0.7455079169, 'walking': 0.0480152833}

Running ECGFounder evaluation on stationary ECG data:
{'cycling': 0.0993723207, 'running': 0.0507893467, 'stationary': 0.8149872463, 'walking': 0.0348510799}

Running ECGFounder evaluation on running PPG data:
{'cycling': 0.0338865304, 'running': 0.8924075057, 'stationary': 0.0204303786, 'walking': 0.0532755833}

You may get a warning:

- Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Download a HF_TOKEN from hugging_face and login for faster downloads. The program can still function without it however and it is not required