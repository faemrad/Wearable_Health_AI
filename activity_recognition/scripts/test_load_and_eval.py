from activity_recognition.scripts import eval_har_HuBERT
from activity_recognition.scripts import eval_har_ECGFounder
import numpy as np

def main():
    stationary_ecg = np.load("data/test_data/S01_stationary_ecg.npy")
    stationary_ppg = np.load("data/test_data/S01L_stationary_ppg.npy")
    running_ecg = np.load("data/test_data/S01_running_ecg.npy")
    running_ppg = np.load("data/test_data/S01L_running_ppg.npy")

    print("Running HuBERT evaluation on running ECG data:")
    result = eval_har_HuBERT.get_probability_by_activity(running_ecg, 64, "ecg")
    print(result)
    print()

    print("Running HuBERT evaluation on stationary PPG data:")
    result = eval_har_HuBERT.get_probability_by_activity(stationary_ppg, 64, "ppg")
    print(result)
    print()

    print("Running ECGFounder evaluation on stationary ECG data:")
    result = eval_har_ECGFounder.get_probability_by_activity(stationary_ecg, 64, "ecg")
    print(result)
    print()

    print("Running ECGFounder evaluation on running PPG data:")
    result = eval_har_ECGFounder.get_probability_by_activity(running_ppg, 64, "ppg")
    print(result)

if __name__ == "__main__":
    main()