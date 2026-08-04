# Default Column Labels:
# key values should be replaced to correspond with the column names in the data files
from unittest import signals

from retrievers.file_reader import FileReader
from processors.file_processor import FileProcessor
import matplotlib.pyplot as plt
import pandas as pd

import neurokit2 as nk

import numpy as np

col_labels = {
    "Shimmer_9696_Timestamp_Shimmer_CAL": "time",
    "Shimmer_9696_Accel_WR_X_CAL": "acc_x",
    "Shimmer_9696_Accel_WR_Y_CAL": "acc_y",
    "Shimmer_9696_Accel_WR_Z_CAL": "acc_z",
    "Gyroscope X": "gyro_x",
    "Gyroscope Y": "gyro_y",
    "Gyroscope Z": "gyro_z",
    "Shimmer_9696_ECG_LA-RA_24BIT_CAL": "ecg",
    "Photoplethysmogram": "ppg"
}




if __name__ == "__main__":
    # Test .edf file
    data = FileReader.read_file_to_df("data/test_data/running_test.csv", add_time=True)
    data = FileProcessor.process_file("data/test_data/running_test.csv", col_labels, 1)

    print(data.head())
    print(data.dtypes)

    cleaned = nk.ecg_clean(data["ecg"], sampling_rate=512, method="neurokit")
    #signals, info = nk.ecg_process(cleaned, sampling_rate=512)

    quality = nk.ecg_quality(cleaned, sampling_rate=512)
    #plt.plot(quality)

    #nk.ecg_plot(signals, info)
    data["ecg"] = cleaned

    plt.plot(data["time"][0:3000], data["ecg"][0:3000])
    
    plt.show()

    signals, info = nk.ecg_process(data["ecg"], sampling_rate=512)
    r_peaks = info["ECG_R_Peaks"]

    # Convert to actual time in ms and get intervals
    rr_intervals_ms = np.diff(r_peaks) / 512 * 1000
    print(rr_intervals_ms)
    print("Estimated HR (bpm):", 60000 / rr_intervals_ms)

    plt.plot(r_peaks[1:], 60000 / rr_intervals_ms)
    plt.xlabel("Sample")
    plt.ylabel("Heart rate (bpm)")
    plt.show()