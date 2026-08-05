# Default Column Labels:
# key values should be replaced to correspond with the column names in the data files
from unittest import signals

from retrievers.file_reader import FileReader
from processors.file_processor import FileProcessor
import matplotlib.pyplot as plt
import pandas as pd



import numpy as np

col_labels = {
    "time": "time",
    "acc_e4:x": "acc_x",
    "acc_e4:y": "acc_y",
    "acc_e4:z": "acc_z",
    "Gyroscope X": "gyro_x",
    "Gyroscope Y": "gyro_y",
    "Gyroscope Z": "gyro_z",
    "Shimmer_9696_ECG_LA-RA_24BIT_CAL": "ecg",
    "ppg:wrist": "ppg",
    "temp": "temp",
    "eda:dry": "eda",
}




if __name__ == "__main__":
    # Test .edf file
    data = FileReader.read_file_to_df("data/test_data/wrist_mix_test.edf", add_time=True)
    data = FileProcessor.process_file("data/test_data/wrist_mix_test.edf", col_labels, None)

    print(data.head())
    print(data.dtypes)



    plt.plot(data["time"][0:1000], data["ppg"][0:1000])
    
    plt.show()