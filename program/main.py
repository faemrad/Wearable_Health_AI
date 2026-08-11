# Default Column Labels:
# key values should be replaced to correspond with the column names in the data files
from unittest import signals

from data_preparation.ppg_sport_preparation import PPGSportPreparation
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
    preperation = PPGSportPreparation(add_time=True)
    

