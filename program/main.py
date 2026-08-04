# Default Column Labels:
# key values should be replaced to correspond with the column names in the data files
from retrievers.file_reader import FileReader

col_labels = {
    "Accelerometer X": "acc_x",
    "Accelerometer Y": "acc_y",
    "Accelerometer Z": "acc_z",
    "Gyroscope X": "gyro_x",
    "Gyroscope Y": "gyro_y",
    "Gyroscope Z": "gyro_z",
    "Electrocardiogram": "ecg",
    "Photoplethysmogram": "ppg"
}


if __name__ == "__main__":
    # Test .edf file
    data = FileReader.read_file_to_df("data/test_data/running_test.csv", add_time=True)
    print(data.head())