import os
from activity_recognition.helpers.file_reader import FileReader
from activity_recognition.helpers.file_processor import FileProcessor

from scipy.signal import resample_poly
from math import gcd
import numpy as np
import pandas as pd

DATA_FOLDER_PATH = "./data"
PPG_SPORT_FQ = 64
PTT_PPG_FQ = 500
PPG_DALIA_PPG_FQ = 64
PPG_DALIA_ACC_FQ = 32
PPG_DALIA_ACT_FQ = 4
PPG_DALIA_ECG_FQ = 700
WRIST_PPG_FQ = 256

TARGET_FQ = 100

class DataCombiner:
    """
    Class to combine and load data from different sources
    """
    def __init__(self):
        pass

    def get_all_data(self) -> list[pd.DataFrame]:
        """
        Returns all data from different sources. Resamples all data to target frequency
        Returns:
            all_data: list of dataframes

                
        Each dataframe has the following columns:
            dataset: str
            subject: str
            activity: str
            dataframe: pd.DataFrame where each row is a datapoint
        """
        all_data = []

        all_data.extend(self.get_ppg_sport())
        all_data.extend(self.get_ppg_dalia())
        all_data.extend(self.get_wrist_ppg_during_exercise())
        all_data.extend(self.get_ptt_ppg_data())
        return all_data

    def get_ppg_sport(self) -> list[pd.DataFrame]:
        """
        Gets data from the ppg-sport dataset

        Returns:
            all_data: list of dataframes
        
        Each dataframe has the following columns:
            dataset: str
            subject: str
            activity: str
            dataframe: pd.DataFrame where each row is a datapoint
        """
        folder = os.path.dirname(f"{DATA_FOLDER_PATH}/ppg-sport/converted_data/")
        all_data = []
        for file in os.listdir(folder):
            if file.endswith(".csv") and file != "subjects_info.csv":
                file_path = os.path.join(folder, file)
                df = FileReader.read_file_to_df(file_path, add_time=True)
                df = FileProcessor._convert_numeric_columns(df)
                df = FileProcessor._drop_units_row_if_present(df)
                keep_cols = ["ecg", "L_imu_x", "L_imu_y", "L_imu_z", "L_ppg"]
                df = df.loc[:, df.columns.intersection(keep_cols)]
                df.rename(columns={"L_ppg": "ppg", "L_imu_x": "imu_x", "L_imu_y": "imu_y", "L_imu_z": "imu_z"}, inplace=True)

                subject = file.split("_")[0]
                activity = file.split("_")[1]
                activity = activity.split(".")[0]

                df = resample_dataframe(df, PPG_SPORT_FQ, TARGET_FQ)

                datapoint = {
                    "dataset": "ppg-sport",
                    "subject": subject,
                    "activity": activity,
                    "dataframe": df
                }
                all_data.append(datapoint)

        return all_data

    def get_ptt_ppg_data(self):
        """
        Extracts data from the pulse-transit-time-ppg dataset

        Returns:
            all_data: list of dataframes
        """
        folder = os.path.dirname(f"{DATA_FOLDER_PATH}/pulse-transit-time-ppg/1.1.0/csv/")
        all_data = []
        for file in os.listdir(folder):
            if file.endswith(".csv")  and file != "subjects_info.csv":
                subject = file.split("_")[0]
                activity = file.split("_")[1]
                activity = activity.split(".")[0]
                file_path = os.path.join(folder, file)
                df = FileReader.read_file_to_df(file_path, add_time=True)
                df = FileProcessor._convert_numeric_columns(df)
                df = FileProcessor._drop_units_row_if_present(df)
                keep_cols = ["ecg", "a_x", "a_y", "a_z", "g_x", "g_y", "g_z"]
                df = df.loc[:, df.columns.intersection(keep_cols)]
                df.rename(columns={"a_x": "acc_x", "a_y": "acc_y", "a_z": "acc_z", "g_x": "gyro_x", "g_y": "gyro_y", "g_z": "gyro_z"}, inplace=True)
                df = resample_dataframe(df, PTT_PPG_FQ, TARGET_FQ)

                datapoint = {
                    "dataset": "ptt-ppg",
                    "subject": subject,
                    "activity": activity,
                    "dataframe": df
                }

                all_data.append(datapoint)

        return all_data


    def get_ppg_dalia(self):
        """
        Extracts data from the PPG_FieldStudy dataset. Only sitting, outdoor-cycle, walking and working activities.

        Returns:
            all_data: list of dataframes
        """
        main_folder = os.path.dirname(f"{DATA_FOLDER_PATH}/ppg+dalia/data/PPG_FieldStudy/")
        int_to_activity = {
            1: "sitting",
            4: "outdoor-cycle",
            7: "walking",
            8: "working"
        }
        all_data = []
        for folder in os.listdir(main_folder):
            if "." not in folder:
                folder_path = os.path.join(main_folder, folder)

                for file in os.listdir(folder_path):
                    if file.endswith(".pkl"):
                        file_path = os.path.join(folder_path, file)
                        df = FileReader.pkl_to_dict(file_path)

                        wrist_data = df["signal"]["wrist"]
                        chest_data = df["signal"]["chest"]
                        
                        ppg_data = wrist_data["BVP"]
                        ecg_data = chest_data["ECG"]
                        acc_x, acc_y, acc_z = wrist_data["ACC"].T

                        initial = extract_activity_start_end(df["activity"])
                        activity = df["activity"]
                        resampled_activity = resample_activity(activity, PPG_DALIA_ACT_FQ, TARGET_FQ)

                        duration_sec = len(ecg_data) / PPG_DALIA_ECG_FQ
                        target_len = round(duration_sec * 100)

                        ecg_data = resample_to_length(ecg_data, PPG_DALIA_ECG_FQ, TARGET_FQ, target_len)
                        ppg_data = resample_to_length(ppg_data, PPG_DALIA_PPG_FQ, TARGET_FQ, target_len)
                        acc_x = resample_to_length(acc_x, PPG_DALIA_ACC_FQ, TARGET_FQ, target_len)
                        acc_y = resample_to_length(acc_y, PPG_DALIA_ACC_FQ, TARGET_FQ, target_len)
                        acc_z = resample_to_length(acc_z, PPG_DALIA_ACC_FQ, TARGET_FQ, target_len)

                        combined_df = pd.DataFrame()
                        combined_df["acc_x"] = acc_x
                        combined_df["acc_y"] = acc_y
                        combined_df["acc_z"] = acc_z
                        combined_df["ecg"] = ecg_data
                        combined_df["ppg"] = ppg_data
                        combined_df["activity"] = resampled_activity
                        combined_df["group_id"] = (
                            combined_df["activity"] != combined_df["activity"].shift()
                        ).cumsum()


                        for (activity_code, group_id), activity_df in combined_df.groupby(["activity", "group_id"]):
                            if activity_code not in int_to_activity:
                                continue  # skip untracked codes like transitions (0), stairs, etc.

                            activity_name = int_to_activity[activity_code]

                            datapoint = {
                                "dataset": "ppg+dalia",
                                "subject": folder,
                                "activity": activity_name,
                                "dataframe": activity_df.drop(columns="group_id"),
                            }
                            all_data.append(datapoint)
                         

        return all_data
        pass

    def get_wrist_ppg_during_exercise(self):
        """
        Extracts data from the wrist-ppg-during-exercise dataset.

        Returns:
            all_data: list of dataframes
        """
        folder_path = f"{DATA_FOLDER_PATH}/wrist-ppg-during-exercise/1.0.0/"
        all_data = []
        for file in os.listdir(folder_path):
            if file.endswith(".dat"):
                file_path = os.path.join(folder_path, file)
                df = FileReader.read_file_to_df(file_path, add_time=True)
                df = FileProcessor._convert_numeric_columns(df)
                df = FileProcessor._drop_units_row_if_present(df)

                keep_cols = ["chest_ecg", "wrist_ppg", "wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z", "wrist_low_noise_accelerometer_x", 
                             "wrist_low_noise_accelerometer_y", "wrist_low_noise_accelerometer_z", "wrist_wide_range_accelerometer_x", 
                             "wrist_wide_range_accelerometer_y", "wrist_wide_range_accelerometer_z"]
                df = df.loc[:, df.columns.intersection(keep_cols)]
                df.rename(columns={"chest_ecg": "ecg", "wrist_ppg": "ppg", "wrist_gyro_x": "gyro_x", "wrist_gyro_y": "gyro_y", "wrist_gyro_z": "gyro_z",
                                   "wrist_low_noise_accelerometer_x": "low_noise_acc_x", "wrist_low_noise_accelerometer_y": "low_noise_acc_y", "wrist_low_noise_accelerometer_z": "low_noise_acc_z", 
                                   "wrist_wide_range_accelerometer_x": "wide_range_acc_x", "wrist_wide_range_accelerometer_y": "wide_range_acc_y", "wrist_wide_range_accelerometer_z": "wide_range_acc_z"}, inplace=True)

                df = resample_dataframe(df, WRIST_PPG_FQ, TARGET_FQ)
  
                datapoint = {
                    "dataset": "wrist-ppg",
                    "subject": file.split("_")[0],
                    "activity": file.split("_")[-1][:-4],
                    "dataframe": df
                }
                all_data.append(datapoint)

        return all_data


def resample_data(array, cur_frequency, new_frequency):
    """
    Resamples an array to a new frequency.

    Args:
        array (np.ndarray): The input array to be resampled.
        cur_frequency (int): The current frequency in Hz.
        new_frequency (int): The new frequency in Hz.

    Returns:
        np.ndarray: The resampled array.
    """
    g = gcd(cur_frequency, new_frequency)
    up, down = new_frequency // g, cur_frequency // g
    resampled_array = resample_poly(array, up, down)
    return resampled_array
    pass

def resample_to_length(array, cur_fq, new_fq, target_len):
    """
    Resamples an array to new target length.

    Args:
        array (np.ndarray): The input array to be resampled.
        cur_fq (int): The current frequency in Hz.
        new_fq (int): The new frequency in Hz.
        target_len (int): The target length of the resampled array.

    Returns:
        np.ndarray: The resampled array.
    """
    g = gcd(cur_fq, new_fq)
    up, down = new_fq // g, cur_fq // g
    resampled = resample_poly(array, up, down)

    diff = target_len - len(resampled)
    if diff > 0:
        resampled = np.pad(resampled, (0, diff), mode='edge')
    elif diff < 0:
        resampled = resampled[:target_len]
    return resampled

def resample_dataframe(df, cur_fq, new_fq, categorical_cols=None):
    """
    Resamples a pd.DataFrame to a new frequency.

    Args:
        df (pd.DataFrame): The input pd.DataFrame to be resampled.
        cur_fq (int): The current frequency in Hz.
        new_fq (int): The new frequency in Hz.

    Returns:
        pd.DataFrame: The resampled pd.DataFrame.
    """
    categorical_cols = categorical_cols or []
    g = gcd(cur_fq, new_fq)
    up, down = new_fq // g, cur_fq // g

    resampled = {}
    for col in df.columns:
        if col in categorical_cols:

            val = new_fq // cur_fq
            resampled[col] = np.repeat(df[col].values, val) if val == new_fq / cur_fq \
                else np.repeat(df[col].values, up)[::down]  
        else:
            resampled[col] = resample_poly(df[col].values, up, down)

   
    min_len = min(len(v) for v in resampled.values())
    resampled = {k: v[:min_len] for k, v in resampled.items()}

    return pd.DataFrame(resampled)

def extract_activity_start_end(array):
    """
    Extracts the start and end indices of each activity in an array. Only used for PPG-Dalia, where all activities for a subject where in one file.

    array (np.ndarray): The input array to be processed.

    Returns:
        dict: A dictionary containing the start and end indices of each activity.
    """
    array = np.asarray(array).flatten()
    start_end = dict()
    start_end[1] = dict()
    start_end[4] = dict()
    start_end[7] = dict()
    start_end[8] = dict()
    for i in range(len(array)-1):
        cur = int(array[i])
        next = int(array[i+1])
        if cur != next:
            if cur in start_end.keys():
                start_end[cur]['end'] = i
            if next in start_end.keys():
                start_end[next]['start'] = i+1
        if i + 1== len(array)-1:
            if next in start_end.keys():
                start_end[next]['end'] = i + 1
        if i == 0 and cur in start_end.keys():
            start_end[cur]['start'] = 0
    return start_end

def adj_start_end(start_end, cur_fq, new_fq):
    """
    Adjusts the start and end indices of each activity in a dictionary.

    Args:
        start_end (dict): A dictionary containing the start and end indices of each activity.
        cur_fq (int): The current frequency in Hz.
        new_fq (int): The new frequency in Hz.

    Returns:
        dict: A dictionary containing the adjusted start and end indices of each activity.
    """
    val = new_fq // cur_fq
    if val != (new_fq / cur_fq):
        raise ValueError(f"New frequency {new_fq} is not divisible by current frequency {cur_fq}.")
    for key in start_end.keys():
        start_end[key]['start'] = start_end[key]['start'] * val
        start_end[key]['end'] = ((start_end[key]['end'] +  1)* val) - 1
    return start_end

def resample_activity(array, cur_fq, new_fq):
    """
    Resamples activity label from PPG-Dalia to be consistent with other datapoints.

    Args:
        array (np.ndarray): The input array to be resampled.
        cur_fq (int): The current frequency in Hz.
        new_fq (int): The new frequency in Hz.

    Returns:
        np.ndarray: The resampled array.
    """
    val = new_fq // cur_fq
    if val != (new_fq / cur_fq):
        raise ValueError(f"{new_fq} not evenly divisible by {cur_fq}")
    return np.repeat(array, val)
