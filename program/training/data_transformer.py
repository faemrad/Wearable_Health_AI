from program.data_preparation.windower import Windower
from program.processors.file_processor import FileProcessor
from program.retrievers.file_reader import FileReader
import os

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from math import gcd
from scipy.signal import resample_poly

class DataTransformer:
    def __init__(self, data_folder: str, model_type: str, frequency: int, column_labels: list[str]):
        self.data_folder = data_folder
        self.model_type = model_type
        self.frequency = frequency
        self.column_labels = column_labels

        self.get_data_files()
        

    def get_data_files(self) -> list:
        """
        Retrieves all CSV files from the specified data folder and stores them as pd.DataFrame objects in a list.

        Returns:
            list: A list of pd.DataFrame objects containing the data from each CSV file.
        """
        all_files = os.listdir(self.data_folder)
        self.data_files = dict()
        for file in all_files:
            df = FileReader.read_file_to_df(os.path.join(self.data_folder, file), add_time=True)
            self.data_files[file] = df

        
        return self.data_files

    def window_data(self, window_size: int, overlap: float):
        """
        Applies a windowing technique to the provided DataFrame.

        Args:
            window_size (int): The size of the window.
            overlap (float): The fraction of overlap between consecutive windows.

        Returns:
            list: A list of windowed pd.DataFrame objects.
        """

        self.windowed_data = []
        windower = Windower(window_size, overlap)

        for key in self.data_files.keys():
            data = self.data_files[key]
            trimmed_key = key.split(".")[0]
            trimmed_key = trimmed_key.split("_")
            user_id = trimmed_key[0]
            activity = trimmed_key[1]
            
            windows = windower.create_windows(data, self.frequency)

            for window in windows:
                new_dict = dict()
                new_dict["activity"] = activity
                new_dict["user_id"] = user_id
                new_dict["window"] = window
                self.windowed_data.append(new_dict)

        return self.windowed_data

    def get_ecg_data(self, window=5, overlap = 1, normalise=True):
        """
        Generates training, validation data and test data for the ECG dataset. 

        Args:
            window (int): The size of the window in seconds.
            overlap (int): The number of overlapping samples between consecutive windows in seconds.

        Returns:
            A pair of tuples containing the training and validation data for the ECG dataset.

        """
        data = self.window_data(window, overlap)

        X, y, groups = self.create_x_y(data, x_labels=["ecg"])
        self.encode_labels()
        self.split_participants()
        
        self.X = self.resample_data(100, self.frequency)
        if normalise:
            self.normalise_data()
        self.apply_split()
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)



    def create_x_y(self, transformed_data: list = None, activity_labels: list[str] = ["badminton", "basketball", "running", "stationary", "tabletennis", "walking"], x_labels = ["R_imu_x", "R_imu_y", "R_imu_z", "R_ppg", "ecg"]):
        """
        Creates the input (X) and output (y) data for training the model.

        Args:
            transformed_data (list): A list of transformed pd.DataFrame objects to be used for training the model.
            activity_labels (list): A list of activity labels corresponding to the transformed data.

        Returns:
            tuple: A tuple containing the input data (X) and output data (y) and participant IDs (groups).
        """
        if transformed_data is None:
            transformed_data = self.windowed_data

        x_list = []
        y_list = []
        groups_list = []

        for item in self.windowed_data:
            filtered_data = item["window"][x_labels].values
            x_list.append(filtered_data)
            y_list.append(item["activity"])
            groups_list.append(item["user_id"])

        X = np.array(x_list)
        y = np.array(y_list)
        groups = np.array(groups_list)

        self.X = X
        self.y = y
        self.groups = groups

        return (X, y, groups)

    def encode_labels(self):
        if not hasattr(self, "groups"):
            self.create_x_y()
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)
        self.label_encoder = le

    def activity_names(self):
        return self.label_encoder.classes_


    def split_participants(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Splits participant IDs into train/val/test sets based on ratios.
        
        groups: array of participant IDs (one per window, or just the unique list)
        train_ratio, val_ratio, test_ratio: must sum to 1.0
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        unique_participants = np.unique(self.groups)
        n = len(unique_participants)
        
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_participants)
        
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        
        self.train_ids = shuffled[:n_train]
        self.val_ids = shuffled[n_train:n_train + n_val]
        self.test_ids = shuffled[n_train + n_val:]
        
        return self.train_ids, self.val_ids, self.test_ids

    def apply_split(self):
        train_mask = np.isin(self.groups, self.train_ids)
        val_mask = np.isin(self.groups, self.val_ids)
        test_mask = np.isin(self.groups, self.test_ids)

        self.X_train, self.y_train = self.X[train_mask], self.y[train_mask]
        self.X_val, self.y_val = self.X[val_mask], self.y[val_mask]
        self.X_test, self.y_test = self.X[test_mask], self.y[test_mask]

    def resample_data(self, new_fq, cur_fq, windowed_data = None):
        """
        Resamples the data to a new frequency.

        Args:
            windowed_data (list): A list of windowed pd.DataFrame objects to be resampled.
            new_fq (int): The new frequency in Hz.
            cur_fq (int): The current frequency in Hz.

        Returns:
            list: A list of resampled pd.DataFrame objects.
        """
        g = gcd(new_fq, cur_fq)
        up, down = new_fq // g, cur_fq // g

        if isinstance(self.X, np.ndarray):
            resampled = resample_poly(self.X, up, down, axis=1)
            return resampled

        resampled_windows = []
        for window in windowed_data:
            original_shape = window.shape
            columns = window.columns

            resampled_values = resample_poly(window.values, up, down, axis=0)  

            resampled_df = pd.DataFrame(resampled_values, columns=columns)
            resampled_windows.append(resampled_df)

        return resampled_windows
        pass

    def normalise_data(self, windowed_data: np.ndarray = None, use_median: bool = False,
                    eps: float = 1e-8, store_stats: bool = False):
        if windowed_data is not None:
            self.X = windowed_data

        if isinstance(self.X, np.ndarray):
            if use_median:
                center = np.median(self.X, axis=1, keepdims=True)
                scale = (np.percentile(self.X, 75, axis=1, keepdims=True)
                        - np.percentile(self.X, 25, axis=1, keepdims=True))
            else:
                center = np.mean(self.X, axis=1, keepdims=True)
                scale = np.std(self.X, axis=1, keepdims=True)

            self.X = (self.X - center) / (scale + eps)

            if store_stats:
                # keep for de-normalization later (RevIN-style)
                self.norm_center = center
                self.norm_scale = scale

            return self.X


