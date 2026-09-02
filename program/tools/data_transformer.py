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
    @staticmethod
    def get_processed_windows_from_file(filepath, file_frequency, desired_frequency, normalise=True) -> pd.DataFrame:
        data = FileReader.read_file_to_df(filepath, add_time=False)
        if "time" in data.columns:
            data = data.drop(columns=["time"])
            
        windowed_data = DataTransformer.window_data(data, 5, 1, file_frequency)
        resampled_data = DataTransformer.resample_window_list(100, desired_frequency, windowed_data)
        if normalise:
            normalised_data = DataTransformer.normalise_window_list(resampled_data)
            return normalised_data
        return resampled_data


    @staticmethod
    def window_data(df: pd.DataFrame, window_size: int, overlap: float, frequency: int) -> list:
        """
        Applies a windowing technique to the provided DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to be windowed.
            window_size (int): The size of the window.
            overlap (float): The fraction of overlap between consecutive windows.

        Returns:
            list: A list of windowed pd.DataFrame objects.
        """

        windowed_data = []
        windower = Windower(window_size, overlap)

        windows = windower.create_windows(df, frequency)

        for window in windows:
            windowed_data.append(window)

        return windowed_data


    @staticmethod
    def resample_data(new_fq, cur_fq, data: pd.DataFrame = None):
        """
        Resamples the data to a new frequency.

        Args:
            data (pd.DataFrame): The input pd.DataFrame to be resampled.
            new_fq (int): The new frequency in Hz.
            cur_fq (int): The current frequency in Hz.

        Returns:
            pd.DataFrame: The resampled pd.DataFrame.
        """

        g = gcd(new_fq, cur_fq)
        up, down = new_fq // g, cur_fq // g


        resampled_values = resample_poly(data, up, down, axis=0)  
        resampled_df = pd.DataFrame(resampled_values, columns=data.columns)

        return resampled_df

    @staticmethod
    def add_time_column(df: pd.DataFrame, frequency: int) -> pd.DataFrame:
        """
        Adds a time column to the DataFrame based on the specified frequency.

        Args:
            df (pd.DataFrame): The input DataFrame to which the time column will be added.
            frequency (int): The frequency of the data in Hz.

        Returns:
            pd.DataFrame: The DataFrame with the added time column.
        """
        n_samples = len(df)
        time_values = np.arange(n_samples) / frequency
        df_with_time = df.copy()
        df_with_time.insert(0, "time", time_values)

        return df_with_time

    @staticmethod
    def add_time_column_to_window_list(windowed_data: list[pd.DataFrame], frequency: int) -> list[pd.DataFrame]:
        """
        Adds a time column to each DataFrame in the windowed_data list based on the specified frequency.

        Args:
            windowed_data (list): A list of windowed pd.DataFrame objects.
            frequency (int): The frequency of the data in Hz.

        Returns:
            list: A list of windowed pd.DataFrame objects with the added time column.
        """
        windowed_data_with_time = []
        for window in windowed_data:
            window_with_time = DataTransformer.add_time_column(window, frequency)
            windowed_data_with_time.append(window_with_time)

        return windowed_data_with_time


    @staticmethod
    def resample_window_list(new_fq, cur_fq, windowed_data: list[pd.DataFrame] = None):
        """
        Resamples the data to a new frequency.

        Args:
            windowed_data (list): A list of windowed pd.DataFrame objects to be resampled.
            new_fq (int): The new frequency in Hz.
            cur_fq (int): The current frequency in Hz.

        Returns:
            list: A list of resampled pd.DataFrame objects.
        """
        

        resampled_windows = []
        for window in windowed_data:
            resampled_df = DataTransformer.resample_data(new_fq, cur_fq, window)            
            resampled_windows.append(resampled_df)

        return resampled_windows
        pass

    @staticmethod
    def normalise_data(windowed_data: np.ndarray = None, use_median: bool = False,
                    eps: float = 1e-8):

        if isinstance(windowed_data, np.ndarray):
            if use_median:
                center = np.median(windowed_data, axis=1, keepdims=True)
                scale = (np.percentile(windowed_data, 75, axis=1, keepdims=True)
                        - np.percentile(windowed_data, 25, axis=1, keepdims=True))
            else:
                center = np.mean(windowed_data, axis=1, keepdims=True)
                scale = np.std(windowed_data, axis=1, keepdims=True)

            normalised_data = (windowed_data - center) / (scale + eps)

            return normalised_data

        elif isinstance(windowed_data, pd.DataFrame):
            if use_median:
                center = windowed_data.median(axis=0)
                scale = windowed_data.quantile(0.75, axis=0) - windowed_data.quantile(0.25, axis=0)
            else:
                center = windowed_data.mean(axis=0)
                scale = windowed_data.std(axis=0)

            normalised_data = (windowed_data - center) / (scale + eps)

            return normalised_data

    @staticmethod
    def normalise_window_list(windowed_data: list[pd.DataFrame] = None, use_median: bool = False,
                    eps: float = 1e-8):

        normalised_windows = []
        for window in windowed_data:
            normalised_df = DataTransformer.normalise_data(window, use_median, eps)            
            normalised_windows.append(normalised_df)

        return normalised_windows


