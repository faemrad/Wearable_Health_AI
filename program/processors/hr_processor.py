import pandas as pd

class HRProcessor:
    def __init__():
        pass
    @staticmethod
    def process_ecg_data(ecg_data: pd.DataFrame, frequency: int):
        """
        Method for processing ECG data to extract heart rate information

        Args:
            ecg_data (pd.DataFrame): DataFrame containing ECG data

        Returns:
            pd.DataFrame: DataFrame containing heart rate information
            int: heart rate variability

        """
        pass
    @staticmethod
    def process_ppg_data(ppg_data: pd.DataFrame, frequency: int):
        """
        Method for processing PPG data to extract heart rate information

        Args:
            ppg_data (pd.DataFrame): DataFrame containing PPG data

        Returns:
            pd.DataFrame: DataFrame containing heart rate information
            int: heart rate variability

        """
        pass

    @staticmethod
    def get_hr_data(data: pd.DataFrame, frequency: int, use_ecg: bool):
        """
        Method for getting heart rate data

        Args:
            data (pd.DataFrame): DataFrame containing data to process
            frequency (int): Frequency of the data
            use_ecg (bool): Whether to use ECG data

        Returns:
            pd.DataFrame: DataFrame containing heart rate information
            int: heart rate variability


        """
        if use_ecg:
            return HRProcessor.process_ecg_data(data, frequency)
        else:
            return HRProcessor.process_ppg_data(data, frequency)
        pass

