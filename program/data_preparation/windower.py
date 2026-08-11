import pandas as pd


class Windower:
    def __init__(self, window_size: int, overlap: int):
        """
        Initializes the Windower class.
        
        Args:
            window_size (int): The size of the window in seconds.
            overlap (int): The number of overlapping samples between consecutive windows in seconds.
        """
        self.window_size = window_size
        self.overlap = overlap

    def create_windows(self, data: pd.DataFrame, frequency: int = None) -> list:
        """
        Create overlapping windows from the input data.

        Args:
            data (pd.DataFrame): The input data to be windowed.
            frequency (int): The frequency of the data in Hz.

        Returns:
            list: A list of overlapping windows.
        """
        if frequency is None:
            if "time" in data.columns:
                time_diffs = data["time"].diff().dropna()
                if not time_diffs.empty:
                    frequency = round(1 / time_diffs.mean())
                else:
                    raise ValueError("Cannot determine frequency from the data.")

        window_length = self.window_size * frequency
        overlap_length = self.overlap * frequency
        step_size = window_length - overlap_length

        windows = []
        for start in range(0, len(data) - window_length + 1, step_size):
            end = start + window_length
            windows.append(data.iloc[start:end])

        return windows
    
    
