import numpy as np
import pandas as pd
import os

class PPGSportPreparation:
    def __init__(self, file_path: str = "./data/PPG-Sport/dataset", output_directory: str = "./data/PPG-Sport/converted_data", add_time: bool = False, frequency: int = 64, first_time=False):
        self.data_files = [f"{file}" for file in os.listdir(file_path) if file.endswith(".npy")]
        self.keys = self.get_unique_keys_ppg_sport(self.data_files)
        print("successfully created unique keys for the data files")
        if first_time:
            self.create_multiple_csv_from_npy(file_path, self.keys, output_directory=output_directory, add_time=add_time, frequency=frequency)
    
    @staticmethod
    def get_unique_keys_ppg_sport(file_paths: list) -> set:
        """
        Gets the unique keys from a list of dictionaries.

        Args:
            file_paths (list): A list of dictionaries where keys are the desired column names and values are the paths to the .npy files.

        Returns:
            set: A set of unique keys from the list of dictionaries.
        """
        key_list = dict()
        for file_path in file_paths:
            if file_path.endswith(".npy"):
                parts = file_path.split("_")
                key = parts[0][:3] + "_" + parts[1]
                category = parts[2][:-4]  # Remove the .npy extension
                if len(parts[0]) > 3:
                    category =  parts[0][3:] + "_" + category
                if key not in key_list:
                    key_list[key] = dict()
                
                key_list[key][category] = file_path
            else:
                pass

        return key_list



    @staticmethod
    def create_csv_from_npy(directory: str, file_paths: dict, output_path: str, add_time: bool = False, frequency: int = 100):
        """
        Creates a CSV file from multiple .npy files.

        Args:
            file_paths (dict): A dictionary where keys are the desired column names and values are the paths to the .npy files.
            output_path (str): The path where the output CSV file will be saved.

        Returns:
            None
        """
        columns = file_paths.keys()
        data_arrays = [np.load(f"{directory}/{file_paths[col]}") for col in columns]
        
        updated_cols = []
        for col in columns:
            if "imu" in col:
                updated_cols.extend([f"{col}_x", f"{col}_y", f"{col}_z"])
            else:
                updated_cols.append(col)
        columns = updated_cols

        combined = np.hstack(data_arrays)
        df = pd.DataFrame(combined, columns=columns)

        if add_time:
            time_array = np.arange(0, len(df) / frequency, 1 / frequency)
            df.insert(0, 'time', time_array)
        print(f"Saving DataFrame to {output_path}")
        df.to_csv(output_path, index=False)

        pass

    @staticmethod
    def create_multiple_csv_from_npy(directory: str, file_paths: dict[dict], output_directory: str = "./data/PPG-Sport/converted_data", add_time: bool = False, frequency: int = 64):
        """
        Creates multiple CSV files from multiple .npy files.

        Args:
            file_paths (dict[dict]): A dictionary where keys are the desired column names and values are the paths to the .npy files. 
            output_directory (str): The path where the output CSV files will be saved.
            Each key in the outer dictionary represents a separate CSV file to be created.
        
        Returns:
            None
        """
        for csv_name, paths in file_paths.items():
            PPGSportPreparation.create_csv_from_npy(directory, paths, f"{output_directory}/{csv_name}.csv", add_time, frequency)
        pass
