from program.tools.file_reader import FileReader
import pandas as pd
import numpy as np

class FileProcessor:
    @staticmethod
    def process_file(file_path: str, desired_columns: str, current_frequency: int) -> pd.DataFrame:
        """
        Processes the data from the file and returns a pandas dataframe with the desired name and number of columns and adjusted frequency.

        Args:
            file_path (str): The path to the file to process.
            desired_columns (dict): The name of the current columns as keys and the desired names as values.
            frequency (int): The desired frequency of the data.

        Returns:
            pd.DataFrame: A pandas dataframe containing the processed data.
        
        """
        # Read the file into a DataFrame
        df = FileReader.read_file_to_df(file_path, add_time=True)

        # Rename the columns
        df = df.rename(columns=desired_columns)
        for col in df.columns:
            if col not in desired_columns.values():
                df = df.drop(columns=[col])

        df = FileProcessor._drop_units_row_if_present(df)
        df = FileProcessor._convert_numeric_columns(df)


        if "time" not in df.columns:
            rate = 1 / current_frequency
            df.insert(0, "time", np.arange(len(df)) * rate)

        elif "time" in df.columns and df["time"].iloc[0] != 0:
            df["time"] = df["time"] - df["time"].iloc[0]

        return df

    @staticmethod
    def _drop_units_row_if_present(df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks whether the first row is a units row (e.g. 'Ticks', 'm/(s^2)', 'mV')
        rather than actual data, and drops it if so.

        A row is treated as a units row only if EVERY value in it fails to
        convert to a number — a real data row should have at least one numeric value.
        """
        if len(df) == 0:
            return df

        first_row = df.iloc[0]
        numeric_first_row = pd.to_numeric(first_row, errors="coerce")

        if numeric_first_row.isna().all():
            print("Warning: units row detected and removed from data")
            df = df.iloc[1:].reset_index(drop=True)

        return df


    @staticmethod
    def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts columns to numeric dtype where possible. Columns that are
        non-numeric (e.g. labels) are left untouched.
        """
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")

            if converted.notna().sum() >= df[col].notna().sum() - 1:
                df[col] = converted
        return df