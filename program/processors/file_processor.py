from program.retrievers.file_reader import FileReader
import pandas as pd
class FileProcessor:
    @staticmethod
    def process_file(file_path: str, desired_columns: str, frequency: int) -> pd.DataFrame:
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

        # Resample the data to the desired frequency
        df = df.resample(f"{frequency}Hz").mean()

        return df