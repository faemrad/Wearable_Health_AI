import pandas as pd
import mne
import wfdb
import numpy as np

class FileReader:
    @staticmethod
    def read_file_to_df(file_path: str, add_time: bool = False) -> pd.DataFrame:
        """
        Gets data from file and combines it into a pandas df. Data is automatically processed.

        Input:
            - file_path: Either the relative path or the full path to the data_file being opened.
            - add_time: Adds a column containing the time when the measurement was taken.
        Output:
            - a pandas dataframe created from the data files.

        """
        if file_path[-4:] == ".dat":
            df = FileReader.dat_to_df(file_path, add_time)
            return df
        elif file_path[-4:] == ".csv":
            df = FileReader.csv_to_df(file_path)
            return df
        elif file_path[-4:] == ".edf":
            df = FileReader.edf_to_df(file_path)
            return df
        else:
            print("Incorrect file format, must be .dat, .csv or .edf file")
            raise ValueError("Unsupported file format")
        
    @staticmethod
    def edf_to_df(file_path: str) -> pd.DataFrame:
        """
        Gets data from edf file to pandas dataframe. Data may need to be processed after being imported.
        

        Input:
            - file_path: The file path without the ending i.e
                    (./file_name NOT ./file_name.dat or ./file_name.hea)

        Output:
            - a pandas dataframe created from the edf file.
        """
        if file_path[-4:] == ".edf":
            try:
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                df = raw.to_data_frame()
                return df
            
            except Exception as e:
                print(f"An error occurred whilst trying to open the file: {e}")
                raise e
        else:
            print("Incorrect file format, must by .edf file")
            raise
            

    @staticmethod
    def dat_to_df(file_path: str, add_time: bool = False) -> pd.DataFrame:
        """
        Gets data from .dat and .hea file and combines it into a pandas df. Data is automatically processed.

        Input:
            - file_path: Either the relative path or the full path to the data_file being opened.
            - add_time: Adds a column containing the time when the measurement was taken.

        Output:
            - a pandas dataframe created from the data files.
        """

        record = wfdb.rdrecord(file_path)
        df = pd.DataFrame(record.p_signal, columns=record.sig_name)
        if add_time:
            df.insert(0, 'time', np.arange(len(df)) / record.fs)
            
        return df

    @staticmethod
    def csv_to_df(file_path: str) -> pd.DataFrame:
        """
        Gets data from .csv file and combines it into a pandas df. Data is automatically processed.

        Input:
            - file_path: Either the relative path or the full path to the data_file being opened.

        Output:
            - a pandas dataframe created from the data files.
        """
        df = pd.read_csv(file_path)
        if len(df.columns) == 1:
            seperator = df.columns[0]
            print("Warning: CSV file has only one column, attempting to read with different separator")
            if seperator == "sep=\t":
                df = pd.read_csv(file_path, sep="\t", skiprows=1)
            else:
                raise TypeError("CSV file has only one column, seperator was not recognised, please check the file and try again")
        return df