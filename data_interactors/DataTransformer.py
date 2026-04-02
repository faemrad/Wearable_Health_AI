import pandas as pd
import matplotlib.pyplot as plt

class DataTransformer:
    def __init__(self):
        pass

    @staticmethod
    def save_df_to_graph(df: pd.DataFrame, filesave_path: str, xkey: str = "time", ykey: str = "ECG") -> None:
        """
        A function for saving a graph of the data in a dataframe. The graph is saved as a .png file.
        Input:
            - df: The dataframe containing the data to be graphed.
            - filesave_path: The path where the graph will be saved.
            - xkey: The column name of the dataframe to be used as the x-axis. Default is "time".
            - ykey: The column name of the dataframe to be used as the y-axis. Default is "ECG".
        """
        if (xkey not in df.keys()) or (ykey not in df.keys()):
            raise KeyError()
        
        start_time = df[xkey].iloc[0]
        x = df[xkey] - start_time
        y = df[ykey]

        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xkey)
        plt.ylabel(ykey)
        plt.title(f"{ykey} vs {xkey}")
        plt.savefig(filesave_path)
        plt.close()

    @staticmethod
    def divide_dataframe(df: pd.DataFrame, chunk_time: float, timekey: str = "time") -> list[pd.DataFrame]:
        """
        A function for dividing a dataframe into smaller dataframes of a specified size. The last dataframe may be smaller than the specified size if the original dataframe cannot be evenly divided.

        Input:
            - df: The dataframe to be divided.
            - chunk_time: The time duration of each chunk in seconds.

        Output:
            - A list of dataframes, each containing a chunk of the original dataframe.
        """
        
        if timekey not in df.keys():
            raise KeyError(f"Column '{timekey}' not found in dataframe.")
    
        time_diffs = df[timekey].diff().dropna()
        
        if (time_diffs.std() / time_diffs.mean()) > 1e-3:
            start_time = df[timekey].iloc[0]
            end_time = df[timekey].iloc[-1]
            chunks = []
            current_start = start_time

            while current_start < end_time:
                current_end = current_start + chunk_time
                chunk = df[(df[timekey] >= current_start) & (df[timekey] < current_end)]
                if not chunk.empty:
                    chunks.append(chunk.reset_index(drop=True))
                current_start = current_end

        else:
            fs = 1 / time_diffs.mean()
            chunk_size = int(chunk_time * fs)
            chunks = [df.iloc[i:i + chunk_size].reset_index(drop=True)
                    for i in range(0, len(df), chunk_size)]
        
        return chunks
    
    @staticmethod
    def save_chunks_to_graphs(chunks: list[pd.DataFrame], filesave_path: str, xkey: str = "time", ykey: str = "ECG") -> None:
        """
        A function for saving a graph of the data in a list of dataframes. The graphs are saved as .png files with the name format "filesave_path_chunkindex.png".

        Input:
            - chunks: A list of dataframes containing the data to be graphed.
            - filesave_path: The path where the graphs will be saved.
            - xkey: The column name of the dataframe to be used as the x-axis. Default is "time".
            - ykey: The column name of the dataframe to be used as the y-axis. Default is "ECG".
        """
        for i, chunk in enumerate(chunks):
            DataTransformer.save_df_to_graph(chunk, f"{filesave_path}_chunk{i}.png", xkey, ykey)



    
    
