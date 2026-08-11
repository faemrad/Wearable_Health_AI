from program.data_preparation.windower import Windower
from program.processors.file_processor import FileProcessor
from program.retrievers.file_reader import FileReader
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Training:
    def __init__(self, data_folder: str, model_type: str, frequency: int, column_labels: list[str]):
        self.data_folder = data_folder
        self.model_type = model_type
        self.frequency = frequency
        self.column_labels = column_labels

        self.get_data_files()
        # self.transformed_data = self.transform_data(self.data_files)

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
    def transform_data(self, data_frames: list):
        """
        Transforms the data by applying a windowing technique to each DataFrame in the provided list.

        Args:
            data_frames (list): A list of pd.DataFrame objects to be transformed.

        Returns:
            list: A list of transformed pd.DataFrame objects after applying the windowing technique.
        """
        pass

    def create_x_y(self, transformed_data: list = None, activity_labels: list[str] = ["badminton", "basketball", "running", "stationary", "tabletennis", "walking"], x_labels = ["L_imu_x", "L_imu_y", "L_imu_z", "L_ppg"]):
        """
        Trains a model using the provided transformed data.

        Args:
            transformed_data (list): A list of transformed pd.DataFrame objects to be used for training the model.
            activity_labels (list): A list of activity labels corresponding to the transformed data.

        Returns:
            None
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
        
        pass

    def encode_labels(self):
        if not hasattr(self, "groups"):
            self.create_x_y()
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)
        self.label_encoder = le

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

    def build_1d_cnn_model(self):
        """
        Builds a 1D CNN for activity classification, sized to match
        the windowed data (self.X) and number of activity classes (self.y).

        Returns:
            None
        """
        num_timesteps = self.X.shape[1]
        num_channels = self.X.shape[2]
        num_classes = len(np.unique(self.y))

        self.model = models.Sequential([
            layers.Input(shape=(num_timesteps, num_channels)),

            layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
            layers.MaxPooling1D(2),

            layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            layers.MaxPooling1D(2),

            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),

            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

        return self.model

    def train(self, epochs: int = 30, batch_size: int = 64):
        """
        Trains the model on the train split, validating against the val split.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.

        Returns:
            History: Keras History object containing training/validation metrics per epoch.
        """
        if not hasattr(self, "model"):
            self.build_1d_cnn_model()

        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size
        )

        return self.history

    def evaluate(self):
        """
        Evaluates the trained model on the held-out test split.

        Returns:
            tuple: (test_loss, test_accuracy)
        """
        if not hasattr(self, "model"):
            raise RuntimeError("No trained model found — call train() first.")

        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        return test_loss, test_acc

    def generate_confusion_matrix(self):
        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=self.label_encoder.classes_)
        disp.plot(xticks_rotation=45)
        plt.show()

    def run_process(self):
        self.window_data(5, 1)
        self.create_x_y()
        self.encode_labels()
        self.split_participants()
        self.apply_split()
        self.train()
        self.evaluate()
        self.generate_confusion_matrix()

if __name__ == "__main__":
    training = Training("./data/PPG-Sport/converted_data", "cnn", 64, ["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "ecg", "ppg", "temp", "eda"])
    training.run_process()