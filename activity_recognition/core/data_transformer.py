"""Loads raw recordings, windows them, and prepares X / y / groups for training."""
from math import gcd

import numpy as np
from scipy.signal import resample_poly
from sklearn.preprocessing import LabelEncoder

from activity_recognition.helpers.data_combiner import DataCombiner
from activity_recognition.core.model_functions.windower import Windower


ACTIVITY_MAPPING = {
    "walk": "walking",
    "walking": "walking",
    "stationary": "stationary",
    "sit": "stationary",
    "sitting": "stationary",
    "working": "stationary",
    "running": "running",
    "run": "running",
    "cycling": "cycling",
    "bike": "cycling",
    "outdoor-cycle": "cycling",
    # "badminton": "badminton",
    # "tabletennis": "tabletennis",
    # "basketball": "basketball",
}


SOURCE_FREQUENCY = 100      
MAX_ABS_AMPLITUDE = 10.5     


class DataTransformer:

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def get_data_files() -> dict:
        """
        Returns {"<dataset>_<subject>_<activity>": DataFrame}.
        """

        records = DataCombiner().get_all_data()
        data_files = {
            f"{d['dataset']}_{d['subject']}_{d['activity']}": d["dataframe"] for d in records
        }
        return data_files

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def get_data(self, window_size=5, overlap=1, normalise=True, datatype="ecg", rescale=True,
                     frequency=100, window_first=False):
        """
        Builds windowed data for every recording with a known activity.

        Args:
            window_size: window length in seconds.
            overlap: overlap between consecutive windows in seconds.
            normalise / rescale: per-window (or per-recording) scaling, see _apply_data_transformations.
            datatype: signal column to use ("ecg" or "ppg").
            frequency: target sampling rate in Hz (recordings are resampled from SOURCE_FREQUENCY).
            window_first: window before scaling (True) or scale the whole recording first (False).
            bandpass: apply the BANDPASS_RANGES filter for `datatype`.

        Returns:
            X (n_windows, window_length, 1), y (encoded labels), groups (subject IDs), datasets.
        """
        data_files = DataTransformer.get_data_files()
        windower = Windower(window_size, overlap)

        print(
            f"Conditions: window={window_size}, overlap={overlap}, normalise={normalise}, "
            f"datatype={datatype}, rescale={rescale}, frequency={frequency}, "
            f"window_first={window_first}"
        )

        windowed_data = []
        total_windows = 0
        impacted_windows = 0
        windows_per_activity = {}

        for key, data in data_files.items():
            dataset, user_id, raw_activity = key.split("_")[:3]

            if raw_activity not in ACTIVITY_MAPPING or datatype not in data.keys():
                continue

            activity = ACTIVITY_MAPPING[raw_activity]
            windows_per_activity.setdefault(activity, 0)

            windows = DataTransformer._apply_data_transformations(
                data[datatype].to_numpy(), SOURCE_FREQUENCY, window_size, overlap, datatype,
                windower=windower, window_first=window_first, default_frequency=frequency,
                rescale=rescale, normalise=normalise
            )

            for window in windows:
                total_windows += 1
                if (not np.isfinite(window).all()) or np.abs(window).max() > MAX_ABS_AMPLITUDE:
                    impacted_windows += 1
                    continue
                windowed_data.append({
                    "dataset": dataset,
                    "activity": activity,
                    "user_id": f"{dataset}_{user_id}",
                    "window": window,
                })
                windows_per_activity[activity] += 1

        print(f"Total windows: {total_windows}, impacted windows: {impacted_windows}")
        for activity, count in windows_per_activity.items():
            print(f"Total windows || {activity}: {count}")

        X, y, groups, datasets = DataTransformer.create_x_y(windowed_data, x_labels=[datatype])
        self.y = y
        self.encode_labels()

        return X, self.y, groups, datasets

    @staticmethod
    def create_x_y(transformed_data: list, x_labels: list):
        """
        Stacks window dicts into arrays.

        Returns:
            (X, y, groups, datasets), where y still holds string activity labels.
        """
        x_list, y_list, groups_list, datasets_list = [], [], [], []

        for item in transformed_data:
            window = item["window"]

            if hasattr(window, "columns"):
                if x_labels[0] not in window.columns:
                    continue
                filtered = window[list(x_labels)].values        
            else:
                filtered = np.asarray(window)
                if filtered.ndim == 1:
                    filtered = filtered[:, np.newaxis]   

            if np.isnan(filtered).any():
                continue

            x_list.append(filtered)
            y_list.append(item["activity"])
            groups_list.append(item["user_id"])
            datasets_list.append(item["dataset"])

        return np.array(x_list), np.array(y_list), np.array(groups_list), np.array(datasets_list)

    def encode_labels(self):
        """Encode self.y (string labels) to integers and keep the encoder for activity_names()."""
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

    def activity_names(self):
        return self.label_encoder.classes_

    @staticmethod
    def _require_array(data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array.")

    @staticmethod
    def log_transform(data: np.ndarray, inplace: bool = False):
        """Signed log1p: sign(x) * log(1 + |x|). Handles zeros and negatives."""
        if not inplace:
            data = data.astype(float, copy=True)
        sign = np.sign(data)
        np.abs(data, out=data)
        np.log1p(data, out=data)
        data *= sign
        return data

    @staticmethod
    def resample_data(new_fq, cur_fq, data):
        """
        Resamples along the last (time) axis.

        Args:
            new_fq: new frequency in Hz.
            cur_fq: current frequency in Hz.
            data: 1-D (samples,) or 2-D (channels, samples) array.
        """
        DataTransformer._require_array(data)
        g = gcd(new_fq, cur_fq)
        return resample_poly(data, new_fq // g, cur_fq // g, axis=data.ndim - 1)

    @staticmethod
    def rescale_data(data, eps: float = 1e-8, feature_range=(-1.0, 1.0)):
        """Min-max scale along the last axis into feature_range."""
        DataTransformer._require_array(data)

        lo, hi = feature_range
        if data.ndim == 1:
            min_val, max_val = data.min(), data.max()
        else:
            min_val = data.min(axis=-1, keepdims=True)
            max_val = data.max(axis=-1, keepdims=True)

        scaled = (data - min_val) / (max_val - min_val + eps)
        return (scaled * (hi - lo) + lo).astype(np.float32)

    @staticmethod
    def normalise_data(data: np.ndarray, flat_tol: float = 1e-6, clip: float = 10.0):
        """
        Robust z-score along the last axis (median / MAD, falling back to std),
        clipped to +-clip. Flat segments become NaN so they can be dropped later.
        """
        DataTransformer._require_array(data)

        data = data.astype(np.float64, copy=False)
        center = np.median(data, axis=-1, keepdims=True)
        mad = 1.4826 * np.median(np.abs(data - center), axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)
        scale = np.where(mad > flat_tol, mad, std)

        out = np.clip((data - center) / np.maximum(scale, flat_tol), -clip, clip)
        out = np.where(scale <= flat_tol, np.nan, out)
        return out.astype(np.float32)

    @staticmethod
    def _apply_data_transformations(data: np.ndarray, frequency, window_size, overlap, datatype, windower=None,
                                    window_first=False, default_frequency=100, rescale=False,
                                    normalise=True, bandpass=False):
        """
        Fill NaNs -> resample to default_frequency -> optional band-pass -> scale + window.
        Order of scaling vs windowing is controlled by window_first.
        """
        DataTransformer._require_array(data)

        nans = np.isnan(data)
        if nans.any():
            data = data.copy() 
            data[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), data[~nans])

        if windower is None:
            windower = Windower(window_size, overlap)

        if frequency != default_frequency:
            data = DataTransformer.resample_data(default_frequency, frequency, data)
            frequency = default_frequency 

        if window_first:
            windows = np.asarray(windower.window_numpy_array(data, frequency))
            if rescale:
                windows = DataTransformer.rescale_data(windows)
            if normalise:
                windows = DataTransformer.normalise_data(windows)
            return windows

        if normalise:
            data = DataTransformer.normalise_data(data)
        if rescale:
            data = DataTransformer.rescale_data(data)

        return windower.window_numpy_array(data, frequency)

    @staticmethod
    def default_data_transformation(data: np.ndarray, datatype, frequency: int, bandpass=False):
        return DataTransformer._apply_data_transformations(
            data, frequency, 8, 2, datatype, windower=None, window_first=False, default_frequency=100,
            rescale=False, normalise=True, bandpass=bandpass
        )

    @staticmethod

    def get_processed_windows_from_np_array(array: np.ndarray, window_size: int, overlap: float,
                                            frequency: int, normalise=True, rescale=False,
                                            window_first=False, target_frequency=100):
        """Inference-time preprocessing, identical to the training path.

        Returns:
            np.ndarray of shape (n_windows, window_length)
        """
        data = np.asarray(array, dtype=np.float64).ravel()
        windows = DataTransformer._apply_data_transformations(
            data, frequency, window_size, overlap, None,
            window_first=window_first, default_frequency=target_frequency,
            rescale=rescale, normalise=normalise,
        )
        return np.asarray(windows, dtype=np.float32)
