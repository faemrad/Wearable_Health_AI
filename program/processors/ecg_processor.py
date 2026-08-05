import numpy as np
from program.processors.processor import Processor
import neurokit2 as nk


class ECGProcessor(Processor):
    @staticmethod
    def process(data, frequency) -> np.ndarray:
        return ECGProcessor._temp_process(data, frequency)
        # TODO: Add actual processing logic for ECG data
        pass

    @staticmethod
    def _temp_process(data, frequency) -> np.ndarray:
        cleaned = nk.ecg_clean(data["ecg"], sampling_rate=frequency, method="neurokit")
        return cleaned