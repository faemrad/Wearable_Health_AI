import numpy as np

class Processor:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def process(data, frequency) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method.")