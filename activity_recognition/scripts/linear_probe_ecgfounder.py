from activity_recognition.core.data_transformer import DataTransformer
from activity_recognition.core.finetune_ecgfounder import FineTuneECGFounder
from activity_recognition.scripts.config import WINDOW_SIZE, OVERLAP, NORMALISE, RESCALE, WINDOW_FIRST, FREQUENCY, DATATYPE, N_EPOCHS, N_SPLITS, BACKBONE_LEARNING_RATE, HEAD_LEARNING_RATE
import numpy as np

def main():
    """
    Performs linear probing of the ECGFounder model using cross-validation and returns the final performance estimate.
    """
    data_transformer = DataTransformer()
    X, y, groups, datasets = data_transformer.get_data(
        window_size=WINDOW_SIZE, overlap=OVERLAP, normalise=NORMALISE, datatype=DATATYPE, rescale=RESCALE,
                     frequency=FREQUENCY, window_first=WINDOW_FIRST)
    
    X = X.swapaxes(1, 2)
    names = data_transformer.activity_names()
    key = np.array([f"{d}__{l}" for d, l in zip(datasets, y)])
    model = FineTuneECGFounder(len(names))
    return model.linear_probe_cv(X, y, groups, names, stratify_labels=key, n_splits=N_SPLITS)


if __name__ == "__main__":
    main()