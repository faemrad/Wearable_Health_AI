from activity_recognition.core.data_transformer import DataTransformer
from activity_recognition.core.finetune_hubert import FineTuneHuBERT
from activity_recognition.scripts.config import WINDOW_SIZE, OVERLAP, NORMALISE, RESCALE, WINDOW_FIRST, FREQUENCY, DATATYPE, N_EPOCHS, N_SPLITS, BACKBONE_LEARNING_RATE, HEAD_LEARNING_RATE
import numpy as np

def main():
    """
    Performs full fine-tuning of the HuBERT model using cross-validation and returns the final performance estimate.
    """
    
    data_transformer = DataTransformer()
    X, y, groups, datasets = data_transformer.get_data(
        window_size=WINDOW_SIZE, overlap=OVERLAP, normalise=NORMALISE, datatype=DATATYPE, rescale=RESCALE,
                     frequency=FREQUENCY, window_first=WINDOW_FIRST)
    
    X = X.swapaxes(1, 2)

    activity_names = data_transformer.activity_names()

    print()
    print("=" * 70)
    print(f"FINE-TUNING (full CV - final performance estimate)")
    print("=" * 70)

    model = FineTuneHuBERT()

    model.fine_tune_cross_validation(
        X_all=X, y_all=y, groups_all=groups,
        class_names=activity_names,
        n_splits=N_SPLITS, n_classes=len(activity_names), n_epochs=N_EPOCHS)
    


if __name__ == "__main__":
    main()