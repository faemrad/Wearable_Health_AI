from activity_recognition.core.data_transformer import DataTransformer
from activity_recognition.core.finetune_ECGFounder import FineTuneECGFounder
from activity_recognition.scripts.config import WINDOW_SIZE, OVERLAP, NORMALISE, RESCALE, WINDOW_FIRST, FREQUENCY, DATATYPE, N_EPOCHS, BACKBONE_LEARNING_RATE, \
    HEAD_LEARNING_RATE, BATCH_SIZE, FREEZE_BACKBONE, SAVE_PATH
import numpy as np


def main(final_model_path: str = SAVE_PATH):
    """
    Performs final fine-tuning of the ECGFounder model and saves the trained model and configuration to disk. All data is used for training with no test set.

    Args:
        final_model_path (str): The path where the final model and configuration will be saved (default is SAVE_PATH).
    """
    data_transformer = DataTransformer()
    X, y, groups, datasets = data_transformer.get_data(
                window_size=WINDOW_SIZE, overlap=OVERLAP, normalise=NORMALISE,
                rescale=RESCALE, window_first=WINDOW_FIRST, frequency=FREQUENCY, datatype=DATATYPE
            )
    
    X = X.swapaxes(1, 2)

    activity_names = data_transformer.activity_names()
    stratify_key = np.array([f"{dataset}__{label}" for dataset, label in zip(datasets, y)])

    print()
    print("=" * 70)
    print(f"Final Model")
    print("=" * 70)

    model = FineTuneECGFounder(len(activity_names), False)

    final_path, config_path = model.fine_tune_final_model(
        X_all=X, y_all=y, groups_all=groups,
        class_names=activity_names, n_classes=len(activity_names),
        n_epochs=N_EPOCHS, backbone_lr=BACKBONE_LEARNING_RATE, head_lr=HEAD_LEARNING_RATE,
        batch_size=BATCH_SIZE, freeze_backbone=FREEZE_BACKBONE,
        save_path=f"{final_model_path}.pth",
        stratify_labels=stratify_key,
        model_name="ECGFounder",
    )

    print(f"Final model saved to: {final_path} || Config saved to: {config_path}")

if __name__ == "__main__":
    main()