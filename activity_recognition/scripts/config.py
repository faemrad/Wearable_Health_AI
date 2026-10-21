DATATYPE = "ecg" 

# Model Parameters
WINDOW_SIZE = 5  # seconds
OVERLAP = 1  # seconds
FREQUENCY = 500  # Hz
NORMALISE = True  # Whether to normalise the data
RESCALE = False  # Whether to rescale the data
WINDOW_FIRST = True  # Whether to apply windowing before scaling 
FREEZE_BACKBONE = False  # Whether to freeze the backbone during training

# Training Parameters
N_EPOCHS = 20  # Number of epochs for training
N_SPLITS = 10  # Number of splits for cross-validation
BATCH_SIZE = 32  # Batch size for training

# Hyperparameters
BACKBONE_LEARNING_RATE = 0.001
HEAD_LEARNING_RATE = 0.001

# Final model
SAVE_PATH = "finetuned_checkpoints/final_model"


# Hugging Face Repositories
HUBERT_REPO = "Jonathan412/hubert-activity-classifier"
ECGFOUNDER_REPO = "Jonathan412/ecgfounder-activity-classifier"

ECG_MODEL_NAME = "ecg_model_four_activities.safetensors"
PPG_MODEL_NAME = "ppg_model_four_activities.safetensors"
CONFIG_NAME = "config.json"