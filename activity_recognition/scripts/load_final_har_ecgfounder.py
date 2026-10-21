from activity_recognition.scripts.config import CONFIG_NAME, ECG_MODEL_NAME, PPG_MODEL_NAME, ECGFOUNDER_REPO, DATATYPE
from huggingface_hub import hf_hub_download
import json
from safetensors.torch import load_file
import torch
from activity_recognition.core import model_builder as mb
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

def main(datatype: str = DATATYPE):
    """
    Loads the final trained ECGFounder model from the Hugging Face Hub and returns the model, configuration, and device.

    Args:
        datatype (str): The type of data (default is DATATYPE).
    Returns:
        tuple: A tuple containing the loaded model, configuration dictionary, and device string.
    """
    if datatype == "ppg":
        model_name = PPG_MODEL_NAME
    else:
        model_name = ECG_MODEL_NAME
    weights_path = hf_hub_download(repo_id=ECGFOUNDER_REPO, filename=model_name)
    config_path = hf_hub_download(repo_id=ECGFOUNDER_REPO, filename=CONFIG_NAME)
    
    

    with open(config_path) as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = mb.BUILDERS[config["model_name"]](config["n_classes"], config.get("dropout", 0.3), device)

    state_dict = load_file(weights_path, device=str(device))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.startswith("backbone.")]
    if missing:
        print(f"WARNING: missing keys loading {ECGFOUNDER_REPO}/{model_name}: {missing}")
    if unexpected:
        print(f"WARNING: unexpected keys loading {ECGFOUNDER_REPO}/{model_name}: {unexpected}")
    model.eval()
    return model, config, device

if __name__ == "__main__":
    model, config, device = main()
    if DATATYPE == "ppg":
        print(f"Loaded model from {ECGFOUNDER_REPO}/{PPG_MODEL_NAME} with config {ECGFOUNDER_REPO}/{CONFIG_NAME}")
    else:
        print(f"Loaded model from {ECGFOUNDER_REPO}/{ECG_MODEL_NAME} with config {ECGFOUNDER_REPO}/{CONFIG_NAME}")
