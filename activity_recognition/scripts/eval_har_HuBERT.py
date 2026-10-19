from activity_recognition.scripts.config import DATATYPE
from activity_recognition.scripts import load_final_har_HuBERT
from activity_recognition.core.data_transformer import DataTransformer
from activity_recognition.core import model_builder as mb
from collections import Counter
import numpy as np
import torch
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

def main(data: np.ndarray, frequency: int, datatype: str = DATATYPE):
    """
    Evaluates the data using the final trained HuBERT model and returns the main prediction, predictions for each window, and the probabilities for each window.

    Args:
        data (np.ndarray): The input data to be evaluated.
        frequency (int): The frequency of the input data.
        datatype (str): The type of data (default is DATATYPE).

    Returns:
        dict: A dictionary containing the main prediction, predictions for each window, window probabilities, and the configuration used for evaluation.
    """
    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 2:
        if 1 in data.shape:
            data = data.reshape(-1)    
        else:
            data = data[:, 0]        
    elif data.ndim != 1:
        raise ValueError(f"Expected 1D or 2D signal, got shape {data.shape}")
    
    model, config, device = load_final_har_HuBERT.main(datatype=datatype)
    windows = DataTransformer.get_processed_windows_from_np_array(
        data, config["window"], config["overlap"], frequency, config["normalise"],
        rescale=config.get("rescale", False),
        window_first=config.get("window_first", False),
        target_frequency=config.get("sampling_rate", 100)
    )


    valid = np.isfinite(windows).all(axis=1) & (np.abs(windows).max(axis=1) <= 10.5)
    windows = windows[valid]
    if len(windows) == 0:
        raise ValueError("No valid windows after preprocessing.")

    x = torch.as_tensor(windows, dtype=torch.float32).to(device)
    x = x.unsqueeze(1)                        
    x = mb.prepare_input(config["model_name"], x)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).cpu().numpy()

    predictions = [config["class_names"][i] for i in pred_idx]

    counts = Counter(pred_idx.tolist())
    main_idx, _ = counts.most_common(1)[0]
    main_prediction = config["class_names"][main_idx]

    return {
        "main_prediction": main_prediction,
        "predictions": predictions,
        "window_probabilities": probs.cpu().numpy().tolist(),
        "config": config
    }

def get_main_prediction(data: np.ndarray, frequency: int, datatype: str = DATATYPE):
    """
    Returns the main prediction for the given data using the final trained HuBERT model.

    Args:
        data (np.ndarray): The input data to be evaluated.
        frequency (int): The frequency of the input data.
        datatype (str): The type of data (default is DATATYPE).

    Returns:
        str: The main prediction label.
    """
    result = main(data, frequency, datatype)
    return result["main_prediction"]

def get_probability_by_activity(data: np.ndarray, frequency: int, datatype: str = DATATYPE):
    """
    Calculates the probability of each activity based on the window probabilities.

    Args:
        data (np.ndarray): The input data to be evaluated.
        frequency (int): The frequency of the input data.
        datatype (str): The type of data (default is DATATYPE).

    Returns:
        dict: A dictionary containing the probability for each activity.
    """
    result = main(data, frequency, datatype)
    config = result["config"]
    probability_sums = [0] * len(result["window_probabilities"][0])
    counter = 0
    for prediction in result["window_probabilities"]:
        for i in range(len(prediction)):
            probability_sums[i] += prediction[i]
        counter += 1
    
    results = dict()
    for i in range(len(probability_sums)):
        results[config["class_names"][i]] = round(probability_sums[i] / counter, 10)
    return results

if __name__ == "__main__":
    dataset = np.load("data/PPG-Sport/dataset/S01L_running_ppg.npy")

    print(get_probability_by_activity(dataset, 64, "ppg"))
