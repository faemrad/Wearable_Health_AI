import json
import torch
import torch.nn as nn
from models.ECGFounder.net1d import Net1D
from models.HuBERT_ECG.hubert_ecg.configuration import HuBERTECGClassificationConfig
from models.HuBERT_ECG.hubert_ecg.modeling_classification import HuBERTECGForClassification



def build_hubert_model(n_classes, dropout=0.3, device="cpu"):
    """
    Builds the full model for HuBERT

    Args:
        n_classes: the number of classes
        dropout: the dropout rate

    Returns:
        the model

    TODO : Connect config to current submodule
    """
    config = HuBERTECGClassificationConfig.from_pretrained(
        "Edoardo-Coppola/hubert-ecg-base",
        num_labels=n_classes,
        classifier_dropout_prob=dropout,
        task_type="multi_class",

    )
    hf_model = HuBERTECGForClassification(config)

    class _LogitsOnly(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.backbone = inner.hubert_ecg
        def forward(self, x):
            return self.inner(x).logits

    return _LogitsOnly(hf_model).to(device)



def build_ecgfounder_model(n_classes, dropout=0.3, device="cpu"):
    """
    Builds the full model for ECGFounder

    Args:
        n_classes: the number of classes
        dropout: the dropout rate

    Returns:
        the model
    """

    model = Net1D(
        in_channels=1, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        verbose=False, use_bn=False, use_do=False, n_classes=n_classes,
    )
    in_features = model.dense.in_features
    model.dense = nn.Linear(in_features, n_classes) 
    return model.to(device)


BUILDERS = {"HuBERT": build_hubert_model, "ECGFounder": build_ecgfounder_model}


def prepare_input(model_name, x):
    """
    Ensures that the input is in the correct format

    Args:
        model_name: the name of the model
        x: the input data
    """
    if model_name == "HuBERT" and x.dim() == 3 and x.shape[1] == 1:
        x = x.squeeze(1)
    return x


def load_model(checkpoint_path, config_path, device=None):
    """
    Loads the model based on the checkpoint_path

    Args:
        checkpoint_path: the path to the checkpoint
        config_path: the path to the config
    """
    with open(config_path) as f:
        config = json.load(f)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BUILDERS[config["model_name"]](config["n_classes"], config.get("dropout", 0.3), device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.startswith("backbone.")]
    if missing:
        print(f"WARNING: missing keys on load: {missing}")
    if unexpected:
        print(f"WARNING: unexpected keys on load: {unexpected}")

    model.eval()
    return model, config, device


def predict(model, config, device, x):
    """
    Runs inference on the model

    Args:
        model: the model
        config: the config
        device: the device
        x: the input data

    Returns:
        the predicted class and the probabilities
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(0) 
    x = x.unsqueeze(0).to(device) 
    x = prepare_input(config["model_name"], x)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax(dim=1).item()

    return config["class_names"][pred_idx], probs.cpu().numpy()[0]