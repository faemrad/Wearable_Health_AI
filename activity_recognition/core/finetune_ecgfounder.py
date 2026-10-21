import models.ECGFounder.finetune_model as ECGFound
import torch
import numpy as np
from activity_recognition.core.model_functions.base_model import BaseFrozenModel
from huggingface_hub import hf_hub_download

class FineTuneECGFounder(BaseFrozenModel):
    """
    ECGFounder model implementation for fine-tuning.
    """
    def __init__(self, n_classes=6, linear_prob=True):
        """
        Initialiases the base model class to be used for each ECGFounder model implementation.
        """
        super().__init__()
        self.checkpoint_path = hf_hub_download(repo_id="PKUDigitalHealth/ECGFounder", filename="1_lead_ECGFounder.pth")
        self.model = ECGFound.ft_1lead_ECGFounder(self.device, self.checkpoint_path, n_classes, linear_prob=linear_prob)
        self.model.to(self.device)
        self.model.eval()

        self._captured_features = None
        self.model.dense.register_forward_hook(self._capture_dense_input)

    def _capture_dense_input(self, module, input, output):
        """
        Captures the input features from the dense layer.
        Args:
            input: the input to the dense layer
            module: the dense layer
            output: the output of the dense layer
        
        Returns:
            None
        """
        self._captured_features = input[0].detach()

    def prepare_input(self, x):
        """
        Prepare the input for the model.
        Args:
            x: the input data
        
        Returns:
            the prepared input
        """
        return x

    def extract_features(self, backbone, loader):
        """
        Extract features from the backbone model.

        Args:
            backbone: the backbone model
            loader: the data loader

        Returns:
            features, labels
        """
        backbone.eval()
        features, labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                _ = backbone(x_batch)
                features.append(self._captured_features.cpu().numpy())
                labels.append(y_batch.numpy())
        return np.concatenate(features), np.concatenate(labels)

    def create_finetune_model(self, n_classes, dropout=0.3):
        """
        Builds a fine-tuning model for the ECGFounder model.
        Args:
            n_classes: the number of classes
            dropout: the dropout rate
        
        Returns:
            the fine-tuning model
        """
        model = ECGFound.ft_1lead_ECGFounder(self.device, self.checkpoint_path, n_classes, linear_prob=False)
        return model