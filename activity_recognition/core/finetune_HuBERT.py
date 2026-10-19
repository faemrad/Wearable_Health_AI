import torch
import torch.nn as nn

import numpy as np

from activity_recognition.core.model_functions.base_model import BaseFrozenModel

from models.HuBERT_ECG.hubert_ecg.configuration import HuBERTECGClassificationConfig
from models.HuBERT_ECG.hubert_ecg.modeling import HuBERTECG
from models.HuBERT_ECG.hubert_ecg.modeling_classification import HuBERTECGForClassification


class FineTuneHuBERT(BaseFrozenModel):
    def __init__(self):
        super().__init__()
        self.model = HuBERTECG.from_pretrained("Edoardo-BS/hubert-ecg-base")
        self.model.to(self.device)

    def extract_features(self, backbone, loader):
        backbone.eval()
        features, labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                x_batch = self.prepare_input(x_batch)

                outputs = backbone(x_batch, return_dict=True)
                pooled = outputs.last_hidden_state.mean(dim=1)
                features.append(pooled.cpu().numpy())
                labels.append(y_batch.numpy())
        return np.concatenate(features), np.concatenate(labels)

    def prepare_input(self, x):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        return x

    def create_finetune_model(self, n_classes, dropout=0.3):
        # Build a classification config from the pretrained backbone's config,
        # then construct HuBERTECGForClassification directly — no custom
        # head wrapper needed, since it's built in.
        cls_config = HuBERTECGClassificationConfig(
            num_labels=n_classes,
            classifier_dropout_prob=dropout,
            task_type="multi_class",  # adjust to match your actual task
            **self.model.config.to_dict(),
        )
        hf_model = HuBERTECGForClassification(cls_config)
        missing, unexpected = hf_model.hubert_ecg.load_state_dict(self.model.state_dict(), strict=False)
        if missing:
            print(f"WARNING: unexpected missing keys when loading backbone: {missing}")

        class _LogitsOnly(nn.Module):
            """Wraps HuBERTECGForClassification so forward() returns a plain
            logits tensor, matching the interface BaseFrozenModel expects
            (same as ECGFounder's Net1D, which returns a tensor directly)."""
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.backbone = inner.hubert_ecg  # exposes the encoder so
                # BaseFrozenModel._find_named_submodule can detect it as the
                # backbone, restoring the backbone_lr/head_lr split and making
                # freeze_backbone actually take effect for HuBERT trials.

            def forward(self, x):
                return self.inner(x).logits

        return _LogitsOnly(hf_model).to(self.device)
