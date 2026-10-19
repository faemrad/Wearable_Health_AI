"""Base class for frozen-feature / fine-tuning pipelines on ECG windows.

The behaviour lives in the mixins; this class owns the state (device, loaders)
and data loading, and composes them so existing imports keep working.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from activity_recognition.core.model_functions.cross_validation import CrossValidationMixin
from activity_recognition.core.model_functions.datasets import Datasets
from activity_recognition.core.model_functions.fine_tuning import FineTuneMixin
from activity_recognition.core.model_functions.frozen_features import FrozenFeatureMixin
from activity_recognition.core.model_functions.helpers import SplitHelpers
from activity_recognition.core.model_functions.plotting import PlottingMixin


class BaseFrozenModel(
    SplitHelpers,
    PlottingMixin,
    FrozenFeatureMixin,
    FineTuneMixin,
    CrossValidationMixin,
):

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.label_encoder = None

    def load_data(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32,
                  num_workers=0, oversample=False):
        train_dataset = Datasets(X_train, y_train, train=True)
        val_dataset = Datasets(X_val, y_val, train=False)
        test_dataset = Datasets(X_test, y_test, train=False)

        loader_kwargs = dict(
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
        )

        if oversample:
            sample_weights = (1.0 / np.bincount(y_train))[y_train]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler, **loader_kwargs
            )
        else:
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs
            )

        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)