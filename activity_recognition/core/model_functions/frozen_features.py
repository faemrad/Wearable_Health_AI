import os

import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from activity_recognition.core.model_functions.datasets import Datasets


class FrozenFeatureMixin:

    def extract_features(self, backbone, loader):
        """
        Override in subclass - model-specific feature extraction.
        Args:
            backbone: the backbone model
            loader: the data loader
        Returns:
            features, labels
        """
        raise NotImplementedError

    def frozen_extraction(self, class_names=None, classifier="logistic", save_prefix="model"):
        """
        Extracts frozen features from the backbone model and trains a classifier on top of them.
        Args:
            class_names: a list of class names (optional)
            classifier: the type of classifier to use ("logistic" or "mlp")
            save_prefix: a prefix for saving the trained classifier
        Returns:
            tuple: a tuple containing the trained classifier and the extracted features and labels for training, validation, and testing
        """
        X_train, y_train = self.extract_features(self.model, self.train_loader)
        X_val, y_val = self.extract_features(self.model, self.val_loader)
        X_test, y_test = self.extract_features(self.model, self.test_loader)

        if classifier == "logistic":
            clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        elif classifier == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                                early_stopping=True, random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

        if classifier == "mlp":
            clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
        else:
            clf.fit(X_train, y_train)

        joblib.dump(clf, f"{save_prefix}_{classifier}.joblib")

        print(f"=== Validation Performance ({save_prefix}, {classifier}) ===")
        print(classification_report(y_val, clf.predict(X_val), target_names=class_names))

        print(f"=== Test Performance ({save_prefix}, {classifier}) ===")
        print(classification_report(y_test, clf.predict(X_test), target_names=class_names))

        return clf, (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def linear_probe_cv(self, X_all, y_all, groups_all, class_names, stratify_labels=None,
                        n_splits=10, Cs=(0.001, 0.01, 0.1, 1.0), model_name=None,
                        save_dir="confusion_matrices/linear_probe"):
        """
        Linear probe: extract frozen embeddings ONCE, then fit a regularised
        logistic regression per fold on the same subject-wise folds as
        fine_tune_cross_validation. C is chosen on the inner val split only.

        Args:
            X_all: the input data
            y_all: the labels
            groups_all: the group labels for stratification
            class_names: the class names
            stratify_labels: the labels to use for stratification (if None, uses y_all)
            n_splits: the number of folds for cross-validation
            Cs: the C values to try
            model_name: the name of the model (used for saving the config)
            save_dir: the directory to save the confusion matrices

        Returns:
            tuple: a tuple containing the confusion matrices for each fold
        """
        if stratify_labels is None:
            stratify_labels = y_all
        model_name = model_name or type(self).__name__.replace("FineTune", "")
        os.makedirs(save_dir, exist_ok=True)

        # 1. Embeddings for every window: backbone frozen, eval mode, no grad
        loader = DataLoader(Datasets(X_all, y_all, train=False), batch_size=32, shuffle=False)
        F, y_check = self.extract_features(self.model, loader)
        assert (y_check == y_all).all(), "feature/label order mismatch"
        print(f"Extracted features: {F.shape}")

        def make(C):
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))

        # 2. Same outer + inner splits as fine_tune_cross_validation
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results, all_true, all_pred = [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X_all, stratify_labels, groups_all)):
            tr, va = self._inner_split(train_idx, stratify_labels, groups_all)
            self._assert_disjoint_groups(groups_all, train=tr, val=va, test=test_idx)

            # 3. Choose C on the inner validation split (macro-F1), never on test
            scores = {
                C: f1_score(y_all[va], make(C).fit(F[tr], y_all[tr]).predict(F[va]), average="macro")
                for C in Cs
            }
            best_C = max(scores, key=scores.get)

            # 4. Evaluate on the held-out subjects
            preds = make(best_C).fit(F[tr], y_all[tr]).predict(F[test_idx])
            report = classification_report(y_all[test_idx], preds, target_names=class_names,
                                           output_dict=True, zero_division=0)
            fold_results.append(report)
            all_true.extend(y_all[test_idx])
            all_pred.extend(preds)
            print(f"Fold {fold_idx + 1}/{n_splits}: C={best_C} | acc={report['accuracy']:.4f} | "
                  f"macro-F1={report['macro avg']['f1-score']:.4f}")

        accs = [r["accuracy"] for r in fold_results]
        f1s = [r["macro avg"]["f1-score"] for r in fold_results]
        print(f"\n{model_name} linear probe | Acc {np.mean(accs):.4f} ± {np.std(accs):.4f} "
              f"| Macro-F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        self.plot_confusion_matrix(
            all_true, all_pred, class_names=class_names,
            title_suffix=f" ({model_name}, linear probe)",
            save_path=os.path.join(save_dir, f"{model_name}_probe_confusion_matrix.png"),
            display=False,
        )
        return fold_results

    def cross_validate(self, X_all, y_all, groups_all, class_names, datasets_all=None,
                       n_splits=1, classifier="logistic", stratify_labels=None,
                       save_dir="confusion_matrices"):
        """
        Performs cross-validation on the given data, using the given classifier.

        Args:
            X_all: the input data
            y_all: the labels
            groups_all: the group labels for stratification
            class_names: the class names
            datasets_all: the dataset labels (optional)
            n_splits: the number of folds for cross-validation
            classifier: the type of classifier to use ("logistic" or "mlp")
            stratify_labels: the labels to use for stratification (if None, uses y_all)
            save_dir: the directory to save the confusion matrices

        Returns:
            tuple: a tuple containing the confusion matrices for each fold
        """

        if stratify_labels is None:
            stratify_labels = y_all

        if datasets_all is not None:
            subject_counts = (
                pd.DataFrame({"dataset": datasets_all, "subject": groups_all})
                .drop_duplicates()
                .groupby("dataset")["subject"]
                .nunique()
            )
            print(f"\nUnique subjects per dataset (n_splits={n_splits}):")
            print(subject_counts)
            too_few = subject_counts[subject_counts < n_splits]
            if not too_few.empty:
                print(f"WARNING: these datasets have fewer subjects than folds, "
                      f"so some folds' TEST set may get zero samples from them:\n{too_few}\n")

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"SGKF summary: {sgkf}")
        fold_results = []
        all_true_labels, all_pred_labels = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X_all, stratify_labels, groups_all)):
            print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")

            if datasets_all is not None:
                print(f"  train dataset counts: {pd.Series(datasets_all[train_idx]).value_counts().to_dict()}")
                print(f"  test  dataset counts: {pd.Series(datasets_all[test_idx]).value_counts().to_dict()}")

            real_train_idx, real_val_idx = self._inner_split(train_idx, y_all, groups_all)
            self._assert_disjoint_groups(groups_all, train=real_train_idx, val=real_val_idx, test=test_idx)

            self.load_data(
                X_all[real_train_idx], y_all[real_train_idx],
                X_all[real_val_idx], y_all[real_val_idx],
                X_all[test_idx], y_all[test_idx],
                oversample=False,
            )

            clf, _, _, (X_test_feat, y_test_feat) = self.frozen_extraction(
                class_names=class_names, classifier=classifier, save_prefix=f"fold_{fold_idx + 1}"
            )

            test_preds = clf.predict(X_test_feat)
            fold_results.append(
                classification_report(y_test_feat, test_preds, target_names=class_names, output_dict=True)
            )
            all_true_labels.extend(y_test_feat)
            all_pred_labels.extend(test_preds)

        accuracies = [r["accuracy"] for r in fold_results]
        for activity in class_names:
            f1s = [r[activity]["f1-score"] for r in fold_results]
            support = [r[activity]["support"] for r in fold_results]
            print(f"\nMean F1 for {activity}: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(f"Sample count for {activity}: {np.sum(support)}")
        print(f"\nMean accuracy across {n_splits} folds: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

        self.plot_confusion_matrix(
            all_true_labels, all_pred_labels, class_names=class_names,
            title_suffix=f" (all folds, {classifier})",
            save_path=os.path.join(save_dir, f"{classifier}_all_folds.png"),
        )
        return fold_results