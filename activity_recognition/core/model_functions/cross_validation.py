import json
import os

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold


class CrossValidationMixin:
    """
    Cross-validation mixin for frozen-feature / fine-tuning pipelines on windows.
    """

    def fine_tune_final_model(self, X_all, y_all, groups_all, class_names, n_classes=6,
                              val_fraction_splits=5, n_epochs=20, backbone_lr=1e-5, head_lr=1e-4,
                              weight_decay=1e-4, early_stopping_patience=8, scheduler_patience=3,
                              freeze_backbone=False, batch_size=32, dropout=0.3,
                              save_path="final_model.pth", stratify_labels=None, window_size=None, overlap=None, normalise=None, rescale=None, sampling_rate=None, window_first=None,
                              model_name=None):
        """
        Fine-tunes the model on the entire dataset, using a stratified group K-fold split to hold out a validation set for early stopping.

        Args:
            X_all: the input data
            y_all: the labels
            groups_all: the group labels for stratification
            class_names: the class names
            n_classes: the number of classes
            val_fraction_splits: the number of splits for the validation set
            n_epochs: the number of epochs to train for
            backbone_lr: the learning rate for the backbone
            head_lr: the learning rate for the classification head
            weight_decay: the weight decay for the optimizer
            early_stopping_patience: the patience for early stopping
            scheduler_patience: the patience for the learning rate scheduler
            freeze_backbone: whether to freeze the backbone during training
            batch_size: the batch size for training
            dropout: the dropout rate for the classification head
            save_path: the path to save the final model to
            stratify_labels: the labels to use for stratification (if None, uses y_all)
            model_name: the name of the model (used for saving the config)

        Returns:
            tuple: a tuple containing the path to the saved model and the path to the saved config
        """
        if stratify_labels is None:
            stratify_labels = y_all

        sgkf = StratifiedGroupKFold(n_splits=val_fraction_splits, shuffle=True, random_state=42)
        train_idx, val_idx = next(sgkf.split(X_all, stratify_labels, groups_all))
        self._assert_disjoint_groups(groups_all, train=train_idx, val=val_idx)

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        print(
            f"\nTraining final model on {len(train_idx)} samples "
            f"(held out {len(val_idx)} samples for early-stopping only, "
            f"no test set - CV already estimated generalization performance)."
        )

        class_weights = self.compute_class_weights(y_train, n_classes)
        self.load_data(X_train, y_train, X_val, y_val, X_val, y_val,
                       batch_size=batch_size, oversample=False)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        _, best_val_f1 = self.fine_tune(
            n_classes=n_classes,
            n_epochs=n_epochs,
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            scheduler_patience=scheduler_patience,
            freeze_backbone=freeze_backbone,
            class_weights=class_weights,
            save_path=save_path,
        )

        config = {
            "model_name": model_name or type(self).__name__.replace("FineTune", ""),
            "n_classes": n_classes,
            "dropout": dropout,
            "class_names": list(class_names),
            "val_f1": best_val_f1,
            "window": window_size,
            "overlap": overlap,
            "normalise": normalise,
            "rescale": rescale,
            "sampling_rate": sampling_rate,
            "window_first": window_first

        }
        config_path = save_path.rsplit(".", 1)[0] + "_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(
            f"\nFinal model saved to {save_path} "
            f"(reached {best_val_f1:.4f} val macro-F1 on its early-stopping "
            f"slice - NOT a generalization estimate, that's what CV was for)"
        )
        print(f"Config saved to {config_path}")

        return save_path, config_path

    def fine_tune_cross_validation(self, X_all, y_all, groups_all, class_names, 
                                   n_splits=1, n_classes=6, n_epochs=20,
                                   backbone_lr=1e-5, head_lr=1e-4, weight_decay=1e-4,
                                   early_stopping_patience=8, scheduler_patience=3,
                                   freeze_backbone=False, batch_size=64, save_dir="finetuned_checkpoints",
                                   stratify_labels=None, model_name=None, plot_per_fold=False,
                                   on_fold_end=None):
        """
        Subject-wise (grouped) stratified K-fold CV with an inner subject-wise
        train/val split for checkpoint selection.

        Args:
            X_all: the input data
            y_all: the labels
            groups_all: the group labels for stratification
            class_names: the class names
            n_splits: the number of folds for cross-validation
            n_classes: the number of classes
            n_epochs: the number of epochs to train for
            backbone_lr: the learning rate for the backbone
            head_lr: the learning rate for the classification head
            weight_decay: the weight decay for the optimizer
            early_stopping_patience: the patience for early stopping
            scheduler_patience: the patience for the learning rate scheduler
            freeze_backbone: whether to freeze the backbone during training
            batch_size: the batch size for training
            save_dir: the directory to save the models to
            stratify_labels: the labels to use for stratification (if None, uses y_all)
            model_name: the name of the model (used for saving the config)
            plot_per_fold: whether to plot the confusion matrix for each fold
            on_fold_end: a callback function to call at the end of each fold, with signature (fold_idx, model, X_val, y_val, X_test, y_test)
        
        Returns:
            list: a list of classification reports for each fold
        """
        if stratify_labels is None:
            stratify_labels = y_all

        model_name = model_name or type(self).__name__.replace("FineTune", "")
        os.makedirs(save_dir, exist_ok=True)

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"SGKF summary: {sgkf}")

        fold_results, all_true_labels, all_pred_labels = [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X_all, stratify_labels, groups_all)):
            print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")

            real_train_idx, real_val_idx = self._inner_split(train_idx, stratify_labels, groups_all)
            self._assert_disjoint_groups(groups_all, train=real_train_idx, val=real_val_idx, test=test_idx)

            y_train_fold = y_all[real_train_idx]
            y_val_fold = y_all[real_val_idx]

            self._warn_sparse_val(y_val_fold, n_classes)

            self.load_data(
                X_all[real_train_idx], y_train_fold,
                X_all[real_val_idx], y_val_fold,
                X_all[test_idx], y_all[test_idx],
                batch_size=batch_size, num_workers=0, oversample=False,
            )

            self.fine_tune(
                n_classes=n_classes, n_epochs=n_epochs,
                backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay,
                early_stopping_patience=early_stopping_patience,
                scheduler_patience=scheduler_patience, freeze_backbone=freeze_backbone,
                class_weights=self.compute_class_weights(y_train_fold, n_classes),
                save_path=os.path.join(save_dir, f"fold_{fold_idx + 1}.pth"),
            )

            fold_plot_path = (
                os.path.join(save_dir, f"{model_name}_fold_{fold_idx + 1}_confusion_matrix.png")
                if plot_per_fold else None
            )

            test_preds, test_labels = self.evaluate_finetuned(
                class_names=class_names,
                title_suffix=f" ({model_name}, fold {fold_idx + 1})",
                save_path=fold_plot_path,
                plot=plot_per_fold,
            )

            report = classification_report(
                test_labels, test_preds, target_names=class_names, output_dict=True
            )
            fold_results.append(report)
            if on_fold_end is not None:
                on_fold_end(fold_idx + 1, report)

            all_true_labels.extend(test_labels)
            all_pred_labels.extend(test_preds)

        self.plot_confusion_matrix(
            all_true_labels, all_pred_labels, class_names=class_names,
            title_suffix=f" ({model_name}, all folds, CV)",
            save_path=os.path.join(save_dir, f"{model_name}_cv_confusion_matrix.png"),
        )

        accuracies = [r["accuracy"] for r in fold_results]
        macro_f1s = [r["macro avg"]["f1-score"] for r in fold_results]

        print()
        print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Mean Macro-F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")

        return fold_results