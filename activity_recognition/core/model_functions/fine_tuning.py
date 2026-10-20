import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score


class FineTuneMixin:
    """Base class for fine-tuning models."""

    BACKBONE_ATTR_NAMES = ["backbone", "encoder"]
    HEAD_ATTR_NAMES = ["dense", "classifier", "fc", "head"]

    def create_finetune_model(self, n_classes):
        """
        Override in subclass - model-specific fine-tuning model creation.
        Args:
            n_classes: the number of classes for the classification head
        """
        raise NotImplementedError

    def prepare_input(self, x):
        """
        Override in subclass - model-specific input preparation (e.g., channel selection, normalization).
        Args:
            x: the input data
        Returns:
            the prepared input data
        """
        return x

    def _find_named_submodule(self, model, attr_names):
        """
        Find a submodule by its attribute name.
        Args:
            model: the model to search
            attr_names: a list of attribute names to check
        Returns:
            a tuple of (name, module) if found, otherwise (None, None)
        """
        for name in attr_names:
            module = getattr(model, name, None)
            if isinstance(module, nn.Module):
                return name, module
        return None, None

    def _split_backbone_head_params(self, model):
        """
        Returns (backbone_params, head_params, backbone_module, head_module).
        If no backbone or head submodule can be identified, returns ([], all_params, None, None).
        Override in subclass if model-specific.
        
        Args:
            model: the model to split
        Returns:
            tuple: a tuple containing the backbone parameters, head parameters, backbone module, and head module
        """
        _, backbone_module = self._find_named_submodule(model, self.BACKBONE_ATTR_NAMES)

        if backbone_module is not None:
            backbone_param_ids = {id(p) for p in backbone_module.parameters()}
            backbone_params = [p for p in backbone_module.parameters() if p.requires_grad]
            head_params = [
                p for p in model.parameters()
                if p.requires_grad and id(p) not in backbone_param_ids
            ]
            return backbone_params, head_params, backbone_module, None

        _, head_module = self._find_named_submodule(model, self.HEAD_ATTR_NAMES)

        if head_module is not None:
            head_param_ids = {id(p) for p in head_module.parameters()}
            head_params = [p for p in head_module.parameters() if p.requires_grad]
            backbone_params = [
                p for p in model.parameters()
                if p.requires_grad and id(p) not in head_param_ids
            ]
            return backbone_params, head_params, None, head_module

        print(
            "WARNING: could not identify a backbone or head submodule "
            f"(looked for {self.BACKBONE_ATTR_NAMES} as backbone, "
            f"{self.HEAD_ATTR_NAMES} as head). All parameters will be "
            "trained as a single group at head_lr, and freeze_backbone "
            "will have no effect. Override BACKBONE_ATTR_NAMES / "
            "HEAD_ATTR_NAMES on your subclass if this is wrong."
        )
        return [], [p for p in model.parameters() if p.requires_grad], None, None


    def fine_tune(self, n_classes, n_epochs=20, backbone_lr=1e-5, head_lr=1e-4,
                  weight_decay=1e-4, head_weight_decay_scale=0.1, label_smoothing=0.1, max_grad_norm=1.0,
                  early_stopping_patience=8, scheduler_patience=3, freeze_backbone=False, min_delta=1e-3,
                  unfreeze_after_epoch=0, class_weights=None, save_path="best_model.pth",
                  max_train_val_gap=None):
        """
        Fine-tunes the model on the training set, using the validation set for early stopping.
        Checkpoints are selected on validation macro-F1 (not accuracy).

        Args:
            n_classes: the number of classes for the classification head
            n_epochs: the number of epochs to train for
            backbone_lr: the learning rate for the backbone
            head_lr: the learning rate for the classification head
            weight_decay: the weight decay for the optimizer
            head_weight_decay_scale: scale factor for weight decay on head parameters
            label_smoothing: label smoothing factor for cross-entropy loss
            max_grad_norm: maximum gradient norm for clipping (None to disable)
            early_stopping_patience: number of epochs with no improvement to wait before stopping
            scheduler_patience: number of epochs with no improvement to wait before reducing LR
            freeze_backbone: whether to freeze the backbone during training
            min_delta: minimum change in validation macro-F1 to qualify as an improvement
            unfreeze_after_epoch: epoch after which to unfreeze the backbone (if frozen)
            class_weights: weights for each class (for imbalanced datasets)
            save_path: path to save the best model checkpoint
            max_train_val_gap: maximum allowed gap between training and validation accuracy (None to disable)

        Returns:
            tuple: a tuple containing the best model and the best validation macro-F1 score
        """
        self.classification_model = self.create_finetune_model(n_classes).to(self.device)

        backbone_params, head_params, backbone_module, head_module = self._split_backbone_head_params(
            self.classification_model
        )

        print(
            f"Backbone param count: {sum(p.numel() for p in backbone_params)} | "
            f"Head param count: {sum(p.numel() for p in head_params)}"
        )

        warmup_active = freeze_backbone and unfreeze_after_epoch > 0

        def set_backbone_requires_grad(flag):
            """
            Sets the requires_grad attribute of the backbone parameters to the given flag.
            Args:
                flag: boolean indicating whether to require gradients for the backbone parameters
            """
            if backbone_module is not None:
                for param in backbone_module.parameters():
                    param.requires_grad = flag
            elif head_module is not None:
                head_param_ids = {id(p) for p in head_module.parameters()}
                for p in self.classification_model.parameters():
                    if id(p) not in head_param_ids:
                        p.requires_grad = flag
            else:
                print("WARNING: freeze_backbone=True but no backbone/head "
                      "submodule could be identified, so nothing was frozen.")

        def build_optimizer_and_scheduler():
            """
            Builds the optimizer and learning rate scheduler.
            Returns:
                tuple: a tuple containing the optimizer and learning rate scheduler
            """
            b_params, h_params, _, _ = self._split_backbone_head_params(self.classification_model)
            groups = []
            if b_params:
                groups.append({"params": b_params, "lr": backbone_lr, "weight_decay": weight_decay})
            if h_params:
                groups.append({"params": h_params, "lr": head_lr,
                               "weight_decay": weight_decay * head_weight_decay_scale})
            opt = optim.AdamW(groups)
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=0.1, patience=scheduler_patience
            )
            return opt, sched

        if freeze_backbone:
            set_backbone_requires_grad(False)

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        optimizer, scheduler = build_optimizer_and_scheduler()

        best_val_acc = -float("inf")
        best_val_f1 = -float("inf")
        epochs_no_improve = 0

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for epoch in range(n_epochs):
            t_epoch = time.time()

            if warmup_active and epoch == unfreeze_after_epoch:
                print(f"  Unfreezing backbone at epoch {epoch + 1} (warmup complete)")
                set_backbone_requires_grad(True)
                optimizer, scheduler = build_optimizer_and_scheduler()
                warmup_active = False

            self.classification_model.train()
            total_loss, total_samples, total_correct, skipped_batches = 0.0, 0, 0, 0

            for x_batch, y_batch in self.train_loader:
                x_batch = self.prepare_input(x_batch.to(self.device))
                y_batch = y_batch.to(self.device).long()

                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self.classification_model(x_batch)
                    loss = criterion(logits, y_batch)

                if not torch.isfinite(loss):
                    skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()

                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]],
                        max_norm=max_grad_norm,
                    )

                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * y_batch.size(0)
                total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total_samples += y_batch.size(0)

            if skipped_batches:
                print(f"  WARNING: skipped {skipped_batches} batches with non-finite loss")

            train_loss = total_loss / max(total_samples, 1)
            train_acc = total_correct / max(total_samples, 1)

            # ---------------- validation ----------------
            self.classification_model.eval()
            val_preds, val_labels = [], []
            val_nonfinite = 0

            with torch.no_grad():
                for x_batch, y_batch in self.val_loader:
                    x_batch = self.prepare_input(x_batch.to(self.device))
                    y_batch = y_batch.to(self.device).long()

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = self.classification_model(x_batch)

                    logits = logits.float()
                    bad = ~torch.isfinite(logits).all(dim=1)
                    preds = logits.argmax(dim=1)
                    preds[bad] = -1  # NaN logits must not silently become class 0
                    val_nonfinite += bad.sum().item()

                    val_preds.append(preds.cpu().numpy())
                    val_labels.append(y_batch.cpu().numpy())

            val_preds = np.concatenate(val_preds)
            val_labels = np.concatenate(val_labels)

            val_acc = float((val_preds == val_labels).mean())
            val_f1 = f1_score(val_labels, val_preds, labels=list(range(n_classes)),
                              average="macro", zero_division=0)

            scheduler.step(val_f1)

            # Epoch 1 prints on its own line; later epochs overwrite in place.
            print(
                f"Epoch {epoch + 1:02d}/{n_epochs} | Train Loss={train_loss:.4f} | "
                f"Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}"
                + (f" | NON-FINITE val logits: {val_nonfinite}" if val_nonfinite else "")
                + f" | epoch time: {time.time() - t_epoch:.1f}s")

            # ---------------- checkpointing / stopping ----------------
            if val_f1 > best_val_f1 + min_delta:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(self.classification_model.state_dict(), save_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if max_train_val_gap is not None:
                gap = train_acc - val_acc
                if gap > max_train_val_gap:
                    print(
                        f"Stopping at epoch {epoch + 1}: "
                        f"train/val gap {gap:.4f} exceeded "
                        f"max_train_val_gap={max_train_val_gap} "
                        f"(Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f})"
                    )
                    break

        self.classification_model.load_state_dict(torch.load(save_path, map_location=self.device))
        print(f"Loaded best model with Val F1={best_val_f1:.4f} (Val Acc={best_val_acc:.4f})")

        return self.classification_model, best_val_f1

    def evaluate_finetuned(self, class_names=None, title_suffix=" (Fine-tuned)",
                           save_path=None, plot=True):
        """
        Evaluates the fine-tuned model on the test set and optionally plots a confusion matrix.

        Args:
            class_names: a list of class names (optional)
            title_suffix: a suffix for the plot title (optional)
            save_path: a path to save the confusion matrix plot as a PNG file (optional)
            plot: whether to plot the confusion matrix (default: True)
        """
        self.classification_model.eval()

        all_preds, all_labels = [], []
        n_bad = 0
        use_amp = self.device.type == "cuda"

        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = self.prepare_input(x_batch.to(self.device))
                y_batch = y_batch.to(self.device).long()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self.classification_model(x_batch)

                logits = logits.float()
                bad = ~torch.isfinite(logits).all(dim=1)
                preds = logits.argmax(dim=1)
                # Count non-finite outputs as errors, not as class 0.
                preds[bad] = (y_batch[bad] + 1) % logits.shape[1]
                n_bad += bad.sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        if n_bad:
            print(f"WARNING: {n_bad} test windows produced non-finite logits (counted as errors)")

        print("=== Test Performance (fine-tuned) ===")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        if plot:
            self.plot_confusion_matrix(
                all_labels, all_preds, class_names=class_names,
                title_suffix=title_suffix, save_path=save_path,
            )

        return np.asarray(all_preds), np.asarray(all_labels)