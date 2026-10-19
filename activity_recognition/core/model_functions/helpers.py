"""Small static helpers shared by the training mixins."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


class SplitHelpers:

    @staticmethod
    def compute_class_weights(y, n_classes, power=0.5):
        """Inverse-frequency^power weights, normalised to mean 1. power=1 is full balancing, 0 is none."""
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        w = (counts.sum() / (n_classes * np.maximum(counts, 1))) ** power
        return (w / w.mean()).astype(np.float32)

    @staticmethod
    def _assert_disjoint_groups(groups_all, **index_sets):
        """Raise if any subject (group) appears in more than one of the given index sets."""
        names = list(index_sets)
        sets = {n: set(np.unique(groups_all[index_sets[n]])) for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = sets[a] & sets[b]
                if overlap:
                    raise RuntimeError(
                        f"Subject leakage between '{a}' and '{b}': {len(overlap)} shared groups, "
                        f"e.g. {list(overlap)[:3]}"
                    )

    @staticmethod
    def _warn_sparse_val(y_val, n_classes, min_count=20):
        val_counts = np.bincount(y_val, minlength=n_classes)
        if (val_counts < min_count).any():
            print(f"  WARNING: sparse inner-val classes {val_counts.tolist()} - val F1 will be noisy")

    @staticmethod
    def _inner_split(train_idx, y, groups, seed=42, n_splits=5):
        """Subject-wise inner train/val split of `train_idx`; returns global indices."""
        inner = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        itr, iva = next(inner.split(train_idx, y[train_idx], groups[train_idx]))
        return train_idx[itr], train_idx[iva]