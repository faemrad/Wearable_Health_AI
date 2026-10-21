import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


class SplitHelpers:
    """Helper functions for train/val splits.

    A collection of static utilities for building subject-wise (group-aware)
    train/validation splits, guarding against subject leakage, and computing
    class weights for imbalanced classification.
    """

    @staticmethod
    def compute_class_weights(y, n_classes, power=0.5):
        """Compute smoothed inverse-frequency class weights.

        Weights are proportional to ``(1 / class_frequency) ** power`` and are
        then normalised so that their mean is 1. ``power=1`` gives full
        balancing (classes contribute equally in expectation), while
        ``power=0`` gives uniform weights (no rebalancing). Intermediate values
        such as the default 0.5 give a softer correction. Classes with zero
        samples are treated as having a count of 1 to avoid division by zero.

        Args:
            y (np.ndarray): Integer class labels of shape ``(n_samples,)``,
                with values in ``[0, n_classes)``.
            n_classes (int): Total number of classes. Classes absent from
                ``y`` still receive a weight entry.
            power (float, optional): Exponent applied to the inverse-frequency
                weights. Defaults to 0.5.

        Returns:
            np.ndarray: Float32 array of shape ``(n_classes,)`` containing the
            per-class weights, normalised to have mean 1.
        """
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        w = (counts.sum() / (n_classes * np.maximum(counts, 1))) ** power
        return (w / w.mean()).astype(np.float32)

    @staticmethod
    def _assert_disjoint_groups(groups_all, **index_sets):
        """Assert that no group (subject) appears in more than one index set.

        Used as a leakage check: every pair of the supplied index sets is
        compared, and an error is raised if any group is present in both.

        Args:
            groups_all (np.ndarray): Group (subject) ID for every sample in the
                full dataset, shape ``(n_samples,)``.
            **index_sets: Named arrays of integer indices into ``groups_all``,
                e.g. ``train=train_idx, val=val_idx, test=test_idx``. The
                keyword names are used in the error message.

        Raises:
            RuntimeError: If any two index sets share at least one group. The
                message names the offending sets, the number of shared groups,
                and up to three example group IDs.
        """
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
        """Print a warning if any validation class has too few samples.

        Sparse classes make validation metrics (e.g. per-class or macro F1)
        high-variance, so this flags the issue without interrupting execution.

        Args:
            y_val (np.ndarray): Integer class labels of the validation set,
                shape ``(n_val_samples,)``.
            n_classes (int): Total number of classes, so that classes with zero
                validation samples are also counted.
            min_count (int, optional): Minimum acceptable number of validation
                samples per class. Defaults to 20.

        Returns:
            None: A warning is printed to stdout if any class count is below
            ``min_count``; nothing is returned.
        """
        val_counts = np.bincount(y_val, minlength=n_classes)
        if (val_counts < min_count).any():
            print(f"  WARNING: sparse inner-val classes {val_counts.tolist()} - val F1 will be noisy")

    @staticmethod
    def _inner_split(train_idx, y, groups, seed=42, n_splits=5):
        """Create a subject-wise inner train/validation split.

        Uses the first fold of a shuffled ``StratifiedGroupKFold`` over the
        samples in ``train_idx``, so that no group appears in both the inner
        train and inner validation sets while class proportions are kept as
        similar as possible. With ``n_splits=5`` this yields roughly an
        80/20 split.

        Args:
            train_idx (np.ndarray): Global indices of the samples to split,
                shape ``(n_train,)``.
            y (np.ndarray): Class labels for the full dataset, indexed by
                global index.
            groups (np.ndarray): Group (subject) IDs for the full dataset,
                indexed by global index.
            seed (int, optional): Random state for fold shuffling. Defaults
                to 42.
            n_splits (int, optional): Number of folds; the validation set is
                approximately ``1 / n_splits`` of the data. Defaults to 5.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(inner_train_idx, inner_val_idx)``,
            both expressed as global indices (i.e. subsets of ``train_idx``).
        """
        inner = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        itr, iva = next(inner.split(train_idx, y[train_idx], groups[train_idx]))
        return train_idx[itr], train_idx[iva]