import os

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


class PlottingMixin:

    def plot_confusion_matrix(self, all_labels, all_preds, class_names=None, title_suffix="",
                              normalise="true", save_path=None, display=True):
        """Renders a confusion matrix and saves it to `save_path` as a PNG.
        Args:
            all_labels: the true labels
            all_preds: the predicted labels
            class_names: a list of class names (optional)
            title_suffix: a suffix for the plot title (optional)
            normalise: whether to normalise the confusion matrix (default: True)
            save_path: a path to save the confusion matrix plot as a PNG file (optional)
            display: whether to display the plot (default: True)
        Returns:
            the path to the saved confusion matrix
        """
        cm = confusion_matrix(all_labels, all_preds, normalize=normalise)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8, 8))
        values_format = ".1%" if normalise else "d"
        accuracy = accuracy_score(all_labels, all_preds)
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True, values_format=values_format)
        plt.title(f"Confusion Matrix{title_suffix} || Accuracy: {accuracy:.1%}")
        plt.tight_layout()

        if save_path is None:
            safe_suffix = (title_suffix or "confusion_matrix").strip().replace(" ", "_").replace("/", "-")
            save_path = os.path.join("confusion_matrices", f"{safe_suffix}.png")

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        if display:
            plt.show(block=False)
            plt.pause(0.5)

        print(f"Confusion matrix saved to {save_path}")
        return save_path