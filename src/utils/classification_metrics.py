from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve
)
from pathlib import Path
import json


def save_classification_report(
    targets: pd.DataFrame,
    outputs: pd.DataFrame,
    plots_dir: Path,
    is_train: bool,
    file_name: str,
) -> None:
    y_pred = outputs.values  # Model predictions
    y_true = targets.values

    # convert to ints
    y_pred = y_pred.round().astype(int)
    y_true = y_true.round().astype(int)

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            f"Channel {i}" for i in range(y_true.shape[1])
        ],  # noqa: E501
        output_dict=True,
        zero_division=0,
    )
    file_name = file_name + "_train" if is_train else file_name + "_test"
    with open(plots_dir / f"{file_name}.json", "w") as f:
        json.dump(report, f, indent=4)


def save_roc_auc_scores(
    targets: pd.DataFrame,
    outputs: pd.DataFrame,
    plots_dir: Path,
    is_train: bool,
    file_name: str,
) -> None:
    """
    Computes ROC-AUC scores per channel, saves them to a JSON file,
    and plots the ROC curves for all channels on a single figure.

    :param targets: Ground truth values (pd.DataFrame)
    :param outputs: Model outputs (probabilities before rounding) (pd.DataFrame)
    :param plots_dir: Directory to save the results
    :param is_train: Whether the data is from training or test set
    :param file_name: Base name for the output file
    """
    y_true = targets.values
    y_pred = outputs.values  # Probabilities before rounding

    roc_auc_scores = {}
    plt.figure(figsize=(10, 8))

    for i in range(y_true.shape[1]):
        try:
            auc_score = roc_auc_score(y_true[:, i], y_pred[:, i])
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])

            roc_auc_scores[f"Channel {i}"] = auc_score
            plt.plot(fpr, tpr, label=f"Channel {i} (AUC = {auc_score:.3f})")

        except ValueError:
            roc_auc_scores[f"Channel {i}"] = None  # Handle cases with only one class

    # Plot settings
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    if is_train:
        plt.title("ROC Curves per Channel (Train)")
    else:
        plt.title("ROC Curves per Channel (Test)")

    plt.legend(loc="lower right")

    # Save JSON results
    file_name = file_name + "_train" if is_train else file_name + "_test"
    with open(plots_dir / f"{file_name}.json", "w") as f:
        json.dump(roc_auc_scores, f, indent=4)

    # Save plot
    plt.savefig(plots_dir / f"{file_name}_roc_auc.png", dpi=300)
    plt.close()
