import pandas as pd
from sklearn.metrics import classification_report
from pathlib import Path
import json


def create_classification_report(
    targets: pd.DataFrame,
    outputs: pd.DataFrame,
) -> dict:
    y_pred = outputs.values  # Model predictions
    y_true = targets.values

    # convert to ints
    y_pred = y_pred.round().astype(int)
    y_true = y_true.round().astype(int)

    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=[f"Channel {i}" for i in range(y_true.shape[1])],
        output_dict=True,
        zero_division=0,
    )
    return clf_report


def save_classification_report(
    clf_report: dict, save_dir: Path, file_name: str, is_train: bool
) -> None:
    """
    Save the classification report to a JSON file.
    Args:
        clf_report (dict): The classification report.
        file_name (str): The name of the file to save the report to.
        is_train (bool): Whether the report is for training or testing data.
    """
    file_name = file_name + "_train" if is_train else file_name + "_test"
    with open(save_dir / f"{file_name}.json", "w") as f:
        json.dump(clf_report, f, indent=4)
