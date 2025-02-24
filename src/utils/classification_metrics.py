import pandas as pd
from sklearn.metrics import classification_report
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
        target_names=[f"Channel {i}" for i in range(y_true.shape[1])],  # noqa: E501
        output_dict=True,
        zero_division=0,
    )
    file_name = file_name + "_train" if is_train else file_name + "_test"
    with open(plots_dir / f"{file_name}.json", "w") as f:
        json.dump(report, f, indent=4)
