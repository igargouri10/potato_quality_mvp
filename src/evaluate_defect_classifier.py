from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PREDICTIONS_PATH = Path("data/processed/defect_image_classification/batch_classifications.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/defect_image_classification/model_metrics.csv")

LABEL_ALIASES = {
    "good": "Good",
    "likely good": "Good",
    "pass": "Good",
    "acceptable": "Good",
    "review": "Bad",
    "warning": "Bad",
    "bad": "Bad",
    "likely bad": "Bad",
    "poor": "Bad",
    "fail": "Bad",
    "reject": "Bad",
    "rejected": "Bad",
}

PREDICTION_ALIASES = {
    "likely good": "Good",
    "good": "Good",
    "review": "Bad",
    "likely bad": "Bad",
    "bad": "Bad",
}


def _normalize_label(value: object, aliases: dict[str, str]) -> str:
    text = str(value).strip().lower()
    return aliases.get(text, text.title())


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series, positive_label: str = "Bad") -> dict[str, float]:
    true_positive = int(((y_true == positive_label) & (y_pred == positive_label)).sum())
    true_negative = int(((y_true != positive_label) & (y_pred != positive_label)).sum())
    false_positive = int(((y_true != positive_label) & (y_pred == positive_label)).sum())
    false_negative = int(((y_true == positive_label) & (y_pred != positive_label)).sum())

    accuracy = _safe_divide(true_positive + true_negative, len(y_true))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    specificity = _safe_divide(true_negative, true_negative + false_positive)

    return {
        "records_evaluated": int(len(y_true)),
        "accuracy": round(accuracy, 4),
        "precision_bad": round(precision, 4),
        "recall_bad": round(recall, 4),
        "f1_bad": round(f1, 4),
        "specificity_good": round(specificity, 4),
        "true_positive_bad": true_positive,
        "true_negative_good": true_negative,
        "false_positive_bad": false_positive,
        "false_negative_bad": false_negative,
    }


def evaluate_defect_classifier(
    labels_path: str | Path,
    predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Evaluate batch visual classifications against human-reviewed labels."""
    labels_path = Path(labels_path)
    predictions_path = Path(predictions_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_path, dtype=str)
    predictions = pd.read_csv(predictions_path, dtype=str)

    required_label_columns = {"batch", "true_label"}
    missing_label_columns = required_label_columns - set(labels.columns)
    if missing_label_columns:
        raise ValueError(f"Label file is missing column(s): {', '.join(sorted(missing_label_columns))}")

    if "batch" not in predictions.columns or "batch_status" not in predictions.columns:
        raise ValueError("Prediction file must contain batch and batch_status columns.")

    labels = labels.copy()
    predictions = predictions.copy()
    labels["batch"] = labels["batch"].astype(str).str.strip().str.lstrip("0")
    predictions["batch"] = predictions["batch"].astype(str).str.strip().str.lstrip("0")
    labels["true_label_normalized"] = labels["true_label"].map(lambda value: _normalize_label(value, LABEL_ALIASES))
    predictions["predicted_label_normalized"] = predictions["batch_status"].map(
        lambda value: _normalize_label(value, PREDICTION_ALIASES)
    )

    join_columns = ["batch"]
    for optional_column in ["po_number", "supplier"]:
        if optional_column in labels.columns and optional_column in predictions.columns:
            labels[optional_column] = labels[optional_column].astype(str).str.strip()
            predictions[optional_column] = predictions[optional_column].astype(str).str.strip()
            join_columns.append(optional_column)

    evaluated = labels.merge(
        predictions,
        on=join_columns,
        how="inner",
        suffixes=("_label", "_prediction"),
    )
    if evaluated.empty:
        raise ValueError("No labels matched the prediction file. Check batch, PO number, and supplier values.")

    metrics = _binary_metrics(
        evaluated["true_label_normalized"],
        evaluated["predicted_label_normalized"],
        positive_label="Bad",
    )

    confusion = pd.crosstab(
        evaluated["true_label_normalized"],
        evaluated["predicted_label_normalized"],
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    )

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path, index=False)
    evaluated_output = output_path.with_name(output_path.stem + "_matched_predictions.csv")
    confusion_output = output_path.with_name(output_path.stem + "_confusion_matrix.csv")
    evaluated.to_csv(evaluated_output, index=False)
    confusion.to_csv(confusion_output)

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "records_evaluated": int(len(evaluated)),
        "metrics_output": str(output_path),
        "matched_predictions_output": str(evaluated_output),
        "confusion_matrix_output": str(confusion_output),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate visual defect classifier performance.")
    parser.add_argument("labels_path", help="CSV with batch and true_label columns.")
    parser.add_argument("--predictions-path", default=str(DEFAULT_PREDICTIONS_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    result = evaluate_defect_classifier(args.labels_path, args.predictions_path, args.output_path)
    print(f"Records evaluated: {result['records_evaluated']}")
    for metric, value in result["metrics"].items():
        print(f"{metric}: {value}")
    print(f"Metrics output: {result['metrics_output']}")
    print(f"Confusion matrix output: {result['confusion_matrix_output']}")
