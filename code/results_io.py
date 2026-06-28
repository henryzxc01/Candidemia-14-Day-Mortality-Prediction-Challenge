from pathlib import Path

import pandas as pd


CV_RESULT_COLUMNS = [
    "method",
    "f1_mean",
    "f1_std",
    "mcc_mean",
    "mcc_std",
    "auroc_mean",
    "auroc_std",
    "cv_folds",
    "seed",
    "key_hyperparams",
]

PREDICTION_COLUMNS = ["id", "prediction", "probability"]


def save_cv_results(rows, output_path):
    df = pd.DataFrame(rows)
    missing = [col for col in CV_RESULT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing CV result columns: {missing}")
    df = df[CV_RESULT_COLUMNS]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


def save_predictions(ids, predictions, probabilities, output_path):
    df = pd.DataFrame(
        {
            "id": ids,
            "prediction": predictions,
            "probability": probabilities,
        }
    )
    df = df[PREDICTION_COLUMNS]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
