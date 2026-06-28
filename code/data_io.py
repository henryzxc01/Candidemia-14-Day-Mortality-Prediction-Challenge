from pathlib import Path

import pandas as pd


LABEL_COLUMN = "Deadin_D14"


def _drop_index_columns(df):
    index_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if index_cols:
        ids = df[index_cols[0]].copy()
        cleaned = df.drop(columns=index_cols)
    else:
        ids = pd.Series(range(len(df)), name="id")
        cleaned = df.copy()
    return cleaned, ids


def load_training_data(x_path, y_path):
    x_df = pd.read_excel(Path(x_path))
    y_df = pd.read_excel(Path(y_path))

    X, ids = _drop_index_columns(x_df)
    y_clean, y_ids = _drop_index_columns(y_df)

    if LABEL_COLUMN not in y_clean.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found in {y_path}")
    if len(X) != len(y_clean):
        raise ValueError(f"X/y length mismatch: {len(X)} != {len(y_clean)}")
    if len(ids) == len(y_ids) and not ids.reset_index(drop=True).equals(y_ids.reset_index(drop=True)):
        raise ValueError("X/y index columns do not match")

    y = y_clean[LABEL_COLUMN].astype(int)
    return X, y, ids


def load_test_features(x_path):
    x_df = pd.read_excel(Path(x_path))
    X, ids = _drop_index_columns(x_df)
    return X, ids
