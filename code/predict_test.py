import argparse
import os
from pathlib import Path


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from data_io import load_test_features, load_training_data
from evaluation import _positive_probability, choose_threshold
from models import get_model_specs
from results_io import save_predictions


DEFAULT_MODEL = "Stacking"
STUDENT_ID = "1258037"
ROOT = Path(__file__).resolve().parents[1]

# Edit these paths directly before the demo if your files are elsewhere.
TRAIN_X_PATH = ROOT / "data" / "train_X.xlsx"
TRAIN_Y_PATH = ROOT / "data" / "train_y.xlsx"
TEST_X_PATH = ROOT / "data" / "test_X.xlsx"
OUTPUT_PATH = ROOT / f"{STUDENT_ID}_Results.xlsx"


def parse_args():
    parser = argparse.ArgumentParser(description="Train final model and create test predictions.")
    parser.add_argument("--train-x", default=TRAIN_X_PATH, type=Path)
    parser.add_argument("--train-y", default=TRAIN_Y_PATH, type=Path)
    parser.add_argument("--test-x", default=TEST_X_PATH, type=Path)
    parser.add_argument("--output", default=OUTPUT_PATH, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    X_train, y_train, _ = load_training_data(args.train_x, args.train_y)
    X_test, test_ids = load_test_features(args.test_x)
    specs = {spec.name: spec for spec in get_model_specs(seed=args.seed)}
    if args.model not in specs:
        raise ValueError(f"Unknown model '{args.model}'. Choices: {sorted(specs)}")

    estimator = specs[args.model].estimator
    estimator.fit(X_train, y_train)
    train_prob = _positive_probability(estimator, X_train)
    threshold = choose_threshold(y_train, train_prob)
    test_prob = _positive_probability(estimator, X_test)
    test_pred = (test_prob >= threshold).astype(int)
    save_predictions(test_ids, test_pred, test_prob, args.output)
    print(f"Wrote {args.output} using {args.model} at threshold {threshold:.2f}")


if __name__ == "__main__":
    main()
