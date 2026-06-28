import argparse
import os
from pathlib import Path


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from data_io import load_training_data
from evaluation import evaluate_models
from models import get_model_specs
from results_io import save_cv_results


DEFAULT_SEED = 42
DEFAULT_FOLDS = 5
STUDENT_ID = "1258037"


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run candidemia 14-day mortality CV experiments.")
    parser.add_argument("--train-x", default=root / "data" / "train_X.xlsx", type=Path)
    parser.add_argument("--train-y", default=root / "data" / "train_y.xlsx", type=Path)
    parser.add_argument("--output", default=root / "cv_results.xlsx", type=Path)
    parser.add_argument("--student-output", default=root / f"{STUDENT_ID}_cv_results.xlsx", type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--cv-folds", default=DEFAULT_FOLDS, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    X, y, _ = load_training_data(args.train_x, args.train_y)
    rows = evaluate_models(get_model_specs(seed=args.seed), X, y, seed=args.seed, cv_folds=args.cv_folds)
    save_cv_results(rows, args.output)
    save_cv_results(rows, args.student_output)

    for row in rows:
        print(
            f"{row['method']}: "
            f"F1={row['f1_mean']:.4f}, "
            f"MCC={row['mcc_mean']:.4f}, "
            f"AUROC={row['auroc_mean']:.4f}"
        )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.student_output}")


if __name__ == "__main__":
    main()
