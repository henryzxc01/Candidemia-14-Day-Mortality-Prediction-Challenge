import argparse
import os
from pathlib import Path


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd

from data_io import load_training_data
from evaluation import evaluate_models
from models import get_stacking_tuning_specs
from results_io import save_cv_results


DEFAULT_SEED = 42
DEFAULT_FOLDS = 5


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run a compact Stacking tuning sweep.")
    parser.add_argument("--train-x", default=root / "data" / "train_X.xlsx", type=Path)
    parser.add_argument("--train-y", default=root / "data" / "train_y.xlsx", type=Path)
    parser.add_argument("--output", default=root / "stacking_tuning_results.xlsx", type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--cv-folds", default=DEFAULT_FOLDS, type=int)
    return parser.parse_args()


def add_average_rank(rows):
    df = pd.DataFrame(rows)
    ranks = df[["f1_mean", "mcc_mean", "auroc_mean"]].rank(ascending=False, method="min")
    df["avg_rank"] = ranks.mean(axis=1).round(4)
    return df.sort_values(["avg_rank", "f1_mean", "mcc_mean", "auroc_mean"], ascending=[True, False, False, False])


def main():
    args = parse_args()
    X, y, _ = load_training_data(args.train_x, args.train_y)
    rows = evaluate_models(get_stacking_tuning_specs(seed=args.seed), X, y, seed=args.seed, cv_folds=args.cv_folds)
    ranked = add_average_rank(rows)
    save_cv_results(ranked.to_dict("records"), args.output)

    for _, row in ranked.iterrows():
        print(
            f"{row['method']}: "
            f"F1={row['f1_mean']:.4f}, "
            f"MCC={row['mcc_mean']:.4f}, "
            f"AUROC={row['auroc_mean']:.4f}, "
            f"avg_rank={row['avg_rank']:.4f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
