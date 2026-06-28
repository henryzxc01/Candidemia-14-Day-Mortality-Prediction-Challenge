import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold


THRESHOLDS = np.round(np.linspace(0.10, 0.90, 81), 2)


def _positive_probability(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    scores = estimator.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def choose_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in THRESHOLDS:
        y_pred = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        score = f1 + mcc
        if score > best_score or (
            np.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def evaluate_model(spec, X, y, seed=42, cv_folds=5):
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_scores = []

    for train_idx, valid_idx in splitter.split(X, y):
        estimator = clone(spec.estimator)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        estimator.fit(X_train, y_train)
        train_prob = _positive_probability(estimator, X_train)
        valid_prob = _positive_probability(estimator, X_valid)
        threshold = choose_threshold(y_train, train_prob)
        valid_pred = (valid_prob >= threshold).astype(int)

        fold_scores.append(
            {
                "f1": f1_score(y_valid, valid_pred, zero_division=0),
                "mcc": matthews_corrcoef(y_valid, valid_pred),
                "auroc": roc_auc_score(y_valid, valid_prob),
                "threshold": threshold,
            }
        )

    return fold_scores


def summarize_scores(spec, fold_scores, seed=42, cv_folds=5):
    def mean(metric):
        return round(float(np.mean([fold[metric] for fold in fold_scores])), 4)

    def std(metric):
        return round(float(np.std([fold[metric] for fold in fold_scores], ddof=1)), 4)

    return {
        "method": spec.name,
        "f1_mean": mean("f1"),
        "f1_std": std("f1"),
        "mcc_mean": mean("mcc"),
        "mcc_std": std("mcc"),
        "auroc_mean": mean("auroc"),
        "auroc_std": std("auroc"),
        "cv_folds": cv_folds,
        "seed": seed,
        "key_hyperparams": spec.key_hyperparams,
    }


def evaluate_models(model_specs, X, y, seed=42, cv_folds=5):
    rows = []
    for spec in model_specs:
        fold_scores = evaluate_model(spec, X, y, seed=seed, cv_folds=cv_folds)
        rows.append(summarize_scores(spec, fold_scores, seed=seed, cv_folds=cv_folds))
    return rows
