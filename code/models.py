from dataclasses import dataclass
import os


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    key_hyperparams: str


def _linear_pipeline(model):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _tree_pipeline(model):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _core_estimators(seed):
    logistic = _linear_pipeline(
        LogisticRegression(
            C=0.2,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
            solver="liblinear",
        )
    )
    bagging = _tree_pipeline(
        BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=seed,
            ),
            n_estimators=400,
            max_samples=0.85,
            max_features=0.85,
            bootstrap=True,
            random_state=seed,
            n_jobs=1,
        )
    )
    random_forest = _tree_pipeline(
        RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    )
    extra_trees = _tree_pipeline(
        ExtraTreesClassifier(
            n_estimators=800,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    )
    gradient_boosting = _tree_pipeline(
        GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.8,
            random_state=seed,
        )
    )
    ada_boost = _tree_pipeline(
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=2,
                min_samples_leaf=6,
                class_weight="balanced",
                random_state=seed,
            ),
            n_estimators=250,
            learning_rate=0.04,
            random_state=seed,
        )
    )

    return {
        "lr": logistic,
        "bag": bagging,
        "rf": random_forest,
        "et": extra_trees,
        "ada": ada_boost,
        "gb": gradient_boosting,
    }


def _make_stacking(seed, base_keys=("lr", "rf", "et", "gb"), final_c=0.5, passthrough=False):
    estimators = _core_estimators(seed)
    final_lr = LogisticRegression(
        C=final_c,
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
        solver="liblinear",
    )
    final_estimator = _linear_pipeline(final_lr) if passthrough else final_lr
    return StackingClassifier(
        estimators=[(key, estimators[key]) for key in base_keys],
        final_estimator=final_estimator,
        cv=3,
        stack_method="predict_proba",
        n_jobs=1,
        passthrough=passthrough,
    )


def get_model_specs(seed=42):
    estimators = _core_estimators(seed)
    soft_voting = VotingClassifier(
        estimators=[
            ("lr", estimators["lr"]),
            ("rf", estimators["rf"]),
            ("et", estimators["et"]),
            ("gb", estimators["gb"]),
            ("ada", estimators["ada"]),
        ],
        voting="soft",
        weights=[1.0, 1.0, 1.0, 1.2, 0.8],
        n_jobs=1,
    )
    stacking = _make_stacking(seed, base_keys=("lr", "rf", "et", "gb"), final_c=0.5, passthrough=False)

    return [
        ModelSpec("LogisticRegression", estimators["lr"], "C=0.2, class_weight=balanced, imputer=median"),
        ModelSpec("Bagging", estimators["bag"], "DecisionTree max_depth=4, n_estimators=400, max_samples=0.85"),
        ModelSpec("RandomForest", estimators["rf"], "n_estimators=500, max_depth=5, min_samples_leaf=5"),
        ModelSpec("ExtraTrees", estimators["et"], "n_estimators=800, max_depth=5, min_samples_leaf=4"),
        ModelSpec("AdaBoost", estimators["ada"], "DecisionTree max_depth=2, n_estimators=250, learning_rate=0.04"),
        ModelSpec("GradientBoosting", estimators["gb"], "n_estimators=250, learning_rate=0.03, max_depth=2"),
        ModelSpec("SoftVoting", soft_voting, "LR/RF/ET/GB/Ada soft voting, weights=1/1/1/1.2/0.8"),
        ModelSpec("Stacking", stacking, "LR/RF/ET/GB base learners, final LR C=0.5, cv=3"),
    ]


def get_stacking_tuning_specs(seed=42):
    variants = [
        ("Stacking_C0.1", ("lr", "rf", "et", "gb"), 0.1, False),
        ("Stacking_C0.5", ("lr", "rf", "et", "gb"), 0.5, False),
        ("Stacking_C1.0", ("lr", "rf", "et", "gb"), 1.0, False),
        ("Stacking_C0.5_PT", ("lr", "rf", "et", "gb"), 0.5, True),
        ("Stacking_NoLR_C0.5", ("rf", "et", "gb"), 0.5, False),
        ("Stacking_Ens_C0.5", ("bag", "rf", "et", "gb", "ada"), 0.5, False),
    ]
    specs = []
    for name, base_keys, final_c, passthrough in variants:
        base_label = "/".join(base_keys).upper()
        specs.append(
            ModelSpec(
                name,
                _make_stacking(seed, base_keys=base_keys, final_c=final_c, passthrough=passthrough),
                f"base={base_label}, final LR C={final_c}, cv=3, passthrough={passthrough}",
            )
        )
    return specs
