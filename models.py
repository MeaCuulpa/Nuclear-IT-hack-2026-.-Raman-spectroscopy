from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


def _get_seed(config, default: int = 42) -> int:
    try:
        return int(config.project.seed)
    except Exception:
        return default


def _is_enabled(config, model_name: str, default: bool = True) -> bool:
    try:
        return bool(getattr(config.training.models, model_name))
    except Exception:
        return default


def make_baseline_models(config) -> dict:
    random_state = _get_seed(config)
    models = {}

    if _is_enabled(config, "linear_svm", False):
        models["linear_svm"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        random_state=random_state,
                        max_iter=20000,
                    ),
                ),
            ]
        )

    if _is_enabled(config, "pca_svm", False):
        models["pca_svm"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=20, svd_solver="full", whiten=False)),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if _is_enabled(config, "logreg", False):
        models["logreg"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        multi_class="auto",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if _is_enabled(config, "rf", True):
        models["rf"] = RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    if _is_enabled(config, "extra_trees", True):
        models["extra_trees"] = ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    if not models:
        raise ValueError("No models enabled in config.training.models")

    return models
