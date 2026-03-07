from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


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

    if _is_enabled(config, "pca_svm", True):
        models["pca_svm"] = Pipeline(
            steps=[
                ("pca", PCA(n_components=0.99, svd_solver="full")),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=3.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if _is_enabled(config, "rf", True):
        models["rf"] = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    if not models:
        raise ValueError("No models enabled in config.training.models")

    return models