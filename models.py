from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


class PLSLogRegClassifier(BaseEstimator, ClassifierMixin):
    """PLS feature extraction followed by multinomial logistic regression."""

    def __init__(
        self,
        n_components: int = 8,
        C: float = 1.0,
        class_weight: str | dict | None = "balanced",
        max_iter: int = 5000,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}

        y_onehot = np.zeros((len(y), len(self.classes_)), dtype=float)
        for row_idx, label in enumerate(y):
            y_onehot[row_idx, class_to_index[label]] = 1.0

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        n_components = int(max(1, min(self.n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])))
        self.pls_ = PLSRegression(n_components=n_components, scale=False)
        self.pls_.fit(X_scaled, y_onehot)
        X_scores = self.pls_.transform(X_scaled)

        self.logreg_ = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.logreg_.fit(X_scores, y)
        return self

    def _transform(self, X):
        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler_.transform(X)
        return self.pls_.transform(X_scaled)

    def predict(self, X):
        return self.logreg_.predict(self._transform(X))

    def predict_proba(self, X):
        return self.logreg_.predict_proba(self._transform(X))


class NamedCatBoostClassifier(CatBoostClassifier if CatBoostClassifier is not None else object):
    pass


SUPPORTED_MODEL_NAMES = (
    "linear_svm",
    "pca_svm",
    "logreg",
    "pls_logreg",
    "rf",
    "extra_trees",
    "catboost",
)


def _get_seed(config, default: int = 42) -> int:
    try:
        return int(config.project.seed)
    except Exception:
        return default


def _get_nested(node, path: str, default=None):
    current = node
    for part in path.split("."):
        if current is None or not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


def get_center_training_cfg(config, center: str):
    return _get_nested(config, f"training.centers.{center}", None)


def _get_center_or_global_value(config, center: str, path: str, fallback_path: str | None, default=None):
    center_cfg = get_center_training_cfg(config, center)
    value = _get_nested(center_cfg, path, None)
    if value is not None:
        return value
    if fallback_path is not None:
        value = _get_nested(config, fallback_path, None)
        if value is not None:
            return value
    return default


def is_model_enabled_for_center(config, center: str, model_name: str, default: bool = False) -> bool:
    center_cfg = get_center_training_cfg(config, center)
    if center_cfg is not None and hasattr(center_cfg, "models") and hasattr(center_cfg.models, model_name):
        return bool(getattr(center_cfg.models, model_name))

    training_models = _get_nested(config, "training.models", None)
    if training_models is not None and hasattr(training_models, model_name):
        return bool(getattr(training_models, model_name))

    return default


def _build_linear_svm(config, center: str):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=float(_get_center_or_global_value(config, center, "linear_svm.C", "training.linear_svm.C", 1.0)),
                    class_weight="balanced",
                    random_state=_get_seed(config),
                    max_iter=20000,
                ),
            ),
        ]
    )


def _build_pca_svm(config, center: str):
    from sklearn.decomposition import PCA

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=int(_get_center_or_global_value(config, center, "pca_svm.n_components", "training.pca_svm.n_components", 20)), svd_solver="full", whiten=False)),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=float(_get_center_or_global_value(config, center, "pca_svm.C", "training.pca_svm.C", 2.0)),
                    gamma=str(_get_center_or_global_value(config, center, "pca_svm.gamma", "training.pca_svm.gamma", "scale")),
                    class_weight="balanced",
                    probability=True,
                    random_state=_get_seed(config),
                ),
            ),
        ]
    )


def _build_logreg(config, center: str):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(_get_center_or_global_value(config, center, "logreg.C", "training.logreg.C", 1.0)),
                    class_weight="balanced",
                    max_iter=int(_get_center_or_global_value(config, center, "logreg.max_iter", "training.logreg.max_iter", 5000)),
                    random_state=_get_seed(config),
                ),
            ),
        ]
    )


def _build_pls_logreg(config, center: str, n_components: int | None = None):
    selected_n = n_components
    if selected_n is None:
        selected_n = int(_get_center_or_global_value(config, center, "pls_logreg.n_components", "training.pls_logreg.n_components", 8))

    return PLSLogRegClassifier(
        n_components=int(selected_n),
        C=float(_get_center_or_global_value(config, center, "pls_logreg.C", "training.pls_logreg.C", 1.0)),
        class_weight=str(_get_center_or_global_value(config, center, "pls_logreg.class_weight", "training.pls_logreg.class_weight", "balanced")),
        max_iter=int(_get_center_or_global_value(config, center, "pls_logreg.max_iter", "training.pls_logreg.max_iter", 5000)),
        random_state=_get_seed(config),
    )


def _build_rf(config, center: str):
    return RandomForestClassifier(
        n_estimators=int(_get_center_or_global_value(config, center, "rf.n_estimators", "training.rf.n_estimators", 800)),
        max_depth=_get_center_or_global_value(config, center, "rf.max_depth", "training.rf.max_depth", None),
        min_samples_split=int(_get_center_or_global_value(config, center, "rf.min_samples_split", "training.rf.min_samples_split", 4)),
        min_samples_leaf=int(_get_center_or_global_value(config, center, "rf.min_samples_leaf", "training.rf.min_samples_leaf", 2)),
        max_features=str(_get_center_or_global_value(config, center, "rf.max_features", "training.rf.max_features", "sqrt")),
        class_weight=str(_get_center_or_global_value(config, center, "rf.class_weight", "training.rf.class_weight", "balanced_subsample")),
        random_state=_get_seed(config),
        n_jobs=-1,
    )


def _build_extra_trees(config, center: str):
    return ExtraTreesClassifier(
        n_estimators=int(_get_center_or_global_value(config, center, "extra_trees.n_estimators", "training.extra_trees.n_estimators", 1000)),
        max_depth=_get_center_or_global_value(config, center, "extra_trees.max_depth", "training.extra_trees.max_depth", None),
        min_samples_split=int(_get_center_or_global_value(config, center, "extra_trees.min_samples_split", "training.extra_trees.min_samples_split", 4)),
        min_samples_leaf=int(_get_center_or_global_value(config, center, "extra_trees.min_samples_leaf", "training.extra_trees.min_samples_leaf", 2)),
        max_features=str(_get_center_or_global_value(config, center, "extra_trees.max_features", "training.extra_trees.max_features", "sqrt")),
        class_weight=str(_get_center_or_global_value(config, center, "extra_trees.class_weight", "training.extra_trees.class_weight", "balanced_subsample")),
        random_state=_get_seed(config),
        n_jobs=-1,
    )


def _build_catboost(config, center: str):
    if CatBoostClassifier is None:
        raise ImportError("catboost is not installed, but catboost model is enabled")

    return NamedCatBoostClassifier(
        iterations=int(_get_center_or_global_value(config, center, "catboost.iterations", "training.catboost.iterations", 500)),
        depth=int(_get_center_or_global_value(config, center, "catboost.depth", "training.catboost.depth", 5)),
        learning_rate=float(_get_center_or_global_value(config, center, "catboost.learning_rate", "training.catboost.learning_rate", 0.05)),
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        random_seed=_get_seed(config),
        verbose=False,
        allow_writing_files=False,
        thread_count=int(_get_center_or_global_value(config, center, "catboost.thread_count", "training.catboost.thread_count", 1)),
        auto_class_weights=str(_get_center_or_global_value(config, center, "catboost.auto_class_weights", "training.catboost.auto_class_weights", "Balanced")),
        l2_leaf_reg=float(_get_center_or_global_value(config, center, "catboost.l2_leaf_reg", "training.catboost.l2_leaf_reg", 5.0)),
        bootstrap_type="Bernoulli",
        subsample=float(_get_center_or_global_value(config, center, "catboost.subsample", "training.catboost.subsample", 0.9)),
    )


BUILDERS = {
    "linear_svm": _build_linear_svm,
    "pca_svm": _build_pca_svm,
    "logreg": _build_logreg,
    "pls_logreg": _build_pls_logreg,
    "rf": _build_rf,
    "extra_trees": _build_extra_trees,
    "catboost": _build_catboost,
}


def build_pls_grid_models(config, center: str) -> Dict[str, BaseEstimator]:
    grid = _get_center_or_global_value(config, center, "pls_logreg.n_components_grid", "training.pls_logreg.n_components_grid", None)
    if grid is None:
        fixed_n = int(_get_center_or_global_value(config, center, "pls_logreg.n_components", "training.pls_logreg.n_components", 8))
        grid = [fixed_n]

    models: Dict[str, BaseEstimator] = {}
    for n_components in grid:
        n_int = int(n_components)
        models[f"pls_logreg_nc{n_int}"] = _build_pls_logreg(config, center, n_components=n_int)
    return models


def make_models_for_center(config, center: str, selected_pls_n_components: int | None = None) -> Dict[str, BaseEstimator]:
    models: Dict[str, BaseEstimator] = {}

    for model_name in SUPPORTED_MODEL_NAMES:
        if not is_model_enabled_for_center(config, center, model_name, default=False):
            continue

        if model_name == "pls_logreg":
            models[model_name] = _build_pls_logreg(config, center, n_components=selected_pls_n_components)
        else:
            models[model_name] = BUILDERS[model_name](config, center)

    if not models:
        raise ValueError(f"No models enabled for center={center}")

    return models
