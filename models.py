from __future__ import annotations

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


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


def _get_value(config, path: str, default):
    current = config
    for part in path.split("."):
        if not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


class PLSLogRegClassifier(BaseEstimator, ClassifierMixin):
    """PLS feature extraction followed by multinomial logistic regression.

    This is more stable for multiclass spectral classification than piping raw
    PLSRegression directly into sklearn Pipeline, because we explicitly fit PLS
    on one-hot encoded targets and only expose X scores to LogisticRegression.
    """

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

        n_components = int(
            max(1, min(self.n_components, X_scaled.shape[0] - 1, X_scaled.shape[1]))
        )
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

    def predict(self, X):
        X_scores = self._transform(X)
        return self.logreg_.predict(X_scores)

    def predict_proba(self, X):
        X_scores = self._transform(X)
        return self.logreg_.predict_proba(X_scores)

    def _transform(self, X):
        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler_.transform(X)
        return self.pls_.transform(X_scaled)


class NamedCatBoostClassifier(CatBoostClassifier if CatBoostClassifier is not None else object):
    """Thin wrapper for safer cloning and cleaner repr in logs."""

    pass


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
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if _is_enabled(config, "pls_logreg", True):
        models["pls_logreg"] = PLSLogRegClassifier(
            n_components=int(_get_value(config, "training.pls_logreg.n_components", 8)),
            C=float(_get_value(config, "training.pls_logreg.C", 1.0)),
            class_weight=str(_get_value(config, "training.pls_logreg.class_weight", "balanced")),
            max_iter=int(_get_value(config, "training.pls_logreg.max_iter", 5000)),
            random_state=random_state,
        )

    if _is_enabled(config, "rf", False):
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
            n_estimators=int(_get_value(config, "training.extra_trees.n_estimators", 1200)),
            max_depth=_get_value(config, "training.extra_trees.max_depth", None),
            min_samples_split=int(_get_value(config, "training.extra_trees.min_samples_split", 4)),
            min_samples_leaf=int(_get_value(config, "training.extra_trees.min_samples_leaf", 2)),
            max_features=str(_get_value(config, "training.extra_trees.max_features", "sqrt")),
            class_weight=str(
                _get_value(config, "training.extra_trees.class_weight", "balanced_subsample")
            ),
            random_state=random_state,
            n_jobs=-1,
        )

    if _is_enabled(config, "catboost", True):
        if CatBoostClassifier is None:
            raise ImportError(
                "catboost is not installed, but config.training.models.catboost=true"
            )

        models["catboost"] = NamedCatBoostClassifier(
            iterations=int(_get_value(config, "training.catboost.iterations", 700)),
            depth=int(_get_value(config, "training.catboost.depth", 6)),
            learning_rate=float(_get_value(config, "training.catboost.learning_rate", 0.03)),
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=int(_get_value(config, "training.catboost.thread_count", 1)),
            auto_class_weights=str(_get_value(config, "training.catboost.auto_class_weights", "Balanced")),
            l2_leaf_reg=float(_get_value(config, "training.catboost.l2_leaf_reg", 5.0)),
            bootstrap_type="Bernoulli",
            subsample=float(_get_value(config, "training.catboost.subsample", 0.9)),
        )

    if not models:
        raise ValueError("No models enabled in config.training.models")

    return models
