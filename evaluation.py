from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold

from dataset import INV_CLASS_MAP

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None


LABELS = sorted(INV_CLASS_MAP.keys())
TARGET_NAMES = [INV_CLASS_MAP[i] for i in LABELS]


def _make_cv(strategy: str, n_splits: int, random_state: int = 42):
    strategy = (strategy or "group").lower()

    if strategy == "stratified_group" and StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return GroupKFold(n_splits=n_splits)


def evaluate_group_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    models: Dict[str, object],
    n_splits: int = 5,
    dataset_name: str = "samplewise",
    cv_strategy: str = "stratified_group",
    random_state: int = 42,
) -> Tuple[list[dict], pd.DataFrame]:
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Not enough unique groups for grouped CV")

    cv = _make_cv(cv_strategy, n_splits=n_splits, random_state=random_state)
    results = []

    for model_name, model in models.items():
        fold_reports = []
        y_true_all = []
        y_pred_all = []

        print(f"\n=== MODEL: {model_name} | DATASET: {dataset_name} | CV: {cv_strategy} ===")
        for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y, groups=groups), start=1):
            x_train, x_valid = x[train_idx], x[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            model.fit(x_train, y_train)
            pred = model.predict(x_valid)

            acc = accuracy_score(y_valid, pred)
            bacc = balanced_accuracy_score(y_valid, pred)
            macro_f1 = f1_score(y_valid, pred, labels=LABELS, average="macro", zero_division=0)

            fold_class_dist = {
                INV_CLASS_MAP[int(label)]: int((y_valid == label).sum()) for label in LABELS
            }
            print(
                f"Fold {fold}: acc={acc:.4f} | bacc={bacc:.4f} | macro_f1={macro_f1:.4f} | valid_classes={fold_class_dist}"
            )
            fold_reports.append(
                {
                    "fold": fold,
                    "acc": acc,
                    "bacc": bacc,
                    "macro_f1": macro_f1,
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "valid_class_distribution": fold_class_dist,
                }
            )
            y_true_all.extend(y_valid.tolist())
            y_pred_all.extend(pred.tolist())

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        oof_acc = accuracy_score(y_true_all, y_pred_all)
        oof_bacc = balanced_accuracy_score(y_true_all, y_pred_all)
        oof_macro_f1 = f1_score(
            y_true_all,
            y_pred_all,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        results.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "cv_strategy": cv_strategy,
                "acc_mean": float(np.mean([row["acc"] for row in fold_reports])),
                "acc_std": float(np.std([row["acc"] for row in fold_reports])),
                "bacc_mean": float(np.mean([row["bacc"] for row in fold_reports])),
                "bacc_std": float(np.std([row["bacc"] for row in fold_reports])),
                "macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_reports])),
                "macro_f1_std": float(np.std([row["macro_f1"] for row in fold_reports])),
                "oof_acc": float(oof_acc),
                "oof_bacc": float(oof_bacc),
                "oof_macro_f1": float(oof_macro_f1),
                "fold_reports": fold_reports,
                "confusion_matrix": confusion_matrix(
                    y_true_all,
                    y_pred_all,
                    labels=LABELS,
                ).tolist(),
                "classification_report": classification_report(
                    y_true_all,
                    y_pred_all,
                    labels=LABELS,
                    target_names=TARGET_NAMES,
                    output_dict=True,
                    zero_division=0,
                ),
            }
        )

from copy import deepcopy
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold

from dataset import INV_CLASS_MAP

try:
    from scipy.special import softmax
except Exception:  # pragma: no cover
    softmax = None

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None


LABELS = sorted(INV_CLASS_MAP.keys())
TARGET_NAMES = [INV_CLASS_MAP[i] for i in LABELS]


def _make_cv(strategy: str, n_splits: int, random_state: int = 42):
    strategy = (strategy or "group").lower()

    if strategy == "stratified_group" and StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return GroupKFold(n_splits=n_splits)


def _clone_model(model):
    try:
        return clone(model)
    except Exception:
        return deepcopy(model)


def _scores_to_proba(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])

    if softmax is not None:
        return softmax(scores, axis=1)

    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _predict_proba(model, x_valid: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_valid)
        proba = np.asarray(proba, dtype=float)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x_valid)
        proba = _scores_to_proba(scores)
    else:  # pragma: no cover
        pred = np.asarray(model.predict(x_valid))
        proba = np.zeros((len(pred), len(LABELS)), dtype=float)
        for row_idx, label in enumerate(pred):
            col_idx = LABELS.index(int(label))
            proba[row_idx, col_idx] = 1.0

    if proba.shape[1] != len(LABELS):
        fixed = np.zeros((proba.shape[0], len(LABELS)), dtype=float)
        model_classes = getattr(model, "classes_", LABELS)
        for model_col, cls in enumerate(model_classes):
            if int(cls) in LABELS:
                fixed[:, LABELS.index(int(cls))] = proba[:, model_col]
        proba = fixed

    row_sums = np.sum(proba, axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return proba / row_sums


def _metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
    }


def _build_result_record(
    dataset_name: str,
    model_name: str,
    cv_strategy: str,
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    fold_reports: list[dict],
) -> dict:
    oof_metrics = _metrics_from_predictions(y_true_all, y_pred_all)
    return {
        "dataset": dataset_name,
        "model": model_name,
        "cv_strategy": cv_strategy,
        "acc_mean": float(np.mean([row["acc"] for row in fold_reports])),
        "acc_std": float(np.std([row["acc"] for row in fold_reports])),
        "bacc_mean": float(np.mean([row["bacc"] for row in fold_reports])),
        "bacc_std": float(np.std([row["bacc"] for row in fold_reports])),
        "macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_reports])),
        "macro_f1_std": float(np.std([row["macro_f1"] for row in fold_reports])),
        "oof_acc": float(oof_metrics["acc"]),
        "oof_bacc": float(oof_metrics["bacc"]),
        "oof_macro_f1": float(oof_metrics["macro_f1"]),
        "fold_reports": fold_reports,
        "confusion_matrix": confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=LABELS,
        ).tolist(),
        "classification_report": classification_report(
            y_true_all,
            y_pred_all,
            labels=LABELS,
            target_names=TARGET_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def _build_fold_reports_from_oof(
    y: np.ndarray,
    y_pred_oof: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    fold_reports = []
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        y_valid = y[valid_idx]
        pred = y_pred_oof[valid_idx]
        metrics = _metrics_from_predictions(y_valid, pred)
        fold_class_dist = {
            INV_CLASS_MAP[int(label)]: int((y_valid == label).sum()) for label in LABELS
        }
        fold_reports.append(
            {
                "fold": fold,
                "acc": metrics["acc"],
                "bacc": metrics["bacc"],
                "macro_f1": metrics["macro_f1"],
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "valid_class_distribution": fold_class_dist,
            }
        )
    return fold_reports


def evaluate_group_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    models: Dict[str, object],
    n_splits: int = 5,
    dataset_name: str = "samplewise",
    cv_strategy: str = "stratified_group",
    random_state: int = 42,
    sample_ids: list[str] | None = None,
    ensemble_config: dict | None = None,
) -> Tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Not enough unique groups for grouped CV")

    cv = _make_cv(cv_strategy, n_splits=n_splits, random_state=random_state)
    splits = list(cv.split(x, y, groups=groups))
    results = []
    oof_store = {}
    fold_index = np.full(len(y), -1, dtype=int)

    for model_name, base_model in models.items():
        y_pred_oof = np.full(len(y), -1, dtype=int)
        y_proba_oof = np.zeros((len(y), len(LABELS)), dtype=float)
        fold_reports = []

        print(f"\n=== MODEL: {model_name} | DATASET: {dataset_name} | CV: {cv_strategy} ===")
        for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
            x_train, x_valid = x[train_idx], x[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            model = _clone_model(base_model)
            model.fit(x_train, y_train)
            proba = _predict_proba(model, x_valid)
            pred = np.asarray([LABELS[idx] for idx in np.argmax(proba, axis=1)])

            y_pred_oof[valid_idx] = pred
            y_proba_oof[valid_idx] = proba
            fold_index[valid_idx] = fold

            metrics = _metrics_from_predictions(y_valid, pred)
            fold_class_dist = {
                INV_CLASS_MAP[int(label)]: int((y_valid == label).sum()) for label in LABELS
            }
            print(
                f"Fold {fold}: acc={metrics['acc']:.4f} | bacc={metrics['bacc']:.4f} | "
                f"macro_f1={metrics['macro_f1']:.4f} | valid_classes={fold_class_dist}"
            )
            fold_reports.append(
                {
                    "fold": fold,
                    "acc": metrics["acc"],
                    "bacc": metrics["bacc"],
                    "macro_f1": metrics["macro_f1"],
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "valid_class_distribution": fold_class_dist,
                }
            )

        result_row = _build_result_record(
            dataset_name=dataset_name,
            model_name=model_name,
            cv_strategy=cv_strategy,
            y_true_all=y,
            y_pred_all=y_pred_oof,
            fold_reports=fold_reports,
        )
        results.append(result_row)
        oof_store[model_name] = {
            "pred": y_pred_oof,
            "proba": y_proba_oof,
        }

    if ensemble_config and bool(ensemble_config.get("enabled", True)):
        members = list(ensemble_config.get("members", []))
        if members:
            missing = [name for name in members if name not in oof_store]
            if missing:
                raise ValueError(f"Ensemble members are missing from trained models: {missing}")

            weights_cfg = ensemble_config.get("weights", {}) or {}
            raw_weights = np.array([float(weights_cfg.get(name, 1.0)) for name in members], dtype=float)
            if np.allclose(raw_weights.sum(), 0.0):
                raw_weights = np.ones_like(raw_weights)
            weights = raw_weights / raw_weights.sum()

            ensemble_proba = np.zeros((len(y), len(LABELS)), dtype=float)
            for weight, member_name in zip(weights, members):
                ensemble_proba += weight * oof_store[member_name]["proba"]

            ensemble_pred = np.asarray([LABELS[idx] for idx in np.argmax(ensemble_proba, axis=1)])
            ensemble_name = str(ensemble_config.get("name", "ensemble_soft"))
            ensemble_fold_reports = _build_fold_reports_from_oof(y, ensemble_pred, splits)

            print(
                f"\n=== ENSEMBLE: {ensemble_name} | MEMBERS: {members} | WEIGHTS: {weights.round(3).tolist()} ==="
            )
            for report in ensemble_fold_reports:
                print(
                    f"Fold {report['fold']}: acc={report['acc']:.4f} | bacc={report['bacc']:.4f} | "
                    f"macro_f1={report['macro_f1']:.4f}"
                )

            results.append(
                _build_result_record(
                    dataset_name=dataset_name,
                    model_name=ensemble_name,
                    cv_strategy=cv_strategy,
                    y_true_all=y,
                    y_pred_all=ensemble_pred,
                    fold_reports=ensemble_fold_reports,
                )
            )
            oof_store[ensemble_name] = {"pred": ensemble_pred, "proba": ensemble_proba}

    results_df = pd.DataFrame(
        [
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "cv_strategy": row["cv_strategy"],
                "acc_mean": row["acc_mean"],
                "acc_std": row["acc_std"],
                "bacc_mean": row["bacc_mean"],
                "bacc_std": row["bacc_std"],
                "macro_f1_mean": row["macro_f1_mean"],
                "macro_f1_std": row["macro_f1_std"],
                "oof_acc": row["oof_acc"],
                "oof_bacc": row["oof_bacc"],
                "oof_macro_f1": row["oof_macro_f1"],
            }
            for row in results
        ]
    )

    oof_df = pd.DataFrame(
        {
            "sample_id": sample_ids if sample_ids is not None else [f"sample_{idx}" for idx in range(len(y))],
            "group": groups,
            "fold": fold_index,
            "true_label": y,
            "true_label_name": [INV_CLASS_MAP[int(label)] for label in y],
        }
    )
    for model_name, payload in oof_store.items():
        oof_df[f"pred__{model_name}"] = payload["pred"]
        oof_df[f"pred_name__{model_name}"] = [INV_CLASS_MAP[int(label)] for label in payload["pred"]]
        for class_idx, class_label in enumerate(LABELS):
            class_name = INV_CLASS_MAP[class_label]
            oof_df[f"proba__{model_name}__{class_name}"] = payload["proba"][:, class_idx]

    return results, results_df, oof_df


def fit_rf_and_feature_importance(
    x: np.ndarray,
    y: np.ndarray,
    wave: np.ndarray,
    output_dir,
    top_k: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf.fit(x, y)
    importance = rf.feature_importances_

    feat_df = pd.DataFrame({"wave": wave, "importance": importance}).sort_values(
        "importance", ascending=False
    )

    plt.figure(figsize=(12, 5))
    plt.plot(wave, importance)
    plt.title("RandomForest feature importance over wave axis")
    plt.xlabel("Wave (cm^-1)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "rf_feature_importance.png", dpi=200)
    plt.close()

    feat_df.head(top_k).to_csv(output_dir / "top_rf_features.csv", index=False)
    return feat_df

    results_df = pd.DataFrame(
        [
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "cv_strategy": row["cv_strategy"],
                "acc_mean": row["acc_mean"],
                "acc_std": row["acc_std"],
                "bacc_mean": row["bacc_mean"],
                "bacc_std": row["bacc_std"],
                "macro_f1_mean": row["macro_f1_mean"],
                "macro_f1_std": row["macro_f1_std"],
                "oof_acc": row["oof_acc"],
                "oof_bacc": row["oof_bacc"],
                "oof_macro_f1": row["oof_macro_f1"],
            }
            for row in results
        ]
    )
    return results, results_df


def fit_rf_and_feature_importance(
    x: np.ndarray,
    y: np.ndarray,
    wave: np.ndarray,
    output_dir,
    top_k: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf.fit(x, y)
    importance = rf.feature_importances_

    feat_df = pd.DataFrame({"wave": wave, "importance": importance}).sort_values(
        "importance", ascending=False
    )

    plt.figure(figsize=(12, 5))
    plt.plot(wave, importance)
    plt.title("RandomForest feature importance over wave axis")
    plt.xlabel("Wave (cm^-1)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "rf_feature_importance.png", dpi=200)
    plt.close()

    feat_df.head(top_k).to_csv(output_dir / "top_rf_features.csv", index=False)
    return feat_df
