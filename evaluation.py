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
