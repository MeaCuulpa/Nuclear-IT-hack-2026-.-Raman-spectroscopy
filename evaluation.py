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


def evaluate_group_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    models: Dict[str, object],
    n_splits: int = 5,
    dataset_name: str = "samplewise",
) -> Tuple[list[dict], pd.DataFrame]:
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Not enough unique groups for GroupKFold")

    gkf = GroupKFold(n_splits=n_splits)
    results = []

    for model_name, model in models.items():
        fold_reports = []
        y_true_all = []
        y_pred_all = []

        print(f"\n=== MODEL: {model_name} | DATASET: {dataset_name} ===")
        for fold, (train_idx, valid_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
            x_train, x_valid = x[train_idx], x[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            model.fit(x_train, y_train)
            pred = model.predict(x_valid)

            acc = accuracy_score(y_valid, pred)
            bacc = balanced_accuracy_score(y_valid, pred)
            macro_f1 = f1_score(y_valid, pred, average="macro")

            print(f"Fold {fold}: acc={acc:.4f} | bacc={bacc:.4f} | macro_f1={macro_f1:.4f}")
            fold_reports.append({"fold": fold, "acc": acc, "bacc": bacc, "macro_f1": macro_f1})
            y_true_all.extend(y_valid.tolist())
            y_pred_all.extend(pred.tolist())

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        results.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "acc_mean": float(np.mean([row["acc"] for row in fold_reports])),
                "acc_std": float(np.std([row["acc"] for row in fold_reports])),
                "bacc_mean": float(np.mean([row["bacc"] for row in fold_reports])),
                "bacc_std": float(np.std([row["bacc"] for row in fold_reports])),
                "macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_reports])),
                "macro_f1_std": float(np.std([row["macro_f1"] for row in fold_reports])),
                "confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
                "classification_report": classification_report(
                    y_true_all,
                    y_pred_all,
                    target_names=[INV_CLASS_MAP[i] for i in sorted(INV_CLASS_MAP)],
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
                "acc_mean": row["acc_mean"],
                "acc_std": row["acc_std"],
                "bacc_mean": row["bacc_mean"],
                "bacc_std": row["bacc_std"],
                "macro_f1_mean": row["macro_f1_mean"],
                "macro_f1_std": row["macro_f1_std"],
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
