from __future__ import annotations

import itertools
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from dataset import INV_CLASS_MAP
from evaluation import (
    LABELS,
    _build_result_record,
    _clone_model,
    _make_cv,
    _metrics_from_predictions,
    _predict_proba,
    evaluate_group_cv,
)
from models import get_center_training_cfg, make_single_model_for_center
from utils import save_json


def _get_nested(node, path: str, default=None):
    current = node
    for part in path.split("."):
        if current is None or not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


def _normalize_search_space(node) -> dict[str, list]:
    if node is None:
        return {}

    normalized = {}
    for param_name, spec in vars(node).items():
        normalized[str(param_name)] = list(spec) if isinstance(spec, list) else [spec]
    return normalized


def _get_search_space(config, center: str, model_name: str) -> dict[str, list]:
    center_cfg = get_center_training_cfg(config, center)
    nested_search = _get_nested(center_cfg, "nested_search", None)
    return _normalize_search_space(_get_nested(nested_search, model_name, None))


def _get_base_params(config, center: str, model_name: str) -> dict:
    center_cfg = get_center_training_cfg(config, center)
    model_cfg = getattr(center_cfg, model_name, None) if center_cfg is not None else None
    return {str(k): v for k, v in vars(model_cfg).items()} if model_cfg is not None else {}


def _candidate_param_grid(config, center: str, model_name: str) -> list[dict]:
    base_params = _get_base_params(config, center, model_name)
    search_space = _get_search_space(config, center, model_name)

    if not search_space:
        return [base_params]

    keys = list(search_space.keys())
    value_lists = [list(search_space[key]) for key in keys]

    candidates = []
    seen = set()

    def register(params: dict):
        key = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
        if key not in seen:
            seen.add(key)
            candidates.append(params)

    register(deepcopy(base_params))

    for combo in itertools.product(*value_lists):
        params = deepcopy(base_params)
        for key, value in zip(keys, combo):
            params[key] = value
        register(params)

    return candidates


def _sub_sample_ids(sample_ids: list[str] | None, indices: np.ndarray) -> list[str] | None:
    if sample_ids is None:
        return None
    arr = np.asarray(sample_ids)
    return arr[indices].tolist()


def _extract_model_proba(oof_df: pd.DataFrame, model_name: str) -> np.ndarray:
    return np.column_stack(
        [oof_df[f"proba__{model_name}__{INV_CLASS_MAP[label]}"].to_numpy(dtype=float) for label in LABELS]
    )


def _score_proba(y_true: np.ndarray, proba: np.ndarray) -> float:
    pred = np.asarray([LABELS[idx] for idx in np.argmax(proba, axis=1)], dtype=int)
    return float(f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0))


def _build_weight_candidates(members: list[str], ensemble_config: dict | None) -> list[dict[str, float]]:
    if not members:
        return []

    explicit = []
    if ensemble_config is not None:
        raw = ensemble_config.get("weight_candidates", None)
        if raw:
            for item in raw:
                explicit.append({name: float(item.get(name, 0.0)) for name in members})

    if explicit:
        candidates = explicit
    else:
        patterns = [
            [1, 1, 1],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [3, 2, 1], [3, 1, 2], [2, 3, 1], [2, 1, 3], [1, 3, 2], [1, 2, 3],
            [3, 1, 1], [1, 3, 1], [1, 1, 3],
            [2, 2, 1], [2, 1, 2], [1, 2, 2],
        ]
        candidates = []
        for pattern in patterns:
            if len(pattern) < len(members):
                continue
            candidates.append(
                {name: float(weight) for name, weight in zip(members, pattern[:len(members)])}
            )

    normalized = []
    seen = set()

    for candidate in candidates:
        weights = np.asarray([max(candidate.get(name, 0.0), 0.0) for name in members], dtype=float)
        if np.allclose(weights.sum(), 0.0):
            continue
        weights /= weights.sum()

        normalized_map = {name: float(weight) for name, weight in zip(members, weights)}
        key = tuple(round(normalized_map[name], 8) for name in members)

        if key not in seen:
            seen.add(key)
            normalized.append(normalized_map)

    return normalized


def _pick_best_weight_map(
    y_true: np.ndarray,
    member_probas: dict[str, np.ndarray],
    members: list[str],
    ensemble_config: dict | None,
):
    candidates = _build_weight_candidates(members, ensemble_config)

    best_weights = None
    best_score = -1.0

    for weight_map in candidates:
        proba = np.zeros_like(next(iter(member_probas.values())))
        for name in members:
            proba += float(weight_map.get(name, 0.0)) * member_probas[name]

        score = _score_proba(y_true, proba)
        if score > best_score:
            best_score = score
            best_weights = weight_map

    return best_weights, float(best_score)


def _tune_single_model(
    config,
    center: str,
    model_name: str,
    x_train,
    y_train,
    groups_train,
    sample_ids_train,
    inner_splits: int,
    cv_strategy: str,
    random_state: int,
):
    candidates = _candidate_param_grid(config, center, model_name)
    tuning_rows = []
    best = None

    for candidate_idx, params in enumerate(candidates, start=1):
        selected_pls_n = params.get("n_components") if model_name == "pls_logreg" else None

        model = make_single_model_for_center(
            config,
            center=center,
            model_name=model_name,
            selected_pls_n_components=selected_pls_n,
            overrides=params,
        )

        _, results_df, oof_df = evaluate_group_cv(
            x_train,
            y_train,
            groups_train,
            {model_name: model},
            n_splits=inner_splits,
            dataset_name=f"inner_tune_{model_name}_center{center}",
            cv_strategy=cv_strategy,
            random_state=random_state,
            sample_ids=sample_ids_train,
            ensemble_config=None,
            verbose=False,
        )

        row = results_df.iloc[0].to_dict()
        tuning_rows.append({"candidate_idx": candidate_idx, "model": model_name, **params, **row})

        score = float(row["oof_macro_f1"])
        if best is None or score > best["score"]:
            best = {
                "params": deepcopy(params),
                "score": score,
                "oof_df": oof_df,
            }

    tuning_df = pd.DataFrame(tuning_rows).sort_values("oof_macro_f1", ascending=False).reset_index(drop=True)
    return best, tuning_df


def run_nested_group_cv_for_center(
    config,
    center: str,
    x,
    y,
    groups,
    sample_ids,
    center_output_dir,
    available_model_names,
    ensemble_config,
):
    center_output_dir = Path(center_output_dir)

    nested_cfg = getattr(config.training, "nested_cv", None)
    outer_splits = int(getattr(nested_cfg, "outer_splits", getattr(config.training, "group_kfold_splits", 3)))
    inner_splits = int(getattr(nested_cfg, "inner_splits", 2))
    cv_strategy = str(getattr(config.training, "cv_strategy", "stratified_group"))
    random_state = int(getattr(nested_cfg, "random_state", config.project.seed))

    outer_n_splits = min(outer_splits, len(np.unique(groups)))
    if outer_n_splits < 2:
        raise ValueError(f"Not enough groups for nested CV in center={center}")

    outer_cv = _make_cv(cv_strategy, n_splits=outer_n_splits, random_state=random_state)
    outer_splits_list = list(outer_cv.split(x, y, groups=groups))

    outer_oof = {}
    outer_fold_index = np.full(len(y), -1, dtype=int)

    tuning_payload = {
        "center": str(center),
        "outer_splits": int(outer_n_splits),
        "inner_splits": int(inner_splits),
        "cv_strategy": cv_strategy,
        "folds": [],
    }

    for model_name in available_model_names:
        outer_oof[model_name] = {
            "pred": np.full(len(y), -1, dtype=int),
            "proba": np.zeros((len(y), len(LABELS)), dtype=float),
            "fold_reports": [],
            "selected_params_by_fold": [],
        }

    ensemble_name = None
    if ensemble_config and ensemble_config.get("enabled", True):
        ensemble_name = str(ensemble_config.get("name", f"ensemble_center{center}"))
        outer_oof[ensemble_name] = {
            "pred": np.full(len(y), -1, dtype=int),
            "proba": np.zeros((len(y), len(LABELS)), dtype=float),
            "fold_reports": [],
            "selected_weights_by_fold": [],
        }

    for outer_fold, (train_idx, valid_idx) in enumerate(outer_splits_list, start=1):
        x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        groups_train, groups_valid = groups[train_idx], groups[valid_idx]
        sample_ids_train = _sub_sample_ids(sample_ids, train_idx)

        valid_groups_sorted = sorted(pd.Series(groups_valid).astype(str).unique().tolist())

        fold_payload = {
            "outer_fold": outer_fold,
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
            "valid_groups": valid_groups_sorted,
            "models": {},
        }

        tuned_valid_probas = {}
        inner_best_probas = {}

        inner_n_splits = min(inner_splits, len(np.unique(groups_train)))
        if inner_n_splits < 2:
            raise ValueError(f"Not enough training groups for inner CV in center={center}, fold={outer_fold}")

        for model_name in available_model_names:
            best, tuning_df = _tune_single_model(
                config=config,
                center=center,
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                groups_train=groups_train,
                sample_ids_train=sample_ids_train,
                inner_splits=inner_n_splits,
                cv_strategy=cv_strategy,
                random_state=random_state + outer_fold * 17,
            )

            tuning_csv = center_output_dir / f"nested_tuning_{model_name}_center{center}_fold{outer_fold}.csv"
            tuning_df.to_csv(tuning_csv, index=False)

            selected_params = deepcopy(best["params"])
            selected_pls_n = selected_params.get("n_components") if model_name == "pls_logreg" else None

            tuned_model = make_single_model_for_center(
                config,
                center=center,
                model_name=model_name,
                selected_pls_n_components=selected_pls_n,
                overrides=selected_params,
            )
            tuned_model = _clone_model(tuned_model)
            tuned_model.fit(x_train, y_train)

            valid_proba = _predict_proba(tuned_model, x_valid)
            valid_pred = np.asarray([LABELS[idx] for idx in np.argmax(valid_proba, axis=1)], dtype=int)

            outer_oof[model_name]["pred"][valid_idx] = valid_pred
            outer_oof[model_name]["proba"][valid_idx] = valid_proba
            outer_fold_index[valid_idx] = outer_fold

            metrics = _metrics_from_predictions(y_valid, valid_pred)

            outer_oof[model_name]["fold_reports"].append(
                {
                    "fold": outer_fold,
                    "acc": metrics["acc"],
                    "bacc": metrics["bacc"],
                    "macro_f1": metrics["macro_f1"],
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "valid_class_distribution": {
                        INV_CLASS_MAP[int(label)]: int((y_valid == label).sum()) for label in LABELS
                    },
                    "selected_params": deepcopy(selected_params),
                    "inner_best_oof_macro_f1": float(best["score"]),
                }
            )
            outer_oof[model_name]["selected_params_by_fold"].append({"fold": outer_fold, **deepcopy(selected_params)})

            inner_best_probas[model_name] = _extract_model_proba(best["oof_df"], model_name)
            tuned_valid_probas[model_name] = valid_proba

            fold_payload["models"][model_name] = {
                "selected_params": deepcopy(selected_params),
                "inner_best_oof_macro_f1": float(best["score"]),
                "outer_valid_macro_f1": float(metrics["macro_f1"]),
                "tuning_csv": tuning_csv.name,
            }

        if ensemble_name is not None:
            members = [name for name in ensemble_config.get("members", []) if name in available_model_names]

            weight_map, inner_score = _pick_best_weight_map(
                y_train,
                {name: inner_best_probas[name] for name in members},
                members,
                ensemble_config,
            )

            ensemble_valid_proba = np.zeros((len(valid_idx), len(LABELS)), dtype=float)
            for member_name in members:
                ensemble_valid_proba += float(weight_map.get(member_name, 0.0)) * tuned_valid_probas[member_name]

            ensemble_valid_pred = np.asarray([LABELS[idx] for idx in np.argmax(ensemble_valid_proba, axis=1)], dtype=int)
            metrics = _metrics_from_predictions(y_valid, ensemble_valid_pred)

            outer_oof[ensemble_name]["pred"][valid_idx] = ensemble_valid_pred
            outer_oof[ensemble_name]["proba"][valid_idx] = ensemble_valid_proba
            outer_oof[ensemble_name]["fold_reports"].append(
                {
                    "fold": outer_fold,
                    "acc": metrics["acc"],
                    "bacc": metrics["bacc"],
                    "macro_f1": metrics["macro_f1"],
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "valid_class_distribution": {
                        INV_CLASS_MAP[int(label)]: int((y_valid == label).sum()) for label in LABELS
                    },
                    "selected_weights": deepcopy(weight_map),
                    "inner_best_oof_macro_f1": float(inner_score),
                    "ensemble_members": members,
                }
            )
            outer_oof[ensemble_name]["selected_weights_by_fold"].append({"fold": outer_fold, **deepcopy(weight_map)})

            fold_payload["ensemble"] = {
                "members": members,
                "selected_weights": deepcopy(weight_map),
                "inner_best_oof_macro_f1": float(inner_score),
                "outer_valid_macro_f1": float(metrics["macro_f1"]),
            }

        tuning_payload["folds"].append(fold_payload)

    results = []
    for model_name, payload in outer_oof.items():
        extra_fields = {"selected_params_by_fold": payload.get("selected_params_by_fold", [])}
        if model_name == ensemble_name:
            extra_fields = {
                "ensemble_members": [name for name in ensemble_config.get("members", []) if name in available_model_names],
                "ensemble_weighting": "inner_cv_selected",
                "selected_weights_by_fold": payload.get("selected_weights_by_fold", []),
            }

        results.append(
            _build_result_record(
                dataset_name=f"nested_samplewise_center{center}",
                model_name=model_name,
                cv_strategy=f"nested_{cv_strategy}",
                y_true_all=y,
                y_pred_all=payload["pred"],
                fold_reports=payload["fold_reports"],
                extra_fields=extra_fields,
            )
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
                "ensemble_members": ",".join(row.get("ensemble_members", [])) if row.get("ensemble_members") else "",
            }
            for row in results
        ]
    ).sort_values("oof_macro_f1", ascending=False).reset_index(drop=True)

    oof_df = pd.DataFrame(
        {
            "sample_id": sample_ids if sample_ids is not None else [f"sample_{idx}" for idx in range(len(y))],
            "group": groups,
            "fold": outer_fold_index,
            "true_label": y,
            "true_label_name": [INV_CLASS_MAP[int(label)] for label in y],
        }
    )

    for model_name, payload in outer_oof.items():
        oof_df[f"pred__{model_name}"] = payload["pred"]
        oof_df[f"pred_name__{model_name}"] = [INV_CLASS_MAP[int(label)] for label in payload["pred"]]
        for class_idx, class_label in enumerate(LABELS):
            class_name = INV_CLASS_MAP[class_label]
            oof_df[f"proba__{model_name}__{class_name}"] = payload["proba"][:, class_idx]

    save_json(tuning_payload, center_output_dir / f"nested_cv_tuning_center{center}.json")
    return results, results_df, oof_df, tuning_payload