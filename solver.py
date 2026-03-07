from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from dataset import INV_CLASS_MAP, RamanDataset
from evaluation import LABELS, TARGET_NAMES, evaluate_group_cv
from models import build_pls_grid_models, get_center_training_cfg, is_model_enabled_for_center, make_models_for_center
from preprocessing import PreprocessingConfig, SpectraPreprocessor
from utils import ensure_dir, save_json, set_seed


class Solver:
    def __init__(self, config):
        self.config = config
        set_seed(int(self.config.project.seed))
        self.output_dir = ensure_dir(self.config.paths.outputs)
        self.dataset = RamanDataset(self.config.paths.data_root)

    def _get_nested(self, node, path: str, default=None):
        current = node
        for part in path.split("."):
            if current is None or not hasattr(current, part):
                return default
            current = getattr(current, part)
        return current

    def _get_center_crop(self, center: str) -> list[float]:
        crop_ranges_by_center = getattr(self.config.preprocessing, "crop_ranges_by_center", None)
        default_crop = list(getattr(self.config.preprocessing, "crop_range", [700, 1800]))
        if crop_ranges_by_center is None:
            return default_crop
        return list(getattr(crop_ranges_by_center, str(center), default_crop))

    def _make_preprocessor_for_center(self, center: str) -> SpectraPreprocessor:
        center_crop = self._get_center_crop(center)
        return SpectraPreprocessor(
            PreprocessingConfig(
                enabled=bool(self.config.preprocessing.enabled),
                crop_min=float(center_crop[0]),
                crop_max=float(center_crop[1]),
                despike_threshold=float(self.config.preprocessing.despike_threshold),
                despike_window=int(self.config.preprocessing.despike_window),
                smooth_window=int(self.config.preprocessing.smooth_window),
                smooth_polyorder=int(self.config.preprocessing.smooth_polyorder),
                baseline_lam=float(self.config.preprocessing.baseline_lam),
                baseline_p=float(self.config.preprocessing.baseline_p),
                baseline_niter=int(self.config.preprocessing.baseline_niter),
                normalization=str(self.config.preprocessing.normalization),
            )
        )

    def _get_center_feature_config(self, center: str) -> dict:
        center_cfg = get_center_training_cfg(self.config, center)
        feature_cfg = getattr(center_cfg, "feature_config", None) if center_cfg is not None else None

        global_agg = [str(value) for value in getattr(self.config.data, "sample_agg_features", ["median"])]
        return {
            "agg_features": [str(value) for value in getattr(feature_cfg, "sample_agg_features", global_agg)] if feature_cfg is not None else global_agg,
            "include_region_feature": bool(getattr(feature_cfg, "include_region_feature", getattr(self.config.data, "include_region_feature", False))) if feature_cfg is not None else bool(getattr(self.config.data, "include_region_feature", False)),
            "append_first_derivative": bool(getattr(feature_cfg, "append_first_derivative", getattr(self.config.data, "append_first_derivative", False))) if feature_cfg is not None else bool(getattr(self.config.data, "append_first_derivative", False)),
        }

    def _build_samplewise_cache_tag(self, center: str, feature_cfg: dict) -> str:
        payload = {
            "center": str(center),
            "agg_features": [str(item) for item in feature_cfg["agg_features"]],
            "include_region_feature": bool(feature_cfg["include_region_feature"]),
            "append_first_derivative": bool(feature_cfg["append_first_derivative"]),
            "preprocessing_enabled": bool(self.config.preprocessing.enabled),
            "crop_range_for_this_center": list(self._get_center_crop(center)),
            "despike_threshold": float(self.config.preprocessing.despike_threshold),
            "despike_window": int(self.config.preprocessing.despike_window),
            "smooth_window": int(self.config.preprocessing.smooth_window),
            "smooth_polyorder": int(self.config.preprocessing.smooth_polyorder),
            "baseline_lam": float(self.config.preprocessing.baseline_lam),
            "baseline_p": float(self.config.preprocessing.baseline_p),
            "baseline_niter": int(self.config.preprocessing.baseline_niter),
            "normalization": str(self.config.preprocessing.normalization),
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(raw).hexdigest()[:12]

    def _build_center_ensemble_config(self, center: str, available_model_names: list[str]) -> dict | None:
        center_cfg = get_center_training_cfg(self.config, center)
        if center_cfg is None:
            return None

        ensemble_cfg = getattr(center_cfg, "ensemble", None)
        if ensemble_cfg is None or not bool(getattr(ensemble_cfg, "enabled", False)):
            return None

        default_members = [name for name in ["extra_trees", "catboost", "pls_logreg"] if name in available_model_names]
        members = [str(value) for value in getattr(ensemble_cfg, "members", default_members)]
        members = [name for name in members if name in available_model_names]
        if not members:
            return None

        weights_node = getattr(ensemble_cfg, "weights", None)
        weights = {}
        if weights_node is not None:
            for name in members:
                if hasattr(weights_node, name):
                    weights[name] = float(getattr(weights_node, name))

        return {
            "enabled": True,
            "name": str(getattr(ensemble_cfg, "name", f"ensemble_center{center}")),
            "members": members,
            "weights": weights,
            "weighting": str(getattr(ensemble_cfg, "weighting", "oof_macro_f1")),
        }

    def _select_best_pls_n_components(
        self,
        center: str,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        file_ids: list[str],
        center_output_dir: Path,
    ) -> tuple[int | None, dict | None]:
        if not is_model_enabled_for_center(self.config, center, "pls_logreg", default=False):
            return None, None

        candidate_models = build_pls_grid_models(self.config, center)
        if not candidate_models:
            return None, None

        if len(candidate_models) == 1:
            only_name = next(iter(candidate_models))
            match = re.search(r"nc(\d+)$", only_name)
            selected = int(match.group(1)) if match else None
            return selected, {"selected_n_components": selected, "grid_results": []}

        print(f"[STEP] Running PLS component sweep for center={center}...")
        pls_results, pls_df, _ = evaluate_group_cv(
            x,
            y,
            groups,
            candidate_models,
            n_splits=int(getattr(self.config.training, "group_kfold_splits", 3)),
            dataset_name=f"pls_grid_center{center}",
            cv_strategy=str(getattr(self.config.training, "cv_strategy", "stratified_group")),
            random_state=int(self.config.project.seed),
            sample_ids=file_ids,
            ensemble_config=None,
        )
        pls_df = pls_df.sort_values("oof_macro_f1", ascending=False).reset_index(drop=True)
        pls_df.to_csv(center_output_dir / f"pls_grid_search_center{center}.csv", index=False)
        save_json(pls_results, center_output_dir / f"pls_grid_search_center{center}.json")

        best_name = str(pls_df.iloc[0]["model"])
        match = re.search(r"nc(\d+)$", best_name)
        if match is None:
            raise ValueError(f"Could not parse n_components from model name: {best_name}")

        selected_n = int(match.group(1))
        print(f"[INFO] Best PLS n_components for center={center}: {selected_n}")
        return selected_n, {
            "selected_n_components": selected_n,
            "grid_results": pls_results,
        }

    def _build_fusion_key(self, sample_id: str) -> str:
        key = str(sample_id)
        key = re.sub(r"center\d+", "center", key, flags=re.IGNORECASE)
        key = re.sub(r"_+", "_", key)
        return key

    def _late_fusion(self, center_payloads: dict[str, dict]) -> dict | None:
        late_cfg = getattr(self.config.training, "late_fusion", None)
        if late_cfg is None or not bool(getattr(late_cfg, "enabled", False)):
            return None

        requested_centers = [str(value) for value in getattr(late_cfg, "centers", list(center_payloads.keys()))]
        use_centers = [center for center in requested_centers if center in center_payloads]
        if len(use_centers) < 2:
            print("[WARN] Late fusion skipped: need at least 2 centers with predictions")
            return None

        weighting = str(getattr(late_cfg, "weighting", "oof_macro_f1"))
        fusion_name = str(getattr(late_cfg, "name", "late_fusion"))

        merge_df = None
        center_scores = {}
        for center in use_centers:
            payload = center_payloads[center]
            oof_df = payload["oof_df"].copy()
            fusion_model_name = str(payload["fusion_model_name"])
            score = float(payload["fusion_score"])
            center_scores[center] = score

            base_cols = ["fusion_key", "true_label", "true_label_name", "group"]
            center_df = oof_df[base_cols].copy()
            for class_label in LABELS:
                class_name = INV_CLASS_MAP[class_label]
                center_df[f"proba__{center}__{class_name}"] = oof_df[
                    f"proba__{fusion_model_name}__{class_name}"
                ]

            if merge_df is None:
                merge_df = center_df
            else:
                merge_df = merge_df.merge(center_df, on=["fusion_key", "true_label", "true_label_name", "group"], how="inner")

        if merge_df is None or merge_df.empty:
            print("[WARN] Late fusion skipped: no overlapping samples between centers")
            return None

        raw_weights = np.array([max(center_scores[center], 1e-8) for center in use_centers], dtype=float) if weighting == "oof_macro_f1" else np.ones(len(use_centers), dtype=float)
        weights = raw_weights / raw_weights.sum()
        weights_map = {center: float(weight) for center, weight in zip(use_centers, weights)}

        final_proba = np.zeros((len(merge_df), len(LABELS)), dtype=float)
        for weight, center in zip(weights, use_centers):
            center_proba = np.column_stack(
                [merge_df[f"proba__{center}__{INV_CLASS_MAP[label]}"] for label in LABELS]
            )
            final_proba += weight * center_proba

        y_true = merge_df["true_label"].to_numpy(dtype=int)
        y_pred = np.asarray([LABELS[idx] for idx in np.argmax(final_proba, axis=1)], dtype=int)
        metrics = {
            "acc": float(accuracy_score(y_true, y_pred)),
            "bacc": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "weights": weights_map,
            "centers": use_centers,
            "n_samples": int(len(merge_df)),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=LABELS,
                target_names=TARGET_NAMES,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        }

        fusion_df = merge_df[["fusion_key", "group", "true_label", "true_label_name"]].copy()
        fusion_df["pred__late_fusion"] = y_pred
        fusion_df["pred_name__late_fusion"] = [INV_CLASS_MAP[int(label)] for label in y_pred]
        for class_idx, class_label in enumerate(LABELS):
            class_name = INV_CLASS_MAP[class_label]
            fusion_df[f"proba__late_fusion__{class_name}"] = final_proba[:, class_idx]

        return {"name": fusion_name, "metrics": metrics, "oof_df": fusion_df}

    def run(self) -> None:
        use_raw_cache = bool(getattr(self.config.data, "use_raw_cache", True))
        force_reload_raw_cache = bool(getattr(self.config.data, "force_reload_raw_cache", False))
        use_aligned_cache = bool(getattr(self.config.data, "use_aligned_cache", True))
        force_rebuild_aligned_cache = bool(getattr(self.config.data, "force_rebuild_aligned_cache", False))
        use_samplewise_cache = bool(getattr(self.config.data, "use_samplewise_cache", True))
        force_rebuild_samplewise_cache = bool(getattr(self.config.data, "force_rebuild_samplewise_cache", False))
        centers_to_run = [str(value) for value in getattr(self.config.data, "centers_to_run", [1500, 2900])]
        cv_strategy = str(getattr(self.config.training, "cv_strategy", "stratified_group"))

        print("[STEP] Loading raw dataset...")
        metadata_df, raw_data = self.dataset.load(
            use_cache=use_raw_cache,
            force_reload=force_reload_raw_cache,
        )
        metadata_df.to_csv(self.output_dir / "metadata_summary.csv", index=False)

        print("\n=== CENTER DISTRIBUTION ===")
        print(metadata_df["center"].value_counts().sort_index())

        print("[STEP] Aligning wave axes inside each center...")
        raw_data = self.dataset.prepare_raw_data(
            raw_data,
            use_cache=use_aligned_cache,
            force_rebuild=force_rebuild_aligned_cache,
        )

        final_report = {
            "n_files": int(len(metadata_df)),
            "n_unique_mice": int(metadata_df["mouse_id"].nunique()),
            "class_distribution": metadata_df["label_name"].value_counts().to_dict(),
            "center_distribution": metadata_df["center"].value_counts().to_dict(),
            "cv_strategy": cv_strategy,
            "results_by_center": {},
        }
        center_payloads_for_fusion = {}

        for center in centers_to_run:
            center_df = metadata_df[metadata_df["center"].astype(str) == center].copy()
            if center_df.empty:
                print(f"[WARN] No files found for center={center}")
                continue

            center_output_dir = ensure_dir(self.output_dir / f"center_{center}")
            feature_cfg = self._get_center_feature_config(center)

            print(f"\n{'=' * 20} CENTER {center} {'=' * 20}")
            print(f"[INFO] Files in center {center}: {len(center_df)}")
            print("[INFO] Class distribution:")
            print(center_df["label_name"].value_counts())
            print(f"[INFO] Feature config for center {center}: {feature_cfg}")

            print(f"[STEP] Building samplewise dataset for center={center}...")
            center_preprocessor = self._make_preprocessor_for_center(center)
            wave_sw, x_sw, y_sw, groups_sw, file_ids_sw = self.dataset.build_samplewise_dataset(
                center_preprocessor,
                agg_methods=feature_cfg["agg_features"],
                use_processed=bool(self.config.preprocessing.enabled),
                use_cache=use_samplewise_cache,
                force_rebuild=force_rebuild_samplewise_cache,
                cache_tag=self._build_samplewise_cache_tag(center=center, feature_cfg=feature_cfg),
                center_filter=center,
                include_region_feature=feature_cfg["include_region_feature"],
                append_first_derivative=feature_cfg["append_first_derivative"],
            )

            print(f"[INFO] Samplewise X shape for center {center}: {x_sw.shape}")
            print(f"[INFO] Samplewise y shape for center {center}: {y_sw.shape}")
            print(f"[INFO] Unique groups for center {center}: {len(set(groups_sw))}")
            print(f"[INFO] Samplewise files for center {center}: {len(file_ids_sw)}")

            selected_pls_n, pls_grid_payload = self._select_best_pls_n_components(
                center=center,
                x=x_sw,
                y=y_sw,
                groups=groups_sw,
                file_ids=file_ids_sw,
                center_output_dir=center_output_dir,
            )

            print(f"[STEP] Creating center-specific models for center={center}...")
            models = make_models_for_center(self.config, center=center, selected_pls_n_components=selected_pls_n)
            print(f"[INFO] Enabled models for center {center}: {list(models.keys())}")

            ensemble_config = self._build_center_ensemble_config(center, list(models.keys()))
            if ensemble_config is not None:
                print(f"[INFO] Ensemble config for center {center}: {ensemble_config}")

            print(f"[STEP] Running samplewise CV for center={center}...")
            sw_results, sw_results_df, sw_oof_df = evaluate_group_cv(
                x_sw,
                y_sw,
                groups_sw,
                models,
                n_splits=int(getattr(self.config.training, "group_kfold_splits", 3)),
                dataset_name=f"samplewise_center{center}",
                cv_strategy=cv_strategy,
                random_state=int(self.config.project.seed),
                sample_ids=file_ids_sw,
                ensemble_config=ensemble_config,
            )
            sw_results_df = sw_results_df.sort_values("oof_macro_f1", ascending=False).reset_index(drop=True)
            sw_oof_df["fusion_key"] = sw_oof_df["sample_id"].map(self._build_fusion_key)

            print(f"\n=== SAMPLEWISE RESULTS | CENTER {center} ===")
            print(sw_results_df)

            sw_results_df.to_csv(center_output_dir / f"cv_results_samplewise_center{center}.csv", index=False)
            sw_oof_df.to_csv(center_output_dir / f"oof_predictions_samplewise_center{center}.csv", index=False)
            save_json(sw_results, center_output_dir / f"samplewise_results_center{center}.json")

            fusion_model_name = ensemble_config["name"] if ensemble_config is not None else str(sw_results_df.iloc[0]["model"])
            fusion_score = None
            for row in sw_results:
                if row["model"] == fusion_model_name:
                    fusion_score = float(row["oof_macro_f1"])
                    break
            if fusion_score is None:
                fusion_score = float(sw_results_df.iloc[0]["oof_macro_f1"])

            center_payloads_for_fusion[center] = {
                "oof_df": sw_oof_df,
                "fusion_model_name": fusion_model_name,
                "fusion_score": fusion_score,
            }

            best_row = sw_results_df.iloc[0].to_dict()
            final_report["results_by_center"][f"center_{center}"] = {
                "n_files": int(len(center_df)),
                "class_distribution": center_df["label_name"].value_counts().to_dict(),
                "feature_config": feature_cfg,
                "selected_pls_n_components": selected_pls_n,
                "best_model": best_row,
                "results": sw_results,
                "pls_grid_search": pls_grid_payload,
                "fusion_model_name": fusion_model_name,
                "fusion_score": fusion_score,
            }

        late_fusion_payload = self._late_fusion(center_payloads_for_fusion)
        if late_fusion_payload is not None:
            fusion_output_dir = ensure_dir(self.output_dir / "fusion")
            late_fusion_payload["oof_df"].to_csv(fusion_output_dir / "final_late_fusion_oof.csv", index=False)
            save_json(late_fusion_payload["metrics"], fusion_output_dir / "final_late_fusion_metrics.json")
            final_report["late_fusion"] = {
                "name": late_fusion_payload["name"],
                "metrics": late_fusion_payload["metrics"],
            }
            print("\n=== FINAL LATE FUSION ===")
            print(late_fusion_payload["metrics"])

        save_json(final_report, Path(self.output_dir) / "final_report.json")
        print(f"\n[DONE] Results saved to: {Path(self.output_dir).resolve()}")
