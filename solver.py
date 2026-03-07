from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dataset import RamanDataset
from evaluation import evaluate_group_cv
from models import make_baseline_models
from preprocessing import PreprocessingConfig, SpectraPreprocessor
from utils import ensure_dir, save_json, set_seed


class Solver:
    def __init__(self, config):
        self.config = config
        set_seed(int(self.config.project.seed))
        self.output_dir = ensure_dir(self.config.paths.outputs)
        self.dataset = RamanDataset(self.config.paths.data_root)

    def _make_preprocessor_for_center(self, center: str) -> SpectraPreprocessor:
        crop_ranges_by_center = getattr(
            self.config.preprocessing,
            "crop_ranges_by_center",
            None,
        )

        default_crop = list(getattr(self.config.preprocessing, "crop_range", [700, 1800]))

        if crop_ranges_by_center is None:
            center_crop = default_crop
        else:
            center_crop = getattr(crop_ranges_by_center, str(center), default_crop)

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

    def _build_samplewise_cache_tag(
        self,
        center: str,
        include_region_feature: bool,
        agg_features: list[str],
        append_first_derivative: bool,
    ) -> str:
        crop_ranges_by_center = getattr(
            self.config.preprocessing,
            "crop_ranges_by_center",
            None,
        )

        default_crop = list(getattr(self.config.preprocessing, "crop_range", [700, 1800]))

        if crop_ranges_by_center is None:
            center_crop = default_crop
        else:
            center_crop = getattr(crop_ranges_by_center, str(center), default_crop)

        payload = {
            "center": str(center),
            "agg_features": [str(item) for item in agg_features],
            "include_region_feature": bool(include_region_feature),
            "append_first_derivative": bool(append_first_derivative),
            "preprocessing_enabled": bool(self.config.preprocessing.enabled),
            "crop_range_for_this_center": list(center_crop),
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

    def run(self) -> None:
        use_raw_cache = bool(getattr(self.config.data, "use_raw_cache", True))
        force_reload_raw_cache = bool(getattr(self.config.data, "force_reload_raw_cache", False))

        use_aligned_cache = bool(getattr(self.config.data, "use_aligned_cache", True))
        force_rebuild_aligned_cache = bool(
            getattr(self.config.data, "force_rebuild_aligned_cache", False)
        )

        use_samplewise_cache = bool(getattr(self.config.data, "use_samplewise_cache", True))
        force_rebuild_samplewise_cache = bool(
            getattr(self.config.data, "force_rebuild_samplewise_cache", False)
        )

        include_region_feature = bool(getattr(self.config.data, "include_region_feature", False))
        append_first_derivative = bool(getattr(self.config.data, "append_first_derivative", True))
        agg_features_cfg = getattr(self.config.data, "sample_agg_features", ["median"])
        agg_features = [str(value) for value in agg_features_cfg]
        centers_to_run_cfg = getattr(self.config.data, "centers_to_run", [1500, 2900])
        centers_to_run = [str(value) for value in centers_to_run_cfg]
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

        print("[STEP] Creating models...")
        models = make_baseline_models(self.config)

        final_report = {
            "n_files": int(len(metadata_df)),
            "n_unique_mice": int(metadata_df["mouse_id"].nunique()),
            "class_distribution": metadata_df["label_name"].value_counts().to_dict(),
            "center_distribution": metadata_df["center"].value_counts().to_dict(),
            "sample_agg_features": agg_features,
            "append_first_derivative": append_first_derivative,
            "include_region_feature": include_region_feature,
            "cv_strategy": cv_strategy,
            "results_by_center": {},
        }

        for center in centers_to_run:
            center_df = metadata_df[metadata_df["center"].astype(str) == center].copy()
            if center_df.empty:
                print(f"[WARN] No files found for center={center}")
                continue

            center_output_dir = ensure_dir(self.output_dir / f"center_{center}")

            print(f"\n{'=' * 20} CENTER {center} {'=' * 20}")
            print(f"[INFO] Files in center {center}: {len(center_df)}")
            print("[INFO] Class distribution:")
            print(center_df["label_name"].value_counts())

            print(f"[STEP] Building samplewise dataset for center={center}...")
            center_preprocessor = self._make_preprocessor_for_center(center)

            wave_sw, x_sw, y_sw, groups_sw, file_ids_sw = self.dataset.build_samplewise_dataset(
                center_preprocessor,
                agg_methods=agg_features,
                use_processed=bool(self.config.preprocessing.enabled),
                use_cache=use_samplewise_cache,
                force_rebuild=force_rebuild_samplewise_cache,
                cache_tag=self._build_samplewise_cache_tag(
                    center=center,
                    include_region_feature=include_region_feature,
                    agg_features=agg_features,
                    append_first_derivative=append_first_derivative,
                ),
                center_filter=center,
                include_region_feature=include_region_feature,
                append_first_derivative=append_first_derivative,
            )

            print(f"[INFO] Samplewise X shape for center {center}: {x_sw.shape}")
            print(f"[INFO] Samplewise y shape for center {center}: {y_sw.shape}")
            print(f"[INFO] Unique groups for center {center}: {len(set(groups_sw))}")
            print(f"[INFO] Samplewise files for center {center}: {len(file_ids_sw)}")

            print(f"[STEP] Running samplewise CV for center={center}...")
            sw_results, sw_results_df = evaluate_group_cv(
                x_sw,
                y_sw,
                groups_sw,
                models,
                n_splits=int(self.config.training.group_kfold_splits),
                dataset_name=f"samplewise_center{center}",
                cv_strategy=cv_strategy,
                random_state=int(self.config.project.seed),
            )

            print(f"\n=== SAMPLEWISE RESULTS | CENTER {center} ===")
            print(sw_results_df.sort_values("oof_macro_f1", ascending=False))

            sw_results_df.to_csv(
                center_output_dir / f"cv_results_samplewise_center{center}.csv",
                index=False,
            )
            save_json(sw_results, center_output_dir / f"samplewise_results_center{center}.json")

            best_row = sw_results_df.sort_values("oof_macro_f1", ascending=False).iloc[0].to_dict()

            final_report["results_by_center"][f"center_{center}"] = {
                "n_files": int(len(center_df)),
                "class_distribution": center_df["label_name"].value_counts().to_dict(),
                "best_model": best_row,
                "results": sw_results,
            }

        save_json(final_report, Path(self.output_dir) / "final_report.json")
        print(f"\n[DONE] Results saved to: {Path(self.output_dir).resolve()}")
