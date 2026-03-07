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

        self.preprocessor = SpectraPreprocessor(
            PreprocessingConfig(
                enabled=bool(self.config.preprocessing.enabled),
                crop_min=float(self.config.preprocessing.crop_range[0]),
                crop_max=float(self.config.preprocessing.crop_range[1]),
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

    def _build_samplewise_cache_tag(self) -> str:
        payload = {
            "agg_method": str(self.config.data.sample_agg_method),
            "preprocessing_enabled": bool(self.config.preprocessing.enabled),
            "crop_range": list(self.config.preprocessing.crop_range),
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
        print("[STEP] Loading raw dataset...")
        metadata_df, raw_data = self.dataset.load(
            use_cache=bool(self.config.data.use_raw_cache),
            force_reload=bool(self.config.data.force_reload_raw_cache),
        )
        metadata_df.to_csv(self.output_dir / "metadata_summary.csv", index=False)

        print("[STEP] Aligning wave axes...")
        raw_data = self.dataset.prepare_raw_data(
            raw_data,
            use_cache=bool(self.config.data.use_aligned_cache),
            force_rebuild=bool(self.config.data.force_rebuild_aligned_cache),
        )

        print("[STEP] Creating models...")
        models = make_baseline_models(self.config)

        print("[STEP] Building samplewise dataset...")
        wave_sw, x_sw, y_sw, groups_sw, file_ids_sw = self.dataset.build_samplewise_dataset(
            self.preprocessor,
            agg_method=str(self.config.data.sample_agg_method),
            use_processed=bool(self.config.preprocessing.enabled),
            use_cache=bool(self.config.data.use_samplewise_cache),
            force_rebuild=bool(self.config.data.force_rebuild_samplewise_cache),
            cache_tag=self._build_samplewise_cache_tag(),
        )
        del wave_sw, file_ids_sw, raw_data

        print(f"[INFO] Samplewise X shape: {x_sw.shape}")
        print(f"[INFO] Samplewise y shape: {y_sw.shape}")
        print(f"[INFO] Unique groups: {len(set(groups_sw))}")

        print("[STEP] Running samplewise CV...")
        sw_results, sw_results_df = evaluate_group_cv(
            x_sw,
            y_sw,
            groups_sw,
            models,
            n_splits=int(self.config.training.group_kfold_splits),
            dataset_name="samplewise",
        )

        print("\n=== SAMPLEWISE RESULTS ===")
        print(sw_results)
        print(sw_results_df)

        sw_results_df.to_csv(self.output_dir / "cv_results_samplewise.csv", index=False)

        final_report = {
            "n_files": int(len(metadata_df)),
            "n_unique_mice": int(metadata_df["mouse_id"].nunique()),
            "class_distribution": metadata_df["label_name"].value_counts().to_dict(),
            "samplewise_results": sw_results,
        }

        save_json(final_report, Path(self.output_dir) / "final_report.json")
        print(f"\n[DONE] Results saved to: {Path(self.output_dir).resolve()}")