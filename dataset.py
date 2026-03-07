from __future__ import annotations

import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from preprocessing import SpectraPreprocessor


CLASS_MAP = {
    "control": 0,
    "endo": 1,
    "exo": 2,
}
INV_CLASS_MAP = {value: key for key, value in CLASS_MAP.items()}


@dataclass
class FileMeta:
    filepath: str
    label_name: str
    label: int
    mouse_id: str
    sample_id: str
    region: str


class RamanDataset:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.metadata_df: Optional[pd.DataFrame] = None
        self.raw_data: Optional[Dict[str, dict]] = None

    def _get_cache_dir(self) -> Path:
        cache_dir = self.root_dir.parent / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @staticmethod
    def infer_region(filename: str) -> str:
        name = filename.lower()
        if "cortex" in name:
            return "cortex"
        if "striatum" in name:
            return "striatum"
        if "cerebellum" in name:
            return "cerebellum"
        return "unknown"

    @staticmethod
    def _should_skip_file(txt_path: Path) -> Tuple[bool, str]:
        name = txt_path.name.lower()
        parts = [part.lower() for part in txt_path.parts]

        if "correlation_results" in parts:
            return True, "auxiliary correlation_results file"

        if name == "summary.txt":
            return True, "summary file"

        if name.endswith("_average.txt"):
            return True, "average spectrum file"

        return False, ""

    @classmethod
    def discover_files(cls, root_dir: Path) -> List[FileMeta]:
        metas: List[FileMeta] = []

        for class_name in CLASS_MAP:
            class_dir = root_dir / class_name
            if not class_dir.exists():
                print(f"[WARN] Missing folder: {class_dir}")
                continue

            for txt_path in sorted(class_dir.rglob("*.txt")):
                should_skip, reason = cls._should_skip_file(txt_path)
                if should_skip:
                    print(f"[SKIP] {reason}: {txt_path}")
                    continue

                metas.append(
                    FileMeta(
                        filepath=str(txt_path),
                        label_name=class_name,
                        label=CLASS_MAP[class_name],
                        mouse_id=txt_path.parent.name,
                        sample_id=txt_path.stem,
                        region=cls.infer_region(txt_path.name),
                    )
                )

        return metas

    @staticmethod
    def robust_read_txt(filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath, sep=r"\s+|\t+", engine="python", comment=None)
        except Exception:
            df = pd.read_csv(filepath, sep="\t", engine="python", comment=None)

        rename_map = {}
        for column in df.columns:
            clean = str(column).strip().replace('"', "")
            rename_map[column] = clean

        df = df.rename(columns=rename_map)

        column_aliases = {
            "#Wvave": "#Wave",
            "Wvave": "#Wave",
            "Wave": "#Wave",
            "Intensity": "#Intensity",
            "X": "#X",
            "Y": "#Y",
        }
        df = df.rename(columns=column_aliases)

        required = ["#X", "#Y", "#Wave", "#Intensity"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(
                f"{filepath}: missing columns {missing}. Found: {df.columns.tolist()}"
            )

        df = df[required].copy()

        for column in required:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=required)

        if df.empty:
            raise ValueError(f"{filepath}: no valid numeric rows after parsing.")

        return df

    @staticmethod
    def long_to_cube(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_unique = np.sort(df["#X"].unique())
        y_unique = np.sort(df["#Y"].unique())
        wave = np.sort(df["#Wave"].unique())

        pivot = df.pivot_table(
            index=["#Y", "#X"],
            columns="#Wave",
            values="#Intensity",
            aggfunc="first",
        )

        full_index = pd.MultiIndex.from_product([y_unique, x_unique], names=["#Y", "#X"])
        pivot = pivot.reindex(full_index)
        pivot = pivot.reindex(columns=wave)

        cube = pivot.values.reshape(len(y_unique), len(x_unique), len(wave))
        return x_unique, y_unique, wave, cube

    @staticmethod
    def flatten_cube(cube: np.ndarray) -> np.ndarray:
        return cube.reshape(-1, cube.shape[-1])

    @staticmethod
    def aggregate_spectra(x: np.ndarray, method: str = "median") -> np.ndarray:
        if method == "median":
            return np.nanmedian(x, axis=0)
        if method == "mean":
            return np.nanmean(x, axis=0)
        raise ValueError(f"Unknown aggregation method: {method}")

    def load(
        self,
        use_cache: bool = True,
        force_reload: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, dict]]:
        cache_path = self._get_cache_dir() / f"raw_cache_{self.root_dir.name}.joblib"

        if use_cache and cache_path.exists() and not force_reload:
            print(f"[CACHE] Loading raw dataset from: {cache_path}")
            payload = joblib.load(cache_path)
            self.metadata_df = payload["metadata_df"]
            self.raw_data = payload["raw_data"]
            return self.metadata_df, self.raw_data

        metas = self.discover_files(self.root_dir)
        if not metas:
            raise RuntimeError(f"No valid .txt files found under {self.root_dir}")

        rows = []
        raw_data: Dict[str, dict] = {}

        for index, meta in enumerate(metas, start=1):
            print(f"[{index}/{len(metas)}] Reading: {meta.filepath}")

            df = self.robust_read_txt(meta.filepath)
            x_unique, y_unique, wave, cube = self.long_to_cube(df)
            x_pixels = self.flatten_cube(cube)

            valid_mask = ~np.isnan(x_pixels).all(axis=1)
            x_pixels = x_pixels[valid_mask]

            if len(x_pixels) == 0:
                print(f"[SKIP] No valid spectra after loading: {meta.filepath}")
                continue

            x_agg = self.aggregate_spectra(x_pixels, method="median")

            file_id = f"{meta.label_name}__{meta.mouse_id}__{meta.sample_id}"
            rows.append(
                {
                    "file_id": file_id,
                    "filepath": meta.filepath,
                    "label_name": meta.label_name,
                    "label": meta.label,
                    "mouse_id": meta.mouse_id,
                    "sample_id": meta.sample_id,
                    "region": meta.region,
                    "n_x": len(x_unique),
                    "n_y": len(y_unique),
                    "n_pixels_valid": len(x_pixels),
                    "n_wave": len(wave),
                    "wave_min": float(np.min(wave)),
                    "wave_max": float(np.max(wave)),
                }
            )

            raw_data[file_id] = {
                "meta": meta,
                "wave": wave,
                "X_pixels": x_pixels,
                "X_agg": x_agg,
            }

        if not rows:
            raise RuntimeError("All files were skipped or invalid.")

        self.metadata_df = pd.DataFrame(rows)
        self.raw_data = raw_data

        if use_cache:
            print(f"[CACHE] Saving raw dataset to: {cache_path}")
            joblib.dump(
                {
                    "metadata_df": self.metadata_df,
                    "raw_data": self.raw_data,
                },
                cache_path,
                compress=3,
            )

        return self.metadata_df, self.raw_data

    @staticmethod
    def check_wave_consistency(raw_data: Dict[str, dict], tol: float = 1e-8) -> bool:
        waves = [item["wave"] for item in raw_data.values()]
        if not waves:
            return True

        reference = waves[0]
        for index, wave in enumerate(waves[1:], start=1):
            if len(wave) != len(reference):
                print(f"[WARN] Wave length mismatch at file index={index}")
                return False
            if not np.allclose(wave, reference, atol=tol, rtol=0):
                print(f"[WARN] Wave values mismatch at file index={index}")
                return False

        print("[OK] All wave axes are consistent.")
        return True

    @classmethod
    def interpolate_to_reference(
        cls,
        raw_data: Dict[str, dict],
        reference_wave: np.ndarray,
    ) -> None:
        for item in raw_data.values():
            wave = item["wave"]
            if len(wave) == len(reference_wave) and np.allclose(wave, reference_wave):
                continue

            x_pixels = item["X_pixels"]
            new_x = []

            for row in x_pixels:
                interpolator = interp1d(
                    wave,
                    row,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                new_x.append(interpolator(reference_wave))

            item["wave"] = reference_wave.copy()
            item["X_pixels"] = np.vstack(new_x)
            item["X_agg"] = cls.aggregate_spectra(item["X_pixels"], method="median")

    def prepare_raw_data(
        self,
        raw_data: Dict[str, dict],
        use_cache: bool = True,
        force_rebuild: bool = False,
    ) -> Dict[str, dict]:
        cache_path = self._get_cache_dir() / f"aligned_raw_cache_{self.root_dir.name}.joblib"

        if use_cache and cache_path.exists() and not force_rebuild:
            print(f"[CACHE] Loading aligned raw data from: {cache_path}")
            aligned_raw = joblib.load(cache_path)
            self.raw_data = aligned_raw
            return aligned_raw

        if self.check_wave_consistency(raw_data):
            if use_cache:
                print(f"[CACHE] Saving aligned raw data to: {cache_path}")
                joblib.dump(raw_data, cache_path, compress=3)
            self.raw_data = raw_data
            return raw_data

        first_key = next(iter(raw_data))
        reference_wave = raw_data[first_key]["wave"]
        print("[INFO] Interpolating all spectra to the reference wave axis...")
        self.interpolate_to_reference(raw_data, reference_wave)
        self.check_wave_consistency(raw_data)

        if use_cache:
            print(f"[CACHE] Saving aligned raw data to: {cache_path}")
            joblib.dump(raw_data, cache_path, compress=3)

        self.raw_data = raw_data
        return raw_data

    def build_samplewise_dataset(
        self,
        preprocessor: SpectraPreprocessor,
        agg_method: str = "median",
        use_processed: bool = True,
        use_cache: bool = True,
        force_rebuild: bool = False,
        cache_tag: str = "default",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        if self.raw_data is None:
            raise RuntimeError("Call load() before building datasets")

        safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in cache_tag)
        cache_path = self._get_cache_dir() / f"samplewise_{self.root_dir.name}_{safe_tag}.joblib"

        if use_cache and cache_path.exists() and not force_rebuild:
            print(f"[CACHE] Loading samplewise dataset from: {cache_path}")
            payload = joblib.load(cache_path)
            return (
                payload["wave_ref"],
                payload["X"],
                payload["y"],
                payload["groups"],
                payload["file_ids"],
            )

        x_list, y_list, groups, file_ids = [], [], [], []
        wave_ref = None

        for file_id, item in self.raw_data.items():
            wave = item["wave"]
            x_pixels = item["X_pixels"]

            if use_processed:
                wave, x_pixels = preprocessor.transform(wave, x_pixels)

            if len(x_pixels) == 0:
                continue

            x_agg = self.aggregate_spectra(x_pixels, method=agg_method)
            x_list.append(x_agg)
            y_list.append(item["meta"].label)
            groups.append(item["meta"].mouse_id)
            file_ids.append(file_id)

            if wave_ref is None:
                wave_ref = wave

        if not x_list:
            raise RuntimeError("No samplewise data could be built.")

        payload = {
            "wave_ref": wave_ref,
            "X": np.vstack(x_list),
            "y": np.array(y_list),
            "groups": np.array(groups),
            "file_ids": file_ids,
        }

        if use_cache:
            print(f"[CACHE] Saving samplewise dataset to: {cache_path}")
            joblib.dump(payload, cache_path, compress=3)

        return (
            payload["wave_ref"],
            payload["X"],
            payload["y"],
            payload["groups"],
            payload["file_ids"],
        )