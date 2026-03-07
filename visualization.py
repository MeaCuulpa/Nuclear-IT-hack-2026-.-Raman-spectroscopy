from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dataset import INV_CLASS_MAP, RamanDataset
from preprocessing import SpectraPreprocessor


def eda_summary(metadata_df) -> None:
    print("\n=== DATASET SUMMARY ===")
    print(metadata_df.head())
    print("\nFiles per class:")
    print(metadata_df["label_name"].value_counts())
    print("\nFiles per mouse_id:")
    print(metadata_df.groupby(["label_name", "mouse_id"]).size())
    print("\nRegions:")
    print(metadata_df["region"].value_counts())
    print("\nPixels stats:")
    print(metadata_df["n_pixels_valid"].describe())


def plot_random_raw_vs_processed(
    raw_data: dict,
    preprocessor: SpectraPreprocessor,
    output_dir,
    n_files: int = 3,
) -> None:
    file_ids = list(raw_data.keys())[:n_files]
    fig, axes = plt.subplots(n_files, 2, figsize=(12, 4 * n_files))
    if n_files == 1:
        axes = np.array([axes])

    for i, file_id in enumerate(file_ids):
        item = raw_data[file_id]
        wave = item["wave"]
        x_pixels = item["X_pixels"]

        raw_spec = x_pixels[np.random.randint(0, len(x_pixels))]
        wave_processed, x_processed = preprocessor.transform(wave, raw_spec[None, :])
        proc_spec = x_processed[0]

        axes[i, 0].plot(wave, raw_spec)
        axes[i, 0].set_title(f"RAW: {file_id}")
        axes[i, 0].set_xlabel("Wave")
        axes[i, 0].set_ylabel("Intensity")

        axes[i, 1].plot(wave_processed, proc_spec)
        axes[i, 1].set_title(f"PROCESSED: {file_id}")
        axes[i, 1].set_xlabel("Wave")
        axes[i, 1].set_ylabel("Intensity")

    plt.tight_layout()
    plt.savefig(output_dir / "raw_vs_processed_examples.png", dpi=200)
    plt.close()


def plot_class_mean_spectra(
    raw_data: dict,
    preprocessor: SpectraPreprocessor,
    output_dir,
    use_processed: bool = True,
) -> None:
    class_specs = {0: [], 1: [], 2: []}
    common_wave = None

    for item in raw_data.values():
        wave = item["wave"]
        x_pixels = item["X_pixels"]
        label = item["meta"].label

        if use_processed:
            wave, x_pixels = preprocessor.transform(wave, x_pixels)

        agg = RamanDataset.aggregate_spectra(x_pixels, method="median")
        class_specs[label].append(agg)
        if common_wave is None:
            common_wave = wave

    plt.figure(figsize=(12, 6))
    for label, arrs in class_specs.items():
        if not arrs:
            continue
        mean_spec = np.mean(np.vstack(arrs), axis=0)
        std_spec = np.std(np.vstack(arrs), axis=0)

        plt.plot(common_wave, mean_spec, label=INV_CLASS_MAP[label])
        plt.fill_between(common_wave, mean_spec - std_spec, mean_spec + std_spec, alpha=0.2)

    plt.title("Mean spectra by class")
    plt.xlabel("Wave (cm^-1)")
    plt.ylabel("Normalized intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mean_spectra_by_class.png", dpi=200)
    plt.close()
