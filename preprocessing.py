from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve


@dataclass
class PreprocessingConfig:
    enabled: bool = True
    crop_min: float = 700.0
    crop_max: float = 1800.0
    despike_threshold: float = 5.0
    despike_window: int = 3
    smooth_window: int = 9
    smooth_polyorder: int = 3
    baseline_lam: float = 1e5
    baseline_p: float = 0.01
    baseline_niter: int = 10
    normalization: str = "l2"


class SpectraPreprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def transform(self, wave: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.enabled:
            return wave, x

        wave2, x2 = crop_spectra(
            wave,
            x,
            wmin=self.config.crop_min,
            wmax=self.config.crop_max,
        )
        x2 = despike_batch(
            x2,
            threshold=self.config.despike_threshold,
            window=self.config.despike_window,
        )
        x2 = smooth_batch(
            x2,
            window_length=self.config.smooth_window,
            polyorder=self.config.smooth_polyorder,
        )
        x2 = baseline_correct_batch(
            x2,
            lam=self.config.baseline_lam,
            p=self.config.baseline_p,
            niter=self.config.baseline_niter,
        )
        x2 = normalize_batch(x2, method=self.config.normalization)
        return wave2, x2


def crop_spectra(
    wave: np.ndarray,
    x: np.ndarray,
    wmin: float = 700.0,
    wmax: float = 1800.0,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (wave >= wmin) & (wave <= wmax)
    return wave[mask], x[:, mask]


def modified_zscore(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return 0.6745 * (x - med) / mad


def despike_spectrum(y: np.ndarray, threshold: float = 5.0, window: int = 3) -> np.ndarray:
    y = y.copy()
    dy = np.diff(y, prepend=y[0])
    z = np.abs(modified_zscore(dy))
    spike_idx = np.where(z > threshold)[0]

    for idx in spike_idx:
        left = max(0, idx - window)
        right = min(len(y), idx + window + 1)
        neighbors = np.concatenate([y[left:idx], y[idx + 1 : right]])
        if len(neighbors) > 0:
            y[idx] = np.median(neighbors)
    return y


def despike_batch(x: np.ndarray, threshold: float = 5.0, window: int = 3) -> np.ndarray:
    return np.vstack([despike_spectrum(row, threshold=threshold, window=window) for row in x])


def smooth_batch(x: np.ndarray, window_length: int = 9, polyorder: int = 3) -> np.ndarray:
    wl = min(window_length, x.shape[1] - 1 if x.shape[1] % 2 == 0 else x.shape[1])
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, x.shape[1] - 1 if x.shape[1] % 2 == 0 else x.shape[1])

    if wl <= polyorder:
        return x.copy()
    return savgol_filter(x, window_length=wl, polyorder=polyorder, axis=1)


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    length = len(y)
    d = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
    w = np.ones(length)

    for _ in range(niter):
        w_matrix = sparse.spdiags(w, 0, length, length)
        z = w_matrix + lam * d.T @ d
        baseline = spsolve(z, w * y)
        w = p * (y > baseline) + (1 - p) * (y <= baseline)

    return baseline


def baseline_correct_batch(x: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    corrected = []
    for row in x:
        baseline = baseline_als(row, lam=lam, p=p, niter=niter)
        corrected.append(row - baseline)
    return np.vstack(corrected)


def normalize_batch(x: np.ndarray, method: str = "l2") -> np.ndarray:
    x = x.copy()

    if method == "l2":
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    if method == "snv":
        mu = x.mean(axis=1, keepdims=True)
        sigma = x.std(axis=1, keepdims=True) + 1e-12
        return (x - mu) / sigma

    if method == "auc":
        area = np.trapz(np.abs(x), axis=1).reshape(-1, 1) + 1e-12
        return x / area

    raise ValueError(f"Unknown normalization method: {method}")
