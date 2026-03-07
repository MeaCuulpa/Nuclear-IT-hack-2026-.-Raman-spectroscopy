from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class PreprocessingConfig:
    enabled: bool = True
    crop_min: float = 700.0
    crop_max: float = 1800.0
    despike_threshold: float = 6.0
    despike_window: int = 3
    smooth_window: int = 11
    smooth_polyorder: int = 3
    baseline_lam: float = 1e5
    baseline_p: float = 0.01
    baseline_niter: int = 10
    normalization: str = "l2"


class SpectraPreprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def transform(self, wave: np.ndarray, x: np.ndarray):
        if not self.config.enabled:
            return wave, x

        wave2, x2 = crop_batch(
            wave,
            x,
            self.config.crop_min,
            self.config.crop_max,
        )

        if x2.size == 0 or wave2.size == 0:
            return wave2, np.empty((0, 0), dtype=float)

        if x2.shape[1] == 0:
            return wave2, np.empty((0, 0), dtype=float)

        x2 = despike_batch(
            x2,
            threshold=self.config.despike_threshold,
            window=self.config.despike_window,
        )

        if x2.size == 0 or x2.shape[1] == 0:
            return wave2, np.empty((0, 0), dtype=float)

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

        x2 = normalize_batch(
            x2,
            method=self.config.normalization,
        )

        return wave2, x2


def crop_batch(wave: np.ndarray, x: np.ndarray, crop_min: float, crop_max: float):
    mask = (wave >= crop_min) & (wave <= crop_max)
    wave2 = wave[mask]

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if mask.sum() == 0:
        return wave2, np.empty((0, 0), dtype=float)

    return wave2, x[:, mask]


def despike_spectrum(y: np.ndarray, threshold: float = 6.0, window: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)

    if y.size == 0:
        return y

    if y.size == 1:
        return y.copy()

    dy = np.diff(y, prepend=y[0])
    med = np.median(dy)
    mad = np.median(np.abs(dy - med))

    if mad == 0:
        return y.copy()

    modified_z = 0.6745 * (dy - med) / mad
    spikes = np.abs(modified_z) > threshold

    y_out = y.copy()
    n = len(y)

    for idx in np.where(spikes)[0]:
        left = max(0, idx - window)
        right = min(n, idx + window + 1)

        local_idx = np.arange(left, right)
        keep = ~spikes[left:right]

        if np.sum(keep) >= 2:
            y_out[idx] = np.median(y[left:right][keep])

    return y_out


def despike_batch(x: np.ndarray, threshold: float = 6.0, window: int = 3) -> np.ndarray:
    if x.size == 0:
        return x
    if x.ndim == 1:
        return despike_spectrum(x, threshold=threshold, window=window).reshape(1, -1)
    return np.vstack(
        [despike_spectrum(row, threshold=threshold, window=window) for row in x]
    )


def smooth_batch(x: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    if x.size == 0:
        return x

    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_features = x.shape[1]
    if n_features < 3:
        return x.copy()

    window_length = min(window_length, n_features)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        return x.copy()

    polyorder = min(polyorder, window_length - 1)
    if polyorder < 1:
        return x.copy()

    return np.vstack(
        [
            savgol_filter(row, window_length=window_length, polyorder=polyorder)
            for row in x
        ]
    )


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    L = len(y)

    if L == 0:
        return y
    if L < 3:
        return y.copy()

    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), format="csr")
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L, format="csr")
        Z = W + lam * (D.T @ D)
        baseline = spsolve(Z.tocsr(), w * y)
        w = p * (y > baseline) + (1 - p) * (y < baseline)

    return baseline


def baseline_correct_batch(x: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    if x.size == 0:
        return x

    if x.ndim == 1:
        x = x.reshape(1, -1)

    corrected = []
    for row in x:
        baseline = baseline_als(row, lam=lam, p=p, niter=niter)
        corrected.append(row - baseline)

    return np.vstack(corrected)


def normalize_batch(x: np.ndarray, method: str = "l2") -> np.ndarray:
    if x.size == 0:
        return x

    if x.ndim == 1:
        x = x.reshape(1, -1)

    x = x.astype(float, copy=True)

    if method == "l2":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    if method == "max":
        denom = np.max(np.abs(x), axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return x / denom

    if method == "none":
        return x

    raise ValueError(f"Unknown normalization method: {method}")