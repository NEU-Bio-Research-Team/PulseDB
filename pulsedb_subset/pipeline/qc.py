from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QCMetrics:
    nan_ratio: float
    flatline_ratio: float
    saturation_ratio: float
    snr_variance_proxy: float


def compute_qc_metrics(x: np.ndarray) -> QCMetrics:
    """Compute lightweight QC metrics for a 1D signal segment."""
    x = np.asarray(x)
    if x.size == 0:
        return QCMetrics(nan_ratio=1.0, flatline_ratio=1.0, saturation_ratio=1.0, snr_variance_proxy=0.0)

    finite = np.isfinite(x)
    nan_ratio = 1.0 - float(finite.mean())

    xf = x[finite]
    if xf.size < 2:
        return QCMetrics(nan_ratio=nan_ratio, flatline_ratio=1.0, saturation_ratio=1.0, snr_variance_proxy=0.0)

    # Flatline ratio: derivative ~ 0
    dx = np.diff(xf)
    eps = 1e-8
    scale = float(np.std(xf)) + eps
    flatline_ratio = float((np.abs(dx) < (1e-4 * scale)).mean())

    # Saturation: too many values stuck at min/max
    xmin = float(np.min(xf))
    xmax = float(np.max(xf))
    if xmax - xmin < eps:
        saturation_ratio = 1.0
    else:
        saturation_ratio = float(((xf == xmin) | (xf == xmax)).mean())

    # SNR proxy: dynamic range relative to high-frequency noise proxy
    p95 = float(np.percentile(np.abs(xf), 95))
    p05 = float(np.percentile(np.abs(xf), 5))
    hf = float(np.std(np.diff(xf))) + eps
    snr_variance_proxy = float((p95 - p05) / hf)

    return QCMetrics(
        nan_ratio=nan_ratio,
        flatline_ratio=flatline_ratio,
        saturation_ratio=saturation_ratio,
        snr_variance_proxy=snr_variance_proxy,
    )


def passes_label_ranges(sbp: float, dbp: float, sbp_range: tuple[float, float], dbp_range: tuple[float, float]) -> bool:
    if not (np.isfinite(sbp) and np.isfinite(dbp)):
        return False
    return (sbp_range[0] <= sbp <= sbp_range[1]) and (dbp_range[0] <= dbp <= dbp_range[1])
