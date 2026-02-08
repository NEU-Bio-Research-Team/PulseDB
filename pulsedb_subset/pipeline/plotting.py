from __future__ import annotations

from pathlib import Path

import matplotlib

# Use non-interactive backend for batch runs
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_histogram(
    values: np.ndarray,
    title: str,
    out_path: Path,
    bins: int = 60,
    xlabel: str = "",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(values)
    v = v[np.isfinite(v)]

    plt.figure(figsize=(7, 4))
    if v.size:
        plt.hist(v, bins=bins, alpha=0.9)
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_example_segment(x: np.ndarray, title: str, out_path: Path, fs: int | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x)
    t = np.arange(x.size)
    if fs and fs > 0:
        t = t / fs

    plt.figure(figsize=(8, 3))
    plt.plot(t, x, linewidth=1.0)
    plt.title(title)
    plt.xlabel("t (s)" if fs else "sample")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
