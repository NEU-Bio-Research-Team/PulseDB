from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    # pulsedb_subset/...
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class SubsetConfig:
    subset_mat_dir: Path
    subset_files: dict[str, str]

    artifacts_dir: Path
    reports_dir: Path
    cache_dir: Path

    fs: int
    segment_sec: int
    channels: list[str]
    channel_indices: dict[str, int]

    eda_qc_sample_per_subset: int
    eda_random_seed: int
    eda_sbp_bins: list[float]

    splits_seed: int
    splits_train_frac: float
    splits_val_frac: float

    chunk_size: int
    shard_size: int
    np_dtype: str

    qc_max_nan_ratio: float
    qc_max_flatline_ratio: float
    qc_max_saturation_ratio: float
    qc_label_sbp_range: tuple[float, float]
    qc_label_dbp_range: tuple[float, float]
    qc_abp_label_max_abs_diff: float

    scaling_fit_subset: str
    scaling_waveform: str
    scaling_demographics: str
    demo_features: list[str]
    gender_feature: str
    add_missing_indicators: bool


def load_config(cfg_path: Path | None = None) -> SubsetConfig:
    root = project_root()
    cfg_path = cfg_path or (root / "config.yaml")
    raw = load_yaml(cfg_path)

    subset_mat_dir = (root / str(raw.get("subset_mat_dir", "../archive"))).resolve()
    subset_files = dict(raw.get("subset_files", {}))

    artifacts_dir = root / str(raw.get("artifacts_dir", "artifacts"))
    reports_dir = root / str(raw.get("reports_dir", "reports"))
    cache_dir = root / str(raw.get("cache_dir", "cache/pulsedb_subset"))

    fs = int(raw.get("fs", 125))
    segment_sec = int(raw.get("segment_sec", 10))
    channels = [str(c) for c in raw.get("channels", ["PPG"])]

    # Explicit mapping from logical channel name -> index in Signals tensor
    # Example:
    # channel_indices:
    #   PPG: 1
    channel_indices_raw = raw.get("channel_indices", {}) or {}
    channel_indices = {str(k): int(v) for k, v in dict(channel_indices_raw).items()}

    eda = raw.get("eda", {}) or {}
    eda_qc_sample_per_subset = int(eda.get("qc_sample_per_subset", 20000))
    eda_random_seed = int(eda.get("random_seed", 42))
    eda_sbp_bins = [float(x) for x in eda.get("sbp_bins", [0, 90, 120, 140, 1000])]

    splits = raw.get("splits", {}) or {}
    splits_seed = int(splits.get("seed", eda_random_seed))
    splits_train_frac = float(splits.get("train_frac", 0.8))
    splits_val_frac = float(splits.get("val_frac", 0.1))

    io = raw.get("io", {}) or {}
    chunk_size = int(io.get("chunk_size", 4096))
    shard_size = int(io.get("shard_size", 8192))
    np_dtype = str(io.get("np_dtype", "float32"))

    qc = raw.get("qc", {}) or {}
    qc_max_nan_ratio = float(qc.get("max_nan_ratio", 0.05))
    qc_max_flatline_ratio = float(qc.get("max_flatline_ratio", 0.20))
    qc_max_saturation_ratio = float(qc.get("max_saturation_ratio", 0.20))
    qc_label_sbp_range = tuple(qc.get("label_sbp_range", [60, 240]))  # type: ignore[assignment]
    qc_label_dbp_range = tuple(qc.get("label_dbp_range", [30, 150]))  # type: ignore[assignment]
    qc_abp_label_max_abs_diff = float(qc.get("abp_label_max_abs_diff", 30))

    scaling = raw.get("scaling", {}) or {}
    scaling_fit_subset = str(scaling.get("fit_subset", "Train"))
    scaling_waveform = str(scaling.get("waveform", "zscore"))
    scaling_demographics = str(scaling.get("demographics", "zscore"))
    demo_features = [str(x) for x in scaling.get("demographics_features", ["age", "height", "weight", "bmi"])]
    gender_feature = str(scaling.get("gender_feature", "gender"))
    add_missing_indicators = bool(scaling.get("add_missing_indicators", True))

    return SubsetConfig(
        subset_mat_dir=subset_mat_dir,
        subset_files=subset_files,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        cache_dir=cache_dir,
        fs=fs,
        segment_sec=segment_sec,
        channels=channels,
        channel_indices=channel_indices,
        eda_qc_sample_per_subset=eda_qc_sample_per_subset,
        eda_random_seed=eda_random_seed,
        eda_sbp_bins=eda_sbp_bins,
        splits_seed=splits_seed,
        splits_train_frac=splits_train_frac,
        splits_val_frac=splits_val_frac,
        chunk_size=chunk_size,
        shard_size=shard_size,
        np_dtype=np_dtype,
        qc_max_nan_ratio=qc_max_nan_ratio,
        qc_max_flatline_ratio=qc_max_flatline_ratio,
        qc_max_saturation_ratio=qc_max_saturation_ratio,
        qc_label_sbp_range=(float(qc_label_sbp_range[0]), float(qc_label_sbp_range[1])),
        qc_label_dbp_range=(float(qc_label_dbp_range[0]), float(qc_label_dbp_range[1])),
        qc_abp_label_max_abs_diff=qc_abp_label_max_abs_diff,
        scaling_fit_subset=scaling_fit_subset,
        scaling_waveform=scaling_waveform,
        scaling_demographics=scaling_demographics,
        demo_features=demo_features,
        gender_feature=gender_feature,
        add_missing_indicators=add_missing_indicators,
    )
