from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm
import yaml

from pulsedb_subset.pipeline.config import load_config
from pulsedb_subset.pipeline.eda_labels import decode_subject_id, read_vector
from pulsedb_subset.pipeline.io_mat import H5MatReader, is_hdf5_mat, load_json
from pulsedb_subset.pipeline.qc import passes_label_ranges
from pulsedb_subset.pipeline.scaler import RunningChannelMoments, RunningMeanStd, ZScoreScaler, save_scalers
from pulsedb_subset.pipeline.shards import ShardWriter
from pulsedb_subset.pipeline.waveform import enforce_nct, infer_segment_axis, select_channel_indices


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _append_dropped(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)


def _load_subject_splits(artifacts_dir: Path) -> dict[str, set[int]] | None:
    """Load global subject-disjoint split sets.

    Returns dict with keys: train/val/test -> set(subject_id).
    """
    path = artifacts_dir / "subject_splits.yaml"
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, set[int]] = {}
    for split in ["train", "val", "test"]:
        key = f"{split}_subjects"
        out[split] = set(int(x) for x in (raw.get(key, []) or []))
    return out


def _build_demo_matrix(
    cfg,
    demo_raw: dict[str, np.ndarray],
    fit_means: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build demo feature matrix.

    Returns (demo, feature_names)

    - Numeric features are float32
    - Gender is kept numeric if present, else NaN
    - Optionally appends missing indicators
    - Optionally imputes missing with provided fit_means
    """
    raw_cols: list[np.ndarray] = []
    names: list[str] = []

    for feat in cfg.demo_features:
        raw_cols.append(np.asarray(demo_raw.get(feat, np.array([])), dtype=np.float32).reshape((-1,)))
        names.append(feat)

    raw_cols.append(
        np.asarray(demo_raw.get(cfg.gender_feature, np.array([])), dtype=np.float32).reshape((-1,))
    )
    names.append(cfg.gender_feature)

    # Determine n
    n = max((c.size for c in raw_cols), default=0)
    if n == 0:
        return np.zeros((0, len(raw_cols)), dtype=np.float32), names

    # Pad/truncate and keep raw (for missing indicators)
    raw_fixed: list[np.ndarray] = []
    for c in raw_cols:
        if c.size == n:
            raw_fixed.append(c)
            continue
        out = np.full((n,), np.nan, dtype=np.float32)
        take = min(n, c.size)
        if take:
            out[:take] = c[:take]
        raw_fixed.append(out)

    # Impute if requested
    imp_fixed: list[np.ndarray] = []
    for name, c in zip(names, raw_fixed):
        if fit_means and name in fit_means:
            m = float(fit_means[name])
            c = c.copy()
            c[~np.isfinite(c)] = m
        imp_fixed.append(c)

    demo = np.stack(imp_fixed, axis=1).astype(np.float32, copy=False)

    if cfg.add_missing_indicators:
        miss = (~np.isfinite(np.stack(raw_fixed, axis=1))).astype(np.float32)
        demo = np.concatenate([demo, miss], axis=1)
        names = names + [f"missing_{name}" for name in names]

    return demo.astype(np.float32, copy=False), names


def _fit_demo_scaler(demo: np.ndarray, feature_names: list[str]) -> tuple[ZScoreScaler, dict[str, float]]:
    # fit only on first block of features (before missing indicators)
    d = demo
    if d.size == 0:
        return ZScoreScaler(mean=np.zeros((d.shape[1],), dtype=np.float32), std=np.ones((d.shape[1],), dtype=np.float32)), {}

    # compute mean/std ignoring NaNs
    mean = np.nanmean(d, axis=0)
    std = np.nanstd(d, axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    for i, name in enumerate(feature_names):
        if name.startswith("missing_"):
            mean[i] = 0.0
            std[i] = 1.0

    means_dict = {name: float(m) for name, m in zip(feature_names, mean) if not name.startswith("missing_")}
    return ZScoreScaler(mean=mean.astype(np.float32), std=std.astype(np.float32)), means_dict


def _qc_mask(cfg, x_nct: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return pass mask for waveform QC and per-segment summary metrics."""
    x = x_nct
    finite = np.isfinite(x)
    nan_ratio_ch = 1.0 - finite.mean(axis=2)
    nan_ratio = nan_ratio_ch.max(axis=1)

    # flatline
    eps = 1e-8
    x0 = np.nan_to_num(x, nan=0.0)
    dx = np.diff(x0, axis=2)
    scale = np.nanstd(x, axis=2) + eps
    thresh = (1e-4 * scale)[..., None]
    flat_ratio_ch = (np.abs(dx) < thresh).mean(axis=2)
    flat_ratio = flat_ratio_ch.max(axis=1)

    # saturation
    xmin = np.nanmin(x, axis=2, keepdims=True)
    xmax = np.nanmax(x, axis=2, keepdims=True)
    rng = np.maximum(xmax - xmin, eps)
    tol = np.maximum(1e-6, 1e-6 * rng)
    sat_ch = ((np.abs(x - xmin) <= tol) | (np.abs(x - xmax) <= tol)).mean(axis=2)
    sat_ratio = sat_ch.max(axis=1)

    pass_mask = (
        (nan_ratio <= cfg.qc_max_nan_ratio)
        & (flat_ratio <= cfg.qc_max_flatline_ratio)
        & (sat_ratio <= cfg.qc_max_saturation_ratio)
    )

    metrics = {
        "nan_ratio": nan_ratio,
        "flatline_ratio": flat_ratio,
        "saturation_ratio": sat_ratio,
    }
    return pass_mask, metrics


def _fit_waveform_scaler(cfg, mat_path: Path, ds_map: dict, keep_mask: np.ndarray | None = None) -> ZScoreScaler:
    signals_path = ds_map.get("signals")
    if not signals_path:
        raise ValueError("Cannot fit waveform scaler: missing signals dataset path")

    with H5MatReader(mat_path) as reader:
        sig_shape = reader.shape(signals_path)
        # Use SBP length as a hint when available
        sbp_hint = read_vector(mat_path, ds_map.get("sbp"))
        seg_axis = infer_segment_axis(sig_shape, n_segments_hint=int(sbp_hint.size) if sbp_hint.size else None)
        n = int(sig_shape[seg_axis])
        _log(f"fit scaler: signals_shape={sig_shape}, segment_axis={seg_axis}, n_segments={n}")
        moments = RunningChannelMoments.create(channels=len(cfg.channels))

        if keep_mask is not None:
            keep_mask = np.asarray(keep_mask, dtype=bool).reshape((-1,))
            if keep_mask.size != n:
                raise ValueError(f"keep_mask size {keep_mask.size} != n_segments {n}")

        for start in tqdm(range(0, n, cfg.chunk_size), desc="Fit scaler (waveform)"):
            stop = min(n, start + cfg.chunk_size)
            if keep_mask is not None:
                local_keep = keep_mask[start:stop]
                if local_keep.size == 0 or not bool(np.any(local_keep)):
                    continue
            raw = reader.read_slice_axis(signals_path, seg_axis, start, stop)
            try:
                x_nct = enforce_nct(raw, segment_len=cfg.fs * cfg.segment_sec)
                x_nct = select_channel_indices(x_nct, cfg.channels, cfg.channel_indices)
                x_nct = x_nct.astype(np.float32, copy=False)
            except Exception:
                continue

            if x_nct.shape[1] != 1:
                raise RuntimeError(f"PPG-only pipeline expected C=1 after channel selection, got C={x_nct.shape[1]}")

            if keep_mask is not None:
                x_nct = x_nct[local_keep]

            pass_mask, _ = _qc_mask(cfg, x_nct)
            kept = x_nct[pass_mask]
            if kept.shape[0] == 0:
                continue
            moments.update_nct(kept)

        return moments.to_scaler()


def main() -> int:
    _log("04_preprocess_cache_shards: start")
    cfg = load_config()

    # Hard-lock: this pipeline run is PPG-only.
    if cfg.channels != ["PPG"]:
        raise RuntimeError(f"PPG-only pipeline expects channels=['PPG'], got channels={cfg.channels}")
    if "PPG" not in cfg.channel_indices:
        raise RuntimeError("Missing channel_indices.PPG in config.yaml")
    ppg_raw_idx = int(cfg.channel_indices["PPG"])
    if ppg_raw_idx != 1:
        raise RuntimeError(f"Expected PPG raw channel index=1, got {ppg_raw_idx}. Re-check debug plots.")
    _log(f"Preprocess on PPG raw channel index={ppg_raw_idx}")

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    schema_path = cfg.artifacts_dir / "schema_map.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing {schema_path}. Run scripts/01_inventory_schema.py first.")
    schemas = load_json(schema_path)
    _log(f"schema_map loaded: {len(schemas)} subsets")

    subject_splits = _load_subject_splits(cfg.artifacts_dir)
    if subject_splits is None:
        raise FileNotFoundError(
            "Missing subject_splits.yaml. Run scripts/02_make_subject_splits.py first to generate subject-disjoint splits."
        )
    _log(
        "subject splits loaded: "
        f"train={len(subject_splits['train'])} val={len(subject_splits['val'])} test={len(subject_splits['test'])}"
    )

    dropped_csv = cfg.artifacts_dir / "dropped_segments.csv"
    if dropped_csv.exists():
        dropped_csv.unlink()
    _log(f"dropped log: {dropped_csv}")

    # ---- Fit scalers on Train only ----
    fit_subset = cfg.scaling_fit_subset
    if fit_subset not in cfg.subset_files:
        raise ValueError(f"scaling.fit_subset='{fit_subset}' not in subset_files")

    fit_mat = (cfg.subset_mat_dir / cfg.subset_files[fit_subset]).resolve()
    if not is_hdf5_mat(fit_mat):
        raise RuntimeError(
            "Waveform preprocessing requires MATLAB v7.3 (HDF5) .mat files for streaming reads. "
            f"Detected non-HDF5 MAT: {fit_mat}.\n\n"
            "Fix options:\n"
            "1) Download/obtain the v7.3 (HDF5) version of the subset files, OR\n"
            "2) Convert in MATLAB: load('VitalDB_Train_Subset.mat'); save('VitalDB_Train_Subset_v73.mat','-v7.3') (repeat for each subset).\n"
        )
    fit_map = (schemas.get(fit_subset, {}) or {}).get("dataset_map", {})

    _log(f"fit scalers on subset={fit_subset} (split=train only)")

    sbp = read_vector(fit_mat, fit_map.get("sbp")).astype(np.float32, copy=False)
    dbp = read_vector(fit_mat, fit_map.get("dbp")).astype(np.float32, copy=False)
    expected_n_fit = int(max(sbp.size, dbp.size))
    sid_fit_raw = read_vector(fit_mat, fit_map.get("subject_id"))
    sid_fit = decode_subject_id(sid_fit_raw, expected_n=expected_n_fit)
    if sid_fit.size == 0:
        raise RuntimeError("Missing subject_id in fit subset; cannot enforce subject-disjoint split")
    if sid_fit.size != expected_n_fit:
        raise RuntimeError(
            f"Fit subset subject_id length mismatch: got={sid_fit.size} expected={expected_n_fit}. "
            "Check /Subset/Subject decoding."
        )

    fit_mask = np.isin(sid_fit, np.array(sorted(subject_splits["train"]), dtype=np.int64))
    _log(f"scaler-fit: Train split=train segments = {int(fit_mask.sum())} / {int(fit_mask.size)}")

    demo_raw_full = {
        "age": read_vector(fit_mat, fit_map.get("age")),
        "gender": read_vector(fit_mat, fit_map.get("gender")),
        "height": read_vector(fit_mat, fit_map.get("height")),
        "weight": read_vector(fit_mat, fit_map.get("weight")),
        "bmi": read_vector(fit_mat, fit_map.get("bmi")),
    }

    demo_full, demo_names = _build_demo_matrix(cfg, demo_raw_full, fit_means=None)
    demo_full = demo_full[fit_mask]
    demo_scaler, demo_fit_means = _fit_demo_scaler(demo_full, demo_names)

    _log("fit waveform scaler (streaming + QC) on Train∩split=train")
    wave_scaler = _fit_waveform_scaler(cfg, fit_mat, fit_map, keep_mask=fit_mask)

    scalers_path = cfg.artifacts_dir / "scalers.json"
    save_scalers(scalers_path, {"waveform": wave_scaler, "demographics": demo_scaler})
    _log(f"wrote scalers: {scalers_path}")

    # ---- Write shards for each subset ----
    for subset_name, filename in cfg.subset_files.items():
        _log(f"subset={subset_name}: start shard writing (subject-disjoint splits)")
        mat_path = (cfg.subset_mat_dir / filename).resolve()
        ds_map = (schemas.get(subset_name, {}) or {}).get("dataset_map", {})

        signals_path = ds_map.get("signals")
        if not signals_path:
            print(f"[WARN] {subset_name}: missing signals path; skipping")
            continue

        sbp = read_vector(mat_path, ds_map.get("sbp")).astype(np.float32, copy=False)
        dbp = read_vector(mat_path, ds_map.get("dbp")).astype(np.float32, copy=False)
        expected_n = int(max(sbp.size, dbp.size))
        subject_id_raw = read_vector(mat_path, ds_map.get("subject_id"))
        sid_all = decode_subject_id(subject_id_raw, expected_n=expected_n) if subject_id_raw.size else np.array([], dtype=np.int64)
        if sid_all.size == 0:
            print(f"[WARN] {subset_name}: missing subject_id; skipping (cannot enforce subject-disjoint splits)")
            continue
        if sid_all.size != expected_n:
            raise RuntimeError(
                f"subset={subset_name}: subject_id length mismatch: got={sid_all.size} expected={expected_n}. "
                "Check /Subset/Subject decoding."
            )

        split_masks = {
            split: np.isin(sid_all, np.array(sorted(subject_splits[split]), dtype=np.int64))
            for split in ["train", "val", "test"]
        }

        demo_raw = {
            "age": read_vector(mat_path, ds_map.get("age")),
            "gender": read_vector(mat_path, ds_map.get("gender")),
            "height": read_vector(mat_path, ds_map.get("height")),
            "weight": read_vector(mat_path, ds_map.get("weight")),
            "bmi": read_vector(mat_path, ds_map.get("bmi")),
        }

        demo_full, demo_names = _build_demo_matrix(cfg, demo_raw, fit_means=demo_fit_means)

        writers = {
            split: ShardWriter(
                out_dir=(cfg.cache_dir / split / subset_name),
                shard_size=cfg.shard_size,
                subset_name=f"{subset_name}:{split}",
                meta={
                    "subset_name": subset_name,
                    "split": split,
                    "fs": cfg.fs,
                    "segment_sec": cfg.segment_sec,
                    "channels_requested": cfg.channels,
                    "channel_indices": cfg.channel_indices,
                    "demo_features": demo_names,
                },
            )
            for split in ["train", "val", "test"]
        }

        with H5MatReader(mat_path) as reader:
            sig_shape = reader.shape(signals_path)
            seg_axis = infer_segment_axis(sig_shape, n_segments_hint=int(sbp.size) if sbp.size else None)
            n = int(sig_shape[seg_axis])
            _log(f"subset={subset_name}: signals_shape={sig_shape}, segment_axis={seg_axis}, n_segments={n}")
            for start in tqdm(range(0, n, cfg.chunk_size), desc=f"Shard {subset_name}"):
                stop = min(n, start + cfg.chunk_size)
                raw = reader.read_slice_axis(signals_path, seg_axis, start, stop)
                try:
                    x_nct = enforce_nct(raw, segment_len=cfg.fs * cfg.segment_sec)
                    x_nct = select_channel_indices(x_nct, cfg.channels, cfg.channel_indices)
                    x_nct = x_nct.astype(np.float32, copy=False)
                except Exception:
                    continue

                if x_nct.shape[1] != 1:
                    raise RuntimeError(
                        f"PPG-only pipeline expected C=1 after channel selection, got C={x_nct.shape[1]} (subset={subset_name})"
                    )

                # align label/demo lengths
                end = min(stop, sbp.size, dbp.size, demo_full.shape[0] if demo_full.size else stop)
                local_n = end - start
                if local_n <= 0:
                    continue

                x_nct = x_nct[:local_n]
                sbp_s = sbp[start:end]
                dbp_s = dbp[start:end]
                demo_s = demo_full[start:end] if demo_full.size else np.zeros((local_n, len(demo_names)), dtype=np.float32)
                sid_s = sid_all[start:end]

                y = np.stack([sbp_s, dbp_s], axis=1).astype(np.float32, copy=False)

                # label range QC
                label_ok = np.array(
                    [
                        passes_label_ranges(float(s), float(d), cfg.qc_label_sbp_range, cfg.qc_label_dbp_range)
                        for s, d in zip(sbp_s, dbp_s)
                    ],
                    dtype=bool,
                )

                wave_ok, metrics = _qc_mask(cfg, x_nct)
                keep_base = label_ok & wave_ok

                # log drops
                dropped_rows = []
                for i in range(local_n):
                    if keep_base[i]:
                        continue
                    reason = "label" if not label_ok[i] else "waveform"
                    dropped_rows.append(
                        {
                            "subset_name": subset_name,
                            "split": "*",
                            "index": int(start + i),
                            "reason": reason,
                            "nan_ratio": float(metrics["nan_ratio"][i]),
                            "flatline_ratio": float(metrics["flatline_ratio"][i]),
                            "saturation_ratio": float(metrics["saturation_ratio"][i]),
                        }
                    )
                _append_dropped(dropped_csv, dropped_rows)

                if not keep_base.any():
                    continue

                for split in ["train", "val", "test"]:
                    sm = split_masks[split][start:end]
                    keep = keep_base & sm
                    if not keep.any():
                        continue

                    x_keep = x_nct[keep]
                    y_keep = y[keep]
                    demo_keep = demo_s[keep]
                    sid_keep = sid_s[keep]

                    # normalize
                    x_keep = wave_scaler.transform(x_keep)
                    demo_keep = demo_scaler.transform(demo_keep)

                    writers[split].add(
                        x_keep.astype(np.float32, copy=False),
                        y_keep,
                        demo_keep,
                        subject_id=sid_keep,
                    )

        for split in ["train", "val", "test"]:
            writers[split].close()
            _log(
                f"subset={subset_name} split={split}: done (shards={writers[split].shard_index}, segments_written={writers[split].total_written})"
            )

    _log(f"dropped segments log: {dropped_csv}")
    _log("04_preprocess_cache_shards: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
