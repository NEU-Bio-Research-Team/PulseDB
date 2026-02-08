from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from pulsedb_subset.pipeline.config import load_config
from pulsedb_subset.pipeline.eda_labels import read_vector
from pulsedb_subset.pipeline.io_mat import H5MatReader, is_hdf5_mat, load_json
from pulsedb_subset.pipeline.plotting import save_example_segment
from pulsedb_subset.pipeline.qc import compute_qc_metrics
from pulsedb_subset.pipeline.waveform import enforce_nct, infer_segment_axis, select_channel_indices


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def main() -> int:
    _log("03_eda_waveform_qc: start")
    cfg = load_config()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    # Hard-lock: this pipeline run is PPG-only.
    if cfg.channels != ["PPG"]:
        raise RuntimeError(f"PPG-only pipeline expects channels=['PPG'], got channels={cfg.channels}")
    if "PPG" not in cfg.channel_indices:
        raise RuntimeError("Missing channel_indices.PPG in config.yaml")
    ppg_raw_idx = int(cfg.channel_indices["PPG"])
    if ppg_raw_idx != 1:
        raise RuntimeError(f"Expected PPG raw channel index=1, got {ppg_raw_idx}. Re-check debug plots.")
    _log(f"QC waveform on PPG raw channel index={ppg_raw_idx}")

    _log(f"qc_sample_per_subset={cfg.eda_qc_sample_per_subset}")

    schema_path = cfg.artifacts_dir / "schema_map.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing {schema_path}. Run scripts/01_inventory_schema.py first.")
    schemas = load_json(schema_path)
    _log(f"schema_map loaded: {len(schemas)} subsets")

    out_rows: list[dict] = []
    examples_dir = cfg.artifacts_dir / "plots" / "waveform_qc"

    n_channels_target = len(cfg.channels)
    total_target = int(cfg.eda_qc_sample_per_subset)

    for subset_name, filename in cfg.subset_files.items():
        mat_path = (cfg.subset_mat_dir / filename).resolve()

        if not is_hdf5_mat(mat_path):
            print(
                f"[WARN] {subset_name}: '{mat_path.name}' is not MATLAB v7.3 (HDF5). "
                "Waveform QC requires v7.3/HDF5 for streaming reads; skipping."
            )
            continue

        ds_map = (schemas.get(subset_name, {}) or {}).get("dataset_map", {})
        signals_path = ds_map.get("signals")
        sbp_path = ds_map.get("sbp")

        if not signals_path:
            print(f"[WARN] {subset_name}: cannot find signals dataset; skipping waveform QC")
            continue

        _log(f"subset={subset_name}: signals_path={signals_path}")

        # Read SBP for stratified sampling
        sbp = read_vector(mat_path, sbp_path)
        if sbp.size == 0:
            idx_all = np.arange(0, total_target, dtype=np.int64)
        else:
            sbp = sbp.astype(np.float32, copy=False)
            idx_all = np.arange(sbp.size, dtype=np.int64)

        rng = np.random.default_rng(cfg.eda_random_seed)

        if sbp.size:
            bins = np.asarray(cfg.eda_sbp_bins, dtype=np.float32)
            bin_id = np.digitize(sbp, bins=bins, right=False)
            selected: list[int] = []
            per_bin = max(1, total_target // max(1, (bins.size + 1)))
            for b in np.unique(bin_id[np.isfinite(sbp)]):
                candidates = idx_all[(bin_id == b) & np.isfinite(sbp)]
                if candidates.size == 0:
                    continue
                take = min(per_bin, candidates.size)
                selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
            if len(selected) < total_target:
                # top-up uniformly
                remain = np.setdiff1d(idx_all, np.array(selected, dtype=np.int64), assume_unique=False)
                if remain.size:
                    extra = min(total_target - len(selected), remain.size)
                    selected.extend(rng.choice(remain, size=extra, replace=False).tolist())
            sample_idx = np.array(selected[:total_target], dtype=np.int64)
        else:
            # no sbp: uniform sample
            if idx_all.size <= total_target:
                sample_idx = idx_all
            else:
                sample_idx = rng.choice(idx_all, size=total_target, replace=False)

        sample_idx.sort()

        _log(f"subset={subset_name}: sampling n={int(sample_idx.size)}")
        before = len(out_rows)

        with H5MatReader(mat_path) as reader:
            sig_shape = reader.shape(signals_path)
            seg_axis = infer_segment_axis(sig_shape, n_segments_hint=int(sbp.size) if sbp.size else None)
            _log(f"subset={subset_name}: signals_shape={sig_shape}, segment_axis={seg_axis}")

            _log("WARNING: snr_variance_proxy is a heuristic quality proxy, not physiological SNR")

            # Example plotting: track best/worst segments by snr_variance_proxy on the selected PPG channel
            best = None
            worst = None

            for idx in tqdm(sample_idx, desc=f"QC {subset_name}"):
                seg = np.asarray(reader.read_index_axis(signals_path, seg_axis, int(idx)))
                # seg is typically (T, C) or (C, T); add batch dim -> (T,C,1) etc not needed.
                # Normalize to a 3D block then enforce.
                if seg.ndim == 1:
                    # single-channel vector
                    seg3 = seg[:, None, None]
                elif seg.ndim == 2:
                    # (T,C) or (C,T)
                    if seg.shape[0] == cfg.fs * cfg.segment_sec:
                        seg3 = seg[:, :, None]
                    elif seg.shape[1] == cfg.fs * cfg.segment_sec:
                        seg3 = np.transpose(seg, (1, 0))[:, :, None]
                    else:
                        continue
                else:
                    continue

                try:
                    x_nct = enforce_nct(seg3, segment_len=cfg.fs * cfg.segment_sec)
                    x_nct = select_channel_indices(x_nct, cfg.channels, cfg.channel_indices)
                except Exception:
                    continue

                x_nct = x_nct.astype(np.float32, copy=False)

                if x_nct.shape[1] != 1:
                    raise RuntimeError(f"PPG-only pipeline expected C=1 after channel selection, got C={x_nct.shape[1]}")

                m = compute_qc_metrics(x_nct[0, 0])
                out_rows.append(
                    {
                        "subset_name": subset_name,
                        "index": int(idx),
                        "channel": int(ppg_raw_idx),
                        "channel_name": "PPG",
                        "nan_ratio": m.nan_ratio,
                        "flatline_ratio": m.flatline_ratio,
                        "saturation_ratio": m.saturation_ratio,
                        "snr_variance_proxy": m.snr_variance_proxy,
                    }
                )

                if best is None or m.snr_variance_proxy > best[0]:
                    best = (m.snr_variance_proxy, x_nct[0, 0].copy())
                if worst is None or m.snr_variance_proxy < worst[0]:
                    worst = (m.snr_variance_proxy, x_nct[0, 0].copy())

            if best is not None:
                save_example_segment(
                    best[1],
                    f"{subset_name} example best (PPG raw_ch={ppg_raw_idx}, snr_variance_proxy={best[0]:.2f})",
                    examples_dir / f"{subset_name}_best_PPG_ch{ppg_raw_idx}.png",
                    fs=cfg.fs,
                )
            if worst is not None:
                save_example_segment(
                    worst[1],
                    f"{subset_name} example worst (PPG raw_ch={ppg_raw_idx}, snr_variance_proxy={worst[0]:.2f})",
                    examples_dir / f"{subset_name}_worst_PPG_ch{ppg_raw_idx}.png",
                    fs=cfg.fs,
                )

        after = len(out_rows)
        _log(f"subset={subset_name}: done (rows_added={after - before})")

    qc_df = pd.DataFrame(out_rows)

    out_csv = cfg.artifacts_dir / "qc_metrics.csv"
    qc_df.to_csv(out_csv, index=False)
    _log(f"wrote: {out_csv} (rows={len(qc_df)})")

    try:
        out_parquet = cfg.artifacts_dir / "qc_metrics.parquet"
        qc_df.to_parquet(out_parquet, index=False)
    except Exception:
        out_parquet = None

    # Report
    report_path = cfg.reports_dir / "eda_waveform_qc.md"
    md = []
    md.append("# EDA level-2: waveform QC (sampled)\n")
    md.append("## Artifacts\n")
    md.append(f"- qc metrics: {out_csv.as_posix()}")
    if out_parquet is not None:
        md.append(f"- qc metrics (parquet): {out_parquet.as_posix()}")
    md.append(f"- examples dir: {examples_dir.as_posix()}\n")

    if not qc_df.empty:
        md.append("## Summary (per subset, channel)\n")
        summ = (
            qc_df.groupby(["subset_name", "channel", "channel_name"])
            .agg(
                n=("snr_variance_proxy", "size"),
                nan_ratio_mean=("nan_ratio", "mean"),
                flatline_mean=("flatline_ratio", "mean"),
                saturation_mean=("saturation_ratio", "mean"),
                snr_p50=("snr_variance_proxy", lambda x: float(np.nanpercentile(x, 50))),
            )
            .reset_index()
        )
        md.append(summ.to_markdown(index=False))

    report_path.write_text("\n\n".join(md) + "\n", encoding="utf-8")
    _log(f"wrote: {report_path}")
    _log("03_eda_waveform_qc: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
