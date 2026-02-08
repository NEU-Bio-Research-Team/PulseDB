from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pulsedb_subset.pipeline.config import load_config
from pulsedb_subset.pipeline.io_mat import H5MatReader
from pulsedb_subset.pipeline.io_mat import is_hdf5_mat
from pulsedb_subset.pipeline.waveform import enforce_nct, infer_segment_axis


def main() -> int:
    """Offline helper: plot a few segments for each raw channel to identify PPG/ECG/ABP.

    Usage:
      python pulsedb_subset/scripts/debug_plot_channels.py

    Then set pulsedb_subset/config.yaml:
      channel_indices:
        PPG: <index>
    """

    cfg = load_config()

    subset_name = cfg.scaling_fit_subset
    if subset_name not in cfg.subset_files:
        raise ValueError(f"Unknown subset '{subset_name}'")

    mat_path = (cfg.subset_mat_dir / cfg.subset_files[subset_name]).resolve()
    if not is_hdf5_mat(mat_path):
        raise RuntimeError("This helper expects MATLAB v7.3/HDF5 .mat files")

    from pulsedb_subset.pipeline.io_mat import load_json

    schema_path = cfg.artifacts_dir / "schema_map.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing {schema_path}. Run scripts/01_inventory_schema.py first.")

    schemas = load_json(schema_path)
    ds_map = (schemas.get(subset_name, {}) or {}).get("dataset_map", {})
    signals_path = ds_map.get("signals")
    if not signals_path:
        raise ValueError(f"Missing signals path for subset={subset_name}")

    seg_len = cfg.fs * cfg.segment_sec

    out_dir = (cfg.artifacts_dir / "plots" / "debug_channels").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with H5MatReader(mat_path) as reader:
        sig_shape = reader.shape(signals_path)
        seg_axis = infer_segment_axis(sig_shape)
        n = int(sig_shape[seg_axis])

        # pick a few evenly spaced indices
        idxs = np.linspace(0, max(0, n - 1), num=3, dtype=int)
        print(f"subset={subset_name}")
        print(f"signals_path={signals_path}")
        print(f"signals_shape={sig_shape} segment_axis={seg_axis} seg_len={seg_len}")
        print(f"plot indices={idxs.tolist()}")
        print(f"saving PNGs to: {out_dir}")

        for idx in idxs:
            seg = np.asarray(reader.read_index_axis(signals_path, seg_axis, int(idx)))
            if seg.ndim != 2:
                print(f"skip idx={idx}: seg.ndim={seg.ndim} shape={seg.shape}")
                continue

            # normalize to (T,C,1) then enforce to (1,C,T)
            if seg.shape[0] == seg_len:
                seg3 = seg[:, :, None]
            elif seg.shape[1] == seg_len:
                seg3 = np.transpose(seg, (1, 0))[:, :, None]
            else:
                print(f"skip idx={idx}: cannot locate time axis (shape={seg.shape})")
                continue

            x_nct = enforce_nct(seg3, segment_len=seg_len)  # (1,C,T)
            C = x_nct.shape[1]

            plt.figure(figsize=(12, 2.5 * C))
            for ch in range(C):
                ax = plt.subplot(C, 1, ch + 1)
                ax.plot(x_nct[0, ch], linewidth=1.0)
                ax.set_title(f"idx={idx} channel={ch}")
            plt.tight_layout()
            out_path = out_dir / f"{subset_name}_idx{idx}.png"
            plt.savefig(out_path, dpi=160)
            plt.close()
            print(f"wrote {out_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
