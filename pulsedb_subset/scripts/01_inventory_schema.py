from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from pulsedb_subset.pipeline.config import load_config
from pulsedb_subset.pipeline.io_mat import save_json
from pulsedb_subset.pipeline.schema import discover_schema, schema_to_manifest_row


def main() -> int:
    cfg = load_config()

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    schemas: dict[str, dict] = {}
    rows = []

    for subset_name, filename in cfg.subset_files.items():
        mat_path = (cfg.subset_mat_dir / filename).resolve()
        schema = discover_schema(subset_name, mat_path)
        schemas[subset_name] = {
            "subset_name": schema.subset_name,
            "mat_path": schema.mat_path,
            "is_hdf5": schema.is_hdf5,
            "dataset_map": schema.dataset_map,
            "nodes": schema.nodes,
        }
        rows.append(schema_to_manifest_row(schema, segment_len=cfg.fs * cfg.segment_sec))

    any_non_hdf5 = any(not s.get("is_hdf5", False) for s in schemas.values())

    manifest_df = pd.DataFrame(rows).sort_values("subset_name")

    manifest_csv = cfg.artifacts_dir / "subset_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    # Optional parquet (requires pyarrow)
    try:
        manifest_parquet = cfg.artifacts_dir / "subset_manifest.parquet"
        manifest_df.to_parquet(manifest_parquet, index=False)
    except Exception:
        pass

    schema_json = cfg.artifacts_dir / "schema_map.json"
    save_json(schema_json, schemas)

    print(f"Wrote: {manifest_csv}")
    print(f"Wrote: {schema_json}")

    if any_non_hdf5:
        print(
            "\n[NOTE] Detected non-HDF5 MATLAB .mat files (not v7.3). "
            "This is OK for schema + label/demographic EDA, but waveform streaming steps "
            "(03_eda_waveform_qc.py, 04_preprocess_cache_shards.py) require v7.3/HDF5." 
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
