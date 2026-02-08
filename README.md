# PulseDB (subset pipeline)

This workspace contains a refactored, paper-friendly EDA + preprocessing pipeline for **PulseDB subset .mat files**.

Design goals:

- Run on limited RAM/SSD by doing **schema discovery + proxy EDA first** (labels/demographics), then **waveform QC on a controlled sample**, then **streaming shard cache**.
- Avoid data leakage: **fit scalers on Train only** and keep evaluation split by protocol.

## Data

Put the official subset files in `archive/` (already present in this repo):

- `VitalDB_Train_Subset.mat`
- `VitalDB_CalBased_Test_Subset.mat`
- `VitalDB_CalFree_Test_Subset.mat`
- `VitalDB_AAMI_Test_Subset.mat`
- `VitalDB_AAMI_Cal_Subset.mat`

## Configure

Edit `pulsedb_subset/config.yaml`:

- `channels`: `[PPG]` (default) or `[ECG, PPG]`
- QC thresholds in `qc:`
- `io.chunk_size` / `io.shard_size`

## Install deps

```bash
pip install -r pulsedb_subset/requirements.txt
```

## Run

```bash
python pulsedb_subset/scripts/01_inventory_schema.py
python pulsedb_subset/scripts/02_eda_labels_demographics.py
python pulsedb_subset/scripts/03_eda_waveform_qc.py
python pulsedb_subset/scripts/04_preprocess_cache_shards.py
python pulsedb_subset/scripts/05_export_paper_artifacts.py
```

Notes:

- If your subset `.mat` files are **not** MATLAB v7.3/HDF5 (e.g. classic MAT v5), the pipeline can still run:
	- `01_inventory_schema.py` (schema + guessed variable names)
	- `02_eda_labels_demographics.py` (labels/demographics; reads only requested vectors)
- But waveform-heavy steps (`03_eda_waveform_qc.py`, `04_preprocess_cache_shards.py`) require **v7.3/HDF5** for streaming reads.

Outputs:

- `pulsedb_subset/artifacts/` (CSV/Parquet/JSON, scalers, dropped segments)
- `pulsedb_subset/reports/` (Markdown + plots)
- `pulsedb_subset/cache/pulsedb_subset/<subset>/shard_####.npz`