from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from pulsedb_subset.pipeline.config import load_config
from pulsedb_subset.pipeline.eda_labels import decode_subject_id, read_vector, summarize_labels_and_demo
from pulsedb_subset.pipeline.io_mat import load_json, save_json
from pulsedb_subset.pipeline.plotting import save_histogram
from pulsedb_subset.pipeline.schema import discover_schema


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _ensure_schema(cfg) -> dict:
    schema_path = cfg.artifacts_dir / "schema_map.json"
    if schema_path.exists():
        return load_json(schema_path)

    schemas = {}
    for subset_name, filename in cfg.subset_files.items():
        mat_path = (cfg.subset_mat_dir / filename).resolve()
        s = discover_schema(subset_name, mat_path)
        schemas[subset_name] = {
            "subset_name": s.subset_name,
            "mat_path": s.mat_path,
            "is_hdf5": s.is_hdf5,
            "dataset_map": s.dataset_map,
            "nodes": s.nodes,
        }
    save_json(schema_path, schemas)
    return schemas


def main() -> int:
    _log("02_eda_labels_demographics: start")
    cfg = load_config()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    _log(f"artifacts_dir={cfg.artifacts_dir}")
    _log(f"reports_dir={cfg.reports_dir}")

    schemas = _ensure_schema(cfg)
    _log(f"schema_map loaded: {len(schemas)} subsets")

    stats_all = []
    missing_all = []
    subject_stats_all = []
    subject_sets: dict[str, set] = {}

    def _subject_set(arr: np.ndarray) -> set:
        arr = np.asarray(arr).reshape((-1,))
        if arr.size == 0:
            return set()
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr[np.isfinite(arr)]
        try:
            return set(np.unique(arr.astype(np.int64, copy=False)))
        except Exception:
            return set(np.unique(arr.astype(str)))

    plots_dir = cfg.artifacts_dir / "plots" / "labels"

    for subset_name, filename in cfg.subset_files.items():
        _log(f"subset={subset_name}: read labels/demographics")
        mat_path = (cfg.subset_mat_dir / filename).resolve()
        ds_map = (schemas.get(subset_name, {}) or {}).get("dataset_map", {})

        sbp = read_vector(mat_path, ds_map.get("sbp"))
        dbp = read_vector(mat_path, ds_map.get("dbp"))

        demo = {
            "age": read_vector(mat_path, ds_map.get("age")),
            "gender": read_vector(mat_path, ds_map.get("gender")),
            "height": read_vector(mat_path, ds_map.get("height")),
            "weight": read_vector(mat_path, ds_map.get("weight")),
            "bmi": read_vector(mat_path, ds_map.get("bmi")),
        }

        subject_id_raw = read_vector(mat_path, ds_map.get("subject_id"))
        expected_n = int(max(sbp.size, dbp.size))
        subject_id = (
            decode_subject_id(subject_id_raw, expected_n=expected_n) if subject_id_raw.size and expected_n > 0 else subject_id_raw
        )
        subject_sets[subset_name] = _subject_set(subject_id)

        stats_df, missing_df, subj_df = summarize_labels_and_demo(
            subset_name=subset_name,
            sbp=sbp,
            dbp=dbp,
            demo=demo,
            subject_id=subject_id if subject_id.size else None,
        )

        stats_all.append(stats_df)
        missing_all.append(missing_df)
        if not subj_df.empty:
            subj_df.insert(0, "subset_name", subset_name)
            subject_stats_all.append(subj_df)

        # Plots
        save_histogram(sbp, f"{subset_name} SBP", plots_dir / f"{subset_name}_sbp_hist.png", xlabel="SBP")
        save_histogram(dbp, f"{subset_name} DBP", plots_dir / f"{subset_name}_dbp_hist.png", xlabel="DBP")

        nseg = int(stats_df.loc[0, "n_segments"]) if (not stats_df.empty and "n_segments" in stats_df.columns) else 0
        _log(f"subset={subset_name}: done (n_segments≈{nseg})")

    stats_by_subset = pd.concat(stats_all, ignore_index=True) if stats_all else pd.DataFrame()
    missingness_by_subset = pd.concat(missing_all, ignore_index=True) if missing_all else pd.DataFrame()
    subject_stats = pd.concat(subject_stats_all, ignore_index=True) if subject_stats_all else pd.DataFrame()

    stats_csv = cfg.artifacts_dir / "stats_by_subset.csv"
    miss_csv = cfg.artifacts_dir / "missingness_by_subset.csv"
    stats_by_subset.to_csv(stats_csv, index=False)
    missingness_by_subset.to_csv(miss_csv, index=False)

    _log(f"wrote: {stats_csv}")
    _log(f"wrote: {miss_csv}")

    if not subject_stats.empty:
        subject_stats.to_csv(cfg.artifacts_dir / "subject_stats_by_subset.csv", index=False)

    # Protocol sanity checks
    def inter(a: str, b: str) -> int:
        return len(subject_sets.get(a, set()) & subject_sets.get(b, set()))

    protocol_lines = []
    if subject_sets.get("Train") and subject_sets.get("CalFree_Test"):
        protocol_lines.append(f"- Train ∩ CalFree_Test subjects: {inter('Train','CalFree_Test')}")
    if subject_sets.get("Train") and subject_sets.get("CalBased_Test"):
        protocol_lines.append(f"- Train ∩ CalBased_Test subjects: {inter('Train','CalBased_Test')}")

    aami_lines = []
    if not subject_stats.empty:
        aami_test = subject_stats[subject_stats["subset_name"] == "AAMI_Test"]
        if not aami_test.empty:
            n_subj = int(aami_test["subject_id"].nunique())
            min_seg = float(aami_test["n_segments"].min())
            aami_lines.append(f"- AAMI_Test: n_subjects={n_subj}, min_segments_per_subject={min_seg:.0f}")

    # Report markdown
    report_path = cfg.reports_dir / "eda_labels_demographics.md"
    md = []
    md.append("# EDA level-1: labels + demographics (proxy)\n")
    md.append("## Artifacts\n")
    md.append(f"- stats: {stats_csv.as_posix()}")
    md.append(f"- missingness: {miss_csv.as_posix()}")
    md.append(f"- plots dir: {plots_dir.as_posix()}\n")

    md.append("## Protocol sanity checks\n")
    md.extend(protocol_lines or ["- subject_id not found or empty; overlap checks skipped."])
    md.append("\n## AAMI quick checks\n")
    md.extend(aami_lines or ["- AAMI subject-level checks skipped (missing subject_id)."])

    md.append("\n## Plots\n")
    for subset_name in cfg.subset_files.keys():
        md.append(f"- {subset_name}: { (plots_dir / f'{subset_name}_sbp_hist.png').as_posix() }")
        md.append(f"- {subset_name}: { (plots_dir / f'{subset_name}_dbp_hist.png').as_posix() }")

    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    _log(f"wrote: {report_path}")
    _log("02_eda_labels_demographics: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
