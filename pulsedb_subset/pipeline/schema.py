from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_mat import (
    NodeInfo,
    guess_dataset_paths,
    is_hdf5_mat,
    scan_hdf5_group_children,
    scan_hdf5_tree,
    scan_v5_mat_vars,
)


@dataclass(frozen=True)
class SubsetSchema:
    subset_name: str
    mat_path: str
    is_hdf5: bool
    nodes: list[dict[str, Any]]
    dataset_map: dict[str, str]


def discover_schema(subset_name: str, mat_path: Path) -> SubsetSchema:
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing subset file: {mat_path}")

    if not is_hdf5_mat(mat_path):
        # For non-v7.3 MAT, scan top-level variables using whosmat (metadata-only)
        # and apply the same guessing heuristics as for HDF5.
        nodes = scan_v5_mat_vars(mat_path)
        dataset_map = guess_dataset_paths(nodes)
        return SubsetSchema(
            subset_name=subset_name,
            mat_path=str(mat_path),
            is_hdf5=False,
            nodes=[n.__dict__ for n in nodes],
            dataset_map=dataset_map,
        )

    # Prefer scanning the well-known '/Subset' group (fast, avoids huge '/#refs#' trees)
    nodes = scan_hdf5_group_children(mat_path, group_path="/Subset")
    dataset_map = guess_dataset_paths(nodes)

    # If we still didn't find signals, fall back to a broader (but capped) scan.
    if "signals" not in dataset_map:
        nodes = scan_hdf5_tree(mat_path)
        dataset_map = guess_dataset_paths(nodes)

    # Some demographics (height/weight) may be stored under refs; probe common paths.
    if dataset_map.get("height") is None or dataset_map.get("weight") is None:
        try:
            import h5py

            with h5py.File(mat_path, "r") as f:
                def _pick(cands: list[str]) -> str | None:
                    for p in cands:
                        pp = p if p.startswith("/") else f"/{p}"
                        if pp in f and hasattr(f[pp], "shape"):
                            return pp
                    return None

                if "height" not in dataset_map:
                    h = _pick([
                        "/Subset/Height",
                        "/#refs#/ht",
                        "/#refs#/HT",
                        "/#refs#/Ht",
                        "/#refs#/#a/ht",
                        "/#refs#/#a/HT",
                        "/#refs#/#a/Ht",
                        "/#refs#/#a/0ht",
                        "/#refs#/#a/0HT",
                    ])
                    if h:
                        dataset_map["height"] = h

                if "weight" not in dataset_map:
                    w = _pick([
                        "/Subset/Weight",
                        "/#refs#/wt",
                        "/#refs#/WT",
                        "/#refs#/Wt",
                        "/#refs#/#a/wt",
                        "/#refs#/#a/WT",
                        "/#refs#/#a/Wt",
                        "/#refs#/#a/0wt",
                        "/#refs#/#a/0WT",
                    ])
                    if w:
                        dataset_map["weight"] = w
        except Exception:
            pass

    return SubsetSchema(
        subset_name=subset_name,
        mat_path=str(mat_path),
        is_hdf5=True,
        nodes=[n.__dict__ for n in nodes],
        dataset_map=dataset_map,
    )


def schema_to_manifest_row(schema: SubsetSchema, segment_len: int | None = None) -> dict[str, Any]:
    # best-effort summary for quick EDA planning
    row: dict[str, Any] = {
        "subset_name": schema.subset_name,
        "mat_path": schema.mat_path,
        "is_hdf5": schema.is_hdf5,
        "signals_path": schema.dataset_map.get("signals", ""),
        "sbp_path": schema.dataset_map.get("sbp", ""),
        "dbp_path": schema.dataset_map.get("dbp", ""),
        "subject_id_path": schema.dataset_map.get("subject_id", ""),
        "age_path": schema.dataset_map.get("age", ""),
        "gender_path": schema.dataset_map.get("gender", ""),
        "height_path": schema.dataset_map.get("height", ""),
        "weight_path": schema.dataset_map.get("weight", ""),
        "bmi_path": schema.dataset_map.get("bmi", ""),
        "signals_ndim": "",
        "signals_dim0": "",
        "signals_dim1": "",
        "signals_dim2": "",
        "n_segments": "",
        "n_channels": "",
        "segment_len": "",
    }

    # Extract shape info for the guessed signals dataset
    if schema.is_hdf5 and row["signals_path"]:
        for n in schema.nodes:
            if n.get("path") == row["signals_path"] and n.get("kind") == "dataset":
                shape = n.get("shape")
                if shape:
                    dims = [int(x) for x in shape]
                    row["signals_shape"] = "x".join(str(x) for x in dims)

                    row["signals_ndim"] = int(len(dims))
                    if len(dims) >= 1:
                        row["signals_dim0"] = int(dims[0])
                    if len(dims) >= 2:
                        row["signals_dim1"] = int(dims[1])
                    if len(dims) >= 3:
                        row["signals_dim2"] = int(dims[2])

                    # Deterministic convention using known segment length when provided.
                    # Raw is expected (T, C, N) in these subset files.
                    if segment_len is not None and len(dims) == 3:
                        T = int(segment_len)
                        if dims[0] == T:
                            row["segment_len"] = int(dims[0])
                            row["n_channels"] = int(dims[1])
                            row["n_segments"] = int(dims[2])
                        elif dims[2] == T:
                            # already (N, C, T)
                            row["n_segments"] = int(dims[0])
                            row["n_channels"] = int(dims[1])
                            row["segment_len"] = int(dims[2])
                row["signals_dtype"] = n.get("dtype", "")
                break

    return row
