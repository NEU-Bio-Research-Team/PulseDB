from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class NodeInfo:
    path: str
    kind: str  # 'group' | 'dataset'
    shape: tuple[int, ...] | None = None
    dtype: str | None = None


def scan_v5_mat_vars(mat_path: Path, max_vars: int = 5000) -> list[NodeInfo]:
    """Scan top-level variables in a non-v7.3 MATLAB .mat.

    Uses scipy.io.whosmat, which reads metadata without loading arrays.
    Returned NodeInfo paths use a pseudo-HDF5 style: '/<varname>'.
    """
    import scipy.io as sio

    vars_meta = sio.whosmat(mat_path)
    out: list[NodeInfo] = []
    for name, shape, matlab_class in vars_meta[:max_vars]:
        # MATLAB v5 doesn't expose a hierarchical tree; represent as flat datasets.
        out.append(
            NodeInfo(
                path=f"/{name}",
                kind="dataset",
                shape=tuple(int(x) for x in shape),
                dtype=str(matlab_class),
            )
        )
    out.sort(key=lambda x: x.path)
    return out


def is_hdf5_mat(mat_path: Path) -> bool:
    """Return True if the .mat is MATLAB v7.3 (HDF5).

    Prefer a fast signature check, but fall back to attempting to open with h5py
    to avoid false negatives for some files.
    """
    try:
        with mat_path.open("rb") as f:
            sig = f.read(8)
        if sig == b"\x89HDF\r\n\x1a\n":
            return True
    except Exception:
        # If we can't read the file header, treat as non-HDF5.
        return False

    # Fallback: attempt open as HDF5
    try:
        import h5py

        with h5py.File(mat_path, "r"):
            return True
    except Exception:
        return False


def scan_hdf5_tree(mat_path: Path, max_nodes: int = 2000) -> list[NodeInfo]:
    import h5py

    out: list[NodeInfo] = []

    def _visit(name: str, obj: Any) -> None:
        if len(out) >= max_nodes:
            return
        if isinstance(obj, h5py.Dataset):
            shape = tuple(int(x) for x in obj.shape)
            dtype = str(obj.dtype)
            out.append(NodeInfo(path=f"/{name}", kind="dataset", shape=shape, dtype=dtype))
        else:
            out.append(NodeInfo(path=f"/{name}", kind="group"))

    with h5py.File(mat_path, "r") as f:
        f.visititems(_visit)

    out.sort(key=lambda x: x.path)
    return out


def scan_hdf5_group_children(mat_path: Path, group_path: str = "/Subset", max_children: int = 5000) -> list[NodeInfo]:
    """Scan a specific HDF5 group and its immediate children.

    This avoids traversing huge reference trees (e.g. '/#refs#') which can cause
    truncated scans when using a global max_nodes budget.
    """
    import h5py

    out: list[NodeInfo] = []
    gp = group_path if group_path.startswith("/") else f"/{group_path}"

    with h5py.File(mat_path, "r") as f:
        # record top-level children (groups/datasets) shallowly
        for name, obj in f.items():
            p = f"/{name}"
            if isinstance(obj, h5py.Dataset):
                out.append(NodeInfo(path=p, kind="dataset", shape=tuple(int(x) for x in obj.shape), dtype=str(obj.dtype)))
            else:
                out.append(NodeInfo(path=p, kind="group"))

        if gp not in f:
            out.sort(key=lambda x: x.path)
            return out

        g = f[gp]
        if isinstance(g, h5py.Dataset):
            out.append(NodeInfo(path=gp, kind="dataset", shape=tuple(int(x) for x in g.shape), dtype=str(g.dtype)))
            out.sort(key=lambda x: x.path)
            return out

        # group node
        out.append(NodeInfo(path=gp, kind="group"))

        n = 0
        for child_name, child in g.items():
            if n >= max_children:
                break
            p = f"{gp.rstrip('/')}/{child_name}"
            if isinstance(child, h5py.Dataset):
                out.append(
                    NodeInfo(path=p, kind="dataset", shape=tuple(int(x) for x in child.shape), dtype=str(child.dtype))
                )
            else:
                out.append(NodeInfo(path=p, kind="group"))
            n += 1

    out.sort(key=lambda x: x.path)
    return out


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def guess_dataset_paths(nodes: Iterable[NodeInfo]) -> dict[str, str]:
    """Best-effort heuristics to locate common arrays in MATLAB v7.3 HDF5 trees.

    Returns a mapping: logical_name -> hdf5_path
    """
    datasets = [n for n in nodes if n.kind == "dataset"]

    # Prefer shallow paths and 'obvious' names.
    def score(path: str) -> tuple[int, int]:
        # fewer segments + shorter string is better
        depth = path.count("/")
        return (depth, len(path))

    by_norm: dict[str, list[str]] = {}
    for d in datasets:
        base = d.path.split("/")[-1]
        by_norm.setdefault(_normalize_name(base), []).append(d.path)

    def pick(*keys: str) -> str | None:
        candidates: list[str] = []
        for k in keys:
            candidates.extend(by_norm.get(_normalize_name(k), []))
        if not candidates:
            # fuzzy contains
            for d in datasets:
                dn = _normalize_name(d.path.split("/")[-1])
                for k in keys:
                    if _normalize_name(k) in dn:
                        candidates.append(d.path)
        if not candidates:
            return None
        candidates = sorted(set(candidates), key=score)
        return candidates[0]

    out: dict[str, str] = {}
    for logical, names in {
        "sbp": ("sbp", "ysbp", "labelsbp"),
        "dbp": ("dbp", "ydbp", "labeldbp"),
        "age": ("age",),
        "gender": ("gender", "sex"),
        "height": ("height", "ht"),
        "weight": ("weight", "wt"),
        "bmi": ("bmi",),
        "subject_id": ("subject", "subjectid", "subjid", "pid", "patientid"),
    }.items():
        p = pick(*names)
        if p:
            out[logical] = p

    # waveform signals are harder; guess using shape heuristics
    # Prefer datasets with ndim in {2,3} and large total size.
    def is_wave_candidate(d: NodeInfo) -> bool:
        if not d.shape:
            return False
        if len(d.shape) not in (2, 3):
            return False
        total = 1
        for s in d.shape:
            total *= max(1, int(s))
        return total >= 100_000  # heuristic

    wave_candidates = [d for d in datasets if is_wave_candidate(d)]

    # Prefer names containing signals/wave/ecg/ppg/abp/x
    name_boost = ["signals", "signal", "wave", "ecg", "ppg", "abp", "x"]

    def wave_score(d: NodeInfo) -> tuple[int, int, int]:
        base = _normalize_name(d.path.split("/")[-1])
        boost = 0
        for token in name_boost:
            if _normalize_name(token) in base:
                boost -= 1
        # bigger is better -> negative
        total = int(np.prod(d.shape)) if d.shape else 0
        return (boost, -total, len(d.path))

    if wave_candidates:
        wave_candidates.sort(key=wave_score)
        out["signals"] = wave_candidates[0].path

    return out


def read_hdf5_array(mat_path: Path, h5_path: str, start: int | None = None, stop: int | None = None) -> np.ndarray:
    """Read an array (optionally sliced along the first axis) from an HDF5-backed MAT.

    Supports numeric datasets directly. If the dataset is an object reference array
    (MATLAB cell/struct), this will try to dereference each element.
    """
    import h5py

    with h5py.File(mat_path, "r") as f:
        ds = f[h5_path]

        # Handle MATLAB references (cell arrays) best-effort
        if hasattr(ds.dtype, "metadata") and ds.dtype.metadata and "ref" in ds.dtype.metadata:
            refs = ds[start:stop] if (start is not None or stop is not None) else ds[()]
            refs = np.asarray(refs).ravel()
            items: list[np.ndarray] = []
            for ref in refs:
                if not isinstance(ref, h5py.Reference) or not ref:
                    continue
                arr = np.array(f[ref])
                items.append(arr)
            if not items:
                return np.empty((0,))
            return np.stack(items, axis=0)

        if start is None and stop is None:
            return np.array(ds)

        # slice on first axis
        sl = slice(start, stop)
        return np.array(ds[sl])


class H5MatReader:
    """Context-managed reader for MATLAB v7.3 (HDF5) .mat files."""

    def __init__(self, mat_path: Path):
        self.mat_path = mat_path
        self._h5 = None

    def __enter__(self) -> "H5MatReader":
        import h5py

        self._h5 = h5py.File(self.mat_path, "r")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def h5(self):
        if self._h5 is None:
            raise RuntimeError("H5MatReader is not open")
        return self._h5

    def dataset(self, h5_path: str):
        return self.h5[h5_path]

    def shape(self, h5_path: str) -> tuple[int, ...]:
        ds = self.dataset(h5_path)
        return tuple(int(x) for x in ds.shape)

    def read_slice(self, h5_path: str, start: int, stop: int) -> np.ndarray:
        ds = self.dataset(h5_path)
        return np.array(ds[slice(start, stop)])

    def read_index(self, h5_path: str, index: int) -> np.ndarray:
        import h5py

        ds = self.dataset(h5_path)
        # MATLAB cell arrays may be stored as ref datasets
        if hasattr(ds.dtype, "metadata") and ds.dtype.metadata and "ref" in ds.dtype.metadata:
            ref = ds[index]
            if isinstance(ref, h5py.Reference) and ref:
                return np.array(self.h5[ref])
            # sometimes ds[index] returns an array of refs
            refs = np.asarray(ref).ravel()
            items = []
            for r in refs:
                if isinstance(r, h5py.Reference) and r:
                    items.append(np.array(self.h5[r]))
            if not items:
                return np.empty((0,))
            return np.stack(items, axis=0)

        # ds[index] returns an array with remaining dims
        return np.array(ds[index])

    def read_slice_axis(self, h5_path: str, axis: int, start: int, stop: int) -> np.ndarray:
        """Slice along a chosen axis (0/1/2) for 3D datasets."""
        ds = self.dataset(h5_path)
        ax = int(axis)
        if ax < 0:
            ax = ds.ndim + ax
        sl = slice(int(start), int(stop))
        if ds.ndim == 1:
            if ax not in (0,):
                raise ValueError(f"Cannot slice axis={axis} for 1D dataset")
            return np.array(ds[sl])
        if ds.ndim == 2:
            if ax == 0:
                return np.array(ds[sl, :])
            if ax == 1:
                return np.array(ds[:, sl])
            raise ValueError(f"Cannot slice axis={axis} for 2D dataset")
        if ds.ndim != 3:
            raise ValueError(f"Unsupported dataset ndim={ds.ndim} for axis slicing")

        if ax == 0:
            return np.array(ds[sl, :, :])
        if ax == 1:
            return np.array(ds[:, sl, :])
        if ax == 2:
            return np.array(ds[:, :, sl])
        raise ValueError(f"Invalid axis={axis} for 3D dataset")

    def read_index_axis(self, h5_path: str, axis: int, index: int) -> np.ndarray:
        """Index along a chosen axis (0/1/2) for 3D datasets."""
        ds = self.dataset(h5_path)
        ax = int(axis)
        if ax < 0:
            ax = ds.ndim + ax
        idx = int(index)
        if ds.ndim == 1:
            if ax not in (0,):
                raise ValueError(f"Cannot index axis={axis} for 1D dataset")
            return np.array(ds[idx])
        if ds.ndim == 2:
            if ax == 0:
                return np.array(ds[idx, :])
            if ax == 1:
                return np.array(ds[:, idx])
            raise ValueError(f"Cannot index axis={axis} for 2D dataset")
        if ds.ndim != 3:
            raise ValueError(f"Unsupported dataset ndim={ds.ndim} for axis indexing")

        if ax == 0:
            return np.array(ds[idx, :, :])
        if ax == 1:
            return np.array(ds[:, idx, :])
        if ax == 2:
            return np.array(ds[:, :, idx])
        raise ValueError(f"Invalid axis={axis} for 3D dataset")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
