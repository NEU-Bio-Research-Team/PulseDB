from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .io_mat import is_hdf5_mat, read_hdf5_array


def _as_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim == 0:
        return x.reshape((1,))
    if x.ndim > 1:
        return x.reshape((-1,))
    return x


def read_vector(mat_path: Path, h5_path: str | None) -> np.ndarray:
    if not h5_path:
        return np.array([])

    if is_hdf5_mat(mat_path):
        arr = read_hdf5_array(mat_path, h5_path)
        return _as_1d(arr)

    # fallback: loadmat
    import scipy.io as sio

    key = h5_path.strip("/").split("/")[-1]

    # IMPORTANT: do NOT load the full .mat (waveforms can be >10GB).
    # variable_names loads only requested arrays plus MATLAB metadata.
    try:
        mat = sio.loadmat(mat_path, variable_names=[key], squeeze_me=True, struct_as_record=False)
    except TypeError:
        # older SciPy compatibility
        mat = sio.loadmat(mat_path, variable_names=[key])

    if key not in mat:
        return np.array([])
    return _as_1d(np.asarray(mat[key]))


_DIGITS_RE = re.compile(r"(\d+)")


def decode_subject_id(raw: np.ndarray, expected_n: int | None = None) -> np.ndarray:
    """Decode VitalDB subject identifiers into a 1D int64 array.

    VitalDB v7.3 MAT files sometimes store subject IDs as MATLAB strings/cell arrays,
    which can surface via `read_vector()` as flattened character-code arrays.

    This function:
    - returns an array of shape (N,) dtype int64
    - when `expected_n` is provided, it validates alignment and will attempt to
      reshape+decode flattened character codes (e.g., length = N * L).
    """

    x = np.asarray(raw)
    if x.size == 0:
        return np.array([], dtype=np.int64)

    x = np.squeeze(x)
    if x.ndim == 0:
        x = x.reshape((1,))

    def _finalize_numeric(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec).reshape((-1,))
        if vec.size == 0:
            return np.array([], dtype=np.int64)
        if np.issubdtype(vec.dtype, np.floating):
            out = np.full(vec.shape, -1, dtype=np.int64)
            ok = np.isfinite(vec)
            if np.any(ok):
                out[ok] = vec[ok].astype(np.int64)
            return out
        return vec.astype(np.int64, copy=False)

    def _decode_codes_matrix(codes: np.ndarray) -> tuple[np.ndarray, int]:
        # codes shape: (N, L)
        codes = np.asarray(codes)
        if codes.ndim != 2:
            codes = codes.reshape((codes.shape[0], -1))

        # Replace non-finite with 0 (NUL)
        if np.issubdtype(codes.dtype, np.floating):
            tmp = np.zeros_like(codes, dtype=np.float64)
            ok = np.isfinite(codes)
            tmp[ok] = codes[ok]
            codes = tmp

        codes_u16 = np.asarray(np.rint(codes), dtype=np.uint16)
        n, l = codes_u16.shape

        # Decode whole matrix as UTF-16LE, then slice per-row.
        big = codes_u16.astype("<u2", copy=False).tobytes().decode("utf-16le", errors="ignore")

        out = np.full((n,), -1, dtype=np.int64)
        invalid = 0
        for i in range(n):
            s = big[i * l : (i + 1) * l].strip().replace("\x00", "")
            m = _DIGITS_RE.search(s)
            if m is None:
                invalid += 1
                continue
            out[i] = int(m.group(1))
        return out, invalid

    # Case 1: plain numeric vector already
    if x.dtype != object and np.issubdtype(x.dtype, np.number):
        flat = np.asarray(x).reshape((-1,))
        if expected_n is not None and flat.size != expected_n:
            # Heuristic: flattened char-code matrix of length N*L
            if flat.size % expected_n == 0:
                l = int(flat.size // expected_n)
                if l > 1:
                    # Most common: codes are (N, L)
                    codes_a = flat.reshape((expected_n, l))
                    ids_a, bad_a = _decode_codes_matrix(codes_a)

                    # Alternate: original codes were (L, N) but flatten/reshape order differed.
                    # Reshape to (L, N) then transpose back to (N, L) before decoding.
                    codes_b = flat.reshape((l, expected_n)).T
                    ids_b, bad_b = _decode_codes_matrix(codes_b)

                    return ids_a if bad_a <= bad_b else ids_b
            raise ValueError(f"subject_id length mismatch: got={flat.size} expected={expected_n} dtype={flat.dtype}")
        return _finalize_numeric(flat)

    # Case 2: object array (strings/bytes/other). Best-effort.
    flat_obj = np.asarray(x, dtype=object).reshape((-1,))
    if expected_n is not None and flat_obj.size != expected_n:
        # Sometimes read_vector() flattens nested arrays; try to interpret as numeric codes.
        try:
            as_float = np.asarray(flat_obj, dtype=np.float64)
            return decode_subject_id(as_float, expected_n=expected_n)
        except Exception as e:
            raise ValueError(
                f"subject_id length mismatch: got={flat_obj.size} expected={expected_n} dtype=object"
            ) from e

    out = np.full((flat_obj.size,), -1, dtype=np.int64)
    invalid = 0
    for i, v in enumerate(flat_obj.tolist()):
        if v is None:
            invalid += 1
            continue
        if isinstance(v, (bytes, bytearray)):
            s = v.decode("utf-8", errors="ignore")
        else:
            s = str(v)
        m = _DIGITS_RE.search(s)
        if m is None:
            invalid += 1
            continue
        out[i] = int(m.group(1))

    # If decoding failed broadly, surface it.
    if expected_n is not None and invalid > 0:
        frac = invalid / max(1, out.size)
        if frac > 0.01:
            raise ValueError(f"subject_id decode produced too many invalid IDs: {invalid}/{out.size}")

    return out


def summarize_labels_and_demo(
    subset_name: str,
    sbp: np.ndarray,
    dbp: np.ndarray,
    demo: dict[str, np.ndarray],
    subject_id: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stats_by_subset row, missingness rows, and optional subject stats."""
    sbp = sbp.astype(np.float32, copy=False)
    dbp = dbp.astype(np.float32, copy=False)

    def sstats(x: np.ndarray) -> dict[str, float]:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"mean": np.nan, "std": np.nan, "q05": np.nan, "q50": np.nan, "q95": np.nan}
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "q05": float(np.percentile(x, 5)),
            "q50": float(np.percentile(x, 50)),
            "q95": float(np.percentile(x, 95)),
        }

    sbp_s = sstats(sbp)
    dbp_s = sstats(dbp)

    stats_row = {
        "subset_name": subset_name,
        "n_segments": int(max(sbp.size, dbp.size)),
        **{f"sbp_{k}": v for k, v in sbp_s.items()},
        **{f"dbp_{k}": v for k, v in dbp_s.items()},
    }

    missing_rows = []
    for k, v in demo.items():
        v = v.astype(np.float32, copy=False) if v.size else v
        miss = 1.0
        if v.size:
            miss = float((~np.isfinite(v)).mean())
        missing_rows.append({"subset_name": subset_name, "field": k, "missing_rate": miss})

    missing_df = pd.DataFrame(missing_rows)

    subject_df = pd.DataFrame()
    if subject_id is not None and subject_id.size and sbp.size:
        sid = _as_1d(subject_id)
        n = min(sid.size, sbp.size, dbp.size)
        sid = sid[:n]
        sbp2 = sbp[:n]
        dbp2 = dbp[:n]
        tmp = pd.DataFrame({"subject_id": sid, "sbp": sbp2, "dbp": dbp2})
        grp = tmp.groupby("subject_id", dropna=False)
        subject_df = grp.agg(
            n_segments=("sbp", "size"),
            sbp_mean=("sbp", "mean"),
            sbp_std=("sbp", "std"),
            dbp_mean=("dbp", "mean"),
            dbp_std=("dbp", "std"),
        ).reset_index()

    return pd.DataFrame([stats_row]), missing_df, subject_df
