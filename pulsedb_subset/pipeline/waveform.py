from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WaveformBatch:
    x: np.ndarray  # (N, C, T)


def to_nct(x: np.ndarray) -> np.ndarray:
    """Best-effort convert a waveform tensor to (N, C, T).

    Supported input shapes:
    - (N, T) -> (N, 1, T)
    - (N, T, C) -> transpose to (N, C, T)
    - (N, C, T) -> unchanged
    - (T, C, N) or (C, T, N) etc: inferred heuristically

    Heuristic:
    - Channel axis tends to be small (<= 8)
    - Time axis tends to be large (>= 100)
    """
    x = np.asarray(x)

    if x.ndim == 2:
        return x[:, None, :]

    if x.ndim != 3:
        raise ValueError(f"Unsupported waveform ndim={x.ndim}, shape={x.shape}")

    shape = list(x.shape)

    # Find likely channel axis
    channel_axis = None
    for ax, s in enumerate(shape):
        if s <= 8:
            channel_axis = ax
            break

    # Find likely time axis
    time_axis = None
    for ax, s in enumerate(shape):
        if s >= 100 and ax != channel_axis:
            time_axis = ax
            break

    if channel_axis is None:
        # default: last dim
        channel_axis = 2
    if time_axis is None:
        # default: middle dim
        time_axis = 1 if channel_axis != 1 else 2

    n_axis = ({0, 1, 2} - {channel_axis, time_axis}).pop()

    # permute to (N, C, T)
    perm = (n_axis, channel_axis, time_axis)
    return np.transpose(x, axes=perm)


def enforce_nct(signals: np.ndarray, segment_len: int) -> np.ndarray:
    """Enforce a single tensor convention: (N, C, T).

    Raw subset files often store signals as (T, C, N). We convert deterministically
    using the known segment length (T = fs * segment_sec).

    Supported inputs:
    - (T, C, N) -> (N, C, T)
    - (N, C, T) -> unchanged
    - (N, T, C) -> (N, C, T)
    - (C, T, N) -> (N, C, T)
    """
    x = np.asarray(signals)
    if x.ndim != 3:
        raise ValueError(f"signals must be 3D, got shape={x.shape}")

    T = int(segment_len)
    if T <= 0:
        raise ValueError(f"Invalid segment_len={segment_len}")

    # (T, C, N)
    if x.shape[0] == T:
        return np.transpose(x, (2, 1, 0))

    # already (N, C, T)
    if x.shape[2] == T:
        return x

    # (N, T, C)
    if x.shape[1] == T:
        return np.transpose(x, (0, 2, 1))

    # (C, T, N)
    if x.shape[1] == T:
        # already handled above; keep for clarity
        pass
    if x.shape[0] <= 8 and x.shape[1] == T:
        return np.transpose(x, (2, 0, 1))

    raise ValueError(f"Unknown signals shape {x.shape}; expected one axis == segment_len={T}")


def select_channel_indices(x_nct: np.ndarray, channels: list[str], channel_indices: dict[str, int]) -> np.ndarray:
    """Select channels by explicit index mapping.

    This is critical to avoid silently training on the wrong modality.
    """
    if x_nct.ndim != 3:
        raise ValueError(f"Expected (N,C,T), got shape={x_nct.shape}")
    if not channels:
        return x_nct

    if channel_indices:
        idxs: list[int] = []
        for name in channels:
            if name not in channel_indices:
                raise KeyError(
                    f"Missing channel_indices mapping for '{name}'. "
                    "Set it in pulsedb_subset/config.yaml (e.g. channel_indices: {PPG: 1})."
                )
            idxs.append(int(channel_indices[name]))

        if max(idxs) >= x_nct.shape[1] or min(idxs) < 0:
            raise ValueError(f"channel_indices={idxs} out of range for C={x_nct.shape[1]}")
        return x_nct[:, idxs, :]

    # No mapping provided.
    # Safe default: only allow implicit selection if requested channels matches C exactly.
    if len(channels) == int(x_nct.shape[1]):
        return x_nct
    raise RuntimeError(
        f"Signals has C={x_nct.shape[1]} channels but config.channels={channels} and no channel_indices mapping. "
        "Refusing to select channels implicitly. Run scripts/debug_plot_channels.py to identify PPG index, then set channel_indices in config.yaml."
    )


def select_channels(x_nct: np.ndarray, n_channels: int) -> np.ndarray:
    if x_nct.ndim != 3:
        raise ValueError(f"Expected (N,C,T), got shape={x_nct.shape}")
    if x_nct.shape[1] < n_channels:
        # pad with zeros
        pad = np.zeros((x_nct.shape[0], n_channels - x_nct.shape[1], x_nct.shape[2]), dtype=x_nct.dtype)
        return np.concatenate([x_nct, pad], axis=1)
    return x_nct[:, :n_channels, :]


def infer_segment_axis(shape: tuple[int, ...], n_segments_hint: int | None = None) -> int:
    """Infer which axis corresponds to the segment/batch dimension (N).

    For PulseDB subset files we often see signals shaped like (T, C, N) e.g. (1250, 3, 465480).
    The pipeline needs to slice along N efficiently.

    Strategy:
    - If n_segments_hint matches exactly one axis -> choose it.
    - Else choose the largest axis that is not an obvious channel axis (<= 8).
    """
    if len(shape) == 0:
        return 0

    # exact match
    if n_segments_hint is not None and n_segments_hint > 0:
        matches = [i for i, s in enumerate(shape) if int(s) == int(n_segments_hint)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # ambiguous; pick the last (commonly N is the last axis)
            return matches[-1]

    # avoid small channel-like axes
    candidates = [i for i, s in enumerate(shape) if int(s) > 8]
    if not candidates:
        # fallback: largest axis
        return int(np.argmax(np.asarray(shape)))

    sizes = [int(shape[i]) for i in candidates]
    return candidates[int(np.argmax(np.asarray(sizes)))]
