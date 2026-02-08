from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RunningMeanStd:
    """Streaming mean/std for vectors using Welford."""

    n: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def create(cls, dim: int, dtype=np.float64) -> "RunningMeanStd":
        return cls(n=0, mean=np.zeros((dim,), dtype=dtype), m2=np.zeros((dim,), dtype=dtype))

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        x = x.astype(np.float64, copy=False)
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.m2 += delta * delta2

    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return self.m2 / (self.n - 1)

    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance(), 1e-12))


@dataclass
class RunningChannelMoments:
    """Streaming moments for per-channel z-score over all values (segments*time)."""

    count: np.ndarray
    sum: np.ndarray
    sumsq: np.ndarray

    @classmethod
    def create(cls, channels: int, dtype=np.float64) -> "RunningChannelMoments":
        return cls(
            count=np.zeros((channels,), dtype=np.int64),
            sum=np.zeros((channels,), dtype=dtype),
            sumsq=np.zeros((channels,), dtype=dtype),
        )

    def update_nct(self, x_nct: np.ndarray) -> None:
        x = np.asarray(x_nct)
        if x.ndim != 3:
            raise ValueError(f"Expected (N,C,T), got {x.shape}")
        n, c, t = x.shape
        for ch in range(c):
            vals = x[:, ch, :].reshape((-1,))
            finite = np.isfinite(vals)
            if not finite.any():
                continue
            v = vals[finite].astype(np.float64, copy=False)
            self.count[ch] += int(v.size)
            self.sum[ch] += float(v.sum())
            self.sumsq[ch] += float(np.square(v).sum())

    def to_scaler(self) -> ZScoreScaler:
        mean = np.zeros_like(self.sum, dtype=np.float32)
        std = np.ones_like(self.sum, dtype=np.float32)
        for ch in range(mean.size):
            if self.count[ch] <= 0:
                continue
            m = self.sum[ch] / self.count[ch]
            v = self.sumsq[ch] / self.count[ch] - m * m
            mean[ch] = float(m)
            std[ch] = float(np.sqrt(max(v, 1e-12)))
        return ZScoreScaler(mean=mean, std=std)


@dataclass(frozen=True)
class ZScoreScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)

        # scalar mean/std
        if self.mean.ndim == 0 or self.mean.size == 1:
            return (x - float(self.mean.reshape(()))) / float(self.std.reshape(()))

        # per-channel for (N,C,T)
        if x.ndim == 3 and self.mean.ndim == 1:
            mean = self.mean[None, :, None]
            std = self.std[None, :, None]
            return (x - mean) / std

        # vector for (N,D) or (D,)
        return (x - self.mean) / self.std

    def to_jsonable(self) -> dict:
        return {"type": "zscore", "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_jsonable(cls, d: dict) -> "ZScoreScaler":
        return cls(mean=np.asarray(d["mean"], dtype=np.float32), std=np.asarray(d["std"], dtype=np.float32))


def save_scalers(path: Path, scalers: dict[str, ZScoreScaler]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {k: v.to_jsonable() for k, v in scalers.items()}
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_scalers(path: Path) -> dict[str, ZScoreScaler]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {k: ZScoreScaler.from_jsonable(v) for k, v in obj.items()}
