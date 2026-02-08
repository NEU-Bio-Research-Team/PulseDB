from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ShardBuffer:
    x_list: list[np.ndarray]
    y_list: list[np.ndarray]
    demo_list: list[np.ndarray]
    subject_list: list[np.ndarray]

    def __init__(self) -> None:
        self.x_list = []
        self.y_list = []
        self.demo_list = []
        self.subject_list = []

    def size(self) -> int:
        if not self.x_list:
            return 0
        return int(sum(arr.shape[0] for arr in self.x_list))


class ShardWriter:
    def __init__(
        self,
        out_dir: Path,
        shard_size: int,
        subset_name: str,
        meta: dict,
    ) -> None:
        self.out_dir = out_dir
        self.shard_size = int(shard_size)
        self.subset_name = subset_name
        self.meta = dict(meta)
        self.buf = ShardBuffer()
        self.shard_index = 0
        self.total_written = 0

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def add(
        self,
        x_nct: np.ndarray,
        y: np.ndarray,
        demo: np.ndarray,
        subject_id: np.ndarray | None = None,
    ) -> None:
        n = int(x_nct.shape[0])
        if n == 0:
            return
        self.buf.x_list.append(x_nct)
        self.buf.y_list.append(y)
        self.buf.demo_list.append(demo)
        if subject_id is None:
            self.buf.subject_list.append(np.full((n,), -1, dtype=np.int64))
        else:
            self.buf.subject_list.append(subject_id.astype(np.int64, copy=False).reshape((-1,)))

        while self.buf.size() >= self.shard_size:
            self._flush_one(self.shard_size)

    def close(self) -> None:
        if self.buf.size() > 0:
            self._flush_one(self.buf.size())

    def _flush_one(self, take_n: int) -> None:
        x = np.concatenate(self.buf.x_list, axis=0)
        y = np.concatenate(self.buf.y_list, axis=0)
        demo = np.concatenate(self.buf.demo_list, axis=0)
        sid = np.concatenate(self.buf.subject_list, axis=0)

        x_out = x[:take_n]
        y_out = y[:take_n]
        demo_out = demo[:take_n]
        sid_out = sid[:take_n]

        remain = slice(take_n, None)
        self.buf.x_list = [x[remain]] if x.shape[0] > take_n else []
        self.buf.y_list = [y[remain]] if y.shape[0] > take_n else []
        self.buf.demo_list = [demo[remain]] if demo.shape[0] > take_n else []
        self.buf.subject_list = [sid[remain]] if sid.shape[0] > take_n else []

        out_path = self.out_dir / f"shard_{self.shard_index:04d}.npz"
        np.savez_compressed(out_path, X=x_out, y=y_out, demo=demo_out, subject_id=sid_out)
        self.total_written += int(take_n)
        self.shard_index += 1
