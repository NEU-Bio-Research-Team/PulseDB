from __future__ import annotations

from pathlib import Path
import shutil

import yaml


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def main() -> int:
    root = _root_dir()
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))

    cache_dir = root / str(cfg.get("cache_dir", "cache/pulsedb_subset"))

    if not cache_dir.exists():
        print(f"Nothing to delete: {cache_dir} does not exist")
        return 0

    if not _is_within(cache_dir, root):
        raise ValueError(f"Refusing to delete outside project: {cache_dir}")

    # Extra guard: never delete project root
    if cache_dir.resolve() == root.resolve():
        raise ValueError("Refusing to delete project root")

    shutil.rmtree(cache_dir)
    print(f"Cache deleted: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
