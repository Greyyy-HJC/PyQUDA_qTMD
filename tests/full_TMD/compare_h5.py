#!/usr/bin/env python3
"""
Compare two HDF5 files with identical key structures.

Usage:
  python tests/full_TMD/compare_h5.py file_a.h5 file_b.h5

Behavior:
- Recursively checks that both files have the same group/dataset key tree.
- For each leaf dataset, prints: dataset_path  max_abs_diff
- If structures differ (missing/extra keys, group vs dataset mismatch), prints an error and exits non-zero.

在命令行指定两个 h5 文件路径进行对比：
- 递归检查两者 key tree 完全一致（Group/Dataset 类型也要一致）
- 对每个末端 Dataset 打印：路径 + max abs difference
- 如果结构不同则报错并退出（返回非 0）
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import h5py
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "ERROR: failed to import h5py. Please ensure h5py is installed in your environment.\n"
        f"Import error: {e}"
    )


class H5StructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetMeta:
    path: str
    shape: Tuple[int, ...]
    dtype: np.dtype


def _is_numeric_dtype(dt: np.dtype) -> bool:
    # bool/int/float/complex all count as numeric for our purpose
    try:
        return np.issubdtype(dt, np.number) or np.issubdtype(dt, np.bool_)
    except TypeError:
        return False


def _collect_objects(h5: "h5py.File") -> Tuple[Dict[str, str], Dict[str, DatasetMeta]]:
    """
    Returns:
      - obj_types: mapping path -> "group" | "dataset"
      - datasets: mapping dataset_path -> DatasetMeta
    """
    obj_types: Dict[str, str] = {}
    datasets: Dict[str, DatasetMeta] = {}

    def visitor(name: str, obj) -> None:
        path = "/" + name if not name.startswith("/") else name
        if isinstance(obj, h5py.Group):
            obj_types[path] = "group"
        elif isinstance(obj, h5py.Dataset):
            obj_types[path] = "dataset"
            datasets[path] = DatasetMeta(path=path, shape=tuple(obj.shape), dtype=np.dtype(obj.dtype))
        else:
            # Rare in typical physics data (e.g. soft/external links); treat as unsupported.
            obj_types[path] = type(obj).__name__

    # include root explicitly for nicer structure diffs
    obj_types["/"] = "group"
    h5.visititems(visitor)
    return obj_types, datasets


def _format_meta(meta: DatasetMeta) -> str:
    return f"shape={meta.shape}, dtype={meta.dtype}"


def _ensure_same_structure(
    types_a: Dict[str, str],
    types_b: Dict[str, str],
    datasets_a: Dict[str, DatasetMeta],
    datasets_b: Dict[str, DatasetMeta],
) -> List[str]:
    paths_a = set(types_a.keys())
    paths_b = set(types_b.keys())

    only_a = sorted(paths_a - paths_b)
    only_b = sorted(paths_b - paths_a)
    if only_a or only_b:
        msg = ["HDF5 key structure mismatch:"]
        if only_a:
            msg.append("  Present only in file A:")
            msg.extend([f"    {p} ({types_a.get(p)})" for p in only_a[:200]])
            if len(only_a) > 200:
                msg.append(f"    ... and {len(only_a) - 200} more")
        if only_b:
            msg.append("  Present only in file B:")
            msg.extend([f"    {p} ({types_b.get(p)})" for p in only_b[:200]])
            if len(only_b) > 200:
                msg.append(f"    ... and {len(only_b) - 200} more")
        raise H5StructureError("\n".join(msg))

    # same paths, now same types
    type_mismatch = sorted([p for p in paths_a if types_a.get(p) != types_b.get(p)])
    if type_mismatch:
        msg = ["HDF5 object type mismatch (group vs dataset):"]
        for p in type_mismatch[:200]:
            msg.append(f"  {p}: A={types_a.get(p)}  B={types_b.get(p)}")
        if len(type_mismatch) > 200:
            msg.append(f"  ... and {len(type_mismatch) - 200} more")
        raise H5StructureError("\n".join(msg))

    # dataset metadata sanity checks (shape + dtype)
    dataset_paths = sorted(datasets_a.keys())
    meta_mismatch: List[str] = []
    for p in dataset_paths:
        ma = datasets_a[p]
        mb = datasets_b[p]
        if ma.shape != mb.shape or ma.dtype != mb.dtype:
            meta_mismatch.append(
                f"  {p}: A({_format_meta(ma)})  B({_format_meta(mb)})"
            )
    if meta_mismatch:
        msg = ["HDF5 dataset metadata mismatch (shape/dtype):"]
        msg.extend(meta_mismatch[:200])
        if len(meta_mismatch) > 200:
            msg.append(f"  ... and {len(meta_mismatch) - 200} more")
        raise H5StructureError("\n".join(msg))

    return dataset_paths


def _max_abs_diff_dataset(
    dset_a: "h5py.Dataset", dset_b: "h5py.Dataset", *, chunk_rows: int
) -> float:
    """
    Compute max(abs(a - b)) for numeric datasets.
    Uses row-chunking along axis 0 for large arrays to limit memory.
    """
    shape = dset_a.shape
    dt = np.dtype(dset_a.dtype)
    if not _is_numeric_dtype(dt):
        raise H5StructureError(f"Non-numeric dataset dtype not supported for diff: {dset_a.name} dtype={dt}")

    # scalar dataset
    if shape == ():
        a = np.asarray(dset_a[()])
        b = np.asarray(dset_b[()])
        diff = a - b
        return float(np.max(np.abs(diff)))

    # small dataset: read whole thing
    n_elem = int(np.prod(shape)) if shape else 0
    # heuristic: if total elements <= 1e6, load in memory
    if n_elem <= 1_000_000:
        a = np.asarray(dset_a[...])
        b = np.asarray(dset_b[...])
        return float(np.max(np.abs(a - b)))

    # large: chunk along first axis
    n0 = shape[0]
    if n0 == 0:
        return 0.0

    maxv = 0.0
    for i in range(0, n0, max(1, chunk_rows)):
        j = min(n0, i + max(1, chunk_rows))
        slc = (slice(i, j),) + (slice(None),) * (len(shape) - 1)
        a = np.asarray(dset_a[slc])
        b = np.asarray(dset_b[slc])
        v = np.max(np.abs(a - b))
        # v may be numpy scalar
        if float(v) > maxv:
            maxv = float(v)
    return maxv


def compare_h5(file_a: str, file_b: str, *, chunk_rows: int = 1024) -> int:
    with h5py.File(file_a, "r") as fa, h5py.File(file_b, "r") as fb:
        types_a, datasets_a = _collect_objects(fa)
        types_b, datasets_b = _collect_objects(fb)

        dataset_paths = _ensure_same_structure(types_a, types_b, datasets_a, datasets_b)

        # Print header for readability
        print(f"# file_a: {file_a}")
        print(f"# file_b: {file_b}")
        print("# dataset_path\tmax_abs_diff")

        for p in dataset_paths:
            da = fa[p]
            db = fb[p]
            maxdiff = _max_abs_diff_dataset(da, db, chunk_rows=chunk_rows)
            # scientific notation, stable width
            print(f"{p}\t{maxdiff:.16e}")

    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare two HDF5 files with identical key structures.")
    ap.add_argument("file_a", type=str, help="Path to HDF5 file A")
    ap.add_argument("file_b", type=str, help="Path to HDF5 file B")
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=1024,
        help="Chunk size along axis-0 when datasets are large (default: 1024)",
    )
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return compare_h5(args.file_a, args.file_b, chunk_rows=args.chunk_rows)
    except H5StructureError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

