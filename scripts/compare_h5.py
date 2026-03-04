#!/usr/bin/env python3
"""
Compare two HDF5 files by common keys.

Usage:
  python scripts/compare_h5.py file_a.h5 file_b.h5

Behavior:
- Recursively scans both files and compares only common dataset keys.
- For each comparable dataset, prints: dataset_path  max_abs_diff
- Missing/extra keys, group-vs-dataset mismatches, or shape/dtype mismatches are reported and skipped.

在命令行指定两个 h5 文件路径进行对比：
- 递归扫描两者 key tree，只对比共同的 Dataset keys
- 对每个可对比 Dataset 打印：路径 + max abs difference
- 缺失/多余 key、Group/Dataset 类型不一致、shape/dtype 不一致会提示并跳过
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def _preview_paths(title: str, paths: List[str], types: Dict[str, str], *, limit: int = 200) -> None:
    if not paths:
        return
    print(title, file=sys.stderr)
    for p in paths[:limit]:
        print(f"  {p} ({types.get(p)})", file=sys.stderr)
    if len(paths) > limit:
        print(f"  ... and {len(paths) - limit} more", file=sys.stderr)


def _select_common_comparable_datasets(
    types_a: Dict[str, str],
    types_b: Dict[str, str],
    datasets_a: Dict[str, DatasetMeta],
    datasets_b: Dict[str, DatasetMeta],
) -> List[str]:
    paths_a = set(types_a)
    paths_b = set(types_b)
    common_paths = paths_a & paths_b

    only_a = sorted(paths_a - paths_b)
    only_b = sorted(paths_b - paths_a)
    _preview_paths("# INFO: keys only in file A (skipped):", only_a, types_a)
    _preview_paths("# INFO: keys only in file B (skipped):", only_b, types_b)

    type_mismatch = sorted([p for p in common_paths if types_a.get(p) != types_b.get(p)])
    if type_mismatch:
        print("# INFO: common keys with type mismatch (skipped):", file=sys.stderr)
        for p in type_mismatch[:200]:
            print(f"  {p}: A={types_a.get(p)}  B={types_b.get(p)}", file=sys.stderr)
        if len(type_mismatch) > 200:
            print(f"  ... and {len(type_mismatch) - 200} more", file=sys.stderr)

    dataset_paths = sorted(
        p for p in common_paths if types_a.get(p) == "dataset" and types_b.get(p) == "dataset"
    )
    comparable_paths: List[str] = []
    meta_mismatch: List[str] = []
    for p in dataset_paths:
        ma = datasets_a[p]
        mb = datasets_b[p]
        if ma.shape == mb.shape and ma.dtype == mb.dtype:
            comparable_paths.append(p)
        else:
            meta_mismatch.append(f"  {p}: A({_format_meta(ma)})  B({_format_meta(mb)})")

    if meta_mismatch:
        print("# INFO: common dataset keys with shape/dtype mismatch (skipped):", file=sys.stderr)
        for line in meta_mismatch[:200]:
            print(line, file=sys.stderr)
        if len(meta_mismatch) > 200:
            print(f"  ... and {len(meta_mismatch) - 200} more", file=sys.stderr)

    if not comparable_paths:
        raise H5StructureError("No common comparable dataset keys found.")

    return comparable_paths


def _max_abs_diff_dataset(
    dset_a: "h5py.Dataset", dset_b: "h5py.Dataset", *, chunk_rows: int
) -> Tuple[float, float, float]:
    """
    Compute both absolute and relative diff for numeric datasets:

      abs_diff = max( abs(abs(a) - abs(b)) )
      rel_diff = max( abs(abs(a) - abs(b)) / abs(a) )

    This allows sign differences between the two sides (and for complex values, compares magnitudes).
    Uses row-chunking along axis 0 for large arrays to limit memory.
    
    Returns:
        (max_abs_diff, max_rel_diff, abs_a_at_max_rel)
    """
    shape = dset_a.shape
    dt = np.dtype(dset_a.dtype)
    if not _is_numeric_dtype(dt):
        raise H5StructureError(f"Non-numeric dataset dtype not supported for diff: {dset_a.name} dtype={dt}")

    # scalar dataset
    if shape == ():
        a = np.asarray(dset_a[()])
        b = np.asarray(dset_b[()])
        aa = np.abs(a)
        ab = np.abs(b)
        diff = np.abs(aa - ab)
        abs_diff = float(diff)
        abs_a_val = float(aa)
        if abs_a_val == 0.0:
            rel_diff = 0.0 if float(diff) == 0.0 else float("inf")
        else:
            rel_diff = float(diff / aa)
        return abs_diff, rel_diff, abs_a_val

    # small dataset: read whole thing
    n_elem = int(np.prod(shape)) if shape else 0
    # heuristic: if total elements <= 1e6, load in memory
    if n_elem <= 1_000_000:
        a = np.asarray(dset_a[...])
        b = np.asarray(dset_b[...])
        aa = np.abs(a)
        ab = np.abs(b)
        diff = np.abs(aa - ab)
        max_abs_diff = float(np.max(diff))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = diff / aa
        ratio = np.where(aa == 0, np.where(diff == 0, 0.0, float("inf")), ratio)
        max_rel_idx = np.argmax(ratio.flatten())
        max_rel_diff = float(ratio.flatten()[max_rel_idx])
        abs_a_at_max_rel = float(aa.flatten()[max_rel_idx])
        return max_abs_diff, max_rel_diff, abs_a_at_max_rel

    # large: chunk along first axis
    n0 = shape[0]
    if n0 == 0:
        return 0.0, 0.0, 0.0

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    abs_a_at_max_rel = 0.0
    for i in range(0, n0, max(1, chunk_rows)):
        j = min(n0, i + max(1, chunk_rows))
        slc = (slice(i, j),) + (slice(None),) * (len(shape) - 1)
        a = np.asarray(dset_a[slc])
        b = np.asarray(dset_b[slc])
        aa = np.abs(a)
        ab = np.abs(b)
        diff = np.abs(aa - ab)
        vmax_abs = float(np.max(diff))
        if vmax_abs > max_abs_diff:
            max_abs_diff = vmax_abs
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = diff / aa
        ratio = np.where(aa == 0, np.where(diff == 0, 0.0, float("inf")), ratio)
        max_rel_idx = np.argmax(ratio.flatten())
        vmax_rel = float(ratio.flatten()[max_rel_idx])
        if vmax_rel > max_rel_diff:
            max_rel_diff = vmax_rel
            abs_a_at_max_rel = float(aa.flatten()[max_rel_idx])
    return max_abs_diff, max_rel_diff, abs_a_at_max_rel


def compare_h5(file_a: str, file_b: str, *, chunk_rows: int = 1024) -> int:
    with h5py.File(file_a, "r") as fa, h5py.File(file_b, "r") as fb:
        types_a, datasets_a = _collect_objects(fa)
        types_b, datasets_b = _collect_objects(fb)

        dataset_paths = _select_common_comparable_datasets(types_a, types_b, datasets_a, datasets_b)

        # Print header for readability
        print(f"# file_a: {file_a}")
        print(f"# file_b: {file_b}")
        print("# dataset_path\tabs_diff=max(abs(abs(a)-abs(b)))\trel_diff=max(abs(abs(a)-abs(b))/abs(a))\tabs(a)_at_max_rel")

        for p in dataset_paths:
            da = fa[p]
            db = fb[p]
            max_abs_diff, max_rel_diff, abs_a_at_max_rel = _max_abs_diff_dataset(da, db, chunk_rows=chunk_rows)
            # scientific notation, stable width
            print(f"{p}\t{max_abs_diff:.16e}\t{max_rel_diff:.16e}\t{abs_a_at_max_rel:.16e}")

    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare two HDF5 files by common dataset keys.")
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
