#!/usr/bin/env python3
"""
Manual diff between two generated data trees (e.g. two `Modeling/` runs, or
two `System ID/` runs), used to confirm that changes to get_data.py,
get_systems.py, or get_prediction.py haven't changed the data/results they
produce.

Run this file by hand, pointing it at two directories you've generated
yourself, e.g.:

    python tests/compare_data_trees.py "/path/to/old/Modeling" Modeling
    python tests/compare_data_trees.py "/path/to/old/System ID" "System ID"

The baseline directory is not modified. Only the second argument is treated
as "current".
"""
import argparse
import filecmp
import pickle
import sys
from pathlib import Path

import numpy as np


def relative_files(root: Path) -> set:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, ndmin=2)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def flatten_arrays(obj):
    """Pull numpy arrays out of a value that may itself be an array, or a
    tuple/list/dict of arrays (e.g. a (A, B, C, D) system realization)."""
    if isinstance(obj, np.ndarray):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from flatten_arrays(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from flatten_arrays(item)
    else:
        yield np.asarray(obj)


def compare_numeric(a_path: Path, b_path: Path, rtol: float, atol: float) -> dict:
    """Load matching files as arrays and classify how they differ."""
    suffix = a_path.suffix.lower()
    try:
        if suffix == ".csv":
            a_arrays, b_arrays = [load_csv(a_path)], [load_csv(b_path)]
        elif suffix == ".pkl":
            a_arrays = list(flatten_arrays(load_pickle(a_path)))
            b_arrays = list(flatten_arrays(load_pickle(b_path)))
        else:
            return {"status": "unsupported"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if len(a_arrays) != len(b_arrays):
        return {"status": "structure_mismatch"}

    exact = True
    max_abs = 0.0
    max_rel = 0.0
    for a, b in zip(a_arrays, b_arrays):
        if a.shape != b.shape:
            return {"status": "shape_mismatch", "shapes": (a.shape, b.shape)}
        if np.array_equal(a, b):
            continue
        exact = False
        try:
            a_f, b_f = a.astype(float), b.astype(float)
        except (TypeError, ValueError):
            # non-numeric (e.g. strings) and unequal: no notion of "close"
            return {"status": "different", "max_abs": None, "max_rel": None}
        diff = np.abs(a_f - b_f)
        max_abs = max(max_abs, float(diff.max(initial=0.0)))
        denom = np.maximum(np.abs(a_f), np.abs(b_f))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(denom > 0, diff / denom, 0.0)
        max_rel = max(max_rel, float(np.max(rel, initial=0.0)))

    if exact:
        return {"status": "exact"}
    if all(np.allclose(a, b, rtol=rtol, atol=atol) for a, b in zip(a_arrays, b_arrays)):
        return {"status": "close", "max_abs": max_abs, "max_rel": max_rel}
    return {"status": "different", "max_abs": max_abs, "max_rel": max_rel}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path, help="baseline directory (the 'known good' data)")
    parser.add_argument("current", type=Path, help="directory to compare against the baseline")
    parser.add_argument("--rtol", type=float, default=1e-5, help="relative tolerance for np.allclose (default 1e-5)")
    parser.add_argument("--atol", type=float, default=1e-8, help="absolute tolerance for np.allclose (default 1e-8)")
    args = parser.parse_args()

    if not args.baseline.is_dir():
        sys.exit(f"Not a directory: {args.baseline}")
    if not args.current.is_dir():
        sys.exit(f"Not a directory: {args.current}")

    a_files = relative_files(args.baseline)
    b_files = relative_files(args.current)
    only_a = sorted(a_files - b_files)
    only_b = sorted(b_files - a_files)
    common = sorted(a_files & b_files)

    byte_identical = []
    numerically_exact = []   # bytes differ, values identical (e.g. formatting)
    close_only = []          # values differ but within tolerance
    different = []           # values differ beyond tolerance / shape / structure
    unsupported_diff = []    # non-csv/pkl files whose bytes differ
    errors = []

    print(f"Baseline: {args.baseline}")
    print(f"Current:  {args.current}")
    print(f"Comparing {len(common)} common files "
          f"({len(only_a)} only in baseline, {len(only_b)} only in current)...\n")

    for rel in common:
        a_path, b_path = args.baseline / rel, args.current / rel
        if filecmp.cmp(a_path, b_path, shallow=False):
            byte_identical.append(rel)
            continue

        result = compare_numeric(a_path, b_path, args.rtol, args.atol)
        status = result["status"]
        if status == "unsupported":
            unsupported_diff.append(rel)
        elif status == "error":
            errors.append((rel, result["error"]))
        elif status == "exact":
            numerically_exact.append(rel)
        elif status == "close":
            close_only.append((rel, result["max_abs"], result["max_rel"]))
        else:
            different.append((rel, status, result))

    if only_a:
        print(f"Only in baseline ({len(only_a)}):")
        for rel in only_a:
            print(f"  - {rel}")
        print()

    if only_b:
        print(f"Only in current ({len(only_b)}):")
        for rel in only_b:
            print(f"  - {rel}")
        print()

    if numerically_exact:
        print(f"Byte-different but numerically identical ({len(numerically_exact)}):")
        for rel in numerically_exact:
            print(f"  - {rel}")
        print()

    if close_only:
        print(f"Differ only within tolerance (rtol={args.rtol}, atol={args.atol}) ({len(close_only)}):")
        for rel, max_abs, max_rel in sorted(close_only, key=lambda x: x[1], reverse=True):
            print(f"  - {rel}: max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}")
        print()

    if different:
        print(f"DIFFERENT beyond tolerance ({len(different)}):")
        for rel, status, result in different:
            if status == "shape_mismatch":
                print(f"  - {rel}: shape mismatch {result['shapes'][0]} vs {result['shapes'][1]}")
            elif status == "structure_mismatch":
                print(f"  - {rel}: different pickled structure (different number of arrays)")
            else:
                max_abs, max_rel = result.get("max_abs"), result.get("max_rel")
                if max_abs is None:
                    print(f"  - {rel}: differs (non-numeric content)")
                else:
                    print(f"  - {rel}: max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}")
        print()

    if unsupported_diff:
        print(f"Differing files with unsupported extension (compared as raw bytes) ({len(unsupported_diff)}):")
        for rel in unsupported_diff:
            print(f"  - {rel}")
        print()

    if errors:
        print(f"Errors while loading ({len(errors)}):")
        for rel, err in errors:
            print(f"  - {rel}: {err}")
        print()

    print(f"Byte-identical files: {len(byte_identical)}/{len(common)}")
    print()

    if only_a or only_b or different or unsupported_diff or errors:
        print("RESULT: DIFFERENT - see details above.")
    elif close_only:
        print(f"RESULT: Data is the same, but {len(close_only)} file(s) needed np.allclose "
              f"(rtol={args.rtol}, atol={args.atol}) rather than exact equality to confirm it.")
    elif numerically_exact:
        print("RESULT: Data is exactly the same numerically (some files differ byte-for-byte, "
              "e.g. due to formatting, but no values changed).")
    else:
        print("RESULT: Files are exactly the same (byte-for-byte).")


if __name__ == "__main__":
    main()
