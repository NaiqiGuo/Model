"""Export System ID data and compare two such exports.

Four categories are covered:
    errors        prediction error CSVs (one number per line)
    heatmaps      prediction error heatmaps (one image per structure/quantity/source)
    training      compressed per-event NPZ data (dt, time, ground, structure)
    realizations  identified state-space matrices (A, B, C, D), as CSVs

build:   pick one category and export only that data, retaining the layout:
         <name>_<structure>/<quantity>/<source>/ for errors
         <name>_<structure>/<quantity>/<source>/training/ for training
         <name>_<structure>/<quantity>/<source>/systems/ for realizations
         <name>_heatmaps/ for heatmaps

compare: pick one category and summarize the differences between two
         build exports. Numbers are compared with a tolerance, so
         floating-point noise is not reported as a difference.
         --silent suppresses terminal output; --to-file saves the report
         to the specified path. Without --to-file, no report file is saved.

Usage:
    python inspect_sid.py build <category> <name>
    python inspect_sid.py compare <category> <folder1> <folder2> [--rel-tol <float>] [--abs-tol <float>] [--silent] [--to-file <path>]

Examples:
    python inspect_sid.py build realizations my_export
    python inspect_sid.py compare errors reference_bridge my_export_bridge --rel-tol 1e-3

"""


import argparse
import filecmp
import shutil
import pickle
from pathlib import Path

SID_DIR = Path("System ID")

STRUCTURES = {
    "bridge": range(1, 23),      # events 1-22
    "frame": range(226, 248),    # events 226-247
}
QUANTITIES = ["acceleration", "displacement"]
SOURCES = ["elastic", "field", "inelastic"]


def build_prefix(name: str) -> str:
    """Use the supplied name directly in the current directory."""
    if not name or not Path(name).name or name in (".", ".."):
        raise ValueError("Provide an export name such as analysis_export")
    return name

def export_groups(name: str):
    """Yield common source and destination paths without creating directories."""
    name = build_prefix(name)
    print(f"Export prefix: {name}")
    for structure, events in STRUCTURES.items():
        out_root = Path(f"{name}_{structure}")
        for quantity in QUANTITIES:
            for source in SOURCES:
                source_dir = SID_DIR / structure / quantity / source
                out_dir = out_root / quantity / source
                yield structure, quantity, source, events, source_dir, out_dir
        print(f"{out_root}: done")


def export_errors(name: str) -> int:
    """Copy prediction error CSVs without changing their contents."""
    copied = 0
    for structure, quantity, source, events, source_dir, out_dir in export_groups(name):
        out_dir.mkdir(parents=True, exist_ok=True)
        for n in events:
            src = source_dir / "System ID Results" / "errors" / f"{n}.csv"
            dst = out_dir / f"{structure}_{source}_{quantity}_event_{n}.csv"
            shutil.copyfile(src, dst)
            copied += 1
    return copied


def export_training(name: str) -> int:
    """Save each event's four training arrays in one lossless compressed NPZ."""
    import numpy as np

    copied = 0
    for _, _, _, events, source_dir, out_dir in export_groups(name):
        for n in events:
            data = {
                item: np.loadtxt(source_dir / "System ID Training Data" / item / f"{n}.csv")
                for item in ("dt", "time", "ground", "structure")
            }
            dst = out_dir / "training" / f"{n}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, **data)
            copied += 1
    return copied


def export_realizations(name: str) -> int:
    """Export raw identified A/B/C/D matrices using the existing CSV format."""
    import numpy as np

    copied = 0
    for _, _, _, events, source_dir, out_dir in export_groups(name):
        for n in events:
            src = source_dir / "System ID Results" / "system realization" / f"{n}.pkl"
            # Only load the local pipeline's trusted pickle artifacts.
            with src.open("rb") as stream:
                matrices = pickle.load(stream)
            if len(matrices) != 4:
                raise ValueError(f"Expected A, B, C, D in {src}")
            dst = out_dir / "systems" / str(n)
            dst.mkdir(parents=True, exist_ok=True)
            for label, matrix in zip("ABCD", matrices):
                matrix = np.asarray(matrix)
                if matrix.ndim != 2:
                    raise ValueError(f"{src}: {label} must be a matrix")
                np.savetxt(dst / f"{label}.csv", matrix, delimiter=",")
                copied += 1
    return copied


def export_heatmaps(name: str) -> int:
    """Copy only the heatmap image for each structure/quantity/source."""
    name = build_prefix(name)
    print(f"Export prefix: {name}")
    out_dir = Path(f"{name}_heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for structure in STRUCTURES:
        for quantity in QUANTITIES:
            for source in SOURCES:
                results_dir = SID_DIR / structure / quantity / source / "System ID Results"
                src = results_dir / "heatmap.png"
                dst = out_dir / f"{structure}_{quantity}_{source}_heatmap.png"
                shutil.copyfile(src, dst)
                copied += 1
    print(f"{out_dir}: done")
    return copied


def list_files(root: Path) -> dict[str, Path]:
    return {
        str(p.relative_to(root)): p
        for p in root.rglob("*")
        if p.is_file()
    }


def category(rel):
    parts = Path(rel).parts
    if Path(rel).suffix.lower() == ".npz" and "training" in parts:
        return "training"
    if Path(rel).suffix.lower() != ".csv":
        return None
    return "training" if "training" in parts else "realizations" if "systems" in parts else "errors"


def compare_arrays(path_a, path_b, rel_tol, abs_tol, delimiter=None, training=False):
    """Compare CSVs or NPZ members using symmetric math.isclose tolerance."""
    import numpy as np

    arrays = []
    for path in (path_a, path_b):
        if isinstance(path, tuple):
            archive, key = path
            with np.load(archive, allow_pickle=False) as data:
                arrays.append(np.atleast_2d(data[key]))
        else:
            arrays.append(np.atleast_2d(np.loadtxt(path, delimiter=delimiter)) if training
                          else np.loadtxt(path, delimiter=delimiter, ndmin=2))
    a, b = arrays
    if a.shape != b.shape:
        return True, f"shape mismatch: A={a.shape}, B={b.shape}"
    invalid_a, invalid_b = int(np.count_nonzero(~np.isfinite(a))), int(np.count_nonzero(~np.isfinite(b)))
    if invalid_a or invalid_b:
        return True, f"non-finite values: A={invalid_a}, B={invalid_b}; shape={a.shape}"
    delta = np.abs(a - b)
    threshold = np.maximum(abs_tol, rel_tol * np.maximum(np.abs(a), np.abs(b)))
    n_bad = int(np.count_nonzero(delta > threshold))
    if not n_bad:
        return False, "exact" if np.array_equal(a, b) else ""
    index = np.unravel_index(np.argmax(delta), delta.shape)
    denominator = np.linalg.norm(a)
    relative = f"{np.linalg.norm(a-b)/denominator:.6g}" if denominator else "undefined (A norm is zero)"
    return True, (f"{n_bad}/{a.size} values differ ({100*n_bad/a.size:.3g}%); shape={a.shape}; "
                  f"max |diff|={delta[index]:.6g} at row={index[0]}, col={index[1]} (zero-based); relative norm vs A={relative}")


def compare(dir_a: Path, dir_b: Path, rel_tol: float, abs_tol: float, selected: str) -> str:
    """Return report text without printing or writing files."""
    for d in (dir_a, dir_b):
        if not d.is_dir():
            raise ValueError(f"not a directory: {d}")

    files_a, files_b = (
        {rel: path for rel, path in list_files(root).items() if category(rel) == selected}
        for root in (dir_a, dir_b)
    )
    for files in (files_a, files_b):
        # Old exports have one extra bridge/frame directory. Match both layouts.
        nested = {Path(rel).parts[0] for rel in files if Path(rel).parts[0] in STRUCTURES}
        if len(nested) > 1:
            raise ValueError("Select one structure's export folder, not a folder containing both bridge and frame")
        for rel in list(files):
            parts = Path(rel).parts
            if len(parts) > 1 and parts[0] in STRUCTURES and parts[1] in QUANTITIES:
                # Prefer a new flat export if an old copy also exists.
                files.setdefault(str(Path(*parts[1:])), files.pop(rel))
    if selected == "training":
        import numpy as np

        # Match NPZ members to the legacy CSV paths without extracting files.
        # When both formats exist, the newer NPZ export takes precedence.
        for files in (files_a, files_b):
            for rel, path in list(files.items()):
                if path.suffix.lower() != ".npz":
                    continue
                del files[rel]
                with np.load(path, allow_pickle=False) as data:
                    required = {"dt", "time", "ground", "structure"}
                    if set(data.files) != required:
                        raise ValueError(f"{path}: expected NPZ members {sorted(required)}")
                    for key in sorted(required):
                        logical = str(Path(rel).parent / key / f"{path.stem}.csv")
                        files[logical] = (path, key)
    only_a = sorted(files_a.keys() - files_b.keys())
    only_b = sorted(files_b.keys() - files_a.keys())
    common = sorted(files_a.keys() & files_b.keys())

    identical, noise_only, real = 0, 0, []
    stats = {s: 0 for s in ("identical", "within tolerance", "different", "only A", "only B")}
    # Realization exports are comma-separated; training and errors use whitespace.
    delimiter = "," if selected == "realizations" else None
    for rel in common:
        different, detail = compare_arrays(files_a[rel], files_b[rel], rel_tol, abs_tol, delimiter,
                                           training=selected == "training")
        if different:
            real.append((rel, detail))
            stats["different"] += 1
        elif (detail == "exact" if isinstance(files_a[rel], tuple) or isinstance(files_b[rel], tuple)
              else filecmp.cmp(files_a[rel], files_b[rel], shallow=False)):
            identical += 1
            stats["identical"] += 1
        else:
            noise_only += 1
            stats["within tolerance"] += 1
    stats["only A"] = len(only_a)
    stats["only B"] = len(only_b)

    lines = []
    lines.append(f"A: {dir_a}")
    lines.append(f"B: {dir_b}")
    lines.append(f"Comparison: {selected}")
    if selected == "training":
        lines.append("Training counts are per array (dt/time/ground/structure); NPZ takes precedence over coexisting legacy CSVs.")
    lines.append(f"tolerance: rel_tol={rel_tol:g}, abs_tol={abs_tol:g}")
    lines.append("")
    lines.append(f"files in both:           {len(common)}")
    lines.append(f"  identical:             {identical}")
    lines.append(f"  float noise only:      {noise_only}")
    lines.append(f"  real differences:      {len(real)}")
    lines.append(f"only in A:               {len(only_a)}")
    lines.append(f"only in B:               {len(only_b)}")
    lines.append("\nBy category:")
    lines.append(f"  {selected}: " + ", ".join(f"{key}={value}" for key, value in stats.items()))
    if not any(stats.values()):
        lines.append("    No data provided in either folder; not compared.")
    if selected == "realizations":
        lines.append("  realizations: raw A/B/C/D before prediction stabilization; matrix differences do not alone establish different input/output behavior.")
        """
        xdot = Ax + Bu
        y = Cx + Du
        xdot = A1x + B1u
        y = C1x + D1u
        """

    if real:
        lines.append("\nReal differences:")
        for rel, detail in real:
            lines.append(f"  {rel}: {detail}")
    if only_a:
        lines.append("\nOnly in A:")
        for rel in only_a:
            lines.append(f"  {rel}")
    if only_b:
        lines.append("\nOnly in B:")
        for rel in only_b:
            lines.append(f"  {rel}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    exporters = {
        "errors": export_errors,
        "training": export_training,
        "realizations": export_realizations,
        "heatmaps": export_heatmaps,
    }
    p_build = sub.add_parser("build", help="export one data category")
    p_build.add_argument("category", choices=exporters, help="export only this data category")
    p_build.add_argument("name", help="export prefix in the current directory, e.g. analysis_export")

    p_cmp = sub.add_parser("compare", help="summarize differences between two folders")
    p_cmp.add_argument("category", choices=("errors", "realizations", "training"), help="compare only this data category")
    p_cmp.add_argument("dir_a", type=Path)
    p_cmp.add_argument("dir_b", type=Path)
    p_cmp.add_argument("--rel-tol", type=float, default=1e-6, help="relative tolerance for numbers (default: 1e-6)")
    p_cmp.add_argument("--abs-tol", type=float, default=1e-9, help="absolute tolerance for numbers (default: 1e-9)")

    p_cmp.add_argument("--silent", action="store_true", help="do not print the comparison report")
    p_cmp.add_argument("--to-file", type=Path, help="save the comparison report to this file")

    args = parser.parse_args()
    if args.command == "build":
        copied = exporters[args.category](args.name)
        print(f"copied {copied} files")
        return 0
    else:
        report = compare(args.dir_a, args.dir_b, args.rel_tol, args.abs_tol, args.category)
        if not args.silent:
            print(report, end="")
        if args.to_file is not None:
            args.to_file.write_text(report, encoding="utf-8")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
