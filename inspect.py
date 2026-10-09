"""Export and compare prediction errors, training data, systems, and environments.

build: a bare name creates a fresh runs/<timestamp>/ directory. An explicit
path prefix is used exactly as supplied (including by the full-run script).
The actual output paths are printed. Compare uses only the supplied directories,
without searching for a latest run.

build: preserve error files in <name>_bridge/ and <name>_frame/; add training/
and systems/<event>/{A,B,C,D}.csv alongside them. Record all installed Python
packages and interpreter metadata in <name>_environment/. Systems are raw
identified matrices before prediction-time stabilization. Existing files are
not deleted. Missing required source data aborts build rather than silently
creating an incomplete snapshot.

The output follows this layout:
    <name>_<structure>/<structure>/<quantity>/<source>/
        <structure>_<source>_<quantity>_event_<N>.csv
e.g. chrystal_frame/frame/acceleration/elastic/frame_elastic_acceleration_event_226.csv

Each file is copied from:
    System ID/<structure>/<quantity>/<source>/System ID Results/errors/<N>.csv
e.g. System ID/frame/acceleration/elastic/System ID Results/errors/226.csv

heatmaps: copy every heatmap.png from System ID/ into one flat folder,
<name>_heatmaps/. Each file is renamed to include its structure, quantity,
and source, e.g.
    <name>_heatmaps/frame_acceleration_elastic_heatmap.png

compare: summarize how two folders differ. Floating-point noise is not
treated as a difference. Files that differ only by floating-point noise
are counted but not listed.

Usage:
    python inspect.py build myfoldername
    python inspect.py heatmaps myfoldername
    python inspect.py compare chrystal_bridge runs/<timestamp>/naiqi_bridge
    python inspect.py compare chrystal_bridge runs/<timestamp>/naiqi_bridge --rel-tol 1e-8

Exit status for compare: 0 if the folders match within tolerance, 1 otherwise.
"""

# Preserve the standard-library API when dependencies import inspect.
# Only direct execution runs the result-inspection CLI below.
from unicodedata import name


if __name__ != "__main__":
    import os as _os
    _stdlib_inspect = _os.path.join(_os.path.dirname(_os.__file__), "inspect.py")
    with open(_stdlib_inspect, "rb") as _source:
        exec(compile(_source.read(), _stdlib_inspect, "exec"), globals())
else:


    import argparse
    import filecmp
    import math
    import re
    import shutil
    import sys
    import json
    import os
    import pickle
    import platform
    import subprocess
    from datetime import datetime, timezone
    from pathlib import Path

    SID_DIR = Path("System ID")

    STRUCTURES = {
        "bridge": range(1, 23),      # events 1-22
        "frame": range(226, 248),    # events 226-247
    }
    QUANTITIES = ["acceleration", "displacement"]
    SOURCES = ["elastic", "field", "inelastic"]

    JUNK_NAMES = {".DS_Store"}
    TOKEN_SPLIT = re.compile(r"[,\s]+")


    def build_prefix(name: str) -> str:
        """Bare names get a fresh run; explicit paths are used verbatim."""
        if not name or not Path(name).name or name in (".", ".."):
            raise ValueError("Provide an export name such as naiqi, or a path ending in that name")
        if os.path.dirname(name):
            return name
        from zoneinfo import ZoneInfo

        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / stamp
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            sys.exit(f"Run directory already exists: {run_dir}. Retry in a second; existing results were not changed.")
        return str(run_dir / name)


    def build(name: str) -> int:
        import numpy as np

        name = build_prefix(name)
        print(f"Export prefix: {name}")
        save_environment(name)
        copied = 0
        for structure, events in STRUCTURES.items():
            out_root = Path(f"{name}_{structure}")
            for quantity in QUANTITIES:
                for source in SOURCES:
                    out_dir = out_root / structure / quantity / source
                    out_dir.mkdir(parents=True, exist_ok=True)
                    errors_dir = SID_DIR / structure / quantity / source / "System ID Results" / "errors"
                    for n in events:
                        src = errors_dir / f"{n}.csv"
                        dst = out_dir / f"{structure}_{source}_{quantity}_event_{n}.csv"
                        shutil.copyfile(src, dst)
                        copied += 1
                        for item in ("dt", "time", "ground", "structure"):
                            training_src = SID_DIR / structure / quantity / source / "System ID Training Data" / item / f"{n}.csv"
                            training_dst = out_dir / "training" / item / f"{n}.csv"
                            training_dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(training_src, training_dst)
                            copied += 1
                        system_src = errors_dir.parent / "system realization" / f"{n}.pkl"
                        # Only load the local pipeline's trusted pickle artifacts.
                        with system_src.open("rb") as stream:
                            matrices = pickle.load(stream)
                        if len(matrices) != 4:
                            raise ValueError(f"Expected A, B, C, D in {system_src}")
                        system_dst = out_dir / "systems" / str(n)
                        system_dst.mkdir(parents=True, exist_ok=True)
                        for label, matrix in zip("ABCD", matrices):
                            matrix = np.asarray(matrix)
                            if matrix.ndim != 2:
                                raise ValueError(f"{system_src}: {label} must be a matrix")
                            np.savetxt(system_dst / f"{label}.csv", matrix, delimiter=",")
                            copied += 1
            print(f"{out_root}: done")
        copied += heatmaps(name)
        return copied


    def save_environment(name):
        """Record the interpreter executing build, rather than an unrelated pip."""
        out = Path(f"{name}_environment")
        packages = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
            check=True, capture_output=True, text=True,
        )
        installed = sorted(json.loads(packages.stdout), key=lambda p: p["name"].lower())
        out.mkdir(parents=True, exist_ok=True)
        (out / "packages.txt").write_text(
            "".join(f"{p['name']}=={p['version']}\n" for p in installed), encoding="utf-8"
        )
        metadata = {
            "python_executable": sys.executable, "python_version": platform.python_version(),
            "platform": platform.platform(), "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "conda_prefix": os.environ.get("CONDA_PREFIX"), "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_count": len(installed),
            "systems_note": "Raw identified A/B/C/D, before prediction-time stabilization. Different state coordinates can produce different matrices with equivalent input/output behavior.",
        }
        (out / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"{out}: recorded {len(installed)} Python packages")


    def environment_path(root):
        for suffix in ("_bridge", "_frame"):
            if root.name.endswith(suffix):
                return root.with_name(root.name[:-len(suffix)] + "_environment")
        return root.with_name(root.name + "_environment")


    def compare_environments(dir_a, dir_b):
        print("\nEnvironment (informational; does not affect numeric comparison status):")
        snapshots = []
        for label, root in (("A", dir_a), ("B", dir_b)):
            env = environment_path(root)
            if not all((env / f).is_file() for f in ("environment.json", "packages.txt")):
                print(f"  {label}: environment information not provided ({env})")
                snapshots.append(None)
                continue
            meta = json.loads((env / "environment.json").read_text())
            print(f"  {label}: Python {meta['python_version']}; {meta['python_executable']}")
            print(f"  {label}: platform={meta.get('platform')}; conda={meta.get('conda_prefix')}; venv={meta.get('virtual_env')}")
            packages = {}
            for line in (env / "packages.txt").read_text().splitlines():
                name, version = line.split("==", 1)
                packages[re.sub(r"[-_.]+", "-", name).lower()] = version
            snapshots.append((meta, packages))
        if any(s is None for s in snapshots):
            return
        (meta_a, a), (meta_b, b) = snapshots
        print(f"  Python version match: {meta_a['python_version'] == meta_b['python_version']}")
        different = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
        print(f"  Packages: {len(different)} version differences, {len(a.keys()-b.keys())} only in A, {len(b.keys()-a.keys())} only in B")
        for k in different:
            print(f"    {k}: A={a[k]}, B={b[k]}")
        for k in sorted(a.keys() - b.keys()):
            print(f"    only in A: {k}=={a[k]}")
        for k in sorted(b.keys() - a.keys()):
            print(f"    only in B: {k}=={b[k]}")


    def heatmaps(name: str) -> int:
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


    def is_junk(path: Path) -> bool:
        """macOS metadata: .DS_Store, and AppleDouble ._* files / __MACOSX/ folders."""
        return path.name in JUNK_NAMES or path.name.startswith("._") or "__MACOSX" in path.parts


    def list_files(root: Path) -> dict[str, Path]:
        return {
            str(p.relative_to(root)): p
            for p in root.rglob("*")
            if p.is_file() and not is_junk(p.relative_to(root))
        }


    def compare_values(a_text: str, b_text: str, rel_tol: float, abs_tol: float) -> tuple[int, float | None]:
        """Return (number of values that differ beyond tolerance, largest absolute difference).

        Tokens are compared as floats when both sides parse as numbers, otherwise as strings.
        A token-count mismatch counts every surplus token as a difference.
        """
        a_tok = [t for t in TOKEN_SPLIT.split(a_text.strip()) if t]
        b_tok = [t for t in TOKEN_SPLIT.split(b_text.strip()) if t]
        n_bad, max_abs = abs(len(a_tok) - len(b_tok)), None
        for x, y in zip(a_tok, b_tok):
            try:
                fx, fy = float(x), float(y)
            except ValueError:
                if x != y:
                    n_bad += 1
                continue
            if not math.isclose(fx, fy, rel_tol=rel_tol, abs_tol=abs_tol):
                n_bad += 1
                diff = abs(fx - fy)
                max_abs = diff if max_abs is None else max(max_abs, diff)
        return n_bad, max_abs


    def category(rel):
        parts = Path(rel).parts
        return "training" if "training" in parts else "systems" if "systems" in parts else "errors"


    def compare_arrays(path_a, path_b, rel_tol, abs_tol):
        """Compare numeric CSV shape and values, using symmetric math.isclose tolerance."""
        import numpy as np

        arrays = []
        for path in (path_a, path_b):
            with path.open() as stream:
                first = stream.readline()
            arrays.append(np.loadtxt(path, delimiter="," if "," in first else None, ndmin=2, dtype=complex))
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
            return False, ""
        index = np.unravel_index(np.argmax(delta), delta.shape)
        denominator = np.linalg.norm(a)
        relative = f"{np.linalg.norm(a-b)/denominator:.6g}" if denominator else "undefined (A norm is zero)"
        return True, (f"{n_bad}/{a.size} values differ ({100*n_bad/a.size:.3g}%); shape={a.shape}; "
                      f"max |diff|={delta[index]:.6g} at row={index[0]}, col={index[1]} (zero-based); relative norm vs A={relative}")


    def compare(dir_a: Path, dir_b: Path, rel_tol: float, abs_tol: float) -> int:
        for d in (dir_a, dir_b):
            if not d.is_dir():
                sys.exit(f"error: not a directory: {d}")

        files_a, files_b = list_files(dir_a), list_files(dir_b)
        only_a = sorted(files_a.keys() - files_b.keys())
        only_b = sorted(files_b.keys() - files_a.keys())
        common = sorted(files_a.keys() & files_b.keys())

        identical, noise_only, real = 0, 0, []
        counts = {k: {s: 0 for s in ("identical", "within tolerance", "different", "only A", "only B")} for k in ("errors", "training", "systems")}
        for rel in common:
            group = counts[category(rel)]
            if files_a[rel].suffix.lower() == ".csv":
                try:
                    different, detail = compare_arrays(files_a[rel], files_b[rel], rel_tol, abs_tol)
                except (ValueError, OSError) as exc:
                    different, detail = True, f"could not read numeric CSV: {exc}"
            else:
                n_bad, max_abs = compare_values(files_a[rel].read_text(errors="replace"), files_b[rel].read_text(errors="replace"), rel_tol, abs_tol)
                different, detail = bool(n_bad), f"{n_bad} values differ; max |diff|={max_abs}"
            if different:
                real.append((rel, detail))
                group["different"] += 1
            elif filecmp.cmp(files_a[rel], files_b[rel], shallow=False):
                identical += 1
                group["identical"] += 1
            else:
                noise_only += 1
                group["within tolerance"] += 1
        for rel in only_a:
            counts[category(rel)]["only A"] += 1
        for rel in only_b:
            counts[category(rel)]["only B"] += 1

        print(f"A: {dir_a}")
        print(f"B: {dir_b}")
        print(f"tolerance: rel_tol={rel_tol:g}, abs_tol={abs_tol:g}")
        print()
        print(f"files in both:           {len(common)}")
        print(f"  identical:             {identical}")
        print(f"  float noise only:      {noise_only}")
        print(f"  real differences:      {len(real)}")
        print(f"only in A:               {len(only_a)}")
        print(f"only in B:               {len(only_b)}")
        print("\nBy category:")
        for name, stats in counts.items():
            print(f"  {name}: " + ", ".join(f"{key}={value}" for key, value in stats.items()))
            if not any(stats.values()):
                print("    No data provided in either folder; not compared.")
        print("  systems: raw A/B/C/D before prediction stabilization; matrix differences do not alone establish different input/output behavior.")

        if real:
            print("\nReal differences:")
            for rel, detail in real:
                print(f"  {rel}: {detail}")
        if only_a:
            print("\nOnly in A:")
            for rel in only_a:
                print(f"  {rel}")
        if only_b:
            print("\nOnly in B:")
            for rel in only_b:
                print(f"  {rel}")

        compare_environments(dir_a, dir_b)
        return 0 if not (real or only_a or only_b) else 1


    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        sub = parser.add_subparsers(dest="command", required=True)

        p_build = sub.add_parser("build", help="export errors, training, systems, and the active Python environment")
        p_build.add_argument("name", help="bare name: new runs/<timestamp>/<name>_* folders; explicit path: use that prefix exactly")

        p_heat = sub.add_parser("heatmaps", help="copy all System ID heatmaps into <name>_heatmaps/")
        p_heat.add_argument("name", help="prefix for the output folder, e.g. 'myfoldername' -> myfoldername_heatmaps/")

        p_cmp = sub.add_parser("compare", help="summarize differences between two folders")
        p_cmp.add_argument("dir_a", type=Path)
        p_cmp.add_argument("dir_b", type=Path)
        p_cmp.add_argument("--rel-tol", type=float, default=1e-6, help="relative tolerance for numbers (default: 1e-6")
        p_cmp.add_argument("--abs-tol", type=float, default=1e-9, help="absolute tolerance for numbers (default: 1e-9)")

        args = parser.parse_args()
        if args.command == "compare" and (not math.isfinite(args.rel_tol) or not math.isfinite(args.abs_tol) or args.rel_tol < 0 or args.abs_tol < 0):
            parser.error("tolerances must be finite and nonnegative")
        if args.command == "build":
            copied = build(args.name)
            print(f"copied {copied} files")
        elif args.command == "heatmaps":
            copied = heatmaps(args.name)
            print(f"copied {copied} files")
        else:
            sys.exit(compare(args.dir_a, args.dir_b, args.rel_tol, args.abs_tol))


    if __name__ == "__main__":
        main()
