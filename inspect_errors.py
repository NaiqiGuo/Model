"""Build error result folders, or compare two of them.

build: copy System ID error results into <name>_bridge/ and <name>_frame/.

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
    python inspect_errors.py build myfoldername
    python inspect_errors.py heatmaps myfoldername
    python inspect_errors.py compare chrystal_bridge naiqi_bridge
    python inspect_errors.py compare chrystal_bridge naiqi_bridge --rel-tol 1e-8

Exit status for compare: 0 if the folders match within tolerance, 1 otherwise.
"""

import argparse
import filecmp
import math
import re
import shutil
import sys
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


def build(name: str) -> int:
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
        print(f"{out_root.name}: done")
    return copied


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


def compare(dir_a: Path, dir_b: Path, rel_tol: float, abs_tol: float) -> int:
    for d in (dir_a, dir_b):
        if not d.is_dir():
            sys.exit(f"error: not a directory: {d}")

    files_a, files_b = list_files(dir_a), list_files(dir_b)
    only_a = sorted(files_a.keys() - files_b.keys())
    only_b = sorted(files_b.keys() - files_a.keys())
    common = sorted(files_a.keys() & files_b.keys())

    identical, noise_only, real = 0, 0, []
    for rel in common:
        if filecmp.cmp(files_a[rel], files_b[rel], shallow=False):
            identical += 1
            continue
        n_bad, max_abs = compare_values(
            files_a[rel].read_text(errors="replace"),
            files_b[rel].read_text(errors="replace"),
            rel_tol, abs_tol,
        )
        if n_bad == 0:
            noise_only += 1
        else:
            real.append((rel, n_bad, max_abs))

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

    if real:
        print("\nReal differences:")
        for rel, n_bad, max_abs in real:
            detail = f"max |diff| = {max_abs:.3g}" if max_abs is not None else "token count mismatch"
            print(f"  {rel}: {n_bad} value(s) differ, {detail}")
    if only_a:
        print("\nOnly in A:")
        for rel in only_a:
            print(f"  {rel}")
    if only_b:
        print("\nOnly in B:")
        for rel in only_b:
            print(f"  {rel}")

    return 0 if not (real or only_a or only_b) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="create <name>_bridge/ and <name>_frame/ from System ID errors")
    p_build.add_argument("name", help="prefix for the output folders, e.g. 'myfoldername' -> myfoldername_bridge/, myfoldername_frame/")

    p_heat = sub.add_parser("heatmaps", help="copy all System ID heatmaps into <name>_heatmaps/")
    p_heat.add_argument("name", help="prefix for the output folder, e.g. 'myfoldername' -> myfoldername_heatmaps/")

    p_cmp = sub.add_parser("compare", help="summarize differences between two folders")
    p_cmp.add_argument("dir_a", type=Path)
    p_cmp.add_argument("dir_b", type=Path)
    p_cmp.add_argument("--rel-tol", type=float, default=1e-6, help="relative tolerance for numbers (default: 1e-6")
    p_cmp.add_argument("--abs-tol", type=float, default=1e-9, help="absolute tolerance for numbers (default: 1e-9)")

    args = parser.parse_args()
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
