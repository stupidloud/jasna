"""Assemble a Windows (NVIDIA) distribution on top of an official Jasna release.

The public repository has neither the proprietary protection module nor the
maintainer's Nuitka release tooling, so this build reuses an official release
package for everything that is hard to obtain (CUDA torch, TensorRT, the
NVIDIA Video Effects SDK, the forked VALI and PyAV wheels, the full-build
ffmpeg, model weights) and replaces the application itself:

    <base>            official jasna-windows-<tag> package, extracted
    <python>          a CPython 3.13 installation (runtime DLLs, pyds, Lib\\)
    <site-packages>   a venv with this checkout's dependencies installed
    <source>          this checkout (the jasna package and its fatbins)
    <launcher>        jasna.exe built from launcher.c

Layout matches the official one: everything flat at the dist root next to
jasna.exe, Lib\\ holding the standard library that Nuitka would have compiled
in. Supporter-only pieces (unet-4x.onnx.enc, protection\\_core.dll) are left
out because the public checkout cannot drive them.
"""
from __future__ import annotations

import argparse
import compileall
import fnmatch
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Directories and files taken verbatim from the official package.
BASE_DIRS = (
    "torch",
    "torch_tensorrt",
    "tensorrt",
    "tensorrt_libs",
    "nvvfx",
    "python_vali",
    "vlc",
    "tools",
    "model_weights",
    "assets",
)
# Packages whose Python side Nuitka compiled into the official jasna.exe, so the
# base only holds their extension modules: the venv supplies the source and the
# base's CUDA builds of the binaries are laid over it afterwards. PyAV and the
# TensorRT bindings are in the same state, but their PyPI wheels match the base
# (same version, same features), so they come from the venv as a whole.
BASE_MERGE_DIRS = ("torchvision",)
BASE_FILE_GLOBS = ("*.dll",)
# Python runtime DLLs come from <python>, so drop the base copies.
BASE_FILE_SKIP = {"python3.dll", "python313.dll"}
MODEL_WEIGHTS_SKIP = {"unet-4x.onnx.enc"}

# site-packages entries the base already provides (or that are build-only).
SITE_PACKAGES_SKIP_GLOBS = (
    # torchgen/ and functorch/ stay: the official package compiled them into
    # jasna.exe, so they only exist in the venv (pinned to the base torch).
    "torch", "torch-*.dist-info",
    "torch_tensorrt", "torch_tensorrt-*.dist-info",
    "tensorrt", "tensorrt-*.dist-info", "tensorrt_libs", "tensorrt_cu*_libs-*.dist-info",
    "python_vali", "python_vali-*.dist-info",
    "nvidia", "nvidia_*", "triton", "triton-*.dist-info",
    "pip", "pip-*.dist-info", "setuptools", "setuptools-*.dist-info",
    "_distutils_hack", "distutils-precedence.pth", "pkg_resources",
    "wheel", "wheel-*.dist-info", "ziglang", "ziglang-*.dist-info",
    "jasna", "jasna-*.dist-info",
    "__pycache__",
)
STDLIB_SKIP = {
    "site-packages", "test", "tests", "idlelib", "ensurepip", "turtledemo",
    "pydoc_data", "lib2to3", "__pycache__", "venv",
}
PYD_SKIP_PREFIXES = ("_test", "winsound")
SOURCE_SKIP = {"__pycache__", "protection", "tests", "keytool"}


def log(msg: str) -> None:
    print(f"[assemble] {msg}", flush=True)


def _ignore_names(*names: str):
    skip = set(names)

    def ignore(_dir: str, entries: list[str]) -> set[str]:
        return {e for e in entries if e in skip}

    return ignore


def copy_tree(src: Path, dst: Path, *, ignore=None) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def take_from_base(base: Path, dist: Path, *, move: bool) -> None:
    for name in BASE_DIRS:
        src = base / name
        if not src.is_dir():
            log(f"base has no {name}/ (skipped)")
            continue
        dst = dist / name
        if move:
            shutil.move(str(src), str(dst))
        else:
            copy_tree(src, dst)
        log(f"base: {name}/")
    for entry in sorted(base.iterdir()):
        if not entry.is_file() or entry.name.lower() in BASE_FILE_SKIP:
            continue
        if any(fnmatch.fnmatch(entry.name.lower(), g) for g in BASE_FILE_GLOBS):
            shutil.copy2(entry, dist / entry.name)
    for name in MODEL_WEIGHTS_SKIP:
        gated = dist / "model_weights" / name
        if gated.exists():
            gated.unlink()
            log(f"dropped supporter-only {name}")


def merge_from_base(base: Path, dist: Path) -> None:
    for name in BASE_MERGE_DIRS:
        src = base / name
        dst = dist / name
        if not src.is_dir():
            log(f"base has no {name}/ to merge (skipped)")
            continue
        if not dst.is_dir():
            raise SystemExit(f"{name}/ must come from site-packages before the base binaries are merged")
        copy_tree(src, dst, ignore=_ignore_names("__pycache__"))
        log(f"base binaries merged into {name}/")
    _relabel_torchvision_build(dist)


def _relabel_torchvision_build(dist: Path) -> None:
    # The venv's torchvision is the CPU wheel of the same release; with the
    # base's CUDA _C.pyd in place its version string should say so too.
    version_py = dist / "torchvision" / "version.py"
    torch_version_py = dist / "torch" / "version.py"
    if not version_py.is_file() or not torch_version_py.is_file():
        return
    torch_local = ""
    for line in torch_version_py.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            _, _, value = line.partition("=")
            torch_local = value.strip().strip("'\"").partition("+")[2]
    if not torch_local:
        return
    text = version_py.read_text(encoding="utf-8")
    relabeled = text.replace("+cpu'", f"+{torch_local}'").replace('+cpu"', f'+{torch_local}"')
    if relabeled != text:
        version_py.write_text(relabeled, encoding="utf-8")
        log(f"torchvision relabeled as +{torch_local}")


def take_python_runtime(python_root: Path, dist: Path) -> None:
    for name in ("python313.dll", "python3.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        src = python_root / name
        if src.is_file():
            shutil.copy2(src, dist / name)
        else:
            log(f"python runtime has no {name} (skipped)")
    dlls = python_root / "DLLs"
    for entry in sorted(dlls.iterdir()):
        if entry.suffix.lower() not in {".pyd", ".dll"}:
            continue
        if entry.name.lower().startswith(PYD_SKIP_PREFIXES):
            continue
        shutil.copy2(entry, dist / entry.name)
    tcl = python_root / "tcl"
    if tcl.is_dir():
        copy_tree(tcl, dist / "tcl")
    lib = python_root / "Lib"
    copy_tree(lib, dist / "Lib", ignore=_ignore_names(*STDLIB_SKIP))
    log(f"python runtime: {platform.python_version()} from {python_root}")


def take_site_packages(site_packages: Path, dist: Path) -> None:
    taken = 0
    for entry in sorted(site_packages.iterdir()):
        if any(fnmatch.fnmatch(entry.name, g) for g in SITE_PACKAGES_SKIP_GLOBS):
            continue
        if entry.suffix == ".pth":
            continue
        dst = dist / entry.name
        if entry.is_dir():
            copy_tree(entry, dst, ignore=_ignore_names("__pycache__"))
        else:
            shutil.copy2(entry, dst)
        taken += 1
    log(f"site-packages: {taken} entries from {site_packages}")


def take_source(source: Path, dist: Path) -> None:
    pkg = source / "jasna"
    copy_tree(pkg, dist / "jasna", ignore=_ignore_names(*SOURCE_SKIP))
    # jasna/protection is an empty submodule in the public checkout; the code
    # imports it lazily, so an absent package simply disables gated models.
    (dist / "jasna" / "protection").mkdir(exist_ok=True)
    for fatbin in sorted((pkg / "media").glob("*.fatbin")):
        shutil.copy2(fatbin, dist / fatbin.name)
    log("source: jasna/ and fatbins")


def write_build_info(dist: Path, source: Path, base_tag: str, python_root: Path) -> None:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(source), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    info = "\n".join(
        [
            "Jasna overlay build (Windows, NVIDIA)",
            f"source commit: {git('rev-parse', 'HEAD')}",
            f"source describe: {git('describe', '--tags', '--always', '--dirty')}",
            f"base release: {base_tag}",
            f"python: {platform.python_version()} ({python_root})",
            "",
            "Built from the public source on top of the official release package.",
            "Supporter-only models are not available in this build.",
        ]
    )
    (dist / "BUILD_INFO.txt").write_text(info + "\n", encoding="utf-8")


def precompile(dist: Path) -> None:
    # torch ships __pycache__ already; everything else gets one so first start
    # does not compile thousands of modules.
    skip = {dist / "torch", dist / "Lib", dist / "tcl"}
    targets = [
        p for p in dist.iterdir()
        if p.is_dir() and p not in skip and not p.name.endswith(".dist-info")
    ]
    targets.append(dist / "Lib")
    ok = compileall.compile_dir  # noqa: E731 - keep the call sites short
    for target in targets:
        ok(str(target), quiet=2, workers=0, ddir=None)
    log("precompiled bytecode")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path, help="extracted official Windows package")
    ap.add_argument("--base-tag", default="unknown", help="release tag the base came from (for BUILD_INFO)")
    ap.add_argument("--python", required=True, type=Path, help="CPython 3.13 installation root")
    ap.add_argument("--site-packages", required=True, type=Path, help="venv site-packages with the dependencies")
    ap.add_argument("--source", required=True, type=Path, help="repository checkout")
    ap.add_argument("--launcher", required=True, type=Path, help="jasna.exe built from launcher.c")
    ap.add_argument("--out", required=True, type=Path, help="dist directory to create")
    ap.add_argument("--move-base", action="store_true", help="move base directories instead of copying (saves disk)")
    ap.add_argument("--no-precompile", action="store_true")
    args = ap.parse_args()

    if sys.version_info[:2] != (3, 13):
        raise SystemExit("run with the same CPython 3.13 that is passed as --python")

    dist: Path = args.out
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir(parents=True)

    take_from_base(args.base.resolve(), dist, move=args.move_base)
    take_python_runtime(args.python.resolve(), dist)
    take_site_packages(args.site_packages.resolve(), dist)
    merge_from_base(args.base.resolve(), dist)
    take_source(args.source.resolve(), dist)
    shutil.copy2(args.launcher, dist / "jasna.exe")
    write_build_info(dist, args.source.resolve(), args.base_tag, args.python.resolve())
    if not args.no_precompile:
        precompile(dist)

    total = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
    log(f"done: {dist} ({total / 2**30:.2f} GiB, {sum(1 for _ in dist.rglob('*'))} entries)")


if __name__ == "__main__":
    main()
