#!/usr/bin/env python
"""Run DynaPyt dynamic analysis over a scoped subset of the test suite.

DynaPyt instruments Python files in place (keeping .py.orig backups), so this
helper operates on a throwaway copy of the repository and never touches the
real src/ tree.

Workflow:
    1. copy the repository into .dynapyt/ (via `git archive`)
    2. instrument .dynapyt/src with the selected DynaPyt analyses
    3. run a scoped pytest subset against the instrumented copy
    4. capture the analysis output into .dynapyt_report/
    5. remove the throwaway copy

Usage:
    uv run python tools/run_dynapyt.py
    uv run python tools/run_dynapyt.py --tests tests/test_detector.py tests/test_readings.py
    uv run python tools/run_dynapyt.py --analysis dynapyt.analyses.TraceAll.TraceAll
    uv run python tools/run_dynapyt.py --keep
"""
import argparse
import io
import os
import shutil
import subprocess  # nosec B404 -- required to run git archive and DynaPyt
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKDIR = ROOT / ".dynapyt"
REPORT_DIR = ROOT / ".dynapyt_report"

DEFAULT_TESTS = ["tests/test_detector.py", "tests/test_readings.py"]
DEFAULT_ANALYSES = ["dynapyt.analyses.BranchCoverage.BranchCoverage"]

IGNORE_COPY = {
    ".git",
    ".dynapyt",
    ".dynapyt_report",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv315",
    ".coverage",
    "__pycache__",
    "build",
    "dist",
    "graphify-out",
    "htmlcov",
}

ENTRY_TEMPLATE = """\
import logging
import pytest

logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-p", "no:odl_plugins", *{tests!r}]))
"""


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests", nargs="+", default=DEFAULT_TESTS,
        help="test paths relative to the repository root",
    )
    parser.add_argument(
        "--analysis", nargs="+", default=DEFAULT_ANALYSES,
        help="full dotted paths of DynaPyt analysis classes",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="keep the .dynapyt copy for debugging",
    )
    return parser.parse_args(argv)


def make_repo_copy():
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True)
    try:
        # nosec B603 B607 -- list-form subprocess call, no shell=True; the
        # command is a fixed literal from this repository.
        proc = subprocess.run(["git", "archive", "HEAD"], cwd=ROOT, capture_output=True, check=True)  # nosec B603 B607
    except (subprocess.CalledProcessError, FileNotFoundError):
        shutil.copytree(
            ROOT, WORKDIR, ignore=shutil.ignore_patterns(*IGNORE_COPY)
        )
        return
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:*") as tar:
        # nosec B202 -- archive is produced locally by `git archive HEAD` from
        # this repository, so its members are trusted.
        tar.extractall(WORKDIR)  # nosec B202


def instrument(analyses):
    cmd = [
        sys.executable, "-m", "dynapyt.run_instrumentation",
        "--directory", str(WORKDIR / "src"), "--analysis", *analyses,
    ]
    # nosec B603 -- list-form subprocess call, no shell=True; command is
    # assembled from trusted local values only.
    subprocess.run(cmd, cwd=WORKDIR, check=True)  # nosec B603


def write_entry(tests):
    entry = WORKDIR / "entry_dynapyt.py"
    entry.write_text(ENTRY_TEMPLATE.format(tests=tests), encoding="utf-8")
    return entry


def run_analysis(entry, analyses):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORKDIR) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-m", "dynapyt.run_analysis",
        "--entry", str(entry), "--analysis", *analyses,
    ]
    # nosec B603 -- list-form subprocess call, no shell=True; command is
    # assembled from trusted local values only.
    return subprocess.run(cmd, cwd=WORKDIR, env=env, capture_output=True, text=True)  # nosec B603


def write_report(proc, analyses):
    if REPORT_DIR.exists():
        shutil.rmtree(REPORT_DIR)
    REPORT_DIR.mkdir(parents=True)
    (REPORT_DIR / "dynapyt_run.log").write_text(
        proc.stdout + proc.stderr, encoding="utf-8"
    )
    branches = [
        line for line in proc.stdout.splitlines() if line.startswith("Branch ")
    ]
    (REPORT_DIR / "branch_coverage.txt").write_text(
        "\n".join(branches) + "\n", encoding="utf-8"
    )
    (REPORT_DIR / "summary.txt").write_text(
        "\n".join([
            f"analyses: {', '.join(analyses)}",
            f"branch events: {len(branches)}",
            f"exit code: {proc.returncode}",
        ]) + "\n",
        encoding="utf-8",
    )


def write_error_report(analyses, error):
    """Write a minimal report so the artifact upload always has files.

    Called when instrumentation or analysis fails before a normal report can
    be produced, so the CI upload step never hits "no files found".
    """
    if REPORT_DIR.exists():
        shutil.rmtree(REPORT_DIR)
    REPORT_DIR.mkdir(parents=True)
    (REPORT_DIR / "dynapyt_run.log").write_text(
        f"ERROR: {error}\n", encoding="utf-8"
    )
    (REPORT_DIR / "branch_coverage.txt").write_text("", encoding="utf-8")
    (REPORT_DIR / "summary.txt").write_text(
        "\n".join([
            f"analyses: {', '.join(analyses)}",
            "branch events: 0",
            "exit code: 1",
            f"error: {error}",
        ]) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    args = parse_args(argv)
    try:
        make_repo_copy()
        print("Instrumenting src/ copy with: " + ", ".join(args.analysis))
        instrument(args.analysis)
        entry = write_entry(args.tests)
        print("Running: " + " ".join(args.tests))
        proc = run_analysis(entry, args.analysis)
        write_report(proc, args.analysis)
        print(f"DynaPyt analysis finished (exit code {proc.returncode}); "
              f"report in {REPORT_DIR}/")
        return proc.returncode
    except Exception as exc:  # noqa: BLE001 -- report failures, don't crash CI
        print(f"DynaPyt analysis failed: {exc}", file=sys.stderr)
        write_error_report(args.analysis, exc)
        return 1
    finally:
        if not args.keep:
            shutil.rmtree(WORKDIR, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
