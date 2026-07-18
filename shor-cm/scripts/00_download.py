#!/usr/bin/env python3
"""Download MAFAULDA per-class zips (resumable) and extract.
Usage: python scripts/00_download.py [--only normal imbalance ...]

Mirror layout changes occasionally. If the primary base 404s, the agent
running this MUST stop and report the URLs tried; do NOT substitute a
different dataset silently.
"""
import argparse, subprocess, sys, zipfile
from pathlib import Path

BASE = "http://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/"
# TODO(agent): verify at runtime by fetching the index page first.
ZIPS = ["normal.zip", "horizontal-misalignment.zip", "vertical-misalignment.zip",
        "imbalance.zip", "underhang.zip", "overhang.zip"]

DATA = Path("data")
RAW = DATA / "raw"
EXTRACT = DATA / "extracted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    EXTRACT.mkdir(parents=True, exist_ok=True)
    zips = [z for z in ZIPS if args.only is None
            or z.split(".")[0] in args.only]
    failed = []
    for z in zips:
        out = RAW / z
        cmd = ["wget", "-c", "-q", "--show-progress", "--tries=5",
               "--timeout=60", "-O", str(out), BASE + z]
        print("GET", BASE + z)
        if subprocess.run(cmd).returncode != 0:
            failed.append(BASE + z)
            continue
        try:
            with zipfile.ZipFile(out) as zf:
                zf.extractall(EXTRACT)
        except zipfile.BadZipFile:
            failed.append(str(out) + " (bad zip, delete and retry)")
    if failed:
        print("FAILED:\n" + "\n".join(failed), file=sys.stderr)
        sys.exit(1)
    n = sum(1 for _ in EXTRACT.rglob("*.csv"))
    print(f"extracted CSV files: {n} (expect ~1951 total across all classes)")


if __name__ == "__main__":
    main()
