#!/usr/bin/env python3
"""Build a pre-resized JPEG cache of Nantes frames on fast storage (the raw data sits on a spinning HDD, which starves
the GPU: 2026-09-08 measurement, iowait 13 %, GPU 0 %). Layout mirrors the source: <cache>/<plane>/<video>/<RUN>.jpg,
grayscale, <size>x<size>, JPEG quality 92. The dataset (`NantesKineticFrames(cache_dir=...)`) uses a cached file when
present and falls back to the original otherwise, so a partial cache is safe.

    python scripts/build_nantes_cache.py --planes embryo_dataset --size 250 --cache ~/.cache/stseg/nantes_250 --only-manifest data/derived/nantes_manifest_F0.csv
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT_DATA = Path("data/raw/nantes_embryo_dataset")


def convert(args: tuple[str, str, int]) -> int:
    src, dst, size = args
    dst = Path(dst)
    if dst.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.open(src).convert("L").resize((size, size), Image.Resampling.BILINEAR).save(dst, "JPEG", quality=92)
        return 1
    except Exception as e:  # truncated files are quarantined already; be defensive anyway
        print("skip", src, e)
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planes", nargs="+", default=["embryo_dataset"])
    ap.add_argument("--size", type=int, default=250)
    ap.add_argument("--cache", default=str(Path("data/cache") / "nantes_250"))
    ap.add_argument("--only-manifest", default=None, help="restrict to the frames listed in this manifest (in-window frames only)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    cache = Path(a.cache)
    if a.only_manifest:
        m = pd.read_csv(a.only_manifest)
        m = m[m.phase.notna() & (m.is_blank != True)]  # noqa: E712
        rel = [Path(p).relative_to(ROOT_DATA / "embryo_dataset") for p in m.path]
    else:
        rel = [p.relative_to(ROOT_DATA / "embryo_dataset") for p in (ROOT_DATA / "embryo_dataset").rglob("*.jpeg")]
    jobs = []
    for plane in a.planes:
        for r in rel:
            src = ROOT_DATA / plane / r
            jobs.append((str(src), str(cache / plane / r.parent / (r.stem + ".jpg")), a.size))
    print(f"{len(jobs)} files -> {cache}")
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, n in enumerate(ex.map(convert, jobs, chunksize=256), 1):
            done += n
            if i % 50000 == 0:
                print(f"  {i}/{len(jobs)} ({done} written)", flush=True)
    print(f"done: {done} written, {len(jobs) - done} already present/skipped")


if __name__ == "__main__":
    main()
