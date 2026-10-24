"""Download one big file with many parallel connections; resumes if interrupted.

    python scripts/fetch.py <url> <out file> [--conns 16]

The file is split into ranges; each finished range is kept as <out>.parts/<n>,
so a rerun only fetches what is missing. Needs a server that supports byte
ranges (Zenodo, S3, Hugging Face do).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 circuit2dsn-fetch"}


def final_url_and_size(url: str) -> tuple[str, int]:
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
        size = int(cr.split("/")[-1]) if "/" in cr else int(r.headers["Content-Length"])
        return r.geturl(), size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out", type=Path)
    ap.add_argument("--conns", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=32)
    args = ap.parse_args()
    if args.out.exists():
        print(f"{args.out} already there")
        return
    url, size = final_url_and_size(args.url)
    chunk = args.chunk_mb << 20
    parts = args.out.with_name(args.out.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [(i, s, min(s + chunk, size) - 1) for i, s in enumerate(range(0, size, chunk))]
    done_bytes = [sum((parts / str(i)).stat().st_size for i, s, e in ranges if (parts / str(i)).exists())]
    lock = threading.Lock()
    t0 = time.time()

    def get(job):
        i, s, e = job
        p = parts / str(i)
        if p.exists() and p.stat().st_size == e - s + 1:
            return
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={s}-{e}"})
                with urllib.request.urlopen(req, timeout=120) as r, open(p.with_suffix(".tmp"), "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        with lock:
                            done_bytes[0] += len(b)
                if p.with_suffix(".tmp").stat().st_size != e - s + 1:
                    raise IOError("short read")
                p.with_suffix(".tmp").replace(p)
                return
            except Exception as ex:                       # noqa: BLE001 - network: retry
                time.sleep(3 * (attempt + 1))
                if attempt == 7:
                    print(f"range {i} failed: {ex}", file=sys.stderr)
                    raise

    stop = threading.Event()

    def report():
        while not stop.wait(30):
            mb = done_bytes[0] / 1e6
            print(f"  {mb:,.0f} / {size / 1e6:,.0f} MB  ({mb / max(time.time() - t0, 1):.1f} MB/s)", flush=True)

    threading.Thread(target=report, daemon=True).start()
    with ThreadPoolExecutor(args.conns) as ex:
        list(ex.map(get, ranges))
    stop.set()
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for i, _, _ in ranges:
            with open(parts / str(i), "rb") as pf:
                shutil.copyfileobj(pf, f)
    tmp.replace(args.out)
    shutil.rmtree(parts)
    print(f"done: {args.out} ({size / 1e6:,.0f} MB in {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
