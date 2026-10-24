"""Read a few members of a large remote zip without downloading all of it
(HTTP range requests). Used to fetch sample CGHD files for testing.

    python scripts/remote_zip.py <url> --list 30
    python scripts/remote_zip.py <url> --get "drafter_1/annotations/C1_D1_P1.xml" --out data/cghd_sample
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path


class HttpFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 7:
                    raise
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--list", type=int, default=0)
    ap.add_argument("--get", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("data/cghd_sample"))
    args = ap.parse_args()
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(args.url), buffer_size=1 << 20))
    names = z.namelist()
    if args.list:
        print(len(names), "members")
        for n in names[: args.list]:
            print(n)
    for name in args.get:
        dst = args.out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(name))
        print("got", dst)


if __name__ == "__main__":
    main()
