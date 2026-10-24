"""Review dashboard: upload a schematic, check components and connections in
the browser, then export EDIF for OrCAD Capture.

    python scripts/review.py                         # dashboard: upload / open
    python scripts/review.py my_circuit.pdf --page 1 # straight into one page
    python scripts/review.py --pcbnet 0002           # demo on a PCBnet design
    python scripts/review.py --resume outputs/review/my_circuit_p1

Opens http://127.0.0.1:8765 (local only). "Files" in the top bar uploads a
PDF / PNG / JPG (runs the detectors and wire tracing) or reopens an earlier
review. Keys: Y yes, E fix, N not a part, S skip, U undo, F zoom, A add part.
Progress is saved after every click in outputs/review/<name>/session.json.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.pipeline import Part, extract_netlist  # noqa: E402
from circuit2dsn.review_state import Session  # noqa: E402
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "circuit2dsn" / "review_ui" / "index.html"
REVIEWS = ROOT / "outputs" / "review"
UPLOADS = ROOT / "inputs" / "uploads"
ALLOWED = (".pdf", ".png", ".jpg", ".jpeg")
MAX_UPLOAD = 100 * 1024 * 1024


class App:
    """The open session plus everything needed to start new ones."""

    def __init__(self, args, session: Session | None):
        self.args = args
        self.session = session
        self.lock = threading.Lock()
        self.classes = Taxonomy.load().classes
        self._detector = self._text = self._orienter = None

    # ---------------------------------------------------------- models (lazy)
    def vlm_backend(self):
        """The local vision model that double-checks the detector (loaded once). None if it cannot start."""
        if getattr(self, "_vlm", None) is None:
            from circuit2dsn import vlm_check

            try:
                self._vlm = (vlm_check.OllamaBackend(self.args.vlm_model or vlm_check.OLLAMA_MODEL)
                             if self.args.vlm == "ollama" else vlm_check.HFBackend(self.args.vlm_model or vlm_check.HF_MODEL))
            except Exception as e:                       # no GPU / Ollama not running: the main pipeline still works
                print(f"vision model not available ({e}); continuing without it")
                self._vlm = False
        return self._vlm or None

    def detectors(self):
        from circuit2dsn.detect import Detector
        from circuit2dsn.orientation import Orienter

        if not self.args.weights.exists():
            raise ValueError(f"No component model at {self.args.weights}. Download best.pt from Colab first.")
        if self._detector is None:
            self._detector = Detector(self.args.weights, conf=self.args.conf)
            if self.args.text_weights.exists():
                self._text = Detector(self.args.text_weights, conf=0.25)
            self._orienter = Orienter.available()
        return self._detector, self._text, self._orienter

    # ---------------------------------------------------------- sessions
    def sessions(self) -> list[dict]:
        out = []
        for f in sorted(REVIEWS.glob("*/session.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                s = Session(f.parent)
                c = s.counts()
                out.append({"folder": str(f.parent), "name": f.parent.name, "source": s.s.get("source", ""),
                            "created": s.s.get("created", ""), "phase": s.s.get("phase"), "counts": c,
                            "exported": bool(s.s.get("exports")),
                            "open": self.session is not None and self.session.folder == f.parent})
            except Exception:  # a broken session folder should not hide the others
                continue
        return out

    def open(self, folder: str) -> None:
        path = Path(folder).resolve()
        if path.parent != REVIEWS.resolve() or not (path / "session.json").exists():
            raise ValueError("not a review folder")
        self.session = Session(path)

    def process(self, file: Path, page_no: int, name: str, mode: str = "review") -> Session:
        """Detect parts (+ text) on one page, trace wires, start a review."""
        from circuit2dsn.detect import load_pages
        from circuit2dsn.textstage import ocr_inside_parts, ocr_page, read_texts

        det, tdet, orienter = self.detectors()
        pages = load_pages(file, self.args.dpi)
        if not 1 <= page_no <= len(pages):
            raise ValueError(f"page {page_no} out of range (1-{len(pages)})")
        page = pages[page_no - 1]
        parts = [Part(x.id, x.cls, (x.x1, x.y1, x.x2, x.y2), x.conf) for x in det.detect(page)]
        texts = []
        if tdet is not None:
            found = tdet.detect(page)
            texts = read_texts(np.asarray(page), [(x.x1, x.y1, x.x2, x.y2) for x in found],
                               [x.cls for x in found], role_confs=[x.conf for x in found])
        else:
            texts = ocr_page(np.asarray(page))      # no trained text model: RapidOCR finds the text
            texts += ocr_inside_parts(np.asarray(page), parts, texts)
        netlist = extract_netlist(np.asarray(page), parts, texts, orienter=orienter)
        from circuit2dsn import vlm_check

        vf = vlm_check.find_vlm_file(file, page_no, [ROOT / "inputs" / "vlm", ROOT / "outputs" / "company"])
        be = None
        if vf is None and self.args.vlm != "none":
            be = self.vlm_backend()
        from circuit2dsn.taxonomy import Taxonomy

        # second opinion: vision model (saved Colab result, or live) + chip-number OCR, combined
        vlm_check.second_opinion(page, netlist, be, vf, list(Taxonomy.load().classes), self.args.vlm_max)
        folder = REVIEWS / f"{name}_p{page_no}"
        if folder.exists():
            folder = REVIEWS / f"{name}_p{page_no}_{time.strftime('%H%M%S')}"
        label = f"{file.name} · page {page_no} of {len(pages)}"
        session = Session.create(folder, page, netlist, label, mode)
        if mode == "auto":
            session.apply({"type": "export", "reviewed": False})
            # anything under 70% sure goes to a person first; the export stays a draft until then
            session.s["phase"] = "verify" if session.verify_items() else "results"
            session.save()
        return session

    def page_count(self, file: Path) -> int:
        if file.suffix.lower() != ".pdf":
            return 1
        import pymupdf

        with pymupdf.open(file) as doc:
            return doc.page_count

    def state(self) -> dict:
        info = {"classes": self.classes, "has_model": self.args.weights.exists(),
                "has_text_model": self.args.text_weights.exists()}
        if self.session is None:
            return {"no_session": True, **info}
        st = self.session.public()
        st.update(info)
        st["folder"] = str(self.session.folder)
        return st


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            elif path == "/page.png":
                if app.session is None:
                    self._send(404, b"no session", "text/plain")
                else:
                    self._send(200, (app.session.folder / "page.png").read_bytes(), "image/png")
            elif path == "/api/state":
                with app.lock:
                    self._json(app.state())
            elif path.startswith("/download/"):
                # files of the open session's latest export only (no paths from the URL)
                name = unquote(path[len("/download/"):])
                sess = app.session
                files = (sess.s["exports"][-1].get("files") or {}).values() if sess and sess.s.get("exports") else []
                if name not in files:
                    self._send(404, b"not found", "text/plain")
                    return
                data = (sess.folder / "export" / name).read_bytes()
                ext = name.rsplit(".", 1)[-1].lower()
                ctype = {"json": "application/json", "csv": "text/csv", "png": "image/png", "log": "text/plain",
                         "zip": "application/zip", "pdf": "application/pdf"}.get(ext, "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                disp = "inline" if ext in ("png", "pdf") else "attachment"     # preview shows in the page
                self.send_header("Content-Disposition", f'{disp}; filename="{name}"')
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/parts":
                # search the installed OrCAD libraries: ?q=lm358&pins=8
                from circuit2dsn import orcad_parts

                q = parse_qs(urlparse(self.path).query)
                pins = q.get("pins", [""])[0]
                self._json({"available": orcad_parts.available(), "total": len(orcad_parts.index()),
                            "parts": orcad_parts.search(q.get("q", [""])[0], 60, int(pins) if pins.isdigit() else None)})
            elif path == "/api/sessions":
                with app.lock:
                    self._json({"sessions": app.sessions()})
            else:
                self._send(404, b"not found", "text/plain")

        def _fail(self, e: Exception):
            if not isinstance(e, ValueError):
                traceback.print_exc()
            msg = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            with app.lock:
                self._json({"error": msg, "state": app.state()})

        def do_POST(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            length = int(self.headers.get("Content-Length", 0))
            try:
                if url.path == "/api/action":
                    action = json.loads(self.rfile.read(length) or b"{}")
                    with app.lock:
                        if app.session is None:
                            raise ValueError("no review open")
                        result = app.session.apply(action)
                        self._json({"state": app.state(), "result": result})
                elif url.path == "/api/open":
                    body = json.loads(self.rfile.read(length) or b"{}")
                    with app.lock:
                        app.open(body["folder"])
                        self._json({"state": app.state()})
                elif url.path == "/api/upload":
                    # raw file body; ?name=<file name>&page=<n>
                    if length > MAX_UPLOAD:
                        raise ValueError("file too large (max 100 MB)")
                    raw_name = unquote(q.get("name", ["upload.png"])[0])
                    if not raw_name.lower().endswith(ALLOWED):
                        raise ValueError("upload a PDF, PNG or JPG")
                    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(raw_name).name)
                    UPLOADS.mkdir(parents=True, exist_ok=True)
                    file = UPLOADS / safe
                    file.write_bytes(self.rfile.read(length))
                    self._json({"file": safe, "pages": app.page_count(file)})
                elif url.path == "/api/process":
                    body = json.loads(self.rfile.read(length) or b"{}")
                    file = (UPLOADS / Path(body["file"]).name)
                    if not file.exists():
                        raise ValueError("uploaded file not found")
                    mode = body.get("mode", "review")
                    if mode not in ("review", "auto"):
                        raise ValueError("mode must be review or auto")
                    session = app.process(file, int(body.get("page", 1)), file.stem, mode)   # slow: outside the lock
                    with app.lock:
                        app.session = session
                        self._json({"state": app.state()})
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:  # keep the UI alive; show the problem
                self._fail(e)

    return Handler


def free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("no free port")


def start_session(args, app: App) -> Session | None:
    if args.resume:
        return Session(args.resume)
    if args.pcbnet:
        from scripts.extract_netlist import from_pcbnet
        from circuit2dsn.orientation import Orienter

        src, pages = from_pcbnet(args.pcbnet, args.gt_text)
        page, parts, texts = pages[0]
        folder = REVIEWS / f"pcbnet_{args.pcbnet}_p1"
        if (folder / "session.json").exists() and not args.new:
            return Session(folder)
        netlist = extract_netlist(np.asarray(page), parts, texts, orienter=Orienter.available())
        return Session.create(folder, page, netlist, f"{src} · page 1")
    if args.input:
        if not args.input.is_file():
            sys.exit(f"not found: {args.input}")
        print("Detecting parts and tracing wires...")
        return app.process(args.input, args.page, args.input.stem)
    return None                                   # dashboard: upload from the browser


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", type=Path, help="PDF / PNG / JPG")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--pcbnet", help="PCBnet design id for a demo (annotated parts, OCR text)")
    ap.add_argument("--gt-text", action="store_true", help="with --pcbnet: annotated text instead of OCR")
    ap.add_argument("--resume", type=Path, help="review folder to continue")
    ap.add_argument("--new", action="store_true", help="start over even if a review exists")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "detector" / "weights" / "best.pt")
    ap.add_argument("--text-weights", type=Path, default=ROOT / "runs" / "text" / "weights" / "best.pt")
    ap.add_argument("--vlm", choices=["none", "ollama", "hf"], default="none",
                    help="local vision model that double-checks every page after the detector (ollama: ollama pull qwen2.5vl:7b)")
    ap.add_argument("--vlm-model", help="override the model name")
    ap.add_argument("--vlm-max", type=int, default=120, help="most vision-model questions per page")
    ap.add_argument("--conf", type=float, default=0.25, help="keep detections down to this confidence (review catches the rest)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    app = App(args, None)
    app.session = start_session(args, app)
    port = free_port(args.port)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{port}/"
    print(f"Review dashboard at {url}\nPress Ctrl+C to stop; progress is saved.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
