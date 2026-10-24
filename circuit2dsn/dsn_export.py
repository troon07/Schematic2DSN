"""Netlist -> verified OrCAD .dsn (+ PDF print and PNG preview), one call.

1. Drawn-schematic EDIF (schematic.py): real symbols, routed wires, power
   symbols, ports, footprints.
2. OrCAD's own importer makes the .dsn; OrCAD's own export of that .dsn is
   compared with the netlist, pin by pin.
3. If that check fails (or the writer fails), the compact label-style EDIF
   (every pin a short wire + net name, connectivity exact by construction)
   is used instead, so a wrong connection is never shipped silently.
4. OrCAD prints the .dsn to PDF; page 1 also becomes a PNG preview.
"""

from __future__ import annotations

import traceback
from pathlib import Path

from . import orcad


def make_dsn(netlist: dict, out: Path, stem: str, pdf: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    info: dict = {"available": orcad.available(), "style": None, "files": {}}
    edf = out / f"{stem}.edf"
    dsn = out / f"{stem}.dsn"
    attempts = []
    try:
        from .schematic import write_schematic

        text, rep = write_schematic(netlist, stem)
        attempts.append(("schematic", text, rep))
    except Exception as e:                               # noqa: BLE001 - fall back, but say why
        info["schematic_error"] = f"{type(e).__name__}: {e}"
        info["schematic_trace"] = traceback.format_exc()[-2000:]
    from .edif import write_edif

    text, rep = write_edif(netlist, stem, "labels")
    attempts.append(("labels", text, rep))

    for style, text, rep in attempts:
        edf.write_text(text, encoding="ascii")
        info.update(style=style, parts=rep.parts, nets=rep.nets, stubs=rep.stubs,
                    unrouted=getattr(rep, "unrouted", 0), pins_without_stub=len(rep.pins_without_stub))
        info["files"]["edif"] = edf.name
        if not info["available"]:
            break                                        # no OrCAD: the EDIF is the deliverable
        expected = orcad.expected_nets(netlist, rep) if style == "schematic" else None
        res = orcad.edif_to_dsn(edf, dsn, expected=expected)
        info.update(ok=res.ok, errors=res.errors, warnings=res.warnings, nets_in_dsn=res.nets,
                    multi_name_nets=res.multi_name_nets, message=res.message, verified=res.verified,
                    pins_checked=res.pins_checked, pins_mismatched=len(res.pins_mismatched),
                    mismatch_examples=[str(m) for m in res.pins_mismatched[:5]])
        if res.ok and (style == "labels" or res.verified):
            info["files"]["dsn"] = res.dsn.name
            if res.olb:
                info["files"]["olb"] = res.olb.name
            info["files"]["dsn_log"] = f"{dsn.stem}_orcad_import.log"
            break
        info.setdefault("fallback_reason", f"{style}: ok={res.ok} verified={res.verified} "
                                           f"mismatched={len(res.pins_mismatched)}")
    if pdf and info.get("files", {}).get("dsn"):
        p = orcad.dsn_to_pdf(dsn, out / f"{stem}_schematic.pdf")
        if p:
            info["files"]["pdf"] = p.name
            try:
                import pymupdf

                doc = pymupdf.open(p)
                pix = doc[0].get_pixmap(dpi=110)
                prev = out / f"{stem}_schematic.png"
                pix.save(prev)
                info["files"]["preview"] = prev.name
                info["pdf_pages"] = len(doc)
            except Exception:                            # noqa: BLE001 - preview is optional
                pass
    return info
