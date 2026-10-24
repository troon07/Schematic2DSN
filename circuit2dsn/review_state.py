"""Human review session: the state behind the review UI, kept apart from the UI
so every action can be tested directly.

Steps
1. parts:       approve / fix (class, designator, value, name) / delete / add
2. connections: approve modules, then nets; split wrong pins out, merge nets
3. export:      corrected netlist JSON + EDIF + training data from corrections

Moving from parts to connections re-traces wires with the corrected parts.
Every change is saved to session.json right away; undo keeps 50 steps.
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .modules import find_modules
from .pipeline import Part, Text, extract_netlist
from .taxonomy import Taxonomy

Image.MAX_IMAGE_PIXELS = None
HIGH = 0.85
VERIFY_BELOW = 0.70     # anything less sure than this must be checked by a person, also in automatic mode


class Session:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.path = self.folder / "session.json"
        self.s = json.loads(self.path.read_text(encoding="utf-8"))
        self.history: list[dict] = []

    # ------------------------------------------------------------ create / save
    @classmethod
    def create(cls, folder: Path, page: Image.Image, netlist: dict, source: str, mode: str = "review") -> "Session":
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        page.convert("RGB").save(folder / "page.png")
        state = {
            "source": source, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "phase": "parts", "mode": mode,
            "netlist": netlist, "texts": netlist.get("texts", []),
            "part_state": {p["id"]: "pending" for p in netlist["parts"]},
            "part_edits": {}, "net_state": {}, "confirmed_pins": [], "module_state": {},
            "modules": [], "next_added": 1, "log": [], "exports": [],
            "unknowns": netlist.get("unidentified", []),
            "unknown_state": {u["id"]: "pending" for u in netlist.get("unidentified", [])},
            "needs_rewire": False,
        }
        (folder / "session.json").write_text(json.dumps(state), encoding="utf-8")
        return cls(folder)

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s), encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------ views
    def public(self) -> dict:
        """State for the UI (without the log)."""
        out = {k: v for k, v in self.s.items() if k != "log"}
        out["verify"] = self.verify_items()
        out["can_undo"] = bool(self.history)
        out["counts"] = self.counts()
        return out

    def review_nets(self) -> list[dict]:
        """Nets worth asking about: 2+ pins, or a real name."""
        out = []
        for n in self.s["netlist"]["nets"]:
            if len(n["pins"]) >= 2 or (n.get("name") and not n["name"].startswith("N")):
                out.append(n)
        return out

    def net_done(self, net: dict) -> bool:
        if self.s["net_state"].get(net["id"]) in ("approved", "edited"):
            return True
        conf = set(self.s["confirmed_pins"])
        return all(p in conf for p in net["pins"])

    def verify_items(self) -> list[dict]:
        """Everything under VERIFY_BELOW not yet checked by a person, least sure first:
        parts, connections (nets) and drawing the detector could not explain."""
        out = []
        ps = self.s["part_state"]
        for p in self.s["netlist"]["parts"]:
            if ps.get(p["id"]) == "pending" and p.get("conf", 1.0) < VERIFY_BELOW:
                out.append({"kind": "part", "id": p["id"], "conf": round(p.get("conf", 1.0), 3),
                            "why": p["vlm_flag"] if p.get("vlm_flag") else f"recognised as {p['cls']} with only {p.get('conf', 1.0):.0%} confidence"})
        for n in self.review_nets():
            if n.get("conf", 1.0) < VERIFY_BELOW and not self.net_done(n):
                why = (n.get("doubts") or [{}])[0].get("what", "uncertain connection")
                out.append({"kind": "net", "id": n["id"], "conf": round(n.get("conf", 1.0), 3),
                            "why": f"connection {n.get('name')}: {why}"})
        st = self.s.get("unknown_state", {})
        for u in self.s.get("unknowns", []):
            if st.get(u["id"]) == "pending":
                out.append({"kind": "unknown", "id": u["id"], "conf": round(1 - u["suspicion"], 3), "why": u["why"]})
        return sorted(out, key=lambda d: d["conf"])

    def counts(self) -> dict:
        ps = self.s["part_state"]
        nets = self.review_nets()
        return {
            "parts_total": sum(1 for v in ps.values() if v != "deleted"),
            "parts_done": sum(1 for v in ps.values() if v in ("approved", "edited", "added")),
            "parts_deleted": sum(1 for v in ps.values() if v == "deleted"),
            "nets_total": len(nets),
            "nets_done": sum(1 for n in nets if self.net_done(n)),
            "modules_total": len(self.s["modules"]),
            "modules_done": sum(1 for m in self.s["modules"] if self.s["module_state"].get(m["id"])),
            "verify_left": len(self.verify_items()),
            "unknowns_left": sum(1 for v in self.s.get("unknown_state", {}).values() if v == "pending"),
        }

    # ------------------------------------------------------------ actions
    def apply(self, action: dict) -> dict:
        kind = action.get("type")
        handler = getattr(self, f"_a_{kind}", None)
        if handler is None:
            raise ValueError(f"unknown action {kind!r}")
        if kind != "undo":
            self.history.append(copy.deepcopy(self.s))
            self.history = self.history[-50:]
        result = handler(action) or {}
        if kind != "undo":
            self.s["log"].append({"t": time.strftime("%H:%M:%S"), **{k: v for k, v in action.items()}})
        self.save()
        return result

    def _a_undo(self, a):
        if not self.history:
            raise ValueError("nothing to undo")
        log = self.s["log"]
        self.s = self.history.pop()
        self.s["log"] = log + [{"t": time.strftime("%H:%M:%S"), "type": "undo"}]

    def _part(self, pid: str) -> dict:
        for p in self.s["netlist"]["parts"]:
            if p["id"] == pid:
                return p
        raise ValueError(f"no part {pid}")

    def _a_part_ok(self, a):
        for pid in a.get("ids") or [a["id"]]:
            if self.s["part_state"].get(pid) == "pending":
                self.s["part_state"][pid] = "approved"

    def _a_part_edit(self, a):
        p = self._part(a["id"])
        edits = self.s["part_edits"].setdefault(a["id"], {})
        if a.get("cls") and a["cls"] != p["cls"]:
            Taxonomy.load().index(a["cls"])          # validates the class name
            edits["cls"] = {"from": p["cls"], "to": a["cls"]}
            p["cls"] = a["cls"]
            self.s["needs_rewire"] = True
        for field in ("designator", "value", "name"):
            if field in a:
                text = (a[field] or "").strip()
                old = p.get(field)
                old = (old[0] if isinstance(old, list) else old) or ""
                if text == old:
                    continue                      # untouched field: not a correction
                edits[field] = text
                p[field] = [text, 1.0] if field != "name" else (text or None)
        if self.s["part_state"].get(a["id"]) != "added":
            self.s["part_state"][a["id"]] = "edited"

    def _a_part_delete(self, a):
        self._part(a["id"])
        self.s["part_state"][a["id"]] = "deleted"
        self.s["part_edits"].setdefault(a["id"], {})["deleted"] = True
        self.s["needs_rewire"] = True

    def _a_part_add(self, a):
        Taxonomy.load().index(a["cls"])
        x1, y1, x2, y2 = (float(v) for v in a["box"])
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        pid = f"A{self.s['next_added']}"
        self.s["next_added"] += 1
        self.s["netlist"]["parts"].append({"id": pid, "cls": a["cls"], "conf": 1.0, "box": box, "name": None,
                                           "designator": None, "value": None, "pins": []})
        self.s["part_state"][pid] = "added"
        self.s["part_edits"][pid] = {"added": True}
        self.s["needs_rewire"] = True
        if a.get("orcad"):
            self._a_part_orcad({"id": pid, **a["orcad"]})
        return {"id": pid}

    def _a_part_orcad(self, a):
        """Use a part from the installed OrCAD libraries for this component
        (found with /api/parts). pin_map (optional): detected pin id -> OrCAD
        pin number, when the automatic pin match should be overridden."""
        from . import orcad_parts

        p = self._part(a["id"])
        row = next((r for r in orcad_parts.index() if r["file"] == a["file"] and r["name"] == a["name"]), None)
        if row is None:
            raise ValueError(f"OrCAD part {a.get('name')} is not in the catalogue")
        p["orcad"] = {"file": row["file"], "name": row["name"], "lib": row["lib"], "prefix": row["prefix"],
                      "fp": row["fp"], "pins": row["pins"], "pin_map": a.get("pin_map") or {}}
        self.s["part_edits"].setdefault(a["id"], {})["orcad"] = f"{row['lib']}:{row['name']}"
        if self.s["part_state"].get(a["id"]) != "added":
            self.s["part_state"][a["id"]] = "edited"

    def _a_part_orcad_clear(self, a):
        p = self._part(a["id"])
        p.pop("orcad", None)
        self.s["part_edits"].setdefault(a["id"], {}).pop("orcad", None)

    def _a_unknown(self, a):
        """An unexplained spot: 'dismiss' (not a component) or 'add' a part there
        (cls, optional orcad {file, name}, optional box tighter than the spot)."""
        spot = next((u for u in self.s.get("unknowns", []) if u["id"] == a["id"]), None)
        if spot is None:
            raise ValueError(f"no spot {a['id']}")
        if a["decision"] == "dismiss":
            self.s["unknown_state"][spot["id"]] = "dismissed"
            return {}
        if a["decision"] != "add":
            raise ValueError("decision must be dismiss or add")
        r = self._a_part_add({"cls": a.get("cls") or "other", "box": a.get("box") or spot["box"],
                              "orcad": a.get("orcad")})
        self.s["unknown_state"][spot["id"]] = "added"
        return r

    def _a_rewire(self, a):
        """Re-trace wires with the corrected parts; user edits survive."""
        rgb = np.asarray(Image.open(self.folder / "page.png").convert("RGB"))
        old_parts = [p for p in self.s["netlist"]["parts"] if self.s["part_state"].get(p["id"]) != "deleted"]
        parts = [Part(p["id"], p["cls"], tuple(p["box"]), p.get("conf", 1.0),
                      name=self.s["part_edits"].get(p["id"], {}).get("name") or None) for p in old_parts]
        texts = [Text(t["content"], tuple(t["box"]), t["role"], t["conf"]) for t in self.s["texts"]]
        from .orientation import Orienter

        nl = extract_netlist(rgb, parts, texts, orienter=Orienter.available())
        # user-typed designators / values beat the text matcher
        for p in nl["parts"]:
            ed = self.s["part_edits"].get(p["id"], {})
            for field in ("designator", "value"):
                if field in ed:
                    p[field] = [ed[field], 1.0]
        nl["texts"] = self.s["texts"]
        picked = {p["id"]: p["orcad"] for p in old_parts if p.get("orcad")}
        for p in nl["parts"]:
            if p["id"] in picked:
                p["orcad"] = picked[p["id"]]
        # keep a person's approvals for connections the re-trace did not change
        old_nets = {frozenset(n["pins"]): self.s["net_state"].get(n["id"]) for n in self.s["netlist"]["nets"]}
        old_conf = set(self.s["confirmed_pins"])
        self.s["netlist"] = nl
        self.s["needs_rewire"] = False
        net_state, confirmed = {}, []
        for n in nl["nets"]:
            st = old_nets.get(frozenset(n["pins"]))
            if st:
                net_state[n["id"]] = st
            if frozenset(n["pins"]) in old_nets and all(p in old_conf for p in n["pins"]):
                confirmed += n["pins"]
        self.s["net_state"], self.s["confirmed_pins"], self.s["module_state"] = net_state, sorted(confirmed), {}
        self.s["modules"] = find_modules(nl)
        if not a.get("keep_phase"):
            self.s["phase"] = "connections"

    def _net(self, nid: str) -> dict:
        for n in self.s["netlist"]["nets"]:
            if n["id"] == nid:
                return n
        raise ValueError(f"no net {nid}")

    def _a_net_ok(self, a):
        for nid in a.get("ids") or [a["id"]]:
            self._net(nid)
            self.s["net_state"].setdefault(nid, "approved")

    def _new_net_id(self) -> str:
        nums = [int(n["id"][3:]) for n in self.s["netlist"]["nets"] if n["id"][3:].isdigit()]
        return f"net{max(nums, default=0) + 1}"

    def _a_net_edit(self, a):
        nl = self.s["netlist"]
        net = self._net(a["id"])
        if a.get("merge_with"):
            other = self._net(a["merge_with"])
            if other is net:
                raise ValueError("cannot merge a net with itself")
            for pid in other["pins"]:
                nl["pins"][pid]["net"] = net["id"]
            net["pins"] = sorted(set(net["pins"]) | set(other["pins"]))
            if not net.get("name") or net["name"].startswith("N"):
                net["name"] = other.get("name") or net.get("name")
            nl["nets"].remove(other)
            self.s["net_state"].pop(other["id"], None)
        remove = [p for p in a.get("remove_pins") or [] if p in net["pins"]]
        if remove:
            if len(remove) == len(net["pins"]):
                raise ValueError("cannot remove every pin of a net")
            nid = self._new_net_id()
            for pid in remove:
                nl["pins"][pid]["net"] = nid
            net["pins"] = [p for p in net["pins"] if p not in remove]
            nl["nets"].append({"id": nid, "name": f"N{nid[3:]}", "pins": sorted(remove), "conf": 0.5,
                               "doubts": [{"what": f"split from {net['name']} by reviewer", "conf": 0.5}]})
            self.s["confirmed_pins"] = [p for p in self.s["confirmed_pins"] if p not in remove]
        if a.get("name"):
            net["name"] = a["name"].strip()
        self.s["net_state"][net["id"]] = "edited"
        self.s["confirmed_pins"] = sorted(set(self.s["confirmed_pins"]) | set(net["pins"]))

    def _a_module(self, a):
        """decision: 'ok' confirms the module's parts and their pins' connections."""
        mod = next((m for m in self.s["modules"] if m["id"] == a["id"]), None)
        if mod is None:
            raise ValueError(f"no module {a['id']}")
        self.s["module_state"][mod["id"]] = a["decision"]
        if a["decision"] == "ok":
            pins = [pid for pid, p in self.s["netlist"]["pins"].items() if p["part"] in mod["parts"]]
            self.s["confirmed_pins"] = sorted(set(self.s["confirmed_pins"]) | set(pins))

    def _a_phase(self, a):
        if a["phase"] not in ("parts", "connections", "export", "results", "verify"):
            raise ValueError("bad phase")
        self.s["phase"] = a["phase"]

    def _a_export(self, a):
        """Write every output file (+ one zip). Training data only after a
        human review: unchecked automatic results would teach the model its
        own mistakes."""
        import zipfile

        from scripts.export_edif import export
        from scripts.extract_netlist import draw_nets

        from .spice import to_bom, to_spice

        if self.s.get("needs_rewire"):
            self._a_rewire({"keep_phase": True})        # parts were added / removed / changed type
        out = self.folder / "export"
        out.mkdir(exist_ok=True)
        nl = self.s["netlist"]
        stem = self.folder.name
        files = {"netlist": f"{stem}_netlist.json", "edif": f"{stem}.edf", "spice": f"{stem}.cir",
                 "bom": f"{stem}_bom.csv", "image": f"{stem}_connections.png", "zip": f"{stem}_all.zip"}
        (out / files["netlist"]).write_text(json.dumps(nl, indent=1), encoding="utf-8")
        # industry NetList.xml format (TD9 TestSequencer / SDDEMO schema)
        try:
            from .netlist_xml import write_xml
            xml_name = f"{stem}_NetList.xml"
            write_xml(nl, str(out / xml_name))
            files["netlist_xml"] = xml_name
        except Exception:  # noqa: BLE001 - export must not fail because of the xml
            pass
        # connectivity self-check of the netlist itself (compact label style, scratch file)
        r = export(nl, out / f"{stem}_selfcheck.edf", stem, "labels")
        (out / f"{stem}_selfcheck.edf").unlink(missing_ok=True)
        (out / files["spice"]).write_text(to_spice(nl, stem), encoding="ascii", errors="replace")
        (out / files["bom"]).write_text(to_bom(nl), encoding="utf-8")
        draw_nets(Image.open(self.folder / "page.png"), nl).save(out / files["image"])
        # the real OrCAD design: drawn schematic (symbols, wires, power symbols,
        # ports, footprints), made by OrCAD's own importer and checked against
        # OrCAD's own netlist of it (needs OrCAD on this PC)
        from .dsn_export import make_dsn

        dsn_info = make_dsn(nl, out, stem)
        for key in ("edif", "dsn", "olb", "dsn_log", "pdf", "preview"):
            if key in dsn_info["files"]:
                files[key] = dsn_info["files"][key]
        with zipfile.ZipFile(out / files["zip"], "w", zipfile.ZIP_DEFLATED) as z:
            for key, fname in files.items():
                if key != "zip":
                    z.write(out / fname, fname)
        reviewed = a.get("reviewed", self.s.get("mode") != "auto")
        training = self._training_data() if reviewed else None
        info = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "edif": str(out / files["edif"]), "netlist": str(out / files["netlist"]),
                "files": files, "dsn": dsn_info, "errors": r["errors"][:10], "mismatch": r["connectivity_mismatch"],
                "shorts": r["shorted_pins"], "training": training, "counts": self.counts(), "reviewed": reviewed,
                "verify_left": len(self.verify_items()), "final": not self.verify_items()}
        self.s["exports"].append(info)
        return info

    def _a_mode(self, a):
        """Switch an automatic result into review (or back to the results page)."""
        if a["mode"] not in ("review", "auto"):
            raise ValueError("bad mode")
        self.s["mode"] = a["mode"]
        self.s["phase"] = "parts" if a["mode"] == "review" else "results"

    def _training_data(self) -> dict:
        """Corrected parts as YOLO labels + text fixes, for the next training round."""
        tax = Taxonomy.load()
        out = self.folder / "training"
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        name = self.folder.name
        shutil.copy(self.folder / "page.png", out / "images" / f"{name}.png")
        W, H = Image.open(self.folder / "page.png").size
        lines = []
        for p in self.s["netlist"]["parts"]:
            if self.s["part_state"].get(p["id"]) == "deleted" or p["cls"] not in tax.classes:
                continue
            x1, y1, x2, y2 = p["box"]
            lines.append(f"{tax.index(p['cls'])} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        fixes = [{"part": pid, **e} for pid, e in self.s["part_edits"].items()]
        (out / "corrections.json").write_text(json.dumps(fixes, indent=1), encoding="utf-8")
        return {"folder": str(out), "parts_labelled": len(lines), "corrections": len(fixes)}
