"""EDIF -> OrCAD Capture .dsn using the OrCAD installation itself (no GUI).

OrCAD ships a TCL shell (tools/bin/tclsh.exe) and Capture's database library
(tools/bin/orDb_Dll_Tcl64.dll). Loaded in that shell, the library provides
Capture's batch commands:
- XMATIC_EDIF2CAP  Capture's own EDIF importer -> .dsn
- XMATIC_CAP2EDIF  Capture's own export of a .dsn, with the connectivity
                   Capture computed; used to verify every pin of every net
- XMATIC_DSN2PDF   Capture's own printout of the .dsn (preview / handout)
- XMATIC_DSN2XML   database dump (net count)

The .dsn format is Cadence's own binary database; only OrCAD's code writes it,
so this needs OrCAD installed on the same PC. Install location: env var
ORCAD_TOOLS (the "tools" folder), else common places are searched.

The import uses circuit2dsn/orcad_edi2cap.cfg: Capture's default settings
with the grid fixed to 100 mil (the default "calculated" grid rescales the
design) and generated net names hidden.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .edif_check import kid, kids, parse

CANDIDATES = [r"D:\tools", r"C:\Cadence\SPB_26.1\tools", r"C:\Cadence\SPB_25.1\tools", r"C:\Cadence\SPB_24.1\tools",
              r"C:\Cadence\SPB_23.1\tools", r"C:\Cadence\SPB_17.4\tools", r"C:\OrCAD\OrCAD_17.4\tools"]
CFG = Path(__file__).with_name("orcad_edi2cap.cfg")

SCRIPT = r"""
load [lindex $argv 0] DboTclWriteBasic
set edf [lindex $argv 1]
set dsn [lindex $argv 2]
set cfg [lindex $argv 3]
set xml [lindex $argv 4]
set back [lindex $argv 5]
if {[file exists $dsn]} { file delete -force $dsn }
set ok [XMATIC_EDIF2CAP $edf $dsn $cfg]
if {![file exists $dsn]} { puts "C2D_RESULT FAIL no dsn written"; exit 2 }
if {[catch {XMATIC_DSN2XML $dsn $xml} err]} { puts "C2D_XML_FAIL $err" }
if {[catch {XMATIC_CAP2EDIF $dsn $back ""} err]} { puts "C2D_BACK_FAIL $err" }
puts "C2D_RESULT OK"
exit 0
"""

PDF_SCRIPT = r"""
load [lindex $argv 0] DboTclWriteBasic
set cfg "COLORS::||JUNCTIONSIZE::Small||DRAWPINARROWS::0||EXCLUDEPROPERTIES::||CREATEPROPERTIESFILE::0||CREATENETANDBOOKMARK::0||ENABLEBIASVOLTAGE::0||ENABLEBIASPOWER::0||ENABLEBIASCURRENT::0||SHOWGRID::0||SHOWDRCMARKERS::0"
if {[catch {XMATIC_DSN2PDF [lindex $argv 1] [lindex $argv 2] [lindex $argv 3] 1 1 Light "" 0 "" $cfg} err]} { puts "C2D_PDF_FAIL $err"; exit 3 }
exit 0
"""


@dataclass
class DsnResult:
    ok: bool
    dsn: Path | None = None
    olb: Path | None = None
    errors: int = -1
    warnings: int = -1
    nets: int = -1                 # nets inside the .dsn (read back)
    multi_name_nets: list = field(default_factory=list)
    message: str = ""
    verified: bool | None = None   # Capture's own connectivity == the netlist (None: not checked)
    pins_checked: int = 0
    pins_mismatched: list = field(default_factory=list)
    capture_nets: list = field(default_factory=list)


def find_tools() -> Path | None:
    roots = [os.environ.get("ORCAD_TOOLS", "")] + CANDIDATES
    for r in roots:
        if r and (Path(r) / "bin" / "tclsh.exe").exists() and (Path(r) / "bin" / "orDb_Dll_Tcl64.dll").exists():
            return Path(r)
    return None


def available() -> bool:
    return find_tools() is not None


def _run(tools: Path, script: str, args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "c2d.tcl"
        sp.write_text(script, encoding="ascii")
        env = dict(os.environ)
        env["PATH"] = f"{tools / 'bin'};{tools / 'capture'};" + env.get("PATH", "")
        env.setdefault("CDS_INST_DIR", str(tools.parent))
        return subprocess.run([str(tools / "bin" / "tclsh.exe"), str(sp), (tools / "bin" / "orDb_Dll_Tcl64.dll").as_posix(),
                               *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env, cwd=str(cwd),
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def edif_to_dsn(edf: Path, dsn: Path, timeout: int = 300, expected: list[set] | None = None) -> DsnResult:
    """expected: nets as sets of (refdes, pin number) / ("PORT", name); when
    given, Capture's own connectivity of the new .dsn is compared with it."""
    tools = find_tools()
    if tools is None:
        return DsnResult(False, message="OrCAD not found (set ORCAD_TOOLS to the OrCAD 'tools' folder)")
    cfg = CFG if CFG.exists() else tools / "capture" / "EDI2CAP.CFG"
    edf, dsn = Path(edf).resolve(), Path(dsn).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "readback.xml"
        back = Path(tmp) / "capture_export.edf"
        try:
            proc = _run(tools, SCRIPT, [p.as_posix() for p in (edf, dsn, cfg, xml, back)], dsn.parent, timeout)
        except subprocess.TimeoutExpired:
            return DsnResult(False, message="OrCAD conversion timed out")
        log = (proc.stdout or "") + (proc.stderr or "")
        res = DsnResult(dsn.exists(), dsn if dsn.exists() else None)
        res.errors += log.count("Semantic error")      # EDIF problems the final count leaves out
        for line in log.splitlines():
            if "errors detected" in line and "warning" in line:
                try:
                    part = line.split(":")[-1].strip()
                    nums = [int(t) for t in part.replace(",", " ").split() if t.isdigit()]
                    # the parser and the writer each print a count: keep the worst
                    res.errors, res.warnings = max(res.errors, nums[0]), max(res.warnings, nums[1])
                except (ValueError, IndexError):
                    pass
        olb = list(dsn.parent.glob("*.OLB")) + list(dsn.parent.glob("*.olb"))
        res.olb = max(olb, key=lambda p: p.stat().st_mtime) if olb else None
        if xml.exists():
            root = ET.parse(xml).getroot()
            nets = list(root.iter("NetScalar"))
            res.nets = len(nets)
            for n in nets:
                names = {a.find("Defn").get("name") for a in n.iter("Alias")}
                if len(names) > 1:
                    res.multi_name_nets.append(sorted(names))
        if back.exists():
            import shutil
            shutil.copy(back, dsn.parent / (dsn.stem + "_capture_netlist.edf"))
            try:
                res.capture_nets = capture_nets(back.read_text(encoding="latin-1"))
            except (ValueError, IndexError, KeyError, TypeError) as e:
                log += f"\nC2D: could not read Capture's export: {e}"
            if expected is not None and res.capture_nets:
                res.pins_checked, res.pins_mismatched = compare_nets(expected, res.capture_nets)
                res.verified = not res.pins_mismatched
        for f in dsn.parent.glob("*.DSNlck"):
            try:
                f.unlink()
            except OSError:
                pass
        (dsn.parent / (dsn.stem + "_orcad_import.log")).write_text(log, encoding="utf-8", errors="replace")
        if not res.ok:
            res.message = "OrCAD did not write the .dsn; see the import log"
        return res


def dsn_to_pdf(dsn: Path, pdf: Path, timeout: int = 300) -> Path | None:
    """Capture's own print of the design (all pages) -> PDF."""
    tools = find_tools()
    if tools is None:
        return None
    dsn, pdf = Path(dsn).resolve(), Path(pdf).resolve()
    if pdf.exists():
        pdf.unlink()
    try:
        _run(tools, PDF_SCRIPT, [dsn.as_posix(), pdf.parent.as_posix(), pdf.name], pdf.parent, timeout)
    except subprocess.TimeoutExpired:
        return None
    for f in dsn.parent.glob("*.DSNlck"):
        try:
            f.unlink()
        except OSError:
            pass
    return pdf if pdf.exists() else None


# ---------------------------------------------------------------- verification

def _s(v):
    return v[1] if isinstance(v, tuple) else v


def _ident(node):
    v = node[1]
    while isinstance(v, list) and v and v[0] in ("rename", "name", "array"):
        v = v[1]
    return _s(v)


def _display_name(node):
    v = node[1]
    while isinstance(v, list) and v and v[0] == "name":
        v = v[1]
    if isinstance(v, list) and v and v[0] == "rename":
        s = v[2] if len(v) > 2 else v[1]
        if isinstance(s, list) and s and s[0] == "stringDisplay":
            s = s[1]
        return _s(s) if not isinstance(s, list) else _ident([None, v[1]])
    return _s(v)


def capture_nets(text: str) -> list[set]:
    """Nets of Capture's own EDIF export: sets of (reference, pin number) and
    ("PORT", name) for power symbols / ports. Same-named nets are one net."""
    tree = parse(text)
    pin_no: dict[str, dict] = {}          # cell -> port identifier -> pin number
    top_names: dict[str, str] = {}
    pages = []
    for lib in kids(tree, "library"):
        for cell in kids(lib, "cell"):
            cname = _ident(cell)
            for view in kids(cell, "view"):
                iface = kid(view, "interface") or []
                ports = {}
                for port in kids(iface, "port"):
                    d = kid(port, "designator")
                    ports[_ident(port)] = _s(d[1]) if d else _display_name(port)
                    top_names[_ident(port)] = _display_name(port)
                pin_no[cname] = ports
                contents = kid(view, "contents")
                if contents:
                    pages += kids(contents, "page") or [contents]
    groups: dict[str, set] = defaultdict(set)
    for page in pages:
        refdes, cell_of = {}, {}
        for inst in kids(page, "instance"):
            iname = _ident(inst)
            cref = kid(kid(inst, "viewRef") or [], "cellRef")
            cell_of[iname] = cref[1] if cref else None
            d = kid(inst, "designator")
            if d:
                v = d[1]
                refdes[iname] = _s(v[1]) if isinstance(v, list) and v[0] == "stringDisplay" else _s(v)
            else:
                refdes[iname] = iname
        for net in kids(page, "net"):
            name = str(_display_name(net)).upper()
            for j in kids(net, "joined"):
                for ref in kids(j, "portRef"):
                    iref = kid(ref, "instanceRef")
                    port = ref[1] if not isinstance(ref[1], list) else _ident(ref)
                    if iref:
                        inst = iref[1]
                        num = pin_no.get(cell_of.get(inst), {}).get(port, port)
                        groups[name].add((str(refdes.get(inst, inst)).upper(), str(num)))
                    else:
                        # Capture writes names that start with a digit as "&5V"
                        groups[name].add(("PORT", str(top_names.get(port, port)).upper().lstrip("&")))
    # power symbols / ports join every net that carries their name
    out: list[set] = []
    by_member: dict = {}
    for name, members in groups.items():
        merged = set(members)
        hit = {id(by_member[m]) for m in members if m in by_member}
        for g in [g for g in out if id(g) in hit]:
            merged |= g
            out.remove(g)
        out.append(merged)
        for m in merged:
            by_member[m] = merged
    return out


def expected_nets(netlist: dict, rep) -> list[set]:
    """Our netlist in the same terms, through the writer's report."""
    out = []
    for net in netlist["nets"]:
        members = set()
        for pid in net["pins"]:
            if pid not in rep.pin_map:
                continue
            ident, num = rep.pin_map[pid]
            if ident == "TOP":
                members.add(("PORT", str(num).upper()))
            else:
                members.add((str(rep.refdes.get(ident, ident)).upper(), str(num)))
        if members:
            out.append(members)
    return out


def compare_nets(expected: list[set], got: list[set]) -> tuple[int, list]:
    """(members checked, members whose net differs). Members of single-member
    nets only count when Capture joined them to something."""
    got_of = {}
    for g in got:
        for m in g:
            got_of[m] = frozenset(g)
    bad, checked = [], 0
    for e in expected:
        fe = frozenset(e)
        for m in e:
            checked += 1
            gm = got_of.get(m, frozenset([m]))
            if gm != fe:
                bad.append((m, sorted(fe - gm)[:4], sorted(gm - fe)[:4]))
    return checked, bad
