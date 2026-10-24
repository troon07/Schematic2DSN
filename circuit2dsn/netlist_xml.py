"""Export a circuit2dsn netlist to the industry 'TD9 TestSequencer' NetList.xml format.

The sir's reference file (NetList.xml) is a per-pin netlist: one <Node> per
device pin, carrying the net it belongs to, the pin's function label, its
electrical type and power level, plus two sections listing the power nets and
the user (signal) nets.

Mapping from our netlist JSON:
  - DeviceLocation  <- part designator (falls back to the part id)
  - PinNumber       <- the pin's number (OCR'd or assigned by the EDIF writer's
                       numbering); falls back to the trailing digits of the pin id
  - NodeName        <- "<Device>_<PinNumber>"
  - NodeDisName     <- "<Device>_<pin label>" for connector-style pins, else NodeName
  - NodeNetName    <- the net's name
  - LabelName      <- the pin's name text (the function printed on the symbol)
  - PowerLevel     <- "0" for GND, the voltage for a supply net ("5" for P5V), else ""
  - NodeType       <- "Power" for power pins, otherwise "Unknown" (we do not
                       infer I/O direction from the drawing)
  - IsPowerPin     <- TRUE when the pin's net is a power net
  - IsPinUsed      <- TRUE when the pin is connected to a named net
"""

from __future__ import annotations

import re
from xml.etree import ElementTree as ET

POWER_NAMES = {"GND", "AGND", "DGND", "PGND", "VSS", "VEE"}
POWER_LEVEL = {"GND": "0", "AGND": "0", "DGND": "0", "PGND": "0", "VSS": "0"}


def _designator(part: dict) -> str:
    d = part.get("designator")
    if isinstance(d, (list, tuple)):
        d = d[0] if d else None
    text = str(d or "").strip()
    if text and text.upper() not in ("NONE",):
        return text
    return str(part.get("id", "")).split(".")[0]


def _pin_number(pid: str, pin: dict) -> str:
    num = pin.get("number")
    if isinstance(num, (list, tuple)):
        num = num[0] if num else None
    text = str(num or "").strip()
    if text:
        text = re.sub(r"\s+", "", text)
        if text:
            return text
    m = re.search(r"\.(\d+)$", pid)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)$", pid)
    return m.group(1) if m else "1"


def _pin_name(pin: dict) -> str:
    nm = pin.get("name")
    if isinstance(nm, (list, tuple)):
        nm = nm[0] if nm else None
    return str(nm or "").strip()


def _net_name(netlist: dict, net_id: str) -> str:
    for n in netlist.get("nets", []):
        if n.get("id") == net_id:
            return str(n.get("name") or net_id)
    return net_id


def _is_power_net(name: str) -> bool:
    u = name.upper().replace(" ", "")
    if u in POWER_NAMES or u in POWER_LEVEL:
        return True
    if u.startswith("V") and any(ch.isdigit() for ch in u):
        return True
    if re.fullmatch(r"P\d+V", u):
        return True
    if "VCC" in u or "VDD" in u or "VSS" in u or "VBAT" in u or "VBUS" in u:
        return True
    return False


def _power_level(name: str) -> str:
    u = name.upper().replace(" ", "")
    if u in POWER_LEVEL:
        return POWER_LEVEL[u]
    if u == "GND" or u.endswith("GND"):
        return "0"
    m = re.search(r"(\d+)", u)
    if m and ("V" in u):
        return m.group(1)
    return ""


def _node_type(net_name: str, is_power: bool) -> str:
    if is_power:
        return "Power"
    return "Unknown"


def build_xml(netlist: dict, application: str = "TD9 TestSequencer",
              board: str = "SDDEMO", version: str = "FVER00001",
              modified: str = "") -> str:
    import datetime

    if not modified:
        modified = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")

    root = ET.Element("FileDescription")
    ET.SubElement(root, "Application", {"ApplicationName": application})
    ET.SubElement(root, "Board", {"BoardName": board})
    ET.SubElement(root, "Version", {"Version": version})
    ET.SubElement(root, "Date", {"FileModified": modified})

    netlist_el = ET.SubElement(root, "NetList")
    nodes_el = ET.SubElement(netlist_el, "Nodes")

    part_of = {p["id"]: p for p in netlist.get("parts", [])}
    net_name_of = {n["id"]: (n.get("name") or n["id"]) for n in netlist.get("nets", [])}

    power_net_names: set[str] = set()
    user_net_names: set[str] = set()

    for pid, pin in (netlist.get("pins") or {}).items():
        part = part_of.get(pin.get("part"), {})
        dev = _designator(part)
        pnum = _pin_number(pid, pin)
        node_name = f"{dev}_{pnum}"
        pname = _pin_name(pin)
        if pname and re.fullmatch(r"[A-Za-z]{1,3}\d{1,2}", pname):
            node_dis = f"{dev}_{pname}"
        else:
            node_dis = node_name

        net_id = pin.get("net")
        net_name = _net_name(netlist, net_id) if net_id else ""
        is_power = _is_power_net(net_name) if net_name else False
        if net_name:
            (power_net_names if is_power else user_net_names).add(net_name)

        is_used = "TRUE" if net_name else "FALSE"
        attrs = {
            "NodeName": node_name,
            "NodeDisName": node_dis,
            "PinNumber": pnum,
            "DeviceLocation": dev,
            "IsPinUsed": is_used,
            "Marking": "UnDefined",
            "NodeNetName": net_name,
            "LabelName": pname,
            "AliaseNetName": "",
            "IsReadOnly": "FALSE",
            "PowerLevel": _power_level(net_name) if is_power else "",
            "NodeType": _node_type(net_name, is_power),
            "IsPowerPin": "TRUE" if is_power else "FALSE",
        }
        ET.SubElement(nodes_el, "Node", attrs)

    # Power nets
    power_el = ET.SubElement(netlist_el, "PowerNets")
    for name in sorted(power_net_names):
        ET.SubElement(power_el, "Net", {
            "NetName": name,
            "NodeTypeLevel": "1",
            "PowerLevel": _power_level(name),
            "IsReadOnly": "FALSE",
            "IsPowerNet": "1",
        })

    # User nets
    user_el = ET.SubElement(netlist_el, "UserNets")
    for name in sorted(user_net_names):
        ET.SubElement(user_el, "Net", {
            "NetName": name,
            "NodeTypeLevel": "1",
            "PowerLevel": "",
            "IsReadOnly": "FALSE",
            "IsPowerNet": "0",
        })

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


def write_xml(netlist: dict, out_path: str, **kwargs) -> str:
    xml = build_xml(netlist, **kwargs)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return xml