"""Text roles: what a piece of schematic text means for the netlist.

The text detector predicts these as classes. PCBnet tags ~60% of its texts
(`details`); the rest get a role from rules using the component the text is
linked to and the text's shape.
"""

from __future__ import annotations

import re

ROLES = ["designator", "value", "net_label", "pin_number", "pin_name", "part_name", "other"]

KIND_TO_ROLE = {
    "designator": "designator",
    "value": "value",
    "netlabel": "net_label",
    "pinnumber": "pin_number",
    "pinname": "pin_name",
    "searchname": "part_name",
}

# Real reference-designator prefixes. A generic "letters+digits" pattern
# would also swallow IC pin names like PB9 / PC13 / N7.
DESIGNATOR = re.compile(r"(R|RN|RP|VR|C|L|FB|U|IC|Q|D|LED|ZD|J|P|CN|X|Y|K|SW|S|F|T|TP|H|M|BT|BZ|LS|MIC|ANT|E|JP)"
                        r"(\d+|\?)[A-Z]?(\.\d+)?")
POWER_TEXT = re.compile(r"[+-]\d+(\.\d+)?\s*V|\d+V\d+|V(CC|DD|SS|EE|IN|BAT|BUS)\w*|A?GND\w*", re.IGNORECASE)
VALUE = re.compile(r"[+-]?\d+([.,]\d+)?\s*([pnuμµmkKMG]|[kKMR]\d+)?\s*(F|H|Ω|R|V|A|Hz|MHz|kHz|W|%|ohm)?"
                   r"(\s*[/,]?\s*\d+([.,]\d+)?\s*(V|A|W|%))*", re.IGNORECASE)
PIN_NUMBER = re.compile(r"\d{1,3}|[A-Z]{1,2}\d{1,2}")
NAMED_SYMBOLS = ("power", "ground", "port")


def _looks_value(t: str) -> bool:
    t = t.strip()
    return bool(VALUE.fullmatch(t)) and any(c.isdigit() for c in t) and not t.isdigit()


def role_of(text, linked_cls: str | None, inside_linked_box: bool) -> str:
    """text: has .content and .kind (lowercased PCBnet `details`)."""
    if text.kind in KIND_TO_ROLE:
        return KIND_TO_ROLE[text.kind]
    t = text.content.strip()
    if linked_cls in NAMED_SYMBOLS:
        return "net_label"               # VCC / +3.3V / GND next to its symbol
    if linked_cls:
        if PIN_NUMBER.fullmatch(t) and t.isdigit():
            return "pin_number"
        if DESIGNATOR.fullmatch(t):
            return "designator"
        if _looks_value(t):
            return "value"
        if linked_cls == "ic_box":
            if PIN_NUMBER.fullmatch(t):
                return "pin_number"
            return "pin_name" if inside_linked_box else "part_name"
        return "part_name"
    # Unlinked: power-like text is a net name; clear designators; grid refs stay "other".
    if POWER_TEXT.fullmatch(t):
        return "net_label"
    if DESIGNATOR.fullmatch(t) and len(t) >= 2 and not re.fullmatch(r"[A-H]\d", t):
        return "designator"
    if _looks_value(t):
        return "value"
    return "other"


VALUE_TEXT = re.compile(r"[+-]?\d+([.,]\d+)?\s*([pnuμµmkKMG]|[kKMR]\d+)?\s*(F|H|Ω|R|V|A|Hz|W|%|ohm)?", re.IGNORECASE)
NET_NAME = re.compile(r"(IN|OUT|INPUT|OUTPUT|COMMON|COM|GND|AGND|DGND|VCC|VDD|VSS|VEE|VIN|VOUT|VBAT|VREF|"
                      r"[+-]?\d+(\.\d+)?V\d*|\d+V\d+|[A-Z][A-Z0-9_]{2,})", re.IGNORECASE)


def role_from_content(text: str) -> str:
    """Role from the words alone, for text found without the trained text
    detector (no layout cues). Designators and values are safe; net names only
    for clear names, since a wrong net_label merges nets."""
    t = text.strip()
    if not any(c.isalnum() for c in t):
        return "other"
    if DESIGNATOR.fullmatch(t):
        return "designator"
    if VALUE_TEXT.fullmatch(t) and any(c.isdigit() for c in t) and not t.isdigit():
        return "value"
    if t.isdigit() and len(t) <= 3:
        return "pin_number"
    if POWER_TEXT.fullmatch(t) or t.upper() in ("INPUT", "OUTPUT", "COMMON", "IN", "OUT", "COM"):
        return "net_label"
    return "other"
