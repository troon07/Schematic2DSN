"""Role-aware cleanup of OCR readings.

The recogniser makes systematic mistakes on schematic text: Ω read as "n",
"Q" or "n2"; tiny digits read as Chinese characters; look-alike letters in
designator numbers; a wire end read as "[" or ".". Knowing the text's role
(designator, value, pin number, net label...) tells us what is legal.

`clean(text, role)` returns the corrected text. Values are put in one
canonical form ("0.1 μF", "100nF" -> "100nF", "10 kΩ" -> "10kΩ") so equal
values compare equal.
"""

from __future__ import annotations

import re

CJK = re.compile(r"[　-〿㐀-鿿＀-￯]")
EDGE_JUNK = "[]()|'\"`.,:;!_-~^"

# Look-alikes, used only where the role says a digit / letter must be.
TO_DIGIT = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1", "i": "1",
                          "T": "7", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "G": "6", "g": "9"})

# Common net / pin names: a reading 1 edit away snaps to them.
VOCAB = ["GND", "AGND", "DGND", "PGND", "VCC", "VDD", "VSS", "VEE", "VIN", "VOUT", "VBAT", "VBUS",
         "3V3", "5V", "12V", "EN", "RST", "RESET", "NRST", "SDA", "SCL", "TX", "RX", "TXD", "RXD",
         "MOSI", "MISO", "SCK", "SCLK", "CS", "NC", "BOOT0", "BOOT1", "SWDIO", "SWCLK", "OUT", "IN",
         "FB", "SW", "BST", "PG", "ADJ", "VREF", "AVDD", "DVDD", "IOVDD", "XTAL1", "XTAL2", "OSC_IN",
         "OSC_OUT", "USB_DP", "USB_DM", "D+", "D-", "CC1", "CC2", "SHIELD", "SHELL", "EP", "PAD"]

SNAP_BELOW = 0.8   # confident readings like "CLK" must not snap to "SCLK"

UNIT_PREFIX = {"p": "p", "n": "n", "u": "u", "μ": "u", "µ": "u", "m": "m", "k": "k", "K": "k", "M": "M", "G": "G"}


def _edit1(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def strip_edges(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()

    def unmatched(ch: str) -> bool:
        pair = {"(": ")", ")": "(", "[": "]", "]": "["}.get(ch)
        return pair is None or t.count(ch) > t.count(pair)

    while len(t) > 1 and t[0] in EDGE_JUNK and t[0] not in "-_" and t[1].isalnum() and unmatched(t[0]):
        t = t[1:]
    while len(t) > 1 and t[-1] in EDGE_JUNK and t[-1] not in "-_" and unmatched(t[-1]):
        t = t[:-1]
    return t


def canonical_value(t: str) -> str:
    """'0.1 μF' -> '0.1uF', '10 kΩ' / '10kn' / '10kQ' -> '10kΩ', '4K7' -> '4.7kΩ'."""
    s = strip_edges(t).replace(" ", "")
    s = s.replace("µ", "u").replace("μ", "u").replace("Ohm", "Ω").replace("ohm", "Ω").replace("R", "Ω") \
        if re.fullmatch(r"[\d.,]+\s*[kKMm]?\s*(R|Ohm|ohm)?", s) else s.replace("µ", "u").replace("μ", "u")
    # Ω misread after a multiplier or digit: 10kn, 10kQ, 5.1kn2, 1MO
    s = re.sub(r"^([\d.]+)([kKM])(n2|n|Q|O)$", lambda m: f"{m.group(1)}{m.group(2).replace('K', 'k')}Ω", s)
    s = re.sub(r"^([\d.]+)(n2|Q)$", r"\1Ω", s)
    # 4K7 / 4k7 / 2R2 notation
    m = re.fullmatch(r"(\d+)([kKMRΩ])(\d+)", s)
    if m:
        mult = {"k": "k", "K": "k", "M": "M", "R": "", "Ω": ""}[m.group(2)]
        s = f"{m.group(1)}.{m.group(3)}{mult}Ω"
    s = s.replace(",", ".")
    m = re.fullmatch(r"([\d.]+)([pnumkKMG]?)([FHΩ]?)(.*)", s)
    if m and m.group(1).replace(".", "").isdigit():
        num, pre, unit, rest = m.groups()
        pre = UNIT_PREFIX.get(pre, pre)
        return f"{num}{pre}{unit}{rest}"
    return s


def clean(text: str, role: str = "", conf: float = 0.0) -> str:
    """conf: OCR confidence; known-name snapping only applies below SNAP_BELOW."""
    t = strip_edges(text)
    role = (role or "").lower()
    if role in ("designator", "value", "pinnumber", "pinname"):
        t = CJK.sub("", t).strip()
    if not t:
        return ""
    if role == "value":
        return canonical_value(t)
    if role == "designator":
        t = t.replace(" ", "")
        if re.fullmatch(r"[A-Z]{1,3}\d+[A-Z]?(\.\d+)?", t):    # U4.1 = unit 1 of U4
            return t
        t = t.replace(".", "")
        m = re.fullmatch(r"([A-Za-z]{1,3})(.+?)([A-Za-z]?)", t)
        if m and len(m.group(2)) <= 4:
            num = m.group(2).translate(TO_DIGIT)
            if num.isdigit():
                return m.group(1).upper() + num + m.group(3)
        return t
    if role == "pinnumber":
        t = t.replace(" ", "")
        if re.fullmatch(r"[A-Z]{1,2}\d{1,2}|\d+|EP|PAD", t):    # 7, B12 (BGA), EP
            return t
        if len(t) <= 3 and any(c.isdigit() for c in t):
            d = t.translate(TO_DIGIT)
            if d.isdigit():
                return d
        return t
    if role in ("pinname", "netlabel", "power"):
        up = t.upper()
        if up not in VOCAB and len(t) >= 3 and conf < SNAP_BELOW and t[0] not in "+-":
            near = [v for v in VOCAB if _edit1(up, v)]
            if len(near) == 1:
                return near[0]
    return t


def same(a: str, b: str, role: str = "") -> bool:
    """Compare two texts the way the netlist cares about."""
    if (role or "").lower() == "value":
        return canonical_value(a).lower() == canonical_value(b).lower()
    return "".join(a.split()) == "".join(b.split())
