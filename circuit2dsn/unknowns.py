"""Spots where the drawing has something the detector did not recognise.

A component the detector misses still leaves evidence in the wire tracing:
its wires stop at it, and its own strokes become short loose lines. So loose
wire ends that touch no detected part, pin or text are collected and grouped;
each group is reported as a box for the reviewer ("unidentified component
here?"), who can pick the right part (any OrCAD library part) or dismiss it.

suspicion: 3+ loose ends together 0.8, two 0.6, one long wire ending in
nothing 0.4. Lower means "check if you like", higher "most likely a part".
"""

from __future__ import annotations

import numpy as np


def find_unidentified(g, part_boxes: list, pins_xy: list, text_boxes: list, pin_groups: set | None = None,
                      max_spots: int = 60) -> list[dict]:
    spots = _loose_ends(g, part_boxes, pins_xy, text_boxes, pin_groups)
    spots += _odd_strokes(g, part_boxes, text_boxes, pin_groups, max(float(g.stroke), 1.0))
    spots = _merge(spots)
    spots.sort(key=lambda d: -d["suspicion"])
    for k, sp in enumerate(spots[:max_spots], 1):
        sp["id"] = f"U{k}"
    return spots[:max_spots]


def _loose_ends(g, part_boxes: list, pins_xy: list, text_boxes: list, pin_groups: set | None = None,
                      ) -> list[dict]:
    ends = np.asarray(g.endpoints, dtype=float)
    if not len(ends):
        return []
    s = max(float(g.stroke), 1.0)
    keep = np.ones(len(ends), bool)
    pad_part, pad_text = 3 * s + 4, 2 * s + 3
    for x1, y1, x2, y2 in part_boxes:
        keep &= ~((ends[:, 0] >= x1 - pad_part) & (ends[:, 0] <= x2 + pad_part) &
                  (ends[:, 1] >= y1 - pad_part) & (ends[:, 1] <= y2 + pad_part))
    for x1, y1, x2, y2 in text_boxes:
        keep &= ~((ends[:, 0] >= x1 - pad_text) & (ends[:, 0] <= x2 + pad_text) &
                  (ends[:, 1] >= y1 - pad_text) & (ends[:, 1] <= y2 + pad_text))
    if pins_xy:
        P = np.asarray(pins_xy, float)
        for i in np.nonzero(keep)[0]:
            if np.min(np.hypot(P[:, 0] - ends[i, 0], P[:, 1] - ends[i, 1])) < 4 * s + 4:
                keep[i] = False
    # next to a junction dot: the dot's blob, not a loose end
    for j in [j for j in getattr(g, "junctions", []) if j.kind == "dot"]:
        keep &= np.hypot(ends[:, 0] - j.x, ends[:, 1] - j.y) > 4 * s + 4
    ends = ends[keep]
    if not len(ends):
        return []
    # only ends of real wires (wire groups that reach a detected pin): a missed
    # part is wired into the circuit, unread letters / stray marks are not
    if pin_groups is not None:
        h0, w0 = g.seg_labels.shape
        real = []
        for x, y in ends:
            lab = g.seg_labels[min(int(y), h0 - 1), min(int(x), w0 - 1)]
            real.append(lab > 0 and int(g.group_of_seg[lab]) in pin_groups)
        ends = ends[np.array(real, bool)]
        if not len(ends):
            return []
    # wire length behind each end (tiny specks are noise, not wires)
    seg_len = np.bincount(g.seg_labels.ravel(), minlength=int(g.seg_labels.max()) + 1)
    h, w = g.seg_labels.shape

    def length_at(x, y):
        r = int(2 * s + 2)
        win = g.seg_labels[max(int(y) - r, 0):min(int(y) + r + 1, h), max(int(x) - r, 0):min(int(x) + r + 1, w)]
        lab = win[win > 0]
        return int(seg_len[np.bincount(lab).argmax()]) if len(lab) else 0

    lens = np.array([length_at(x, y) for x, y in ends])
    ok = lens >= 3 * s + 3
    ends, lens = ends[ok], lens[ok]
    # group loose ends lying close together
    R = max(20.0, 10 * s)
    parent = list(range(len(ends)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(ends)):
        d = np.hypot(ends[i + 1:, 0] - ends[i, 0], ends[i + 1:, 1] - ends[i, 1])
        for j in np.nonzero(d < R)[0]:
            parent[find(i + 1 + j)] = find(i)
    groups: dict = {}
    for i in range(len(ends)):
        groups.setdefault(find(i), []).append(i)
    spots = []
    for members in groups.values():
        pts = ends[members]
        n = len(members)
        if n == 1 and lens[members[0]] < 20 * s:
            continue                      # one short loose line: a stray mark, not worth asking
        pad = R / 2
        box = [float(pts[:, 0].min() - pad), float(pts[:, 1].min() - pad),
               float(pts[:, 0].max() + pad), float(pts[:, 1].max() + pad)]
        conf = 0.8 if n >= 3 else 0.6 if n == 2 else 0.4
        why = (f"{n} loose wire ends here: probably a component that was not recognised" if n >= 2
               else "a wire ends here with nothing attached")
        spots.append({"box": [round(v, 1) for v in box], "suspicion": conf, "why": why, "ends": n})
    return spots


def _odd_strokes(g, part_boxes, text_boxes, pin_groups, s) -> list[dict]:
    """Wire-connected strokes that are neither horizontal nor vertical
    (curves, diagonals) with no part or text over them: symbol outlines that
    joined the wires, so they leave no loose ends."""
    import cv2

    skel = (g.seg_labels > 0).astype(np.uint8)
    if not skel.any():
        return []
    L = int(3 * s + 4)
    axis = cv2.morphologyEx(skel, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)) |         cv2.morphologyEx(skel, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    axis = cv2.dilate(axis, np.ones((3, 3), np.uint8))
    odd = skel & (1 - axis)
    h, w = odd.shape
    for (x1, y1, x2, y2), pad in [(b, 3 * s + 4) for b in part_boxes] + [(b, 2 * s + 3) for b in text_boxes]:
        odd[max(int(y1 - pad), 0):min(int(y2 + pad) + 1, h), max(int(x1 - pad), 0):min(int(x2 + pad) + 1, w)] = 0
    for j in [j for j in getattr(g, "junctions", []) if j.kind == "dot"]:
        r = int(4 * s + 4)
        odd[max(int(j.y) - r, 0):int(j.y) + r + 1, max(int(j.x) - r, 0):int(j.x) + r + 1] = 0
    if pin_groups is not None:
        ys, xs = np.nonzero(odd)
        bad = ~np.isin(g.group_of_seg[g.seg_labels[ys, xs]], list(pin_groups))
        odd[ys[bad], xs[bad]] = 0
    R = int(max(6, 4 * s))
    blobs = cv2.dilate(odd, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blobs, 8)
    out = []
    for k in range(1, n):
        x, y, bw, bh, _ = stats[k]
        mass = int(odd[lab == k].sum())
        if mass < 4 * s + 6:
            continue
        out.append({"box": [float(x), float(y), float(x + bw), float(y + bh)], "suspicion": 0.7, "ends": 0,
                    "why": "curved or slanted lines on the wires here, but no part was recognised"})
    return out


def _merge(spots: list[dict]) -> list[dict]:
    """Overlapping spots become one (highest suspicion, union box)."""
    spots = sorted(spots, key=lambda d: -d["suspicion"])
    out: list[dict] = []
    for sp in spots:
        b = sp["box"]
        for o in out:
            ob = o["box"]
            if not (b[2] < ob[0] or ob[2] < b[0] or b[3] < ob[1] or ob[3] < b[1]):
                o["box"] = [min(b[0], ob[0]), min(b[1], ob[1]), max(b[2], ob[2]), max(b[3], ob[3])]
                o["ends"] = o.get("ends", 0) + sp.get("ends", 0)
                break
        else:
            out.append(dict(sp))
    return [{**o, "box": [round(v, 1) for v in o["box"]]} for o in out]
