"""Wire extraction: schematic image -> groups of electrically joined wire.

Classical line-art pipeline, no training needed for clean EDA schematics:

1. ink mask: pixels that differ from the page background colour
2. erase component bodies (otherwise a resistor's zig-zag shorts its pins)
3. skeletonise to 1 px lines; find junction clusters and segments between them
4. junction rules
     - junction dot (blob thicker than the line)  -> all arms connected
     - 4 arms forming two straight lines, no dot  -> crossing, NOT connected:
                                                     only opposite arms join
     - anything else (T, L, Y)                    -> all arms connected
5. union-find over segments -> wire groups

Every decision is recorded with a confidence so the review UI can ask about
the doubtful ones (e.g. a crossing whose dot is borderline).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

EIGHT = np.ones((3, 3), dtype=bool)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


@dataclass
class Junction:
    x: float
    y: float
    arms: list[int]            # segment labels
    kind: str                  # "dot", "crossing", "tee", "corner"
    conf: float                # confidence that `kind` is right


@dataclass
class WireGraph:
    seg_labels: np.ndarray      # int32 map, 0 = no segment
    group_of_seg: np.ndarray    # seg label -> group id
    junctions: list[Junction]
    stroke: float               # median half-width of lines in px
    skeleton: np.ndarray
    endpoints: np.ndarray       # (N, 2) x, y of skeleton end pixels
    ink: np.ndarray = field(repr=False, default=None)

    def group_at(self, x: float, y: float, radius: float) -> tuple[int | None, float]:
        """Group of the nearest wire pixel within `radius`, and its distance."""
        h, w = self.seg_labels.shape
        r = int(np.ceil(radius))
        x0, x1 = max(int(x) - r, 0), min(int(x) + r + 1, w)
        y0, y1 = max(int(y) - r, 0), min(int(y) + r + 1, h)
        win = self.seg_labels[y0:y1, x0:x1]
        ys, xs = np.nonzero(win)
        if len(xs) == 0:
            return None, float("inf")
        d = np.hypot(xs + x0 - x, ys + y0 - y)
        k = int(np.argmin(d))
        if d[k] > radius:
            return None, float("inf")
        return int(self.group_of_seg[win[ys[k], xs[k]]]), float(d[k])


def ink_mask(rgb: np.ndarray, threshold: int = 60) -> np.ndarray:
    """Pixels that differ from the dominant (background) colour."""
    small = rgb[::4, ::4].reshape(-1, 3) // 16
    packed = (small[:, 0].astype(np.int32) << 8) | (small[:, 1].astype(np.int32) << 4) | small[:, 2]
    mode = np.bincount(packed).argmax()
    bg = rgb[::4, ::4].reshape(-1, 3)[packed == mode].mean(0)
    diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(2)
    return diff > threshold


def erase_boxes(mask: np.ndarray, boxes, shrink: float = 0.0) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    for x1, y1, x2, y2 in boxes:
        a, b = int(np.ceil(x1 + shrink)), int(np.ceil(y1 + shrink))
        c, d = int(np.floor(x2 - shrink)), int(np.floor(y2 - shrink))
        if c >= a and d >= b:
            out[max(b, 0):min(d + 1, h), max(a, 0):min(c + 1, w)] = False
    return out


def erase_text(mask: np.ndarray, boxes, pad: float = 1.0, run_factor: float = 1.0, keep_dots: bool = True) -> np.ndarray:
    """Remove text glyphs but keep wire lines that run through the text box.

    Net labels sit on wire ends and pin numbers sit beside pin stubs; their
    glyphs touch the wire and each other, bridging separate nets. Inside each
    text box we drop everything except long horizontal/vertical runs (wires),
    which glyphs do not have. Run length is ~1.5x the typical text height.
    """
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return mask
    heights = np.array([min(b[2] - b[0], b[3] - b[1]) for b in boxes])
    # A glyph stroke is shorter than the text height; a wire through the box
    # (measured on the whole page, so it may continue outside) is longer.
    run = int(max(8, run_factor * np.median(heights)))
    m8 = mask.astype(np.uint8)
    lines = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)) | \
        cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((run, 1), np.uint8))
    # Junction dots often sit under a net label; keep them too.
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    stroke = float(np.median(dist[mask])) if mask.any() else 1.0
    core = (dist >= max(2.5, 1.42 * stroke + 1.5)).astype(np.uint8)
    rad = int(np.ceil(dist.max())) if core.any() else 0
    if rad:
        core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)))
    lines = ((lines | core) if keep_dots else lines) & m8
    out = mask.copy()
    h, w = mask.shape
    for x1, y1, x2, y2 in boxes:
        a, b = max(int(x1 - pad), 0), max(int(y1 - pad), 0)
        c, d = min(int(np.ceil(x2 + pad)) + 1, w), min(int(np.ceil(y2 + pad)) + 1, h)
        out[b:d, a:c] = lines[b:d, a:c].astype(bool)
    return out


def _neighbours(skel: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    return cv2.filter2D(skel.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT).astype(np.int32) * skel


def build_graph(mask: np.ndarray, dot_margin: float = 1.5, keep_ink: bool = False,
                gray: np.ndarray | None = None) -> WireGraph:
    """gray: the page in grey (rgb.min(2)). With it, the junction classifier
    (junction_model.py) decides whether a 4-way crossing has a dot: at low
    resolution a small dot is no thicker than two crossing lines, so the
    thickness rule alone misses about a third of them."""
    skel = skeletonize(mask)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    stroke = float(np.median(dist[skel])) if skel.any() else 1.0

    nb = _neighbours(skel)
    junction_px = skel & (nb >= 3)
    end_ys, end_xs = np.nonzero(skel & (nb == 1))

    # Merge junction pixels closer than ~a stroke width: an X drawn with thick
    # lines skeletonises into two nearby 3-way nodes, not one 4-way node.
    r = max(2, int(np.ceil(2 * stroke)) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    region = cv2.dilate(junction_px.astype(np.uint8), kernel).astype(bool) & skel
    region |= junction_px
    clusters, n_clusters = ndimage.label(region, structure=EIGHT)

    segs, n_segs = ndimage.label(skel & ~region, structure=EIGHT)
    uf = UnionFind(n_segs + 1)
    junctions: list[Junction] = []

    if n_clusters:
        # which segments touch which cluster
        grown = ndimage.grey_dilation(clusters, footprint=EIGHT)
        touch = (segs > 0) & (grown > 0)
        pairs = np.unique(np.stack([grown[touch], segs[touch]], 1), axis=0)
        arms_of: dict[int, list[int]] = {}
        for c, s in pairs:
            arms_of.setdefault(int(c), []).append(int(s))

        centres = ndimage.center_of_mass(region, clusters, range(1, n_clusters + 1))
        thick = ndimage.maximum(dist, clusters, range(1, n_clusters + 1))
        seg_pixels = ndimage.find_objects(segs)
        cluster_slices = ndimage.find_objects(clusters)
        lookup = segs.copy()   # segments + junction pixels, for point lookup
        reach = 4 * r + 6

        # learned dot decision for 4-way junctions (the only case where "no dot" cuts a connection)
        learned: dict[int, float] = {}
        if gray is not None:
            from . import junction_model

            four = [c for c, a in arms_of.items() if len(a) == 4]
            probs = junction_model.p_dot_many(gray, [(centres[c - 1][1], centres[c - 1][0], max(2.0, 2 * stroke))
                                                     for c in four]) if four else None
            if probs is not None:
                learned = dict(zip(four, probs))

        for c, arms in arms_of.items():
            cy, cx = centres[c - 1]
            # Direction each arm leaves the junction (mean offset of its nearby pixels).
            dirs = []
            for s in arms:
                sl = seg_pixels[s - 1]
                ys, xs = np.nonzero(segs[sl] == s)
                ys, xs = ys + sl[0].start, xs + sl[1].start
                near = np.hypot(xs - cx, ys - cy) < reach
                if near.any():
                    v = np.array([xs[near].mean() - cx, ys[near].mean() - cy])
                else:
                    v = np.array([xs.mean() - cx, ys.mean() - cy])
                n = np.linalg.norm(v)
                dirs.append(v / n if n > 0 else v)

            # Where two lines cross, the centre is ~1.42x as thick as a line
            # (diagonal to the corner); a junction dot is a disc well beyond that.
            dot_score = float(thick[c - 1]) - (1.42 * stroke + dot_margin)
            has_dot = dot_score >= 0
            if c in learned:
                # the classifier decides; a clearly fat blob (>= 1 px past the
                # crossing thickness) is a dot whatever the classifier says
                has_dot = learned[c] >= 0.5 or dot_score >= 1.0
                if has_dot and dot_score < 0:
                    dot_score = max(dot_score, 2 * learned[c] - 1)     # confidence from the classifier
            kind, conf = "tee", 0.9
            if len(arms) == 4 and not has_dot:
                # Four wires meeting without a dot is a crossing by convention.
                # The straightest pair is one wire; the other two arms are the
                # other wire, which may bend over a hop arc instead of running
                # straight through. Less sure when nothing is straight.
                (a, b), (c2, d), straight, other = _opposite_pairs(dirs)
                uf.union(arms[a], arms[b])
                uf.union(arms[c2], arms[d])
                conf = 0.55 + 0.15 * min(-dot_score, 1.5) + 0.15 * max(straight, 0) + 0.1 * max(other, 0)
                kind = "crossing" if other > 0.85 else "hop"
                junctions.append(Junction(cx, cy, arms, kind, round(min(0.99, conf), 3)))
                continue
            if has_dot:
                # dot_score = px of thickness beyond the crossing threshold;
                # clear dots are >= 1 px past it, borderline ones near 0.
                kind, conf = "dot", min(0.99, 0.75 + 0.2 * dot_score)
            elif len(arms) == 2:
                kind, conf = "corner", 0.95
            for s in arms[1:]:
                uf.union(arms[0], s)
            junctions.append(Junction(cx, cy, arms, kind, round(conf, 3)))
            # A pin often meets a wire right at a T / dot, so junction pixels
            # must resolve to the wire. (Crossings stay unlabelled: ambiguous.)
            sl = cluster_slices[c - 1]
            lookup[sl][clusters[sl] == c] = arms[0]
    else:
        lookup = segs

    group = np.array([uf.find(i) for i in range(n_segs + 1)], dtype=np.int32)
    return WireGraph(lookup, group, junctions, stroke, skel,
                     np.stack([end_xs, end_ys], 1) if len(end_xs) else np.zeros((0, 2), int),
                     mask if keep_ink else None)


def _opposite_pairs(dirs: list[np.ndarray]):
    """Pair 4 arm directions into two wires.

    Returns ((a, b), (c, d), straight, other): (a, b) is the most nearly
    opposite pair, `straight` its score and `other` the score of the
    remaining pair (1 = perfectly opposite, i.e. a straight line).
    """
    best = None
    for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
        for (p, q), (r, s) in (((a, b), (c, d)), ((c, d), (a, b))):
            s1, s2 = -float(dirs[p] @ dirs[q]), -float(dirs[r] @ dirs[s])
            if best is None or (s1, s2) > (best[2], best[3]):
                best = ((p, q), (r, s), s1, s2)
    return best
