"""Synthetic training data for NEW symbol classes (ids 32 and up).

Draws standard ISA-5.1 / ISO 10628 symbols (valves, fittings, instruments) in the Digitize-PID style
and pastes them, with short pipes and tag text, into empty areas of your existing training tiles.
Each new tile keeps its original labels, so the model keeps learning the first 32 classes too.

    python src/synth_symbols.py --preview          # check the drawings first: docs/synth_preview.png
    python src/synth_symbols.py                    # build data/synth/ and data/synth.zip (~6000 + 1000 tiles)
    python src/synth_symbols.py --train 3000 --val 500   # smaller, faster run

Then train on Colab with both tiles.zip and synth.zip (see docs/train_new_symbols.md).
"""
import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
TILES = ROOT / "data" / "tiles"
OUT = ROOT / "data" / "synth"
NAMES = ROOT / "models" / "class_names.json"
SS = 4                                  # supersampling for smooth lines

# id, name, category, kind (valve = sits in a pipe, inst = instrument with signal lines)
NEW_CLASSES = [
    (32, "Safety relief valve", "valve"),
    (33, "Three-way valve", "valve"),
    (34, "Angle valve", "valve"),
    (35, "Solenoid valve", "valve"),
    (36, "Motor-operated valve", "valve"),
    (37, "Pressure regulator", "valve"),
    (38, "Y-strainer", "fitting"),
    (39, "Steam trap", "fitting"),
    (40, "Centrifugal pump", "fitting"),
    (41, "PLC function", "instrument"),
    (42, "Computer function", "instrument"),
    (43, "Interlock", "instrument"),
]

FONT_FILES = {
    False: ["DejaVuSansCondensed.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "/System/Library/Fonts/Supplemental/Arial Narrow.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arialn.ttf", "C:/Windows/Fonts/arial.ttf"],
    True: ["DejaVuSansCondensed-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
           "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/Library/Fonts/Arial Bold.ttf", "C:/Windows/Fonts/arialnb.ttf", "C:/Windows/Fonts/arialbd.ttf"],
}
_fonts = {}
BOLD = False                            # set per symbol, so text weight matches line weight


def font(size, bold=None):
    size, bold = max(6, int(size)), BOLD if bold is None else bold
    if (size, bold) not in _fonts:
        for f in FONT_FILES[bold] + FONT_FILES[False]:
            try:
                _fonts[size, bold] = ImageFont.truetype(f, size)
                break
            except OSError:
                continue
        else:
            _fonts[size, bold] = ImageFont.load_default(size)
    return _fonts[size, bold]


# ------------------------------------------------------------------ drawing in a local frame
class Sym:
    """Records shapes around (0, 0), then renders them rotated/mirrored (text stays upright)."""

    def __init__(self, lw):
        self.lw, self.ops, self.ports = lw, [], []

    def line(self, pts, dash=None, w=None):
        self.ops.append(("line", pts, dash, w))

    def poly(self, pts, fill=False):
        self.ops.append(("poly", pts, fill, None))

    def circle(self, cx, cy, r, fill=False, n=72):
        self.poly([(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)], fill)

    def arc(self, cx, cy, r, a0, a1, n=36):
        return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                 cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]

    def text(self, x, y, s, size):
        self.ops.append(("text", [(x, y)], s, size))

    def port(self, x, y, dx, dy, signal=False):
        self.ports.append((x, y, dx, dy, signal))

    def render(self, rot=0, mirror=False):
        """-> (L image of ink at native size, ports in image px, offset of (0,0) in the image)"""
        def T(p):
            x, y = p
            if mirror:
                x = -x
            for _ in range(rot % 4):
                x, y = -y, x
            return x, y

        pts_all = [T(p) for op in self.ops if op[0] != "text" for p in op[1]]
        for op in self.ops:
            if op[0] == "text":
                x, y = T(op[1][0])
                w = font(op[3]).getlength(op[2]) / 2 + 2
                pts_all += [(x - w, y - op[3] * 0.6), (x + w, y + op[3] * 0.6)]
        pad = self.lw * 2 + 2
        x0, y0 = min(p[0] for p in pts_all) - pad, min(p[1] for p in pts_all) - pad
        x1, y1 = max(p[0] for p in pts_all) + pad, max(p[1] for p in pts_all) + pad
        W, H = math.ceil(x1 - x0), math.ceil(y1 - y0)
        im = Image.new("L", (W * SS, H * SS), 0)
        d = ImageDraw.Draw(im)

        def S(p):
            x, y = T(p)
            return ((x - x0) * SS, (y - y0) * SS)

        for kind, pts, a, b in self.ops:
            if kind == "line":
                w = round((b or self.lw) * SS)
                if a:                                   # dashed
                    for seg in _dashes([S(p) for p in pts], a[0] * SS, a[1] * SS):
                        d.line(seg, fill=255, width=w)
                else:
                    d.line([S(p) for p in pts], fill=255, width=w, joint="curve")
            elif kind == "poly":
                q = [S(p) for p in pts]
                if a:
                    d.polygon(q, fill=255)
                else:
                    d.line(q + [q[0], q[1]], fill=255, width=round(self.lw * SS), joint="curve")
            elif kind == "text":
                d.text(S(pts[0]), a, fill=255, font=font(b * SS), anchor="mm")
        im = im.resize((W, H), Image.LANCZOS)
        ports = []
        for x, y, dx, dy, sig in self.ports:
            px, py = T((x, y))
            ddx, ddy = T((dx, dy))
            ports.append((px - x0, py - y0, round(ddx), round(ddy), sig))
        return im, ports


def _dashes(pts, on, off):
    out = []
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        L = math.hypot(bx - ax, by - ay)
        t = 0.0
        while t < L:
            t2 = min(L, t + on)
            out.append([(ax + (bx - ax) * t / L, ay + (by - ay) * t / L), (ax + (bx - ax) * t2 / L, ay + (by - ay) * t2 / L)])
            t = t2 + off
    return out


# ------------------------------------------------------------------ the symbols
def bowtie(s, L, H, fill=False):
    s.poly([(-L / 2, -H / 2), (-L / 2, H / 2), (0, 0)], fill)
    s.poly([(L / 2, -H / 2), (L / 2, H / 2), (0, 0)], fill)
    s.port(-L / 2, 0, -1, 0)
    s.port(L / 2, 0, 1, 0)


def tri_down(s, L, H, fill=False):           # third triangle, base at the bottom, tip at the centre
    s.poly([(-H / 2, L / 2), (H / 2, L / 2), (0, 0)], fill)
    s.port(0, L / 2, 0, 1)


def stem(s, top):
    s.line([(0, 0), (0, top)])


def safety_relief(s, rng, L):
    H = L * rng.uniform(0.55, 0.7)
    s.poly([(-H / 2, L / 2), (H / 2, L / 2), (0, 0)])             # inlet from below
    s.poly([(L / 2, -H / 2), (L / 2, H / 2), (0, 0)])             # outlet to the side
    s.port(0, L / 2, 0, 1)
    s.port(L / 2, 0, 1, 0)
    top, n, w = -L * rng.uniform(0.85, 1.05), rng.choice([3, 4, 5]), L * rng.uniform(0.14, 0.2)
    y0 = -L * 0.22
    s.line([(0, 0), (0, y0)])
    zz = [(0, y0)] + [((w if i % 2 else -w), y0 + (top - y0) * (i + 0.5) / n) for i in range(n)] + [(0, top)]
    s.line(zz)
    if rng.random() < 0.7:
        s.line([(-w * 1.3, top), (w * 1.3, top)])


def three_way(s, rng, L):
    H = L * rng.uniform(0.55, 0.7)
    bowtie(s, L, H)
    tri_down(s, L, H)
    if rng.random() < 0.4:                                         # optional handwheel
        stem(s, -L * 0.4)
        s.line([(-L * 0.18, -L * 0.4), (L * 0.18, -L * 0.4)])


def angle(s, rng, L):
    H = L * rng.uniform(0.55, 0.7)
    s.poly([(-L / 2, -H / 2), (-L / 2, H / 2), (0, 0)])
    s.poly([(-H / 2, L / 2), (H / 2, L / 2), (0, 0)])
    s.port(-L / 2, 0, -1, 0)
    s.port(0, L / 2, 0, 1)
    if rng.random() < 0.6:
        stem(s, -L * 0.42)
        s.line([(-L * 0.2, -L * 0.42), (L * 0.2, -L * 0.42)])


def actuated(s, rng, L, letter):
    H = L * rng.uniform(0.55, 0.7)
    bowtie(s, L, H)
    top = -L * rng.uniform(0.38, 0.5)
    stem(s, top)
    a = L * rng.uniform(0.38, 0.48)
    if letter == "S":
        s.poly([(-a / 2, top), (a / 2, top), (a / 2, top - a), (-a / 2, top - a)])
    else:
        s.circle(0, top - a / 2, a / 2)
    s.text(0, top - a / 2, letter, a * 0.62)


def solenoid(s, rng, L):
    actuated(s, rng, L, "S")


def motor_operated(s, rng, L):
    actuated(s, rng, L, "M")


def regulator(s, rng, L):
    H = L * rng.uniform(0.55, 0.7)
    bowtie(s, L, H)
    top = -L * rng.uniform(0.38, 0.48)
    stem(s, top)
    r = L * rng.uniform(0.3, 0.38)
    dome = s.arc(0, top, r, 180, 360)
    s.poly(dome)                                                   # diaphragm (flat side down)
    side = 1 if rng.random() < 0.5 else -1                        # sensing line to the downstream pipe
    xs, ys = side * r, top - r * 0.35
    xe = side * L * rng.uniform(0.85, 1.1)
    dash = (L * 0.08, L * 0.06) if rng.random() < 0.5 else None
    s.line([(xs, ys), (xe, ys), (xe, -s.lw)], dash=dash, w=max(1.5, s.lw * 0.7))
    s.ports = [p for p in s.ports if p[2] != side] + [(xe, 0, side, 0, False)]


def y_strainer(s, rng, L):
    if rng.random() < 0.5:
        return y_strainer_flanged(s, rng, L)
    s.line([(-L / 2, 0), (L / 2, 0)])
    s.port(-L / 2, 0, -1, 0)
    s.port(L / 2, 0, 1, 0)
    a = math.radians(rng.uniform(40, 50))
    ln, w = L * rng.uniform(0.55, 0.7), L * rng.uniform(0.24, 0.3)
    ux, uy = math.cos(a), math.sin(a)                             # leg goes down at ~45 degrees
    nx, ny = -uy, ux
    cx, cy = -L * 0.2 + ux * w * 0.5, uy * w * 0.5
    ends = []
    for k in (-0.5, 0.5):                                          # two long sides, starting on the pipe
        px, py = cx + nx * w * k, cy + ny * w * k
        t = -py / uy
        start = (px + ux * t, 0.0)
        end = (px + ux * ln, py + uy * ln)
        s.line([start, end])
        ends.append(end)
    s.line(ends)                                                   # cap
    if rng.random() < 0.3:
        e0, e1 = ends
        s.line([(e0[0] + ux * w * 0.25, e0[1] + uy * w * 0.25), (e1[0] + ux * w * 0.25, e1[1] + uy * w * 0.25)])
    m0 = (cx + ux * w * 0.3, cy + uy * w * 0.3)
    m1 = (cx + ux * (ln - w * 0.3), cy + uy * (ln - w * 0.3))
    s.line([m0, m1], dash=(L * 0.06, L * 0.05), w=max(1.5, s.lw * 0.6))


def y_strainer_flanged(s, rng, L):
    """Second common style: flanged body (end bars, like an H) and a solid diagonal leg with a cap bar."""
    half = L * rng.uniform(0.38, 0.48)
    fh = L * rng.uniform(0.28, 0.36)                               # flange bar height
    s.line([(-half, 0), (half, 0)], w=s.lw * rng.uniform(1.0, 1.6))
    for x in (-half, half):
        s.line([(x, -fh / 2), (x, fh / 2)], w=s.lw * rng.uniform(1.2, 1.8))
    s.port(-half, 0, -1, 0)
    s.port(half, 0, 1, 0)
    a = math.radians(rng.uniform(40, 50))
    ux, uy = math.cos(a), math.sin(a)
    nx, ny = -uy, ux
    ln, w = L * rng.uniform(0.5, 0.65), max(s.lw * 1.6, L * rng.uniform(0.08, 0.13))
    x0 = half * rng.uniform(0.2, 0.6)                              # leg leaves the body near one end
    start = (x0, 0.0)
    end = (x0 - ux * ln, uy * ln) if rng.random() < 0.5 else (x0 + ux * ln, uy * ln)
    dx, dy = (end[0] - start[0]) / ln, (end[1] - start[1]) / ln
    px, py = -dy, dx
    s.poly([(start[0] + px * w / 2, start[1] + py * w / 2), (end[0] + px * w / 2, end[1] + py * w / 2),
            (end[0] - px * w / 2, end[1] - py * w / 2), (start[0] - px * w / 2, start[1] - py * w / 2)], fill=True)
    c = w * rng.uniform(1.6, 2.2)                                  # cap bar across the end of the leg
    s.line([(end[0] + px * c, end[1] + py * c), (end[0] - px * c, end[1] - py * c)], w=s.lw * 1.4)


def steam_trap(s, rng, L):
    r = L * rng.uniform(0.28, 0.36)
    if rng.random() < 0.6:
        s.circle(0, 0, r)
    else:
        s.poly([(-r, -r), (r, -r), (r, r), (-r, r)])
    if rng.random() < 0.5:
        s.text(0, 0, "T", r * 1.25)
    else:
        s.line([(-r * 0.5, -r * 0.45), (r * 0.5, -r * 0.45)])
        s.line([(0, -r * 0.45), (0, r * 0.55)])
    s.port(-r, 0, -1, 0)
    s.port(r, 0, 1, 0)


def pump(s, rng, L):
    r = L * rng.uniform(0.42, 0.5)
    s.circle(0, 0, r)
    s.line([(0, -r), (r * rng.uniform(1.0, 1.3), -r)])            # tangential discharge
    if rng.random() < 0.8:                                         # base
        b = r * rng.uniform(1.2, 1.5)
        s.line([(-r * 0.6, r * 0.8), (-b * 0.75, r * 1.35), (b * 0.75, r * 1.35), (r * 0.6, r * 0.8)])
    if rng.random() < 0.3:
        s.poly([(-r * 0.35, -r * 0.4), (-r * 0.35, r * 0.4), (r * 0.45, 0)])
    s.port(-r, 0, -1, 0)
    s.port(r * 1.15, -r, 1, 0)


def _tag(rng):
    letters = rng.choice(["FY", "TY", "PY", "LY", "UY", "HS", "FIC", "TIC", "PLC", "XY", "KY", "LIC"])
    return letters, str(rng.randint(10, 9999)) if rng.random() < 0.8 else ""


def _inst_text(s, rng, size, center_line):
    letters, num = _tag(rng)
    fs = size * rng.uniform(0.2, 0.25)
    if num:
        s.text(0, -size * 0.14, letters, fs)
        s.text(0, size * 0.16, num, fs)
    else:
        s.text(0, 0, letters, fs)
    if center_line:
        s.line([(-size * 0.42, 0), (size * 0.42, 0)])


def _signal_ports(s, half):
    s.port(0, -half, 0, -1, True)
    s.port(0, half, 0, 1, True)
    s.port(-half, 0, -1, 0, True)
    s.port(half, 0, 1, 0, True)


def plc(s, rng, L):
    a = L * rng.uniform(1.15, 1.5) / 2
    s.poly([(-a, -a), (a, -a), (a, a), (-a, a)])
    s.poly([(0, -a), (a, 0), (0, a), (-a, 0)])
    _inst_text(s, rng, a * 1.6, rng.random() < 0.3)
    _signal_ports(s, a)


def computer(s, rng, L):
    a = L * rng.uniform(1.15, 1.5) / 2
    h = a * math.sqrt(3) / 2
    s.poly([(-a, 0), (-a / 2, -h), (a / 2, -h), (a, 0), (a / 2, h), (-a / 2, h)])
    _inst_text(s, rng, a * 1.9, rng.random() < 0.4)
    _signal_ports(s, a)


def interlock(s, rng, L):
    a = L * rng.uniform(0.6, 0.85)
    s.poly([(0, -a), (a, 0), (0, a), (-a, 0)])
    t = rng.choice(["I", "IL", "I", "I-" + str(rng.randint(1, 99))])
    s.text(0, 0, t, a * (0.75 if len(t) == 1 else 0.5))
    _signal_ports(s, a)


DRAW = {32: safety_relief, 33: three_way, 34: angle, 35: solenoid, 36: motor_operated, 37: regulator,
        38: y_strainer, 39: steam_trap, 40: pump, 41: plc, 42: computer, 43: interlock}
SIZE = {32: (55, 90), 33: (55, 90), 34: (55, 90), 35: (55, 90), 36: (55, 90), 37: (55, 90),
        38: (60, 100), 39: (60, 100), 40: (75, 130), 41: (75, 105), 42: (75, 105), 43: (55, 85)}
UPRIGHT = {41, 42, 43, 40}             # instruments and pumps are not rotated (only mirrored where it makes sense)


def make_symbol(cid, rng):
    global BOLD
    lw = rng.uniform(3.2, 6.0)                 # Digitize-PID lines are 4-6 px at tile scale
    BOLD = lw > 4.4
    L = rng.uniform(*SIZE[cid])
    s = Sym(lw)
    DRAW[cid](s, rng, L)
    if cid in UPRIGHT:
        rot = 0
    elif cid == 32:                    # relief valves: spring up, occasionally turned on its side
        rot = 0 if rng.random() < 0.85 else rng.choice([1, 3])
    else:
        rot = rng.choice([0, 1, 2, 3])
    mirror = cid in (32, 34, 37, 38, 40) and rng.random() < 0.5
    im, ports = s.render(rot, mirror)
    return im, ports, lw


# ------------------------------------------------------------------ pasting into tiles
def read_boxes(path, W, H):
    boxes = []
    if path.exists():
        for line in path.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:])
                boxes.append((c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return boxes


def overlaps(b, boxes, m=0):
    return any(b[0] < o[2] + m and b[2] > o[0] - m and b[1] < o[3] + m and b[3] > o[1] - m for o in boxes)


def draw_pipe(d, x, y, dx, dy, lw, length, busy, W, H, ink, signal):
    """Pipe from a port outwards; stops before other symbols or at the tile edge."""
    step, n = 4, 0
    ex, ey = x, y
    while n < length:
        nx, ny = ex + dx * step, ey + dy * step
        if not (0 <= nx < W and 0 <= ny < H) or overlaps((nx - 2, ny - 2, nx + 2, ny + 2), busy, 4):
            break
        ex, ey, n = nx, ny, n + step
    if n < 6:
        return None
    w = max(1, round(lw * (0.55 if signal else 1)))
    if signal and random.random() < 0.7:
        L = math.hypot(ex - x, ey - y)
        t = 0
        while t < L:
            t2 = min(L, t + 9)
            d.line([(x + dx * t, y + dy * t), (x + dx * t2, y + dy * t2)], fill=ink, width=w)
            t = t2 + 6
    else:
        d.line([(x, y), (ex, ey)], fill=ink, width=w)
    return (min(x, ex) - w, min(y, ey) - w, max(x, ex) + w, max(y, ey) + w)


def add_tag_text(d, rng, box, busy, W, H, ink):
    text = rng.choice([f"{rng.randint(10, 300)}{rng.choice(['BV', 'GV', 'SV', 'NRV', 'CV', 'PSV', 'ST'])}",
                       f"{rng.choice(['PSV', 'XV', 'MOV', 'PCV', 'SV', 'P', 'ST'])}-{rng.randint(100, 9999)}",
                       f"{rng.randint(1, 12)}\"", f"{rng.choice('ABCDEFGHKLMNPRSTUVWXYZ')}{rng.choice('ABCDEFGHKLMNPRSTUVWXYZ')}-{rng.randint(10000, 99999)}"])
    f = font(rng.uniform(20, 34), bold=rng.random() < 0.5)
    tw, th = f.getlength(text), f.size
    x0, y0, x1, y1 = box
    for _ in range(8):
        side = rng.choice(["above", "below", "right", "left"])
        tx = {"above": (x0 + x1 - tw) / 2, "below": (x0 + x1 - tw) / 2, "right": x1 + 6, "left": x0 - tw - 6}[side]
        ty = {"above": y0 - th - 6, "below": y1 + 4, "right": (y0 + y1 - th) / 2, "left": (y0 + y1 - th) / 2}[side]
        tb = (tx, ty, tx + tw, ty + th)
        if tb[0] >= 0 and tb[1] >= 0 and tb[2] < W and tb[3] < H and not overlaps(tb, busy, 3):
            d.text((tx, ty), text, fill=ink, font=f)
            busy.append(tb)
            return


def synth_tile(img, boxes, classes, rng, max_new=3):
    """Paste 1..max_new new symbols into empty areas of one tile. Returns the new labels (cls, x0, y0, x1, y1)."""
    W, H = img.size
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    dark = gray < gray.mean() - 60
    busy = [b[1:] for b in boxes]
    ink_level = rng.randint(0, 45)
    added = []
    d = ImageDraw.Draw(img)
    for cid in classes[: rng.randint(1, max_new)]:
        sym, ports, lw = make_symbol(cid, rng)
        sw, sh = sym.size
        if sw >= W - 20 or sh >= H - 20:
            continue
        for _ in range(60):
            x, y = rng.randint(8, W - sw - 8), rng.randint(8, H - sh - 8)
            b = (x, y, x + sw, y + sh)
            if overlaps(b, busy, 14) or dark[y:y + sh, x:x + sw].mean() > 0.01:
                continue
            a = np.asarray(sym, dtype=np.float32) / 255.0
            region = np.asarray(img.crop(b), dtype=np.float32)
            region = region * (1 - a[..., None]) + ink_level * a[..., None]
            img.paste(Image.fromarray(region.clip(0, 255).astype(np.uint8)), (x, y))
            ys, xs = np.nonzero(a > 0.25)
            tight = (x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1)
            busy.append(tight)
            ink = (ink_level,) * 3
            pipes = []
            for px, py, dx, dy, sig in ports:
                if rng.random() < (0.75 if sig else 0.97):
                    seg = draw_pipe(d, x + px, y + py, dx, dy, lw, rng.randint(25, 260),
                                    [o for o in busy if o is not tight] + pipes, W, H, ink, sig)
                    if seg:
                        pipes.append(seg)
            busy += pipes
            if rng.random() < 0.6:
                add_tag_text(d, rng, tight, busy, W, H, ink)
            added.append((cid, *tight))
            break
    return added


def to_yolo(c, x0, y0, x1, y1, W, H):
    return f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"


# ------------------------------------------------------------------ main
def class_list():
    rows = json.loads(NAMES.read_text(encoding="utf-8")) if NAMES.exists() else []
    old = {int(r["id"]): r["name"] for r in rows if int(r["id"]) < 32}
    if len(old) != 32:
        sys.exit(f"Expected classes 0..31 in {NAMES}")
    return [old[i] for i in range(32)] + [n for _, n, _ in NEW_CLASSES]


def update_class_names():
    """Add the new classes to models/class_names.json (existing entries and your edits are kept)."""
    rows = json.loads(NAMES.read_text(encoding="utf-8"))
    have = {int(r["id"]) for r in rows}
    added = [{"id": i, "name": n, "category": c} for i, n, c in NEW_CLASSES if i not in have]
    if added:
        NAMES.write_text(json.dumps(sorted(rows + added, key=lambda r: int(r["id"])), indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
        print(f"Added {len(added)} new classes to {NAMES.relative_to(ROOT)}")


def tile_list(split):
    imgs = sorted((TILES / "images" / split).glob("*.jpg"))
    if not imgs:
        sys.exit(f"No tiles in {TILES / 'images' / split}. Run src/tile_dataset.py first.")
    return imgs


def balanced_classes(counts, rng, k=3):
    ids = sorted(counts, key=lambda c: (counts[c], rng.random()))
    return ids[:k]


def build(split, n, rng, counts):
    out_i, out_l = OUT / "images" / split, OUT / "labels" / split
    out_i.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)
    tiles = tile_list(split)
    made = 0
    for i in range(n * 3):
        if made >= n:
            break
        src = rng.choice(tiles)
        img = Image.open(src).convert("RGB")
        W, H = img.size
        boxes = read_boxes(TILES / "labels" / split / (src.stem + ".txt"), W, H)
        new = synth_tile(img, boxes, balanced_classes(counts, rng), rng)
        if not new:
            continue
        for c, *_ in new:
            counts[c] += 1
        stem = f"{src.stem}_syn{made}"
        img.save(out_i / f"{stem}.jpg", quality=rng.randint(80, 95))
        lines = [to_yolo(c, *b, W, H) for c, *b in boxes] + [to_yolo(c, *b, W, H) for c, *b in new]
        (out_l / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        made += 1
        if made % 500 == 0:
            print(f"  {split}: {made}/{n}", flush=True)
    return made


def preview(rng):
    """Gallery of every new symbol + a few finished tiles with their boxes."""
    cell, cols = 150, 8
    names = {i: n for i, n, _ in NEW_CLASSES}
    rows = len(NEW_CLASSES)
    sheet = Image.new("RGB", (cell * cols + 230, cell * rows), "white")
    d = ImageDraw.Draw(sheet)
    for r, (cid, name, _) in enumerate(NEW_CLASSES):
        d.text((8, r * cell + cell // 2), f"{cid}  {name}", fill=(31, 78, 156), font=font(17, False), anchor="lm")
        for c in range(cols):
            im, _, _ = make_symbol(cid, rng)
            k = min(1.0, (cell - 16) / max(im.size))
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
            box = Image.new("RGB", (cell, cell), (228, 228, 228))
            ink = Image.new("RGB", im.size, (20, 20, 20))
            box.paste(ink, ((cell - im.width) // 2, (cell - im.height) // 2), im)
            sheet.paste(box, (230 + c * cell, r * cell))
    (ROOT / "docs").mkdir(exist_ok=True)
    sheet.save(ROOT / "docs" / "synth_preview.png")

    tiles = tile_list("train")
    counts = {i: 0 for i, _, _ in NEW_CLASSES}
    shots = []
    while len(shots) < 6:
        src = rng.choice(tiles)
        img = Image.open(src).convert("RGB")
        boxes = read_boxes(TILES / "labels" / "train" / (src.stem + ".txt"), *img.size)
        new = synth_tile(img, boxes, balanced_classes(counts, rng), rng)
        if not new:
            continue
        for c, *_ in new:
            counts[c] += 1
        dd = ImageDraw.Draw(img)
        for _, *b in boxes:
            dd.rectangle(b, outline=(150, 150, 150), width=2)
        for c, *b in new:
            dd.rectangle(b, outline=(220, 38, 38), width=3)
            dd.text((b[0], b[1] - 3), names[c], fill=(220, 38, 38), font=font(16, False), anchor="lb")
        shots.append(img)
    w, h = shots[0].size
    grid = Image.new("RGB", (w * 3, h * 2), "white")
    for i, im in enumerate(shots):
        grid.paste(im, ((i % 3) * w, (i // 3) * h))
    grid.save(ROOT / "docs" / "synth_preview_tiles.jpg", quality=88)
    print("Saved docs/synth_preview.png (every new symbol, 8 random variants)")
    print("Saved docs/synth_preview_tiles.jpg (red = new symbols, grey = existing labels)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="only draw preview pictures in docs/")
    ap.add_argument("--train", type=int, default=6000, help="synthetic training tiles (default 6000)")
    ap.add_argument("--val", type=int, default=1000, help="synthetic validation tiles (default 1000)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    random.seed(args.seed)

    if args.preview:
        preview(rng)
        return

    names = class_list()
    if OUT.exists():
        shutil.rmtree(OUT)
    counts = {i: 0 for i, _, _ in NEW_CLASSES}
    print(f"Building {args.train} train + {args.val} val synthetic tiles in {OUT.relative_to(ROOT)}/ ...")
    build("train", args.train, rng, counts)
    train_counts = dict(counts)
    build("val", args.val, rng, {i: 0 for i in counts})

    yaml = ["# Original tiles + synthetic tiles with 12 new symbol classes (written by src/synth_symbols.py)",
            "path: /content/datasets",
            "train: [tiles/images/train, synth/images/train]",
            "val: [tiles/images/val, synth/images/val]",
            f"nc: {len(names)}",
            "names:"] + [f"  {i}: {json.dumps(n)}" for i, n in enumerate(names)]
    (OUT / "data_v2.yaml").write_text("\n".join(yaml) + "\n")
    update_class_names()
    print("New symbols in training tiles: " + ", ".join(f"{n} {train_counts[i]}" for i, n, _ in NEW_CLASSES))
    print("Zipping ...", flush=True)
    shutil.make_archive(str(ROOT / "data" / "synth"), "zip", root_dir=ROOT / "data", base_dir="synth")
    print(f"Done: data/synth.zip ({(ROOT / 'data' / 'synth.zip').stat().st_size / 1e6:.0f} MB). "
          "Upload it to Google Drive > FYP, next to tiles.zip.")


if __name__ == "__main__":
    main()
