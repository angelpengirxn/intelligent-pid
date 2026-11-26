"""Build a one-page synthetic P&ID test sheet + YOLO ground truth.

Coordinate system: y increases DOWNWARD (same convention as src/synth_symbols.py).
Outputs: test_pid_sheet.png, test_pid_sheet.pdf (real text layer), test_pid_sheet.txt (YOLO labels).
"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

W, H = 1000.0, 700.0
DPI = 500                      # 1000 units * 5 px/unit = 5000 px wide
PX = DPI / 72.0 * (10.0 / W) * 72.0 / 1.0   # not used; see unit_to_px below
LW = 0.75                      # pipe / symbol line width in points
LWB = 1.4                      # border
FS = 4.2                       # tag text size
ink = "black"

fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
ax.set_facecolor("white"); fig.patch.set_facecolor("white")

boxes = []          # (cls, x0, y0, x1, y1) in drawing units


# ----------------------------------------------------------------- primitives
class Rec:
    """Collects the points of one symbol so its bounding box can be measured."""
    def __init__(self): self.pts = []
    def add(self, pts): self.pts.extend(pts)


def line(pts, rec=None, w=LW, dash=None, color=ink):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    kw = {} if dash is None else {"dashes": dash}
    ax.plot(xs, ys, color=color, lw=w, solid_capstyle="butt", zorder=2, **kw)
    if rec is not None: rec.add(pts)


def poly(pts, rec=None, fill=False, w=LW):
    ax.add_patch(Polygon(pts, closed=True, fill=fill,
                         facecolor=ink if fill else "none",
                         edgecolor=ink, lw=w, joinstyle="miter", zorder=2))
    if rec is not None: rec.add(pts)


def circle(cx, cy, r, rec=None, fill=False, w=LW, n=72):
    pts = [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
           for i in range(n)]
    poly(pts, rec, fill, w)


def text(x, y, s, size=FS, rec=None, ha="center", va="center", weight="normal", rot=0):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=ink, zorder=3,
            family="DejaVu Sans", weight=weight, rotation=rot)


def commit(cls, rec, pad=0.6):
    xs = [p[0] for p in rec.pts]; ys = [p[1] for p in rec.pts]
    boxes.append((cls, min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad))


# ----------------------------------------------------------------- symbol set
# Each function draws one symbol centred on (x, y) with half-length s, and records its box.
# vert=True rotates the symbol 90 degrees (for a vertical pipe run).

def _rot(pts, x, y, vert):
    if not vert: return [(x + a, y + b) for a, b in pts]
    return [(x - b, y + a) for a, b in pts]


def bowtie(x, y, s, rec, vert=False, fill=False, h=0.62):
    hh = s * h
    poly(_rot([(-s, -hh), (-s, hh), (0, 0)], x, y, vert), rec, fill)
    poly(_rot([(s, -hh), (s, hh), (0, 0)], x, y, vert), rec, fill)


def tri_in(x, y, s, rec, vert=False, fill=False, h=0.62):
    """Third triangle, base away from centre, tip at the centre (3-way / angle bodies)."""
    hh = s * h
    poly(_rot([(-hh, s), (hh, s), (0, 0)], x, y, vert), rec, fill)


def handwheel(x, y, s, rec, vert=False, up=-1):
    top = up * s * 0.95
    line(_rot([(0, 0), (0, top)], x, y, vert), rec)
    line(_rot([(-s * 0.42, top), (s * 0.42, top)], x, y, vert), rec)


def c00(x, y, s, vert=True):                       # Gate valve (bar)
    r = Rec(); bowtie(x, y, s, r, vert); handwheel(x, y, s, r, vert); commit(0, r)

def c01(x, y, s, vert=True):                       # Ball valve
    r = Rec(); bowtie(x, y, s, r, vert); circle(x, y, s * 0.36, r); commit(1, r)

def c02(x, y, s, vert=False):                      # Globe valve
    r = Rec(); bowtie(x, y, s, r, vert); circle(x, y, s * 0.33, r, fill=True); commit(2, r)

def c03(x, y, s, vert=False):                      # Gate valve
    r = Rec(); bowtie(x, y, s, r, vert); commit(3, r)

def c04(x, y, s, vert=False):                      # Ball valve (flanged)
    r = Rec(); bowtie(x, y, s, r, vert); circle(x, y, s * 0.36, r)
    for k in (-1, 1):
        line(_rot([(k * s, -s * 0.72), (k * s, s * 0.72)], x, y, vert), r, w=LW * 1.5)
    commit(4, r)

def c05(x, y, s, vert=False):                      # Butterfly valve
    r = Rec()
    line(_rot([(-s, s * 0.6), (-s, -s * 0.1)], x, y, vert), r)
    line(_rot([(-s, -s * 0.1), (s, -s * 0.1)], x, y, vert), r)
    line(_rot([(s, s * 0.6), (s, -s * 0.1)], x, y, vert), r)
    cx, cy = _rot([(0, -s * 0.1)], x, y, vert)[0]
    circle(cx, cy, s * 0.3, r, fill=True)
    commit(5, r)

def c06(x, y, s, vert=True):                       # Lever-operated valve
    r = Rec(); bowtie(x, y, s, r, vert)
    line(_rot([(0, 0), (0, -s * 1.0)], x, y, vert), r)
    line(_rot([(-s * 0.75, -s * 1.0), (s * 0.5, -s * 1.0)], x, y, vert), r, w=LW * 1.4)
    commit(6, r)

def c07(x, y, s, vert=False):                      # Non-return valve
    r = Rec()
    poly(_rot([(-s, -s * 0.62), (-s, s * 0.62), (s, 0)], x, y, vert), r)
    line(_rot([(s, -s * 0.72), (s, s * 0.72)], x, y, vert), r, w=LW * 1.4)
    commit(7, r)

def c08(x, y, s, vert=False):                      # Diaphragm valve
    r = Rec(); bowtie(x, y, s, r, vert)
    arc = [(s * math.cos(math.radians(a)), -s * 0.62 * 1.0 - 0 + s * 0.62 * 0 + s * math.sin(math.radians(a)) * 0.45)
           for a in range(180, 361, 10)]
    line(_rot(arc, x, y, vert), r)
    commit(8, r)

def c09(x, y, s, vert=False):                      # Needle valve
    r = Rec(); bowtie(x, y, s, r, vert)
    poly(_rot([(-s * 0.26, s * 1.05), (s * 0.26, s * 1.05), (0, 0)], x, y, vert), r, fill=True)
    commit(9, r)

def c10(x, y, s, vert=True):                       # Gate valve (partially open)
    r = Rec()
    hh = s * 0.62
    poly(_rot([(-s, -hh), (-s, hh), (0, 0)], x, y, vert), r, fill=False)
    poly(_rot([(s, -hh), (s, hh), (0, 0)], x, y, vert), r, fill=True)
    commit(10, r)

def c11(x, y, s, vert=False):                      # Gate valve (normally closed)
    r = Rec(); bowtie(x, y, s, r, vert, fill=True); commit(11, r)

def c12(x, y, s, vert=True):                       # Globe valve (normally closed)
    r = Rec(); bowtie(x, y, s, r, vert, fill=True); circle(x, y, s * 0.3, r, fill=True); commit(12, r)

def c13(x, y, s, vert=False):                      # Control valve (diaphragm actuator)
    r = Rec(); bowtie(x, y, s, r, vert)
    line(_rot([(0, 0), (0, -s * 0.8)], x, y, vert), r)
    dome = [(s * 0.55 * math.cos(math.radians(a)), -s * 0.8 + s * 0.45 * math.sin(math.radians(a)))
            for a in range(180, 361, 10)]
    poly(_rot(dome + [dome[0]], x, y, vert), r)
    commit(13, r)

def c14(x, y, s, vert=True):                       # Plug valve (open)
    r = Rec(); circle(x, y, s * 0.6, r)
    for k in (-1, 1):
        line(_rot([(0, k * s * 0.6), (0, k * s)], x, y, vert), r)
    commit(14, r)

def c15(x, y, s, vert=True):                       # Plug valve (closed)
    r = Rec(); circle(x, y, s * 0.6, r, fill=True)
    for k in (-1, 1):
        line(_rot([(0, k * s * 0.6), (0, k * s)], x, y, vert), r)
    commit(15, r)

def c16(x, y, s, vert=False):                      # Line blind flange
    r = Rec()
    for dx in (-s * 0.25, -s * 0.05):
        line(_rot([(dx, -s * 0.6), (dx, s * 0.6)], x, y, vert), r, w=LW * 1.3)
    line(_rot([(-s * 0.05, 0), (s * 0.45, 0)], x, y, vert), r)
    cx, cy = _rot([(s * 0.75, 0)], x, y, vert)[0]
    circle(cx, cy, s * 0.3, r)
    commit(16, r)

def c17(x, y, s, vert=True):                       # Spectacle blind (closed)
    r = Rec()
    cf = _rot([(0, -s * 0.3)], x, y, vert)[0]; co = _rot([(0, -s * 0.95)], x, y, vert)[0]
    circle(cf[0], cf[1], s * 0.34, r, fill=True)
    circle(co[0], co[1], s * 0.3, r)
    for dy in (s * 0.25, s * 0.55):
        line(_rot([(-s * 0.45, dy), (s * 0.45, dy)], x, y, vert), r, w=LW * 1.3)
    commit(17, r)

def c18(x, y, s, vert=False):                      # Spectacle blind (open)
    r = Rec()
    a = _rot([(-s * 0.75, 0)], x, y, vert)[0]; b = _rot([(-s * 0.2, 0)], x, y, vert)[0]
    circle(a[0], a[1], s * 0.34, r, fill=True)
    circle(b[0], b[1], s * 0.3, r)
    for dx in (s * 0.3, s * 0.55):
        line(_rot([(dx, -s * 0.6), (dx, s * 0.6)], x, y, vert), r, w=LW * 1.3)
    commit(18, r)

def c19(x, y, s, vert=False):                      # Reducer
    r = Rec()
    poly(_rot([(-s * 0.7, -s * 0.62), (s * 0.7, -s * 0.34), (s * 0.7, s * 0.34), (-s * 0.7, s * 0.62)],
              x, y, vert), r)
    commit(19, r)

def c20(x, y, s, vert=False):                      # Flange
    r = Rec()
    for dx in (-s * 0.12, s * 0.12):
        line(_rot([(dx, -s * 0.62), (dx, s * 0.62)], x, y, vert), r, w=LW * 1.4)
    commit(20, r)

def c21(x, y, s, vert=False):                      # In-line mixer
    r = Rec()
    poly(_rot([(-s * 0.5, -s * 0.8), (s * 0.5, -s * 0.8), (s * 0.5, s * 0.8), (-s * 0.5, s * 0.8)],
              x, y, vert), r)
    for k in range(3):
        y0 = -s * 0.8 + k * s * 0.533
        line(_rot([(-s * 0.5, y0), (s * 0.5, y0 + s * 0.533)], x, y, vert), r, w=LW * 0.9)
        line(_rot([(s * 0.5, y0), (-s * 0.5, y0 + s * 0.533)], x, y, vert), r, w=LW * 0.9)
    commit(21, r)

def c22(x, y, s, vert=False):                      # Jacketed pipe
    r = Rec()
    line(_rot([(-s, -s * 0.5), (s, -s * 0.5)], x, y, vert), r)
    line(_rot([(-s, s * 0.5), (s, s * 0.5)], x, y, vert), r)
    for k in (-1, 1):
        e = [(k * s + k * s * 0.12 * math.cos(math.radians(a)), s * 0.5 * math.sin(math.radians(a)))
             for a in range(90, 271, 10)]
        line(_rot(e, x, y, vert), r, w=LW * 0.9)
    line(_rot([(-s * 0.8, 0), (s * 0.8, 0)], x, y, vert), r, w=LW * 0.8, dash=(4, 3))
    commit(22, r)

def c23(x, y, s, vert=True):                       # Flow-direction arrow
    r = Rec()
    poly(_rot([(-s * 0.45, -s * 0.5), (s * 0.45, -s * 0.5), (0, s * 0.55)], x, y, vert), r, fill=True)
    commit(23, r)

def c24(x, y, s, vert=True):                       # Sight glass
    r = Rec(); circle(x, y, s * 0.62, r)
    d = s * 0.62 / math.sqrt(2)
    line(_rot([(-d, -d), (d, d)], x, y, vert), r)
    line(_rot([(-d, d), (d, -d)], x, y, vert), r)
    for k in (-1, 1):
        line(_rot([(0, k * s * 0.62), (0, k * s * 1.05)], x, y, vert), r)
    commit(24, r)


# -------- instruments (bubbles) --------
def bubble(x, y, s, cls, letters, num, kind):
    """kind: 'field' | 'room' | 'aux' | 'dcs' | 'box' | 'tagbox'"""
    r = Rec()
    if kind == "dcs":
        poly([(x - s, y - s), (x + s, y - s), (x + s, y + s), (x - s, y + s)], r)
        circle(x, y, s * 0.92, r)
    elif kind == "box":
        poly([(x - s * 0.95, y - s * 0.85), (x + s * 0.95, y - s * 0.85),
              (x + s * 0.95, y + s * 0.85), (x - s * 0.95, y + s * 0.85)], r, w=LW * 1.3)
    elif kind == "tagbox":
        poly([(x - s * 1.05, y - s), (x + s * 1.05, y - s),
              (x + s * 1.05, y + s), (x - s * 1.05, y + s)], r, w=LW * 1.3)
    else:
        circle(x, y, s, r)
    if kind == "room":
        line([(x - s, y), (x + s, y)], r)
    if kind == "aux":
        line([(x - s, y - s * 0.09), (x + s, y - s * 0.09)], r)
        line([(x - s, y + s * 0.09), (x + s, y + s * 0.09)], r)
    if kind == "dcs":
        line([(x - s * 0.92, y), (x + s * 0.92, y)], r)
    if kind == "tagbox":
        text(x, y - s * 0.52, letters, FS * 0.92)
        text(x, y, num, FS * 0.92)
        text(x, y + s * 0.52, "LG-10", FS * 0.92)
    elif num:
        text(x, y - s * 0.34, letters, FS * 1.0)
        text(x, y + s * 0.36, num, FS * 1.0)
    else:
        text(x, y, letters, FS * 1.05)
    commit(cls, r)


def c41(x, y, s):                                  # PLC function
    r = Rec()
    poly([(x - s, y - s), (x + s, y - s), (x + s, y + s), (x - s, y + s)], r)
    poly([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], r)
    text(x, y - s * 0.3, "KY", FS * 0.9); text(x, y + s * 0.32, "7140", FS * 0.9)
    commit(41, r)


def c42(x, y, s):                                  # Computer function
    r = Rec(); h = s * math.sqrt(3) / 2
    poly([(x - s, y), (x - s / 2, y - h), (x + s / 2, y - h),
          (x + s, y), (x + s / 2, y + h), (x - s / 2, y + h)], r)
    text(x, y - s * 0.26, "UY", FS * 0.9); text(x, y + s * 0.3, "7141", FS * 0.9)
    commit(42, r)


def c43(x, y, s):                                  # Interlock
    r = Rec()
    poly([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], r)
    text(x, y, "I", FS * 1.1)
    commit(43, r)


# -------- new valve / fitting classes (shapes follow src/synth_symbols.py) --------
def c32(x, y, s):                                  # Safety relief valve (inlet below, outlet right)
    r = Rec(); hh = s * 0.62
    poly([(x - hh, y + s), (x + hh, y + s), (x, y)], r)
    poly([(x + s, y - hh), (x + s, y + hh), (x, y)], r)
    top = y - s * 0.95; y0 = y - s * 0.22; w = s * 0.17
    line([(x, y), (x, y0)], r)
    zz = [(x, y0)] + [(x + (w if i % 2 else -w), y0 + (top - y0) * (i + 0.5) / 4) for i in range(4)] + [(x, top)]
    line(zz, r)
    line([(x - w * 1.3, top), (x + w * 1.3, top)], r)
    commit(32, r)

def c33(x, y, s, vert=False):                      # Three-way valve
    r = Rec(); bowtie(x, y, s, r, vert); tri_in(x, y, s, r, vert); handwheel(x, y, s, r, vert)
    commit(33, r)

def c34(x, y, s):                                  # Angle valve (inlet left, outlet down)
    r = Rec(); hh = s * 0.62
    poly([(x - s, y - hh), (x - s, y + hh), (x, y)], r)
    poly([(x - hh, y + s), (x + hh, y + s), (x, y)], r)
    handwheel(x, y, s, r, False)
    commit(34, r)

def _actuated(x, y, s, letter, square):
    r = Rec(); bowtie(x, y, s, r, False)
    top = y - s * 0.44 * 2
    line([(x, y), (x, top)], r)
    a = s * 0.86
    if square:
        poly([(x - a / 2, top), (x + a / 2, top), (x + a / 2, top - a), (x - a / 2, top - a)], r)
    else:
        circle(x, top - a / 2, a / 2, r)
    text(x, top - a / 2, letter, FS * 1.05)
    return r

def c35(x, y, s): commit(35, _actuated(x, y, s, "S", True))     # Solenoid valve
def c36(x, y, s): commit(36, _actuated(x, y, s, "M", False))    # Motor-operated valve

def c37(x, y, s):                                  # Pressure regulator
    r = Rec(); bowtie(x, y, s, r, False)
    top = y - s * 0.86
    line([(x, y), (x, top)], r)
    rr = s * 0.66
    dome = [(x + rr * math.cos(math.radians(a)), top + rr * math.sin(math.radians(a)))
            for a in range(180, 361, 10)]
    poly(dome + [dome[0]], r)
    xs, ys = x + rr, top - rr * 0.35
    xe = x + s * 1.9
    line([(xs, ys), (xe, ys), (xe, y)], r, w=LW * 0.8, dash=(4, 3))
    commit(37, r)

def c38(x, y, s):                                  # Y-strainer (flanged body + solid leg)
    r = Rec(); half = s * 0.9; fh = s * 0.64
    line([(x - half, y), (x + half, y)], r, w=LW * 1.3)
    for k in (-1, 1):
        line([(x + k * half, y - fh / 2), (x + k * half, y + fh / 2)], r, w=LW * 1.6)
    a = math.radians(45); ux, uy = math.cos(a), math.sin(a)
    ln, lw_ = s * 1.15, s * 0.22
    x0 = x + half * 0.35
    end = (x0 + ux * ln, y + uy * ln)
    px, py = -uy, ux
    poly([(x0 + px * lw_ / 2, y + py * lw_ / 2), (end[0] + px * lw_ / 2, end[1] + py * lw_ / 2),
          (end[0] - px * lw_ / 2, end[1] - py * lw_ / 2), (x0 - px * lw_ / 2, y - py * lw_ / 2)], r, fill=True)
    c = lw_ * 1.9
    line([(end[0] + px * c, end[1] + py * c), (end[0] - px * c, end[1] - py * c)], r, w=LW * 1.4)
    commit(38, r)

def c39(x, y, s):                                  # Steam trap
    r = Rec(); rr = s * 0.62
    circle(x, y, rr, r)
    text(x, y, "T", FS * 1.1)
    commit(39, r)

def c40(x, y, s):                                  # Centrifugal pump
    r = Rec(); rr = s * 0.95
    circle(x, y, rr, r)
    line([(x, y - rr), (x + rr * 1.15, y - rr)], r)
    b = rr * 1.35
    line([(x - rr * 0.6, y + rr * 0.8), (x - b * 0.75, y + rr * 1.35),
          (x + b * 0.75, y + rr * 1.35), (x + rr * 0.6, y + rr * 0.8)], r)
    commit(40, r)

def c25(x, y, s, letters, num): bubble(x, y, s, 25, letters, num, "field")
def c26(x, y, s, letters, num): bubble(x, y, s, 26, letters, num, "field")
def c27(x, y, s, letters, num): bubble(x, y, s, 27, letters, num, "room")
def c28(x, y, s, letters, num): bubble(x, y, s, 28, letters, num, "aux")
def c29(x, y, s, letters):      bubble(x, y, s, 29, letters, "", "box")
def c30(x, y, s, letters, num): bubble(x, y, s, 30, letters, num, "dcs")
def c31(x, y, s, letters, num): bubble(x, y, s, 31, letters, num, "tagbox")


# ============================================================ sheet layout
S = 7.0          # standard symbol half-size (14 units = 70 px on the PNG)
B = 9.0          # instrument bubble radius

def pipe(pts, w=LW, dash=None):
    line(pts, None, w=w, dash=dash)

def signal(p0, p1):
    line([p0, p1], None, w=LW * 0.8, dash=(3.5, 2.5))

def tag(x, y, s, size=FS, ha="center", va="center", rot=0):
    text(x, y, s, size, ha=ha, va=va, rot=rot)

# ---- border and title block ----
pipe([(18, 18), (W - 18, 18), (W - 18, H - 18), (18, H - 18), (18, 18)], w=LWB)
pipe([(24, 24), (W - 24, 24), (W - 24, H - 24), (24, H - 24), (24, 24)], w=LW)
TBX, TBY = W - 268, H - 112
pipe([(TBX, TBY), (W - 24, TBY), (W - 24, H - 24), (TBX, H - 24), (TBX, TBY)], w=LWB)
for dy in (26, 52, 70):
    pipe([(TBX, TBY + dy), (W - 24, TBY + dy)])
tag(TBX + 8, TBY + 13, "UNIVERSITI TEKNOLOGI PETRONAS", FS * 1.05, ha="left")
tag(TBX + 8, TBY + 39, "SYNTHETIC TEST SHEET - P&ID SYMBOL DETECTION", FS * 0.9, ha="left")
tag(TBX + 8, TBY + 61, "INSTRUMENT AIR AND UTILITY WATER SYSTEM", FS * 0.9, ha="left")
tag(TBX + 8, TBY + 82, "DWG No.  UTP-FYP-PID-001", FS * 0.9, ha="left")
tag(W - 32, TBY + 82, "REV  0", FS * 0.9, ha="right")

tag(40, 40, "NOTES:", FS * 0.9, ha="left")
tag(40, 54, "1. ALL VALVE SIZES IN NB (mm).", FS * 0.85, ha="left")
tag(40, 66, "2. INSTRUMENT TAGS TO ISA-5.1.", FS * 0.85, ha="left")

# ---- main runs ----
R1, R2, R3, R4 = 120.0, 268.0, 414.0, 562.0
X0, X1 = 56.0, 944.0

pipe([(X0, R1), (X1, R1)])
pipe([(X0, R2), (X1, R2)])
pipe([(X0, R3), (X1, R3)])
pipe([(X0, R4), (X1, R4)])

tag(150, R1 - 9, '6"-IA-7101-A1A', FS * 0.9)
tag(150, R2 - 9, '4"-IA-7102-A1A', FS * 0.9)
tag(150, R3 - 9, '3"-CW-7201-B2B', FS * 0.9)
tag(150, R4 - 9, '2"-ST-7301-C1C', FS * 0.9)

# ---- row 1: instrument-air header ----
c20(96, R1, S);                       tag(96, R1 + 20, "FL-01", FS * 0.85)
c19(206, R1, S)
c03(268, R1, S);                      tag(268, R1 + 20, "25GV", FS * 0.85)
c38(344, R1, S);                      tag(344, R1 - 20, "ST-7101", FS * 0.85)
c07(420, R1, S);                      tag(420, R1 + 20, "NRV", FS * 0.85)
c13(498, R1, S)
c26(574, R1, B * 0.86, "RO", "7103")
c33(660, R1, S);                      tag(660, R1 + 22, "15TWV", FS * 0.85)
c36(742, R1, S);                      tag(742, R1 + 20, "MOV-01", FS * 0.85)
c02(822, R1, S);                      tag(822, R1 + 20, "25GLV", FS * 0.85)
c23(886, R1 + 26, S, vert=True)
pipe([(886, R1), (886, R1 + 46)])

# ---- row 2: distribution header ----
c40(96, R2, S * 1.25);                tag(96, R2 + 32, "P-101A", FS * 0.9)
c04(190, R2, S);                      tag(190, R2 + 20, "25BV", FS * 0.85)
c08(262, R2, S);                      tag(262, R2 + 22, "15DV", FS * 0.85)
c09(334, R2, S);                      tag(334, R2 + 24, "15NV", FS * 0.85)
c05(410, R2, S);                      tag(410, R2 + 22, "50BFV", FS * 0.85)
c35(486, R2, S);                      tag(486, R2 + 20, "SOV-02", FS * 0.85)
c37(574, R2, S)
c11(672, R2, S);                      tag(672, R2 + 20, "25GV", FS * 0.85)
c39(748, R2, S);                      tag(748, R2 + 22, "ST-02", FS * 0.85)
c18(826, R2, S)
c16(896, R2, S)

# ---- row 3: cooling water ----
c22(104, R3, S)
c21(196, R3, S)
c34(286, R3, S);                      tag(286, R3 - 20, "15AV", FS * 0.85)
pipe([(286, R3), (286, R4)])
c42(372, R3 - 62, B * 1.05)
signal((372, R3 - 62 + B * 1.05), (372, R3))
c13(452, R3, S)
c07(534, R3, S);                      tag(534, R3 + 20, "NRV", FS * 0.85)
c19(606, R3, S)
c03(676, R3, S);                      tag(676, R3 + 20, "50GV", FS * 0.85)
c32(760, R3 - 34, S)
pipe([(760, R3), (760, R3 - 34 + S)])
pipe([(760 + S, R3 - 34), (830, R3 - 34)])
tag(800, R3 - 44, "PSV-7107", FS * 0.85)
c20(884, R3, S)

# ---- row 4: steam / condensate ----
c03(110, R4, S);                      tag(110, R4 + 20, "25GV", FS * 0.85)
c39(188, R4, S);                      tag(188, R4 + 22, "ST-03", FS * 0.85)
c38(266, R4, S)
c11(352, R4, S);                      tag(352, R4 + 20, "15GV", FS * 0.85)
c05(430, R4, S);                      tag(430, R4 + 22, "80BFV", FS * 0.85)
c02(508, R4, S)
c36(590, R4, S);                      tag(590, R4 + 20, "MOV-02", FS * 0.85)
c35(676, R4, S)
c07(756, R4, S);                      tag(756, R4 + 20, "NRV", FS * 0.85)
c04(834, R4, S);                      tag(834, R4 + 20, "50BV", FS * 0.85)
c19(900, R4, S)

# ---- vertical branches with vertical-oriented symbols ----
def branch(x, ytop, ybot, items):
    pipe([(x, ytop), (x, ybot)])
    for f, yy, *rest in items:
        f(x, yy, S)

pipe([(230, R1), (230, R2)]);  c00(230, (R1 + R2) / 2 - 16, S); c01(230, (R1 + R2) / 2 + 26, S)
tag(248, (R1 + R2) / 2 - 16, "25GV", FS * 0.85, ha="left")
tag(248, (R1 + R2) / 2 + 26, "15BV", FS * 0.85, ha="left")

pipe([(392, R1), (392, R2)]);  c06(392, (R1 + R2) / 2 - 14, S); c24(392, (R1 + R2) / 2 + 30, S)
tag(410, (R1 + R2) / 2 - 14, "15LV", FS * 0.85, ha="left")
tag(410, (R1 + R2) / 2 + 30, "SG-01", FS * 0.85, ha="left")

pipe([(620, R1), (620, R2)]);  c10(620, (R1 + R2) / 2 - 14, S); c14(620, (R1 + R2) / 2 + 30, S)
tag(638, (R1 + R2) / 2 + 30, "25PV", FS * 0.85, ha="left")

pipe([(862, R1), (862, R2)]);  c12(862, (R1 + R2) / 2 - 14, S); c15(862, (R1 + R2) / 2 + 30, S)

pipe([(150, R2), (150, R3)]);  c17(150, (R2 + R3) / 2 + 6, S)
pipe([(470, R3), (470, R4)]);  c01(470, (R3 + R4) / 2 - 12, S); c23(470, (R3 + R4) / 2 + 28, S, vert=True)
tag(488, (R3 + R4) / 2 - 12, "25BV", FS * 0.85, ha="left")
pipe([(712, R3), (712, R4)]);  c00(712, (R3 + R4) / 2 - 12, S); c14(712, (R3 + R4) / 2 + 28, S)
pipe([(906, R3), (906, R4)]);  c06(906, (R3 + R4) / 2 + 6, S)

# ---- instruments with signal lines ----
INST = [
    (c25, 268, R1 - 62, "PI",  "7101", 268, R1),
    (c27, 498, R1 - 72, "FIC", "7103", 498, R1),
    (c25, 660, R1 - 62, "TI",  "7104", 660, R1),
    (c28, 822, R1 - 72, "PIC", "7105", 822, R1),
    (c30, 190, R2 + 68, "LIC", "7110", 190, R2),
    (c25, 410, R2 + 62, "FI",  "7111", 410, R2),
    (c27, 672, R2 + 72, "TIC", "7112", 672, R2),
    (c25, 452, R3 - 62, "PT",  "7202", 452, R3),
    (c28, 676, R3 + 70, "LT",  "7203", 676, R3),
    (c27, 352, R4 + 72, "TIC", "7301", 352, R4),
    (c25, 590, R4 + 62, "PI",  "7302", 590, R4),
    (c30, 620, R4 - 78, "FIC", "7303", 620, R4),
]
for f, bx, by, lt, nm, px_, py_ in INST:
    f(bx, by, B, lt, nm)
    signal((bx, by + (B if by < py_ else -B)), (px_, py_))

c29(742, R1 - 66, B * 0.95, "HS")
signal((742, R1 - 66 + B * 0.95), (742, R1))
c41(486, R2 + 66, B * 1.0)
signal((486, R2 + 66 - B), (486, R2))
c43(906, R2 + 64, B * 0.9)
signal((906, R2 + 64 - B * 0.9), (906, R2))
c31(104, R3 - 66, B * 0.95, "646", "214")
signal((104, R3 - 66 + B * 0.95), (104, R3))

# ============================================================ output
import pathlib
out = pathlib.Path(__file__).parent
fig.savefig(out / "test_pid_sheet.png", dpi=DPI, facecolor="white")
fig.savefig(out / "test_pid_sheet.pdf", facecolor="white")

PXW = int(round(W / 100.0 * DPI)); PXH = int(round(H / 100.0 * DPI))
lines = []
for cls, x0, y0, x1, y1 in boxes:
    cx = (x0 + x1) / 2 / W; cy = (y0 + y1) / 2 / H
    bw = (x1 - x0) / W; bh = (y1 - y0) / H
    lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
(out / "test_pid_sheet.txt").write_text("\n".join(lines) + "\n")

from collections import Counter
cnt = Counter(b[0] for b in boxes)
print(f"{PXW}x{PXH} px, {len(boxes)} symbols, {len(cnt)} distinct classes")
print("missing classes:", sorted(set(range(44)) - set(cnt)))
