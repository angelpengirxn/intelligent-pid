"""Build P&ID-style test sheets that contain every trained symbol class.

Why: the model knows 44 classes, but the three real drawings in data/real_test/ only show a
handful of them. These sheets put every class on one drawing, so you can check that each class
is detected at all, and so per-class recall means something.

How: symbols are cut out of the training data itself (data/tiles for classes 0-31, data/synth
for classes 32-43) and composited onto a blank A3 drawing at 300 DPI with pipe runs, valve
codes and ISA instrument tags. The cut-outs keep the exact line weight and image quality the
model was trained on, so a class that is missed here is missed because of the model, not
because the test symbol was drawn differently.

Read this before quoting the numbers: because the symbols come from the training images, these
sheets are a COMPLETENESS check ("can the model find each class, and does tag reading work"),
not a measure of generalisation. The real-drawing evaluation in src/evaluate.py is what shows
how the model performs on drawings it has never seen. Report them separately.

Run:
    python3 src/make_test_sheet.py

Writes into data/test_sheets/:
    all_symbols_clean.png / .txt   pipe runs, tags, clean line work
    all_symbols_scan.png  / .txt   the same sheet degraded like a scan (skew, blur, noise, JPEG)
    all_symbols_key.png            a legend: every class drawn once with its id and name

The .txt files are YOLO ground truth for the matching PNG, so the sheets can be scored with
    python3 src/evaluate.py          (after copying the pair into data/real_test/)
or just uploaded to the running web app to look at by eye.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "test_sheets"
NAMES = json.loads((ROOT / "models" / "class_names.json").read_text())
N_CLASSES = len(NAMES)

# class ids -> (images dir, labels dir)
SOURCES = [(ROOT / "data/tiles/images/train", ROOT / "data/tiles/labels/train", range(0, 32)),
           (ROOT / "data/synth/images/train", ROOT / "data/synth/labels/train", range(32, N_CLASSES))]

SHEET = (4961, 3508)          # A3 at 300 DPI, landscape: the app upscales to 4960 px anyway
PER_CLASS = 3                 # instances of each class on the sheet
PAD = 5                       # pixels of background kept around each cut-out
INK = 25                      # near-black, like a printed drawing

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/System/Library/Fonts/Supplemental/Arial.ttf",
         "/Library/Fonts/Arial.ttf",
         "C:/Windows/Fonts/arial.ttf"]
_fcache = {}


def font(size):
    if size not in _fcache:
        for f in FONTS:
            try:
                _fcache[size] = ImageFont.truetype(f, size)
                break
            except OSError:
                continue
        else:
            _fcache[size] = ImageFont.load_default()
    return _fcache[size]


VALVE_CODE = {"Ball valve": "BV", "Ball valve (flanged)": "BV", "Gate valve": "GV",
              "Gate valve (bar)": "GV", "Gate valve (partially open)": "GV",
              "Gate valve (normally closed)": "GV", "Globe valve": "GLV",
              "Globe valve (normally closed)": "GLV", "Butterfly valve": "BFV",
              "Non-return valve": "NRV", "Diaphragm valve": "DV", "Needle valve": "NV",
              "Plug valve (open)": "PLV", "Plug valve (closed)": "PLV",
              "Safety relief valve": "SRV", "Three-way valve": "BV", "Angle valve": "GLV",
              "Solenoid valve": "SOV", "Motor-operated valve": "GV",
              "Pressure regulator": "PRV", "Control valve": "FCV"}
BORE = ["15", "25", "40", "50", "80", "100", "150", "200"]
ISA = ["PI", "TI", "FI", "LI", "PG", "TG", "PIC", "TIC", "FIC", "LIC", "PT", "TT", "FT", "LT"]


# ---------------------------------------------------------------- cutting symbols out
def whiten(crop):
    """Lift a JPEG tile's slightly grey background to pure white.

    Without this the cut-out leaves a visible grey rectangle on the sheet, which is an edge the
    detector could key on and which would not appear on a real drawing.
    """
    a = np.asarray(crop, dtype=np.float32)
    bg = float(np.percentile(a, 99))
    if bg >= 250:
        return crop
    a = np.clip(a * (255.0 / max(bg, 1.0)), 0, 255)
    return Image.fromarray(a.astype("uint8"), "L")


def boxes_of(path, W, H):
    out = []
    for line in path.read_text().split("\n"):
        p = line.split()
        if len(p) < 5:
            continue
        c, x, y, w, h = int(p[0]), *(float(v) for v in p[1:5])
        out.append((c, (x - w / 2) * W, (y - h / 2) * H, w * W, h * H))
    return out


def crop_bank(want=PER_CLASS * 3):
    """{class id: [grayscale cut-out with PAD of background around the symbol]}"""
    bank = defaultdict(list)
    for img_dir, lab_dir, ids in SOURCES:
        if not lab_dir.is_dir():
            sys.exit(f"missing {lab_dir} - unzip the dataset there first")
        need = set(ids)
        labs = sorted(lab_dir.glob("*.txt"))
        random.Random(7).shuffle(labs)
        for lab in labs:
            if not need:
                break
            hit = {c for c, *_ in boxes_of(lab, 1, 1)} & need
            if not hit:
                continue
            img = next((p for e in (".jpg", ".png", ".jpeg")
                        if (p := img_dir / (lab.stem + e)).exists()), None)
            if img is None:
                continue
            im = Image.open(img).convert("L")
            bs = boxes_of(lab, *im.size)
            for c, x, y, w, h in bs:
                if c not in need or w < 12 or h < 12 or w > 260 or h > 260:
                    continue
                r = (x - PAD, y - PAD, x + w + PAD, y + h + PAD)
                if r[0] < 0 or r[1] < 0 or r[2] > im.width or r[3] > im.height:
                    continue
                # the cut-out must hold one symbol only
                if any(o is not (c, x, y, w, h) and x - PAD < ox + ow and ox < x + w + PAD
                       and y - PAD < oy + oh and oy < y + h + PAD
                       for o in bs for _, ox, oy, ow, oh in [o] if o != (c, x, y, w, h)):
                    continue
                bank[c].append(whiten(im.crop(tuple(int(v) for v in r))))
                if len(bank[c]) >= want:
                    need.discard(c)
    missing = [i for i in range(N_CLASSES) if not bank[i]]
    if missing:
        print("WARNING no cut-out found for classes:", missing)
    return bank


# ---------------------------------------------------------------- composing the sheet
def new_sheet(title):
    im = Image.new("L", SHEET, 255)
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, SHEET[0] - 41, SHEET[1] - 41], outline=INK, width=6)
    d.rectangle([60, 60, SHEET[0] - 61, SHEET[1] - 61], outline=INK, width=2)
    # title block, bottom right
    x0, y0 = SHEET[0] - 1500, SHEET[1] - 420
    d.rectangle([x0, y0, SHEET[0] - 61, SHEET[1] - 61], outline=INK, width=4)
    for i, t in enumerate(["UNIVERSITI TEKNOLOGI PETRONAS", title,
                           "SYNTHETIC TEST SHEET - NOT A REAL PLANT DRAWING",
                           "DWG No. TEST-001        REV 0        SCALE NTS"]):
        d.text((x0 + 30, y0 + 30 + i * 80), t, fill=INK, font=font(46 if i < 2 else 36))
    return im, d


def paste(sheet, crop, cx, cy):
    """Darken-composite the cut-out so white background does not erase the pipe underneath."""
    x, y = int(cx - crop.width / 2), int(cy - crop.height / 2)
    box = (x, y, x + crop.width, y + crop.height)
    sheet.paste(ImageChops.darker(sheet.crop(box), crop), box)
    return (x + PAD, y + PAD, crop.width - 2 * PAD, crop.height - 2 * PAD)   # the symbol's box


def build(bank, rng):
    sheet, d = new_sheet("P&ID SYMBOL DETECTION TEST - ALL 44 CLASSES")
    items = [(i, bank[i][k % len(bank[i])]) for i in range(N_CLASSES) if bank[i]
             for k in range(PER_CLASS)]
    rng.shuffle(items)

    rows = list(range(300, SHEET[1] - 560, 290))
    gt, loop, it = [], 7101, iter(items)
    for row_i, y in enumerate(rows):
        d.line([180, y, SHEET[0] - 180, y], fill=INK, width=4)          # the pipe run
        d.text((110, y - 60), f'4"-P-{1000 + row_i * 10}', fill=INK, font=font(40))
        x = 300
        while x < SHEET[0] - 420:
            try:
                cid, crop = next(it)
            except StopIteration:
                it = iter([])
                break
            cat = NAMES[cid]["category"]
            name = NAMES[cid]["name"]
            above = cat == "instrument" and crop.height < 180
            cy = y - crop.height // 2 - 70 if above else y
            if above:
                d.line([x, y, x, cy + crop.height // 2], fill=INK, width=3)
            bx, by, bw, bh = paste(sheet, crop, x, cy)
            gt.append((cid, bx, by, bw, bh))

            # text next to the symbol, in the form the pipeline knows how to read
            fs = max(22, min(34, int(crop.height * 0.42)))
            if cat == "valve" and name in VALVE_CODE and rng.random() < 0.75:
                d.text((x, y + crop.height // 2 + 12), rng.choice(BORE) + VALVE_CODE[name],
                       fill=INK, font=font(fs), anchor="ma")
            elif above and rng.random() < 0.8:
                d.text((x, cy - crop.height // 2 - fs - 16), f"9-{rng.choice(ISA)}-{loop}",
                       fill=INK, font=font(fs), anchor="ma")
                loop += 1
            x += crop.width + rng.randint(175, 255)
        if not items:
            break
    # a few vertical tie-ins, so the sheet does not read as pure rows
    for _ in range(6):
        a, b = sorted(rng.sample(range(len(rows)), 2))
        if b - a == 1:
            xv = rng.randint(400, SHEET[0] - 500)
            d.line([xv, rows[a], xv, rows[b]], fill=INK, width=4)
    print(f"placed {len(gt)} symbols, {len({c for c, *_ in gt})} distinct classes")
    return sheet, gt


def scanned(sheet, gt, rng):
    """Same sheet, degraded the way a scanned drawing is: slight skew, blur, noise, JPEG."""
    im = sheet.rotate(rng.uniform(-0.4, 0.4), resample=Image.BICUBIC, fillcolor=255)
    im = im.filter(ImageFilter.GaussianBlur(0.7))
    px = im.load()
    for _ in range(60000):                                   # speckle
        x, y = rng.randrange(im.width), rng.randrange(im.height)
        px[x, y] = max(0, px[x, y] - rng.randint(40, 120))
    return im, gt            # the rotation is small enough that the boxes still hold


def key_sheet(bank):
    cols, cell = 6, (780, 300)
    rows = (N_CLASSES + cols - 1) // cols
    im = Image.new("L", (cols * cell[0] + 80, rows * cell[1] + 160), 255)
    d = ImageDraw.Draw(im)
    d.text((40, 40), "Symbol key - the 44 trained classes", fill=INK, font=font(56))
    for i in range(N_CLASSES):
        cx = 40 + (i % cols) * cell[0] + cell[0] // 2
        cy = 150 + (i // cols) * cell[1] + 90
        if bank[i]:
            paste(im, bank[i][0], cx, cy)
        d.text((cx, cy + 105), f'{i}  {NAMES[i]["name"]}', fill=INK,
               font=font(32), anchor="ma")
    return im


def save(name, im, gt=None):
    OUT.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(OUT / f"{name}.png", optimize=True)
    if gt is not None:
        W, H = im.size
        (OUT / f"{name}.txt").write_text("\n".join(
            f"{c} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
            for c, x, y, w, h in gt) + "\n")
    print("wrote", OUT / f"{name}.png")


if __name__ == "__main__":
    rng = random.Random(3)
    print("cutting symbols out of the training data ...")
    bank = crop_bank()
    print("have cut-outs for", sum(1 for i in range(N_CLASSES) if bank[i]), "of", N_CLASSES, "classes")

    sheet, gt = build(bank, rng)
    save("all_symbols_clean", sheet, gt)
    save("all_symbols_scan", *scanned(sheet, gt, rng))
    save("all_symbols_key", key_sheet(bank))
