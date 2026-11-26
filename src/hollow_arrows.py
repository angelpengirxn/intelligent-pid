"""Copy every training tile that contains a flow arrow (class 23),
hollow out the arrow (keep the outline, fill white), and save it as a new training tile."""
from pathlib import Path
import numpy as np
from PIL import Image

ARROW = 23
R = 2          # outline thickness kept, in pixels
IMG = Path("data/tiles/images/train")
LBL = Path("data/tiles/labels/train")

def erode(mask, r):
    """True only where the whole (2r+1)x(2r+1) neighbourhood is True."""
    out = mask.copy()
    p = np.pad(mask, r, constant_values=False)
    h, w = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out &= p[r + dy:r + dy + h, r + dx:r + dx + w]
    return out

made = 0
for lbl in sorted(LBL.glob("*.txt")):
    if lbl.stem.endswith("_hollow"):
        continue
    lines = lbl.read_text().splitlines()
    arrows = [l.split() for l in lines if l.split() and int(l.split()[0]) == ARROW]
    if not arrows:
        continue
    img = np.array(Image.open(IMG / f"{lbl.stem}.jpg").convert("L"))
    H, W = img.shape
    for _, xc, yc, w, h in arrows:
        xc, yc, w, h = float(xc) * W, float(yc) * H, float(w) * W, float(h) * H
        x1, y1 = max(0, int(xc - w / 2)), max(0, int(yc - h / 2))
        x2, y2 = min(W, int(xc + w / 2) + 1), min(H, int(yc + h / 2) + 1)
        crop = img[y1:y2, x1:x2]
        inside = erode(crop < 128, R)          # solid interior of the triangle
        crop[inside] = 255                     # make it white, outline stays
    Image.fromarray(img).convert("RGB").save(IMG / f"{lbl.stem}_hollow.jpg", quality=95)
    (LBL / f"{lbl.stem}_hollow.txt").write_text("\n".join(lines))
    made += 1

print(f"Created {made} hollow-arrow tiles")