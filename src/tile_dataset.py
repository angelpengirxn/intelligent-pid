import random
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SRC = Path("data/raw/digitize_pid")
TEST = Path("data/raw/test_samples")
OUT = Path("data/tiles")
TILE, OVERLAP = 640, 128          # tile size and overlap in pixels
STRIDE = TILE - OVERLAP
MIN_VISIBLE = 0.6                 # keep a box only if 60%+ of it is inside the tile
KEEP_EMPTY = 0.1                  # keep 10% of tiles with no symbols (teaches "background")
random.seed(42)

test_stems = {p.stem for p in TEST.glob("*")}
labels = {p.stem: p for p in SRC.rglob("*.txt")}

def starts(length):
    s = list(range(0, max(length - TILE, 0) + 1, STRIDE))
    if s[-1] + TILE < length:
        s.append(length - TILE)
    return s

counts = {"train": 0, "val": 0}
for img_path in sorted(SRC.rglob("*")):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    if img_path.stem in test_stems:          # never tile your test drawings
        continue
    lbl = labels.get(img_path.stem)
    if lbl is None:
        continue
    split = "val" if any("val" in part for part in img_path.parts) else "train"

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    boxes = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c, xc, yc, w, h = int(p[0]), *map(float, p[1:])
        boxes.append((c, (xc - w/2) * W, (yc - h/2) * H, (xc + w/2) * W, (yc + h/2) * H))

    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    for y0 in starts(H):
        for x0 in starts(W):
            lines = []
            for c, x1, y1, x2, y2 in boxes:
                ix1, iy1 = max(x1, x0), max(y1, y0)
                ix2, iy2 = min(x2, x0 + TILE), min(y2, y0 + TILE)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                if (ix2 - ix1) * (iy2 - iy1) < MIN_VISIBLE * (x2 - x1) * (y2 - y1):
                    continue
                cx = ((ix1 + ix2) / 2 - x0) / TILE
                cy = ((iy1 + iy2) / 2 - y0) / TILE
                bw, bh = (ix2 - ix1) / TILE, (iy2 - iy1) / TILE
                lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if not lines and random.random() > KEEP_EMPTY:
                continue
            name = f"{img_path.stem}_{x0}_{y0}"
            img.crop((x0, y0, x0 + TILE, y0 + TILE)).save(OUT / "images" / split / f"{name}.jpg", quality=95)
            (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            counts[split] += 1
    print(f"done {img_path.name}")

print("Tiles:", counts)