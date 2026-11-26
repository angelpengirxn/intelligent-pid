import random
from pathlib import Path
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

SRC = Path("data/raw/digitize_pid")
TILES = Path("data/tiles")
TILE = 640

# pick one drawing that was tiled
tile_imgs = list((TILES / "images" / "train").glob("*.jpg"))
stem = random.choice(tile_imgs).stem.rsplit("_", 2)[0]
orig = next(p for p in SRC.rglob(f"{stem}.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

img = Image.open(orig).convert("RGB")
d = ImageDraw.Draw(img)
count = 0

for lbl in (TILES / "labels" / "train").glob(f"{stem}_*.txt"):
    x0, y0 = map(int, lbl.stem.rsplit("_", 2)[1:])
    for line in lbl.read_text().splitlines():
        c, xc, yc, w, h = line.split()
        xc, yc = float(xc) * TILE + x0, float(yc) * TILE + y0
        w, h = float(w) * TILE, float(h) * TILE
        d.rectangle([xc - w/2, yc - h/2, xc + w/2, yc + h/2], outline="red", width=4)
        d.text((xc - w/2, yc - h/2 - 14), c, fill="red")
        count += 1

img.save("docs/tile_check_full.jpg", quality=90)
print(f"{orig.name}: drew {count} boxes → docs/tile_check_full.jpg")