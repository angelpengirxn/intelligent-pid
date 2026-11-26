from pathlib import Path
from PIL import Image

root = Path("data/raw/digitize_pid")
out = Path("docs/symbol_crops")
out.mkdir(parents=True, exist_ok=True)

images = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
saved = set()

for lbl in root.rglob("*.txt"):
    img_path = images.get(lbl.stem)
    if not img_path:
        continue
    img = None
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        if cls in saved:
            continue
        if img is None:
            img = Image.open(img_path)
        W, H = img.size
        xc, yc, w, h = map(float, parts[1:])
        pad = 10
        box = (int((xc - w/2) * W) - pad, int((yc - h/2) * H) - pad,
               int((xc + w/2) * W) + pad, int((yc + h/2) * H) + pad)
        img.crop(box).save(out / f"class_{cls:02d}.png")
        saved.add(cls)

print(f"Saved {len(saved)} symbol crops to {out}")