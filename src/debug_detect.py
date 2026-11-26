"""Diagnostic: run the trained model on one drawing and save what it sees, for debugging.

Run from the pid-analysis folder:
    python src/debug_detect.py "data/raw/real_samples/compressed air pid.pdf"
Writes docs/debug/: raw_detections.json (before merging), final.json (what the app shows),
crops of every detection per class, and a full annotated drawing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import ImageDraw  # noqa: E402

from app import pipeline as P  # noqa: E402

src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/real_samples/compressed air pid.pdf")
out = Path("docs/debug")
(out / "crops").mkdir(parents=True, exist_ok=True)
for old in (out / "crops").glob("*.png"):
    old.unlink()

data = src.read_bytes()
img, words = P.load_document(data, src.name)
model = P.get_model()
if model is None:
    sys.exit("No model at models/best.pt")
info = P.class_info(model)
W, H = img.size
print(f"{src.name}: {W}x{H}px, {len(words)} text lines in PDF")

# --- raw tiled predictions, low threshold, before any merging
import numpy as np  # noqa: E402
arr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
coords = [(x, y) for y in P._starts(H) for x in P._starts(W)]
raw = []
for i in range(0, len(coords), P.BATCH):
    batch = coords[i:i + P.BATCH]
    res = model.predict([P._tile(arr, x, y) for x, y in batch], imgsz=P.TILE, conf=0.10, verbose=False)
    for (x, y), r in zip(batch, res):
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            raw.append({"cls_id": int(c), "cls": info[int(c)]["name"], "conf": round(float(s), 3),
                        "xyxy": [round(float(b[0] + x), 1), round(float(b[1] + y), 1),
                                 round(float(b[2] + x), 1), round(float(b[3] + y), 1)],
                        "tile": [x, y]})
(out / "raw_detections.json").write_text(json.dumps(raw, indent=1))
print(f"raw detections (conf>=0.10, before merging): {len(raw)}")

# --- what the app shows
result = P.analyze(data, src.name)
final = [{k: d.get(k) for k in ("cls", "cat", "conf", "tag", "name", "b")} for d in result["detections"]]
(out / "final.json").write_text(json.dumps(final, indent=1))
print(f"final detections shown in app: {len(final)}")

# --- crops per class (padded) and annotated drawing
counts = {}
for n, d in enumerate(result["detections"]):
    x, y, w, h = d["b"]
    pad = max(w, h) * 0.6
    crop = img.crop((x - pad, y - pad, x + w + pad, y + h + pad))
    safe = "".join(ch if ch.isalnum() else "_" for ch in d["cls"])
    counts[safe] = counts.get(safe, 0) + 1
    crop.save(out / "crops" / f"{safe}_{counts[safe]:02d}_{d['conf']:.2f}.png")
annot = img.copy()
dr = ImageDraw.Draw(annot)
for d in result["detections"]:
    x, y, w, h = d["b"]
    dr.rectangle([x, y, x + w, y + h], outline="red", width=4)
    dr.text((x, y - 14), f'{d["cls"]} {d["conf"]:.2f} {d.get("tag") or ""}', fill="red")
annot.thumbnail((5000, 5000))
annot.save(out / "annotated.jpg", quality=85)
print("saved to docs/debug/  (crops/, annotated.jpg, raw_detections.json, final.json)")
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {v:3d}  {k}")
