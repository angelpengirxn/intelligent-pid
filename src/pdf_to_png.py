import fitz
from pathlib import Path

src = Path("data/raw/real_samples")
out = Path("data/real_test/images")
out.mkdir(parents=True, exist_ok=True)

for f in src.iterdir():
    if f.suffix.lower() == ".pdf":
        for i, page in enumerate(fitz.open(f)):
            page.get_pixmap(dpi=300).save(out / f"{f.stem}_p{i+1}.png")
    elif f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        (out / f.name).write_bytes(f.read_bytes())
print("Done:", len(list(out.iterdir())), "images")