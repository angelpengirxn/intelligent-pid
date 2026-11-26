from pathlib import Path
from PIL import Image, ImageDraw

crops = sorted(Path("docs/symbol_crops").glob("class_*.png"))
cell, cols = 160, 8
rows = (len(crops) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell, rows * (cell + 20)), "white")
draw = ImageDraw.Draw(sheet)

for i, p in enumerate(crops):
    img = Image.open(p).convert("RGB")
    img.thumbnail((cell - 10, cell - 10))
    x, y = (i % cols) * cell, (i // cols) * (cell + 20)
    sheet.paste(img, (x + 5, y + 5))
    draw.text((x + 5, y + cell), p.stem, fill="black")

sheet.save("docs/symbol_sheet.png")
print("Saved docs/symbol_sheet.png")