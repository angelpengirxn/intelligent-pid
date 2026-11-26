from pathlib import Path
from collections import Counter

root = Path("data/raw/digitize_pid")
for split in ["train", "val"]:
    counts = Counter()
    for f in root.rglob(f"*{split}*/**/*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    print(f"\n=== {split.upper()} ===")
    print("Class ID | Count")
    for cls in sorted(counts):
        print(f"{cls:>8} | {counts[cls]}")