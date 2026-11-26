"""Evaluate the system on real drawings labelled in makesense.ai (Step 4).

Compares two set-ups on the same pages:
  A) Model only   - YOLOv8 + tiling + merging (no drawing text)
  B) Full system  - the same detections, plus the drawing's text (tag reading, valve labels,
                    weak-box rescue and dropping unconfirmed boxes), i.e. exactly what the app shows

Before labelling (once):
    python src/evaluate.py --labels-list
  -> writes data/real_test/labels.txt. In makesense.ai, import it as the label list so the class
     numbers match the model (same order as models/class_names.json).

After labelling: Actions > Export Annotations > "YOLO format" zip. Unzip the .txt files into
    data/real_test/labels/        (e.g. "compressed air pid_p1.txt")
then run:
    python src/evaluate.py

Each label file is matched to its source: "<name>_p<N>.txt" -> page N of data/raw/real_samples/<name>.pdf
(so the PDF's text layer is used, like the app), otherwise the image of the same name in data/real_test/images.

Results go to docs/evaluation/: evaluation.xlsx (summary, per class, mistakes, instrument tags to check)
and one picture per page (green = correct, red = false detection, orange = missed, purple = wrong name).
"""
import argparse
import copy
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import pipeline  # noqa: E402

IMAGES = ROOT / "data" / "real_test" / "images"
LABELS = ROOT / "data" / "real_test" / "labels"
SOURCES = ROOT / "data" / "raw" / "real_samples"
OUT = ROOT / "docs" / "evaluation"
MODES = [("model", "A) Model only"), ("full", "B) Full system")]
GREEN, RED, ORANGE, PURPLE = (22, 163, 74), (220, 38, 38), (234, 120, 0), (147, 51, 234)


# ---------------------------------------------------------------- inputs
def class_rows():
    return sorted(json.loads(pipeline.NAMES_PATH.read_text(encoding="utf-8")), key=lambda r: int(r["id"]))


def write_label_list():
    rows = class_rows()
    ids = [int(r["id"]) for r in rows]
    if ids != list(range(len(rows))):
        sys.exit(f"class_names.json ids must be 0..{len(rows) - 1} with no gaps; found {ids}")
    path = ROOT / "data" / "real_test" / "labels.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(r["name"] for r in rows) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)} ({len(rows)} classes).")
    print("makesense.ai: on the 'Create labels' screen choose 'Load labels from file' and pick this file.")
    hidden = [r["name"] for r in rows if r.get("show") is False]
    if hidden:
        print("Hidden in the app (no need to label, they are ignored): " + ", ".join(hidden))


def find_source(stem: str):
    """label stem -> (file bytes, filename, page number)"""
    m = re.match(r"^(.*)_p(\d+)$", stem)
    if m:
        pdf = SOURCES / f"{m.group(1)}.pdf"
        if pdf.exists():
            return pdf.read_bytes(), pdf.name, int(m.group(2))
    for ext in (".png", ".jpg", ".jpeg"):
        img = IMAGES / f"{stem}{ext}"
        if img.exists():
            return img.read_bytes(), img.name, 1
    return None


def read_labels(path: Path, W: int, H: int, n_classes: int):
    boxes = []
    for i, line in enumerate(path.read_text().split("\n"), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            print(f"  ! {path.name} line {i}: expected 'class cx cy w h', skipped")
            continue
        c = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:])
        if not 0 <= c < n_classes:
            print(f"  ! {path.name} line {i}: class {c} is not 0..{n_classes - 1}, skipped")
            continue
        boxes.append({"cls_id": c, "xyxy": [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]})
    return boxes


# ---------------------------------------------------------------- matching
def xyxy(d):
    x, y, w, h = d["b"]
    return [x, y, x + w, y + h]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, thr):
    """Standard detection matching: highest confidence first, each label used once.
    Returns per prediction: (status, gt index, iou) with status tp / wrong_class / fp; and missed gt indices."""
    used = set()
    out = [None] * len(preds)
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i]["conf"]):
        p = preds[pi]
        best, bi = 0.0, None
        for gi, g in enumerate(gts):           # same class first
            if gi not in used and g["cls_id"] == p["cls_id"]:
                v = iou(xyxy(p), g["xyxy"])
                if v >= thr and v > best:
                    best, bi = v, gi
        if bi is not None:
            used.add(bi)
            out[pi] = ("tp", bi, best)
            continue
        for gi, g in enumerate(gts):           # right place, wrong class?
            if gi not in used:
                v = iou(xyxy(p), g["xyxy"])
                if v >= thr and v > best:
                    best, bi = v, gi
        out[pi] = ("wrong_class", bi, best) if bi is not None else ("fp", None, 0.0)
    wrong_gts = {o[1] for o in out if o[0] == "wrong_class"}
    missed = [gi for gi in range(len(gts)) if gi not in used]
    return out, missed, wrong_gts


def average_precision(scored, n_gt):
    """All-point interpolated AP (as in PASCAL VOC / COCO AP@0.5) from [(conf, is_tp), ...]."""
    if n_gt == 0:
        return None
    if not scored:
        return 0.0
    scored = sorted(scored, key=lambda s: -s[0])
    tp = np.cumsum([1 if t else 0 for _, t in scored])
    fp = np.cumsum([0 if t else 1 for _, t in scored])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------------------------------------------------------- run
def run_page(data, filename, page_no):
    img, _ = pipeline.load_document(data, filename, page_no)
    t0 = time.time()
    raw = pipeline.detect(img, pipeline.RESCUE_CONF)          # one YOLO pass, shared by both set-ups
    t_model = time.time() - t0
    model_only = [copy.deepcopy(d) for d in raw if d["conf"] >= pipeline.CONF]

    real_detect = pipeline.detect
    pipeline.detect = lambda *_a, **_k: copy.deepcopy(raw)
    try:
        t0 = time.time()
        full = pipeline.analyze(data, filename, page_no)["detections"]
        t_text = time.time() - t0
    finally:
        pipeline.detect = real_detect
    return img, {"model": model_only, "full": full}, t_model, t_model + t_text


def draw_overlay(img, preds, status, gts, missed, wrong_gts, names, path):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    lw = max(3, round(max(im.size) / 1200))
    for gi in missed:
        if gi in wrong_gts:
            continue
        d.rectangle(gts[gi]["xyxy"], outline=ORANGE, width=lw * 2)
    for p, (st, gi, _) in zip(preds, status):
        col = {"tp": GREEN, "wrong_class": PURPLE, "fp": RED}[st]
        d.rectangle(xyxy(p), outline=col, width=lw)
        if st == "wrong_class":
            x1, y1 = xyxy(p)[:2]
            d.text((x1, max(0, y1 - 14 * lw)), f'{names[p["cls_id"]]} (is {names[gts[gi]["cls_id"]]})', fill=PURPLE)
    k = 3000 / max(im.size)
    if k < 1:
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-list", action="store_true", help="write data/real_test/labels.txt for makesense.ai and stop")
    ap.add_argument("--labels", type=Path, default=LABELS, help="folder of YOLO .txt label files")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a box as found (default 0.5)")
    args = ap.parse_args()

    if args.labels_list:
        write_label_list()
        return
    files = sorted(args.labels.glob("*.txt")) if args.labels.exists() else []
    files = [f for f in files if f.name != "labels.txt"]
    if not files:
        sys.exit(f"No label files in {args.labels}. Finish Step 4 first: export YOLO labels from makesense.ai "
                 "and unzip the .txt files there.")
    if pipeline.get_model() is None:
        sys.exit(f"No model at {pipeline.MODEL_PATH}")

    rows = class_rows()
    names = {int(r["id"]): r["name"] for r in rows}
    known = {int(k) for k in pipeline.get_model().names}
    hidden = {int(r["id"]) for r in rows if r.get("show") is False} | {int(r["id"]) for r in rows if int(r["id"]) not in known}
    if len(known) < len(rows):
        print(f"Model knows {len(known)} classes: labels of classes {len(known)}-{len(rows) - 1} are ignored until it is retrained.")
    OUT.mkdir(parents=True, exist_ok=True)

    scored = {m: defaultdict(list) for m, _ in MODES}         # class -> [(conf, is_tp)]
    counts = {m: defaultdict(lambda: [0, 0, 0]) for m, _ in MODES}   # class -> [tp, fp, fn]
    located = {m: [0, 0] for m, _ in MODES}                   # [found in right place (any class), total labels]
    n_gt = defaultdict(int)
    mistakes, tags, pages, timing = [], [], [], {m: 0.0 for m, _ in MODES}
    ignored = 0

    for f in files:
        src = find_source(f.stem)
        if src is None:
            print(f"! {f.name}: no matching PDF page or image, skipped")
            continue
        data, filename, page_no = src
        print(f"{f.stem}: analysing {filename} page {page_no} ...", flush=True)
        img, preds_by_mode, t_model, t_full = run_page(data, filename, page_no)
        timing["model"] += t_model
        timing["full"] += t_full
        W, H = img.size
        gts_all = read_labels(f, W, H, len(rows))
        gts = [g for g in gts_all if g["cls_id"] not in hidden]
        ignored += len(gts_all) - len(gts)
        for g in gts:
            n_gt[g["cls_id"]] += 1
        pages.append((f.stem, len(gts)))

        for mode, label in MODES:
            preds = [p for p in preds_by_mode[mode] if p["cls_id"] not in hidden]
            status, missed, wrong_gts = match(preds, gts, args.iou)
            for p, (st, gi, v) in zip(preds, status):
                scored[mode][p["cls_id"]].append((p["conf"], st == "tp"))
                counts[mode][p["cls_id"]][0 if st == "tp" else 1] += 1
                if st != "tp":
                    mistakes.append([label, f.stem, "Wrong name" if st == "wrong_class" else "False detection",
                                     names[gts[gi]["cls_id"]] if gi is not None else "-", names[p["cls_id"]],
                                     p.get("name") or "", round(p["conf"], 2), round(v, 2), *[round(c) for c in xyxy(p)]])
                if mode == "full" and st == "tp" and p.get("cat") == "inst":
                    tags.append([f.stem, names[p["cls_id"]], p.get("tag") or "", p.get("name") or "",
                                 round(p["conf"], 2), *[round(c) for c in xyxy(p)]])
            for gi in missed:
                counts[mode][gts[gi]["cls_id"]][2] += 1
                if gi not in wrong_gts:
                    mistakes.append([label, f.stem, "Missed", names[gts[gi]["cls_id"]], "-", "", "", "",
                                     *[round(c) for c in gts[gi]["xyxy"]]])
            located[mode][0] += len(gts) - len([gi for gi in missed if gi not in wrong_gts])
            located[mode][1] += len(gts)
            draw_overlay(img, preds, status, gts, missed, wrong_gts, names, OUT / f"{f.stem}_{mode}.png")

    # ------------------------------------------------ summary
    summary = {}
    for mode, label in MODES:
        tp = sum(c[0] for c in counts[mode].values())
        fp = sum(c[1] for c in counts[mode].values())
        fn = sum(c[2] for c in counts[mode].values())
        p, r, f1 = prf(tp, fp, fn)
        aps = {c: average_precision(scored[mode][c], n_gt[c]) for c in n_gt}
        aps = {c: a for c, a in aps.items() if a is not None}
        summary[mode] = dict(label=label, tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1,
                             map50=sum(aps.values()) / len(aps) if aps else 0.0, aps=aps,
                             located=located[mode][0] / located[mode][1] if located[mode][1] else 0.0,
                             secs=timing[mode] / max(1, len(pages)))

    print(f"\n{len(pages)} page(s), {sum(n for _, n in pages)} labelled symbols"
          + (f" ({ignored} ignored: hidden or not in this model)" if ignored else "") + f", IoU >= {args.iou}\n")
    print(f'{"":17s}{"Precision":>10s}{"Recall":>9s}{"F1":>7s}{"mAP@0.5":>9s}{"Located":>9s}{"s/page":>8s}')
    for s in summary.values():
        print(f'{s["label"]:17s}{s["p"]:10.3f}{s["r"]:9.3f}{s["f1"]:7.3f}{s["map50"]:9.3f}{s["located"]:9.1%}{s["secs"]:8.1f}')
    print("\nLocated = symbol boxed in the right place, even if its name is wrong.")

    write_excel(summary, counts, n_gt, names, mistakes, tags, pages, args.iou, ignored)
    print(f"\nSaved {(OUT / 'evaluation.xlsx').relative_to(ROOT)} and page pictures in {OUT.relative_to(ROOT)}/")
    if tags:
        print(f"Open the 'Instrument tags' sheet and mark each of the {len(tags)} tags Yes/No to get tag accuracy.")


def write_excel(summary, counts, n_gt, names, mistakes, tags, pages, iou_thr, ignored):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    head_fill, head_font = PatternFill("solid", fgColor="1F4E9C"), Font(bold=True, color="FFFFFF")

    def header(ws, cols, widths):
        ws.append(cols)
        for i, w in enumerate(widths, start=1):
            ws.cell(row=1, column=i).fill = head_fill
            ws.cell(row=1, column=i).font = head_font
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A2"

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    header(ws, ["Set-up", "Precision", "Recall", "F1", "mAP@0.5", "Located (any name)",
                "Correct", "False detections", "Missed / wrong name", "Seconds per page"],
           [20, 11, 10, 9, 10, 18, 9, 16, 20, 16])
    for s in summary.values():
        ws.append([s["label"], s["p"], s["r"], s["f1"], s["map50"], s["located"], s["tp"], s["fp"], s["fn"], s["secs"]])
        for c in range(2, 6):
            ws.cell(row=ws.max_row, column=c).number_format = "0.000"
        ws.cell(row=ws.max_row, column=6).number_format = "0.0%"
        ws.cell(row=ws.max_row, column=10).number_format = "0.0"
    ws.append([])
    ws.append(["Pages", ", ".join(f"{p} ({n} labels)" for p, n in pages)])
    ws.append(["Match rule", f"Same class and IoU >= {iou_thr}"])
    if ignored:
        ws.append(["Ignored labels", f"{ignored} (classes hidden in the app, or not yet in this model)"])
    ws.append(["A) Model only", "YOLOv8 + tiling + merging, confidence >= " + str(pipeline.CONF)])
    ws.append(["B) Full system", "A + drawing text: tag/label reading, rescued weak boxes, unconfirmed boxes dropped"])
    ws.append(["Note", "mAP here uses only the boxes each set-up outputs, so it is lower than the validation mAP "
                       "from training (which ranks every box down to confidence 0.001). Compare A with B, not with training."])

    pc = wb.create_sheet("Per class")
    cols = ["Class", "Labels"]
    for _, label in MODES:
        short = label.split(")")[0] + ")"
        cols += [f"{short} Correct", f"{short} False", f"{short} Missed", f"{short} Precision", f"{short} Recall", f"{short} AP@0.5"]
    header(pc, cols, [30, 8] + [11, 9, 10, 12, 10, 11] * len(MODES))
    for c in sorted(set(n_gt) | {k for m, _ in MODES for k in counts[m]}, key=lambda c: -n_gt.get(c, 0)):
        row = [names[c], n_gt.get(c, 0)]
        for m, _ in MODES:
            tp, fp, fn = counts[m][c]
            p, r, _ = prf(tp, fp, fn)
            ap = summary[m]["aps"].get(c)
            row += [tp, fp, fn, round(p, 3), round(r, 3), round(ap, 3) if ap is not None else "n/a (no labels)"]
        pc.append(row)

    mk = wb.create_sheet("Mistakes")
    header(mk, ["Set-up", "Page", "Type", "Correct class", "Predicted class", "Name shown", "Confidence", "IoU",
                "x1", "y1", "x2", "y2"], [16, 22, 16, 26, 26, 26, 11, 7, 7, 7, 7, 7])
    for r in mistakes:
        mk.append(r)

    tg = wb.create_sheet("Instrument tags")
    header(tg, ["Page", "Symbol", "Tag read", "Name shown", "Confidence", "x1", "y1", "x2", "y2",
                "Tag correct? (Yes/No)"], [22, 26, 16, 32, 11, 7, 7, 7, 7, 20])
    dv = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    tg.add_data_validation(dv)
    for r in tags:
        tg.append(r + [""])
        dv.add(tg.cell(row=tg.max_row, column=10))
    if tags:
        last = tg.max_row
        tg.append([])
        tg.append(["Tag accuracy", "", "", "", "", "", "", "", "",
                   f'=IFERROR(COUNTIF(J2:J{last},"Yes")/COUNTIF(J2:J{last},"?*"),"fill in column J")'])
        tg.cell(row=tg.max_row, column=10).number_format = "0.0%"
        tg.cell(row=tg.max_row, column=1).font = Font(bold=True)

    wb.save(OUT / "evaluation.xlsx")


if __name__ == "__main__":
    main()
