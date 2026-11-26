"""P&ID analysis pipeline: load drawing -> tiled YOLO detection -> merge -> OCR tags -> ISA decode."""
import base64
import io
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "best.pt"))
NAMES_PATH = Path(os.getenv("CLASS_NAMES", ROOT / "models" / "class_names.json"))
TILE, OVERLAP = 640, 128          # must match tile_dataset.py
STRIDE = TILE - OVERLAP
CONF = float(os.getenv("CONF", 0.25))
RESCUE_CONF = 0.10               # weaker detections are kept only if a valid tag is read inside them
IOU = 0.5
CROSS_IOU = 0.6                   # same spot, different class -> keep the best one
BATCH = 16
PDF_DPI = 300
DEVICE = os.getenv("DEVICE", "")        # e.g. "cpu" on Hugging Face Spaces
DISPLAY_MAX = 2400                # longest side of the preview image sent to the browser

_model = None


# ---------------------------------------------------------------- model
def get_model():
    """Load YOLO once. Returns None if no trained weights exist yet."""
    global _model
    if _model is None and MODEL_PATH.exists():
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
    return _model


CATEGORY_ALIASES = {"valve": "valve", "valves": "valve", "instrument": "inst", "instruments": "inst",
                    "inst": "inst", "fitting": "fitting", "fittings": "fitting", "other": "fitting"}


def class_info(model) -> dict:
    """{class_id: {"name", "cat"}} from the model, overridden by models/class_names.json if present.
    The file is re-read on every request, so renaming takes effect without restarting the server."""
    info = {int(k): {"name": v, "cat": category(v), "show": True, "min_conf": 0.0, "needs_label": False}
            for k, v in model.names.items()}
    if NAMES_PATH.exists():
        try:
            rows = json.loads(NAMES_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            print(f"[class_names.json] ignored, could not read it: {e}")
            rows = []
        for row in rows:
            i = int(row.get("id", -1))
            if i not in info:
                continue
            if row.get("name"):
                info[i]["name"] = str(row["name"])
            cat = str(row.get("category", "")).strip().lower()
            info[i]["cat"] = CATEGORY_ALIASES.get(cat, category(info[i]["name"]))
            info[i]["show"] = bool(row.get("show", True))       # "show": false hides this class everywhere
            info[i]["needs_label"] = bool(row.get("needs_label", False))  # keep only if a label on the drawing confirms it
            try:                                                 # "min_conf": 0.6 drops weaker detections of this class
                info[i]["min_conf"] = float(row.get("min_conf", 0.0))
            except (TypeError, ValueError):
                pass
    return info


_ocr = None


def get_ocr():
    """OCR engine, first one that works: RapidOCR (pip install rapidocr-onnxruntime),
    EasyOCR (pip install easyocr), Tesseract (brew install tesseract), otherwise none."""
    global _ocr
    if _ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = ("rapidocr", RapidOCR())
        except Exception:
            try:
                import easyocr
                _ocr = ("easyocr", easyocr.Reader(["en"], gpu=False, verbose=False))
            except Exception:
                try:
                    import pytesseract
                    pytesseract.get_tesseract_version()
                    _ocr = ("tesseract", None)
                except Exception:
                    _ocr = ("none", None)
    return _ocr


def ocr_available() -> bool:
    return get_ocr()[0] != "none"


# ---------------------------------------------------------------- input
MAX_PAGES = int(os.getenv("MAX_PAGES", 30))


def _open_pdf(data: bytes):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    return fitz.open(stream=data, filetype="pdf")


def page_count(data: bytes, filename: str) -> int:
    """Number of pages that will be analysed (images count as 1)."""
    if filename.lower().endswith(".pdf"):
        return max(1, min(len(_open_pdf(data)), MAX_PAGES))
    return 1


def load_document(data: bytes, filename: str, page_no: int = 1):
    """One PDF page (300 DPI) or an image -> (RGB PIL image, words). page_no starts at 1.
    For a PDF with a text layer, words = [(x0, y0, x1, y1, text), ...] in image pixels,
    so tags can be read exactly without OCR. For images and scanned PDFs, words = []."""
    if filename.lower().endswith(".pdf"):
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        page = doc[page_no - 1]
        pix = page.get_pixmap(dpi=PDF_DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        k = PDF_DPI / 72.0
        words = []                       # one entry per text line: (x0, y0, x1, y1, text, dir_x, dir_y)
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                if text:
                    x0, y0, x1, y1 = line["bbox"]
                    dx, dy = line["dir"]
                    words.append((x0 * k, y0 * k, x1 * k, y1 * k, text, dx, dy))
        return img, words
    return normalize_image(Image.open(io.BytesIO(data))), []


IMAGE_MIN_SIDE = int(os.getenv("IMAGE_MIN_SIDE", 4960))   # A3 at 300 DPI, the scale the model was trained on


def normalize_image(img: Image.Image) -> Image.Image:
    """Make PNG/JPG uploads look like a 300 DPI render:
    - transparent backgrounds become white (otherwise they turn black and hide the black lines)
    - small images are upscaled so symbols are roughly the size the model learned (~60-130 px)."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    img = img.convert("RGB")
    longest = max(img.size)
    if longest < IMAGE_MIN_SIDE:
        k = IMAGE_MIN_SIDE / longest
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    return img


def load_image(data: bytes, filename: str) -> Image.Image:
    return load_document(data, filename)[0]


def preview_data_url(img: Image.Image) -> str:
    """Downscaled JPEG for the browser. Box coordinates stay in full-size pixels."""
    prev = img.copy()
    prev.thumbnail((DISPLAY_MAX, DISPLAY_MAX))
    buf = io.BytesIO()
    prev.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- detection
def _starts(length: int):
    s = list(range(0, max(length - TILE, 0) + 1, STRIDE))
    if s[-1] + TILE < length:
        s.append(length - TILE)
    return s


def _tile(arr: np.ndarray, x: int, y: int) -> np.ndarray:
    crop = arr[y:y + TILE, x:x + TILE]
    if crop.shape[0] == TILE and crop.shape[1] == TILE:
        return crop
    out = np.full((TILE, TILE, 3), 255, dtype=arr.dtype)   # pad small drawings with white
    out[:crop.shape[0], :crop.shape[1]] = crop
    return out


def _nms(boxes, scores, classes, iou_thr):
    """Per-class non-maximum suppression: keep the best box where tiles overlap."""
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        idx = idx[np.argsort(-scores[idx])]
        while idx.size:
            i = idx[0]
            keep.append(int(i))
            if idx.size == 1:
                break
            rest = idx[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / np.maximum(area_i + area_r - inter, 1e-6)
            idx = rest[iou < iou_thr]
    return keep


def _drop_fragments(boxes, scores, classes, keep):
    """Remove partial boxes (cut at a tile edge) that sit mostly inside a more confident box."""
    keep = sorted(keep, key=lambda i: -scores[i])
    final = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        inside = False
        for j in final:
            a1, b1, a2, b2 = boxes[j]
            iw = max(0, min(x2, a2) - max(x1, a1))
            ih = max(0, min(y2, b2) - max(y1, b1))
            if iw * ih / area > 0.7:
                inside = True
                break
        if not inside:
            final.append(i)
    return final


def detect(img: Image.Image, conf: float = CONF):
    """Slide a 640 px window over the whole drawing, run YOLO on each tile, merge results."""
    model = get_model()
    if model is None:
        return []
    W, H = img.size
    arr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])   # RGB -> BGR for ultralytics
    coords = [(x, y) for y in _starts(H) for x in _starts(W)]

    all_boxes, all_scores, all_cls = [], [], []
    for i in range(0, len(coords), BATCH):
        batch = coords[i:i + BATCH]
        results = model.predict([_tile(arr, x, y) for x, y in batch],
                                imgsz=TILE, conf=conf, verbose=False, **({"device": DEVICE} if DEVICE else {}))
        for (x, y), r in zip(batch, results):
            if len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            xyxy[:, [0, 2]] += x
            xyxy[:, [1, 3]] += y
            all_boxes.append(xyxy)
            all_scores.append(r.boxes.conf.cpu().numpy())
            all_cls.append(r.boxes.cls.cpu().numpy().astype(int))
    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_cls)
    keep = _nms(boxes, scores, classes, IOU)
    # One symbol must not get two labels (e.g. "Control-room" 0.77 and "Field-mounted" 0.50 on the same bubble):
    # across classes, overlapping boxes keep only the most confident one.
    idx = np.array(keep, dtype=int)
    if idx.size:
        cross = _nms(boxes[idx], scores[idx], np.zeros(idx.size, dtype=int), CROSS_IOU)
        keep = [int(idx[i]) for i in cross]
    keep = _drop_fragments(boxes, scores, classes, keep)

    info = class_info(model)
    keep = [i for i in keep
            if info[int(classes[i])]["show"] and scores[i] >= info[int(classes[i])]["min_conf"]]
    dets = []
    for n, i in enumerate(sorted(keep, key=lambda i: (boxes[i][1], boxes[i][0]))):
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        c = info[int(classes[i])]
        dets.append({
            "id": f"d{n + 1}",
            "cls_id": int(classes[i]),
            "needs_label": c["needs_label"],
            "cls": c["name"],
            "cat": c["cat"],
            "conf": round(float(scores[i]), 3),
            "b": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
        })
    return dets


def category(name: str) -> str:
    n = name.lower()
    if "valve" in n:
        return "valve"
    if any(k in n for k in ("instrument", "display", "box")):
        return "inst"
    return "fitting"


def relabel(d: dict, cls_id: int) -> dict:
    """User correction: give a detection another symbol type, renaming it the way analyze() would."""
    info = class_info(get_model()) if get_model() is not None else {}
    if cls_id not in info:
        raise ValueError(f"Unknown symbol type {cls_id}")
    if "orig_cls" not in d:                                    # remember what the model said, once
        d["orig_cls"], d["orig_cls_id"], d["orig_name"] = d["cls"], d.get("cls_id"), d.get("name") or d["cls"]
    new = info[cls_id]
    size = re.search(r"\((\d+(?:\.\d+)?) NB\)", d.get("orig_name") or "")
    d["cls"], d["cls_id"], d["cat"] = new["name"], cls_id, new["cat"]
    tag = d.get("tag") or ""
    if new["cat"] == "inst":
        d["tag_parts"] = parse_tag(tag)
        d["desc"] = decode_isa(tag) if tag else ""
        d["name"] = d["desc"] or new["name"]
    else:
        d["tag_parts"], d["desc"] = None, ""
        d["name"] = new["name"] + (f" ({size.group(1)} NB)" if size else "")
    if cls_id == d["orig_cls_id"]:                             # changed back to the model's answer
        d["name"] = d.pop("orig_name")
        d.pop("orig_cls")
        d.pop("orig_cls_id")
        d.pop("corrected", None)
        if d["cat"] == "inst":
            d["desc"] = decode_isa(tag) if tag else ""
    else:
        d["corrected"] = True
    return d


def class_list() -> list:
    """Symbol types the user can choose from (hidden classes left out), in id order."""
    model = get_model()
    if model is None:
        return []
    return [{"id": i, "name": c["name"], "cat": c["cat"]} for i, c in sorted(class_info(model).items()) if c["show"]]


# ---------------------------------------------------------------- OCR + ISA
# Tag = [area-]LETTERS-NUMBER[/NUMBER...], e.g. "PIC-204", "9-TI-1011/1012/1013", "PI-102A"
TAG_RE = re.compile(r"^(?:(\d{1,4})-)?([A-Z]{1,5})-(\d{1,6}[A-Z]?(?:/\d{1,6}[A-Z]?)*)$")


def normalize_tag(text: str) -> str:
    """'9-TI 1011 / 1012/1013' -> '9-TI-1011/1012/1013'; 'PIC204' -> 'PIC-204'."""
    t = re.sub(r"[^A-Z0-9\-/ ]", "", (text or "").upper())
    t = re.sub(r"\s*/\s*", "/", t).strip()
    t = re.sub(r"(?<=[A-Z])[\s-]*(?=\d)", "-", t)        # letters followed by digits -> 'TI-1011'
    t = re.sub(r"(?<=\d)[\s-]+(?=[A-Z])", "-", t)        # area digits followed by letters -> '9-TI'
    t = re.sub(r"\s+", "", t)
    return t


def parse_tag(tag: str):
    """'9-TI-1011/1012/1013' -> {'area': '9', 'letters': 'TI', 'numbers': ['1011', '1012', '1013']}."""
    m = TAG_RE.match(tag or "")
    if not m:
        return None
    return {"area": m.group(1) or "", "letters": m.group(2), "numbers": m.group(3).split("/")}


def number_key(n: str):
    """Sort '1011' < '1012' < '1012A' numerically."""
    m = re.match(r"(\d+)([A-Z]?)", n or "")
    return (int(m.group(1)), m.group(2)) if m else (10 ** 9, n or "")


OCR_DPI = 600          # bubbles are re-rendered from the PDF at this resolution for OCR
OCR_INSET = 0.06       # blank out the outer 6% of the box (the bubble outline)


def _mask_outline(im: Image.Image) -> Image.Image:
    """Keep only the inside of the bubble so the circle outline isn't read as characters."""
    from PIL import ImageDraw
    w, h = im.size
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).ellipse([w * OCR_INSET, h * OCR_INSET, w * (1 - OCR_INSET), h * (1 - OCR_INSET)], fill=255)
    out = Image.new("RGB", im.size, "white")
    out.paste(im.convert("RGB"), mask=mask)
    return out


def bubble_image(box, img: Image.Image, page=None, mask: bool = True) -> Image.Image:
    """High-resolution picture of one bubble: re-rendered from the PDF if we have it, else upscaled from the image."""
    x, y, w, h = box
    if page is not None:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        k = PDF_DPI / 72.0
        pix = page.get_pixmap(dpi=OCR_DPI, clip=fitz.Rect(x / k, y / k, (x + w) / k, (y + h) / k))
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        im = img.crop((x, y, x + w, y + h))
        im = im.resize((max(1, im.width * 2), max(1, im.height * 2)), Image.LANCZOS)
    return _mask_outline(im) if mask else im


def _circle_clean(im: Image.Image, shrink: float = 0.86):
    """Find the bubble circle in a loose crop, keep only its inside (outline and leader lines removed)."""
    try:
        import cv2
    except ImportError:
        return None
    from PIL import ImageDraw
    g = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    side = min(g.shape)
    c = cv2.HoughCircles(cv2.medianBlur(g, 5), cv2.HOUGH_GRADIENT, dp=1, minDist=side, param1=120, param2=30,
                         minRadius=int(side * 0.25), maxRadius=int(side * 0.5))
    if c is None:
        return None
    cx, cy, r = (float(v) for v in c[0][0])
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).ellipse([cx - r * shrink, cy - r * shrink, cx + r * shrink, cy + r * shrink], fill=255)
    out = Image.new("RGB", im.size, "white")
    out.paste(im.convert("RGB"), mask=m)
    return out.crop((int(cx - r), int(cy - r), int(cx + r), int(cy + r)))


def _prep(im: Image.Image, border: float = 0.15) -> Image.Image:
    """Black-and-white with a white margin: OCR finds text that touches the edge more easily."""
    from PIL import ImageOps
    g = im.convert("L").point(lambda v: 255 if v > 150 else 0)
    b = int(max(g.size) * border)
    return ImageOps.expand(g, border=b, fill=255).convert("RGB")


def _rec_line(im: Image.Image) -> str:
    """Recognise one line of text (no text detection step)."""
    kind, engine = get_ocr()
    if kind == "rapidocr":
        res, _ = engine(np.asarray(im), use_det=False, use_cls=False, use_rec=True)
        if not res:
            return ""
        r = res[0]
        return str(r[0] if isinstance(r[0], str) else r[1])
    return " ".join(t for t, _ in _ocr_lines(im))


def _ocr_lines(im: Image.Image):
    """[(text, score), ...] top to bottom."""
    kind, engine = get_ocr()
    if kind == "rapidocr":
        res, _ = engine(np.asarray(im))
        res = sorted(res or [], key=lambda r: (r[0][0][1] + r[0][2][1]) / 2)
        return [(str(t), float(sc)) for _, t, sc in res]
    if kind == "easyocr":
        res = engine.readtext(np.asarray(im))
        res = sorted(res, key=lambda r: (r[0][0][1] + r[0][2][1]) / 2)
        return [(str(t), float(sc)) for _, t, sc in res]
    if kind == "tesseract":
        import pytesseract
        text = pytesseract.image_to_string(im, config="--psm 6")
        return [(t, 0.5) for t in text.splitlines() if t.strip()]
    return []


def _fix_ocr(text: str) -> str:
    """Common OCR slips in tag letters: lowercase l or | for I ('9-Pl' -> '9-PI')."""
    return text.replace("l", "I").replace("|", "I")


def read_tag(img: Image.Image, box, page=None, angles=(0, 90, 270)) -> str:
    return read_tag_detail(img, box, page, angles)["tag"]


def read_tag_detail(img: Image.Image, box, page=None, angles=(0, 90, 270)) -> dict:
    """OCR one bubble. Returns {"tag": full tag or "", "number": loop number if only that was readable}."""
    if not ocr_available():
        return {"tag": "", "number": ""}
    tag = _read_tag_simple(img, box, page, angles)
    if tag:
        return {"tag": tag, "number": ""}
    # Harder labels (bold, sideways, off-centre box): find the circle, clean it, read line by line
    x, y, w, h = box
    loose = [x - w * 0.3, y - h * 0.3, w * 1.6, h * 1.6]
    cleaned = _circle_clean(bubble_image(loose, img, page, mask=False))
    numbers = []
    if cleaned is not None:
        for ang in angles:
            r = cleaned.rotate(ang, expand=True, fillcolor="white") if ang else cleaned
            try:
                whole = [t for t, _ in _ocr_lines(_prep(r))]
                rw, rh = r.size
                split = [_rec_line(_prep(r.crop((0, 0, rw, rh // 2)))), _rec_line(_prep(r.crop((0, rh // 2, rw, rh))))]
            except Exception:
                continue
            for lines in (whole, split):
                cand = _combine(_fix_ocr(t) for t in lines if t)
                p = parse_tag(cand)
                if p and len(p["letters"]) >= 2:
                    return {"tag": cand, "number": ""}
                numbers += re.findall(r"\d{3,6}", " ".join(lines))
    from collections import Counter
    number = Counter(numbers).most_common(1)[0][0] if numbers else ""
    return {"tag": "", "number": number}


def _read_tag_simple(img: Image.Image, box, page=None, angles=(0, 90, 270)) -> str:
    """OCR one bubble, trying each text direction; keep the reading that forms a valid tag."""
    x, y, w, h = box
    pad = 0.08                                   # a slightly larger crop catches letters cut by a tight box
    crops = [box, [x - w * pad, y - h * pad, w * (1 + 2 * pad), h * (1 + 2 * pad)]]
    best, best_score = "", 0.0
    for crop in crops:
        base = bubble_image(crop, img, page)
        for ang in angles:
            im = base.rotate(ang, expand=True, fillcolor="white") if ang else base
            try:
                lines = _ocr_lines(im)
            except Exception:
                continue
            if not lines:
                continue
            tag = _combine(_fix_ocr(t) for t, _ in lines)
            parts = parse_tag(tag)
            if not parts:
                continue
            score = sum(sc for _, sc in lines) / len(lines)
            if len(parts["letters"]) >= 2:       # 'TI' is far more likely than a lone 'T'
                score += 0.2
            if score > best_score:
                best, best_score = tag, score
            if score >= 0.85:
                return best
    return best


def text_angles(words):
    """Try the drawing's main text direction first (rotated drawings write tags sideways)."""
    from collections import Counter
    dirs = Counter((round(w[5]), round(w[6])) for w in words if len(w) > 6)
    main = dirs.most_common(1)[0][0] if dirs else (1, 0)
    return {(0, 1): (90, 0, 270), (0, -1): (270, 0, 90), (-1, 0): (180, 90, 270)}.get(main, (0, 90, 270))


def _combine(lines) -> str:
    """Text lines inside a bubble -> one tag.
    ['PIC', '204'] -> 'PIC-204'; ['9-TI', '1011/1012/1013'] -> '9-TI-1011/1012/1013'; ['FT-101'] -> 'FT-101'."""
    parts = [normalize_tag(line) for line in lines]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) >= 2 and re.fullmatch(r"(\d{1,4}-)?[A-Z]{1,5}", parts[0]):
        return normalize_tag(f"{parts[0]} {parts[1]}")
    return parts[0]


def tag_from_words(words, box) -> str:
    """Read a tag from the PDF's own text layer: words whose centre lies inside the symbol box."""
    x, y, w, h = box
    x0, y0, x1, y1 = x + w * 0.08, y + h * 0.08, x + w * 0.92, y + h * 0.92
    inside = [wd for wd in words if x0 <= (wd[0] + wd[2]) / 2 <= x1 and y0 <= (wd[1] + wd[3]) / 2 <= y1]
    if not inside:
        return ""
    # Reading order follows the text direction, so rotated drawings work too:
    # 'across' = position along the text, 'down' = which line it is (first line, second line, ...)
    def frame(wd):
        dx, dy = (wd[5], wd[6]) if len(wd) > 6 else (1.0, 0.0)
        cx, cy = (wd[0] + wd[2]) / 2, (wd[1] + wd[3]) / 2
        height = abs(dx) * (wd[3] - wd[1]) + abs(dy) * (wd[2] - wd[0])   # text height in its own frame
        return cx * dx + cy * dy, cx * -dy + cy * dx, max(height, 1.0)
    items = sorted((frame(wd) + (wd[4],) for wd in inside), key=lambda t: (t[1], t[0]))
    lines, last_down = [], None
    for across, down, height, text in items:   # group into text lines
        if last_down is None or abs(down - last_down) > height * 0.6:
            lines.append([(across, text)])
        else:
            lines[-1].append((across, text))
        last_down = down
    return _combine(" ".join(t for _, t in sorted(line)) for line in lines)


ISA_FIRST = {
    "A": "Analysis", "B": "Burner/combustion", "C": "Conductivity", "D": "Density",
    "E": "Voltage", "F": "Flow", "G": "Gauging", "H": "Hand", "I": "Current",
    "J": "Power", "K": "Time", "L": "Level", "M": "Moisture", "P": "Pressure",
    "Q": "Quantity", "R": "Radiation", "S": "Speed", "T": "Temperature",
    "U": "Multivariable", "V": "Vibration", "W": "Weight", "X": "Unclassified",
    "Y": "Event/state", "Z": "Position",
}
ISA_MODIFIER = {"D": "differential", "F": "ratio", "Q": "totalizing", "S": "safety", "K": "rate of change"}
ISA_SUCCEEDING = {
    "A": "alarm", "C": "controller", "E": "element", "G": "glass/gauge", "H": "high",
    "I": "indicating", "K": "control station", "L": "light", "O": "orifice", "P": "test point",
    "R": "recorder", "S": "switch", "T": "transmitter", "V": "valve", "W": "well",
    "Y": "relay/converter", "Z": "actuator",
}


def decode_isa(tag: str) -> str:
    """'PIC-204' -> 'Pressure indicating controller'. Returns '' if the tag doesn't look like ISA."""
    parsed = parse_tag(tag)
    if not parsed:
        return ""
    letters = parsed["letters"]
    first = ISA_FIRST.get(letters[0])
    if not first:
        return ""
    rest = letters[1:]
    words = [first]
    if rest and rest[0] in ISA_MODIFIER and len(rest) > 1:
        words.append(ISA_MODIFIER[rest[0]])
        rest = rest[1:]
    if rest.endswith("CV"):
        return " ".join(words + [ISA_SUCCEEDING.get(c, c) for c in rest[:-2]] + ["control valve"]).capitalize()
    words += [ISA_SUCCEEDING.get(c, c) for c in rest]
    if rest.endswith("I"):            # "I" as the last letter = indicator (PI, TI, LI)
        words[-1] = "indicator"
    return " ".join(words).capitalize()


# ---------------------------------------------------------------- full run
# Valve type codes written next to valves on many P&IDs, e.g. "15BV", "150BF", "200NRV", "9-SRV-7101"
VALVE_CODES = {
    "BV": "Ball valve", "BF": "Butterfly valve", "BFV": "Butterfly valve", "GV": "Gate valve",
    "GLV": "Globe valve", "NRV": "Non-return valve", "CHV": "Check valve", "PLV": "Plug valve",
    "NV": "Needle valve", "DV": "Diaphragm valve", "SRV": "Safety relief valve",
    "PSV": "Pressure safety valve", "PRV": "Pressure relief valve", "TCV": "Temperature control valve",
    "PCV": "Pressure control valve", "FCV": "Flow control valve", "LCV": "Level control valve",
}
_CODE_RE = re.compile(r"^(\d{1,4})?\s*(" + "|".join(sorted(VALVE_CODES, key=len, reverse=True)) + r")$")
ROUND_HINTS = ("instrument", "display", "box", "orifice", "sight glass")   # symbols that may really be a bubble


def valve_code(text: str):
    """'15BV' -> ('15', 'BV'); '9-SRV-7101' -> ('', 'SRV'); otherwise None."""
    t = normalize_tag(text)
    m = _CODE_RE.match(t.replace("-", ""))
    if m:
        return m.group(1) or "", m.group(2)
    p = parse_tag(t)
    if p and p["letters"] in VALVE_CODES:
        return "", p["letters"]
    return None


def nearby_valve_label(words, box, reach=2.5):
    """Closest valve code written next to a symbol (within `reach` symbol sizes)."""
    x, y, w, h = box
    cx, cy, size = x + w / 2, y + h / 2, max(w, h)
    best = None
    for wd in words:
        code = valve_code(wd[4])
        if not code:
            continue
        tx, ty = (wd[0] + wd[2]) / 2, (wd[1] + wd[3]) / 2
        dist = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
        if dist <= reach * size and (best is None or dist < best[0]):
            best = (dist, wd[4].strip(), code)
    return best


def _roundish(d) -> bool:
    x, y, w, h = d["b"]
    aspect = w / max(h, 1e-6)
    return 0.7 <= aspect <= 1.4 and (d["cat"] == "inst" or any(k in d["cls"].lower() for k in ROUND_HINTS))


def analyze(data: bytes, filename: str, page_no: int = 1) -> dict:
    """Analyse one page of a drawing (page_no starts at 1)."""
    img, words = load_document(data, filename, page_no)
    W, H = img.size
    candidates = detect(img, RESCUE_CONF)
    use_ocr = ocr_available()
    page = None
    if filename.lower().endswith(".pdf") and use_ocr:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        page = fitz.open(stream=data, filetype="pdf")[page_no - 1]
    angles = text_angles(words)

    dets = []
    for d in candidates:
        weak = d["conf"] < CONF
        tag, name = "", ""

        # 1) instrument bubbles: read the tag inside (PDF text first, OCR for drawn/raster text)
        if _roundish(d):
            number = ""
            if words:
                tag = tag_from_words(words, d["b"])
            if not tag and use_ocr:
                got = read_tag_detail(img, d["b"], page, angles)
                tag, number = got["tag"], got["number"]
            p = parse_tag(tag)
            if tag and decode_isa(tag) and not (p and p["letters"] in VALVE_CODES):
                if d["cat"] != "inst":                 # e.g. a bubble the model called "Sight glass"
                    d["cls"], d["cat"] = "Instrument bubble", "inst"
                name = decode_isa(tag)
            elif number:                               # letters unreadable, loop number readable
                if d["cat"] != "inst":
                    d["cls"], d["cat"] = "Instrument bubble", "inst"
                tag, name = number, "Instrument (tag unclear)"
                d["conf"] = min(d["conf"], 0.69)       # flag for review
            elif d["cat"] != "inst":
                tag = ""

        d["_tag"], d["_name"], d["_weak"] = tag, name, weak
        dets.append(d)

    # 2) valves: give each valve-type label on the drawing (15BV, 150BF, 200NRV, 9-SRV-7101)
    #    to the single closest symbol, so a label is never shared by two symbols
    if words:
        pairs = []
        for i, d in enumerate(dets):
            if d["_name"] or d["cat"] not in ("valve", "fitting"):
                continue
            x, y, w, h = d["b"]
            cx, cy, size = x + w / 2, y + h / 2, max(w, h)
            for j, wd in enumerate(words):
                code = valve_code(wd[4])
                if not code:
                    continue
                dist = (((wd[0] + wd[2]) / 2 - cx) ** 2 + ((wd[1] + wd[3]) / 2 - cy) ** 2) ** 0.5
                if dist <= 2.5 * size:
                    pairs.append((dist, i, j, code))
        used_d, used_w = set(), set()
        for dist, i, j, (size_nb, code) in sorted(pairs):
            if i in used_d or j in used_w:
                continue
            used_d.add(i)
            used_w.add(j)
            d = dets[i]
            d["_tag"], d["cat"] = normalize_tag(words[j][4]), "valve"
            d["_name"] = VALVE_CODES[code] + (f" ({size_nb} NB)" if size_nb else "")

    final = []
    for d in dets:
        tag, name, weak = d.pop("_tag"), d.pop("_name"), d.pop("_weak")
        needs_label = d.pop("needs_label", False)
        if (weak or (needs_label and words)) and not name:   # unconfirmed by the drawing's text -> drop
            continue
        is_inst = d["cat"] == "inst"
        d["tag"] = tag
        d["tag_parts"] = parse_tag(tag) if is_inst else None
        d["desc"] = decode_isa(tag) if is_inst else ""
        d["name"] = name or d["desc"] or d["cls"]
        final.append(d)
    dets = final

    for n, d in enumerate(dets, start=1):
        d["id"] = f"d{n}"
    for d in dets:
        d["page"] = page_no
    return {
        "filename": filename,
        "page": page_no,
        "width": W,
        "height": H,
        "image": preview_data_url(img),
        "model": MODEL_PATH.name if get_model() is not None else None,
        "ocr": use_ocr or bool(words),
        "text_source": "pdf" if words else ("ocr" if use_ocr else None),
        "detections": dets,
    }
