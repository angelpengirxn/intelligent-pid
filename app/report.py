"""Excel report and chatbot answers built from one analysis result."""
import io
import json
import os
from collections import Counter

from .pipeline import number_key

CAT_NAMES = {"valve": "Valves", "fitting": "Fittings", "inst": "Instruments"}
REVIEW_BELOW = 0.70


def _name(d: dict) -> str:
    """Decoded ISA meaning when the tag was read, otherwise the symbol type."""
    return d.get("name") or d.get("desc") or d["cls"]


CAT_ORDER = {"valve": 0, "inst": 1, "fitting": 2}


def sort_key(d: dict):
    """Tagged items first, in tag-number order (area, loop number, letters); untagged items after."""
    p = d.get("tag_parts")
    if p:
        return (0, int(p["area"] or 0), number_key(p["numbers"][0]), p["letters"], d.get("page", 1))
    return (1, d.get("page", 1), CAT_ORDER.get(d["cat"], 3), d["b"][1], d["b"][0])


def needs_review(d: dict) -> bool:
    return d["conf"] < REVIEW_BELOW and not d.get("corrected")


def checked(d: dict) -> str:
    return f'Corrected (model said {d["orig_cls"]})' if d.get("corrected") else ""


def full_tag(p: dict, number: str) -> str:
    return f'{p["area"] + "-" if p["area"] else ""}{p["letters"]}-{number}'


def build_excel(job: dict) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    head_fill = PatternFill("solid", fgColor="1F4E9C")
    head_font = Font(bold=True, color="FFFFFF")
    warn_font = Font(bold=True, color="B23B2A")

    wb = Workbook()
    ws = wb.active
    ws.title = "Symbols"
    headers = ["No.", "Page", "Tag", "Name", "Symbol type", "Category", "Confidence",
               "Needs review", "Checked by user", "x (px)", "y (px)", "Width (px)", "Height (px)"]
    ws.append(headers)
    for c in ws[1]:
        c.fill, c.font = head_fill, head_font
        c.alignment = Alignment(horizontal="center")

    for n, d in enumerate(sorted(job["detections"], key=sort_key), start=1):
        x, y, w, h = d["b"]
        review = needs_review(d)
        ws.append([n, d.get("page", 1), d.get("tag") or "", _name(d), d["cls"], CAT_NAMES.get(d["cat"], d["cat"]),
                   d["conf"], "Yes" if review else "No", checked(d),
                   round(x), round(y), round(w), round(h)])
        ws.cell(row=n + 1, column=7).number_format = "0.00"
        if review:
            ws.cell(row=n + 1, column=8).font = warn_font

    widths = [6, 7, 14, 32, 28, 14, 12, 13, 34, 9, 9, 11, 11]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    # Instrument list: one row per instrument number, e.g. 9-TI-1011/1012/1013 -> three rows
    il = wb.create_sheet("Instrument list")
    il.append(["Tag", "Page", "Name", "Symbol type", "Shared bubble", "Confidence", "Needs review"])
    for c in il[1]:
        c.fill, c.font = head_fill, head_font
        c.alignment = Alignment(horizontal="center")
    rows = []
    for d in job["detections"]:
        p = d.get("tag_parts")
        if not p:
            continue
        for num in p["numbers"]:
            rows.append(((int(p["area"] or 0), number_key(num), p["letters"], d.get("page", 1)),
                         [full_tag(p, num), d.get("page", 1), _name(d), d["cls"],
                          d["tag"] if len(p["numbers"]) > 1 else "",
                          d["conf"], "Yes" if needs_review(d) else "No"]))
    for _, row in sorted(rows, key=lambda r: r[0]):
        il.append(row)
        il.cell(row=il.max_row, column=6).number_format = "0.00"
        if row[6] == "Yes":
            il.cell(row=il.max_row, column=7).font = warn_font
    if not rows:
        il.append(["No instrument tags were read on this drawing."])
    for i, w in enumerate([22, 7, 34, 28, 26, 12, 13], start=1):
        il.column_dimensions[get_column_letter(i)].width = w
    il.freeze_panes = "A2"

    # Summary sheet
    s = wb.create_sheet("Summary")
    s.append(["Drawing", job["filename"]])
    done, total = job.get("pages_done", 1), job.get("page_count", 1)
    s.append(["Pages analysed", f"{done} of {total}" + ("" if done == total else " (the other pages were still being analysed; export again when they finish)")])
    s.append(["Model", job.get("model") or "none"])
    s.append(["Total symbols", len(job["detections"])])
    s.append(["Needs review (< 0.70)", sum(needs_review(d) for d in job["detections"])])
    removed = job.get("removed", [])
    s.append(["Corrected by user", sum(bool(d.get("corrected")) for d in job["detections"])])
    s.append(["Removed as wrong detections", len(removed)])
    s.append([])
    s.append(["Name", "Count"])
    for c in s[s.max_row]:
        c.fill, c.font = head_fill, head_font
    for name, count in Counter(_name(d) for d in job["detections"]).most_common():
        s.append([name, count])
    s.column_dimensions["A"].width = 32
    s.column_dimensions["B"].width = 26
    for row in s.iter_rows(min_row=1, max_row=7):
        row[0].font = Font(bold=True)

    # Removed: what the model detected that the user marked as not a symbol (useful for retraining)
    if removed:
        rm = wb.create_sheet("Removed")
        rm.append(["Page", "Model said", "Confidence", "x (px)", "y (px)", "Width (px)", "Height (px)"])
        for c in rm[1]:
            c.fill, c.font = head_fill, head_font
        for d in sorted(removed, key=lambda d: (d.get("page", 1), d["b"][1], d["b"][0])):
            x, y, w, h = d["b"]
            rm.append([d.get("page", 1), d.get("orig_cls") or d["cls"], d["conf"], round(x), round(y), round(w), round(h)])
            rm.cell(row=rm.max_row, column=3).number_format = "0.00"
        for i, w in enumerate([7, 30, 12, 9, 9, 11, 11], start=1):
            rm.column_dimensions[get_column_letter(i)].width = w

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- chatbot
def _rule_based(job: dict, q: str) -> str:
    dets = job["detections"]
    ql = q.lower()
    if not dets:
        return "No symbols were detected on this drawing, so there is nothing to answer from yet."
    for key, cat in (("valve", "valve"), ("instrument", "inst"), ("fitting", "fitting")):
        if key in ql:
            items = [d for d in dets if d["cat"] == cat]
            names = Counter(_name(d) for d in items)
            detail = ", ".join(f"{n} × {c}" for c, n in names.most_common())
            return f"{len(items)} {CAT_NAMES[cat].lower()} detected" + (f": {detail}." if detail else ".")
    if "tag" in ql:
        tags = [d["tag"] for d in sorted(dets, key=sort_key) if d.get("tag")]
        return ("Tags found (in number order): " + ", ".join(tags) + ".") if tags else "No tags were read. OCR may be off or the text was unclear."
    if any(k in ql for k in ("review", "check", "low", "unsure")):
        low = [d for d in dets if needs_review(d)]
        if not low:
            return "All detections are at or above 0.70 confidence."
        return f"{len(low)} detection(s) below 0.70: " + "; ".join(
            f'{_name(d)} {d.get("tag") or ""} ({d["conf"]:.2f})'.replace("  ", " ") for d in low) + "."
    counts = Counter(CAT_NAMES[d["cat"]] for d in dets)
    return f"{len(dets)} symbols detected: " + ", ".join(f"{n} {c.lower()}" for c, n in counts.items()) + \
        ". Ask about valves, instruments, tags or what needs review."


GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_ITEMS = 400          # symbols sent to GPT per drawing (keeps the request small)
MAX_HISTORY = 10         # previous chat messages sent back for follow-up questions


def gpt_available() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def drawing_context(job: dict) -> str:
    """Compact, structured description of one analysed drawing for the language model."""
    dets = sorted(job["detections"], key=sort_key)
    counts = Counter(CAT_NAMES.get(d["cat"], d["cat"]) for d in dets)
    by_name = Counter(_name(d) for d in dets)
    lines = [
        f"Drawing file: {job['filename']} ({job.get('pages_done', 1)} of {job.get('page_count', 1)} page(s) analysed)",
        f"Total symbols detected: {len(dets)} (" + ", ".join(f"{n} {c.lower()}" for c, n in counts.items()) + ")",
        "Counts by name: " + "; ".join(f"{name}: {n}" for name, n in by_name.most_common()),
        f"Checked by the user: {sum(bool(d.get('corrected')) for d in dets)} symbol name(s) corrected, "
        f"{len(job.get('removed', []))} wrong detection(s) removed (removed ones are not listed or counted).",
        "",
        "Symbols (page | category | name | tag on drawing | model's symbol type | confidence | needs review):",
    ]
    for d in dets[:MAX_ITEMS]:
        lines.append(" | ".join([
            f'p{d.get("page", 1)}', CAT_NAMES.get(d["cat"], d["cat"]), _name(d), d.get("tag") or "-", d["cls"],
            f'{d["conf"]:.2f}', "yes" if needs_review(d) else ("no, corrected by user" if d.get("corrected") else "no")]))
    if len(dets) > MAX_ITEMS:
        lines.append(f"... and {len(dets) - MAX_ITEMS} more symbols not listed")
    inst = []
    for d in dets:
        p = d.get("tag_parts")
        if p:
            inst += [f"{full_tag(p, n)} = {_name(d)}" for n in p["numbers"]]
    if inst:
        lines += ["", "Instrument index (one entry per instrument number): " + "; ".join(inst)]
    return "\n".join(lines)


SYSTEM_PROMPT = """You are the assistant inside a P&ID Analyzer app, helping a process engineer understand one \
piping and instrumentation diagram that the app has analysed.

How the data was produced: a YOLOv8 model detected symbols; names come from the drawing's own text where \
available (ISA-5.1 instrument tags such as 9-PI-7101 = pressure indicator, and valve codes such as 15BV = \
15 NB ball valve, NRV = non-return valve, SRV = safety relief valve). NB means nominal bore (pipe size in mm).

Rules:
- Answer only from the detection data below. Do not invent symbols, tags, line numbers or process conditions.
- If the data cannot answer the question (e.g. flow paths, pipe connections, set pressures), say so plainly \
and suggest checking the drawing.
- Mention when an item has confidence below 0.70 ("needs review").
- Be concise: short sentences, lists for several items, tags exactly as written.

Detection data:
{context}"""


def answer(job: dict, question: str, history=None) -> dict:
    """{"answer": text, "source": "gpt" | "basic"}. Uses OpenAI when OPENAI_API_KEY is set."""
    if not gpt_available():
        return {"answer": _rule_based(job, question), "source": "basic"}
    try:
        from openai import OpenAI
        import openai
    except ImportError:
        return {"answer": _rule_based(job, question), "source": "basic"}

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=drawing_context(job))}]
    for m in (history or [])[-MAX_HISTORY:]:
        role, text = m.get("role"), str(m.get("content", ""))[:2000]
        if role in ("user", "assistant") and text:
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": question[:2000]})

    try:
        client = OpenAI(timeout=60)
        resp = client.chat.completions.create(model=GPT_MODEL, messages=messages, temperature=0.2)
        return {"answer": resp.choices[0].message.content.strip(), "source": "gpt"}
    except openai.AuthenticationError:
        problem = "the OpenAI API key was rejected. Check OPENAI_API_KEY in your .env file"
    except openai.RateLimitError:
        problem = "the OpenAI account has no remaining quota or hit a rate limit. Check billing at platform.openai.com"
    except openai.NotFoundError:
        problem = f"the model '{GPT_MODEL}' isn't available to this key. Set OPENAI_MODEL in .env to a model you can use"
    except openai.APIConnectionError:
        problem = "the server couldn't reach OpenAI (no internet connection?)"
    except Exception as e:  # never break the chat
        problem = f"GPT failed ({type(e).__name__})"
    return {"answer": f"{_rule_based(job, question)}\n\n(Basic answer: {problem}.)", "source": "basic"}
