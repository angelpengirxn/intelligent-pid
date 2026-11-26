"""FastAPI server: serves the web page and the analysis API.

Run from the project folder:
    uvicorn app.main:app --reload
then open http://127.0.0.1:8000
"""
import asyncio
import os
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from typing import Optional

from pydantic import BaseModel

from . import pipeline, report


def _load_env_file():
    """Read KEY=value lines from .env in the project folder (e.g. OPENAI_API_KEY), without overriding real env vars."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()
report.GPT_MODEL = os.getenv("OPENAI_MODEL", report.GPT_MODEL)

STATIC = Path(os.getenv("STATIC_DIR", Path(__file__).parent / "static"))
MAX_MB = 50
ALLOWED = (".pdf", ".png", ".jpg", ".jpeg")

app = FastAPI(title="P&ID Analyzer")
JOBS: "OrderedDict[str, dict]" = OrderedDict()   # last 20 results, kept in memory


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    model = pipeline.get_model()
    return {
        "model": pipeline.MODEL_PATH.name if model is not None else None,
        "model_path": str(pipeline.MODEL_PATH),
        "classes": [c["name"] for c in pipeline.class_info(model).values()] if model is not None else [],
        "ocr": pipeline.ocr_available(),
        "gpt": report.gpt_available(),
        "gpt_model": report.GPT_MODEL if report.gpt_available() else None,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Upload a drawing. Analyses page 1 now; other pages are fetched with /api/page/{job_id}/{n}."""
    name = file.filename or "drawing"
    if not name.lower().endswith(ALLOWED):
        raise HTTPException(400, "Upload a PDF, PNG or JPG file.")
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_MB} MB.")
    try:
        n_pages = await run_in_threadpool(pipeline.page_count, data, name)
        result = await run_in_threadpool(pipeline.analyze, data, name, 1)
    except Exception as e:
        raise HTTPException(422, f"Could not read this file ({type(e).__name__}: {e}).")

    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"filename": name, "data": data, "page_count": n_pages, "model": result["model"],
                    "pages": {1: result}, "lock": asyncio.Lock()}
    while len(JOBS) > 20:
        JOBS.popitem(last=False)
    return {**result, "job_id": job_id, "page_count": n_pages}


@app.get("/api/page/{job_id}/{page_no}")
async def page(job_id: str, page_no: int):
    """Analyse (or return the cached result of) one page of an uploaded PDF."""
    job = _job(job_id)
    if not 1 <= page_no <= job["page_count"]:
        raise HTTPException(404, f"This document has {job['page_count']} page(s).")
    async with job["lock"]:                       # one page at a time per document
        if page_no not in job["pages"]:
            try:
                job["pages"][page_no] = await run_in_threadpool(
                    pipeline.analyze, job["data"], job["filename"], page_no)
            except Exception as e:
                raise HTTPException(422, f"Could not read page {page_no} ({type(e).__name__}: {e}).")
    return {**job["pages"][page_no], "job_id": job_id, "page_count": job["page_count"]}


def _job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "This result has expired. Upload the drawing again.")
    return job


def _document(job: dict) -> dict:
    """All analysed pages combined, for the Excel report and the chatbot.
    Symbols the user marked as wrong are left out (listed separately in "removed")."""
    dets, removed = [], []
    for n in sorted(job["pages"]):
        for d in job["pages"][n]["detections"]:
            (removed if d.get("wrong") else dets).append({**d, "id": f'p{n}-{d["id"]}', "page": n})
    first = job["pages"][min(job["pages"])]
    return {"filename": job["filename"], "model": job["model"], "width": first["width"], "height": first["height"],
            "page_count": job["page_count"], "pages_done": len(job["pages"]), "detections": dets, "removed": removed}


@app.get("/api/classes")
def classes():
    """Symbol types for the correction dropdown."""
    return pipeline.class_list()


class EditIn(BaseModel):
    job_id: str
    page: int = 1
    id: str
    cls_id: Optional[int] = None    # change the symbol type
    wrong: Optional[bool] = None    # True = not a symbol (remove), False = restore


@app.post("/api/edit")
def edit(body: EditIn):
    """User correction of one detection; the change is kept for the Excel report and the chatbot."""
    job = _job(body.job_id)
    res = job["pages"].get(body.page)
    d = next((d for d in res["detections"] if d["id"] == body.id), None) if res else None
    if d is None:
        raise HTTPException(404, "Symbol not found. Upload the drawing again.")
    if body.cls_id is not None:
        try:
            pipeline.relabel(d, body.cls_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
    if body.wrong is not None:
        if body.wrong:
            d["wrong"] = True
        else:
            d.pop("wrong", None)
    return d


@app.get("/api/report/{job_id}")
def excel_report(job_id: str):
    job = _document(_job(job_id))
    stem = Path(job["filename"]).stem
    return Response(
        report.build_excel(job),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{stem}_report.xlsx"'},
    )


class ChatIn(BaseModel):
    job_id: str
    question: str
    history: list = []          # earlier messages: [{"role": "user"|"assistant", "content": "..."}]


@app.post("/api/chat")
async def chat(body: ChatIn):
    job = _document(_job(body.job_id))
    if not body.question.strip():
        raise HTTPException(400, "Type a question first.")
    return await run_in_threadpool(report.answer, job, body.question, body.history)
