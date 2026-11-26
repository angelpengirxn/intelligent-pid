# Intelligent P&ID Analysis Through OCR and YOLO-Based Detection

Turns a scanned P&ID back into structured data — 44 symbol classes detected with YOLOv8, instrument tags read by OCR and decoded against ISA-5.1, delivered as an Excel inventory and a grounded chatbot. Developed as a final year project at Universiti Teknologi PETRONAS, 2026.

## What it does
1. Upload a P&ID (PDF, PNG or JPG)
2. **YOLOv8** detects symbols across the whole drawing (640 px tiles with overlap, merged afterwards)
3. Tags are read from the PDF text layer, or by **OCR** for drawn or scanned text
4. Instrument tags are decoded with **ISA-5.1** letters (e.g. `9-PI-7101` → Pressure indicator);
   valves are named from the drawing's own codes (e.g. `15BV` → Ball valve, 15 NB)
5. Results show on the drawing, in a symbol list, as an **Excel report**, and through a **chatbot**

## Project structure
```
app/            web app: FastAPI backend (main.py, pipeline.py, report.py) + frontend (static/index.html)
models/         class_names.json (symbol names, categories, show/hide) + best.pt (not in Git, see below)
src/            data preparation and helper scripts (tiling, checks, diagnostics, sharing)
hf_space/       launcher files for Hugging Face Spaces
```

## Run locally
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-app.txt
# put the trained model at models/best.pt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

## Model and data
- Trained on Dataset-P&ID / Digitize-PID (Paliwal et al., 2021): 500 synthetic P&IDs, 32 symbol classes
- `models/best.pt` is not stored in this repository (size); it is kept on Google Drive and in the Hugging Face Space
- `data/` (tiles and raw drawings) is not stored in this repository

## Reference
Paliwal, S., Jain, A., Sharma, M., & Vig, L. (2021). Digitize-PID: Automatic digitization of piping and
instrumentation diagrams. arXiv:2109.03794.
