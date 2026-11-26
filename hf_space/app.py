"""Hugging Face Space launcher (Gradio SDK).
Serves the P&ID Analyzer web app (FastAPI + custom frontend) on port 7860.
Finds the app files wherever they were uploaded (inside app/ or flattened to the top level)."""
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("DEVICE", "cpu")      # run the model on CPU (no GPU quota needed)
os.environ["GRADIO_SSR_MODE"] = "false"    # no separate page server in front, so "/" reaches this app

# ZeroGPU Spaces must declare at least one GPU function at startup ("spaces" is imported first,
# as Hugging Face recommends). The model runs on CPU, so visitors never use up GPU quota;
# this function only satisfies the check.
try:
    import spaces

    @spaces.GPU(duration=10)
    def gpu_check():
        return True
except ImportError:                         # running outside Hugging Face
    pass


def find(name):
    hits = [p for p in ROOT.rglob(name) if ".git" not in p.parts and "__pycache__" not in p.parts]
    return min(hits, key=lambda p: len(p.parts)) if hits else None


print("Space files:")
for p in sorted(ROOT.rglob("*")):
    if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts:
        print("  ", p.relative_to(ROOT), f"({p.stat().st_size // 1024} KB)")

main_py = next((p for p in [ROOT / "app" / "main.py"] if p.exists()), None) or find("main.py")
if main_py is None:
    sys.exit("main.py not found: upload the app/ folder (main.py, pipeline.py, report.py, static/index.html)")
code_dir = main_py.parent
for f in ("pipeline.py", "report.py"):
    if not (code_dir / f).exists():
        sys.exit(f"{f} not found next to {main_py.relative_to(ROOT)}")

for var, name in (("MODEL_PATH", "best.pt"), ("CLASS_NAMES", "class_names.json")):
    hit = find(name)
    if hit and var not in os.environ:
        os.environ[var] = str(hit)
index = code_dir / "static" / "index.html"
if not index.exists():
    hit = find("index.html")
    if hit:
        os.environ["STATIC_DIR"] = str(hit.parent)
print("code:", code_dir, "| model:", os.environ.get("MODEL_PATH"), "| names:", os.environ.get("CLASS_NAMES"),
      "| page:", os.environ.get("STATIC_DIR", index.parent))

# Register the code folder as the "app" package (this file is also called app.py)
pkg = types.ModuleType("app")
pkg.__path__ = [str(code_dir)]
sys.modules["app"] = pkg

import gradio as gr                         # noqa: E402

import app.main as app_main                 # noqa: E402

if (code_dir / "static" / "index.html").exists() is False and find("index.html"):
    app_main.STATIC = find("index.html").parent      # page uploaded without its static/ folder

with gr.Blocks(title="P&ID Analyzer") as demo:
    gr.Markdown("P&ID Analyzer is running. Open the Space's main page to use it.")

# The app's own routes: "/" = P&ID Analyzer page, "/api/..." = upload, Excel and chat
APP_ROUTES = [r for r in app_main.app.router.routes
              if getattr(r, "path", "") == "/" or getattr(r, "path", "").startswith("/api/")]


def add_app_routes(server):
    """Put the app's routes in front of Gradio's, so they win (works after the server has started)."""
    routes = server.router.routes
    for r in reversed(APP_ROUTES):
        if r not in routes:
            routes.insert(0, r)


if __name__ == "__main__":
    # Let Gradio pick host and port from Hugging Face's settings, start it, then add the app's routes.
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"), ssr_mode=False, prevent_thread_lock=True)
    add_app_routes(demo.app)
    print("P&ID Analyzer routes added:", [r.path for r in APP_ROUTES])
    demo.block_thread()
