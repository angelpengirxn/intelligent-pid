"""Push the latest app to the Hugging Face Space in one commit.

    python src/deploy_space.py            # code + launcher + class names
    python src/deploy_space.py --model    # also upload models/best.pt (67 MB, only after retraining)
    python src/deploy_space.py --dry-run  # show what would change, upload nothing

One-time setup:  pip install huggingface_hub   then   hf auth login   (paste a token with Write access)
"""
import sys
from pathlib import Path

SPACE = "angeliann/intelligent-pid-analyser"
ROOT = Path(__file__).resolve().parent.parent

# local file -> path inside the Space
FILES = {
    "hf_space/app.py": "app.py",
    "hf_space/requirements.txt": "requirements.txt",
    "hf_space/packages.txt": "packages.txt",
    "app/__init__.py": "app/__init__.py",
    "app/main.py": "app/main.py",
    "app/pipeline.py": "app/pipeline.py",
    "app/report.py": "app/report.py",
    "app/static/index.html": "app/static/index.html",
    "models/class_names.json": "class_names.json",
}
MODEL = {"models/best.pt": "best.pt"}
# old copies from the first (flattened) web upload; the app/ folder replaces them
OLD = ["main.py", "pipeline.py", "report.py", "index.html", "__init__.py"]


def main():
    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    except ImportError:
        sys.exit("Install it first:  pip install huggingface_hub")

    files = dict(FILES)
    if "--model" in sys.argv:
        files.update(MODEL)
    missing = [f for f in files if not (ROOT / f).exists() and not f.endswith("__init__.py")]
    if missing:
        sys.exit(f"Missing local files: {missing}")

    api = HfApi()
    existing = set(api.list_repo_files(SPACE, repo_type="space"))
    ops = []
    for local, remote in files.items():
        path = ROOT / local
        data = path.read_bytes() if path.exists() else b""
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=data))
    ops += [CommitOperationDelete(path_in_repo=p) for p in OLD if p in existing]

    print(f"Space: {SPACE}")
    for op in ops:
        kind = "upload" if isinstance(op, CommitOperationAdd) else "delete"
        print(f"  {kind:6s} {op.path_in_repo}")
    if "--dry-run" in sys.argv:
        print("Dry run: nothing uploaded.")
        return

    api.create_commit(SPACE, repo_type="space", operations=ops,
                      commit_message="Update app from local project")
    print(f"Done. The Space is rebuilding: https://huggingface.co/spaces/{SPACE}")
    print("Direct link: https://" + SPACE.replace("/", "-") + ".hf.space")


if __name__ == "__main__":
    main()
