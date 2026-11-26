from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="SherifAhmed/digitize-pid-yolo",
    repo_type="dataset",
    local_dir="data/raw/digitize_pid",
)
print("Done!")