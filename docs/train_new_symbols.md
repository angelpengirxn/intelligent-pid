# Train the model on the 12 new symbols (Google Colab)

## 1. Make the synthetic data (on the Mac)
```bash
source venv/bin/activate
python src/synth_symbols.py --preview      # look at docs/synth_preview.png and docs/synth_preview_tiles.jpg first
python src/synth_symbols.py                # about 10 minutes -> data/synth.zip
```
Upload `data/synth.zip` to Google Drive > FYP (next to `tiles.zip`).

## 2. Colab (Runtime > Change runtime type > T4 GPU), one cell each

```python
from google.colab import drive; drive.mount('/content/drive')
!pip -q install ultralytics
!mkdir -p /content/datasets
!unzip -q -o /content/drive/MyDrive/FYP/tiles.zip -d /content/datasets
!unzip -q -o /content/drive/MyDrive/FYP/synth.zip -d /content/datasets
!ls /content/datasets            # must show: synth  tiles
```

If `ls` shows `images  labels` instead of `tiles`, run: `!mkdir -p /content/datasets/tiles && mv /content/datasets/images /content/datasets/labels /content/datasets/tiles/`

```python
from ultralytics import YOLO
model = YOLO('/content/drive/MyDrive/FYP/runs/baseline_v8s/weights/best.pt')   # start from your 32-class model
model.train(data='/content/datasets/synth/data_v2.yaml', epochs=25, patience=8, imgsz=640, batch=32,
            fliplr=0, degrees=0, project='/content/drive/MyDrive/FYP/runs', name='new_symbols_v8s')
```

- Training goes straight to Drive, so if Colab disconnects, resume with:
  `YOLO('/content/drive/MyDrive/FYP/runs/new_symbols_v8s/weights/last.pt').train(resume=True)`
- The results table at the end lists mAP50 for every class. The 12 new ones are ids 32-43.

## 3. Use it in the app
Download `runs/new_symbols_v8s/weights/best.pt` from Drive and replace `models/best.pt`
(keep the old one as `models/best_32.pt`). Restart uvicorn. The new names are already in `models/class_names.json`.
