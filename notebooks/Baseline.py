import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, time, glob
import pandas as pd
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
YAML_PATH = BASE_DIR / "data" / "processed" / "dataset.yaml"
RESULTS   = BASE_DIR / "results"
RESULTS.mkdir(exist_ok=True)

print("="*55)
print("  BASELINE: YOLOv8n - Fine-tune 5 Epoch")
print("  Ly do chon: Phu hop bai toan Object Detection")
print("  Metrics: mAP@0.5, Precision, Recall, Inference time")
print("="*55)

try:
    from ultralytics import YOLO
except ImportError:
    print("[LOI] Chua cai ultralytics. Chay: pip install ultralytics")
    sys.exit(1)

print("\n[1] Bat dau train baseline YOLOv8n (5 epoch) ...")
model = YOLO("yolov8n.pt")

train_results = model.train(
    data     = str(YAML_PATH),
    epochs   = 5,
    imgsz    = 640,
    batch    = 16,
    name     = "baseline_yolov8n",
    project  = str(RESULTS / "runs"),
    exist_ok = True,
    verbose  = True,
)
print("[1] Train xong!")

print("\n[2] Danh gia tren tap Validation ...")
metrics = model.val(verbose=False)

map50    = float(metrics.box.map50)
prec     = float(metrics.box.mp)
recall   = float(metrics.box.mr)
map5095  = float(metrics.box.map)

print(f"\n{'='*55}")
print(f"  KET QUA BASELINE - YOLOv8n (5 epoch)")
print(f"{'='*55}")
print(f"  mAP@0.5      : {map50:.4f}   ({map50*100:.2f}%)")
print(f"  mAP@0.5:0.95 : {map5095:.4f}")
print(f"  Precision    : {prec:.4f}   ({prec*100:.2f}%)")
print(f"  Recall       : {recall:.4f}  ({recall*100:.2f}%)")
print(f"{'='*55}")

print("\n[3] Do Inference Time ...")
val_imgs = glob.glob(str(BASE_DIR / "data" / "processed" / "images" / "val" / "**" / "*.jpg"),
                     recursive=True)[:50]
if val_imgs:
    t0 = time.perf_counter()
    for img_path in val_imgs:
        model.predict(img_path, verbose=False)
    elapsed = (time.perf_counter() - t0) / len(val_imgs) * 1000
    print(f"  Inference time (avg): {elapsed:.1f} ms/anh")
else:
    elapsed = None
    print("  [Bo qua] Khong tim thay anh val.")

result_row = {
    "Model"           : "YOLOv8n (baseline, 5 epoch)",
    "mAP@0.5 (%)"     : round(map50*100, 2),
    "mAP@0.5:0.95 (%)": round(map5095*100, 2),
    "Precision (%)"   : round(prec*100, 2),
    "Recall (%)"      : round(recall*100, 2),
    "Inference (ms)"  : round(elapsed, 1) if elapsed else "N/A",
}
df = pd.DataFrame([result_row])
csv_path = RESULTS / "ket_qua_baseline.csv"
df.to_csv(csv_path, index=False)
print(f"\n[OK] Luu ket qua: {csv_path}")

