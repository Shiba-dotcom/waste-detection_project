import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS  = os.path.join(BASE_DIR, "results")
CSV_PATH = os.path.join(BASE_DIR, "results", "runs", "prev_yolov8", "detect", "result", "results.csv")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(df["epoch"], df["metrics/mAP50(B)"], color="#2196F3", linewidth=2)
axes[0].set_title("mAP@0.5 qua các Epoch", fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("mAP@0.5")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["epoch"], df["metrics/precision(B)"], color="#4CAF50", linewidth=2)
axes[1].set_title("Precision qua các Epoch", fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Precision")
axes[1].grid(True, alpha=0.3)

axes[2].plot(df["epoch"], df["metrics/recall(B)"], color="#FF9800", linewidth=2)
axes[2].set_title("Recall qua các Epoch", fontweight="bold")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Recall")
axes[2].grid(True, alpha=0.3)

plt.suptitle("Lịch sử Training YOLOv8n (50 Epoch)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "bieu_do_4_training_history.png"), dpi=150)
plt.close()
print("[Chart 4] Saved: bieu_do_4_training_history.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
loss_cols = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
loss_names = ["Box Loss", "Cls Loss", "DFL Loss"]
colors = ["#F44336", "#9C27B0", "#00BCD4"]

for ax, col, name, color in zip(axes, loss_cols, loss_names, colors):
    ax.plot(df["epoch"], df[col], color=color, linewidth=2)
    ax.set_title(f"{name} qua các Epoch", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

plt.suptitle("Đường cong Loss - YOLOv8n (50 Epoch)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "bieu_do_5_loss_curves.png"), dpi=150)
plt.close()
print("[Chart 5] Saved: bieu_do_5_loss_curves.png")

last = df.iloc[-1]
ep5  = df[df["epoch"] == 5].iloc[0] if 5 in df["epoch"].values else df.iloc[4]

result_table = pd.DataFrame([
    {
        "Model": "YOLOv8n (baseline, 5 epoch)",
        "mAP@0.5 (%)": round(ep5["metrics/mAP50(B)"]*100, 2),
        "mAP@0.5:0.95 (%)": round(ep5["metrics/mAP50-95(B)"]*100, 2),
        "Precision (%)": round(ep5["metrics/precision(B)"]*100, 2),
        "Recall (%)": round(ep5["metrics/recall(B)"]*100, 2),
        "Inference (ms)": "N/A",
    },
    {
        "Model": "YOLOv8n (50 epoch)",
        "mAP@0.5 (%)": round(last["metrics/mAP50(B)"]*100, 2),
        "mAP@0.5:0.95 (%)": round(last["metrics/mAP50-95(B)"]*100, 2),
        "Precision (%)": round(last["metrics/precision(B)"]*100, 2),
        "Recall (%)": round(last["metrics/recall(B)"]*100, 2),
        "Inference (ms)": "N/A",
    }
])
result_table.to_csv(os.path.join(RESULTS, "ket_qua_baseline.csv"), index=False)
print("\n[OK] Saved: ket_qua_baseline.csv")
print("\nKet qua baseline (5 epoch):")
print(f"  mAP50: {ep5['metrics/mAP50(B)']:.4f} | Precision: {ep5['metrics/precision(B)']:.4f} | Recall: {ep5['metrics/recall(B)']:.4f}")
print("\nKet qua day du (50 epoch):")
print(f"  mAP50: {last['metrics/mAP50(B)']:.4f} | Precision: {last['metrics/precision(B)']:.4f} | Recall: {last['metrics/recall(B)']:.4f}")
print("\n[HOAN TAT] Bieu do training history da luu trong results/")
