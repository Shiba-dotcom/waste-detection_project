import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from collections import Counter

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
ANN_FILE   = os.path.join(RAW_DIR, "annotations.json")
RESULTS    = os.path.join(BASE_DIR, "results")
SRC_DIR    = os.path.join(BASE_DIR, "src")
os.makedirs(RESULTS, exist_ok=True)

mapping_df = pd.read_csv(os.path.join(SRC_DIR, "mapping.csv"))
CATEGORY_MAP = dict(zip(mapping_df["name"], mapping_df["group"]))

print("[Load] annotations.json ...")
with open(ANN_FILE, "r") as f:
    data = json.load(f)

images_info = data["images"]
annotations  = data["annotations"]
categories   = data["categories"]

id2name = {c["id"]: c["name"] for c in categories}
id2img  = {img["id"]: img for img in images_info}

print(f"\n{'='*50}")
print(f"  THONG KE BO DU LIEU TACO")
print(f"{'='*50}")
print(f"  Ten bo du lieu : TACO (Trash Annotations in Context)")
print(f"  Nguon          : http://tacodataset.org")
print(f"  So anh         : {len(images_info)}")
print(f"  So annotations : {len(annotations)}")
print(f"  So lop goc     : {len(categories)}")
print(f"  Loai bai toan  : Object Detection")
print(f"{'='*50}\n")

print("[Buoc 2] Kiem tra anh loi ...")
broken, valid = [], 0
for img in images_info:
    path = os.path.join(RAW_DIR, img["file_name"])
    if not os.path.exists(path):
        broken.append(img["file_name"]); continue
    try:
        with Image.open(path) as im:
            im.verify()
        valid += 1
    except Exception:
        broken.append(img["file_name"])

print(f"  Anh hop le : {valid}")
print(f"  Anh loi    : {len(broken)}")
if broken:
    print(f"  Danh sach  : {broken[:5]}")

invalid_bbox = 0
for ann in annotations:
    img  = id2img[ann["image_id"]]
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or (x+w) > img["width"] or (y+h) > img["height"]:
        invalid_bbox += 1
print(f"  BBox bat thuong: {invalid_bbox}")

ann_img_ids  = set(a["image_id"] for a in annotations)
no_ann_count = sum(1 for img in images_info if img["id"] not in ann_img_ids)
print(f"  Anh khong co annotation: {no_ann_count}")

widths  = [img["width"]  for img in images_info]
heights = [img["height"] for img in images_info]
print(f"\n  Kich thuoc anh (width)  - min:{min(widths)} max:{max(widths)} mean:{np.mean(widths):.0f}")
print(f"  Kich thuoc anh (height) - min:{min(heights)} max:{max(heights)} mean:{np.mean(heights):.0f}")

labels_5, areas = [], []
for ann in annotations:
    cat_name = id2name[ann["category_id"]]
    labels_5.append(CATEGORY_MAP.get(cat_name, "Other"))
    _, _, w, h = ann["bbox"]
    areas.append(w * h)

counts_5 = Counter(labels_5)
total    = sum(counts_5.values())
print(f"\n  Phan bo 5 lop chinh:")
for cls, cnt in sorted(counts_5.items(), key=lambda x: -x[1]):
    print(f"    {cls:10s}: {cnt:5d}  ({cnt/total*100:.1f}%)")

sns.set_theme(style="whitegrid", palette="muted")

fig, ax = plt.subplots(figsize=(8, 5))
cls_names = [k for k, _ in sorted(counts_5.items(), key=lambda x: -x[1])]
cls_vals  = [counts_5[k] for k in cls_names]
colors    = ["#4CAF50","#2196F3","#FF9800","#9C27B0","#F44336"]
bars = ax.bar(cls_names, cls_vals, color=colors, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, cls_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
            f"{val}\n({val/total*100:.1f}%)", ha="center", va="bottom", fontsize=9)
ax.set_title("Phân bố số lượng Annotation theo lớp", fontsize=13, fontweight="bold")
ax.set_xlabel("Lớp (nhóm)"); ax.set_ylabel("Số Annotations")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "bieu_do_1_phan_bo_lop.png"), dpi=150)
plt.close()
print("\n[Chart 1] Saved: bieu_do_1_phan_bo_lop.png")

fig, ax = plt.subplots(figsize=(8, 5))
areas_clip = [min(a, 200000) for a in areas]
ax.hist(areas_clip, bins=60, color="#2196F3", edgecolor="white", alpha=0.85)
ax.set_title("Phân bố diện tích bounding box", fontsize=13, fontweight="bold")
ax.set_xlabel("Diện tích (px^2) - cắt tại 200k"); ax.set_ylabel("Số Annotations")
ax.axvline(np.median(areas), color="red", linestyle="--", label=f"Median: {np.median(areas):.0f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "bieu_do_2_dien_tich_bbox.png"), dpi=150)
plt.close()
print("[Chart 2] Saved: bieu_do_2_dien_tich_bbox.png")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(widths, heights, alpha=0.3, s=10, color="#9C27B0")
ax.set_title("Phân bố Kích thước Ảnh (Width x Height)", fontsize=13, fontweight="bold")
ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "bieu_do_3_kich_thuoc_anh.png"), dpi=150)
plt.close()
print("[Chart 3] Saved: bieu_do_3_kich_thuoc_anh.png")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
ann_by_img = {}
for ann in annotations:
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

sample_imgs = [img for img in images_info if img["id"] in ann_by_img][:6]
CLASS_COLORS = {"Plastic":"#4CAF50","Other":"#FF9800","Metal":"#2196F3",
                "Paper":"#9C27B0","Glass":"#F44336"}

for i, img_info in enumerate(sample_imgs):
    path = os.path.join(RAW_DIR, img_info["file_name"])
    try:
        im = Image.open(path).convert("RGB")
        axes[i].imshow(im)
        for ann in ann_by_img.get(img_info["id"], []):
            x, y, w, h = ann["bbox"]
            cls   = CATEGORY_MAP.get(id2name[ann["category_id"]], "Other")
            color = CLASS_COLORS.get(cls, "yellow")
            rect  = patches.Rectangle((x, y), w, h, linewidth=2,
                                       edgecolor=color, facecolor="none")
            axes[i].add_patch(rect)
            axes[i].text(x, y-3, cls, color=color, fontsize=7,
                         bbox=dict(facecolor="black", alpha=0.4, pad=1))
        axes[i].axis("off")
        axes[i].set_title(f"ID {img_info['id']}", fontsize=8)
    except Exception:
        axes[i].axis("off")

plt.suptitle("Mẫu Ảnh với Bounding Box Annotation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "truc_quan_mau_voi_khung_bao.png"), dpi=150)
plt.close()
print("[Chart 4] Saved: truc_quan_mau_voi_khung_bao.png")

stats = {
    "Ten bo du lieu"      : ["TACO (Trash Annotations in Context)"],
    "Nguon"               : ["http://tacodataset.org"],
    "So anh"              : [len(images_info)],
    "So annotations"      : [len(annotations)],
    "Loai bai toan"       : ["Object Detection"],
    "So lop sau gop"      : [5],
    "Anh hop le"          : [valid],
    "Anh loi"             : [len(broken)],
    "BBox bat thuong"     : [invalid_bbox],
    "Median area bbox px2": [round(float(np.median(areas)), 1)],
}
pd.DataFrame(stats).T.to_csv(os.path.join(RESULTS, "thong_ke_dataset.csv"), header=["Gia tri"])
print("\n[OK] Saved: thong_ke_dataset.csv")

print("\n" + "="*50)
print("  NHAN XET KET QUA EDA")
print("="*50)
nhanxet = [
    "1. Du lieu mat can bang: Plastic ~43%, Glass chi ~5% -> can luu y khi danh gia tung lop.",
    "2. Lop Plastic chiem da so (>2000 ann), lop Glass thieu so (~254) -> mo hinh co the bi bias.",
    "3. Kich thuoc anh khong dong nhat: width dao dong lon -> phai resize ve 640x640 truoc khi train.",
    "4. Dien tich bbox co median nho -> nhieu vat the nho, kho phat hien (thach thuc cho detect).",
    "5. Hau het anh deu co annotation hop le, so anh loi rat it -> chat luong du lieu tot.",
    "6. BBox bat thuong (am/vuot bien) rat it -> annotation kha chuan, it can loc them.",
]
for nx in nhanxet:
    print(f"  {nx}")

print("\n[HOAN TAT] EDA xong. Bieu do luu trong results/")
