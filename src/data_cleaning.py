"""
data_cleaning.py - Lam sach du lieu TACO truoc khi train
Nhom 2 - Waste Detection

Chay truoc Training_data.py de:
  1. Kiem tra & loai bo annotation trung lap
  2. Kiem tra gia tri thieu (missing fields)
  3. Kiem tra nhan khong hop le (orphan category)
  4. Loai bo BBox bat thuong (am, vuot bien, qua nho)
  5. Kiem tra anh loi (truncated, grayscale, kich thuoc)
  6. Xuat annotations_cleaned.json de dung cho cac buoc sau
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import os
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter, defaultdict

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
ANN_FILE  = os.path.join(RAW_DIR, "annotations.json")
SRC_DIR   = os.path.join(BASE_DIR, "src")
RESULTS   = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS, exist_ok=True)

# File output sau khi cleaning
CLEANED_ANN_FILE = os.path.join(RAW_DIR, "annotations_cleaned.json")

# Config
MIN_BBOX_PX = 5       # BBox nho hon 5px (w hoac h) se bi loai
MIN_IMG_SIZE = 32      # Anh nho hon 32x32 se bi loai

# ── Load data ────────────────────────────────────────────────────────
print("=" * 60)
print("  DATA CLEANING - TACO Dataset")
print("=" * 60)

print("\n[Load] annotations.json ...")
with open(ANN_FILE, "r") as f:
    data = json.load(f)

images_info = data["images"]
annotations = data["annotations"]
categories  = data["categories"]

# Luu so luong goc truoc khi mutate
ORIG_IMG_COUNT = len(images_info)
ORIG_ANN_COUNT = len(annotations)

id2name = {c["id"]: c["name"] for c in categories}
id2img  = {img["id"]: img for img in images_info}

print(f"  So anh goc         : {ORIG_IMG_COUNT}")
print(f"  So annotations goc : {ORIG_ANN_COUNT}")
print(f"  So categories      : {len(categories)}")

# Tracking bo loc
removed_reasons = defaultdict(int)
log_messages = []

def log(msg):
    print(f"  {msg}")
    log_messages.append(msg)


# ═══════════════════════════════════════════════════════════════════
# BUOC 1: KIEM TRA DONG TRUNG LAP (Duplicates)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("[Buoc 1] Kiem tra dong trung lap (Duplicates)")
print(f"{'─'*60}")

# 1a. Duplicate annotations (cung image_id + category_id + bbox)
seen_ann = set()
dup_ann_indices = []
for i, ann in enumerate(annotations):
    key = (ann["image_id"], ann["category_id"], tuple(ann["bbox"]))
    if key in seen_ann:
        dup_ann_indices.append(i)
    else:
        seen_ann.add(key)

log(f"Duplicate annotations: {len(dup_ann_indices)}")
if dup_ann_indices:
    log(f"  -> Se loai bo {len(dup_ann_indices)} annotation trung lap")
    # Loai bo tu cuoi len dau de khong lech index
    for idx in sorted(dup_ann_indices, reverse=True):
        annotations.pop(idx)
    removed_reasons["duplicate_annotation"] = len(dup_ann_indices)

# 1b. Duplicate image IDs
img_ids = [img["id"] for img in images_info]
dup_img_ids = [iid for iid, cnt in Counter(img_ids).items() if cnt > 1]
log(f"Duplicate image IDs: {len(dup_img_ids)}")
if dup_img_ids:
    log(f"  -> Image IDs trung: {dup_img_ids[:5]}")

# 1c. Duplicate file names
file_names = [img["file_name"] for img in images_info]
dup_fnames = [fn for fn, cnt in Counter(file_names).items() if cnt > 1]
log(f"Duplicate file names: {len(dup_fnames)}")
if dup_fnames:
    log(f"  -> File names trung: {dup_fnames[:5]}")


# ═══════════════════════════════════════════════════════════════════
# BUOC 2: KIEM TRA GIA TRI THIEU (Missing Values)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("[Buoc 2] Kiem tra gia tri thieu (Missing Values)")
print(f"{'─'*60}")

# 2a. Kiem tra truong bat buoc trong images
required_img_fields = ["id", "file_name", "width", "height"]
missing_img = defaultdict(int)
bad_img_ids = set()

for img in images_info:
    for field in required_img_fields:
        if field not in img or img[field] is None:
            missing_img[field] += 1
            bad_img_ids.add(img.get("id"))

if missing_img:
    log(f"Images thieu truong:")
    for field, cnt in missing_img.items():
        log(f"  - {field}: {cnt} anh thieu")
else:
    log("Images: Tat ca truong bat buoc day du")

# 2b. Kiem tra truong bat buoc trong annotations
required_ann_fields = ["id", "image_id", "category_id", "bbox"]
missing_ann = defaultdict(int)
bad_ann_indices = []

for i, ann in enumerate(annotations):
    is_bad = False
    for field in required_ann_fields:
        if field not in ann or ann[field] is None:
            missing_ann[field] += 1
            is_bad = True
    if is_bad:
        bad_ann_indices.append(i)

if missing_ann:
    log(f"Annotations thieu truong:")
    for field, cnt in missing_ann.items():
        log(f"  - {field}: {cnt} annotations thieu")
    # Loai bo annotations thieu truong
    for idx in sorted(bad_ann_indices, reverse=True):
        annotations.pop(idx)
    removed_reasons["missing_fields"] = len(bad_ann_indices)
    log(f"  -> Da loai bo {len(bad_ann_indices)} annotations thieu truong")
else:
    log("Annotations: Tat ca truong bat buoc day du")

# 2c. Anh khong co annotation
ann_img_ids = set(a["image_id"] for a in annotations)
no_ann_imgs = [img for img in images_info if img["id"] not in ann_img_ids]
log(f"Anh khong co annotation: {len(no_ann_imgs)}")


# ═══════════════════════════════════════════════════════════════════
# BUOC 3: KIEM TRA NHAN KHONG HOP LE (Orphan Categories)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("[Buoc 3] Kiem tra nhan khong hop le")
print(f"{'─'*60}")

# 3a. category_id trong annotation khong ton tai trong categories
valid_cat_ids = set(c["id"] for c in categories)
orphan_ann_indices = []
for i, ann in enumerate(annotations):
    if ann["category_id"] not in valid_cat_ids:
        orphan_ann_indices.append(i)

log(f"Annotations voi category_id khong hop le: {len(orphan_ann_indices)}")
if orphan_ann_indices:
    for idx in sorted(orphan_ann_indices, reverse=True):
        log(f"  - ann_id={annotations[idx]['id']}, cat_id={annotations[idx]['category_id']}")
        annotations.pop(idx)
    removed_reasons["orphan_category"] = len(orphan_ann_indices)

# 3b. Kiem tra mapping.csv co phu tat ca categories khong
mapping_df = pd.read_csv(os.path.join(SRC_DIR, "mapping.csv"))
mapped_names = set(mapping_df["name"])
cat_names = set(c["name"] for c in categories)
unmapped = cat_names - mapped_names
log(f"Categories khong co trong mapping.csv: {len(unmapped)}")
if unmapped:
    for name in unmapped:
        log(f"  - '{name}' -> se mac dinh vao 'Other'")


# ═══════════════════════════════════════════════════════════════════
# BUOC 4: LOAI BO BBOX BAT THUONG (Anomalies)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("[Buoc 4] Kiem tra & loai bo BBox bat thuong")
print(f"{'─'*60}")

anomaly_indices = []
anomaly_details = defaultdict(int)

for i, ann in enumerate(annotations):
    img = id2img.get(ann["image_id"])
    if img is None:
        anomaly_indices.append(i)
        anomaly_details["image_id_not_found"] += 1
        continue

    x, y, w, h = ann["bbox"]

    # Kiem tra gia tri NaN / Inf
    bbox_vals = [x, y, w, h]
    if any(not isinstance(v, (int, float)) for v in bbox_vals):
        anomaly_indices.append(i)
        anomaly_details["invalid_type"] += 1
        continue

    if any(np.isnan(v) or np.isinf(v) for v in bbox_vals):
        anomaly_indices.append(i)
        anomaly_details["nan_or_inf"] += 1
        continue

    # BBox am hoac zero
    if w <= 0 or h <= 0:
        anomaly_indices.append(i)
        anomaly_details["zero_or_negative_size"] += 1
        continue

    if x < 0 or y < 0:
        anomaly_indices.append(i)
        anomaly_details["negative_position"] += 1
        continue

    # BBox vuot ra ngoai anh
    if (x + w) > img["width"] or (y + h) > img["height"]:
        anomaly_indices.append(i)
        anomaly_details["out_of_bounds"] += 1
        continue

    # BBox qua nho (< MIN_BBOX_PX pixel)
    if w < MIN_BBOX_PX or h < MIN_BBOX_PX:
        anomaly_indices.append(i)
        anomaly_details["too_small"] += 1
        continue

    # BBox = toan bo anh (co the la loi annotation)
    if w >= img["width"] * 0.99 and h >= img["height"] * 0.99:
        anomaly_indices.append(i)
        anomaly_details["covers_entire_image"] += 1
        continue

log(f"Tong BBox bat thuong: {len(anomaly_indices)}")
for reason, cnt in anomaly_details.items():
    log(f"  - {reason}: {cnt}")

if anomaly_indices:
    for idx in sorted(anomaly_indices, reverse=True):
        annotations.pop(idx)
    removed_reasons["bbox_anomaly"] = len(anomaly_indices)
    log(f"  -> Da loai bo {len(anomaly_indices)} annotations bat thuong")


# ═══════════════════════════════════════════════════════════════════
# BUOC 5: KIEM TRA ANH LOI, KICH THUOC
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("[Buoc 5] Kiem tra anh loi, truncated, grayscale, kich thuoc")
print(f"{'─'*60}")

broken_imgs = []
truncated_imgs = []
grayscale_imgs = []
too_small_imgs = []
valid_count = 0

for img in images_info:
    path = os.path.join(RAW_DIR, img["file_name"])

    # Kiem tra file ton tai
    if not os.path.exists(path):
        broken_imgs.append(img["file_name"])
        continue

    try:
        with Image.open(path) as im:
            # Verify header
            im.verify()
    except Exception:
        broken_imgs.append(img["file_name"])
        continue

    try:
        with Image.open(path) as im:
            # Load toan bo pixel de kiem tra truncated
            im.load()

            # Kiem tra grayscale
            if im.mode in ("L", "LA", "1"):
                grayscale_imgs.append(img["file_name"])

            # Kiem tra kich thuoc qua nho
            w_px, h_px = im.size
            if w_px < MIN_IMG_SIZE or h_px < MIN_IMG_SIZE:
                too_small_imgs.append(img["file_name"])
                continue

        valid_count += 1
    except Exception:
        truncated_imgs.append(img["file_name"])

log(f"Anh hop le          : {valid_count}")
log(f"Anh loi (header)     : {len(broken_imgs)}")
log(f"Anh truncated (body) : {len(truncated_imgs)}")
log(f"Anh grayscale        : {len(grayscale_imgs)}")
log(f"Anh qua nho (<{MIN_IMG_SIZE}px) : {len(too_small_imgs)}")

if broken_imgs:
    log(f"  Danh sach anh loi: {broken_imgs[:10]}")
if truncated_imgs:
    log(f"  Danh sach truncated: {truncated_imgs[:10]}")
if grayscale_imgs:
    log(f"  Danh sach grayscale: {grayscale_imgs[:10]}")

# Loai bo images va annotations lien quan
bad_filenames = set(broken_imgs + truncated_imgs + too_small_imgs)
if bad_filenames:
    bad_img_ids_from_files = set()
    for img in images_info:
        if img["file_name"] in bad_filenames:
            bad_img_ids_from_files.add(img["id"])

    before = len(annotations)
    annotations = [a for a in annotations if a["image_id"] not in bad_img_ids_from_files]
    images_info = [img for img in images_info if img["file_name"] not in bad_filenames]
    removed_ann = before - len(annotations)

    removed_reasons["bad_image"] = len(bad_filenames)
    log(f"  -> Loai bo {len(bad_filenames)} anh va {removed_ann} annotations lien quan")


# ═══════════════════════════════════════════════════════════════════
# BUOC 6: THONG KE TONG KET
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("[Buoc 6] Thong ke sau khi cleaning")
print(f"{'─'*60}")

# Cap nhat lai id maps
id2img_clean = {img["id"]: img for img in images_info}

log(f"So anh sau cleaning       : {len(images_info)}")
log(f"So annotations sau cleaning: {len(annotations)}")

# Thong ke so luong da loai bo
total_removed = sum(removed_reasons.values())
log(f"\nTong so da loai bo/sua:")
for reason, cnt in removed_reasons.items():
    log(f"  - {reason}: {cnt}")
log(f"  TONG: {total_removed}")

# Kiem tra YOLO conversion validity (preview)
print(f"\n{'─'*60}")
print("[Buoc 7] Preview kiem tra YOLO label validity")
print(f"{'─'*60}")

mapping = dict(zip(mapping_df["name"], mapping_df["group"]))
yolo_issues = 0
for ann in annotations:
    img = id2img_clean.get(ann["image_id"])
    if img is None:
        continue
    x, y, w, h = ann["bbox"]
    img_w, img_h = img["width"], img["height"]

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Kiem tra gia tri YOLO phai nam trong [0, 1]
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1
            and 0 < w_norm <= 1 and 0 < h_norm <= 1):
        yolo_issues += 1

log(f"YOLO labels se co gia tri ngoai [0,1]: {yolo_issues}")
if yolo_issues == 0:
    log("-> Tat ca labels se hop le khi convert sang YOLO format!")
else:
    log(f"-> Can xem lai {yolo_issues} annotations truoc khi train")


# ═══════════════════════════════════════════════════════════════════
# BUOC 8: LUU FILE DA LAM SACH
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("[Buoc 8] Luu annotations_cleaned.json")
print(f"{'─'*60}")

cleaned_data = {
    "images": images_info,
    "annotations": annotations,
    "categories": categories,
}

with open(CLEANED_ANN_FILE, "w") as f:
    json.dump(cleaned_data, f, indent=2)

log(f"Da luu: {CLEANED_ANN_FILE}")
log(f"  Images     : {len(images_info)}")
log(f"  Annotations: {len(annotations)}")

# Luu bao cao cleaning ra CSV
cleaning_report = {
    "Metric": [],
    "Value": [],
}

report_items = [
    ("So anh goc", ORIG_IMG_COUNT),
    ("So annotations goc", ORIG_ANN_COUNT),
    ("So anh sau cleaning", len(images_info)),
    ("So annotations sau cleaning", len(annotations)),
    ("Duplicate annotations loai", removed_reasons.get("duplicate_annotation", 0)),
    ("Missing fields loai", removed_reasons.get("missing_fields", 0)),
    ("Orphan category loai", removed_reasons.get("orphan_category", 0)),
    ("BBox bat thuong loai", removed_reasons.get("bbox_anomaly", 0)),
    ("Anh loi/truncated/nho loai", removed_reasons.get("bad_image", 0)),
    ("YOLO labels ngoai [0,1]", yolo_issues),
    ("Anh grayscale (giu lai)", len(grayscale_imgs)),
]

for metric, value in report_items:
    cleaning_report["Metric"].append(metric)
    cleaning_report["Value"].append(value)

report_path = os.path.join(RESULTS, "bao_cao_cleaning.csv")
pd.DataFrame(cleaning_report).to_csv(report_path, index=False)
log(f"Da luu bao cao: {report_path}")

print(f"\n{'='*60}")
print("  [HOAN TAT] Data Cleaning xong!")
print(f"  Su dung 'annotations_cleaned.json' cho cac buoc tiep theo")
print(f"{'='*60}")
