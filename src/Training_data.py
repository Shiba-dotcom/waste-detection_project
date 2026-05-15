import json
import pandas as pd
import os
import shutil

# Uu tien dung file da lam sach, fallback sang file goc
cleaned_path = "../data/raw/annotations_cleaned.json"
original_path = "../data/raw/annotations.json"

if os.path.exists(cleaned_path):
    ann_path = cleaned_path
    print("[INFO] Su dung annotations_cleaned.json (da lam sach)")
else:
    ann_path = original_path
    print("[WARN] Khong tim thay annotations_cleaned.json, dung file goc")
    print("       Hay chay 'python src/data_cleaning.py' truoc!")

with open(ann_path, "r") as f:
    data = json.load(f)

mapping_df = pd.read_csv("mapping.csv")
mapping = dict(zip(mapping_df["name"], mapping_df["group"]))


idtn = {c["id"]: c["name"] for c in data["categories"]}

base_out = "../data/processed" # Thư mục chứa 
img_out = os.path.join(base_out, "images") # Thư mục chứa file hình
label_out = os.path.join(base_out, "labels") # Thư mục hứa file annotation (định dạng txt)

if os.path.exists(base_out):
    shutil.rmtree(base_out)


os.makedirs(img_out, exist_ok=True)
os.makedirs(label_out, exist_ok=True)

classes = sorted(set(mapping.values()))
class2id = {c: i for i, c in enumerate(classes)}



img_map = {img["id"]: img for img in data["images"]}

raw_img_dir = "../data/raw"

for img_id, img in img_map.items():
    src = os.path.join(raw_img_dir, img["file_name"])
    dst = os.path.join(img_out, img["file_name"])
    
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    shutil.copy(src, dst)


for ann in data["annotations"]:
    img_id = ann["image_id"]
    img_info = img_map[img_id]

    img_name = os.path.splitext(img_info["file_name"])[0]

    # [9] Mapping lại nhãn
    cat_name = idtn[ann["category_id"]]
    label = mapping.get(cat_name, "Other")
    class_id = class2id[label]

    x, y, w, h = ann["bbox"]

    img_w, img_h = img_info["width"], img_info["height"]

    # [8] Chuyển annotation (định dạng COCO) sang format YOLO
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    
    txt_path = os.path.join(label_out, f"{img_name}.txt")

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, "a") as f:
        f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")


yaml_path = os.path.join(base_out, "dataset.yaml")

with open(yaml_path, "w") as f:
    f.write(f"""
path: {base_out}
train: images/train
val: images/val

names:
""")
    for i, c in enumerate(classes):
        f.write(f"  {i}: {c}\n")

print("Đã xử lý xong.")



"""

[5] 
shutil.rmtree(base_out) có tác dụng reset tránh giữ lại dữ liệu cũ gấy sai lệch kết quả

[7]
Copy sang để tránh thay đổi dataset, cũng như là tách biệt dữ liệu gốc với dữ liêuj tiền xử lý.


[9]
cat_name = idtn[ann["category_id"]]
label = mapping.get(cat_name, "Other")
class_id = class2id[label]
    
Trong đó:
ann["category_id"] lấy mã lớp của object hiện tại từ annotation.

idtn: id to name
idtn = {c["id"]: c["name"] for c in data["categories"]} : lấy tên lớp thật tương ứng với category_id của object hiện tại.

label = mapping.get(cat_name, "Other")
nếu tìm thấy class trong mapping thì lấy lớp tương ứng
Nếu không tìm thấy thì mặc định gán vào lớp "Other"

class_id = class2id[label] chuyển nhãn dán (label) về id tương ứng do YOLO không đọc chữ, mà là nhận số nguyên
"""

# TODO: thống kê số mẫu mỗi lớp
# train/val split
# baseline model