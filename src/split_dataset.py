import os
import random
import shutil

random.seed(42)

base = "../data/processed"

img_dir = os.path.join(base, "images")
label_dir = os.path.join(base, "labels")

train_ratio = 0.8

train_img_dir = os.path.join(img_dir, "train")
val_img_dir = os.path.join(img_dir, "val")

train_label_dir = os.path.join(label_dir, "train")
val_label_dir = os.path.join(label_dir, "val")

# Chặn split lại
if os.path.exists(train_img_dir) or os.path.exists(val_img_dir):
    print("Dataset already split.")
    exit()

# Tạo folder
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Lấy toàn bộ ảnh recursive, bỏ qua train/val
images = []

for root, dirs, files in os.walk(img_dir):
    dirs[:] = [d for d in dirs if d not in ["train", "val"]]

    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(root, file))

print(f"Found {len(images)} images")

random.shuffle(images)

split_idx = int(len(images) * train_ratio)

train_imgs = images[:split_idx]
val_imgs = images[split_idx:]


def move_files(img_list, split):
    for img_path in img_list:
        relative_path = os.path.relpath(img_path, img_dir)

        dst_img = os.path.join(img_dir, split, relative_path)
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)

        label_path = os.path.join(
            label_dir,
            os.path.splitext(relative_path)[0] + ".txt"
        )

        dst_label = os.path.join(
            label_dir,
            split,
            os.path.splitext(relative_path)[0] + ".txt"
        )
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)

        shutil.move(img_path, dst_img)

        if os.path.exists(label_path):
            shutil.move(label_path, dst_label)

move_files(train_imgs, "train")
move_files(val_imgs, "val")

print(f"Train: {len(train_imgs)} images")
print(f"Val: {len(val_imgs)} images")
print("Split completed.")