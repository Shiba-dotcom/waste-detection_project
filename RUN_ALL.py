"""
RUN_ALL.py - Chay toan bo pipeline Tuan 2 theo thu tu
Nhom 2 - Waste Detection (TACO Dataset)

Cach dung:
    python RUN_ALL.py
"""
import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))

steps = [
    ("EDA & Data Cleaning",     os.path.join(BASE, "notebooks", "01_EDA.py")),
    ("Baseline YOLOv8n",        os.path.join(BASE, "notebooks", "02_Baseline.py")),
    ("Training History Charts", os.path.join(BASE, "notebooks", "03_Results_Charts.py")),
]

for name, script in steps:
    print(f"\n{'='*60}")
    print(f"  BUOC: {name}")
    print(f"{'='*60}")
    ret = subprocess.run([sys.executable, script], cwd=BASE)
    if ret.returncode != 0:
        print(f"[LOI] {name} that bai. Dung pipeline.")
        sys.exit(1)

print("\n[HOAN TAT] Toan bo pipeline Tuan 2 da chay thanh cong!")
print("Ket qua luu trong: results/")
print("  - bieu_do_1_phan_bo_lop.png       (EDA: Phan bo lop)")
print("  - bieu_do_2_dien_tich_bbox.png    (EDA: Dien tich bbox)")
print("  - bieu_do_3_kich_thuoc_anh.png    (EDA: Kich thuoc anh)")
print("  - bieu_do_4_sample_images.png     (EDA: Mau anh voi bbox)")
print("  - bieu_do_5_training_history.png  (Baseline: mAP/P/R vs Epoch)")
print("  - bieu_do_6_loss_curves.png       (Baseline: Loss curves)")
print("  - thong_ke_dataset.csv            (Thong ke dataset)")
print("  - ket_qua_baseline.csv            (Ket qua metrics)")
