# 📋 Kế Hoạch Tuần 2 – Nhóm 2: Waste Detection (TACO Dataset)
**Thời gian:** 11/5 – 18/5/2026 | **Mốc nộp:** 18/5/2026  

---

## 🔍 Tóm tắt tình hình hiện tại (từ báo cáo tuần 1)

| Hạng mục | Đã hoàn thành |
|---|---|
| Bộ dữ liệu | TACO dataset (~1500 ảnh, ~4784 annotations, 5 nhóm lớp) |
| Kiến trúc dự kiến | YOLOv8n / YOLOv8s (Object Detection) |
| Hướng thực hiện | 4 giai đoạn: Data Prep → Modeling → Evaluation → Integration |
| Metrics dự kiến | mAP@0.5, Precision, Recall, IoU |
| Câu hỏi nghiên cứu | Đã xác định 3 câu hỏi chính |
| Cấu trúc thư mục | **Đã tạo** (`project/data`, `notebooks`, `src`, `results`, `report`) |

---

## ⚠️ Yêu cầu bắt buộc của giáo viên (Tuần 2)

> [!IMPORTANT]
> Tuần 2 chưa cần mô hình nâng cao, chỉ cần **quy trình xử lý dữ liệu + baseline chạy được**.

> [!CAUTION]
> Các lỗi **KHÔNG được phép** (bị trừ điểm nặng):
> - Code không chạy được
> - Không đọc được dữ liệu thật bằng Python
> - Không có thống kê dữ liệu
> - Chỉ dùng Accuracy (phải dùng mAP, Precision, Recall, F1)
> - Không có baseline
> - Copy code từ mạng không giải thích
> - Không có nhận xét kết quả (tối thiểu 5 nhận xét)
> - Không ghi rõ AI hỗ trợ phần nào

---

## 📁 Bước 0 – Kiểm tra & Hoàn thiện cấu trúc thư mục

> [!NOTE]
> Cấu trúc `project/` đã có sẵn, chỉ cần tạo thêm các thư mục con và đặt dữ liệu đúng chỗ.

**Cấu trúc cần có:**
```
project/
├── data/
│   ├── raw/          ← Đặt TACO dataset gốc vào đây
│   └── processed/    ← Dữ liệu sau khi xử lý
├── notebooks/        ← Jupyter Notebook chính
├── src/              ← Code Python module
├── results/          ← Biểu đồ, bảng kết quả
└── report/           ← File báo cáo .docx
```

**Việc cần làm:**
- [ ] Tải TACO dataset về (từ http://tacodataset.org hoặc GitHub: `pedropro/TACO`)
- [ ] Đặt dữ liệu gốc vào `project/data/raw/`
- [ ] Kiểm tra file `requirements.txt` đã đủ thư viện chưa

---

## 📥 Bước 1 – Tải & Đọc bộ dữ liệu TACO bằng Python

**Thư viện cần cài:**
```
ultralytics       # YOLOv8
opencv-python     # Xử lý ảnh
Pillow            # Đọc ảnh
pycocotools       # Đọc annotation COCO format
matplotlib        # Vẽ biểu đồ
seaborn           # Visualize
pandas            # Xử lý bảng dữ liệu
numpy             # Tính toán
tqdm              # Progress bar
```

**Việc cần làm trong notebook:**
- [ ] Load file `annotations.json` (định dạng COCO) bằng `pycocotools`
- [ ] Đọc danh sách ảnh, annotation, categories
- [ ] In ra bảng thống kê cơ bản:

| Thuộc tính | Giá trị |
|---|---|
| Tên bộ dữ liệu | TACO (Trash Annotations in Context) |
| Nguồn dữ liệu | http://tacodataset.org |
| Số mẫu (ảnh) | ~1500 |
| Số annotations | ~4784 |
| Loại bài toán | Object Detection |
| Số lớp (sau gộp) | 5 (Plastic, Metal, Glass, Paper, Other) |
| Tỷ lệ các lớp | Phải tính từ dữ liệu thật |

---

## 🧹 Bước 2 – Làm sạch dữ liệu (Data Cleaning)

> [!NOTE]
> Với dữ liệu ảnh + COCO annotation, việc "làm sạch" tập trung vào kiểm tra chất lượng ảnh và annotation.

**Checklist làm sạch:**

| Loại xử lý | Hành động cụ thể |
|---|---|
| Ảnh lỗi / không mở được | Kiểm tra từng ảnh, loại bỏ ảnh corrupt |
| Kích thước ảnh không đồng nhất | Thống kê min/max/mean width, height |
| Annotation thiếu / rỗng | Kiểm tra ảnh không có annotation |
| Nhãn không thống nhất | Map 60+ lớp chi tiết → 5 nhóm lớp chính |
| Bounding box bất thường | Kiểm tra bbox âm, bbox lớn hơn ảnh |

**Code minh họa:**
```python
# Kiểm tra ảnh lỗi
from PIL import Image
broken_images = []
for img_info in coco.imgs.values():
    try:
        img = Image.open(img_path)
        img.verify()
    except:
        broken_images.append(img_info['file_name'])

# Map nhãn về 5 nhóm
CATEGORY_MAP = {
    # Plastic group
    'plastic': 'Plastic', 'bottle': 'Plastic', 'bag': 'Plastic', ...
    # Metal group  
    'can': 'Metal', 'tin': 'Metal', ...
    # Glass group
    'glass': 'Glass', ...
    # Paper group
    'paper': 'Paper', 'cardboard': 'Paper', ...
    # Other
    ...  : 'Other'
}
```

---

## 📊 Bước 3 – Phân tích dữ liệu ban đầu (EDA)

> [!IMPORTANT]
> Bắt buộc có **ít nhất 2 biểu đồ** và **nhận xét giải thích** – không được chỉ chèn ảnh không giải thích.

**Biểu đồ cần vẽ (đề xuất 3–4 biểu đồ):**

| # | Biểu đồ | Mục đích |
|---|---|---|
| 1 | Bar chart: Số annotations mỗi lớp | Phát hiện mất cân bằng dữ liệu |
| 2 | Histogram: Phân bố diện tích bbox | Phát hiện vật thể nhỏ/lớn |
| 3 | Scatter plot: Width vs Height của ảnh | Kiểm tra độ đồng nhất kích thước |
| 4 | Sample images với bbox | Trực quan hóa chất lượng dữ liệu |

**Nhận xét mẫu (tối thiểu 5 nhận xét):**
1. Phân bố nhãn giữa các lớp có cân bằng không?
2. Lớp nào chiếm đa số / thiểu số?
3. Kích thước ảnh có đồng nhất không?
4. Bbox có xu hướng nhỏ (vật thể nhỏ, khó detect)?
5. Có hiện tượng che khuất / chồng lấn annotation không?

---

## ⚙️ Bước 4 – Tiền xử lý dữ liệu (Data Preprocessing)

**Mục tiêu:** Chuẩn bị đủ để chạy baseline YOLOv8.

| Bước | Hành động |
|---|---|
| Resize ảnh | Resize về 640×640 (chuẩn YOLO), giữ tỉ lệ aspect ratio |
| Normalize | Chuẩn hóa pixel về [0, 1] |
| Chuyển format | Convert annotation COCO → YOLO format (`.txt`) |
| Chia train/val/test | Tỉ lệ 70/15/15 hoặc 80/10/10 |
| Lưu processed data | Lưu vào `project/data/processed/` |

**Cấu trúc YOLO format cần có:**
```
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

## 🤖 Bước 5 – Xây dựng Baseline

> [!NOTE]
> Baseline cho Object Detection với ảnh → theo gợi ý giáo viên: **"KNN/SVM trên ảnh đã flatten hoặc CNN rất nhỏ"**.
> Tuy nhiên với YOLO project, baseline hợp lý hơn là **YOLOv8n pretrained (zero-shot hoặc fine-tune 1–5 epoch)**.

**Lựa chọn baseline (chọn 1 trong 2):**

### Option A – Baseline đơn giản (SVM/CNN nhỏ) – Dễ giải thích
```python
# Flatten ảnh → SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report

X = [img.flatten() for img in images]  # flatten ảnh 64x64
model = SVC(kernel='rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Option B – YOLOv8n pretrained (baseline phù hợp hơn với đề tài)
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Fine-tune sơ bộ (3-5 epochs để có baseline)
results = model.train(
    data='taco.yaml',
    epochs=5,          # Chỉ 5 epoch cho baseline
    imgsz=640,
    batch=16,
    name='baseline_yolov8n'
)

# Evaluate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")
```

**→ Khuyến nghị: Dùng Option B (YOLOv8n 5 epoch) vì phù hợp đề tài Object Detection.**

---

## 📈 Bước 6 – Đánh giá kết quả Baseline

**Metrics bắt buộc phải báo cáo (đề tài Object Detection):**

| Metric | Ý nghĩa | Cách lấy |
|---|---|---|
| **mAP@0.5** | Độ chính xác tổng thể | `metrics.box.map50` |
| **Precision** | Tránh báo động giả | `metrics.box.p` |
| **Recall** | Không bỏ sót rác | `metrics.box.r` |
| **Inference time** | Tốc độ xử lý | Đo thời gian predict |
| **Confusion Matrix** | Nếu có thể | Từ validation set |

> [!WARNING]
> **KHÔNG được dùng Accuracy** như metric chính – giáo viên đã cấm rõ ràng và nhóm cũng đã giải thích lý do trong báo cáo tuần 1.

**Bảng kết quả mẫu cần có trong báo cáo:**

| Model | mAP@0.5 | Precision | Recall | Inference (ms) |
|---|---|---|---|---|
| YOLOv8n (baseline, 5 epoch) | ? | ? | ? | ? |

---

## 📝 Bước 7 – Viết Báo cáo Tuần 2 (4–6 trang)

**Cấu trúc báo cáo theo yêu cầu giáo viên:**

| # | Mục | Nội dung |
|---|---|---|
| 1 | **Tóm tắt đề tài** | Nhắc lại ngắn gọn bài toán phát hiện rác thải |
| 2 | **Mô tả bộ dữ liệu** | Số ảnh, annotation, nhãn, loại dữ liệu sau khi kiểm tra |
| 3 | **Làm sạch & tiền xử lý** | Trình bày các bước đã làm |
| 4 | **Phân tích dữ liệu ban đầu** | Bảng thống kê + biểu đồ + nhận xét |
| 5 | **Baseline** | Mô hình, cách làm, lý do chọn |
| 6 | **Kết quả ban đầu** | Bảng metric, confusion matrix nếu có |
| 7 | **Nhận xét** | Điểm tốt, điểm yếu, vấn đề phát hiện (≥5 nhận xét) |
| 8 | **Kế hoạch tuần 3** | Phương pháp chính sẽ triển khai |

**Hình thức nộp:**
- Tên file: `Nhom2_Tuan2_MaDeTai.docx`
- Thành viên nhóm: Họ tên + MSSV
- Phân công nhiệm vụ: Ai làm phần nào
- Tài liệu tham khảo: IEEE hoặc APA
- Ghi rõ AI hỗ trợ phần nào

---

## 🗓️ Phân công & Timeline (11/5 – 18/5)

| Ngày | Việc cần làm | Người thực hiện |
|---|---|---|
| 11–12/5 | Tải TACO dataset, tạo cấu trúc thư mục, cài thư viện | Cả nhóm |
| 12–13/5 | Đọc dữ liệu, kiểm tra annotation, làm sạch cơ bản | Thành viên A |
| 13–14/5 | Chuyển COCO → YOLO format, chia train/val/test | Thành viên B |
| 14–15/5 | Vẽ biểu đồ EDA, viết nhận xét phân tích | Thành viên C |
| 15–16/5 | Chạy baseline YOLOv8n (5 epoch), ghi lại metrics | Thành viên A/B |
| 16–17/5 | Viết báo cáo, tổng hợp kết quả, kiểm tra format | Cả nhóm |
| 17–18/5 | Review code chạy lại toàn bộ notebook, nộp bài | Cả nhóm |

---

## ✅ Checklist trước khi nộp

- [ ] Notebook chạy được từ đầu đến cuối không lỗi
- [ ] Có bảng thống kê dữ liệu (số ảnh, annotation, lớp, tỉ lệ)
- [ ] Có ít nhất 2 biểu đồ + nhận xét đi kèm
- [ ] Có baseline chạy được với metrics hợp lệ
- [ ] Metrics **KHÔNG** chỉ có Accuracy (phải có mAP, Precision, Recall)
- [ ] Có ít nhất 5 nhận xét kết quả
- [ ] Dữ liệu đã xử lý lưu trong `project/data/processed/`
- [ ] Biểu đồ lưu trong `project/results/`
- [ ] Báo cáo `.docx` lưu trong `project/report/`
- [ ] Ghi rõ phân công nhiệm vụ trong báo cáo
- [ ] Ghi rõ AI hỗ trợ phần nào
- [ ] Tài liệu tham khảo đúng format IEEE/APA
- [ ] Tên file đúng: `Nhom2_Tuan2_34`

---


