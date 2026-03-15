# Hệ thống Phân loại Phương tiện Việt Nam (InceptionV3)

Ứng dụng Streamlit local để phân loại ảnh phương tiện với 5 lớp:
- bicycle
- bus
- car
- motorcycle
- truck

## 1. Cấu trúc dự án

```text
.
|- app.py
|- requirements.txt
|- data/
|  |- processed/
|     |- train/<class_name>/
|     |- val/<class_name>/
|     |- test/<class_name>/
|- models/
|  |- checkpoints/inception_v3_best.h5
|  |- final/inception_v3_final.h5
|  |- logs/
|- src/
|  |- data/
|  |- models/
|  |- utils/
|- scripts/
|  |- qa/evaluate_model.py
|  |- qa/webapp_smoke_test.py
|- MODEL_CARD.md
```

## 2. Cài đặt local

### Yêu cầu trước
- Python 3.11+
- Windows PowerShell (hoặc shell tương đương)

### Tạo và kích hoạt môi trường ảo

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Cài dependency

```powershell
pip install -r requirements.txt
```

## 3. Chạy ứng dụng

```powershell
streamlit run app.py
```

Địa chỉ mặc định: http://localhost:8501

## 4. Yêu cầu về model và dữ liệu

### 🚀 Cài đặt Mô hình (Bắt buộc)
Vì file trọng số (weights) của AI khá nặng nên không được lưu trực tiếp trong mã nguồn. Để chạy được ứng dụng, bạn vui lòng làm theo 2 bước sau:
1. Tải file mô hình `inception_v3_final.h5` tại mục **[Releases](https://github.com/Hieu-Techie/Vehicle-Classification-System---InceptionV3/releases/download/v1.0.0/inception_v3_final.h5)** của dự án này.
2. Tạo thư mục `models/final/` trong thư mục gốc của dự án (nếu chưa có).
3. Copy file vừa tải về bỏ vào đường dẫn chuẩn: `models/final/inception_v3_final.h5`.

Ứng dụng cần:
- Final model tại `models/final/inception_v3_final.h5`
- Cấu hình 5 class trong `src/utils/config.py`
- Log đánh giá trong `models/logs/`

Nguồn dataset (Kaggle):
- https://kaggle.com/datasets/8160678ef05f80141b0a318b74b345852ff1db0be8095541e2c6305c04c14ca2

Lưu ý:
- Thư mục `data/` được loại khỏi GitHub để repo nhẹ hơn.
- Tải dataset từ Kaggle và đặt vào `data/processed/` khi chạy local.

### Cách tải dataset từ Kaggle

1. Mở trang dataset và lấy dataset slug:
	- https://kaggle.com/datasets/8160678ef05f80141b0a318b74b345852ff1db0be8095541e2c6305c04c14ca2
2. Cài Kaggle CLI và đặt API credential (`kaggle.json`) vào user profile.
3. Tải và giải nén dataset vào thư mục `data/` của dự án.

```powershell
pip install kaggle
kaggle datasets download -d 8160678ef05f80141b0a318b74b345852ff1db0be8095541e2c6305c04c14ca2 -p data
Expand-Archive -Path data\*.zip -DestinationPath data -Force
```

4. Đảm bảo cấu trúc sau tồn tại trước khi chạy app:
	- `data/processed/train/<class_name>/`
	- `data/processed/val/<class_name>/`
	- `data/processed/test/<class_name>/`

Cấu trúc dataset cần dùng:

```text
data/processed/
|- train/bicycle|bus|car|motorcycle|truck
|- val/bicycle|bus|car|motorcycle|truck
|- test/bicycle|bus|car|motorcycle|truck
```

## 5. Lệnh kiểm tra chất lượng (QA)

Phần này gồm các lệnh dùng để kiểm tra chất lượng model và độ ổn định của ứng dụng trước khi phát hành.

Chạy đánh giá model:

```powershell
python scripts/qa/evaluate_model.py
```

File đầu ra:
- `models/logs/evaluation_results.json`
- `models/logs/confusion_matrix.csv`
- `models/logs/classification_report.md`

Chạy smoke test web app:

```powershell
python scripts/qa/webapp_smoke_test.py
```

File đầu ra:
- `models/logs/webapp_smoke_test_report.json`

## 6. Snapshot metric hiện tại

Theo file mới nhất `models/logs/evaluation_results.json`:
- Test accuracy: 0.9975
- Test samples: 1972

## 7. Docker

### Build image

Docker Desktop (hoặc Docker daemon tương đương) cần được bật trước khi build/run.

```powershell
docker build -t vehicle-classification-system:local .
```

### Chạy container

Container cần mount thư mục `models/` local để app load:
- `models/final/inception_v3_final.h5`
- `models/logs/evaluation_results.json`

```powershell
docker run --rm -p 8501:8501 -v "${PWD}/models:/app/models:ro" vehicle-classification-system:local
```

### Chạy bằng Docker Compose

```powershell
docker compose up --build
```

## 8. Sẵn sàng để đẩy lên GitHub

- Xem `docs/release_checklist.md` trước khi push.
- Không đưa `venv/`, model weights và cache local vào git.
- Nên xóa cache source như `__pycache__/` trước khi public.
- Dữ liệu được host riêng trên Kaggle, vì vậy toàn bộ `data/` đã được ignore trong `.gitignore`.

## 9. Xử lý sự cố

- Nếu model không load được, kiểm tra file `models/final/inception_v3_final.h5`.
- Nếu gặp lỗi TensorFlow, đảm bảo đã cài `tensorflow==2.15.0`.
- Nếu upload không hợp lệ, định dạng hỗ trợ là `.jpg`, `.jpeg`, `.png` và giới hạn 10 MB.

