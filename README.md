# Vehicle Classification System with InceptionV3

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)

Hệ thống phân loại phương tiện từ ảnh tĩnh với 5 lớp, dùng InceptionV3 và giao diện Streamlit chạy local.

A 5-class vehicle image classification system using InceptionV3 with a local Streamlit interface.

## 📋 Mục lục / Table of Contents

- [✨ Tính năng / Features](#-tính-năng--features)
- [💻 Yêu cầu hệ thống / Requirements](#-yêu-cầu-hệ-thống--requirements)
- [🚀 Cài đặt / Installation](#-cài-đặt--installation)
- [📖 Cách sử dụng / Usage](#-cách-sử-dụng--usage)
- [🧠 Mô hình và dữ liệu / Model & Dataset](#-mô-hình-và-dữ-liệu--model--dataset)
- [🧪 Kiểm thử chất lượng / QA](#-kiểm-thử-chất-lượng--qa)
- [📊 Kết quả hiện tại / Current Metrics](#-kết-quả-hiện-tại--current-metrics)
- [🐳 Docker](#-docker)
- [📁 Cấu trúc thư mục / Project Structure](#-cấu-trúc-thư-mục--project-structure)
- [🛠️ Xử lý sự cố / Troubleshooting](#️-xử-lý-sự-cố--troubleshooting)
- [🤝 Đóng góp / Contributing](#-đóng-góp--contributing)
- [📝 Giấy phép / License](#-giấy-phép--license)
- [📧 Liên hệ / Contact](#-liên-hệ--contact)
- [🌟 Acknowledgments](#-acknowledgments)

## ✨ Tính năng / Features

- 🚗 Phân loại 5 lớp phương tiện: bicycle, bus, car, motorcycle, truck
- 🧠 Mô hình InceptionV3 fine-tuned cho bài toán phân loại ảnh giao thông
- 🌐 Web app Streamlit upload ảnh và dự đoán trực tiếp
- 🧪 Có script đánh giá model và smoke test web app
- 🐳 Hỗ trợ chạy local và Docker

## 💻 Yêu cầu hệ thống / Requirements

- Python 3.11+
- pip
- Docker Desktop (tuỳ chọn, nếu chạy container)
- PowerShell/Bash

## 🚀 Cài đặt / Installation

1. Clone repository

```powershell
git clone https://github.com/Hieu-Techie/Vehicle-Classification-System---InceptionV3.git
cd "Vehicle Classification System - InceptionV3"
```

2. Tạo môi trường ảo

```powershell
python -m venv venv
```

3. Kích hoạt môi trường ảo

```powershell
# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

4. Cài thư viện

```powershell
pip install -r requirements.txt
```

## 📖 Cách sử dụng / Usage

1. Đảm bảo bạn đã có:
- Model ở đường dẫn: models/final/inception_v3_final.h5
- Dataset ở đường dẫn: data/processed/

2. Chạy ứng dụng

```powershell
streamlit run app.py
```

3. Mở trình duyệt tại:
- http://localhost:8501

4. Upload ảnh định dạng jpg, jpeg hoặc png để dự đoán lớp phương tiện.

## 🧠 Mô hình và dữ liệu / Model & Dataset

### Trọng số mô hình / Trained Weights

Vì file mô hình nặng nên không lưu trực tiếp trong repository.

Tải file model tại release:
- https://github.com/Hieu-Techie/Vehicle-Classification-System---InceptionV3/releases/download/v1.0.0/inception_v3_final.h5

Sau đó đặt vào:
- models/final/inception_v3_final.h5

### Dataset nguồn / Dataset Source

Kaggle dataset:
- https://kaggle.com/datasets/8160678ef05f80141b0a318b74b345852ff1db0be8095541e2c6305c04c14ca2

Tải bằng Kaggle CLI:

```powershell
pip install kaggle
kaggle datasets download -d 8160678ef05f80141b0a318b74b345852ff1db0be8095541e2c6305c04c14ca2 -p data
Expand-Archive -Path data\*.zip -DestinationPath data -Force
```

Cấu trúc bắt buộc:

```text
data/processed/
|- train/bicycle|bus|car|motorcycle|truck
|- val/bicycle|bus|car|motorcycle|truck
|- test/bicycle|bus|car|motorcycle|truck
```

## 🧪 Kiểm thử chất lượng / QA

Đánh giá model:

```powershell
python scripts/qa/evaluate_model.py
```

Kết quả lưu tại:
- models/logs/evaluation_results.json
- models/logs/confusion_matrix.csv
- models/logs/classification_report.md

Smoke test web app:

```powershell
python scripts/qa/webapp_smoke_test.py
```

Kết quả lưu tại:
- models/logs/webapp_smoke_test_report.json

## 📊 Kết quả hiện tại / Current Metrics

- Test accuracy: 0.9975
- Test samples: 1972

## 🐳 Docker

Build image:

```powershell
docker build -t vehicle-classification-system:local .
```

Run container:

```powershell
docker run --rm -p 8501:8501 -v "${PWD}/models:/app/models:ro" vehicle-classification-system:local
```

Run với Docker Compose:

```powershell
docker compose up --build
```

## 📁 Cấu trúc thư mục / Project Structure

```text
.
|- app.py
|- requirements.txt
|- requirements.docker.txt
|- Dockerfile
|- docker-compose.yml
|- src/
|  |- data/
|  |- models/
|  |- utils/
|- scripts/
|  |- qa/
|     |- evaluate_model.py
|     |- webapp_smoke_test.py
|- models/
|  |- checkpoints/
|  |- final/
|  |- logs/
|- data/
|  |- processed/
|- MODEL_CARD.md
```

## 🛠️ Xử lý sự cố / Troubleshooting

- Model không load: kiểm tra đúng tên và vị trí file model trong models/final.
- App không chạy: đảm bảo đã kích hoạt môi trường ảo và cài đủ dependencies.
- Docker lỗi kết nối: bật Docker Desktop trước khi build/run.
- Upload thất bại: chỉ hỗ trợ jpg, jpeg, png và dung lượng ảnh trong giới hạn app.

## 🤝 Đóng góp / Contributing

Hiện tại dự án chưa nhận pull request trực tiếp để đảm bảo ổn định bản phát hành.

Bạn vẫn có thể hỗ trợ bằng cách:
- Mở Issue để báo lỗi hoặc đề xuất tính năng.
- Mô tả rõ bước tái hiện, log lỗi và ảnh minh họa (nếu có).
- Theo dõi các bản cập nhật tiếp theo khi repo mở lại quy trình đóng góp.

## 📝 Giấy phép / License

Dự án sử dụng giấy phép MIT.

Xem chi tiết tại file [LICENSE](LICENSE).

## 📧 Liên hệ / Contact

Nếu bạn có câu hỏi hoặc góp ý, vui lòng mở Issue tại:
- https://github.com/Hieu-Techie/Vehicle-Classification-System---InceptionV3/issues

## 🌟 Acknowledgments

- TensorFlow Documentation: https://www.tensorflow.org/learn
- Streamlit Documentation: https://docs.streamlit.io/
- Kaggle Datasets: https://www.kaggle.com/datasets

