# Model Card: InceptionV3 Vehicle Classifier

## 1. Thông tin model

- Tên model: bộ phân loại phương tiện dựa trên InceptionV3
- Framework: TensorFlow / Keras
- File weights cuối: `models/final/inception_v3_final.h5`
- Mã suy luận: `src/models/predictor.py`
- Số lớp phân loại: 5

Nhãn class:
- bicycle
- bus
- car
- motorcycle
- truck

## 2. Mục đích sử dụng

Model được thiết kế cho demo suy luận local và mục đích học tập qua Streamlit.
Model trả về một trong năm nhóm phương tiện từ một ảnh đầu vào.

Không dành cho hệ thống quyết định pháp lý hoặc an toàn cao.

## 3. Tóm tắt dữ liệu huấn luyện

- Nguồn dataset (Kaggle): https://kaggle.com/datasets/8160678ef05f80141b0a318b74b345852ff1db0be8095541e2c6305c04c14ca2
- Chính sách repo: `data/` không đưa lên GitHub; dataset được tải riêng.
- Thư mục dữ liệu: `data/processed/`
- Tổng số ảnh: 19,714
- Tóm tắt split:
  - train: 15,771
  - val: 1,971
  - test: 1,972

Số lượng theo từng class:

| Class | Train | Val | Test | Total |
|-------|------:|----:|-----:|------:|
| bicycle | 3200 | 400 | 400 | 4000 |
| bus | 3200 | 400 | 400 | 4000 |
| car | 3200 | 400 | 400 | 4000 |
| motorcycle | 2971 | 371 | 372 | 3714 |
| truck | 3200 | 400 | 400 | 4000 |

## 4. Kết quả đánh giá mới nhất

Nguồn: `models/logs/evaluation_results.json` (sinh bởi `scripts/qa/evaluate_model.py`)

- Test accuracy: 0.9975
- Macro F1: 0.9975
- Weighted F1: 0.9975

Chỉ số theo từng class:

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| bicycle | 1.0000 | 0.9950 | 0.9975 | 400 |
| bus | 0.9975 | 1.0000 | 0.9988 | 400 |
| car | 0.9975 | 0.9975 | 0.9975 | 400 |
| motorcycle | 0.9947 | 1.0000 | 0.9973 | 372 |
| truck | 0.9975 | 0.9950 | 0.9962 | 400 |

Confusion matrix tại:
- `models/logs/confusion_matrix.csv`

## 5. Giới hạn

- Rủi ro domain shift: hiệu năng có thể giảm với góc chụp/thời tiết/ánh sáng khác dữ liệu train.
- Mất cân bằng dữ liệu: class motorcycle có ít mẫu hơn các class còn lại.
- Confidence từ model không phải xác suất đã được hiệu chỉnh cho bài toán rủi ro cao.

## 6. Ràng buộc vận hành

- Định dạng input: `.jpg`, `.jpeg`, `.png`
- Kích thước file tối đa: 10 MB
- Input shape: 299x299x3
- Ngưỡng confidence trong app: 0.5

## 7. Bảo trì

Khi train lại hoặc thay model, cần cập nhật:
- `models/final/inception_v3_final.h5`
- `models/logs/evaluation_results.json`
- `models/logs/confusion_matrix.csv`
- `models/logs/classification_report.md`
- File model card này
