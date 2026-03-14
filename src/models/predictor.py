"""
Module dự đoán loại phương tiện giao thông.

Cung cấp các hàm để:
- Load model InceptionV3 từ file .h5 (với caching Streamlit)
- Tiền xử lý ảnh đầu vào
- Thực hiện dự đoán và trả về kết quả có cấu trúc
"""

import os
import sys

import numpy as np
from PIL import Image
import streamlit as st

# Thêm thư mục gốc vào sys.path để import được src.*
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.config import (
    FINAL_MODEL_DIR,
    FINAL_MODEL_FILENAME,
    IMAGE_SIZE,
    INPUT_SHAPE,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_MAPPING,
    CONFIDENCE_THRESHOLD,
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
)

# Đường dẫn đầy đủ đến file model
MODEL_PATH = os.path.join(FINAL_MODEL_DIR, FINAL_MODEL_FILENAME)


@st.cache_resource(show_spinner="Đang tải mô hình InceptionV3...")
def tai_model():
    """
    Tải model InceptionV3 từ file .h5 với caching Streamlit.

    Do model được train trên Kaggle (Keras 3 / TF 2.16+) nhưng local dùng
    Keras 2 (TF 2.15), không thể load_model() trực tiếp.
    Giải pháp: rebuild kiến trúc local rồi load weights từ file .h5.

    Model chỉ được load một lần duy nhất nhờ @st.cache_resource.

    Trả về:
        Keras model đã tải, hoặc None nếu xảy ra lỗi
    """
    try:
        import tensorflow as tf  # Import muộn để tránh ảnh hưởng cold start
        from tensorflow.keras.applications import InceptionV3
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
        from tensorflow.keras.models import Model

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy file model: {MODEL_PATH}\n"
                "Hãy đảm bảo file inception_v3_final.h5 nằm trong models/final/"
            )

        # Rebuild kiến trúc model giống với notebook training
        # (Keras 3 format không tương thích với Keras 2 khi dùng load_model)
        base_model = InceptionV3(
            weights=None,
            include_top=False,
            input_shape=INPUT_SHAPE,
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(NUM_CLASSES, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=outputs)

        # Load trọng số đã train từ file .h5
        model.load_weights(MODEL_PATH)

        return model

    except Exception as e:
        st.error(f"❌ Lỗi khi tải model: {e}")
        return None


def tien_xu_ly_anh(image: Image.Image) -> np.ndarray:
    """
    Tiền xử lý ảnh đầu vào theo chuẩn InceptionV3.

    Các bước xử lý:
    1. Chuyển sang RGB (loại bỏ alpha channel nếu có)
    2. Resize về 299×299 bằng LANCZOS (chất lượng cao)
    3. Normalize pixel values từ [0, 255] về [0, 1]
    4. Thêm batch dimension

    Tham số:
        image: PIL Image object (bất kỳ mode nào)

    Trả về:
        numpy array shape (1, 299, 299, 3) dtype float32
    """
    # Chuyển sang RGB — xử lý RGBA, grayscale, palette mode
    image = image.convert("RGB")

    # Resize về kích thước chuẩn InceptionV3 (299×299)
    image = image.resize(IMAGE_SIZE, Image.LANCZOS)

    # Chuyển sang numpy array và normalize về [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Thêm batch dimension: (299, 299, 3) → (1, 299, 299, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def du_doan(model, image: Image.Image) -> dict:
    """
    Thực hiện dự đoán loại phương tiện từ ảnh.

    Thứ tự class trong output model (alphabetical theo flow_from_directory):
        index 0 → bicycle
        index 1 → bus
        index 2 → car
        index 3 → motorcycle
        index 4 → truck

    Tham số:
        model: Keras model đã được tải bằng tai_model()
        image: PIL Image object

    Trả về:
        dict với các key:
        - class_folder   (str):   tên folder class (bicycle, bus, car, motorcycle, truck)
        - class_name_vi  (str):   tên tiếng Việt
        - class_name_en  (str):   tên tiếng Anh
        - confidence     (float): độ tự tin của dự đoán [0..1]
        - all_probabilities (dict): xác suất của cả 5 classes
        - is_confident   (bool):  True nếu confidence >= CONFIDENCE_THRESHOLD

    Ngoại lệ:
        RuntimeError: khi có lỗi trong quá trình dự đoán
    """
    try:
        # Tiền xử lý ảnh
        img_array = tien_xu_ly_anh(image)

        # Chạy dự đoán — verbose=0 để không in tiến trình
        predictions = model.predict(img_array, verbose=0)[0]

        # Lấy index có xác suất cao nhất
        predicted_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        # Tra thông tin class từ CLASS_MAPPING theo folder name
        class_info = next(
            info
            for info in CLASS_MAPPING.values()
            if info["folder"] == predicted_class
        )

        # Xây dựng dict xác suất đầy đủ cho biểu đồ
        all_probs = {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        }

        return {
            "class_folder": predicted_class,
            "class_name_vi": class_info["name_vi"],
            "class_name_en": class_info["name_en"],
            "confidence": confidence,
            "all_probabilities": all_probs,
            "is_confident": confidence >= CONFIDENCE_THRESHOLD,
        }

    except Exception as e:
        raise RuntimeError(f"Lỗi khi thực hiện dự đoán: {e}") from e


def kiem_tra_file_hop_le(file_name: str, file_size_bytes: int) -> tuple[bool, str]:
    """
    Kiểm tra file upload có hợp lệ về định dạng và kích thước.

    Tham số:
        file_name:        tên file (dùng để kiểm tra phần mở rộng)
        file_size_bytes:  kích thước file tính bằng bytes

    Trả về:
        (is_valid, error_message) — error_message là chuỗi rỗng nếu hợp lệ
    """
    # Kiểm tra phần mở rộng file
    ext = os.path.splitext(file_name.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Định dạng không được hỗ trợ: '{ext}'. "
            f"Chỉ chấp nhận: JPG, JPEG, PNG"
        )

    # Kiểm tra kích thước file
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        size_mb = file_size_bytes / 1024 / 1024
        return False, (
            f"File quá lớn: {size_mb:.1f} MB "
            f"(giới hạn tối đa {MAX_FILE_SIZE_MB} MB)"
        )

    return True, ""
