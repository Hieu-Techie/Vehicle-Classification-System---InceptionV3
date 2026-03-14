"""
Cấu hình chung cho dự án Vehicle Classification System.

Module này chứa tất cả constants, class mapping, và cấu hình
được sử dụng xuyên suốt dự án. Tránh sử dụng magic numbers.
"""

import os

# ======================== ĐƯỜNG DẪN DỰ ÁN ========================
# Thư mục gốc của dự án (2 cấp lên từ src/utils/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Đường dẫn dữ liệu
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DIR, "test")

# Đường dẫn model
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
FINAL_MODEL_DIR = os.path.join(MODELS_DIR, "final")
LOGS_DIR = os.path.join(MODELS_DIR, "logs")

# ======================== CLASS MAPPING ========================
# Mapping từ tên class sang thông tin chi tiết
# Thứ tự theo alphabetical — khớp với flow_from_directory của Keras:
#   bicycle: 0, bus: 1, car: 2, motorcycle: 3, truck: 4
NUM_CLASSES = 5
CLASS_NAMES = ["bicycle", "bus", "car", "motorcycle", "truck"]

CLASS_MAPPING = {
    0: {
        "folder": "bicycle",
        "name_vi": "Xe đạp",
        "name_en": "Bicycle",
        "description": "Xe đạp",
    },
    1: {
        "folder": "bus",
        "name_vi": "Xe buýt",
        "name_en": "Bus",
        "description": "Xe buýt",
    },
    2: {
        "folder": "car",
        "name_vi": "Xe ô tô con",
        "name_en": "Car",
        "description": "Xe ô tô con",
    },
    3: {
        "folder": "motorcycle",
        "name_vi": "Xe máy",
        "name_en": "Motorcycle",
        "description": "Xe máy",
    },
    4: {
        "folder": "truck",
        "name_vi": "Xe tải",
        "name_en": "Truck",
        "description": "Xe tải",
    },
}

# Mapping ngược: tên folder → class_id
FOLDER_TO_CLASS_ID = {info["folder"]: class_id for class_id, info in CLASS_MAPPING.items()}

# Mapping ngược: tên folder → tên tiếng Việt
FOLDER_TO_NAME_VI = {info["folder"]: info["name_vi"] for info in CLASS_MAPPING.values()}

# ======================== MÔ HÌNH ========================
IMAGE_SIZE = (299, 299)           # Kích thước chuẩn InceptionV3
INPUT_SHAPE = (299, 299, 3)       # Shape đầu vào (RGB)
BATCH_SIZE = 32                   # Batch size cho training
RANDOM_SEED = 42                  # Seed cho reproducibility

# Tên file model
BEST_MODEL_FILENAME = "inception_v3_best.h5"
FINAL_MODEL_FILENAME = "inception_v3_final.h5"
TRAINING_HISTORY_FILENAME = "training_history.json"

# ======================== TRAINING HYPERPARAMETERS ========================
# Stage 1: Freeze base, train head only
STAGE1_EPOCHS = 10
STAGE1_LEARNING_RATE = 1e-3

# Stage 2: Unfreeze top layers, fine-tune
STAGE2_EPOCHS = 40
STAGE2_LEARNING_RATE = 1e-5
STAGE2_UNFREEZE_LAYERS = 50      # Số layers mở khóa từ cuối

# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-7

# ======================== DATA AUGMENTATION ========================
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "horizontal_flip": True,
    "zoom_range": 0.2,
    "shear_range": 0.1,
    "fill_mode": "nearest",
}

# ======================== INFERENCE ========================
CONFIDENCE_THRESHOLD = 0.5        # Ngưỡng confidence để hiển thị kết quả
MAX_FILE_SIZE_MB = 10             # Giới hạn file upload (MB)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ======================== FILE FORMATS ẢNH HỢP LỆ ========================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
