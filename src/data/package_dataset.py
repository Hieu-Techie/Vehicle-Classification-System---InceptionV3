"""
Script đóng gói dữ liệu thành file zip để upload lên Kaggle.

Chức năng:
- DP-016: Zip toàn bộ data/processed/ thành dataset.zip
- DP-018: Tạo metadata file cho Kaggle Dataset

Cách sử dụng:
    python src/data/package_dataset.py                   # Zip dataset
    python src/data/package_dataset.py --with-metadata   # Zip + tạo metadata Kaggle
"""

import os
import sys
import json
import zipfile
from datetime import datetime

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.config import (
    DATA_DIR,
    PROCESSED_DIR,
    CLASS_NAMES,
    CLASS_MAPPING,
    IMAGE_EXTENSIONS,
)


EXPECTED_SPLITS = ("train", "val", "test")
CLASS_LABELS = {
    info["folder"]: info["name_vi"] for info in CLASS_MAPPING.values()
}
DEFAULT_KAGGLE_DATASET_SLUG = "hieu-techie/vietnamese-vehicle-classification"


def lay_kaggle_dataset_slug() -> str:
    """Lấy slug Kaggle Dataset từ env, có fallback mặc định."""
    explicit_slug = os.getenv("KAGGLE_DATASET_SLUG", "").strip()
    if explicit_slug:
        return explicit_slug

    kaggle_username = os.getenv("KAGGLE_USERNAME", "").strip()
    if kaggle_username:
        return f"{kaggle_username}/vietnamese-vehicle-classification"

    return DEFAULT_KAGGLE_DATASET_SLUG


def dem_anh_trong_folder(folder: str) -> int:
    """
    Đếm số file ảnh trong một thư mục.

    Tham số:
        folder: Đường dẫn thư mục

    Trả về:
        Số lượng file ảnh
    """
    if not os.path.exists(folder):
        return 0
    count = 0
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            count += 1
    return count


def xac_thuc_cau_truc_du_lieu() -> list[str]:
    """
    Xác thực data/processed có đúng cấu trúc 5 class hiện tại không.

    Trả về:
        Danh sách class hợp lệ theo thứ tự cấu hình

    Raise:
        ValueError nếu phát hiện dataset local đang là bản legacy hoặc thiếu class
    """
    expected_classes = set(CLASS_NAMES)
    errors = []

    for split in EXPECTED_SPLITS:
        split_dir = os.path.join(PROCESSED_DIR, split)
        if not os.path.isdir(split_dir):
            errors.append(f"Thiếu thư mục split: {split_dir}")
            continue

        actual_classes = {
            entry
            for entry in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, entry))
        }

        if actual_classes != expected_classes:
            errors.append(
                "Split "
                f"'{split}' có class folders {sorted(actual_classes)} "
                f"nhưng project đang yêu cầu {sorted(expected_classes)}"
            )

    if errors:
        raise ValueError(
            "Cấu trúc data/processed không khớp project 5 class hiện tại. "
            "Hãy thay dataset local bằng các folder bicycle/bus/car/motorcycle/truck trước khi package.\n- "
            + "\n- ".join(errors)
        )

    return CLASS_NAMES


def zip_dataset(output_path: str) -> int:
    """
    Nén toàn bộ thư mục processed/ thành file zip.

    Tham số:
        output_path: Đường dẫn file zip đầu ra

    Trả về:
        Tổng số file đã nén
    """
    tong_file = 0

    print(f"📦 Đang nén dữ liệu vào: {output_path}")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROCESSED_DIR):
            for filename in files:
                file_path = os.path.join(root, filename)
                # Tạo đường dẫn tương đối trong zip
                arcname = os.path.relpath(file_path, PROCESSED_DIR)
                zf.write(file_path, arcname)
                tong_file += 1

    # Kích thước file zip
    zip_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Hoàn thành! {tong_file} file → {zip_size_mb:.1f} MB")
    return tong_file


def tao_kaggle_metadata(output_dir: str) -> None:
    """
    Tạo file dataset-metadata.json cho Kaggle Datasets API.

    Tham số:
        output_dir: Thư mục chứa file metadata
    """
    # Thống kê dữ liệu
    valid_classes = xac_thuc_cau_truc_du_lieu()
    thong_ke = {}
    for split in EXPECTED_SPLITS:
        thong_ke[split] = {}
        for class_name in valid_classes:
            folder = os.path.join(PROCESSED_DIR, split, class_name)
            thong_ke[split][class_name] = dem_anh_trong_folder(folder)

    tong_anh = sum(
        count for split_data in thong_ke.values() for count in split_data.values()
    )

    # Kaggle dataset-metadata.json
    metadata = {
        "title": "Vietnamese Vehicle Classification Dataset",
        "id": lay_kaggle_dataset_slug(),
        "licenses": [{"name": "CC-BY-SA-4.0"}],
        "description": (
            "Dataset phân loại phương tiện giao thông gồm 5 lớp: "
            "bicycle (Xe đạp), bus (Xe buýt), car (Xe ô tô con), "
            "motorcycle (Xe máy), truck (Xe tải). "
            f"Tổng cộng {tong_anh} ảnh, đã chia train/val/test theo tỷ lệ 80/10/10."
        ),
        "keywords": [
            "image-classification",
            "deep-learning",
            "transfer-learning",
            "inceptionv3",
            "vehicle-classification",
            "vietnam",
            "traffic",
        ],
    }

    metadata_path = os.path.join(output_dir, "dataset-metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"📝 Đã tạo metadata: {metadata_path}")

    rows = []
    for class_name in valid_classes:
        rows.append(
            "| "
            f"{class_name} | {CLASS_LABELS.get(class_name, class_name)} | "
            f"{thong_ke['train'].get(class_name, 0)} | "
            f"{thong_ke['val'].get(class_name, 0)} | "
            f"{thong_ke['test'].get(class_name, 0)} |"
        )

    rows_text = "\n".join(rows)
    train_tree_lines = [f"│   ├── {class_name}/" for class_name in valid_classes[:-1]]
    train_tree_lines.append(f"│   └── {valid_classes[-1]}/")
    class_tree = "\n".join(train_tree_lines)

    # Tạo file README cho dataset
    readme_content = f"""# Vietnamese Vehicle Classification Dataset
## Bộ dữ liệu Phân loại Phương tiện Giao thông Việt Nam

### Thống kê
- **Tổng số ảnh**: {tong_anh}
- **Số classes**: {len(valid_classes)}
- **Ngày tạo**: {datetime.now().strftime('%Y-%m-%d')}

### Classes
| Class | Tên tiếng Việt | Train | Val | Test |
|-------|----------------|-------|-----|------|
{rows_text}

### Cấu trúc thư mục
```
processed/
├── train/
{class_tree}
├── val/
│   └── ...
└── test/
    └── ...
```

### Mục đích sử dụng
- Transfer Learning với InceptionV3
- Nghiên cứu / Học tập
- Input size: 299x299x3 (RGB)
"""
    readme_path = os.path.join(output_dir, "dataset_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"📝 Đã tạo README: {readme_path}")


def main():
    """Hàm chính — đóng gói dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Đóng gói dataset thành zip")
    parser.add_argument(
        "--with-metadata",
        action="store_true",
        help="Tạo thêm metadata cho Kaggle Datasets",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(DATA_DIR, "dataset.zip"),
        help="Đường dẫn file zip đầu ra",
    )
    args = parser.parse_args()

    xac_thuc_cau_truc_du_lieu()

    # Zip dataset
    zip_dataset(args.output)

    # Tạo metadata nếu cần
    if args.with_metadata:
        tao_kaggle_metadata(DATA_DIR)

    print("\n🎉 Đóng gói hoàn tất!")


if __name__ == "__main__":
    main()
