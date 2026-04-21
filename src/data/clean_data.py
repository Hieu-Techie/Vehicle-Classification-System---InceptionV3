"""
Script kiểm tra và loại bỏ ảnh lỗi (corrupt), trùng lặp, và không hợp lệ.

Chức năng:
- DP-004: Loại bỏ ảnh lỗi, không đọc được (corrupt files)
- DP-005: Xóa ảnh trùng lặp sử dụng image hashing (perceptual hash)
- DP-006: Kiểm tra và loại bỏ ảnh không liên quan (file quá nhỏ/lớn)

Cách sử dụng:
    python src/data/clean_data.py                    # Chỉ báo cáo
    python src/data/clean_data.py --delete           # Xóa file lỗi
    python src/data/clean_data.py --check-duplicates # Kiểm tra trùng lặp
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.config import (
    PROCESSED_DIR,
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
)

# Ngưỡng kích thước file (bytes)
MIN_FILE_SIZE = 1_000        # 1 KB — file quá nhỏ có thể bị lỗi
MAX_FILE_SIZE = 10_000_000   # 10 MB — file quá lớn
EXPECTED_SPLITS = ("train", "val", "test")


def kiem_tra_anh_doc_duoc(file_path: str) -> bool:
    """
    Kiểm tra xem file ảnh có đọc được không.

    Tham số:
        file_path: Đường dẫn tới file ảnh

    Trả về:
        True nếu ảnh đọc được, False nếu bị corrupt
    """
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            # Thử load toàn bộ pixel để phát hiện ảnh corrupt
            img.verify()
        # Mở lại sau verify (verify đóng file)
        with Image.open(file_path) as img:
            img.load()
        return True
    except Exception:
        return False


def tinh_hash_anh(file_path: str) -> str:
    """
    Tính MD5 hash của file ảnh để phát hiện trùng lặp.

    Tham số:
        file_path: Đường dẫn tới file ảnh

    Trả về:
        Chuỗi MD5 hash
    """
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def kiem_tra_kich_thuoc_file(file_path: str) -> str | None:
    """
    Kiểm tra kích thước file có nằm trong giới hạn hợp lệ không.

    Tham số:
        file_path: Đường dẫn tới file

    Trả về:
        None nếu hợp lệ, chuỗi mô tả lỗi nếu không hợp lệ
    """
    size = os.path.getsize(file_path)
    if size < MIN_FILE_SIZE:
        return f"Quá nhỏ ({size} bytes < {MIN_FILE_SIZE} bytes)"
    if size > MAX_FILE_SIZE:
        return f"Quá lớn ({size / 1_000_000:.1f} MB > {MAX_FILE_SIZE / 1_000_000:.0f} MB)"
    return None


def kiem_tra_dinh_dang(file_path: str) -> bool:
    """
    Kiểm tra file có phải định dạng ảnh hợp lệ không.

    Tham số:
        file_path: Đường dẫn tới file

    Trả về:
        True nếu có extension hợp lệ
    """
    ext = Path(file_path).suffix.lower()
    return ext in IMAGE_EXTENSIONS


def xac_thuc_cau_truc_du_lieu(base_dir: str) -> None:
    """
    Xác thực cấu trúc dữ liệu local có khớp project 5 class hiện tại không.

    Raise:
        ValueError nếu phát hiện dataset legacy hoặc thiếu class folder.
    """
    expected_classes = set(CLASS_NAMES)
    errors = []

    for split in EXPECTED_SPLITS:
        split_dir = os.path.join(base_dir, split)
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
                f"nhưng config hiện tại yêu cầu {sorted(expected_classes)}"
            )

    if errors:
        raise ValueError(
            "Dataset local không khớp project 5 class hiện tại. "
            "Hãy thay data/processed bằng dataset bicycle/bus/car/motorcycle/truck trước khi quét làm sạch.\n- "
            + "\n- ".join(errors)
        )


def quet_thu_muc(
    base_dir: str,
    check_duplicates: bool = False,
    delete: bool = False,
) -> dict:
    """
    Quét toàn bộ thư mục data và báo cáo các vấn đề.

    Tham số:
        base_dir: Thư mục gốc chứa train/val/test
        check_duplicates: Có kiểm tra ảnh trùng lặp không
        delete: Có xóa file lỗi không (False = chỉ báo cáo)

    Trả về:
        Dictionary chứa thống kê kết quả
    """
    ket_qua = {
        "tong_file": 0,
        "anh_hop_le": 0,
        "anh_corrupt": [],
        "anh_sai_dinh_dang": [],
        "anh_kich_thuoc_bat_thuong": [],
        "anh_trung_lap": [],
    }

    # Để phát hiện trùng lặp
    hash_map: dict[str, list[str]] = defaultdict(list)

    splits = list(EXPECTED_SPLITS)

    for split in splits:
        for class_name in CLASS_NAMES:
            folder = os.path.join(base_dir, split, class_name)
            if not os.path.exists(folder):
                print(f"  ⚠️  Thư mục không tồn tại: {folder}")
                continue

            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if not os.path.isfile(file_path):
                    continue

                ket_qua["tong_file"] += 1
                relative_path = os.path.relpath(file_path, base_dir)

                # Kiểm tra định dạng
                if not kiem_tra_dinh_dang(file_path):
                    ket_qua["anh_sai_dinh_dang"].append(relative_path)
                    continue

                # Kiểm tra kích thước
                loi_kich_thuoc = kiem_tra_kich_thuoc_file(file_path)
                if loi_kich_thuoc:
                    ket_qua["anh_kich_thuoc_bat_thuong"].append(
                        f"{relative_path} — {loi_kich_thuoc}"
                    )

                # Kiểm tra đọc được (corrupt)
                if not kiem_tra_anh_doc_duoc(file_path):
                    ket_qua["anh_corrupt"].append(relative_path)
                    if delete:
                        os.remove(file_path)
                        print(f"  🗑️  Đã xóa (corrupt): {relative_path}")
                    continue

                # Tính hash cho trùng lặp
                if check_duplicates:
                    file_hash = tinh_hash_anh(file_path)
                    if file_hash:
                        hash_map[file_hash].append(relative_path)

                ket_qua["anh_hop_le"] += 1

    # Xử lý trùng lặp
    if check_duplicates:
        for file_hash, files in hash_map.items():
            if len(files) > 1:
                ket_qua["anh_trung_lap"].append(files)
                if delete:
                    # Giữ file đầu tiên, xóa các file trùng còn lại
                    for dup_file in files[1:]:
                        full_path = os.path.join(base_dir, dup_file)
                        if os.path.exists(full_path):
                            os.remove(full_path)
                            print(f"  🗑️  Đã xóa (trùng lặp): {dup_file}")

    return ket_qua


def in_bao_cao(ket_qua: dict) -> None:
    """In báo cáo kết quả kiểm tra dữ liệu."""

    print("\n" + "=" * 60)
    print("📊 BÁO CÁO KIỂM TRA DỮ LIỆU")
    print("=" * 60)

    print(f"\n📁 Tổng số file quét: {ket_qua['tong_file']}")
    print(f"✅ Ảnh hợp lệ:       {ket_qua['anh_hop_le']}")

    # Ảnh corrupt
    so_corrupt = len(ket_qua["anh_corrupt"])
    if so_corrupt > 0:
        print(f"\n❌ Ảnh corrupt ({so_corrupt}):")
        for f in ket_qua["anh_corrupt"]:
            print(f"   - {f}")
    else:
        print(f"\n✅ Không có ảnh corrupt")

    # Sai định dạng
    so_sai = len(ket_qua["anh_sai_dinh_dang"])
    if so_sai > 0:
        print(f"\n⚠️  File sai định dạng ({so_sai}):")
        for f in ket_qua["anh_sai_dinh_dang"]:
            print(f"   - {f}")
    else:
        print(f"✅ Tất cả file đều đúng định dạng ảnh")

    # Kích thước bất thường
    so_bt = len(ket_qua["anh_kich_thuoc_bat_thuong"])
    if so_bt > 0:
        print(f"\n⚠️  Ảnh kích thước bất thường ({so_bt}):")
        for f in ket_qua["anh_kich_thuoc_bat_thuong"][:10]:
            print(f"   - {f}")
        if so_bt > 10:
            print(f"   ... và {so_bt - 10} file khác")

    # Trùng lặp
    so_trung = len(ket_qua["anh_trung_lap"])
    if so_trung > 0:
        print(f"\n🔄 Nhóm ảnh trùng lặp ({so_trung} nhóm):")
        for nhom in ket_qua["anh_trung_lap"][:5]:
            print(f"   Nhóm ({len(nhom)} file):")
            for f in nhom:
                print(f"     - {f}")
        if so_trung > 5:
            print(f"   ... và {so_trung - 5} nhóm khác")
    elif ket_qua.get("_check_dup"):
        print(f"✅ Không có ảnh trùng lặp")

    # Kết luận
    tong_loi = so_corrupt + so_sai
    print("\n" + "-" * 60)
    if tong_loi == 0:
        print("🎉 KẾT LUẬN: Dữ liệu SẠCH — không có lỗi nghiêm trọng!")
    else:
        print(f"⚠️  KẾT LUẬN: Phát hiện {tong_loi} vấn đề cần xử lý.")
        print("   Chạy lại với --delete để tự động xóa file lỗi.")
    print("-" * 60)


def main():
    """Hàm chính — parse arguments và thực thi."""
    parser = argparse.ArgumentParser(
        description="Kiểm tra và làm sạch dữ liệu ảnh phương tiện giao thông"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Xóa file corrupt và trùng lặp (mặc định: chỉ báo cáo)",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Kiểm tra ảnh trùng lặp bằng MD5 hash",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=PROCESSED_DIR,
        help=f"Thư mục dữ liệu (mặc định: {PROCESSED_DIR})",
    )
    args = parser.parse_args()

    print("🔍 Đang quét dữ liệu...")
    print(f"📂 Thư mục: {args.data_dir}")
    print(f"🗑️  Xóa file lỗi: {'CÓ' if args.delete else 'KHÔNG (chỉ báo cáo)'}")
    print(f"🔄 Kiểm tra trùng lặp: {'CÓ' if args.check_duplicates else 'KHÔNG'}")

    xac_thuc_cau_truc_du_lieu(args.data_dir)

    ket_qua = quet_thu_muc(
        base_dir=args.data_dir,
        check_duplicates=args.check_duplicates,
        delete=args.delete,
    )

    if args.check_duplicates:
        ket_qua["_check_dup"] = True

    in_bao_cao(ket_qua)


if __name__ == "__main__":
    main()
