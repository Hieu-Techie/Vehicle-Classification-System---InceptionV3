"""
Ứng dụng Web Phân loại Phương tiện Giao thông Việt Nam.

Xây dựng bằng Streamlit + TensorFlow/Keras (InceptionV3).
Upload ảnh → tiền xử lý → dự đoán → hiển thị kết quả + biểu đồ.

Cách chạy:
    streamlit run app.py
"""

import os
import sys
import json

import streamlit as st
from PIL import Image

# Thêm thư mục gốc vào sys.path để import được src.*
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import CLASS_MAPPING, CLASS_NAMES, CONFIDENCE_THRESHOLD, LOGS_DIR
from src.models.predictor import du_doan, kiem_tra_file_hop_le, tai_model

# ─── Cấu hình trang ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phân loại Xe Việt Nam",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _doc_eval_results() -> dict | None:
    """Đọc kết quả đánh giá model từ file evaluation_results.json."""
    eval_path = os.path.join(LOGS_DIR, "evaluation_results.json")
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _hien_thi_sidebar() -> None:
    """Hiển thị thông tin dự án và kết quả training ở sidebar."""
    with st.sidebar:
        st.title("ℹ️ Thông tin Dự án")
        st.markdown("---")

        # Thông tin mô hình
        st.subheader("🤖 Mô hình")
        st.markdown(
            "- **Kiến trúc**: InceptionV3  \n"
            "- **Transfer Learning**: ImageNet  \n"
            "- **Fine-tuning**: 2 giai đoạn  \n"
            "- **Input**: 299 × 299 × 3  "
        )

        # Kết quả training
        st.subheader("📊 Kết quả Training")
        eval_data = _doc_eval_results()
        if eval_data:
            test_acc = eval_data.get("test_accuracy", 0) * 100
            st.metric("Test Accuracy", f"{test_acc:.1f}%")

            # F1-score từng class
            report = eval_data.get("classification_report", {})
            if report:
                st.markdown("**F1-score theo class:**")
                for class_folder in CLASS_NAMES:
                    if class_folder in report:
                        f1 = report[class_folder]["f1-score"] * 100
                        name_vi = next(
                            (
                                info["name_vi"]
                                for info in CLASS_MAPPING.values()
                                if info["folder"] == class_folder
                            ),
                            class_folder,
                        )
                        st.markdown(f"&nbsp;&nbsp;• {name_vi}: **{f1:.1f}%**")
        else:
            st.caption("Chưa có dữ liệu đánh giá.")

        st.markdown("---")

        # Danh sách 5 loại xe
        st.subheader("🏷️ 5 Loại xe")
        for info in CLASS_MAPPING.values():
            st.markdown(f"• **{info['name_vi']}** ({info['name_en']})")

        st.markdown("---")
        st.caption("Vehicle Classification System v1.0  \nModel: InceptionV3 · TF 2.15")


def _ve_bieu_do_xac_suat(all_probabilities: dict, predicted_class: str) -> None:
    """
    Vẽ horizontal bar chart hiển thị xác suất của cả 5 classes.

    Cột dự đoán được tô màu xanh lá, các cột còn lại màu xanh dương nhạt.

    Tham số:
        all_probabilities: dict {class_folder: probability}
        predicted_class:   folder name của class được dự đoán
    """
    import pandas as pd
    import altair as alt

    # Xây dựng DataFrame cho biểu đồ
    rows = []
    for class_folder, prob in all_probabilities.items():
        name_vi = next(
            (
                info["name_vi"]
                for info in CLASS_MAPPING.values()
                if info["folder"] == class_folder
            ),
            class_folder,
        )
        rows.append(
            {
                "label": name_vi,
                "prob": round(prob * 100, 1),
                "is_top": class_folder == predicted_class,
            }
        )

    df = pd.DataFrame(rows)

    # Horizontal bar chart bằng Altair
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X(
                "prob:Q",
                title="Xác suất (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y(
                "label:N",
                title="",
                sort=alt.SortField("prob", order="descending"),
            ),
            color=alt.condition(
                alt.datum.is_top,
                alt.value("#00b894"),   # xanh lá — class dự đoán
                alt.value("#74b9ff"),   # xanh dương nhạt — các class còn lại
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Loại xe"),
                alt.Tooltip("prob:Q", title="Xác suất (%)", format=".1f"),
            ],
        )
        .properties(height=alt.Step(45))
    )

    st.altair_chart(chart, use_container_width=True)


# ─── Main App ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Hàm chính của ứng dụng Streamlit."""

    # Sidebar
    _hien_thi_sidebar()

    # Tải model (có caching — chỉ load 1 lần)
    model = tai_model()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🚗 Phân loại Phương tiện Giao thông Việt Nam")
    st.markdown(
        "Tải lên ảnh một phương tiện, hệ thống sẽ phân loại vào **5 nhóm**: "
        "Xe đạp &nbsp;·&nbsp; Xe buýt &nbsp;·&nbsp; Xe ô tô con &nbsp;·&nbsp; Xe máy &nbsp;·&nbsp; Xe tải"
    )
    st.markdown("---")

    # ── Kiểm tra model đã tải thành công chưa ────────────────────────────────
    if model is None:
        st.error(
            "⚠️ Không thể tải model. Hãy đảm bảo file "
            "`models/final/inception_v3_final.h5` tồn tại và không bị lỗi."
        )
        st.stop()

    # ── Upload ảnh ────────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📁 Chọn ảnh phương tiện",
        type=["jpg", "jpeg", "png"],
        help="Hỗ trợ JPG, JPEG, PNG · Tối đa 10 MB",
    )

    if uploaded_file is None:
        st.info("👆 Hãy tải lên một ảnh để bắt đầu phân loại.")
        return

    # ── Validate file ─────────────────────────────────────────────────────────
    is_valid, error_msg = kiem_tra_file_hop_le(
        uploaded_file.name,
        uploaded_file.size,
    )
    if not is_valid:
        st.error(f"❌ {error_msg}")
        return

    # ── Đọc ảnh từ file upload ────────────────────────────────────────────────
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"❌ Không thể đọc file ảnh: {e}")
        return

    # ── Layout 2 cột: ảnh bên trái | kết quả bên phải ────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("🖼️ Ảnh đã tải lên")
        st.image(image, use_column_width=True, caption=uploaded_file.name)

    with col_result:
        st.subheader("🔍 Kết quả Phân loại")

        # Nút phân loại
        phan_loai_clicked = st.button(
            "🚀 Phân loại ngay",
            type="primary",
            use_container_width=True,
        )

        if not phan_loai_clicked:
            st.caption("Nhấn nút bên trên để thực hiện phân loại.")
            return

        # ── Thực hiện dự đoán ─────────────────────────────────────────────
        with st.spinner("Đang phân tích ảnh..."):
            try:
                ket_qua = du_doan(model, image)
            except Exception as e:
                st.error(f"❌ Lỗi khi phân loại: {e}")
                return

        # ── Hiển thị kết quả chính ────────────────────────────────────────
        confidence_pct = ket_qua["confidence"] * 100

        if ket_qua["is_confident"]:
            st.success("✅ Kết quả dự đoán")
        else:
            st.warning(
                f"⚠️ Độ tự tin thấp (dưới {CONFIDENCE_THRESHOLD * 100:.0f}%) "
                "— kết quả có thể chưa chính xác"
            )

        st.metric(
            label="Loại phương tiện",
            value=ket_qua["class_name_vi"],
            delta=f"{ket_qua['class_name_en']} · {confidence_pct:.1f}%",
        )

        # ── Biểu đồ xác suất 5 classes ───────────────────────────────────
        st.markdown("**Xác suất các loại xe:**")
        _ve_bieu_do_xac_suat(ket_qua["all_probabilities"], ket_qua["class_folder"])


if __name__ == "__main__":
    main()
