"""Phase 4 model evaluation script.

Runs evaluation on data/processed/test and writes:
- models/logs/evaluation_results.json
- models/logs/confusion_matrix.csv
- models/logs/classification_report.md
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure project root imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (  # noqa: E402
    CLASS_MAPPING,
    CLASS_NAMES,
    INPUT_SHAPE,
    LOGS_DIR,
    TEST_DIR,
    FINAL_MODEL_DIR,
    FINAL_MODEL_FILENAME,
)
from src.models.predictor import tien_xu_ly_anh  # noqa: E402


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1_score: float
    support: int


def load_model():
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    model_path = Path(FINAL_MODEL_DIR) / FINAL_MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    base_model = InceptionV3(weights=None, include_top=False, input_shape=INPUT_SHAPE)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.load_weights(str(model_path))
    return model


def collect_test_images() -> list[tuple[int, Path]]:
    test_root = Path(TEST_DIR)
    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    samples: list[tuple[int, Path]] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = test_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory in test set: {class_dir}")

        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in exts:
                samples.append((class_idx, image_path))

    if not samples:
        raise RuntimeError("No test images found.")
    return samples


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def compute_class_metrics(cm: np.ndarray) -> dict[str, ClassMetrics]:
    result: dict[str, ClassMetrics] = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        support = int(cm[idx, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        result[class_name] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            support=support,
        )
    return result


def write_confusion_matrix_csv(cm: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + CLASS_NAMES)
        for i, class_name in enumerate(CLASS_NAMES):
            writer.writerow([class_name] + cm[i].tolist())


def write_classification_report_md(
    class_metrics: dict[str, ClassMetrics],
    accuracy: float,
    macro: ClassMetrics,
    weighted: ClassMetrics,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Classification Report\n\n")
        f.write("| Class | Precision | Recall | F1-score | Support |\n")
        f.write("|-------|-----------|--------|----------|---------|\n")
        for class_name in CLASS_NAMES:
            m = class_metrics[class_name]
            f.write(
                f"| {class_name} | {m.precision:.4f} | {m.recall:.4f} | {m.f1_score:.4f} | {m.support} |\n"
            )

        total_support = sum(v.support for v in class_metrics.values())
        f.write(
            f"| accuracy | - | - | {accuracy:.4f} | {total_support} |\n"
            f"| macro avg | {macro.precision:.4f} | {macro.recall:.4f} | {macro.f1_score:.4f} | {macro.support} |\n"
            f"| weighted avg | {weighted.precision:.4f} | {weighted.recall:.4f} | {weighted.f1_score:.4f} | {weighted.support} |\n"
        )


def main() -> None:
    logs_dir = Path(LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()
    samples = collect_test_images()

    y_true: list[int] = []
    y_pred: list[int] = []

    for true_idx, image_path in samples:
        with Image.open(image_path) as img:
            input_arr = tien_xu_ly_anh(img)
        probs = model.predict(input_arr, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        y_true.append(true_idx)
        y_pred.append(pred_idx)

    cm = confusion_matrix(y_true, y_pred, len(CLASS_NAMES))
    class_metrics = compute_class_metrics(cm)

    accuracy = float((np.array(y_true) == np.array(y_pred)).mean())
    macro_precision = float(np.mean([m.precision for m in class_metrics.values()]))
    macro_recall = float(np.mean([m.recall for m in class_metrics.values()]))
    macro_f1 = float(np.mean([m.f1_score for m in class_metrics.values()]))

    supports = np.array([m.support for m in class_metrics.values()], dtype=float)
    total_support = float(supports.sum())
    if total_support > 0:
        weighted_precision = float(np.average([m.precision for m in class_metrics.values()], weights=supports))
        weighted_recall = float(np.average([m.recall for m in class_metrics.values()], weights=supports))
        weighted_f1 = float(np.average([m.f1_score for m in class_metrics.values()], weights=supports))
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    macro = ClassMetrics(macro_precision, macro_recall, macro_f1, int(total_support))
    weighted = ClassMetrics(weighted_precision, weighted_recall, weighted_f1, int(total_support))

    report = {
        "test_accuracy": accuracy,
        "num_test_samples": int(total_support),
        "class_names": CLASS_NAMES,
        "classification_report": {
            c: {
                "precision": class_metrics[c].precision,
                "recall": class_metrics[c].recall,
                "f1-score": class_metrics[c].f1_score,
                "support": class_metrics[c].support,
                "name_vi": next(v["name_vi"] for v in CLASS_MAPPING.values() if v["folder"] == c),
            }
            for c in CLASS_NAMES
        },
        "macro avg": {
            "precision": macro.precision,
            "recall": macro.recall,
            "f1-score": macro.f1_score,
            "support": macro.support,
        },
        "weighted avg": {
            "precision": weighted.precision,
            "recall": weighted.recall,
            "f1-score": weighted.f1_score,
            "support": weighted.support,
        },
        "confusion_matrix": cm.tolist(),
    }

    with (logs_dir / "evaluation_results.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    write_confusion_matrix_csv(cm, logs_dir / "confusion_matrix.csv")
    write_classification_report_md(class_metrics, accuracy, macro, weighted, logs_dir / "classification_report.md")

    print(f"Evaluation done. Test samples: {int(total_support)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved: {logs_dir / 'evaluation_results.json'}")
    print(f"Saved: {logs_dir / 'confusion_matrix.csv'}")
    print(f"Saved: {logs_dir / 'classification_report.md'}")


if __name__ == "__main__":
    main()
