from __future__ import annotations

import json
import time
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictor import du_doan, kiem_tra_file_hop_le  # noqa: E402
from scripts.qa.evaluate_model import load_model  # noqa: E402
from src.utils.config import CLASS_NAMES, LOGS_DIR, TEST_DIR, MAX_FILE_SIZE_MB  # noqa: E402

 
def pick_sample_images() -> list[Path]:
    samples: list[Path] = []
    test_root = Path(TEST_DIR)

    for class_name in CLASS_NAMES:
        class_dir = test_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        first = next((p for p in sorted(class_dir.iterdir()) if p.is_file()), None)
        if first is None:
            raise RuntimeError(f"No sample image found for class {class_name}")
        samples.append(first)

    return samples


def test_valid_upload_formats() -> dict:
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    valid_cases = [
        ("sample.jpg", max_bytes // 2),
        ("sample.jpeg", max_bytes // 2),
        ("sample.png", max_bytes // 2),
    ]

    results = []
    passed = True
    for file_name, file_size in valid_cases:
        ok, err = kiem_tra_file_hop_le(file_name, file_size)
        results.append({"file": file_name, "size": file_size, "ok": ok, "error": err})
        if not ok:
            passed = False

    return {"name": "valid_upload_formats", "passed": passed, "details": results}


def test_invalid_files() -> dict:
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    invalid_cases = [
        ("sample.gif", max_bytes // 2, "invalid_extension"),
        ("sample.jpg", max_bytes + 1, "oversize"),
    ]

    results = []
    passed = True
    for file_name, file_size, expected in invalid_cases:
        ok, err = kiem_tra_file_hop_le(file_name, file_size)
        case_ok = (not ok) and bool(err)
        results.append(
            {
                "file": file_name,
                "size": file_size,
                "expected": expected,
                "ok": ok,
                "error": err,
                "case_passed": case_ok,
            }
        )
        if not case_ok:
            passed = False

    return {"name": "invalid_file_handling", "passed": passed, "details": results}


def test_inference_stability(model, runs_per_image: int = 5) -> dict:
    samples = pick_sample_images()
    run_times_ms = []
    sample_results = []
    passed = True

    for img_path in samples:
        with Image.open(img_path) as image:
            for _ in range(runs_per_image):
                start = time.perf_counter()
                pred = du_doan(model, image)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                run_times_ms.append(elapsed_ms)

            # Re-open image for deterministic single-record check.
        with Image.open(img_path) as image_check:
            pred = du_doan(model, image_check)

        probs_sum = sum(pred["all_probabilities"].values())
        structure_ok = all(
            key in pred
            for key in [
                "class_folder",
                "class_name_vi",
                "class_name_en",
                "confidence",
                "all_probabilities",
                "is_confident",
            ]
        )
        prob_ok = abs(probs_sum - 1.0) < 1e-4
        case_ok = structure_ok and prob_ok
        if not case_ok:
            passed = False

        sample_results.append(
            {
                "image": str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "predicted_class": pred["class_folder"],
                "confidence": pred["confidence"],
                "probabilities_sum": probs_sum,
                "case_passed": case_ok,
            }
        )

    avg_ms = sum(run_times_ms) / len(run_times_ms) if run_times_ms else 0.0
    max_ms = max(run_times_ms) if run_times_ms else 0.0

    return {
        "name": "inference_stability",
        "passed": passed,
        "runs_per_image": runs_per_image,
        "num_images": len(samples),
        "avg_inference_ms": avg_ms,
        "max_inference_ms": max_ms,
        "details": sample_results,
    }


def main() -> None:
    model = load_model()

    checks = [
        test_valid_upload_formats(),
        test_invalid_files(),
        test_inference_stability(model=model, runs_per_image=5),
    ]

    overall_passed = all(c["passed"] for c in checks)

    report = {
        "overall_passed": overall_passed,
        "checks": checks,
    }

    logs_dir = Path(LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = logs_dir / "webapp_smoke_test_report.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Smoke tests passed: {overall_passed}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
