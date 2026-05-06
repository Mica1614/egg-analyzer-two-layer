"""
test_api.py — Runs 5 test images per class through the Egg Analyzer API.

Layer 1 classes tested: dead, fertile, infertile
Layer 2 classes tested: initial_stage, late_stage, middle_stage  (via fertile images)

Results (annotated images) are saved automatically by the server to results/.
A summary table is printed to stdout.
"""

import os
import sys
import glob
import requests
from pathlib import Path

API_URL = "http://localhost:8000/analyze"

# ── Test image sources ────────────────────────────────────────────────────────
L1_TEST_BASE = Path(
    r"C:\Users\Administrator\Desktop\egg-cubator"
    r"\egg-analyzer-model-training\egg_analyzer_layer_one_model\split_dataset\test"
)
L2_TEST_BASE = Path(
    r"C:\Users\Administrator\Desktop\egg-cubator"
    r"\egg-analyzer-model-training\egg_analyzer_layer_two_model\split_dataset\test"
)

SAMPLES_PER_CLASS = 5

test_groups = [
    # (group_label,  folder_path)
    ("dead",          L1_TEST_BASE / "dead"),
    ("infertile",     L1_TEST_BASE / "infertile"),
    ("fertile",       L1_TEST_BASE / "fertile"),
    ("initial_stage", L2_TEST_BASE / "initial_stage"),
    ("middle_stage",  L2_TEST_BASE / "middle_stage"),
    ("late_stage",    L2_TEST_BASE / "late_stage"),
]

# ─────────────────────────────────────────────────────────────────────────────

def collect_images(folder: Path, n: int) -> list[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    imgs = []
    for ext in exts:
        imgs.extend(sorted(folder.glob(ext)))
    return imgs[:n]


def run_test(img_path: Path) -> dict | None:
    try:
        with open(img_path, "rb") as f:
            resp = requests.post(
                API_URL,
                files={"file": (img_path.name, f, "image/jpeg")},
                timeout=30,
            )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ERROR on {img_path.name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  Egg Analyzer API - Test Suite  (5 images per class)")
    print("=" * 80)

    all_passed = True

    for group_label, folder in test_groups:
        print(f"\n{'-'*70}")
        print(f"  Class: {group_label.upper()}")
        print(f"  Folder: {folder}")
        print(f"{'-'*70}")

        if not folder.exists():
            print(f"  [SKIP] Folder does not exist: {folder}")
            continue

        images = collect_images(folder, SAMPLES_PER_CLASS)
        if not images:
            print(f"  [SKIP] No images found in folder.")
            continue

        for i, img_path in enumerate(images, 1):
            result = run_test(img_path)
            if result is None:
                all_passed = False
                continue

            l1      = result["layer1_class"]
            l1_conf = result["layer1_confidence"]
            l2      = result.get("layer2_class") or "—"
            l2_conf = result.get("layer2_confidence") or 0.0
            label   = result["annotation_label"]
            ms      = result["inference_ms"]
            saved   = Path(result["result_image_path"]).name

            print(
                f"  [{i}] {img_path.name[:50]:<50}"
                f"  L1: {l1:<10} {l1_conf:5.1f}%"
                + (f"  |  L2: {l2:<15} {l2_conf:5.1f}%" if l2 != "—" else "")
                + f"  ({ms:.0f}ms)"
            )
            print(f"       Label: {label}")
            print(f"       Saved: results/{saved}")

    print(f"\n{'='*80}")
    print("  Test run complete.")
    print("  Annotated images saved in egg-analyzer/results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
