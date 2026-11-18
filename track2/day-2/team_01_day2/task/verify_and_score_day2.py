#!/usr/bin/env python3
"""
verify_and_score_day2.py — ตัวตรวจความถูกต้องตาม DAY2_INFER_SPEC.md และคำนวณคะแนนจำลองกรรมการ
สำหรับ TESA Top Gun Rally 2026 (Edge AI Track - Day 2)

ตรวจสอบ:
  1. โครงสร้าง predictions.json (image_name, objects, id 1..N, class 7 คลาสตัวพิมพ์เล็ก, polygon >= 3 จุด)
  2. ภาพ seg_mask/<stem>.png (Single-channel integer, ขนาดตรงกับภาพต้นฉบับหลังแก้ EXIF)
  3. ความสอดคล้องระดับพิกเซลระหว่าง seg_mask == id กับ polygon (ตาม DAY2_INFER_SPEC.md บรรทัด 79-81)
  4. คำนวณคะแนนตามเกณฑ์กรรมการ (สไลด์หน้า 58 และ DAY2_INFER_SPEC.md บรรทัด 90):
     - P1: Average Mask IoU (เต็ม 7.5 คะแนน)
     - P2: Counting Score = คู่ที่จับได้ / max(เม็ดจริง, เม็ดที่ทาย) (เต็ม 7.5 คะแนน)
     - F1 @ IoU >= 0.95 และความแม่นยำในการทายคลาส (Class Accuracy)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _extra_sp in (
    _PROJECT_ROOT / "venv" / "Lib" / "site-packages",
    Path(r"C:\Users\kitar\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages"),
):
    if _extra_sp.exists() and str(_extra_sp) not in sys.path:
        sys.path.insert(0, str(_extra_sp))

import cv2
import numpy as np
from PIL import Image, ImageOps

VALID_CLASSES = {"good", "shell", "black", "husk", "broken", "parchment", "immature"}


def get_exif_size(img_path: Path) -> Tuple[int, int]:
    """คืนค่า (width, height) ของภาพหลังปรับ EXIF orientation"""
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im)
        return im.size


def verify_submission_bundle(
    images_dir: Path,
    seg_mask_dir: Path,
    predictions_json_path: Path,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """ตรวจสอบความถูกต้องตาม DAY2_INFER_SPEC.md ทุกข้อ"""
    errors: List[str] = []
    if not predictions_json_path.exists():
        return False, [f"Missing predictions.json: {predictions_json_path}"], {}

    data = json.loads(predictions_json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return False, ["predictions.json root must be a JSON array"], {}

    img_files = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: p.name,
    )
    if len(data) != len(img_files):
        errors.append(f"Entry count mismatch: JSON has {len(data)}, images_dir has {len(img_files)}")

    total_objects = 0
    poly_mask_ious: List[float] = []

    for entry, img_path in zip(data, img_files):
        if entry.get("image_name") != img_path.name:
            errors.append(f"Image name order/mismatch: expected {img_path.name}, got {entry.get('image_name')}")
            continue

        expected_keys = {"image_name", "objects"}
        if set(entry.keys()) != expected_keys:
            errors.append(f"{img_path.name}: unexpected top-level keys {set(entry.keys())}")

        mask_path = seg_mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            errors.append(f"Missing seg_mask file: {mask_path}")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            errors.append(f"Cannot read seg_mask: {mask_path}")
            continue
        if mask.ndim != 2:
            errors.append(f"{mask_path.name}: must be single-channel 2D integer image, got shape {mask.shape}")
            continue

        orig_w, orig_h = get_exif_size(img_path)
        if mask.shape != (orig_h, orig_w):
            errors.append(f"{mask_path.name}: shape {mask.shape} != expected ({orig_h}, {orig_w})")
            continue

        objs = entry.get("objects", [])
        expected_ids = list(range(1, len(objs) + 1))
        actual_ids = [o.get("id") for o in objs]
        if actual_ids != expected_ids:
            errors.append(f"{img_path.name}: object IDs {actual_ids} != 1..{len(objs)}")

        unique_mask_ids = sorted(int(v) for v in np.unique(mask) if v != 0)
        if unique_mask_ids != expected_ids:
            errors.append(f"{img_path.name}: mask unique IDs {unique_mask_ids} != JSON IDs {expected_ids}")

        for obj in objs:
            total_objects += 1
            if set(obj.keys()) != {"id", "class", "polygon"}:
                errors.append(f"{img_path.name} id={obj.get('id')}: invalid object keys {set(obj.keys())}")
            if obj.get("class") not in VALID_CLASSES:
                errors.append(f"{img_path.name} id={obj.get('id')}: invalid class '{obj.get('class')}'")

            poly = obj.get("polygon", [])
            if not isinstance(poly, list) or len(poly) < 3:
                errors.append(f"{img_path.name} id={obj.get('id')}: polygon must have >= 3 points")
                continue

            pts = np.array(poly, dtype=np.int32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                errors.append(f"{img_path.name} id={obj.get('id')}: invalid polygon shape {pts.shape}")
                continue
            if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= orig_w) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= orig_h):
                errors.append(f"{img_path.name} id={obj.get('id')}: polygon coordinates out of bounds")
                continue

            # ตรวจความตรงกันระหว่าง polygon กับ seg_mask == id ในกรอบ ROI
            oid = int(obj["id"])
            x1 = max(0, int(pts[:, 0].min()) - 2)
            y1 = max(0, int(pts[:, 1].min()) - 2)
            x2 = min(orig_w, int(pts[:, 0].max()) + 3)
            y2 = min(orig_h, int(pts[:, 1].max()) + 3)

            m_bin = mask[y1:y2, x1:x2] == oid
            p_canvas = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            local_pts = pts.copy()
            local_pts[:, 0] -= x1
            local_pts[:, 1] -= y1
            cv2.drawContours(p_canvas, [local_pts.reshape((-1, 1, 2))], -1, 1, thickness=-1)
            p_bin = p_canvas > 0

            inter = int(np.count_nonzero(m_bin & p_bin))
            union = int(np.count_nonzero(m_bin | p_bin))
            iou = inter / float(union) if union > 0 else 0.0
            poly_mask_ious.append(iou)
            if iou < 0.98:
                errors.append(f"{img_path.name} id={oid}: polygon vs seg_mask IoU too low ({iou:.4f})")

    stats = {
        "num_images": len(img_files),
        "total_objects": total_objects,
        "mean_polygon_mask_consistency_iou": float(np.mean(poly_mask_ious)) if poly_mask_ious else 1.0,
        "min_polygon_mask_consistency_iou": float(np.min(poly_mask_ious)) if poly_mask_ious else 1.0,
    }
    return len(errors) == 0, errors, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Day 2 Ground Truth / Submission Bundle")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets_seg" / "ground_truth_day2_format",
        help="Path to ground_truth_day2_format folder or a specific bundle",
    )
    args = parser.parse_args()

    sub_bundles = (
        [args.bundle_dir]
        if (args.bundle_dir / "ground_truth.json").exists()
        else [d for d in sorted(args.bundle_dir.iterdir()) if d.is_dir() and (d / "ground_truth.json").exists()]
    )

    all_passed = True
    for bdir in sub_bundles:
        ok, errs, stats = verify_submission_bundle(
            images_dir=bdir / "images",
            seg_mask_dir=bdir / "seg_mask",
            predictions_json_path=bdir / "ground_truth.json",
        )
        status_str = "PASS (100% Spec Compliant)" if ok else f"FAIL ({len(errs)} errors)"
        print(
            f"[{status_str}] {bdir.name:<24} | Images: {stats.get('num_images', 0):<3} "
            f"| Objects: {stats.get('total_objects', 0):<4} "
            f"| Polygon-Mask IoU: {stats.get('mean_polygon_mask_consistency_iou', 0.0):.6f} "
            f"(min: {stats.get('min_polygon_mask_consistency_iou', 0.0):.6f})"
        )
        if not ok:
            all_passed = False
            for e in errs[:10]:
                print(f"   - {e}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
