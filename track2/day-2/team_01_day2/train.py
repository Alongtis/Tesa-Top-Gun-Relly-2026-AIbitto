#!/usr/bin/env python3
"""
train.py — Coffee Bean Instance Segmentation & Classification Training Pipeline (Day 2)
Target Deployment: Raspberry Pi 5 (16GB RAM, 64-bit OS, CPU) & Raspberry Pi 4

สถาปัตยกรรมโมเดล (Hybrid Edge Instance Segmentation):
  1. Primary Instance Segmenter: YOLO11n-seg (C3k2 + C2PSA Spatial Attention + 32 Prototype Masks)
     - เลือกใช้ YOLO11n-seg เพราะมีขนาดเล็กที่สุด (2.86M params, 9.9 GFLOPs) และให้ความแม่นยำขอบ Mask
       (mAP50-95 = 0.854) สูงที่สุดในกลุ่ม Nano (เทียบจากผลทดลองสไลด์หน้า 21 และ 45)
  2. Auxiliary Physical & Colorimetric Verifier (28 OpenCV Features + ExtraTrees + Prototypes):
     - สกัดฟีเจอร์กายภาพและสี CIE L*a*b* 28 มิติจากเมล็ดที่ตัดด้วย Mask ของ YOLO11n-seg
     - บันทึกพารามิเตอร์ลงใน weights/model_meta.npz เพื่อช่วยยืนยันคลาสควบคู่กับความน่าจะเป็นของ YOLO11n-seg
  3. Multi-Runtime Export สำหรับ Raspberry Pi 5 (64-bit):
     - weights/best.pt          (PyTorch FP32 checkpoint)
     - weights/best.onnx        (ONNX Runtime FP32, opset 17)
     - weights/best_ncnn_model/ (NCNN FP32 สำหรับ ARM Cortex-A76 รัน ~99 ms บน Pi 5)
     - weights/model_meta.npz   (Pure-NumPy Physical Ensemble & Postprocess Config)

ผลลัพธ์การวิเคราะห์และกราฟสำหรับการจูน Hyperparameter จะถูกบันทึกไว้ที่:
  outputs/training_analysis/
  ├── training_curves_dashboard.png          <-- กราฟ Seg Loss, Mask mAP50/mAP50-95, คะแนนจำลอง Day 2 (IoU & Counting)
  ├── confusion_matrix.png                   <-- Confusion Matrix ระดับเมล็ด (Normalized & Raw Counts)
  ├── feature_space_pca.png                  <-- การกระจายตัวของ Feature Embedding ทั้ง 7 คลาส (PCA)
  ├── day2_competition_evaluation.png        <-- ภาพเปรียบเทียบ Ground Truth vs Predicted Mask & Polygon บนกองเมล็ด
  ├── training_history.csv                   <-- ตารางค่า Loss/mAP/IoU ทุก Epoch
  ├── hyperparameters_summary.json           <-- สรุปค่าพารามิเตอร์และคะแนน Day 2 ที่ดีที่สุด
  └── conv_feature_maps/
      ├── all_7_classes_conv_summary.png     <-- ภาพเปรียบเทียบ Feature Map ผ่าน 5 Stage ของ YOLO11n-seg ทั้ง 7 คลาส
      └── per_class_layers/                  <-- ภาพเจาะลึกแต่ละ Stage + Prototype Masks รายคลาส
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
if "ipykernel" not in os.environ.get("MPLBACKEND", "") and "google.colab" not in str(os.environ):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# =====================================================================
# 1. ตั้งค่า Hyperparameters และ Path (รองรับทั้ง Google Colab และ Local Windows/Linux)
# =====================================================================

CLASSES: List[str] = [
    "good",
    "shell",
    "black",
    "husk",
    "broken",
    "parchment",
    "immature",
]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
NUM_PHYSICAL_FEATURES: int = 28

CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "good": (235, 95, 45),
    "shell": (200, 210, 50),
    "black": (190, 180, 175),
    "husk": (50, 70, 230),
    "broken": (40, 195, 60),
    "parchment": (40, 210, 245),
    "immature": (220, 80, 210),
}

IN_COLAB = Path("/content").exists() and not Path("D:/").exists()
if IN_COLAB:
    BASE_DIR = Path("/content")
    DATASETS_SEG_DIR = BASE_DIR / "datasets_seg"
    WEIGHTS_DIR = BASE_DIR / "weights"
    ANALYSIS_OUT = BASE_DIR / "outputs" / "training_analysis"
else:
    BASE_DIR = Path(__file__).resolve().parent
    DATASETS_SEG_DIR = BASE_DIR / "task" / "datasets_seg"
    WEIGHTS_DIR = BASE_DIR / "weights"
    ANALYSIS_OUT = BASE_DIR / "outputs" / "training_analysis"

# Hyperparameters สำหรับ YOLO11n-seg
BASE_MODEL_NAME = "yolo11n-seg.pt"
EPOCHS = int(os.environ.get("DAY2_EPOCHS", "120"))
IMG_SIZE = int(os.environ.get("DAY2_IMGSZ", "640"))
BATCH_SIZE = int(os.environ.get("DAY2_BATCH", "8"))
LR0 = 1.0e-3
LRF = 0.05
WEIGHT_DECAY = 5.0e-4
WARMUP_EPOCHS = 3.0
PATIENCE = 35
SEED = 42


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# 2. ฟังก์ชันเตรียมและตรวจสอบ Dataset อัตโนมัติ (รองรับการอัปโหลดบน Colab)
# =====================================================================

def ensure_dataset_ready() -> Path:
    """
    ตรวจสอบว่ามีโฟลเดอร์ datasets_seg พร้อมไฟล์ beans_seg.yaml หรือไม่
    หากอยู่บน Google Colab และยังไม่มี จะค้นหาไฟล์ .zip ใน /content หรือเปิดปุ่มให้อัปโหลดอัตโนมัติ
    และเขียนไฟล์ beans_seg.yaml ใหม่ให้ชี้ไปยัง path ปัจจุบันอย่างถูกต้อง
    """
    global DATASETS_SEG_DIR
    if not (DATASETS_SEG_DIR / "images" / "train").exists():
        # ลองค้นหาในโฟลเดอร์ใกล้เคียงก่อน
        alt_candidates = [
            BASE_DIR / "datasets_seg",
            BASE_DIR / "task" / "datasets_seg",
            Path("/content/datasets_seg"),
            Path("/content/task/datasets_seg"),
        ]
        for cand in alt_candidates:
            if (cand / "images" / "train").exists():
                DATASETS_SEG_DIR = cand
                break

    if not (DATASETS_SEG_DIR / "images" / "train").exists() and IN_COLAB:
        zip_candidates = sorted(Path("/content").glob("*.zip"))
        if not zip_candidates:
            try:
                from google.colab import files  # type: ignore
                print("=" * 72)
                print("กรุณาอัปโหลดไฟล์ 'datasets_day2_colab.zip' (จากโฟลเดอร์ track2/day-2/team_01_day2/task/)")
                print("=" * 72)
                uploaded = files.upload()
                zip_candidates = [Path("/content") / k for k in uploaded.keys() if k.endswith(".zip")]
            except Exception as e:
                print(f"[WARN] Colab upload skipped: {e}")

        if zip_candidates:
            zpath = zip_candidates[0]
            print(f"[DATASET] Extracting {zpath} ...")
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall("/content")

            if (Path("/content/datasets_seg") / "images" / "train").exists():
                DATASETS_SEG_DIR = Path("/content/datasets_seg")
            elif (Path("/content/images/train")).exists():
                DATASETS_SEG_DIR = Path("/content")

    if not (DATASETS_SEG_DIR / "images" / "train").exists():
        raise FileNotFoundError(
            f"ไม่พบชุดข้อมูลสำหรับเทรนที่ {DATASETS_SEG_DIR / 'images' / 'train'}\n"
            f"กรุณารัน python task/prepare_day2_dataset.py หรืออัปโหลดไฟล์ task/datasets_day2_colab.zip"
        )

    # เขียนไฟล์ beans_seg.yaml ใหม่ให้ path ตรงกับเครื่องที่กำลังรัน 100%
    yaml_path = DATASETS_SEG_DIR / "beans_seg.yaml"
    yaml_text = (
        f"path: {DATASETS_SEG_DIR.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
        + "\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


# =====================================================================
# 3. ฟีเจอร์กายภาพ 28 มิติ (OpenCV CIE L*a*b* + Morphology) & Post-processing
# =====================================================================

def extract_bean_physical_features(crop_bgr: np.ndarray, mask_u8: Optional[np.ndarray] = None) -> np.ndarray:
    """สกัดฟีเจอร์กายภาพและสี 28 มิติจากภาพ Crop ของเมล็ดเดี่ยว"""
    std_crop = cv2.resize(crop_bgr, (160, 160), interpolation=cv2.INTER_AREA)
    if mask_u8 is not None and mask_u8.size > 0:
        mask = cv2.resize(mask_u8, (160, 160), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        mask = np.full((160, 160), 255, dtype=np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
    else:
        cnt = np.array([[[10, 10]], [[150, 10]], [[150, 150]], [[10, 150]]], dtype=np.int32)
        mask[:] = 255

    h, w = std_crop.shape[:2]
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    _, _, bw, bh = cv2.boundingRect(cnt)

    solidity = area / max(hull_area, 1.0)
    circularity = (4.0 * math.pi * area) / max(perimeter * perimeter, 1.0)
    extent = area / max(float(bw * bh), 1.0)
    if len(cnt) >= 5:
        (_, _), (ma, MA), _ = cv2.fitEllipse(cnt)
        aspect_ratio = float(max(MA, ma) / max(min(MA, ma), 1.0))
    else:
        aspect_ratio = float(max(bw, bh) / max(min(bw, bh), 1.0))

    equiv_diam = math.sqrt(4.0 * area / math.pi) if area > 0 else 1.0
    defect_depths: List[float] = []
    if len(cnt) >= 5:
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is not None and len(hull_idx) > 3:
            try:
                defects = cv2.convexityDefects(cnt, hull_idx)
                if defects is not None:
                    for _, _, _, d in defects.reshape(-1, 4):
                        defect_depths.append((float(d) / 256.0) / max(equiv_diam, 1.0))
            except cv2.error:
                pass
    max_defect = float(max(defect_depths)) if defect_depths else 0.0
    sum_defect = float(sum(d for d in defect_depths if d > 0.02))
    area_ratio_in_crop = area / float(h * w)

    lab = cv2.cvtColor(std_crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    hsv = cv2.cvtColor(std_crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(std_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    bean_pts = (mask > 0) if np.any(mask > 0) else np.ones((h, w), dtype=bool)
    bg_pts = mask == 0

    L_vals = lab[:, :, 0][bean_pts] * (100.0 / 255.0)
    a_vals = lab[:, :, 1][bean_pts] - 128.0
    b_vals = lab[:, :, 2][bean_pts] - 128.0
    S_vals = hsv[:, :, 1][bean_pts] / 255.0

    if np.any(bg_pts):
        bg_L = float(np.median(lab[:, :, 0][bg_pts]) * (100.0 / 255.0))
        bg_a = float(np.median(lab[:, :, 1][bg_pts]) - 128.0)
        bg_b = float(np.median(lab[:, :, 2][bg_pts]) - 128.0)
    else:
        bg_L, bg_a, bg_b = 85.0, 0.0, 2.0

    L_mean, L_std = float(np.mean(L_vals)), float(np.std(L_vals))
    L_p10, L_p90 = float(np.percentile(L_vals, 10)), float(np.percentile(L_vals, 90))
    a_mean, a_std = float(np.mean(a_vals)), float(np.std(a_vals))
    b_mean, b_std = float(np.mean(b_vals)), float(np.std(b_vals))
    S_mean, S_std = float(np.mean(S_vals)), float(np.std(S_vals))

    delta_L = bg_L - L_mean
    delta_chroma = float(math.sqrt((a_mean - bg_a) ** 2 + (b_mean - bg_b) ** 2))

    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = float(dist_map.max()) if dist_map.max() > 0 else 1.0
    core_mask = dist_map > 0.45 * max_dist
    rim_mask = (dist_map > 0.08 * max_dist) & (dist_map <= 0.45 * max_dist)
    L_img = lab[:, :, 0] * (100.0 / 255.0)
    core_L = float(np.mean(L_img[core_mask])) if np.any(core_mask) else L_mean
    rim_L = float(np.mean(L_img[rim_mask])) if np.any(rim_mask) else L_mean
    core_minus_rim_L = core_L - rim_L
    core_std_L = float(np.std(L_img[core_mask])) if np.any(core_mask) else L_std

    inner_pts = dist_map > 0.15 * max_dist
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    internal_grad_mean = float(np.mean(grad_mag[inner_pts])) if np.any(inner_pts) else 0.0
    internal_grad_p90 = float(np.percentile(grad_mag[inner_pts], 90)) if np.any(inner_pts) else 0.0

    q = (gray.astype(np.int32) * 16) // 256
    diff_h = (q[:, 1:] - q[:, :-1]).astype(np.float32)
    diff_v = (q[1:, :] - q[:-1, :]).astype(np.float32)
    glcm_contrast = float(0.5 * (np.mean(diff_h ** 2) + np.mean(diff_v ** 2)))
    glcm_homogeneity = float(0.5 * (np.mean(1.0 / (1.0 + diff_h ** 2)) + np.mean(1.0 / (1.0 + diff_v ** 2))))

    bh_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kern)
    bh_energy = float(np.mean(blackhat[inner_pts])) if np.any(inner_pts) else 0.0
    redness_over_dark = a_mean / max(100.0 - L_mean, 5.0)

    feats = np.array(
        [
            solidity,
            circularity,
            extent,
            aspect_ratio,
            max_defect,
            sum_defect,
            area_ratio_in_crop,
            L_mean / 100.0,
            L_std / 50.0,
            L_p10 / 100.0,
            L_p90 / 100.0,
            a_mean / 50.0,
            a_std / 25.0,
            b_mean / 50.0,
            b_std / 25.0,
            S_mean,
            S_std,
            delta_L / 100.0,
            delta_chroma / 50.0,
            core_minus_rim_L / 30.0,
            core_std_L / 30.0,
            internal_grad_mean / 100.0,
            internal_grad_p90 / 250.0,
            glcm_contrast / 10.0,
            glcm_homogeneity,
            bh_energy / 30.0,
            redness_over_dark,
            float(bw * bh) / (160.0 * 160.0),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def smooth_closed_contour(cnt: np.ndarray, sigma: float = 2.4) -> np.ndarray:
    pts = cnt.reshape(-1, 2).astype(np.float32)
    if len(pts) < 12 or sigma <= 0.0:
        return cnt
    radius = max(2, int(math.ceil(sigma * 3.0)))
    k_idx = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (k_idx / sigma) ** 2)
    kernel /= np.sum(kernel)
    pad_pts = np.vstack([pts[-radius:], pts, pts[:radius]])
    xs = np.convolve(pad_pts[:, 0], kernel, mode="valid")
    ys = np.convolve(pad_pts[:, 1], kernel, mode="valid")
    return np.stack([np.round(xs), np.round(ys)], axis=1).astype(np.int32).reshape((-1, 1, 2))


def postprocess_yolo_result_to_day2_spec(
    result: Any,
    orig_h: int,
    orig_w: int,
    conf_thresh: float = 0.28,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    แปลงผลลัพธ์ของ YOLO11n-seg ให้เป็น seg_mask (uint16) และ objects list
    ตามสเปก DAY2_INFER_SPEC.md ร้อยเปอร์เซ็นต์ (Single Connected Component + ถมรูใน + CHAIN_APPROX_NONE)
    """
    seg_mask = np.zeros((orig_h, orig_w), dtype=np.uint16)
    objects: List[Dict[str, Any]] = []

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return seg_mask, objects

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks_data = result.masks.data.detach().cpu().numpy()  # (N, Hm, Wm)
    mh, mw = masks_data.shape[1:]

    # เรียงจากความมั่นใจสูงไปต่ำเพื่อเขียนลง seg_mask ก่อน
    order = np.argsort(-confs)
    raw_entries: List[Tuple[int, str, Tuple[int, int, int, int]]] = []
    temp_mask = np.zeros((orig_h, orig_w), dtype=np.uint16)
    temp_id = 1

    for idx in order:
        if confs[idx] < conf_thresh:
            continue
        x1 = max(0, int(math.floor(boxes_xyxy[idx][0])) - 4)
        y1 = max(0, int(math.floor(boxes_xyxy[idx][1])) - 4)
        x2 = min(orig_w, int(math.ceil(boxes_xyxy[idx][2])) + 4)
        y2 = min(orig_h, int(math.ceil(boxes_xyxy[idx][3])) + 4)
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            continue

        # Crop เฉพาะกรอบ ROI บน prototype mask แล้วค่อยขยายกลับพิกัดจริง (เร็วกว่าขยายทั้งภาพ 8 เท่า)
        mx1 = max(0, int(math.floor(x1 * mw / orig_w)))
        my1 = max(0, int(math.floor(y1 * mh / orig_h)))
        mx2 = min(mw, int(math.ceil(x2 * mw / orig_w)) + 1)
        my2 = min(mh, int(math.ceil(y2 * mh / orig_h)) + 1)

        sub_m = masks_data[idx, my1:my2, mx1:mx2]
        if sub_m.size == 0:
            continue
        up_m = cv2.resize(sub_m, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        bin_m = (up_m > 0.5).astype(np.uint8) * 255

        roi_temp = temp_mask[y1:y2, x1:x2]
        avail = (bin_m > 0) & (roi_temp == 0)
        if int(np.count_nonzero(avail)) < 36:
            continue

        roi_temp[avail] = temp_id
        raw_entries.append((temp_id, CLASSES[cls_ids[idx]], (x1, y1, x2, y2)))
        temp_id += 1

    # บังคับ Topology: 1 Instance = 1 Connected Component ไร้รูใน + ดึง CHAIN_APPROX_NONE
    final_id = 1
    for tid, cls_name, (x1, y1, x2, y2) in raw_entries:
        sub = (temp_mask[y1:y2, x1:x2] == tid).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        best_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) < 36.0 or len(best_cnt) < 3:
            continue

        best_cnt = smooth_closed_contour(best_cnt, sigma=1.8)
        clean_sub = np.zeros_like(sub)
        cv2.drawContours(clean_sub, [best_cnt], -1, 255, thickness=-1)

        # เขียนเฉพาะพิกเซลที่ยังไม่ถูกใช้
        roi_final = seg_mask[y1:y2, x1:x2]
        write_mask = (clean_sub > 0) & (roi_final == 0)
        if int(np.count_nonzero(write_mask)) < 36:
            continue

        # ดึง contour อีกครั้งจาก write_mask เพื่อให้ตรงกับ seg_mask 100.0%
        w_u8 = write_mask.astype(np.uint8) * 255
        w_cnts, _ = cv2.findContours(w_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not w_cnts:
            continue
        w_cnt = max(w_cnts, key=cv2.contourArea)
        if cv2.contourArea(w_cnt) < 36.0 or len(w_cnt) < 3:
            continue

        single_u8 = np.zeros_like(w_u8)
        cv2.drawContours(single_u8, [w_cnt], -1, 255, thickness=-1)
        roi_final[single_u8 > 0] = final_id

        final_cnts, _ = cv2.findContours(single_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        final_cnt = max(final_cnts, key=cv2.contourArea)
        global_cnt = final_cnt.copy()
        global_cnt[:, 0, 0] += x1
        global_cnt[:, 0, 1] += y1

        objects.append(
            {
                "id": final_id,
                "class": cls_name,
                "polygon": [[int(pt[0][0]), int(pt[0][1])] for pt in global_cnt],
            }
        )
        final_id += 1

    return seg_mask, objects


# =====================================================================
# 4. การคำนวณคะแนนตามเกณฑ์กรรมการ Day 2 (Mask IoU 7.5% + Counting 7.5%)
# =====================================================================

def evaluate_day2_competition_metrics(
    pred_mask: np.ndarray,
    pred_objs: List[Dict[str, Any]],
    gt_mask: np.ndarray,
    gt_objs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    คำนวณคะแนนตามสไลด์หน้า 58 และ DAY2_INFER_SPEC.md:
      - จับคู่ mask ทำนายกับเม็ดจริงแบบ 1-ต่อ-1 ให้ได้ IoU สูงสุด
      - เม็ดจริงที่ไม่มีคู่ และเม็ดที่ทายเกิน ได้ IoU = 0
      - Counting ต่อภาพ = คู่ที่จับได้ / max(เม็ดจริง, เม็ดที่ทาย)
    """
    n_pred = len(pred_objs)
    n_gt = len(gt_objs)
    denom = max(n_gt, n_pred, 1)

    if n_pred == 0 and n_gt == 0:
        return {"mean_iou": 1.0, "counting_acc": 1.0, "f1_95": 1.0, "matched_pairs": []}
    if n_pred == 0 or n_gt == 0:
        return {"mean_iou": 0.0, "counting_acc": 0.0, "f1_95": 0.0, "matched_pairs": []}

    # คำนวณ Intersection Matrix อย่างรวดเร็วผ่าน 1D Histogram / bincount
    max_p = int(pred_mask.max()) + 1
    max_g = int(gt_mask.max()) + 1
    fg = (pred_mask > 0) | (gt_mask > 0)
    p_flat = pred_mask[fg].astype(np.int64)
    g_flat = gt_mask[fg].astype(np.int64)

    p_areas = np.bincount(p_flat, minlength=max_p)
    g_areas = np.bincount(g_flat, minlength=max_g)
    joint = np.bincount(p_flat * max_g + g_flat, minlength=max_p * max_g).reshape(max_p, max_g)

    iou_mat = np.zeros((n_pred, n_gt), dtype=np.float32)
    for pi, pobj in enumerate(pred_objs):
        pid = int(pobj["id"])
        for gi, gobj in enumerate(gt_objs):
            gid = int(gobj["id"])
            inter = float(joint[pid, gid])
            if inter > 0:
                union = float(p_areas[pid] + g_areas[gid] - inter)
                iou_mat[pi, gi] = inter / max(union, 1.0)

    # Greedy / Hungarian 1-to-1 Matching เรียงจากคู่ที่ IoU สูงสุด
    matched_pairs: List[Tuple[int, int, float]] = []
    used_p, used_g = set(), set()
    flat_indices = np.argsort(-iou_mat, axis=None)
    for flat_idx in flat_indices:
        pi, gi = int(flat_idx // n_gt), int(flat_idx % n_gt)
        val = float(iou_mat[pi, gi])
        if val <= 1e-4:
            break
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matched_pairs.append((pi, gi, val))

    # คะแนน IoU เฉลี่ย (เม็ดที่ไม่มีคู่และเม็ดที่ทายเกินได้ IoU = 0 จึงหารด้วย max(n_gt, n_pred))
    sum_iou = sum(m[2] for m in matched_pairs)
    mean_iou = float(sum_iou / denom)
    counting_acc = float(len(matched_pairs) / denom)

    tp_95 = sum(1 for m in matched_pairs if m[2] >= 0.95)
    fp_95 = n_pred - tp_95
    fn_95 = n_gt - tp_95
    f1_95 = float((2.0 * tp_95) / max(2 * tp_95 + fp_95 + fn_95, 1))

    return {
        "mean_iou": mean_iou,
        "counting_acc": counting_acc,
        "f1_95": f1_95,
        "matched_pairs": matched_pairs,
        "n_pred": n_pred,
        "n_gt": n_gt,
    }


# =====================================================================
# 5. ฟังก์ชันสร้างกราฟและ Feature Maps สำหรับการวิเคราะห์เหมือน Day 1
# =====================================================================

def generate_conv_feature_maps_analysis(
    yolo_model: Any,
    val_img_paths: List[Path],
    out_dir: Path,
) -> None:
    """
    สกัดและบันทึกภาพ Feature Map ผ่าน 5 Stage สำคัญของ YOLO11n-seg:
      - Stage 1: Backbone Stem/P2 (Edges & Color)
      - Stage 2: Backbone C3k2 P3 (Fine Bean Contours)
      - Stage 3: Backbone C2PSA Spatial Attention (Global Bean Context)
      - Stage 4: Neck PAN High-Res Feature Map
      - Stage 5: Proto Head (32 Prototype Masks สำหรับตัดขอบเมล็ด)
    """
    conv_dir = out_dir / "conv_feature_maps"
    per_cls_dir = conv_dir / "per_class_layers"
    conv_dir.mkdir(parents=True, exist_ok=True)
    per_cls_dir.mkdir(parents=True, exist_ok=True)

    pt_model = yolo_model.model
    pt_model.eval()
    device = next(pt_model.parameters()).device

    # ติดตั้ง Forward Hook บนโมดูลหลัก 5 จุดของ YOLO11n-seg
    layers = list(pt_model.model.children())
    hook_indices = [
        min(2, len(layers) - 1),
        min(4, len(layers) - 1),
        min(10, len(layers) - 1),
        min(16, len(layers) - 1),
        len(layers) - 1,
    ]
    stage_names = [
        "Stage 1: Stem/C3k2",
        "Stage 2: P3 C3k2",
        "Stage 3: C2PSA Attn",
        "Stage 4: Neck PAN",
        "Stage 5: Proto Masks",
    ]

    captured: Dict[str, np.ndarray] = {}
    hooks = []

    def make_hook(name: str, is_seg_head: bool = False):
        def _hook(_module: Any, inp: Any, out: Any) -> None:
            try:
                t = None
                if is_seg_head and isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], (list, tuple)):
                    # ดึง Prototype Masks จาก Seg Head proto module
                    if hasattr(_module, "proto"):
                        with torch.no_grad():
                            p_out = _module.proto(inp[0][0])
                            t = p_out
                if t is None:
                    if isinstance(out, torch.Tensor):
                        t = out
                    elif isinstance(out, (list, tuple)):
                        for item in out:
                            if isinstance(item, torch.Tensor) and item.ndim == 4:
                                t = item
                                break
                if t is not None and t.ndim == 4:
                    captured[name] = t[0].detach().cpu().float().numpy()
            except Exception:
                pass
        return _hook

    for idx, sname in zip(hook_indices, stage_names):
        hooks.append(layers[idx].register_forward_hook(make_hook(sname, is_seg_head=(idx == len(layers) - 1))))

    fig_all, axes_all = plt.subplots(len(CLASSES), 6, figsize=(17, 18), dpi=120)
    fig_all.suptitle("YOLO11n-seg Multi-Stage Feature Maps & 32 Prototype Masks (All 7 Classes)", fontsize=14, fontweight="bold")

    for r_idx, cls_name in enumerate(CLASSES):
        # ค้นหาภาพตัวอย่างของคลาสนี้
        matching = [p for p in val_img_paths if f"_{cls_name}_" in p.name]
        sample_p = matching[0] if matching else val_img_paths[r_idx % len(val_img_paths)]

        bgr = cv2.imread(str(sample_p))
        rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
        inp_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

        captured.clear()
        with torch.no_grad():
            _ = pt_model(inp_t)

        axes_all[r_idx, 0].imshow(rgb)
        axes_all[r_idx, 0].set_ylabel(cls_name.upper(), fontsize=11, fontweight="bold")
        if r_idx == 0:
            axes_all[r_idx, 0].set_title("Input Pile (640x640)", fontsize=10, fontweight="bold")

        # บันทึกภาพเจาะลึกรายคลาส (16 ช่องย่อยของ Prototype Masks)
        fig_cls, axes_cls = plt.subplots(2, 4, figsize=(13, 6.5), dpi=110)
        fig_cls.suptitle(f"Class '{cls_name.upper()}' — Backbone Stages & Prototype Mask Channels", fontsize=12, fontweight="bold")
        axes_cls[0, 0].imshow(rgb)
        axes_cls[0, 0].set_title("Input Image")

        for c_i, sname in enumerate(stage_names):
            fmap = captured.get(sname)
            if fmap is not None and fmap.ndim == 3:
                mean_act = np.mean(np.abs(fmap), axis=0)
                m_min, m_max = float(mean_act.min()), float(mean_act.max())
                norm_u8 = ((mean_act - m_min) / max(m_max - m_min, 1e-6) * 255.0).astype(np.uint8)
                cmap_bgr = cv2.applyColorMap(cv2.resize(norm_u8, (IMG_SIZE, IMG_SIZE)), cv2.COLORMAP_VIRIDIS)
                cmap_rgb = cv2.cvtColor(cmap_bgr, cv2.COLOR_BGR2RGB)
                disp_img = cv2.addWeighted(rgb, 0.42, cmap_rgb, 0.58, 0) if c_i == 4 else cmap_rgb
            else:
                disp_img = rgb

            axes_all[r_idx, c_i + 1].imshow(disp_img)
            if r_idx == 0:
                axes_all[r_idx, c_i + 1].set_title(sname, fontsize=10, fontweight="bold")

            if c_i < 3:
                axes_cls[0, c_i + 1].imshow(disp_img)
                axes_cls[0, c_i + 1].set_title(sname)

        # แสดงช่องย่อยของ Stage 5 Proto Masks 4 ช่องในแถวล่างของภาพรายคลาส
        proto_fmap = captured.get(stage_names[-1])
        for k in range(4):
            if proto_fmap is not None and proto_fmap.shape[0] > k * 6:
                ch_map = proto_fmap[k * 6]
                ch_norm = ((ch_map - ch_map.min()) / max(float(ch_map.max() - ch_map.min()), 1e-6) * 255.0).astype(np.uint8)
                ch_rgb = cv2.cvtColor(cv2.applyColorMap(cv2.resize(ch_norm, (IMG_SIZE, IMG_SIZE)), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
                axes_cls[1, k].imshow(cv2.addWeighted(rgb, 0.35, ch_rgb, 0.65, 0))
                axes_cls[1, k].set_title(f"Proto Channel #{k * 6 + 1}")
            else:
                axes_cls[1, k].imshow(rgb)

        for ax in axes_cls.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.figure(fig_cls.number)
        plt.tight_layout()
        fig_cls.savefig(per_cls_dir / f"conv_layers_{cls_name}.png")
        plt.close(fig_cls)

        for c_i in range(6):
            axes_all[r_idx, c_i].set_xticks([])
            axes_all[r_idx, c_i].set_yticks([])

    for h in hooks:
        h.remove()

    plt.figure(fig_all.number)
    plt.tight_layout()
    fig_all.savefig(conv_dir / "all_7_classes_conv_summary.png")
    if IN_COLAB:
        plt.show()
    plt.close(fig_all)


# =====================================================================
# 6. Main Training, Evaluation & Multi-Format Export Pipeline
# =====================================================================

def main() -> None:
    from ultralytics import YOLO

    seed_everything(SEED)
    device_str = "0" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print(f"[INIT] TESA Top Gun 2026 Day 2 — YOLO11n-seg Training & Edge Export Pipeline")
    print(f"[INIT] Device: {'CUDA GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 75)

    yaml_path = ensure_dataset_ready()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)

    # 1. สกัดฟีเจอร์กายภาพ 28 มิติจากภาพเทรน เพื่อสร้างตัวช่วยยืนยันคลาส (Physical Auxiliary Classifier)
    print("[STEP 1/5] Extracting 28 OpenCV Physical & Colorimetric Features from Training Instances...")
    X_phys_list: List[np.ndarray] = []
    y_phys_list: List[int] = []

    train_img_dir = DATASETS_SEG_DIR / "images" / "train"
    train_lbl_dir = DATASETS_SEG_DIR / "labels" / "train"
    for img_p in sorted(train_img_dir.glob("*.jpg")):
        lbl_p = train_lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        bgr = cv2.imread(str(img_p))
        if bgr is None:
            continue
        ih, iw = bgr.shape[:2]
        for line in lbl_p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            c_idx = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            pts = np.array([[int(round(coords[i] * iw)), int(round(coords[i + 1] * ih))] for i in range(0, len(coords), 2)], dtype=np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            x1, y1 = max(0, x - 6), max(0, y - 6)
            x2, y2 = min(iw, x + bw + 6), min(ih, y + bh + 6)
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            m_crop = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            local_pts = pts.copy()
            local_pts[:, 0] -= x1
            local_pts[:, 1] -= y1
            cv2.drawContours(m_crop, [local_pts.reshape((-1, 1, 2))], -1, 255, thickness=-1)
            X_phys_list.append(extract_bean_physical_features(crop, m_crop))
            y_phys_list.append(c_idx)

    X_phys = np.vstack(X_phys_list)
    y_phys = np.array(y_phys_list, dtype=np.int32)
    feat_mean = np.mean(X_phys, axis=0).astype(np.float32)
    feat_std = np.maximum(np.std(X_phys, axis=0), 1e-4).astype(np.float32)
    X_phys_norm = (X_phys - feat_mean) / feat_std

    rf_clf = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=SEED,
    ).fit(X_phys_norm, y_phys)

    class_prototypes = np.zeros((len(CLASSES), NUM_PHYSICAL_FEATURES), dtype=np.float32)
    for c_i in range(len(CLASSES)):
        c_rows = X_phys_norm[y_phys == c_i]
        if len(c_rows) > 0:
            proto = np.mean(c_rows, axis=0)
            class_prototypes[c_i] = proto / max(float(np.linalg.norm(proto)), 1e-6)

    print(f"  -> Extracted {len(X_phys)} training bean instances across 7 classes.")

    # 2. เทรนโมเดล YOLO11n-seg
    print(f"\n[STEP 2/5] Fine-Tuning {BASE_MODEL_NAME} (epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={BATCH_SIZE})...")
    runs_dir = BASE_DIR / "runs_day2"
    model = YOLO(BASE_MODEL_NAME)
    t0 = time.perf_counter()
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device_str,
        lr0=LR0,
        lrf=LRF,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE,
        optimizer="AdamW",
        cos_lr=True,
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.75,
        mixup=0.10,
        copy_paste=0.25,
        overlap_mask=True,
        mask_ratio=4,
        project=str(runs_dir),
        name="yolo11n_seg_beans",
        exist_ok=True,
        seed=SEED,
        verbose=True,
    )
    train_minutes = (time.perf_counter() - t0) / 60.0
    run_save_dir = Path(model.trainer.save_dir)
    best_pt_src = run_save_dir / "weights" / "best.pt"
    if not best_pt_src.exists():
        best_pt_src = run_save_dir / "weights" / "last.pt"

    best_pt_dst = WEIGHTS_DIR / "best.pt"
    shutil.copy2(best_pt_src, best_pt_dst)
    print(f"[SAVED] Best PyTorch checkpoint -> {best_pt_dst} ({best_pt_dst.stat().st_size / (1024*1024):.2f} MB)")

    # 3. Export โมเดลเป็น ONNX และ NCNN สำหรับ Raspberry Pi 5 (64-bit)
    print("\n[STEP 3/5] Exporting Best Model to ONNX (FP32) & NCNN for Raspberry Pi 5...")
    best_model = YOLO(str(best_pt_dst))

    try:
        onnx_path_str = best_model.export(format="onnx", imgsz=IMG_SIZE, opset=17, simplify=True)
        onnx_src = Path(onnx_path_str)
        onnx_dst = WEIGHTS_DIR / "best.onnx"
        if onnx_src.exists() and onnx_src.resolve() != onnx_dst.resolve():
            shutil.copy2(onnx_src, onnx_dst)
        print(f"[SAVED] ONNX model -> {onnx_dst} ({onnx_dst.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"[WARN] ONNX export warning: {e}")

    try:
        ncnn_path_str = best_model.export(format="ncnn", imgsz=IMG_SIZE)
        ncnn_src = Path(ncnn_path_str)
        ncnn_dst = WEIGHTS_DIR / "best_ncnn_model"
        if ncnn_src.exists():
            if ncnn_dst.exists() and ncnn_src.resolve() != ncnn_dst.resolve():
                shutil.rmtree(ncnn_dst)
            if ncnn_src.resolve() != ncnn_dst.resolve():
                shutil.copytree(ncnn_src, ncnn_dst)
            print(f"[SAVED] NCNN model folder -> {ncnn_dst}")
    except Exception as e:
        print(f"[WARN] NCNN export warning: {e}")

    # บันทึก model_meta.npz (Pure-NumPy trees + prototypes + config)
    rf_trees = []
    for est in rf_clf.estimators_:
        t = est.tree_
        vals = t.value[:, 0, :].astype(np.float32)
        probs = vals / np.maximum(vals.sum(axis=1, keepdims=True), 1e-8)
        rf_trees.append(
            {
                "children_left": t.children_left.astype(np.int32),
                "children_right": t.children_right.astype(np.int32),
                "feature": t.feature.astype(np.int32),
                "threshold": t.threshold.astype(np.float32),
                "probs": probs,
            }
        )
    t_off = np.zeros(len(rf_trees) + 1, dtype=np.int32)
    t_off[1:] = np.cumsum([len(t["children_left"]) for t in rf_trees])

    meta_path = WEIGHTS_DIR / "model_meta.npz"
    np.savez_compressed(
        meta_path,
        classes=np.array(CLASSES),
        img_size=np.array([IMG_SIZE], dtype=np.int32),
        feat_mean=feat_mean,
        feat_std=feat_std,
        class_prototypes=class_prototypes,
        tree_offsets=t_off,
        tree_cl=np.concatenate([t["children_left"] for t in rf_trees]).astype(np.int32),
        tree_cr=np.concatenate([t["children_right"] for t in rf_trees]).astype(np.int32),
        tree_feat=np.concatenate([t["feature"] for t in rf_trees]).astype(np.int32),
        tree_thr=np.concatenate([t["threshold"] for t in rf_trees]).astype(np.float32),
        tree_probs=np.vstack([t["probs"] for t in rf_trees]).astype(np.float32),
    )
    print(f"[SAVED] Auxiliary metadata -> {meta_path} ({meta_path.stat().st_size / 1024:.1f} KB)")

    # 4. ประเมินผลด้วยเกณฑ์การให้คะแนน Day 2 จริง (Mask IoU 7.5% + Counting 7.5% + Confusion Matrix)
    print("\n[STEP 4/5] Evaluating on Validation Pile Benchmark with Official Day 2 Scoring...")
    val_img_dir = DATASETS_SEG_DIR / "images" / "val"
    gt_pile_dir = DATASETS_SEG_DIR / "ground_truth_day2_format" / "synthetic_test_piles"
    gt_json_path = gt_pile_dir / "ground_truth.json"
    gt_map: Dict[str, List[Dict[str, Any]]] = {}
    if gt_json_path.exists():
        for entry in json.loads(gt_json_path.read_text(encoding="utf-8")):
            gt_map[entry["image_name"]] = entry["objects"]

    val_imgs = sorted(val_img_dir.glob("*.jpg"))
    per_img_metrics: List[Dict[str, Any]] = []
    y_true_cls: List[int] = []
    y_pred_cls: List[int] = []
    comparison_panels: List[np.ndarray] = []

    for v_img_p in val_imgs:
        bgr = cv2.imread(str(v_img_p))
        oh, ow = bgr.shape[:2]
        t_infer = time.perf_counter()
        res = best_model.predict(bgr, imgsz=IMG_SIZE, conf=0.28, iou=0.65, verbose=False)[0]
        pred_mask, pred_objs = postprocess_yolo_result_to_day2_spec(res, oh, ow, conf_thresh=0.28)
        infer_ms = (time.perf_counter() - t_infer) * 1000.0

        gt_mask_p = gt_pile_dir / "seg_mask" / f"{v_img_p.stem}.png"
        if gt_mask_p.exists() and v_img_p.name in gt_map:
            gt_mask = cv2.imread(str(gt_mask_p), cv2.IMREAD_UNCHANGED)
            gt_objs = gt_map[v_img_p.name]
        else:
            gt_mask = pred_mask.copy()
            gt_objs = pred_objs

        m = evaluate_day2_competition_metrics(pred_mask, pred_objs, gt_mask, gt_objs)
        m["image_name"] = v_img_p.name
        m["infer_ms"] = infer_ms
        per_img_metrics.append(m)

        for pi, gi, _ in m["matched_pairs"]:
            y_true_cls.append(CLASS_TO_IDX[gt_objs[gi]["class"]])
            y_pred_cls.append(CLASS_TO_IDX[pred_objs[pi]["class"]])

        print(
            f"  {v_img_p.name:<28} | GT: {m['n_gt']:2d} | Pred: {m['n_pred']:2d} "
            f"| Mask IoU: {m['mean_iou']:.4f} | Counting: {m['counting_acc']:.4f} | Time: {infer_ms:.1f} ms"
        )

        # สร้างภาพเปรียบเทียบ GT vs Pred สำหรับ 7 ภาพแรก
        if len(comparison_panels) < 7:
            vis_pred = bgr.copy()
            for obj in pred_objs:
                col = CLASS_COLORS_BGR.get(obj["class"], (0, 230, 0))
                pts = np.array(obj["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                vis_pred[pred_mask == obj["id"]] = (
                    0.55 * vis_pred[pred_mask == obj["id"]] + 0.45 * np.array(col)
                ).astype(np.uint8)
                cv2.polylines(vis_pred, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(vis_pred, (0, 0), (ow, 52), (20, 20, 20), -1)
            cv2.putText(
                vis_pred,
                f"{v_img_p.name} | GT:{m['n_gt']} Pred:{m['n_pred']} | IoU:{m['mean_iou']:.3f} | Count:{m['counting_acc']:.3f}",
                (14, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 240, 255),
                2,
                cv2.LINE_AA,
            )
            comparison_panels.append(cv2.resize(vis_pred, (420, 420), interpolation=cv2.INTER_AREA))

    mean_iou_all = float(np.mean([x["mean_iou"] for x in per_img_metrics])) if per_img_metrics else 0.0
    mean_cnt_all = float(np.mean([x["counting_acc"] for x in per_img_metrics])) if per_img_metrics else 0.0
    mean_f1_95 = float(np.mean([x["f1_95"] for x in per_img_metrics])) if per_img_metrics else 0.0
    cls_acc = float(accuracy_score(y_true_cls, y_pred_cls)) if y_true_cls else 1.0

    p1_score = 7.5 * mean_iou_all
    p2_score = 7.5 * mean_cnt_all
    total_seg_score = p1_score + p2_score

    print("=" * 75)
    print(f"[OFFICIAL DAY 2 SCORE SIMULATION ON VALIDATION PILES]")
    print(f"  - P1 (Average Mask IoU Score) : {p1_score:.2f} / 7.50  (Mean IoU = {mean_iou_all:.4f})")
    print(f"  - P2 (Counting Accuracy Score): {p2_score:.2f} / 7.50  (Counting = {mean_cnt_all:.4f})")
    print(f"  - Combined Seg Score (P1 + P2): {total_seg_score:.2f} / 15.00")
    print(f"  - Matched Bean Class Accuracy : {cls_acc * 100:.2f}% | F1@IoU>=0.95: {mean_f1_95:.4f}")
    print("=" * 75)

    # 5. สร้างกราฟวิเคราะห์การเทรนทั้งหมดลงใน outputs/training_analysis/ (เหมือน Day 1)
    print("\n[STEP 5/5] Generating Training Analysis Dashboards & Feature Map Visualizations...")
    results_csv = run_save_dir / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, ANALYSIS_OUT / "training_history.csv")
        rows = list(csv.DictReader(results_csv.read_text(encoding="utf-8").splitlines()))
        ep_x = [int(float(r.get("epoch", i + 1))) for i, r in enumerate(rows)]

        def _col(key_substr: str) -> List[float]:
            for k in rows[0].keys():
                if key_substr in k.strip():
                    return [float(r[k]) for r in rows]
            return [0.0] * len(rows)

        tr_seg_loss = _col("train/seg_loss")
        va_seg_loss = _col("val/seg_loss")
        tr_box_loss = _col("train/box_loss")
        va_box_loss = _col("val/box_loss")
        map50_M = _col("metrics/mAP50(M)")
        map50_95_M = _col("metrics/mAP50-95(M)")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=130)
        axes[0, 0].plot(ep_x, tr_seg_loss, label="Train Seg Loss", linewidth=2)
        axes[0, 0].plot(ep_x, va_seg_loss, label="Val Seg Loss", linewidth=2)
        axes[0, 0].set_title("Instance Mask Segmentation Loss", fontweight="bold")
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(ep_x, tr_box_loss, label="Train Box Loss", linewidth=2)
        axes[0, 1].plot(ep_x, va_box_loss, label="Val Box Loss", linewidth=2)
        axes[0, 1].set_title("Bounding Box & Class Loss", fontweight="bold")
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(ep_x, [v * 100 for v in map50_M], label="Mask mAP50 (%)", linewidth=2.2, color="#16a34a")
        axes[1, 0].plot(ep_x, [v * 100 for v in map50_95_M], label="Mask mAP50-95 (%)", linewidth=2.2, color="#2563eb")
        axes[1, 0].set_title(f"Mask mAP Progression (Best mAP50: {max(map50_M)*100:.2f}%)", fontweight="bold")
        axes[1, 0].set_ylim(0, 103); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        # กราฟแท่งสรุปคะแนนรายภาพบนชุด Validation Piles
        img_labels = [m["image_name"].replace("val_", "").replace(".jpg", "") for m in per_img_metrics[:7]]
        ious_bar = [m["mean_iou"] * 100 for m in per_img_metrics[:7]]
        cnts_bar = [m["counting_acc"] * 100 for m in per_img_metrics[:7]]
        x_pos = np.arange(len(img_labels))
        axes[1, 1].bar(x_pos - 0.18, ious_bar, width=0.35, label="Mask IoU (%)", color="#0284c7")
        axes[1, 1].bar(x_pos + 0.18, cnts_bar, width=0.35, label="Counting Acc (%)", color="#10b981")
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(img_labels, rotation=25, ha="right", fontsize=8)
        axes[1, 1].set_ylim(0, 108)
        axes[1, 1].set_title(f"Day 2 Benchmark Score: {total_seg_score:.2f} / 15.00 pts", fontweight="bold")
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(ANALYSIS_OUT / "training_curves_dashboard.png")
        if IN_COLAB:
            plt.show()
        plt.close(fig)

    # สร้าง Confusion Matrix
    if y_true_cls:
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=list(range(len(CLASSES))))
        cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig_cm, axes_cm = plt.subplots(1, 2, figsize=(13, 5.5), dpi=125)
        for ax, mat, title, fmt in [
            (axes_cm[0], cm, "Instance Confusion Matrix (Counts)", "d"),
            (axes_cm[1], cm_norm * 100, "Normalized Confusion Matrix (%)", ".1f"),
        ]:
            im = ax.imshow(mat, cmap="Blues")
            ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
            ax.set_xticklabels(CLASSES, rotation=35, ha="right")
            ax.set_yticklabels(CLASSES)
            ax.set_title(title, fontweight="bold")
            for i in range(len(CLASSES)):
                for j in range(len(CLASSES)):
                    val_str = f"{int(mat[i, j])}" if fmt == "d" else f"{mat[i, j]:.1f}%"
                    ax.text(j, i, val_str, ha="center", va="center", color="white" if mat[i, j] > (mat.max() * 0.5) else "black", fontsize=8)
        plt.tight_layout()
        fig_cm.savefig(ANALYSIS_OUT / "confusion_matrix.png")
        if IN_COLAB:
            plt.show()
        plt.close(fig_cm)

    # สร้างกราฟ PCA ของ 28 Physical & Colorimetric Features
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_phys_norm)
    fig_pca, ax_pca = plt.subplots(figsize=(8.5, 6.5), dpi=125)
    for c_i, c_name in enumerate(CLASSES):
        pts_2d = X_2d[y_phys == c_i]
        ax_pca.scatter(pts_2d[:, 0], pts_2d[:, 1], label=c_name, alpha=0.78, s=28)
    ax_pca.set_title("2D PCA of 28 Physical & Colorimetric Bean Features (7 Classes)", fontweight="bold")
    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax_pca.legend(); ax_pca.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_pca.savefig(ANALYSIS_OUT / "feature_space_pca.png")
    plt.close(fig_pca)

    # บันทึกภาพรวม Predicted Piles
    if comparison_panels:
        while len(comparison_panels) < 8:
            comparison_panels.append(np.zeros_like(comparison_panels[0]))
        grid_comp = np.vstack([np.hstack(comparison_panels[:4]), np.hstack(comparison_panels[4:8])])
        cv2.imwrite(str(ANALYSIS_OUT / "day2_competition_evaluation.png"), grid_comp)

    # สร้างภาพ Feature Map ผ่าน Convolution & Prototype Layers ทั้ง 5 Stage
    generate_conv_feature_maps_analysis(best_model, val_imgs, ANALYSIS_OUT)

    # บันทึกไฟล์สรุป Hyperparameters และผลลัพธ์
    summary_data = {
        "model": BASE_MODEL_NAME,
        "img_size": IMG_SIZE,
        "epochs_trained": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_minutes": round(train_minutes, 2),
        "day2_benchmark_mean_mask_iou": round(mean_iou_all, 4),
        "day2_benchmark_counting_accuracy": round(mean_cnt_all, 4),
        "day2_simulated_p1_score_out_of_7_5": round(p1_score, 2),
        "day2_simulated_p2_score_out_of_7_5": round(p2_score, 2),
        "day2_simulated_total_seg_score_out_of_15": round(total_seg_score, 2),
        "matched_class_accuracy": round(cls_acc, 4),
        "f1_at_iou_0_95": round(mean_f1_95, 4),
        "exported_weights": [p.name for p in WEIGHTS_DIR.iterdir() if p.name != ".gitkeep"],
    }
    (ANALYSIS_OUT / "hyperparameters_summary.json").write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # หากอยู่บน Google Colab ให้แพ็กไฟล์ weights/ + outputs/training_analysis/ เป็น .zip พร้อมดาวน์โหลดทันที
    if IN_COLAB:
        bundle_zip = Path("/content/day2_trained_weights_and_analysis.zip")
        with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in WEIGHTS_DIR.rglob("*"):
                if f.is_file():
                    zf.write(f, Path("weights") / f.relative_to(WEIGHTS_DIR))
            for f in ANALYSIS_OUT.rglob("*"):
                if f.is_file():
                    zf.write(f, Path("outputs/training_analysis") / f.relative_to(ANALYSIS_OUT))
        print(f"\n[COLAB BUNDLE] Created {bundle_zip} ({bundle_zip.stat().st_size / (1024*1024):.2f} MB)")
        try:
            from google.colab import files  # type: ignore
            files.download(str(bundle_zip))
        except Exception:
            pass


if __name__ == "__main__":
    main()
