#!/usr/bin/env python3
"""
train.py — Coffee Bean Defect Detection & Classification Training Pipeline (Day 1)
Target Deployment: Raspberry Pi 4 (4GB RAM, CPU) & Edge Devices

สถาปัตยกรรมโมเดล (Hybrid Dual-Branch Multimodal Network):
  1. Stage 1 (Localization): OpenCV Local Illumination-Normalized CIE L*a*b* + Watershed Bean Detector
  2. Stage 2 (Classification):
     - Branch A (Deep ConvNet): MobileNetV3-Small (5 Convolutional Stages + Squeeze-and-Excitation)
     - Branch B (Physical & Colorimetric): 28 ฟีเจอร์กายภาพจาก OpenCV (CIE L*a*b* relative to paper,
       Core-vs-Rim Cavity Ratio สำหรับแยก shell, Solidity/Convexity Defects สำหรับแยก broken,
       GLCM Texture & Redness สำหรับแยก husk vs black)
     - Fusion Head + Deep Cosine Prototype Ensemble บันทึกรวมในไฟล์ weights/best.pt ไฟล์เดียว

ผลลัพธ์การวิเคราะห์และกราฟสำหรับการจูน Hyperparameter จะถูกบันทึกไว้ที่:
  outputs/training_analysis/
  ├── training_curves_dashboard.png          <-- กราฟ Loss, Accuracy, Learning Rate, Per-Class F1
  ├── confusion_matrix.png                   <-- Confusion Matrix (Validation & Full Dataset)
  ├── feature_space_pca.png                  <-- การกระจายตัวของ Feature Embedding ทั้ง 7 คลาส (PCA)
  ├── training_history.csv                   <-- ตารางค่า Loss/Acc ทุก Epoch สำหรับนำไปเทียบจูนค่า
  ├── hyperparameters_summary.json           <-- สรุปค่าพารามิเตอร์และผลลัพธ์ที่ดีที่สุด
  └── conv_feature_maps/
      ├── all_7_classes_conv_summary.png     <-- ภาพเปรียบเทียบผลลัพธ์ผ่าน Conv Layer ทั้ง 5 Stage ของทั้ง 7 คลาส
      ├── per_class_layers/                  <-- ภาพเจาะลึกแต่ละ Conv Layer + Filter ย่อยของแต่ละคลาส
      └── epoch_progression/                 <-- ภาพวิวัฒนาการของ Conv Layer ในแต่ละช่วง Epoch ของการเทรน
"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.decomposition import PCA

# ลำดับคลาสมาตรฐานตาม INFER_SPEC.md
CLASSES = [
    "good",
    "shell",
    "black",
    "husk",
    "broken",
    "parchment",
    "immature",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_PHYSICAL_FEATURES = 28


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image_exif_corrected(img_path: Path) -> np.ndarray:
    """อ่านภาพและปรับหมุนตาม EXIF Orientation ให้ถูกต้องตามสเปก"""
    pil_img = Image.open(img_path)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# =====================================================================
# 1. OpenCV Bean Localization & Physical Feature Extraction (28 Features)
# =====================================================================

def detect_beans_in_frame(
    img_bgr: np.ndarray,
    work_scale: float = 0.25,
    pad_ratio: float = 0.08,
) -> List[Dict[str, Any]]:
    """
    ค้นหาและสกัดเมล็ดกาแฟทุกเม็ดออกจากภาพถ่ายด้วย Local Illumination Normalization + CIE L*a*b*
    รองรับทั้งภาพถ่ายเมล็ดเดียวและหลายเม็ด ทนต่อแสงเงาไล่เฉดและตัดขอบโต๊ะอัตโนมัติ
    """
    orig_h, orig_w = img_bgr.shape[:2]
    max_dim = max(orig_h, orig_w)
    if max_dim < 900:
        work_scale = 1.0
    elif max_dim < 1800:
        work_scale = 0.5

    ws_w = max(64, int(round(orig_w * work_scale)))
    ws_h = max(64, int(round(orig_h * work_scale)))
    small = cv2.resize(img_bgr, (ws_w, ws_h), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_chan, a_chan, b_chan = cv2.split(lab)

    ksize = max(31, int(round(min(ws_w, ws_h) * 0.18)) | 1)
    bg_L = cv2.GaussianBlur(L_chan, (ksize, ksize), 0)
    bg_a = cv2.GaussianBlur(a_chan, (ksize, ksize), 0)
    bg_b = cv2.GaussianBlur(b_chan, (ksize, ksize), 0)

    dark_diff = np.clip(bg_L - L_chan, 0, 255)
    chroma_diff = np.sqrt(np.maximum(0.0, (a_chan - bg_a) ** 2 + (b_chan - bg_b) ** 2))

    fg_score = dark_diff * 0.75 + chroma_diff * 1.85
    fg_score_u8 = np.clip(fg_score, 0, 255).astype(np.uint8)
    fg_score_u8 = cv2.GaussianBlur(fg_score_u8, (5, 5), 0)

    margin_x = int(ws_w * 0.05)
    margin_top = int(ws_h * 0.05)
    margin_bot = int(ws_h * 0.88)
    roi_mask = np.zeros_like(fg_score_u8)
    roi_mask[margin_top:margin_bot, margin_x : ws_w - margin_x] = 255
    fg_masked = cv2.bitwise_and(fg_score_u8, fg_score_u8, mask=roi_mask)

    otsu_val, _ = cv2.threshold(
        fg_masked[margin_top:margin_bot, margin_x : ws_w - margin_x],
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    thresh_val = max(28.0, float(otsu_val) * 1.08)
    _, binary = cv2.threshold(fg_masked, thresh_val, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    raw_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = ws_w * ws_h
    min_bean_area = img_area * 0.00045
    max_bean_area = img_area * 0.25000

    valid_areas = [cv2.contourArea(c) for c in raw_contours if min_bean_area <= cv2.contourArea(c) <= max_bean_area]
    median_area = float(np.median(valid_areas)) if valid_areas else (img_area * 0.002)

    contours = []
    for cnt in raw_contours:
        area = cv2.contourArea(cnt)
        if not (min_bean_area <= area <= max_bean_area):
            continue
        if len(valid_areas) >= 3 and area > median_area * 1.55:
            mask_single = np.zeros_like(binary)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)
            dist = cv2.distanceTransform(mask_single, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
            sure_fg = sure_fg.astype(np.uint8)
            n_labels, labels = cv2.connectedComponents(sure_fg)
            if n_labels > 2:
                markers = labels + 1
                unknown = cv2.subtract(mask_single, sure_fg)
                markers[unknown == 255] = 0
                ws_input = cv2.cvtColor(mask_single, cv2.COLOR_GRAY2BGR)
                cv2.watershed(ws_input, markers)
                for lbl in range(2, n_labels + 1):
                    sub_mask = np.uint8(markers == lbl) * 255
                    sub_cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if sub_cnts:
                        contours.append(max(sub_cnts, key=cv2.contourArea))
                continue
        contours.append(cnt)

    beans: List[Dict[str, Any]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_bean_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > 3.5:
            continue

        x1 = int(round(x / work_scale))
        y1 = int(round(y / work_scale))
        bw = int(round(w / work_scale))
        bh = int(round(h / work_scale))
        x2 = x1 + bw
        y2 = y1 + bh

        pad_x = int(round(bw * pad_ratio))
        pad_y = int(round(bh * pad_ratio))
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(orig_w, x2 + pad_x)
        py2 = min(orig_h, y2 + pad_y)

        crop_bgr = img_bgr[py1:py2, px1:px2].copy()
        if crop_bgr.size == 0:
            continue

        beans.append(
            {
                "bbox_padded": (px1, py1, px2, py2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "crop_bgr": crop_bgr,
                "rel_area": float((bw * bh) / max(orig_w * orig_h, 1)),
            }
        )

    # Fallback: หากภาพที่ส่งมาเป็นภาพที่ผู้ใช้ Crop เมล็ดมาแล้วเต็มเฟรม
    if not beans:
        beans.append(
            {
                "bbox_padded": (0, 0, orig_w, orig_h),
                "center": (orig_w / 2.0, orig_h / 2.0),
                "crop_bgr": img_bgr.copy(),
                "rel_area": 0.0035,
            }
        )

    beans.sort(key=lambda b: (int(b["center"][1] // 250), b["center"][0]))
    return beans


def segment_single_bean_crop(crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """สร้าง Mask ของเมล็ดกาแฟภายในภาพ Crop เดี่ยว (แยกเมล็ดออกจากขอบกระดาษขาว)"""
    h, w = crop_bgr.shape[:2]
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_c, a_c, b_c = cv2.split(lab)

    # เก็บค่าสีพื้นหลังบริเวณขอบทั้ง 4 ด้านของภาพ Crop
    border_pixels_L = np.concatenate([L_c[0, :], L_c[-1, :], L_c[:, 0], L_c[:, -1]])
    border_pixels_a = np.concatenate([a_c[0, :], a_c[-1, :], a_c[:, 0], a_c[:, -1]])
    border_pixels_b = np.concatenate([b_c[0, :], b_c[-1, :], b_c[:, 0], b_c[:, -1]])

    bg_L = float(np.percentile(border_pixels_L, 75))
    bg_a = float(np.median(border_pixels_a))
    bg_b = float(np.median(border_pixels_b))

    dark_diff = np.clip(bg_L - L_c, 0, 255)
    chroma_diff = np.sqrt((a_c - bg_a) ** 2 + (b_c - bg_b) ** 2)
    score = np.clip(dark_diff * 0.75 + chroma_diff * 1.85, 0, 255).astype(np.uint8)
    score = cv2.GaussianBlur(score, (5, 5), 0)

    otsu_t, _ = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(score, max(20.0, float(otsu_t) * 0.95), 255, cv2.THRESH_BINARY)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros((h, w), dtype=np.uint8)
    if cnts:
        main_cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [main_cnt], -1, 255, thickness=cv2.FILLED)
    else:
        clean_mask[:] = 255
        main_cnt = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.int32)

    return clean_mask, main_cnt


def extract_bean_physical_features(crop_bgr: np.ndarray, rel_area: float = 0.0035) -> np.ndarray:
    """
    สกัดฟีเจอร์กายภาพ 28 มิติจากภาพเมล็ดกาแฟ 1 เม็ดด้วย OpenCV:
      - รูปทรงเรขาคณิตและความเว้าแหว่ง (Solidity, Circularity, Aspect Ratio, Extent, Convexity Defects)
      - ค่าสี CIE L*a*b* และ HSV ที่ชดเชยแสงพื้นหลังกระดาษแล้ว (Relative Colorimetry)
      - อัตราส่วนความสว่าง/ขอบเขตระหว่างใจกลางเม็ด (Inner Core) กับขอบเม็ด (Outer Rim) สำหรับแยก Shell & Good
      - พื้นผิว GLCM Contrast, Homogeneity และ Black-Hat Groove Energy สำหรับแยก Husk, Black, Parchment
    """
    std_crop = cv2.resize(crop_bgr, (160, 160), interpolation=cv2.INTER_AREA)
    mask, cnt = segment_single_bean_crop(std_crop)
    h, w = std_crop.shape[:2]

    # 1. รูปทรงเรขาคณิต (7 ฟีเจอร์)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    _, _, bw, bh = cv2.boundingRect(cnt)

    solidity = area / max(hull_area, 1.0)
    circularity = (4.0 * np.pi * area) / max(perimeter * perimeter, 1.0)
    extent = area / max(float(bw * bh), 1.0)
    if len(cnt) >= 5:
        (_, _), (ma, MA), _ = cv2.fitEllipse(cnt)
        aspect_ratio = float(max(MA, ma) / max(min(MA, ma), 1.0))
    else:
        aspect_ratio = float(max(bw, bh) / max(min(bw, bh), 1.0))

    equiv_diam = np.sqrt(4.0 * area / np.pi) if area > 0 else 1.0
    defect_depths = []
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

    # 2. ค่าสี CIE L*a*b* และ HSV เทียบกับพื้นหลังกระดาษ (12 ฟีเจอร์)
    lab = cv2.cvtColor(std_crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    hsv = cv2.cvtColor(std_crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(std_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    bean_pts = mask > 0
    bg_pts = mask == 0
    if not np.any(bean_pts):
        bean_pts = np.ones((h, w), dtype=bool)

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

    L_mean = float(np.mean(L_vals))
    L_std = float(np.std(L_vals))
    L_p10 = float(np.percentile(L_vals, 10))
    L_p90 = float(np.percentile(L_vals, 90))
    a_mean = float(np.mean(a_vals))
    a_std = float(np.std(a_vals))
    b_mean = float(np.mean(b_vals))
    b_std = float(np.std(b_vals))
    S_mean = float(np.mean(S_vals))
    S_std = float(np.std(S_vals))

    delta_L = bg_L - L_mean
    delta_chroma = float(np.sqrt((a_mean - bg_a) ** 2 + (b_mean - bg_b) ** 2))

    # 3. โครงสร้างโพรงด้านใน vs ขอบเม็ด (Core vs Rim สำหรับแยก Shell, Good, Parchment) (5 ฟีเจอร์)
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = float(dist_map.max()) if dist_map.max() > 0 else 1.0
    core_mask = (dist_map > 0.45 * max_dist)
    rim_mask = (dist_map > 0.08 * max_dist) & (dist_map <= 0.45 * max_dist)

    L_img = lab[:, :, 0] * (100.0 / 255.0)
    core_L = float(np.mean(L_img[core_mask])) if np.any(core_mask) else L_mean
    rim_L = float(np.mean(L_img[rim_mask])) if np.any(rim_mask) else L_mean
    core_minus_rim_L = core_L - rim_L
    core_std_L = float(np.std(L_img[core_mask])) if np.any(core_mask) else L_std

    # ความหนาแน่นของเส้นขอบภายในเม็ด (Internal Edge & Cavity Ring Density)
    inner_mask_u8 = np.uint8(dist_map > 0.15 * max_dist) * 255
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    inner_pts = inner_mask_u8 > 0
    internal_grad_mean = float(np.mean(grad_mag[inner_pts])) if np.any(inner_pts) else 0.0
    internal_grad_p90 = float(np.percentile(grad_mag[inner_pts], 90)) if np.any(inner_pts) else 0.0

    # 4. พื้นผิว GLCM Contrast, Black-Hat Groove และสัดส่วนสีแดงต่อความมืด (4 ฟีเจอร์)
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
            rel_area * 100.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# =====================================================================
# 2. Dataset & On-the-Fly Augmentation
# =====================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_crop_to_tensor(crop_bgr: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """แปลงภาพ Crop BGR เป็น PyTorch Tensor ที่รักษาสัดส่วน (Letterbox with edge median color)"""
    h, w = crop_bgr.shape[:2]
    scale = img_size / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    # เติมขอบด้วยสีพื้นหลังจริงของภาพเพื่อรักษา Aspect Ratio ของเมล็ดกาแฟ (สำคัญมากสำหรับ broken/shell)
    border_vals = np.concatenate([resized[0, :], resized[-1, :], resized[:, 0], resized[:, -1]], axis=0)
    pad_color = [int(x) for x in np.median(border_vals, axis=0)]

    canvas = np.full((img_size, img_size, 3), pad_color, dtype=np.uint8)
    y0 = (img_size - nh) // 2
    x0 = (img_size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(norm.transpose(2, 0, 1)).float()


def augment_bean_crop_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    """สุ่มหมุน 360 องศา พลิกซ้าย-ขวา ย่อ-ขยาย และปรับแสงเงาสำหรับการเทรน"""
    img = crop_bgr.copy()
    h, w = img.shape[:2]

    border_vals = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]], axis=0)
    border_color = tuple(int(x) for x in np.median(border_vals, axis=0))

    # 1. สุ่มพลิกภาพ
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.5:
        img = cv2.flip(img, 0)

    # 2. สุ่มหมุน 0-360 องศา + สุ่มสเกล (0.88 - 1.12)
    angle = random.uniform(0.0, 360.0)
    scale = random.uniform(0.88, 1.12)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    tx = random.uniform(-0.05 * w, 0.05 * w)
    ty = random.uniform(-0.05 * h, 0.05 * h)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )

    # 3. สุ่มปรับความสว่างและคอนทราสต์เล็กน้อย (จำลองสภาพแสงที่ต่างกัน)
    if random.random() < 0.8:
        alpha = random.uniform(0.90, 1.10)  # contrast
        beta = random.uniform(-12.0, 12.0)  # brightness
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return img


class CoffeeBeanDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        img_size: int = 224,
        is_train: bool = True,
        repeats: int = 1,
    ):
        self.samples = samples
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.img_size = img_size
        self.is_train = is_train
        self.repeats = repeats if is_train else 1

    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        item = self.samples[idx % len(self.samples)]
        crop = item["crop_bgr"]
        phys = item["phys_raw"].copy()

        if self.is_train:
            crop = augment_bean_crop_bgr(crop)
            # เติม Gaussian noise เล็กน้อยให้ฟีเจอร์กายภาพเพื่อป้องกัน Overfitting
            phys_norm = (phys - self.feat_mean) / self.feat_std
            phys_norm = phys_norm + np.random.normal(0.0, 0.05, size=phys_norm.shape).astype(np.float32)
        else:
            phys_norm = (phys - self.feat_mean) / self.feat_std

        img_tensor = preprocess_crop_to_tensor(crop, self.img_size)
        phys_tensor = torch.from_numpy(phys_norm).float()
        return img_tensor, phys_tensor, int(item["label"])


# =====================================================================
# 3. Hybrid Multimodal ConvNet Architecture (MobileNetV3-Small + Physical Branch)
# =====================================================================

class HybridBeanNet(nn.Module):
    """
    โมเดลผสมผสาน Convolutional Neural Network (MobileNetV3-Small 5 Stages)
    เข้ากับ Physical & Colorimetric Feature Branch (28 OpenCV Features)
    พร้อมรองรับการดึง Feature Maps จากทุก Convolutional Stage เพื่อแสดงผลภาพ
    """

    def __init__(self, num_classes: int = 7, num_phys_features: int = NUM_PHYSICAL_FEATURES, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # 13 blocks (0..12), output = [B, 576, 7, 7]

        # กิ่งประมวลผลฟีเจอร์กายภาพจาก OpenCV (28 -> 64)
        self.phys_mlp = nn.Sequential(
            nn.Linear(num_phys_features, 64),
            nn.BatchNorm1d(64),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Hardswish(inplace=True),
        )

        # รวม Conv Global Avg+Max Pool (576) + Physical Embedding (64) = 640 -> 192 Embedding
        self.fusion_proj = nn.Sequential(
            nn.Linear(576 + 64, 192),
            nn.BatchNorm1d(192),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.30),
        )
        self.classifier = nn.Linear(192, num_classes)

    def extract_conv_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        ส่งข้อมูลภาพผ่าน Convolutional Layer ทีละ Stage และเก็บผลลัพธ์ของทุก Stage
        เพื่อใช้สร้างภาพ Visualization ของการคำนวณผ่านชั้น Convolution
        """
        stages: Dict[str, torch.Tensor] = {}
        # Stage 1 (Stem + Edge/Color Conv): features[0..1] -> [B, 16, 56, 56]
        x = self.features[0](x)
        x = self.features[1](x)
        stages["Stage 1: Conv Stem & Edges (56x56, 16ch)"] = x

        # Stage 2 (Surface & Groove Conv): features[2..3] -> [B, 24, 28, 28]
        x = self.features[2](x)
        x = self.features[3](x)
        stages["Stage 2: Texture & Groove (28x28, 24ch)"] = x

        # Stage 3 (Cavity & Shape Conv): features[4..6] -> [B, 40, 14, 14]
        for i in range(4, 7):
            x = self.features[i](x)
        stages["Stage 3: Cavity & Fracture (14x14, 40ch)"] = x

        # Stage 4 (Part-level Defect Attention Conv): features[7..8] -> [B, 48, 14, 14]
        for i in range(7, 9):
            x = self.features[i](x)
        stages["Stage 4: SE-Attention Defects (14x14, 48ch)"] = x

        # Stage 5 (Deep Semantic Conv 1x1): features[9..12] -> [B, 576, 7, 7]
        for i in range(9, 13):
            x = self.features[i](x)
        stages["Stage 5: Semantic Conv (7x7, 576ch)"] = x
        return stages

    def forward(
        self, img: torch.Tensor, phys: torch.Tensor, return_embedding: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fmap = self.features(img)  # [B, 576, 7, 7]
        avg_pool = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(fmap, 1).flatten(1)
        conv_vec = 0.7 * avg_pool + 0.3 * max_pool

        phys_vec = self.phys_mlp(phys)
        combined = torch.cat([conv_vec, phys_vec], dim=1)
        emb = self.fusion_proj(combined)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, F.normalize(emb, p=2, dim=1)
        return logits, emb


# =====================================================================
# 4. Visualization: Convolutional Layer Feature Maps & Training Charts
# =====================================================================

def tensor_to_rgb_img(img_tensor: torch.Tensor) -> np.ndarray:
    """แปลง Normalized Tensor [3, H, W] กลับเป็นภาพ RGB [H, W, 3] ช่วง [0..255]"""
    arr = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = (arr * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_channel_grid_image(fmap: torch.Tensor, top_k: int = 4, out_size: int = 160) -> Tuple[np.ndarray, np.ndarray]:
    """
    จาก Feature Map [C, H, W] ของชั้น Convolution:
      1) สร้างภาพ Heatmap ค่าเฉลี่ยพลังงานของทุก Channel (Mean Activation Map)
      2) สร้างตาราง 2x2 ของ Filter Channel ที่มีการตอบสนองสูงสุด 4 อันดับแรก (Top-4 Filters)
    """
    fmap_np = fmap.detach().cpu().numpy()  # [C, H, W]
    abs_fmap = np.abs(fmap_np)

    # 1. Mean Activation Map
    mean_map = np.mean(abs_fmap, axis=0)
    m_min, m_max = float(mean_map.min()), float(mean_map.max())
    norm_mean = ((mean_map - m_min) / max(m_max - m_min, 1e-6) * 255.0).astype(np.uint8)
    norm_mean_resized = cv2.resize(norm_mean, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    heatmap_bgr = cv2.applyColorMap(norm_mean_resized, cv2.COLORMAP_VIRIDIS)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 2. Top-4 Active Channels 2x2 Grid
    chan_energies = np.mean(abs_fmap, axis=(1, 2))
    top_indices = np.argsort(chan_energies)[::-1][:top_k]
    half = out_size // 2
    grid_rgb = np.zeros((out_size, out_size, 3), dtype=np.uint8)

    for i, c_idx in enumerate(top_indices):
        r = i // 2
        c = i % 2
        ch = fmap_np[c_idx]
        c_min, c_max = float(ch.min()), float(ch.max())
        ch_u8 = ((ch - c_min) / max(c_max - c_min, 1e-6) * 255.0).astype(np.uint8)
        ch_resized = cv2.resize(ch_u8, (half - 2, half - 2), interpolation=cv2.INTER_NEAREST)
        ch_color = cv2.cvtColor(cv2.applyColorMap(ch_resized, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        y0, x0 = r * half + 1, c * half + 1
        grid_rgb[y0 : y0 + half - 2, x0 : x0 + half - 2] = ch_color
        cv2.putText(
            grid_rgb,
            f"ch{c_idx}",
            (x0 + 4, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return heatmap_rgb, grid_rgb


@torch.no_grad()
def save_conv_layer_visualizations(
    model: HybridBeanNet,
    class_rep_samples: Dict[str, Dict[str, Any]],
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    out_dir: Path,
    img_size: int = 224,
    epoch_tag: str = "final",
) -> None:
    """
    บันทึกภาพผลลัพธ์เมื่อข้อมูลไหลผ่านแต่ละชั้นของ Convolution Layer (Stage 1 -> Stage 5):
      - รายละเอียดแยกตามคลาส (Mean Activation + Top-4 Individual Conv Filters + Deep CAM Overlay)
      - ภาพรวมทั้ง 7 คลาสเปรียบเทียบกันในแผ่นเดียว
    """
    model.eval()
    device = next(model.parameters()).device

    per_class_dir = out_dir / "per_class_layers"
    epoch_dir = out_dir / "epoch_progression"
    per_class_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    stage_names = [
        "Stage 1: Conv Stem & Edges (56x56, 16ch)",
        "Stage 2: Texture & Groove (28x28, 24ch)",
        "Stage 3: Cavity & Fracture (14x14, 40ch)",
        "Stage 4: SE-Attention Defects (14x14, 48ch)",
        "Stage 5: Semantic Conv (7x7, 576ch)",
    ]
    short_titles = [
        "Conv Stage 1\nEdges & Color (56x56)",
        "Conv Stage 2\nGroove & Surface (28x28)",
        "Conv Stage 3\nCavity & Shape (14x14)",
        "Conv Stage 4\nSE Defect Map (14x14)",
        "Conv Stage 5\nSemantic (7x7) + CAM",
    ]

    # สร้างภาพรวม 7 แถว (7 คลาส) x 7 คอลัมน์ (Input, Mask, Stage 1..5)
    fig_all, axes_all = plt.subplots(len(CLASSES), 7, figsize=(21, 19), dpi=140)
    fig_all.suptitle(
        f"Convolutional Layer Feature Maps Progression Across All 7 Coffee Bean Classes ({epoch_tag})",
        fontsize=15,
        fontweight="bold",
        y=0.992,
    )

    for row_idx, cls_name in enumerate(CLASSES):
        sample = class_rep_samples[cls_name]
        crop_bgr = sample["crop_bgr"]
        phys_norm = (sample["phys_raw"] - feat_mean) / feat_std

        img_t = preprocess_crop_to_tensor(crop_bgr, img_size).unsqueeze(0).to(device)
        phys_t = torch.from_numpy(phys_norm).float().unsqueeze(0).to(device)

        stages_dict = model.extract_conv_stages(img_t)
        logits, _ = model(img_t, phys_t)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_cls = CLASSES[int(np.argmax(probs))]
        pred_conf = float(np.max(probs)) * 100.0

        orig_rgb = tensor_to_rgb_img(img_t[0])
        mask_u8, cnt = segment_single_bean_crop(cv2.resize(crop_bgr, (img_size, img_size)))
        mask_vis = cv2.cvtColor(cv2.resize(crop_bgr, (img_size, img_size)), cv2.COLOR_BGR2RGB)
        cv2.drawContours(mask_vis, [cnt], -1, (0, 230, 60), 3)

        # วาดลงในตารางรวม 7 คลาส
        axes_all[row_idx, 0].imshow(orig_rgb)
        axes_all[row_idx, 0].set_ylabel(f"{cls_name.upper()}", fontsize=11, fontweight="bold")
        if row_idx == 0:
            axes_all[row_idx, 0].set_title("Input Bean\n(224x224)", fontsize=10, fontweight="bold")

        axes_all[row_idx, 1].imshow(mask_vis)
        if row_idx == 0:
            axes_all[row_idx, 1].set_title("OpenCV Contour\n& Physical ROI", fontsize=10, fontweight="bold")

        # สำหรับบันทึกรายคลาสแบบละเอียด (2 แถว x 6 คอลัมน์: แถวบน Mean Heatmap, แถวล่าง Top-4 Filters)
        if epoch_tag == "final":
            fig_cls, axes_cls = plt.subplots(2, 6, figsize=(18, 6.5), dpi=140)
            fig_cls.suptitle(
                f"Class: '{cls_name}' | Predicted: '{pred_cls}' ({pred_conf:.1f}%) — Layer-by-Layer Convolutional Feature Maps",
                fontsize=13,
                fontweight="bold",
            )
            axes_cls[0, 0].imshow(orig_rgb)
            axes_cls[0, 0].set_title("Input Image (224x224)", fontsize=9, fontweight="bold")
            axes_cls[1, 0].imshow(mask_vis)
            axes_cls[1, 0].set_title("OpenCV Contour Mask", fontsize=9, fontweight="bold")

        for s_idx, s_name in enumerate(stage_names):
            fmap = stages_dict[s_name][0]  # [C, H, W]
            heatmap_rgb, top4_grid_rgb = make_channel_grid_image(fmap, top_k=4, out_size=img_size)

            # ถ้าเป็น Stage 5 สุดท้าย ให้ทำ Overlay CAM ลงบนภาพเมล็ดกาแฟด้วยเพื่อให้เห็นว่าโมเดลโฟกัสจุดไหน
            if s_idx == 4:
                cam_overlay = cv2.addWeighted(orig_rgb, 0.45, heatmap_rgb, 0.55, 0)
                axes_all[row_idx, s_idx + 2].imshow(cam_overlay)
            else:
                axes_all[row_idx, s_idx + 2].imshow(heatmap_rgb)

            if row_idx == 0:
                axes_all[row_idx, s_idx + 2].set_title(short_titles[s_idx], fontsize=10, fontweight="bold")

            if epoch_tag == "final":
                axes_cls[0, s_idx + 1].imshow(heatmap_rgb if s_idx < 4 else cam_overlay)
                axes_cls[0, s_idx + 1].set_title(f"{short_titles[s_idx]}\n(Mean Activation)", fontsize=8.5, fontweight="bold")
                axes_cls[1, s_idx + 1].imshow(top4_grid_rgb)
                axes_cls[1, s_idx + 1].set_title(f"Stage {s_idx + 1}: Top-4 Filters", fontsize=8.5)

        for c in range(7):
            axes_all[row_idx, c].set_xticks([])
            axes_all[row_idx, c].set_yticks([])

        if epoch_tag == "final":
            for r in range(2):
                for c in range(6):
                    axes_cls[r, c].axis("off")
            plt.tight_layout()
            fig_cls.savefig(per_class_dir / f"conv_layers_{cls_name}.png", bbox_inches="tight")
            plt.close(fig_cls)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if epoch_tag == "final":
        fig_all.savefig(out_dir / "all_7_classes_conv_summary.png", bbox_inches="tight")
    fig_all.savefig(epoch_dir / f"conv_maps_{epoch_tag}.png", bbox_inches="tight")
    plt.close(fig_all)


def save_training_dashboard_and_charts(
    history: List[Dict[str, float]],
    y_val_true: List[int],
    y_val_pred: List[int],
    y_all_true: List[int],
    y_all_pred: List[int],
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    out_dir: Path,
) -> None:
    """
    สร้างกราฟวิเคราะห์ผลการเทรนสำหรับการจูน Hyperparameters:
      1) training_curves_dashboard.png (Loss curve, Accuracy curve, LR schedule, Per-Class F1)
      2) confusion_matrix.png (Validation Split & Full 175-Bean Dataset)
      3) feature_space_pca.png (2D PCA Scatter plot ของ Feature Embeddings ทั้ง 7 คลาส)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(h["epoch"]) for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    tr_acc = [h["train_acc"] * 100.0 for h in history]
    va_acc = [h["val_acc"] * 100.0 for h in history]
    va_ens_acc = [h["val_ensemble_acc"] * 100.0 for h in history]
    lrs = [h["lr"] for h in history]

    best_idx = int(np.argmax(va_ens_acc))
    best_ep = epochs[best_idx]

    # 1. Training Curves Dashboard (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    fig.suptitle(
        "Coffee Bean Classification — Training & Validation Diagnostic Dashboard",
        fontsize=15,
        fontweight="bold",
    )

    # Panel 1: Loss Curve
    ax = axes[0, 0]
    ax.plot(epochs, tr_loss, label="Train Loss (with Mixup + Augmentation)", color="#1f77b4", linewidth=2.2)
    ax.plot(epochs, va_loss, label="Validation Loss (Clean Crops)", color="#d62728", linewidth=2.2)
    ax.axvline(best_ep, color="#2ca02c", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_ep})")
    ax.set_title("Cross-Entropy Loss (Train vs Validation)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Panel 2: Accuracy Curve
    ax = axes[0, 1]
    ax.plot(epochs, tr_acc, label="Train Accuracy (%)", color="#1f77b4", linewidth=2.0, alpha=0.85)
    ax.plot(epochs, va_acc, label="Val Accuracy - ConvNet Only (%)", color="#ff7f0e", linewidth=2.0)
    ax.plot(epochs, va_ens_acc, label="Val Accuracy - Hybrid Ensemble (%)", color="#2ca02c", linewidth=2.5)
    ax.scatter([best_ep], [va_ens_acc[best_idx]], color="#2ca02c", s=70, zorder=5)
    ax.set_title(
        f"Classification Accuracy (Peak Val Hybrid Acc: {va_ens_acc[best_idx]:.2f}%)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    # Panel 3: Learning Rate Schedule & Train-Val Gap
    ax = axes[1, 0]
    ax.plot(epochs, lrs, color="#9467bd", linewidth=2.2, label="Learning Rate (Cosine Annealing)")
    ax.set_title("Learning Rate Schedule Across Epochs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Panel 4: Per-Class Validation F1-Score Bar Chart
    ax = axes[1, 1]
    report = classification_report(
        y_val_true, y_val_pred, labels=list(range(len(CLASSES))), target_names=CLASSES, output_dict=True, zero_division=0
    )
    cls_f1s = [report[c]["f1-score"] * 100.0 for c in CLASSES]
    bars = ax.bar(CLASSES, cls_f1s, color=["#2ca02c", "#17becf", "#333333", "#8c564b", "#e377c2", "#bcbd22", "#ff7f0e"])
    ax.set_ylim(0, 112)
    ax.set_title("Validation Per-Class F1-Score (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1-Score (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for b, val in zip(bars, cls_f1s):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            val + 2.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "training_curves_dashboard.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Confusion Matrices (Validation Holdout & All 175 Beans)
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150)
    for ax_i, (yt, yp, title) in enumerate(
        [
            (y_val_true, y_val_pred, f"Validation Holdout Set ({len(y_val_true)} Beans)"),
            (y_all_true, y_all_pred, f"All Extracted Beans ({len(y_all_true)} Beans)"),
        ]
    ):
        cm = confusion_matrix(yt, yp, labels=list(range(len(CLASSES))))
        acc_val = accuracy_score(yt, yp) * 100.0
        im = axes_cm[ax_i].imshow(cm, cmap="Blues")
        axes_cm[ax_i].set_title(f"{title}\nAccuracy: {acc_val:.2f}%", fontsize=12, fontweight="bold")
        axes_cm[ax_i].set_xticks(range(len(CLASSES)))
        axes_cm[ax_i].set_yticks(range(len(CLASSES)))
        axes_cm[ax_i].set_xticklabels(CLASSES, rotation=35, ha="right")
        axes_cm[ax_i].set_yticklabels(CLASSES)
        axes_cm[ax_i].set_xlabel("Predicted Class")
        axes_cm[ax_i].set_ylabel("True Class")
        thresh = cm.max() / 2.0
        for r in range(len(CLASSES)):
            for c in range(len(CLASSES)):
                axes_cm[ax_i].text(
                    c,
                    r,
                    str(cm[r, c]),
                    ha="center",
                    va="center",
                    color="white" if cm[r, c] > thresh else "black",
                    fontweight="bold",
                )
        fig_cm.colorbar(im, ax=axes_cm[ax_i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_cm.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig_cm)

    # 3. 2D PCA Cluster Visualization of Learned Multimodal Embeddings
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_embeddings)
    fig_pca, ax_pca = plt.subplots(figsize=(9, 7), dpi=150)
    palette = ["#2ca02c", "#17becf", "#222222", "#8c564b", "#d62728", "#bcbd22", "#ff7f0e"]
    for idx, cls_name in enumerate(CLASSES):
        mask_c = all_labels == idx
        ax_pca.scatter(
            coords[mask_c, 0],
            coords[mask_c, 1],
            label=cls_name,
            color=palette[idx],
            s=65,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.7,
        )
    ax_pca.set_title("2D PCA Projection of Learned Multimodal Bean Embeddings (175 Beans)", fontsize=12, fontweight="bold")
    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax_pca.grid(True, linestyle="--", alpha=0.35)
    ax_pca.legend(title="Bean Class", loc="best")
    plt.tight_layout()
    fig_pca.savefig(out_dir / "feature_space_pca.png", bbox_inches="tight")
    plt.close(fig_pca)


# =====================================================================
# 5. Main Training Loop & Evaluation
# =====================================================================

def evaluate_hybrid(
    model: HybridBeanNet,
    samples: List[Dict[str, Any]],
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    rf_clf: Any,
    prototypes: np.ndarray,
    img_size: int,
    device: torch.device,
) -> Tuple[float, float, float, List[int], List[int], np.ndarray]:
    """ประเมินผลบนชุดข้อมูลด้วยทั้ง ConvNet เดี่ยว และ Hybrid Ensemble (ConvNet + Prototype + OpenCV RF)"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    y_true: List[int] = []
    y_cnn_pred: List[int] = []
    y_ens_pred: List[int] = []
    embeddings_list: List[np.ndarray] = []

    with torch.no_grad():
        for item in samples:
            crop = item["crop_bgr"]
            phys_norm = (item["phys_raw"] - feat_mean) / feat_std
            label = int(item["label"])

            # Multi-view (Original + Horizontal Flip) เพื่อความเสถียรของฟีเจอร์
            t1 = preprocess_crop_to_tensor(crop, img_size)
            t2 = preprocess_crop_to_tensor(cv2.flip(crop, 1), img_size)
            imgs_t = torch.stack([t1, t2], dim=0).to(device)
            phys_t = torch.from_numpy(np.stack([phys_norm, phys_norm], axis=0)).float().to(device)

            logits, embs = model(imgs_t, phys_t, return_embedding=True)
            mean_logits = torch.mean(logits, dim=0, keepdim=True)
            loss = criterion(mean_logits, torch.tensor([label], device=device))
            total_loss += float(loss.item())

            cnn_prob = F.softmax(mean_logits, dim=1)[0].cpu().numpy()
            mean_emb = F.normalize(torch.mean(embs, dim=0, keepdim=True), p=2, dim=1)[0].cpu().numpy()
            embeddings_list.append(mean_emb)

            # Cosine similarity กับ Class Prototypes
            cos_sim = np.dot(prototypes, mean_emb)
            proto_prob = np.exp(cos_sim * 6.0)
            proto_prob = proto_prob / np.sum(proto_prob)

            # OpenCV Physical Random Forest / ExtraTrees Probability
            rf_prob = rf_clf.predict_proba(phys_norm.reshape(1, -1))[0]

            # Hybrid Ensemble Probability
            ens_prob = 0.55 * cnn_prob + 0.20 * proto_prob + 0.25 * rf_prob

            y_true.append(label)
            y_cnn_pred.append(int(np.argmax(cnn_prob)))
            y_ens_pred.append(int(np.argmax(ens_prob)))

    avg_loss = total_loss / max(len(samples), 1)
    cnn_acc = accuracy_score(y_true, y_cnn_pred)
    ens_acc = accuracy_score(y_true, y_ens_pred)
    return avg_loss, float(cnn_acc), float(ens_acc), y_true, y_ens_pred, np.vstack(embeddings_list)


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train Hybrid Coffee Bean Classifier + Conv Visualizer")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=base_dir.parent / "datasets",
        help="Path to raw 7-class dataset directory",
    )
    parser.add_argument(
        "--weights-out",
        type=Path,
        default=base_dir / "weights" / "best.pt",
        help="Output path for trained weights (best.pt)",
    )
    parser.add_argument(
        "--analysis-out",
        type=Path,
        default=base_dir / "outputs" / "training_analysis",
        help="Output directory for accuracy/loss charts and Conv layer visualizations",
    )
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs (default: 45)")
    parser.add_argument("--batch-size", type=int, default=28, help="Mini-batch size (default: 28)")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Initial learning rate (default: 1.2e-3)")
    parser.add_argument("--weight-decay", type=float, default=2e-4, help="AdamW weight decay (default: 2e-4)")
    parser.add_argument("--label-smoothing", type=float, default=0.06, help="Label smoothing factor (default: 0.06)")
    parser.add_argument("--mixup-alpha", type=float, default=0.20, help="Mixup alpha regularization (default: 0.20)")
    parser.add_argument("--img-size", type=int, default=224, help="Crop input resolution (default: 224)")
    parser.add_argument("--aug-repeats", type=int, default=4, help="Augmented repeats per epoch for training beans")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    t0 = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Training Device: {device} | Image Size: {args.img_size}x{args.img_size} | Epochs: {args.epochs}")

    # 1. สกัดเมล็ดกาแฟทั้ง 175 เม็ด และคำนวณฟีเจอร์กายภาพ 28 มิติจากภาพต้นฉบับ
    all_samples: List[Dict[str, Any]] = []
    class_rep_samples: Dict[str, Dict[str, Any]] = {}

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = args.dataset / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Dataset class folder not found: {cls_dir}")

        img_paths = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        cls_beans: List[Dict[str, Any]] = []
        for img_p in img_paths:
            img_bgr = load_image_exif_corrected(img_p)
            beans = detect_beans_in_frame(img_bgr)
            for b_i, b in enumerate(beans):
                phys_raw = extract_bean_physical_features(b["crop_bgr"], b["rel_area"])
                sample = {
                    "class_name": cls_name,
                    "label": cls_idx,
                    "source_image": img_p.name,
                    "bean_idx": b_i + 1,
                    "crop_bgr": b["crop_bgr"],
                    "rel_area": b["rel_area"],
                    "phys_raw": phys_raw,
                }
                cls_beans.append(sample)
        print(f"  -> Extracted {len(cls_beans):2d} beans for class '{cls_name}'")
        if cls_beans:
            class_rep_samples[cls_name] = cls_beans[min(3, len(cls_beans) - 1)]
        all_samples.extend(cls_beans)

    print(f"[DATA] Total physical beans extracted: {len(all_samples)}")

    # 2. แบ่ง Train / Validation ระดับเมล็ดต้นฉบับอย่างเคร่งครัด (Strict Split BEFORE Featurization/Augmentation)
    # แบ่ง Stratified คลาสละ 20 เม็ดเป็น Train (140 เม็ด) และ 5 เม็ดเป็น Val (35 เม็ด) โดยกระจายทุกรูป (circle/grid/square)
    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    for cls_idx in range(len(CLASSES)):
        cls_items = [s for s in all_samples if s["label"] == cls_idx]
        # เลือกทุกๆ ลำดับที่ 5 เป็น Validation เพื่อให้กระจายครบทั้ง circle, grid, square
        for i, item in enumerate(cls_items):
            if i % 5 == 2:
                val_samples.append(item)
            else:
                train_samples.append(item)

    print(f"[SPLIT] Strict Bean-Level Split -> Train: {len(train_samples)} beans | Validation: {len(val_samples)} beans")

    # 3. คำนวณ Mean/Std ของฟีเจอร์กายภาพจากชุด Train เท่านั้น (ป้องกัน Data Leakage 100%)
    X_train_phys = np.vstack([s["phys_raw"] for s in train_samples])
    feat_mean = np.mean(X_train_phys, axis=0).astype(np.float32)
    feat_std = np.maximum(np.std(X_train_phys, axis=0), 1e-4).astype(np.float32)

    X_train_norm = (X_train_phys - feat_mean) / feat_std
    y_train_arr = np.array([s["label"] for s in train_samples], dtype=np.int64)

    # เทรนโมเดลฝั่งฟีเจอร์กายภาพ (ExtraTrees บน 28 OpenCV features ที่ถูกปรับสเกลด้วย Train Split)
    rf_clf = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=args.seed,
    )
    rf_clf.fit(X_train_norm, y_train_arr)

    # 4. สร้าง DataLoader และโมเดล HybridBeanNet
    train_ds = CoffeeBeanDataset(
        train_samples,
        feat_mean=feat_mean,
        feat_std=feat_std,
        img_size=args.img_size,
        is_train=True,
        repeats=args.aug_repeats,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = HybridBeanNet(num_classes=len(CLASSES), num_phys_features=NUM_PHYSICAL_FEATURES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)

    history: List[Dict[str, float]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state_dict = None
    best_prototypes = None

    args.analysis_out.mkdir(parents=True, exist_ok=True)
    conv_vis_dir = args.analysis_out / "conv_feature_maps"

    # บันทึกภาพผลลัพธ์ผ่าน Convolution Layer ก่อนเริ่มเทรน (Epoch 0 - Initial Pretrained Filters)
    save_conv_layer_visualizations(
        model, class_rep_samples, feat_mean, feat_std, conv_vis_dir, img_size=args.img_size, epoch_tag="epoch_00_init"
    )

    print("-" * 88)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val CNN':>9} | {'Val Hybrid':>10} | {'LR':>9}")
    print("-" * 88)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs_t, phys_t, labels_t in train_loader:
            imgs_t = imgs_t.to(device)
            phys_t = phys_t.to(device)
            labels_t = labels_t.to(device)

            optimizer.zero_grad()

            # Mixup Regularization บนภาพและฟีเจอร์กายภาพ
            if args.mixup_alpha > 0.0 and random.random() < 0.45:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                perm = torch.randperm(imgs_t.size(0), device=device)
                mixed_imgs = lam * imgs_t + (1.0 - lam) * imgs_t[perm]
                mixed_phys = lam * phys_t + (1.0 - lam) * phys_t[perm]
                logits, _ = model(mixed_imgs, mixed_phys)
                loss = lam * criterion(logits, labels_t) + (1.0 - lam) * criterion(logits, labels_t[perm])
            else:
                logits, _ = model(imgs_t, phys_t)
                loss = criterion(logits, labels_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += float(loss.item()) * imgs_t.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += int((preds == labels_t).sum().item())
            total_train += int(imgs_t.size(0))

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        train_loss = running_loss / max(total_train, 1)
        train_acc = correct_train / max(total_train, 1)

        # คำนวณ Class Prototypes จากชุด Train ปัจจุบัน
        model.eval()
        proto_mat = np.zeros((len(CLASSES), 192), dtype=np.float32)
        with torch.no_grad():
            for c_i in range(len(CLASSES)):
                c_items = [s for s in train_samples if s["label"] == c_i]
                c_imgs = torch.stack([preprocess_crop_to_tensor(s["crop_bgr"], args.img_size) for s in c_items]).to(device)
                c_phys = torch.from_numpy(
                    np.vstack([(s["phys_raw"] - feat_mean) / feat_std for s in c_items])
                ).float().to(device)
                _, c_embs = model(c_imgs, c_phys, return_embedding=True)
                mean_proto = F.normalize(torch.mean(c_embs, dim=0, keepdim=True), p=2, dim=1)[0].cpu().numpy()
                proto_mat[c_i] = mean_proto

        val_loss, val_cnn_acc, val_ens_acc, y_val_true, y_val_pred, _ = evaluate_hybrid(
            model, val_samples, feat_mean, feat_std, rf_clf, proto_mat, args.img_size, device
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "train_acc": round(train_acc, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_cnn_acc, 5),
                "val_ensemble_acc": round(val_ens_acc, 5),
                "lr": round(current_lr, 7),
            }
        )

        is_best = (val_ens_acc > best_val_acc) or (
            abs(val_ens_acc - best_val_acc) < 1e-6 and val_loss < best_val_loss
        )
        if is_best:
            best_val_acc = val_ens_acc
            best_val_loss = val_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_prototypes = proto_mat.copy()

        if epoch in (1, 15, 30, args.epochs) or is_best:
            print(
                f"{epoch:5d} | {train_loss:10.4f} | {train_acc*100:8.2f}% | {val_loss:10.4f} | "
                f"{val_cnn_acc*100:8.2f}% | {val_ens_acc*100:9.2f}% | {current_lr:9.6f}"
                + ("  <-- BEST" if is_best else "")
            )

        # บันทึกภาพผลลัพธ์ผ่าน Convolution Layer ตามช่วงเวลาเพื่อดูวิวัฒนาการของ Feature Map
        if epoch in (10, 25):
            save_conv_layer_visualizations(
                model,
                class_rep_samples,
                feat_mean,
                feat_std,
                conv_vis_dir,
                img_size=args.img_size,
                epoch_tag=f"epoch_{epoch:02d}",
            )

    print("-" * 88)

    # 5. โหลดโมเดลที่ดีที่สุดกลับมาประเมินผลและสร้างภาพกราฟ + ภาพ Convolution Layer ครบทั้ง 7 คลาส
    model.load_state_dict(best_state_dict)
    model.eval()

    _, val_cnn_best, val_ens_best, y_val_true, y_val_pred, _ = evaluate_hybrid(
        model, val_samples, feat_mean, feat_std, rf_clf, best_prototypes, args.img_size, device
    )
    _, all_cnn_acc, all_ens_acc, y_all_true, y_all_pred, all_embs = evaluate_hybrid(
        model, all_samples, feat_mean, feat_std, rf_clf, best_prototypes, args.img_size, device
    )

    print(f"[RESULT] Best Validation Holdout Accuracy (35 unseen beans): {val_ens_best*100:.2f}% (ConvNet: {val_cnn_best*100:.2f}%)")
    print(f"[RESULT] Overall Accuracy across all 175 Beans:              {all_ens_acc*100:.2f}% (ConvNet: {all_cnn_acc*100:.2f}%)")

    # 6. บันทึกภาพการคำนวณผ่าน Convolution Layer ทั้ง 5 Stage ของโมเดลที่ดีที่สุด
    print("[VISUAL] Generating Convolutional Layer Feature Map visualizations...")
    save_conv_layer_visualizations(
        model, class_rep_samples, feat_mean, feat_std, conv_vis_dir, img_size=args.img_size, epoch_tag="final"
    )

    # 7. บันทึกกราฟ Loss, Accuracy, Confusion Matrix, PCA และไฟล์ CSV
    print("[CHART]  Generating Accuracy/Loss curves, Confusion Matrix, and PCA plots...")
    save_training_dashboard_and_charts(
        history=history,
        y_val_true=y_val_true,
        y_val_pred=y_val_pred,
        y_all_true=y_all_true,
        y_all_pred=y_all_pred,
        all_embeddings=all_embs,
        all_labels=np.array(y_all_true),
        out_dir=args.analysis_out,
    )

    csv_path = args.analysis_out / "training_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_ensemble_acc", "lr"]
        )
        writer.writeheader()
        writer.writerows(history)

    # 8. คำนวณ Full-Dataset Class Prototypes & RF สำหรับใช้ในการส่งงานจริง (หลังจากวัดผล Validation เสร็จสิ้นแล้ว)
    X_all_phys = np.vstack([s["phys_raw"] for s in all_samples])
    y_all_arr = np.array([s["label"] for s in all_samples], dtype=np.int64)
    X_all_norm = (X_all_phys - feat_mean) / feat_std

    rf_prod = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=args.seed,
    )
    rf_prod.fit(X_all_norm, y_all_arr)

    prod_prototypes = np.zeros((len(CLASSES), 192), dtype=np.float32)
    with torch.no_grad():
        for c_i in range(len(CLASSES)):
            c_items = [s for s in all_samples if s["label"] == c_i]
            c_imgs = torch.stack([preprocess_crop_to_tensor(s["crop_bgr"], args.img_size) for s in c_items]).to(device)
            c_phys = torch.from_numpy(
                np.vstack([(s["phys_raw"] - feat_mean) / feat_std for s in c_items])
            ).float().to(device)
            _, c_embs = model(c_imgs, c_phys, return_embedding=True)
            prod_prototypes[c_i] = F.normalize(torch.mean(c_embs, dim=0, keepdim=True), p=2, dim=1)[0].cpu().numpy()

    # แปลง ExtraTreesClassifier เป็น Pure-NumPy arrays เพื่อให้ infer.py รันได้โดยไม่ต้องพึ่ง scikit-learn บน Raspberry Pi 4
    rf_trees = []
    for est in rf_prod.estimators_:
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

    # บันทึกทุกอย่างลงใน weights/best.pt ไฟล์เดียวเพื่อให้ infer.py โหลดใช้งานออฟไลน์บน Raspberry Pi 4 ได้ทันที
    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "architecture": "HybridBeanNet_MobileNetV3Small_OpenCV28",
        "classes": CLASSES,
        "img_size": args.img_size,
        "num_phys_features": NUM_PHYSICAL_FEATURES,
        "model_state_dict": best_state_dict,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "class_prototypes": prod_prototypes,
        "rf_trees": rf_trees,
        "metrics": {
            "val_holdout_accuracy": round(val_ens_best * 100.0, 2),
            "val_cnn_only_accuracy": round(val_cnn_best * 100.0, 2),
            "overall_bean_accuracy": round(all_ens_acc * 100.0, 2),
            "best_val_loss": round(best_val_loss, 4),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "mixup_alpha": args.mixup_alpha,
            "img_size": args.img_size,
            "aug_repeats": args.aug_repeats,
            "seed": args.seed,
        },
    }
    torch.save(checkpoint_payload, args.weights_out)

    # Export ONNX + Pure-NumPy metadata สำหรับ infer.py (รันบน Raspberry Pi OS 32-bit ได้โดยไม่ต้องใช้ PyTorch)
    class _ConvExtractor(nn.Module):
        def __init__(self, net: HybridBeanNet):
            super().__init__()
            self.features = net.features

        def forward(self, img: torch.Tensor) -> torch.Tensor:
            fmap = self.features(img)
            avg_p = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
            max_p = F.adaptive_max_pool2d(fmap, 1).flatten(1)
            return 0.7 * avg_p + 0.3 * max_p

    def _fold_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> Tuple[np.ndarray, np.ndarray]:
        w = linear.weight.detach().cpu().numpy().astype(np.float32)
        b = linear.bias.detach().cpu().numpy().astype(np.float32)
        g = bn.weight.detach().cpu().numpy().astype(np.float32)
        be = bn.bias.detach().cpu().numpy().astype(np.float32)
        m = bn.running_mean.detach().cpu().numpy().astype(np.float32)
        v = bn.running_var.detach().cpu().numpy().astype(np.float32)
        sc = g / np.sqrt(v + float(bn.eps))
        return (w * sc[:, None]).astype(np.float32), ((b - m) * sc + be).astype(np.float32)

    model_cpu = model.cpu().eval()
    onnx_out = args.weights_out.with_suffix(".onnx")
    torch.onnx.export(
        _ConvExtractor(model_cpu).eval(),
        torch.randn(1, 3, args.img_size, args.img_size),
        str(onnx_out),
        opset_version=13,
        dynamo=False,
        input_names=["img"],
        output_names=["conv_vec"],
    )
    W_p1, b_p1 = _fold_linear_bn(model_cpu.phys_mlp[0], model_cpu.phys_mlp[1])
    W_p2, b_p2 = _fold_linear_bn(model_cpu.phys_mlp[4], model_cpu.phys_mlp[5])
    W_fus, b_fus = _fold_linear_bn(model_cpu.fusion_proj[0], model_cpu.fusion_proj[1])
    W_cls = model_cpu.classifier.weight.detach().cpu().numpy().astype(np.float32)
    b_cls = model_cpu.classifier.bias.detach().cpu().numpy().astype(np.float32)
    tree_offsets = np.zeros(len(rf_trees) + 1, dtype=np.int32)
    tree_offsets[1:] = np.cumsum([len(t["children_left"]) for t in rf_trees])
    npz_out = args.weights_out.parent / "model_meta.npz"
    np.savez_compressed(
        npz_out,
        classes=np.array(CLASSES),
        img_size=np.array([args.img_size], dtype=np.int32),
        feat_mean=feat_mean,
        feat_std=feat_std,
        class_prototypes=prod_prototypes,
        W_p1=W_p1, b_p1=b_p1,
        W_p2=W_p2, b_p2=b_p2,
        W_fus=W_fus, b_fus=b_fus,
        W_cls=W_cls, b_cls=b_cls,
        tree_offsets=tree_offsets,
        tree_cl=np.concatenate([t["children_left"] for t in rf_trees]).astype(np.int32),
        tree_cr=np.concatenate([t["children_right"] for t in rf_trees]).astype(np.int32),
        tree_feat=np.concatenate([t["feature"] for t in rf_trees]).astype(np.int32),
        tree_thr=np.concatenate([t["threshold"] for t in rf_trees]).astype(np.float32),
        tree_probs=np.vstack([t["probs"] for t in rf_trees]).astype(np.float32),
    )
    print(f"[SAVED]  RPi 32-bit ONNX+NPZ bundle: {onnx_out} + {npz_out}")

    summary_json = args.analysis_out / "hyperparameters_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "metrics": checkpoint_payload["metrics"],
                "hyperparameters": checkpoint_payload["hyperparameters"],
                "weights_file": str(args.weights_out),
                "weights_size_mb": round(args.weights_out.stat().st_size / (1024 * 1024), 2),
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[SAVED]  Model weights saved to: {args.weights_out} ({args.weights_out.stat().st_size / (1024*1024):.2f} MB)")
    print(f"[SAVED]  Training charts & Conv maps saved to: {args.analysis_out}")


if __name__ == "__main__":
    main()
