#!/usr/bin/env python3
"""
infer.py — Coffee Bean Defect Classification Inference Pipeline (Day 1)
Target Execution:
  - Raspberry Pi 4 (32-bit armv7l / 64-bit aarch64 Raspberry Pi OS, CPU)
  - Ubuntu / Linux (x86_64 / aarch64, CPU) & Windows

ใช้เพียง OpenCV DNN (.onnx) + Pure NumPy (.npz) + Pillow (ไม่ต้องติดตั้ง PyTorch บน Raspi 32-bit)
คำสั่งมาตรฐานตาม INFER_SPEC.md:
    python infer.py --input test/ --weights weights/ --out outputs/
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

# ลำดับชื่อคลาสภาษาอังกฤษตัวพิมพ์เล็กตาม INFER_SPEC.md
CLASSES = [
    "good",
    "shell",
    "black",
    "husk",
    "broken",
    "parchment",
    "immature",
]
VALID_CLASSES_SET = set(CLASSES)
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
NUM_PHYSICAL_FEATURES = 28

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def configure_cpu_runtime() -> None:
    """ตั้งค่า CPU threads ให้เหมาะกับ Raspberry Pi 4 (4-Core ARM)"""
    n_threads = min(4, os.cpu_count() or 4)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_threads))
    cv2.setNumThreads(n_threads)


# =====================================================================
# Image Reading (EXIF Corrected), Bean Localization & Deterministic Preprocessing
# =====================================================================

def load_image_exif_corrected(img_path: Path) -> np.ndarray:
    """อ่านภาพและปรับหมุนตาม EXIF Orientation ให้ถูกต้องก่อนทำ Preprocessing"""
    try:
        with Image.open(img_path) as pil_img:
            pil_img.load()
            oriented = ImageOps.exif_transpose(pil_img).convert("RGB")
            arr = np.array(oriented)
    except Exception as exc:
        raise RuntimeError(f"Cannot decode image '{img_path}': {exc}") from exc

    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        raise RuntimeError(f"Invalid image tensor shape for '{img_path}': {arr.shape}")
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def detect_beans_in_frame(
    img_bgr: np.ndarray,
    pad_ratio: float = 0.08,
) -> List[Dict[str, Any]]:
    """
    ตรวจจับเมล็ดกาแฟทุกเม็ดในภาพด้วย Local Illumination Normalization + CIE L*a*b* + Watershed
    รองรับทั้งภาพหลายเม็ด (จัดเรียงหรือวางกระจัดกระจาย) และภาพถ่ายซูมเมล็ดเดียว
    """
    orig_h, orig_w = img_bgr.shape[:2]
    max_dim = max(orig_h, orig_w)
    work_scale = min(1.0, 1024.0 / float(max_dim))
    ws_w = max(64, int(round(orig_w * work_scale)))
    ws_h = max(64, int(round(orig_h * work_scale)))
    small = cv2.resize(img_bgr, (ws_w, ws_h), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_chan, a_chan, b_chan = cv2.split(lab)

    ksize = max(51, int(round(max(ws_w, ws_h) * 0.45)) | 1)
    bg_L = cv2.GaussianBlur(L_chan, (ksize, ksize), 0)
    bg_a = cv2.GaussianBlur(a_chan, (ksize, ksize), 0)
    bg_b = cv2.GaussianBlur(b_chan, (ksize, ksize), 0)

    dark_diff = np.clip(bg_L - L_chan, 0, 255)
    chroma_diff = np.sqrt(np.maximum(0.0, (a_chan - bg_a) ** 2 + (b_chan - bg_b) ** 2))
    fg_score_u8 = np.clip(dark_diff * 0.75 + chroma_diff * 1.85, 0, 255).astype(np.uint8)
    fg_score_u8 = cv2.GaussianBlur(fg_score_u8, (5, 5), 0)

    margin_x = int(ws_w * 0.02)
    margin_top = int(ws_h * 0.02)
    margin_bot = int(ws_h * 0.94)
    roi_mask = np.zeros_like(fg_score_u8)
    roi_mask[margin_top:margin_bot, margin_x : ws_w - margin_x] = 255
    fg_masked = cv2.bitwise_and(fg_score_u8, fg_score_u8, mask=roi_mask)

    roi_slice = fg_masked[margin_top:margin_bot, margin_x : ws_w - margin_x]
    if roi_slice.size == 0 or np.max(roi_slice) < 20:
        return [
            {
                "bbox_padded": (0, 0, orig_w, orig_h),
                "center": (orig_w / 2.0, orig_h / 2.0),
                "crop_bgr": img_bgr.copy(),
                "rel_area": 0.0035,
            }
        ]

    otsu_val, _ = cv2.threshold(roi_slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = max(28.0, float(otsu_val) * 1.08)
    _, binary = cv2.threshold(fg_masked, thresh_val, 255, cv2.THRESH_BINARY)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern, iterations=1)

    raw_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = ws_w * ws_h
    min_bean_area = img_area * 0.00045
    max_bean_area = img_area * 0.65000

    valid_areas = [
        cv2.contourArea(c)
        for c in raw_contours
        if min_bean_area <= cv2.contourArea(c) <= max_bean_area
    ]
    median_area = float(np.median(valid_areas)) if valid_areas else (img_area * 0.002)

    contours = []
    for cnt in raw_contours:
        area = cv2.contourArea(cnt)
        if not (min_bean_area <= area <= max_bean_area):
            continue
        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
        if w_b > ws_w * 0.55 and (y_b + h_b) >= margin_bot - 5 and (w_b / max(h_b, 1)) > 3.0:
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
                markers[cv2.subtract(mask_single, sure_fg) == 255] = 0
                cv2.watershed(cv2.cvtColor(mask_single, cv2.COLOR_GRAY2BGR), markers)
                for lbl in range(2, n_labels + 1):
                    sub_cnts, _ = cv2.findContours(
                        np.uint8(markers == lbl) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if sub_cnts:
                        contours.append(max(sub_cnts, key=cv2.contourArea))
                continue
        contours.append(cnt)

    beans: List[Dict[str, Any]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_bean_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) / max(min(w, h), 1) > 3.5:
            continue

        x1 = int(round(x / work_scale))
        y1 = int(round(y / work_scale))
        bw = int(round(w / work_scale))
        bh = int(round(h / work_scale))
        pad_x = int(round(bw * pad_ratio))
        pad_y = int(round(bh * pad_ratio))
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(orig_w, x1 + bw + pad_x)
        py2 = min(orig_h, y1 + bh + pad_y)

        crop_bgr = img_bgr[py1:py2, px1:px2].copy()
        if crop_bgr.size > 0:
            beans.append(
                {
                    "bbox_padded": (px1, py1, px2, py2),
                    "center": (x1 + bw / 2.0, y1 + bh / 2.0),
                    "crop_bgr": crop_bgr,
                    "rel_area": float((bw * bh) / max(orig_w * orig_h, 1)),
                }
            )

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
    """สร้าง Mask ของเมล็ดกาแฟภายในภาพ Crop เดี่ยว"""
    h, w = crop_bgr.shape[:2]
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_c, a_c, b_c = cv2.split(lab)

    bg_L = float(np.percentile(np.concatenate([L_c[0, :], L_c[-1, :], L_c[:, 0], L_c[:, -1]]), 75))
    bg_a = float(np.median(np.concatenate([a_c[0, :], a_c[-1, :], a_c[:, 0], a_c[:, -1]])))
    bg_b = float(np.median(np.concatenate([b_c[0, :], b_c[-1, :], b_c[:, 0], b_c[:, -1]])))

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
    """สกัดฟีเจอร์กายภาพ 28 มิติจากภาพ Crop เมล็ดกาแฟ (คงที่ ไม่สุ่ม)"""
    std_crop = cv2.resize(crop_bgr, (160, 160), interpolation=cv2.INTER_AREA)
    mask, cnt = segment_single_bean_crop(std_crop)
    h, w = std_crop.shape[:2]

    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
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
    delta_chroma = float(np.sqrt((a_mean - bg_a) ** 2 + (b_mean - bg_b) ** 2))

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

    clamped_rel_area = min(max(rel_area, 0.0005), 0.015)

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
            clamped_rel_area * 100.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def preprocess_crop_to_nchw(crop_bgr: np.ndarray, img_size: int = 224) -> np.ndarray:
    """Preprocessing แบบคงที่ (Letterbox Resize + Pad ด้วยสีขอบกระดาษ + ImageNet Normalize) ไม่ทำ TTA"""
    h, w = crop_bgr.shape[:2]
    scale = img_size / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    border_vals = np.concatenate([resized[0, :], resized[-1, :], resized[:, 0], resized[:, -1]], axis=0)
    pad_color = [int(x) for x in np.median(border_vals, axis=0)]

    canvas = np.full((img_size, img_size, 3), pad_color, dtype=np.uint8)
    y0 = (img_size - nh) // 2
    x0 = (img_size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(norm.transpose(2, 0, 1), dtype=np.float32)


def _hardswish(x: np.ndarray) -> np.ndarray:
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def predict_forest_flat_numpy(meta: Dict[str, np.ndarray], x_1d: np.ndarray) -> np.ndarray:
    """ทำนายความน่าจะเป็นจาก Decision Trees แบบ Flat Pure-NumPy (ไม่พึ่ง scikit-learn/pickle)"""
    offsets = meta["tree_offsets"]
    cl = meta["tree_cl"]
    cr = meta["tree_cr"]
    feat = meta["tree_feat"]
    thr = meta["tree_thr"]
    probs = meta["tree_probs"]

    n_trees = len(offsets) - 1
    n_classes = int(probs.shape[1])
    total = np.zeros(n_classes, dtype=np.float64)

    for i in range(n_trees):
        base = int(offsets[i])
        node = 0
        while cl[base + node] != -1:
            if x_1d[feat[base + node]] <= thr[base + node]:
                node = int(cl[base + node])
            else:
                node = int(cr[base + node])
        total += probs[base + node]

    return (total / max(n_trees, 1)).astype(np.float32)


# =====================================================================
# Standard Interface Required by INFER_SPEC.md: load_model() & predict_image()
# =====================================================================

def _find_companion_npz(onnx_file: Path, weights_dir: Path) -> Path | None:
    """ค้นหาไฟล์ metadata (.npz) คู่กับไฟล์ .onnx โดยไม่จำกัดชื่อไฟล์"""
    candidates = [
        onnx_file.parent / f"{onnx_file.stem}_meta.npz",
        onnx_file.parent / f"{onnx_file.stem}.npz",
        onnx_file.parent / "model_meta.npz",
    ]
    for c in candidates:
        if c.is_file():
            return c
    same_dir_npz = sorted(onnx_file.parent.glob("*.npz"))
    if same_dir_npz:
        return same_dir_npz[0]
    recursive_npz = sorted(weights_dir.rglob("*.npz"))
    return recursive_npz[0] if recursive_npz else None


def _resolve_weights_source(weights_path: Path) -> Tuple[str, Path, Path | None]:
    """
    ค้นหาไฟล์ weight อัตโนมัติภายในโฟลเดอร์ weights/ (ไม่จำกัดชื่อไฟล์และรองรับหลายไฟล์ในโฟลเดอร์):
      - หากส่งโฟลเดอร์ weights/ เข้ามา: ค้นหาคู่ไฟล์ .onnx + .npz ก่อน (ทั้งชั้นนอกและโฟลเดอร์ย่อย)
        เพื่อให้รันได้ทันทีบน Raspberry Pi OS 32-bit/CPU โดยไม่ต้องใช้ PyTorch
      - หากไม่พบ .onnx + .npz จึงค้นหาไฟล์ PyTorch (.pt / .pth) ในโฟลเดอร์ให้อัตโนมัติ
      - ยังคงรองรับกรณีส่ง path ของไฟล์เดี่ยวเข้ามาตรงๆ ด้วย
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights path not found: {weights_path}")

    if weights_path.is_file():
        ext = weights_path.suffix.lower()
        if ext == ".onnx":
            meta_file = _find_companion_npz(weights_path, weights_path.parent)
            if meta_file is None:
                raise FileNotFoundError(f"Missing companion metadata .npz in: {weights_path.parent}")
            return "onnx", weights_path, meta_file
        if ext in (".pt", ".pth"):
            return "pt", weights_path, None
        raise ValueError(f"Unsupported weights file extension '{weights_path.suffix}'. Expected .onnx, .pt, or .pth")

    weights_dir = weights_path

    # 1. ค้นหาไฟล์ .onnx (ให้ความสำคัญกับชื่อ best.onnx ก่อน หากไม่มีให้เลือกไฟล์ .onnx ใดๆ ที่พบ)
    onnx_all = sorted(
        list({p.resolve(): p for p in list(weights_dir.glob("*.onnx")) + list(weights_dir.rglob("*.onnx"))}.values()),
        key=lambda p: (0 if p.name == "best.onnx" else 1, len(p.parts), p.name.lower()),
    )
    for onnx_file in onnx_all:
        meta_file = _find_companion_npz(onnx_file, weights_dir)
        if meta_file is not None and meta_file.is_file():
            return "onnx", onnx_file, meta_file

    # 2. Fallback: ค้นหาไฟล์ .pt / .pth ในโฟลเดอร์ (รองรับไม่จำกัดชื่อไฟล์)
    pt_all = sorted(
        list({
            p.resolve(): p
            for ext in ("*.pt", "*.pth")
            for p in list(weights_dir.glob(ext)) + list(weights_dir.rglob(ext))
        }.values()),
        key=lambda p: (0 if p.name == "best.pt" else 1, len(p.parts), p.name.lower()),
    )
    if pt_all:
        return "pt", pt_all[0], None

    raise FileNotFoundError(f"No usable weight files (.onnx+.npz or .pt/.pth) found inside: {weights_dir}")


def _build_pytorch_hybrid_net(num_classes: int, num_phys_features: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import mobilenet_v3_small

    class _HybridBeanNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = mobilenet_v3_small(weights=None).features
            self.phys_mlp = nn.Sequential(
                nn.Linear(num_phys_features, 64),
                nn.BatchNorm1d(64),
                nn.Hardswish(inplace=True),
                nn.Dropout(0.20),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.Hardswish(inplace=True),
            )
            self.fusion_proj = nn.Sequential(
                nn.Linear(576 + 64, 192),
                nn.BatchNorm1d(192),
                nn.Hardswish(inplace=True),
                nn.Dropout(0.30),
            )
            self.classifier = nn.Linear(192, num_classes)

        def forward(self, img: torch.Tensor, phys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            fmap = self.features(img)
            conv_vec = 0.7 * F.adaptive_avg_pool2d(fmap, 1).flatten(1) + 0.3 * F.adaptive_max_pool2d(fmap, 1).flatten(1)
            emb = self.fusion_proj(torch.cat([conv_vec, self.phys_mlp(phys)], dim=1))
            return self.classifier(emb), F.normalize(emb, p=2, dim=1)

    return _HybridBeanNet()


def load_model(weights_path: Path | str) -> Dict[str, Any]:
    """
    โหลดโมเดลจาก weights/ (รองรับทั้ง OpenCV DNN .onnx + .npz และไฟล์ PyTorch .pt)
    """
    mode, primary_file, meta_file = _resolve_weights_source(Path(weights_path))

    if mode == "onnx":
        try:
            net = cv2.dnn.readNetFromONNX(str(primary_file))
            if int(cv2.__version__.split(".")[0]) < 5:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as exc:
            raise RuntimeError(f"Failed to load ONNX model via OpenCV DNN '{primary_file}': {exc}") from exc

        try:
            with np.load(meta_file, allow_pickle=False) as data:
                meta = {k: data[k] for k in data.files}
        except Exception as exc:
            raise RuntimeError(f"Failed to load metadata '{meta_file}': {exc}") from exc

        classes = [str(c) for c in meta["classes"].tolist()]
        if classes != CLASSES:
            raise ValueError(f"Model classes {classes} do not match expected spec classes {CLASSES}")

        return {
            "backend": "onnx",
            "net": net,
            "meta": meta,
            "classes": classes,
            "img_size": int(meta["img_size"][0]),
            "weights_file": str(primary_file),
        }

    # PyTorch .pt mode (กรณีผู้ใช้ระบุไฟล์ .pt โดยตรง เช่น weights/best2.pt)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot load '{primary_file}' because PyTorch is not installed on this device. "
            "For 32-bit Raspberry Pi OS, pass the weights/ directory containing best.onnx + model_meta.npz."
        ) from exc

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        ckpt = torch.load(primary_file, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load PyTorch checkpoint '{primary_file}': {exc}") from exc

    classes = list(ckpt.get("classes", CLASSES))
    if classes != CLASSES:
        raise ValueError(f"Model classes {classes} do not match expected spec classes {CLASSES}")

    img_size = int(ckpt.get("img_size", 224))
    num_phys = int(ckpt.get("num_phys_features", NUM_PHYSICAL_FEATURES))
    pt_model = _build_pytorch_hybrid_net(len(classes), num_phys)
    pt_model.load_state_dict(ckpt["model_state_dict"])
    pt_model.eval()

    meta: Dict[str, Any] = {
        "feat_mean": np.asarray(ckpt["feat_mean"], dtype=np.float32),
        "feat_std": np.asarray(ckpt["feat_std"], dtype=np.float32),
        "class_prototypes": np.asarray(ckpt["class_prototypes"], dtype=np.float32),
    }
    if ckpt.get("rf_trees"):
        rf_trees = ckpt["rf_trees"]
        offsets = np.zeros(len(rf_trees) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum([len(t["children_left"]) for t in rf_trees])
        meta.update(
            {
                "tree_offsets": offsets,
                "tree_cl": np.concatenate([t["children_left"] for t in rf_trees]).astype(np.int32),
                "tree_cr": np.concatenate([t["children_right"] for t in rf_trees]).astype(np.int32),
                "tree_feat": np.concatenate([t["feature"] for t in rf_trees]).astype(np.int32),
                "tree_thr": np.concatenate([t["threshold"] for t in rf_trees]).astype(np.float32),
                "tree_probs": np.vstack([t["probs"] for t in rf_trees]).astype(np.float32),
            }
        )
    elif ckpt.get("rf_classifier") is not None:
        meta["rf_classifier"] = ckpt["rf_classifier"]

    return {
        "backend": "pt",
        "pt_model": pt_model,
        "meta": meta,
        "classes": classes,
        "img_size": img_size,
        "weights_file": str(primary_file),
    }


def predict_image(image_path: Path | str, model_bundle: Dict[str, Any]) -> str:
    """
    ทำนายผล 1 คลาสต่อ 1 ภาพ (ทุกเมล็ดในภาพถือเป็นคลาสเดียวกัน)
    ใช้ Preprocessing แบบคงที่ (ไม่มี random augmentation หรือ TTA)
    """
    img_bgr = load_image_exif_corrected(Path(image_path))
    beans = detect_beans_in_frame(img_bgr)

    backend = model_bundle["backend"]
    meta: Dict[str, Any] = model_bundle["meta"]
    classes: List[str] = model_bundle["classes"]
    img_size: int = model_bundle["img_size"]

    feat_mean = meta["feat_mean"]
    feat_std = meta["feat_std"]
    prototypes = meta["class_prototypes"]

    total_image_prob = np.zeros(len(classes), dtype=np.float64)

    for b in beans:
        phys_raw = extract_bean_physical_features(b["crop_bgr"], b["rel_area"])
        phys_norm = np.clip((phys_raw - feat_mean) / feat_std, -5.0, 5.0).astype(np.float32)
        nchw = preprocess_crop_to_nchw(b["crop_bgr"], img_size)

        if backend == "onnx":
            net = model_bundle["net"]
            net.setInput(nchw[np.newaxis, :, :, :])
            conv_vec = net.forward()[0]
            p_vec = _hardswish(phys_norm @ meta["W_p1"].T + meta["b_p1"])
            p_vec = _hardswish(p_vec @ meta["W_p2"].T + meta["b_p2"])
            emb = _hardswish(np.concatenate([conv_vec, p_vec], axis=0) @ meta["W_fus"].T + meta["b_fus"])
            logits = emb @ meta["W_cls"].T + meta["b_cls"]
            cnn_prob = _softmax(logits)
            emb_norm = emb / max(float(np.linalg.norm(emb)), 1e-12)
        else:
            import torch
            import torch.nn.functional as F

            pt_model = model_bundle["pt_model"]
            with torch.no_grad():
                logits_t, emb_t = pt_model(
                    torch.from_numpy(nchw).unsqueeze(0),
                    torch.from_numpy(phys_norm).unsqueeze(0),
                )
                cnn_prob = F.softmax(logits_t, dim=1)[0].cpu().numpy()
                emb_norm = emb_t[0].cpu().numpy()

        proto_prob = _softmax(np.dot(prototypes, emb_norm) * 6.0)

        if "tree_offsets" in meta:
            tree_prob = predict_forest_flat_numpy(meta, phys_norm)
            bean_prob = 0.55 * cnn_prob + 0.20 * proto_prob + 0.25 * tree_prob
        elif "rf_classifier" in meta:
            tree_prob = meta["rf_classifier"].predict_proba(phys_norm.reshape(1, -1))[0]
            bean_prob = 0.55 * cnn_prob + 0.20 * proto_prob + 0.25 * tree_prob
        else:
            bean_prob = 0.75 * cnn_prob + 0.25 * proto_prob

        total_image_prob += bean_prob

    pred_class = classes[int(np.argmax(total_image_prob))]
    if pred_class not in VALID_CLASSES_SET:
        raise RuntimeError(f"Predicted invalid class '{pred_class}' for '{image_path}'")
    return pred_class


# =====================================================================
# CLI Runner & Output Spec Enforcement
# =====================================================================

def collect_input_images(input_path: Path, out_path: Path) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    out_resolved = out_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in VALID_IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension for '{input_path}'. Expected .jpg, .jpeg, or .png")
        return [input_path]

    if input_path.is_dir():
        # ใช้ไฟล์ภาพในระดับชั้นนอกสุดของโฟลเดอร์ก่อน เพื่อป้องกันไฟล์ซ้ำจากโฟลเดอร์ย่อย (เช่น datasets/)
        top_level = [
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
        ]
        if top_level:
            candidates = top_level
        else:
            candidates = []
            for p in input_path.rglob("*"):
                if not p.is_file():
                    continue
                try:
                    p.resolve().relative_to(out_resolved)
                    continue
                except ValueError:
                    pass
                if p.suffix.lower() in VALID_IMAGE_EXTS:
                    candidates.append(p)

        candidates.sort(key=lambda p: (p.name.lower(), str(p)))
        if not candidates:
            raise FileNotFoundError(f"No valid images (.jpg/.jpeg/.png) found in: {input_path}")

        seen_names: Dict[str, Path] = {}
        for p in candidates:
            if p.name in seen_names:
                raise ValueError(
                    f"Duplicate input filename '{p.name}' found at '{seen_names[p.name]}' and '{p}'. "
                    "Input filenames must be unique."
                )
            seen_names[p.name] = p
        return candidates

    raise ValueError(f"Input path is neither a file nor a directory: {input_path}")


def prepare_output_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls_name in CLASSES:
        (out_dir / cls_name).mkdir(parents=True, exist_ok=True)


def validate_predictions_csv(csv_path: Path, expected_filenames: List[str]) -> None:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))
    if not reader or reader[0] != ["filename", "class"]:
        raise RuntimeError(f"Invalid CSV header in {csv_path}")
    rows = reader[1:]
    if len(rows) != len(expected_filenames):
        raise RuntimeError(f"Row count mismatch in {csv_path}: expected {len(expected_filenames)}, got {len(rows)}")
    seen = set()
    for fname, cls_name in rows:
        if fname in seen:
            raise RuntimeError(f"Duplicate filename in CSV: {fname}")
        if cls_name not in VALID_CLASSES_SET:
            raise RuntimeError(f"Invalid class '{cls_name}' in CSV for '{fname}'")
        seen.add(fname)
    if seen != set(expected_filenames):
        raise RuntimeError("Output filenames do not match input filenames")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coffee Bean Image Classification Inference (Day 1)")
    parser.add_argument("--input", type=Path, required=True, help="Path to test image directory or single image file")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to weights folder (auto-discovers weight files inside)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Path to output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_cpu_runtime()

    try:
        if args.input.resolve() == args.out.resolve():
            raise ValueError("--input and --out directories must be separate")

        image_files = collect_input_images(args.input, args.out)
        model_bundle = load_model(args.weights)
        prepare_output_dirs(args.out)

        rows: List[Tuple[str, str]] = []
        for img_path in image_files:
            pred_class = predict_image(img_path, model_bundle)
            dest_file = args.out / pred_class / img_path.name
            shutil.copy2(img_path, dest_file)
            rows.append((img_path.name, pred_class))

        csv_path = args.out / "predictions.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "class"])
            writer.writerows(rows)

        validate_predictions_csv(csv_path, [p.name for p in image_files])
        print(f"[OK] Processed {len(rows)} image(s) -> {csv_path}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
