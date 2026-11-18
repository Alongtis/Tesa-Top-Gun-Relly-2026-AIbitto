#!/usr/bin/env python3
"""
prepare_day2_dataset.py — Day 2 Instance Segmentation Dataset & Synthetic Pile Generator
สำหรับ TESA Top Gun Rally 2026 (Edge AI Track - Day 2)

ฟีเจอร์หลัก:
  1. Coarse-to-Fine 1px Auto-Masking:
     - ค้นหาตำแหน่งเมล็ดด้วย CIE L*a*b* Local Illumination Normalization
     - ซูมเข้าไปสกัดขอบ Mask ระดับพิกเซลจริง (1:1 Native Resolution) ในแต่ละกรอบเมล็ด
     - บังคับ Topology ตาม DAY2_INFER_SPEC.md (Single Connected Component + ถมรูใน + CHAIN_APPROX_NONE)
     - รับประกันความตรงกัน 100% ระหว่าง uint16 seg_mask/*.png และ polygon ใน JSON
  2. Active-Region Cropping (ตามสไลด์หน้า 6 "ภาพครอปเฉพาะบริเวณเมล็ด"):
     - ครอปเฉพาะบริเวณกลุ่มเมล็ดจากภาพใหญ่ 3072x4096 เพื่อให้เมล็ดมีความละเอียดสูงขึ้น 3.5 เท่าเมื่อเข้า YOLO-seg
  3. Realistic Synthetic Pile Generator (แก้ปัญหา Train เม็ดห่าง vs Test เม็ดวางกองชิด/ซ้อนกัน):
     - สุ่มนำเมล็ดจากคลัง 175 เม็ดมาหมุน/ปรับขนาดเล็กน้อย แล้วจัดเรียงเป็น "กองเมล็ดที่ชิดและเกยกัน"
     - รองรับทั้งพื้นหลังกระดาษจริง พื้นหลังลายไม้จำลอง (เหมือนสไลด์หน้า 3/12/22) และป้ายชื่อคลาสจำลอง (Negative Sample)
  4. Quality Inspection Visuals:
     - สร้างแผ่น Dashboard ตรวจสอบคุณภาพMask ขอบ 1px และภาพกองจำลองในโฟลเดอร์ task/quality_inspection/
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# รองรับการรันผ่าน Python ใน venv ของโปรเจกต์บน Windows
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

# ลำดับคลาสตามสเปก DAY2_INFER_SPEC.md (ตัวพิมพ์เล็กทั้งหมด)
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

# สีประจำคลาส (BGR) สำหรับวาดภาพ Visualization ให้แยกง่ายด้วยสายตา
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "good": (235, 95, 45),       # ฟ้าคราม (เหมือนสไลด์หน้า 22/46)
    "shell": (200, 210, 50),     # เขียวอมฟ้า (Cyan/Teal เหมือนสไลด์หน้า 22/46)
    "black": (190, 180, 175),    # เทาเงินสว่าง (เห็นชัดบนเม็ดดำ)
    "husk": (50, 70, 230),       # แดงส้ม
    "broken": (40, 195, 60),     # เขียวสด
    "parchment": (40, 210, 245), # เหลืองทอง
    "immature": (220, 80, 210),  # ม่วงชมพู
}


def load_image_exif_corrected(img_path: Path) -> np.ndarray:
    """อ่านภาพพร้อมปรับหมุนตาม EXIF Orientation ให้ถูกต้องตาม DAY2_INFER_SPEC.md"""
    with Image.open(img_path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def smooth_closed_contour(cnt: np.ndarray, sigma: float = 3.8) -> np.ndarray:
    """
    ปรับเส้นขอบ Contour ปิดให้เรียบเนียนระดับ Sub-pixel ด้วย Periodic 1D Gaussian Smoothing
    ขจัดรอยหยักขั้นบันได (staircase jitter) โดยคงรูปทรงจริงของเมล็ดกาแฟ
    """
    pts = cnt.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    if n < 12 or sigma <= 0.0:
        return cnt

    radius = max(2, int(math.ceil(sigma * 3.0)))
    k_idx = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (k_idx / sigma) ** 2)
    kernel /= np.sum(kernel)

    # ต่อหัวท้ายแบบวงกลม (Periodic Wrapping) เพื่อไม่ให้จุดเริ่มต้น/จุดจบของเส้นเกิดรอยหัก
    pad_pts = np.vstack([pts[-radius:], pts, pts[:radius]])
    xs_smooth = np.convolve(pad_pts[:, 0], kernel, mode="valid")
    ys_smooth = np.convolve(pad_pts[:, 1], kernel, mode="valid")
    smoothed = np.stack([ np.round(xs_smooth), np.round(ys_smooth) ], axis=1).astype(np.int32)
    return smoothed.reshape((-1, 1, 2))


def ensure_single_clean_component(
    binary_mask: np.ndarray,
    smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    บังคับให้ binary_mask ของเมล็ด 1 เม็ดเป็น Single Connected Component ที่ไม่มีรูด้านใน
    พร้อมปรับเส้นขอบให้เรียบเนียนด้วย Periodic Gaussian Contour Smoothing (ถ้ากำหนด smooth_sigma > 0)
    และคืนค่า (clean_mask_u8, external_contour_chain_approx_none)
    เพื่อให้ตรงตาม DAY2_INFER_SPEC.md บรรทัด 79-81 ร้อยเปอร์เซ็นต์
    """
    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros_like(binary_mask), np.empty((0, 1, 2), dtype=np.int32)

    best_cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best_cnt) < 16.0 or len(best_cnt) < 3:
        return np.zeros_like(binary_mask), np.empty((0, 1, 2), dtype=np.int32)

    if smooth_sigma > 0.0 and len(best_cnt) >= 16:
        best_cnt = smooth_closed_contour(best_cnt, sigma=smooth_sigma)

    clean_mask = np.zeros_like(binary_mask)
    cv2.drawContours(clean_mask, [best_cnt], -1, 255, thickness=-1)

    # ดึง contour อีกครั้งจาก mask ที่วาดแล้วเพื่อให้พิกัดตรงกันระดับพิกเซล 100%
    final_cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not final_cnts:
        return np.zeros_like(binary_mask), np.empty((0, 1, 2), dtype=np.int32)
    final_cnt = max(final_cnts, key=cv2.contourArea)
    return clean_mask, final_cnt


def refine_bean_mask_fullres(
    img_bgr: np.ndarray,
    coarse_cnt_orig: np.ndarray,
    pad_px: int = 72,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    ปรับความคมชัดและความเรียบของขอบ Mask เมล็ดกาแฟที่ความละเอียดภาพจริง (1:1 Full Resolution)
    แก้ปัญหาเมล็ด 2 สี (เช่น black ครึ่งเหลืองครึ่งดำ, shell ขอบสว่าง, immature ขอบสะท้อนแสง):
      - ไม่ใช้ Per-Bean Otsu (เพราะสีดำเข้มจะดึง Threshold จนกินเนื้อเมล็ดสีอ่อนแหว่ง)
      - ใช้ Background-Referenced Chroma + Saturation + Shadow-Suppressed Darkness Score
      - ใช้ Morphological Closing + Signed Distance Field (SDF) + Periodic Gaussian Curve Smoothing
    """
    orig_h, orig_w = img_bgr.shape[:2]
    x, y, w, h = cv2.boundingRect(coarse_cnt_orig)
    x1 = max(0, x - pad_px)
    y1 = max(0, y - pad_px)
    x2 = min(orig_w, x + w + pad_px)
    y2 = min(orig_h, y + h + pad_px)

    roi_bgr = img_bgr[y1:y2, x1:x2]
    rh, rw = roi_bgr.shape[:2]

    coarse_local = coarse_cnt_orig.copy()
    coarse_local[:, 0, 0] -= x1
    coarse_local[:, 0, 1] -= y1
    coarse_mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.drawContours(coarse_mask, [coarse_local], -1, 255, thickness=-1)

    # โซนพื้นหลังกระดาษรอบนอก (ห่างจากตัวเมล็ดออกไป 35px เพื่อไม่ให้โดนเงาใกล้ขอบเมล็ด)
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
    bg_ring = cv2.dilate(coarse_mask, k_bg, iterations=1) == 0

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_c, a_c, b_c = cv2.split(lab)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    S_c = hsv[:, :, 1]

    if np.count_nonzero(bg_ring) >= 80:
        bg_L = float(np.median(L_c[bg_ring]))
        bg_a = float(np.median(a_c[bg_ring]))
        bg_b = float(np.median(b_c[bg_ring]))
        bg_S = float(np.median(S_c[bg_ring]))
    else:
        bg_L = float(np.percentile(L_c, 88))
        bg_a = float(np.median(a_c))
        bg_b = float(np.median(b_c))
        bg_S = float(np.percentile(S_c, 15))

    # 1. แยกเมล็ดออกจากพื้นหลังและตัดเงาข้างเมล็ด (Shadow-Suppressed + Capped Otsu):
    #    - เงาบนกระดาษขาว: มืดลงเล็กน้อย (dark_diff ~ 10-25) แต่ไม่มีสี (chroma_diff ~ 1-4, sat_diff ~ 0-6)
    #    - ขอบเมล็ดสีอ่อน (shell, immature, ครึ่งเหลืองของ black): มีสีชัดเจน (chroma_diff >= 10, sat_diff >= 22)
    #    - สาเหตุที่ Otsu ปกติกินเนื้อเมล็ด 2 สี: ส่วนสีดำสนิทมีคะแนนสูงมาก (~200) จึงดึงค่าเฉลี่ย Otsu ขึ้นไปที่ ~95
    #      วิธีแก้: จำกัดเพดานคะแนนสูงสุด (Clip ที่ 88.0) ก่อนคำนวณ Otsu เพื่อให้ส่วนมืดและส่วนสว่างของเมล็ดอยู่กลุ่มเดียวกัน!
    dark_diff = np.clip(bg_L - L_c, 0.0, 255.0)
    chroma_diff = np.sqrt((a_c - bg_a) ** 2 + (b_c - bg_b) ** 2)
    sat_diff = np.clip(S_c - bg_S, 0.0, 255.0)

    # หักความมืดระดับเงาจางๆ (20.0) ออก เพื่อไม่ให้เงาตกกระทบรอบเมล็ดติดเข้ามาเป็น Halo
    shadow_free_dark = np.maximum(0.0, dark_diff - 20.0) * 0.95
    raw_fg = shadow_free_dark + chroma_diff * 2.55 + sat_diff * 0.55
    raw_fg = cv2.GaussianBlur(raw_fg, (5, 5), 0)

    # จำกัดเพดานที่ 88.0 เพื่อไม่ให้ส่วนสีดำดึง Otsu ทิ้งส่วนสีอ่อนของเมล็ด
    capped_fg = np.clip(raw_fg, 0.0, 88.0).astype(np.uint8)

    search_zone = cv2.dilate(
        coarse_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65)),
        iterations=1,
    )
    valid_vals = capped_fg[search_zone > 0]
    otsu_t, _ = cv2.threshold(valid_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # บังคับช่วง Threshold ให้อยู่เหนือเงากระดาษ (>= 36) แต่ไม่สูงเกินขอบเมล็ดสีอ่อน (<= 50)
    thresh_val = float(np.clip(float(otsu_t) * 1.04, 36.0, 50.0))
    fine_bin = ((capped_fg >= thresh_val) & (search_zone > 0)).astype(np.uint8) * 255

    # 2. ปิดรอยแหว่งและเชื่อมเนื้อเมล็ดให้เป็นก้อนเดียวกันโดยไม่ขยายขอบออกนอกเมล็ด
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fine_bin = cv2.morphologyEx(fine_bin, cv2.MORPH_CLOSE, k_close, iterations=2)
    fine_bin = cv2.morphologyEx(fine_bin, cv2.MORPH_OPEN, k_open, iterations=1)

    # ถมรูด้านในก่อนทำ SDF smoothing
    fine_bin, _ = ensure_single_clean_component(fine_bin, smooth_sigma=0.0)

    # 3. Signed Distance Field (SDF) + Gaussian Smoothing เพื่อลบมุมเหลี่ยมและทำให้ขอบโค้งมนตามธรรมชาติ
    fine_blur = cv2.GaussianBlur(fine_bin, (11, 11), 2.8)
    _, fine_smooth = cv2.threshold(fine_blur, 127, 255, cv2.THRESH_BINARY)

    # 4. Periodic 1D Gaussian Contour Smoothing (sigma=4.0) เพื่อให้เส้นขอบเรียบเนียนไร้ฟันปลา 100%
    clean_local_mask, clean_local_cnt = ensure_single_clean_component(fine_smooth, smooth_sigma=4.0)
    if len(clean_local_cnt) < 3 or cv2.contourArea(clean_local_cnt) < 0.5 * cv2.contourArea(coarse_local):
        clean_local_mask, clean_local_cnt = ensure_single_clean_component(coarse_mask, smooth_sigma=4.2)

    cnt_global = clean_local_cnt.copy()
    cnt_global[:, 0, 0] += x1
    cnt_global[:, 0, 1] += y1

    bx, by, bw, bh = cv2.boundingRect(cnt_global)
    return clean_local_mask, cnt_global, (bx, by, bx + bw, by + bh)


def detect_and_segment_beans_1px(
    img_bgr: np.ndarray,
    work_scale: float = 0.25,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    สกัด Instance Masks และ Polygons ระดับ 1px ของเมล็ดกาแฟทุกเม็ดในภาพ
    คืนค่า (รายการเมล็ดเรียงตามตำแหน่ง, ภาพ seg_mask uint16 ขนาดเท่าภาพจริง)
    """
    orig_h, orig_w = img_bgr.shape[:2]
    ws_w = int(round(orig_w * work_scale))
    ws_h = int(round(orig_h * work_scale))
    small = cv2.resize(img_bgr, (ws_w, ws_h), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_chan, a_chan, b_chan = cv2.split(lab)

    bg_L = cv2.GaussianBlur(L_chan, (151, 151), 0)
    bg_a = cv2.GaussianBlur(a_chan, (151, 151), 0)
    bg_b = cv2.GaussianBlur(b_chan, (151, 151), 0)

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

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k5, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=1)

    raw_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = ws_w * ws_h
    min_bean_area = img_area * 0.00045
    max_bean_area = img_area * 0.04000

    valid_areas = [cv2.contourArea(c) for c in raw_contours if min_bean_area <= cv2.contourArea(c) <= max_bean_area]
    median_area = float(np.median(valid_areas)) if valid_areas else (img_area * 0.002)

    coarse_contours = []
    for cnt in raw_contours:
        area = cv2.contourArea(cnt)
        if not (min_bean_area <= area <= max_bean_area):
            continue
        if area > median_area * 1.55:
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
                        coarse_contours.append(max(sub_cnts, key=cv2.contourArea))
                continue
        coarse_contours.append(cnt)

    raw_beans: List[Dict[str, Any]] = []
    for cnt in coarse_contours:
        area = cv2.contourArea(cnt)
        if not (min_bean_area <= area <= max_bean_area):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) / max(min(w, h), 1) > 3.5:
            continue

        cnt_orig = (cnt.astype(np.float32) / work_scale).astype(np.int32)
        _, fine_cnt_global, bbox_tight = refine_bean_mask_fullres(img_bgr, cnt_orig)
        if len(fine_cnt_global) < 3:
            continue

        tx1, ty1, tx2, ty2 = bbox_tight
        raw_beans.append(
            {
                "bbox_tight": bbox_tight,
                "center": ((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0),
                "contour": fine_cnt_global,
            }
        )

    # เรียงลำดับจากบนลงล่าง ซ้ายไปขวา
    raw_beans.sort(key=lambda b: (int(b["center"][1] // 250), b["center"][0]))

    # วาดลงบน seg_mask (uint16) เพื่อป้องกันพิกเซลซ้อนทับกัน 100%
    seg_mask = np.zeros((orig_h, orig_w), dtype=np.uint16)
    for idx, b in enumerate(raw_beans, start=1):
        cv2.drawContours(seg_mask, [b["contour"]], -1, int(idx), thickness=-1)

    # สกัด Polygon จาก seg_mask จริงเพื่อให้ตรงตาม DAY2_INFER_SPEC.md ทุกพิกเซล
    final_beans: List[Dict[str, Any]] = []
    new_seg_mask = np.zeros((orig_h, orig_w), dtype=np.uint16)
    cur_id = 1
    for idx, b in enumerate(raw_beans, start=1):
        tx1, ty1, tx2, ty2 = b["bbox_tight"]
        sub = (seg_mask[ty1:ty2, tx1:tx2] == idx).astype(np.uint8) * 255
        clean_sub, local_cnt = ensure_single_clean_component(sub)
        if len(local_cnt) < 3:
            continue

        new_seg_mask[ty1:ty2, tx1:tx2][clean_sub > 0] = cur_id
        global_cnt = local_cnt.copy()
        global_cnt[:, 0, 0] += tx1
        global_cnt[:, 0, 1] += ty1

        gx, gy, gw, gh = cv2.boundingRect(global_cnt)
        poly_pts = [[int(pt[0][0]), int(pt[0][1])] for pt in global_cnt]
        crop_mask_u8 = (new_seg_mask[gy : gy + gh, gx : gx + gw] == cur_id).astype(np.uint8) * 255
        final_beans.append(
            {
                "id": cur_id,
                "bbox_tight": (gx, gy, gx + gw, gy + gh),
                "center": (gx + gw / 2.0, gy + gh / 2.0),
                "contour": global_cnt,
                "polygon": poly_pts,
                "mask_crop": crop_mask_u8,
                "area_px": int(np.count_nonzero(crop_mask_u8)),
            }
        )
        cur_id += 1

    return final_beans, new_seg_mask


def compute_active_pile_roi(
    beans: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.16,
) -> Tuple[int, int, int, int]:
    """คำนวณกรอบครอปเฉพาะบริเวณกลุ่มเมล็ด (ตามคำแนะนำในสไลด์หน้า 6)"""
    if not beans:
        return (0, 0, img_w, img_h)
    min_x = min(b["bbox_tight"][0] for b in beans)
    min_y = min(b["bbox_tight"][1] for b in beans)
    max_x = max(b["bbox_tight"][2] for b in beans)
    max_y = max(b["bbox_tight"][3] for b in beans)

    span_w = max(max_x - min_x, 400)
    span_h = max(max_y - min_y, 400)
    side = int(round(max(span_w, span_h) * (1.0 + 2.0 * margin_ratio)))
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2

    rx1 = max(0, cx - side // 2)
    ry1 = max(0, cy - side // 2)
    rx2 = min(img_w, rx1 + side)
    ry2 = min(img_h, ry1 + side)
    rx1 = max(0, rx2 - side)
    ry1 = max(0, ry2 - side)
    return (rx1, ry1, rx2, ry2)


def render_vis_overlay(
    img_bgr: np.ndarray,
    seg_mask: np.ndarray,
    objects: List[Dict[str, Any]],
    title_banner: str = "",
) -> np.ndarray:
    """วาดภาพ Overlay mask + เส้นขอบ Polygon + ป้ายชื่อคลาส/ID สำหรับตรวจด้วยสายตา"""
    vis = img_bgr.copy()
    overlay = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    line_t = max(2, int(round(max(h, w) / 700)))
    font_scale = max(0.5, max(h, w) / 1800.0)

    for obj in objects:
        obj_id = int(obj["id"])
        cls_name = obj["class"]
        color = CLASS_COLORS_BGR.get(cls_name, (0, 220, 0))
        pts = np.array(obj["polygon"], dtype=np.int32).reshape((-1, 1, 2))

        m = seg_mask == obj_id
        overlay[m] = color
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 255, 255), thickness=line_t + 1, lineType=cv2.LINE_AA)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=line_t, lineType=cv2.LINE_AA)

    vis = cv2.addWeighted(overlay, 0.42, vis, 0.58, 0.0)

    for obj in objects:
        obj_id = int(obj["id"])
        cls_name = obj["class"]
        color = CLASS_COLORS_BGR.get(cls_name, (0, 220, 0))
        pts = np.array(obj["polygon"], dtype=np.int32)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        label = f"#{obj_id} {cls_name}"
        cv2.putText(vis, label, (max(5, cx - 35), cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (15, 15, 15), line_t + 2, cv2.LINE_AA)
        cv2.putText(vis, label, (max(5, cx - 35), cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, line_t - 1), cv2.LINE_AA)

    if title_banner:
        bh = max(44, int(h * 0.048))
        cv2.rectangle(vis, (0, 0), (w, bh), (25, 25, 25), -1)
        cv2.putText(
            vis,
            title_banner,
            (18, int(bh * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 1.15,
            (0, 240, 255),
            max(2, line_t),
            cv2.LINE_AA,
        )
    return vis


def generate_wood_background(width: int, height: int, rng: random.Random) -> np.ndarray:
    """สร้างพื้นหลังลายไม้เหมือนโต๊ะในสไลด์หน้า 1, 3, 12, 22, 46"""
    y_grid = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x_grid = np.linspace(0, 1, width, dtype=np.float32)[None, :]

    freq1 = rng.uniform(14.0, 24.0)
    freq2 = rng.uniform(35.0, 55.0)
    phase = rng.uniform(0.0, 6.28)
    warp = 0.025 * np.sin(x_grid * 6.0 + phase)

    stripes = (
        0.55 * np.sin((y_grid + warp) * freq1 * math.pi + phase)
        + 0.30 * np.sin((y_grid - warp * 0.5) * freq2 * math.pi)
        + 0.15 * np.cos(y_grid * 95.0 * math.pi)
    )
    noise = np.random.default_rng(rng.randint(0, 999999)).normal(0, 0.06, (height, width)).astype(np.float32)
    t = np.clip((stripes + 1.0) * 0.5 + noise, 0.0, 1.0)

    # โทนสีไม้สัก/โอ๊คตามภาพสไลด์ (BGR)
    dark_wood = np.array([58, 96, 132], dtype=np.float32)
    light_wood = np.array([105, 152, 192], dtype=np.float32)
    wood = (dark_wood[None, None, :] * (1.0 - t[:, :, None]) + light_wood[None, None, :] * t[:, :, None])
    wood_u8 = np.clip(wood, 0, 255).astype(np.uint8)
    wood_u8 = cv2.GaussianBlur(wood_u8, (9, 3), 0)
    return wood_u8


def generate_paper_background(
    paper_patches: List[np.ndarray],
    width: int,
    height: int,
    rng: random.Random,
) -> np.ndarray:
    """สร้างพื้นหลังกระดาษจริงพร้อมความลาดเอียงของแสงธรรมชาติ"""
    base = rng.choice(paper_patches)
    bg = cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # สุ่มความลาดเอียงของแสงเหมือนภาพถ่ายจริง
    gx = np.linspace(rng.uniform(-12, 12), rng.uniform(-12, 12), width, dtype=np.float32)[None, :, None]
    gy = np.linspace(rng.uniform(-12, 12), rng.uniform(-12, 12), height, dtype=np.float32)[:, None, None]
    tint = np.array([rng.uniform(-5, 5), rng.uniform(-4, 4), rng.uniform(-5, 5)], dtype=np.float32)[None, None, :]
    bg = np.clip(bg + gx + gy + tint, 155, 250).astype(np.uint8)
    return bg


def rotate_bean_rgba(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    angle_deg: float,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """หมุนและปรับขนาดเมล็ดเดี่ยวพร้อม Mask โดยไม่ให้ขอบขาด"""
    h, w = mask_u8.shape[:2]
    diag = int(math.ceil(math.hypot(w, h) * scale * 1.25)) + 8
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    M[0, 2] += diag / 2.0 - cx
    M[1, 2] += diag / 2.0 - cy

    rot_bgr = cv2.warpAffine(img_bgr, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    rot_mask = cv2.warpAffine(mask_u8, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    _, rot_mask = cv2.threshold(rot_mask, 127, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(rot_mask > 0)
    if len(xs) == 0:
        return img_bgr, mask_u8
    x1, x2 = max(0, int(xs.min()) - 2), min(diag, int(xs.max()) + 3)
    y1, y2 = max(0, int(ys.min()) - 2), min(diag, int(ys.max()) + 3)
    return rot_bgr[y1:y2, x1:x2], rot_mask[y1:y2, x1:x2]


def synthesize_pile_image(
    bean_bank: Dict[str, List[Dict[str, Any]]],
    paper_patches: List[np.ndarray],
    target_classes: List[str],
    num_beans: int,
    canvas_size: Tuple[int, int],
    bg_type: str,
    add_label_card: bool,
    rng: random.Random,
    touching_mode: str = "pile",
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    สร้างภาพกองเมล็ดกาแฟจำลอง (เม็ดวางชิดและซ้อนกันเล็กน้อยเหมือนภาพ Test Day 2 & Day 3)
    คืนค่า (canvas_bgr, seg_mask_u16, objects_list) ที่ตรงตามสเปก DAY2_INFER_SPEC.md 100%
    """
    cw, ch = canvas_size
    if bg_type == "wood":
        canvas = generate_wood_background(cw, ch, rng)
    else:
        canvas = generate_paper_background(paper_patches, cw, ch, rng)

    card_rect = None
    if add_label_card:
        # วาดป้ายชื่อคลาสจำลองด้านล่าง (Negative Sample ห้ามถูกนับเป็นเมล็ด)
        card_w = int(cw * rng.uniform(0.36, 0.46))
        card_h = int(ch * rng.uniform(0.11, 0.14))
        cx1 = (cw - card_w) // 2 + rng.randint(-40, 40)
        cy1 = ch - card_h - int(ch * 0.04)
        cx2, cy2 = cx1 + card_w, cy1 + card_h
        card_rect = (cx1, cy1, cx2, cy2)

        cv2.rectangle(canvas, (cx1 + 4, cy1 + 5), (cx2 + 4, cy2 + 5), (40, 40, 40), -1)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (242, 240, 236), -1)
        cv2.putText(
            canvas,
            target_classes[0].capitalize(),
            (cx1 + card_w // 4, cy1 + int(card_h * 0.65)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )

    raw_id_mask = np.zeros((ch, cw), dtype=np.uint16)
    placed_meta: Dict[int, str] = {}
    occupied_centers: List[Tuple[float, float, float]] = []

    max_y_limit = (card_rect[1] - 25) if card_rect is not None else (ch - 45)
    pile_cx = cw * rng.uniform(0.46, 0.54)
    pile_cy = min(ch * rng.uniform(0.45, 0.53), max_y_limit * 0.52)

    cur_raw_id = 1
    for _ in range(num_beans):
        cls_name = rng.choice(target_classes)
        sample = rng.choice(bean_bank[cls_name])

        # ปรับขนาดเมล็ดให้สอดคล้องกับขนาดภาพ (ประมาณ 80-135 px บนภาพ 960x960)
        orig_bh, orig_bw = sample["mask"].shape[:2]
        base_scale = (min(cw, ch) / 1080.0) * rng.uniform(0.84, 1.08)
        angle = rng.uniform(0.0, 360.0)
        b_bgr, b_mask = rotate_bean_rgba(sample["bgr"], sample["mask"], angle, base_scale)

        # ปรับแสงเล็กน้อยให้หลากหลาย
        gain = rng.uniform(0.93, 1.07)
        b_bgr = np.clip(b_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        bh, bw = b_mask.shape[:2]
        radius = 0.5 * math.hypot(bw, bh) * 0.42

        placed = False
        for attempt in range(120):
            if touching_mode == "dense_pile" and occupied_centers and rng.random() < 0.82:
                # วางชิดหรือเกยขอบเม็ดที่มีอยู่แล้วเล็กน้อย (จำลองกองเม็ดชิดและซ้อน)
                anchor_x, anchor_y, anchor_r = rng.choice(occupied_centers)
                theta = rng.uniform(0.0, 2.0 * math.pi)
                dist = (anchor_r + radius) * rng.uniform(0.76, 1.06)
                px = int(round(anchor_x + math.cos(theta) * dist - bw / 2.0))
                py = int(round(anchor_y + math.sin(theta) * dist - bh / 2.0))
            else:
                spread_x = cw * (0.28 if attempt < 60 else 0.36)
                spread_y = (max_y_limit - 50) * (0.28 if attempt < 60 else 0.36)
                px = int(round(rng.gauss(pile_cx, spread_x * 0.55) - bw / 2.0))
                py = int(round(rng.gauss(pile_cy, spread_y * 0.55) - bh / 2.0))

            if px < 20 or py < 20 or (px + bw) >= (cw - 20) or (py + bh) >= max_y_limit:
                continue

            roi_id = raw_id_mask[py : py + bh, px : px + bw]
            bean_bool = b_mask > 0
            bean_area = int(np.count_nonzero(bean_bool))
            if bean_area < 100:
                break

            overlap_pixels = int(np.count_nonzero((roi_id > 0) & bean_bool))
            # อนุญาตให้เกยขอบกันได้สูงสุด 12% ของพื้นที่เม็ด เพื่อจำลองเม็ดซ้อนโดยไม่ทับเม็ดล่างขาดเป็น 2 ท่อน
            if overlap_pixels > int(bean_area * 0.12):
                continue

            # ตรวจว่าเม็ดด้านล่างที่ถูกทับบางส่วนต้องไม่เหลือพื้นที่น้อยเกินไป
            affected_ids = np.unique(roi_id[(roi_id > 0) & bean_bool])
            too_much_occ = False
            for aid in affected_ids:
                rem = int(np.count_nonzero(raw_id_mask == aid)) - int(np.count_nonzero((roi_id == aid) & bean_bool))
                if rem < 250:
                    too_much_occ = True
                    break
            if too_much_occ:
                continue

            # วาดเงาตกกระทบเบาๆ ใต้เมล็ด
            sy1, sy2 = min(ch, py + 4), min(ch, py + bh + 4)
            sx1, sx2 = min(cw, px + 3), min(cw, px + bw + 3)
            sh_h, sh_w = sy2 - sy1, sx2 - sx1
            if sh_h > 0 and sh_w > 0:
                sh_mask = cv2.GaussianBlur(b_mask[:sh_h, :sh_w].astype(np.float32) / 255.0, (9, 9), 0)[:, :, None]
                canvas[sy1:sy2, sx1:sx2] = np.clip(
                    canvas[sy1:sy2, sx1:sx2].astype(np.float32) * (1.0 - 0.22 * sh_mask),
                    0,
                    255,
                ).astype(np.uint8)

            # เกลี่ยขอบ Alpha 1px ให้เนียนสนิทกับพื้นหลัง
            alpha = cv2.GaussianBlur(b_mask.astype(np.float32) / 255.0, (3, 3), 0)[:, :, None]
            roi_canvas = canvas[py : py + bh, px : px + bw].astype(np.float32)
            canvas[py : py + bh, px : px + bw] = np.clip(
                b_bgr.astype(np.float32) * alpha + roi_canvas * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)

            roi_id[bean_bool] = cur_raw_id
            placed_meta[cur_raw_id] = cls_name
            occupied_centers.append((px + bw / 2.0, py + bh / 2.0, radius))
            cur_raw_id += 1
            placed = True
            break

        if not placed:
            continue

    # สร้าง final seg_mask (1..N) และ objects list ที่รับประกัน Single Connected Component 100%
    final_seg_mask = np.zeros((ch, cw), dtype=np.uint16)
    final_objects: List[Dict[str, Any]] = []
    final_id = 1

    for rid, cls_name in placed_meta.items():
        ys, xs = np.where(raw_id_mask == rid)
        if len(xs) < 100:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        sub = (raw_id_mask[y1:y2, x1:x2] == rid).astype(np.uint8) * 255
        clean_sub, local_cnt = ensure_single_clean_component(sub, smooth_sigma=2.0)
        if len(local_cnt) < 3 or cv2.contourArea(local_cnt) < 100:
            continue

        final_seg_mask[y1:y2, x1:x2][clean_sub > 0] = final_id

        # ดึง contour อีกครั้งหลังเขียนลง final_seg_mask
        global_cnt = local_cnt.copy()
        global_cnt[:, 0, 0] += x1
        global_cnt[:, 0, 1] += y1
        poly_pts = [[int(pt[0][0]), int(pt[0][1])] for pt in global_cnt]

        final_objects.append(
            {
                "id": final_id,
                "class": cls_name,
                "polygon": poly_pts,
            }
        )
        final_id += 1

    return canvas, final_seg_mask, final_objects


def objects_to_yolo_seg_lines(objects: List[Dict[str, Any]], img_w: int, img_h: int) -> List[str]:
    """แปลงรายการ objects เป็นฟอร์แมต YOLO Segmentation: <class_idx> x1 y1 x2 y2 ... (พิกัด 0..1)"""
    lines: List[str] = []
    for obj in objects:
        cls_idx = CLASS_TO_IDX[obj["class"]]
        poly = obj["polygon"]
        # ลดจุดลงเล็กน้อยเฉพาะในไฟล์เทรน YOLO เพื่อไม่ให้บรรทัดยาวเกินไป แต่คงรูปทรงละเอียด (ส่วน JSON ส่งงานใช้ CHAIN_APPROX_NONE เต็ม)
        step = max(1, len(poly) // 90)
        sampled = poly[::step]
        if len(sampled) < 3:
            sampled = poly
        coords_str = " ".join(
            f"{min(1.0, max(0.0, pt[0] / img_w)):.6f} {min(1.0, max(0.0, pt[1] / img_h)):.6f}"
            for pt in sampled
        )
        lines.append(f"{cls_idx} {coords_str}")
    return lines


def save_day2_ground_truth_bundle(
    bundle_dir: Path,
    entries: List[Tuple[str, np.ndarray, np.ndarray, List[Dict[str, Any]]]],
) -> None:
    """
    บันทึกชุดข้อมูลในฟอร์แมตส่งงาน Day 2 เป๊ะๆ:
      - images/<image_name>
      - seg_mask/<image_stem>.png (uint16 single-channel)
      - ground_truth.json (เหมือน predictions.json ทุกประการ)
      - vis/<image_name>
    """
    img_dir = bundle_dir / "images"
    mask_dir = bundle_dir / "seg_mask"
    vis_dir = bundle_dir / "vis"
    for d in (img_dir, mask_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    gt_json_list: List[Dict[str, Any]] = []
    for img_name, img_bgr, seg_mask_u16, objects in sorted(entries, key=lambda x: x[0]):
        stem = Path(img_name).stem
        cv2.imwrite(str(img_dir / img_name), img_bgr)
        cv2.imwrite(str(mask_dir / f"{stem}.png"), seg_mask_u16.astype(np.uint16))

        vis = render_vis_overlay(
            img_bgr,
            seg_mask_u16,
            objects,
            title_banner=f"{img_name} | Beans: {len(objects)}",
        )
        # ย่อภาพ vis หากขนาดใหญ่เกิน 1600px เพื่อให้เปิดตรวจง่าย
        vh, vw = vis.shape[:2]
        if max(vh, vw) > 1600:
            s = 1600.0 / max(vh, vw)
            vis = cv2.resize(vis, (int(round(vw * s)), int(round(vh * s))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(vis_dir / img_name), vis)

        gt_json_list.append(
            {
                "image_name": img_name,
                "objects": objects,
            }
        )

    (bundle_dir / "ground_truth.json").write_text(
        json.dumps(gt_json_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_edge_zoom_sheet(bean_bank: Dict[str, List[Dict[str, Any]]], out_path: Path) -> None:
    """สร้างแผ่นภาพซูมระดับพิกเซล (1:1 Crop) ของทั้ง 7 คลาส เพื่อตรวจความแนบสนิทของขอบ Mask"""
    cell_w, cell_h = 260, 260
    cols = 4
    rows = len(CLASSES)
    header_h = 75
    canvas = np.full((header_h + rows * cell_h, (cols + 1) * cell_w, 3), 28, dtype=np.uint8)

    cv2.putText(
        canvas,
        "Day 2 Auto-Mask Quality Check: 1px Full-Resolution Boundary Fit (All 7 Classes)",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (0, 240, 255),
        2,
        cv2.LINE_AA,
    )

    for r_idx, cls_name in enumerate(CLASSES):
        y0 = header_h + r_idx * cell_h
        color = CLASS_COLORS_BGR[cls_name]
        samples = bean_bank[cls_name][:cols]

        # คอลัมน์แรก: ชื่อคลาสและสถิติ
        cv2.rectangle(canvas, (8, y0 + 8), (cell_w - 8, y0 + cell_h - 8), (48, 48, 48), -1)
        cv2.rectangle(canvas, (8, y0 + 8), (cell_w - 8, y0 + cell_h - 8), color, 2)
        cv2.putText(canvas, cls_name.upper(), (24, y0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"Total: {len(bean_bank[cls_name])} beans",
            (24, y0 + 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        avg_area = int(np.mean([s["area_px"] for s in bean_bank[cls_name]]))
        cv2.putText(
            canvas,
            f"Avg Area: {avg_area:,} px",
            (24, y0 + 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Green = 1px Contour",
            (24, y0 + 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 245, 80),
            1,
            cv2.LINE_AA,
        )

        for c_idx, s in enumerate(samples):
            x0 = (c_idx + 1) * cell_w
            bgr = s["bgr_padded"].copy()
            mask_p = s["mask_padded"]

            cnts, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            overlay = bgr.copy()
            overlay[mask_p > 0] = color
            blended = cv2.addWeighted(overlay, 0.25, bgr, 0.75, 0)
            cv2.drawContours(blended, cnts, -1, (0, 255, 0), 2, cv2.LINE_AA)

            bh, bw = blended.shape[:2]
            scale = min((cell_w - 24) / max(bw, 1), (cell_h - 24) / max(bh, 1))
            nw, nh = max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))
            resized = cv2.resize(blended, (nw, nh), interpolation=cv2.INTER_AREA)

            py = y0 + (cell_h - nh) // 2
            px = x0 + (cell_w - nw) // 2
            canvas[py : py + nh, px : px + nw] = resized
            cv2.putText(
                canvas,
                f"{s['source']} ({ len(cnts[0]) if cnts else 0 } pts)",
                (x0 + 14, y0 + cell_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(out_path), canvas)


def create_stats_dashboard(
    real_counts: Dict[str, int],
    train_img_count: int,
    val_img_count: int,
    total_synth_beans: int,
    out_path: Path,
) -> None:
    """วาดแผ่นสรุปสถิติชุดข้อมูล Day 2 ด้วย OpenCV (ไม่ต้องพึ่ง matplotlib)"""
    w, h = 1280, 720
    img = np.full((h, w, 3), 24, dtype=np.uint8)

    cv2.putText(
        img,
        "TESA Top Gun 2026 (Day 2) - Segmentation Dataset & Synthetic Pile Summary",
        (35, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (0, 235, 255),
        2,
        cv2.LINE_AA,
    )

    # กล่องสรุปตัวเลขหลัก 4 กล่องด้านบน
    cards = [
        ("Real Extracted Beans", f"{sum(real_counts.values())} beans", "From 21 Original Images (7 Classes)"),
        ("YOLO-Seg Train Images", f"{train_img_count} images", "21 ROI Crops + Synthetic Touching Piles"),
        ("Validation Benchmark", f"{val_img_count} images", "Single-Class & Mixed Piles + uint16 GT"),
        ("Total Annotated Instances", f"{sum(real_counts.values()) + total_synth_beans:,}", "1px Topology-Verified Polygons"),
    ]
    for i, (title, big_val, sub) in enumerate(cards):
        cx1 = 35 + i * 305
        cy1 = 80
        cx2 = cx1 + 288
        cy2 = 210
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (42, 45, 52), -1)
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (90, 180, 240), 2)
        cv2.putText(img, title, (cx1 + 16, cy1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, big_val, (cx1 + 16, cy1 + 84), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (80, 245, 160), 2, cv2.LINE_AA)
        cv2.putText(img, sub, (cx1 + 16, cy1 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1, cv2.LINE_AA)

    # กราฟแท่งแสดงจำนวนเมล็ดต่อคลาสและสถานะการตรวจสเปก
    cv2.putText(
        img,
        "Extracted High-Res Bean Bank per Class & DAY2_INFER_SPEC Verification:",
        (35, 265),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    for idx, cls_name in enumerate(CLASSES):
        by = 300 + idx * 54
        cnt = real_counts.get(cls_name, 0)
        color = CLASS_COLORS_BGR[cls_name]
        cv2.putText(img, f"{idx}: {cls_name:<10}", (45, by + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2, cv2.LINE_AA)

        bar_x1 = 230
        bar_w = int(cnt * 16)
        cv2.rectangle(img, (bar_x1, by + 4), (bar_x1 + bar_w, by + 34), color, -1)
        cv2.putText(
            img,
            f"{cnt} real beans -> ~{cnt * 12} synthetic pile instances (100% single component, 0 holes)",
            (bar_x1 + bar_w + 18, by + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), img)


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    day1_dataset_dir = _PROJECT_ROOT / "day-1" / "datasets"

    datasets_seg_dir = task_dir / "datasets_seg"
    quality_dir = task_dir / "quality_inspection"
    bank_dir = datasets_seg_dir / "extracted_beans_bank"
    gt_root_dir = datasets_seg_dir / "ground_truth_day2_format"

    yolo_img_train = datasets_seg_dir / "images" / "train"
    yolo_img_val = datasets_seg_dir / "images" / "val"
    yolo_lbl_train = datasets_seg_dir / "labels" / "train"
    yolo_lbl_val = datasets_seg_dir / "labels" / "val"

    for d in (
        datasets_seg_dir,
        quality_dir,
        quality_dir / "real_annotated_vis",
        quality_dir / "synthetic_piles_vis",
        bank_dir,
        gt_root_dir,
        yolo_img_train,
        yolo_img_val,
        yolo_lbl_train,
        yolo_lbl_val,
    ):
        d.mkdir(parents=True, exist_ok=True)

    bean_bank: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CLASSES}
    real_counts: Dict[str, int] = {c: 0 for c in CLASSES}
    paper_patches: List[np.ndarray] = []

    real_full_entries: List[Tuple[str, np.ndarray, np.ndarray, List[Dict[str, Any]]]] = []
    real_crop_entries: List[Tuple[str, np.ndarray, np.ndarray, List[Dict[str, Any]]]] = []
    real_vis_thumbs: List[np.ndarray] = []

    print("=" * 75)
    print(f"{'CLASS':<12} | {'IMAGE':<22} | {'BEANS':<8} | {'ACTIVE ROI SIZE'}")
    print("=" * 75)

    for cls_name in CLASSES:
        cls_dir = day1_dataset_dir / cls_name
        (bank_dir / cls_name).mkdir(parents=True, exist_ok=True)
        img_paths = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

        for img_path in img_paths:
            img_bgr = load_image_exif_corrected(img_path)
            orig_h, orig_w = img_bgr.shape[:2]

            # เก็บตัวอย่างพื้นหลังกระดาษขาวด้านบนของภาพไว้ใช้สร้าง Synthetic Piles
            paper_patches.append(img_bgr[int(orig_h * 0.08) : int(orig_h * 0.25), int(orig_w * 0.15) : int(orig_w * 0.85)].copy())

            beans, seg_mask_u16 = detect_and_segment_beans_1px(img_bgr)
            real_counts[cls_name] += len(beans)

            objects_full: List[Dict[str, Any]] = []
            for b in beans:
                tx1, ty1, tx2, ty2 = b["bbox_tight"]
                objects_full.append(
                    {
                        "id": int(b["id"]),
                        "class": cls_name,
                        "polygon": b["polygon"],
                    }
                )

                # บันทึกเมล็ดเดี่ยวพร้อม Alpha Mask (RGBA PNG) เข้าคลัง extracted_beans_bank
                crop_bgr = img_bgr[ty1:ty2, tx1:tx2].copy()
                crop_mask = b["mask_crop"]
                rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = crop_mask
                bean_filename = f"{img_path.stem}_bean_{b['id']:02d}.png"
                cv2.imwrite(str(bank_dir / cls_name / bean_filename), rgba)

                # เก็บขอบแบบมี padding เล็กน้อยไว้แสดงผลในแผ่นตรวจคุณภาพ 1px
                px1, py1 = max(0, tx1 - 18), max(0, ty1 - 18)
                px2, py2 = min(orig_w, tx2 + 18), min(orig_h, ty2 + 18)
                mask_pad = (seg_mask_u16[py1:py2, px1:px2] == b["id"]).astype(np.uint8) * 255

                bean_bank[cls_name].append(
                    {
                        "bgr": crop_bgr,
                        "mask": crop_mask,
                        "bgr_padded": img_bgr[py1:py2, px1:px2].copy(),
                        "mask_padded": mask_pad,
                        "area_px": b["area_px"],
                        "source": f"{img_path.stem}#{b['id']}",
                    }
                )

            real_full_entries.append((img_path.name, img_bgr, seg_mask_u16, objects_full))

            # สร้างภาพ Active Pile ROI Cropped (ตามสไลด์หน้า 6: ภาพครอปเฉพาะบริเวณเมล็ด)
            rx1, ry1, rx2, ry2 = compute_active_pile_roi(beans, orig_w, orig_h, margin_ratio=0.16)
            crop_img_bgr = img_bgr[ry1:ry2, rx1:rx2].copy()
            crop_seg_u16 = seg_mask_u16[ry1:ry2, rx1:rx2].copy()
            ch, cw = crop_img_bgr.shape[:2]

            # ย่อเป็น 960x960 สำหรับเทรน YOLO-seg ให้รวดเร็วและคมชัด
            target_sz = 960
            scale_c = target_sz / float(cw)
            crop_resized_bgr = cv2.resize(crop_img_bgr, (target_sz, target_sz), interpolation=cv2.INTER_AREA)
            crop_resized_mask = cv2.resize(crop_seg_u16, (target_sz, target_sz), interpolation=cv2.INTER_NEAREST)

            objects_crop: List[Dict[str, Any]] = []
            clean_crop_mask = np.zeros((target_sz, target_sz), dtype=np.uint16)
            cid = 1
            for b in beans:
                sub = (crop_resized_mask == b["id"]).astype(np.uint8) * 255
                c_sub, c_cnt = ensure_single_clean_component(sub, smooth_sigma=2.4)
                if len(c_cnt) < 3:
                    continue
                clean_crop_mask[c_sub > 0] = cid
                objects_crop.append(
                    {
                        "id": cid,
                        "class": cls_name,
                        "polygon": [[int(pt[0][0]), int(pt[0][1])] for pt in c_cnt],
                    }
                )
                cid += 1

            real_crop_entries.append((img_path.name, crop_resized_bgr, clean_crop_mask, objects_crop))

            # บันทึกภาพ ROI Crop ทั้ง 21 ภาพเข้าชุดเทรน YOLO-seg
            train_name = f"real_roi_{img_path.stem}.jpg"
            cv2.imwrite(str(yolo_img_train / train_name), crop_resized_bgr)
            yolo_lines = objects_to_yolo_seg_lines(objects_crop, target_sz, target_sz)
            (yolo_lbl_train / f"real_roi_{img_path.stem}.txt").write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

            # สร้างภาพ Visualization ของภาพจริง
            vis_crop = render_vis_overlay(
                crop_resized_bgr,
                clean_crop_mask,
                objects_crop,
                title_banner=f"{img_path.name} | {len(objects_crop)} beans (1px Auto-Mask)",
            )
            cv2.imwrite(str(quality_dir / "real_annotated_vis" / f"vis_{img_path.name}"), vis_crop)
            real_vis_thumbs.append(cv2.resize(vis_crop, (420, 420), interpolation=cv2.INTER_AREA))

            print(f"{cls_name:<12} | {img_path.name:<22} | {len(beans):<8} | {cw}x{ch} -> 960x960")

    # 1. บันทึก Ground Truth ของภาพจริง (ทั้งแบบ Full และแบบ ROI Cropped)
    save_day2_ground_truth_bundle(gt_root_dir / "real_21_full", real_full_entries)
    save_day2_ground_truth_bundle(gt_root_dir / "real_21_roi_cropped", real_crop_entries)

    # 2. สร้างภาพรวม 21 ภาพจริง (7 แถว x 3 คอลัมน์)
    if len(real_vis_thumbs) == 21:
        rows_21 = [np.hstack(real_vis_thumbs[r * 3 : (r + 1) * 3]) for r in range(7)]
        cv2.imwrite(str(quality_dir / "01_summary_21_real_masks.jpg"), np.vstack(rows_21))

    # 3. สร้างแผ่นตรวจความคมระดับ 1px ของทั้ง 7 คลาส
    create_edge_zoom_sheet(bean_bank, quality_dir / "02_summary_1px_edge_zoom.jpg")

    # 4. สร้างภาพ Synthetic Touching Piles สำหรับ Train (56 ภาพ) และ Validation/Test Benchmark (14 ภาพ)
    print("-" * 75)
    print("Generating Synthetic Touching/Overlapping Pile Images (Paper & Wood Backgrounds)...")
    rng = random.Random(202602)
    total_synth_beans = 0
    synth_val_entries: List[Tuple[str, np.ndarray, np.ndarray, List[Dict[str, Any]]]] = []
    synth_preview_thumbs: List[np.ndarray] = []

    # 4.1 ชุด Train Synthetic Piles:
    #   - กองคลาสเดี่ยว (Single-Class Piles เหมือน Test Day 2): คลาสละ 5 ภาพ = 35 ภาพ (ทั้งพื้นกระดาษ, พื้นไม้, และมีป้ายชื่อคลาส)
    #   - กองคละคลาส (Multi-Class Piles เผื่อเม็ดปนและ Day 3): 21 ภาพ
    train_pile_idx = 1
    for cls_name in CLASSES:
        for v in range(5):
            bg_type = "wood" if (v % 2 == 1) else "paper"
            add_card = (v == 4)
            n_beans = rng.randint(16, 34)
            canvas, smask, objs = synthesize_pile_image(
                bean_bank=bean_bank,
                paper_patches=paper_patches,
                target_classes=[cls_name],
                num_beans=n_beans,
                canvas_size=(960, 960),
                bg_type=bg_type,
                add_label_card=add_card,
                rng=rng,
                touching_mode="dense_pile",
            )
            total_synth_beans += len(objs)
            fname = f"synth_train_single_{cls_name}_{v + 1:02d}.jpg"
            cv2.imwrite(str(yolo_img_train / fname), canvas)
            lines = objects_to_yolo_seg_lines(objs, 960, 960)
            (yolo_lbl_train / f"{Path(fname).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            train_pile_idx += 1

    for m_idx in range(21):
        bg_type = "wood" if (m_idx % 2 == 1) else "paper"
        add_card = (m_idx % 5 == 0)
        k_cls = rng.randint(3, 7)
        chosen_classes = rng.sample(CLASSES, k=k_cls)
        n_beans = rng.randint(18, 36)
        canvas, smask, objs = synthesize_pile_image(
            bean_bank=bean_bank,
            paper_patches=paper_patches,
            target_classes=chosen_classes,
            num_beans=n_beans,
            canvas_size=(960, 960),
            bg_type=bg_type,
            add_label_card=add_card,
            rng=rng,
            touching_mode="dense_pile",
        )
        total_synth_beans += len(objs)
        fname = f"synth_train_mixed_{m_idx + 1:02d}.jpg"
        cv2.imwrite(str(yolo_img_train / fname), canvas)
        lines = objects_to_yolo_seg_lines(objs, 960, 960)
        (yolo_lbl_train / f"{Path(fname).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 4.2 ชุด Validation & Simulated Day 2 Test Benchmark (14 ภาพ):
    #   - 7 ภาพกองคลาสเดี่ยวบนพื้นกระดาษขาว (จำลอง Test Day 2 บนกระดาษ)
    #   - 7 ภาพกองคลาสเดี่ยวบนพื้นไม้พร้อมป้าย (จำลองสไลด์หน้า 3/12/22/46)
    for cls_name in CLASSES:
        for mode_idx, (bg_type, add_card, suffix) in enumerate(
            [("paper", False, "paper_pile"), ("wood", True, "wood_pile")]
        ):
            n_beans = rng.randint(22, 35)
            canvas, smask, objs = synthesize_pile_image(
                bean_bank=bean_bank,
                paper_patches=paper_patches,
                target_classes=[cls_name],
                num_beans=n_beans,
                canvas_size=(960, 960),
                bg_type=bg_type,
                add_label_card=add_card,
                rng=rng,
                touching_mode="dense_pile",
            )
            total_synth_beans += len(objs)
            fname = f"val_{cls_name}_{suffix}.jpg"
            cv2.imwrite(str(yolo_img_val / fname), canvas)
            lines = objects_to_yolo_seg_lines(objs, 960, 960)
            (yolo_lbl_val / f"{Path(fname).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

            synth_val_entries.append((fname, canvas, smask, objs))
            vis_pile = render_vis_overlay(
                canvas,
                smask,
                objs,
                title_banner=f"{fname} | {len(objs)} touching beans ({cls_name})",
            )
            cv2.imwrite(str(quality_dir / "synthetic_piles_vis" / f"vis_{fname}"), vis_pile)
            synth_preview_thumbs.append(cv2.resize(vis_pile, (440, 440), interpolation=cv2.INTER_AREA))

    save_day2_ground_truth_bundle(gt_root_dir / "synthetic_test_piles", synth_val_entries)

    # สร้างแผ่นรวมภาพกองจำลอง (7 แถว x 2 คอลัมน์: ซ้าย=กองบนกระดาษ, ขวา=กองบนโต๊ะไม้+ป้ายคลาส)
    if len(synth_preview_thumbs) == 14:
        rows_synth = [
            np.hstack([synth_preview_thumbs[r * 2], synth_preview_thumbs[r * 2 + 1]])
            for r in range(7)
        ]
        cv2.imwrite(str(quality_dir / "03_summary_synthetic_piles.jpg"), np.vstack(rows_synth))

    # 5. เขียนไฟล์ beans_seg.yaml สำหรับเทรน YOLO Segmentation
    yaml_content = (
        f"path: {datasets_seg_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASSES))
        + "\n"
    )
    (datasets_seg_dir / "beans_seg.yaml").write_text(yaml_content, encoding="utf-8")

    # 6. สร้างกราฟสรุปสถิติชุดข้อมูล
    train_count = len(list(yolo_img_train.glob("*.jpg")))
    val_count = len(list(yolo_img_val.glob("*.jpg")))
    create_stats_dashboard(
        real_counts=real_counts,
        train_img_count=train_count,
        val_img_count=val_count,
        total_synth_beans=total_synth_beans,
        out_path=quality_dir / "04_dataset_statistics_report.png",
    )

    print("=" * 75)
    print(f"[SUCCESS] Real Beans Extracted (1px Mask): {sum(real_counts.values())} beans from 21 images")
    print(f"[SUCCESS] YOLO-Seg Train Set Created     : {train_count} images ({yolo_img_train})")
    print(f"[SUCCESS] YOLO-Seg Val/Benchmark Created : {val_count} images ({yolo_img_val})")
    print(f"[SUCCESS] Day 2 Ground Truth Bundles     : {gt_root_dir}")
    print(f"[SUCCESS] Visual Quality Inspection Dir  : {quality_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
