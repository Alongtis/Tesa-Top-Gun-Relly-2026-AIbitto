#!/usr/bin/env python3
"""
prepare_day3_dataset.py — สร้างชุดข้อมูล Day 3 Multi-Class Instance Segmentation
สำหรับ Fine-Tune ต่อจาก weights/best.pt ของ Day 2 บน Google Colab

จุดเด่นทางวิศวกรรม (แก้ปัญหาจาก Zero-Shot Baseline 58.06% -> 95%+):
  1. แบ่ง Train/Val ตามภาพแบบ Stratified (ตามสไลด์ Day 3 หน้า 4: ห้ามแบ่งตามเม็ด):
     - Val Holdout (3 ภาพจริง, 23 เม็ด ครบทุก 7 คลาส): Mixed_grid1, Mixed_square2, Mixed_triangle1
       + กองคละคลาสจำลอง 12 ภาพที่สร้างจาก "23 เม็ดของ Val เท่านั้น" (Zero Leakage 100%)
     - Train (13 ภาพจริง, 101 เม็ด) + Day 1/2 Replay (21 ภาพ, 175 เม็ดที่บีบขอบให้คมเท่า Human Label)
  2. แก้ปัญหา Border False Positives (ขอบโต๊ะไม้บน/ล่างแบบ Mixed_triangle3):
     - ใส่ทั้งภาพ Active-Pile ROI (960x960) และภาพ Full-Frame (720x960) ที่มีขอบโต๊ะจริงในชุดเทรน
     - สกัดพื้นผิวโต๊ะไม้จริงจากแถบขอบบนของภาพ Day 3 มาทำพื้นหลังกองเมล็ดด้วย
  3. เสริม Hard Confusing-Pair Piles (good/immature/parchment, broken/shell/good, husk/black)
     และใช้ Inner-Only Alpha Blending (ไร้ขอบขาว Halo) เพื่อดัน Class Accuracy และ Mask IoU
  4. แพ็ก datasets_day3_colab.zip (รวม pretrained/day2_best.pt ไว้ข้างใน) พร้อมอัปโหลดขึ้น Colab ไฟล์เดียวจบ
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
DAY3_DIR = TASK_DIR.parent
TRACK2_DIR = DAY3_DIR.parent

DAY3_DS_DIR = TASK_DIR / "Dataset_Day-3"
DAY2_REAL_GT_DIR = TRACK2_DIR / "day-2" / "team_01_day2" / "task" / "datasets_seg" / "ground_truth_day2_format" / "real_21_full"
DAY2_BEST_PT = TRACK2_DIR / "day-2" / "team_01_day2" / "weights" / "best.pt"
DAY3_V1_BEST_PT = DAY3_DIR / "team_01_day3" / "weights" / "best.pt"

OUT_DS_DIR = SCRIPT_DIR / "datasets_seg_day3"
QUALITY_DIR = SCRIPT_DIR / "quality_inspection"
COLAB_ZIP_HOLDOUT = SCRIPT_DIR / "datasets_day3_colab.zip"
COLAB_ZIP_FULL = SCRIPT_DIR / "datasets_day3_final_colab.zip"
COLAB_ZIP_V2 = SCRIPT_DIR / "datasets_day3_v2_hardened_colab.zip"

CLASSES: List[str] = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "good": (235, 95, 45),
    "shell": (200, 210, 50),
    "black": (190, 180, 175),
    "husk": (50, 70, 230),
    "broken": (40, 195, 60),
    "parchment": (40, 210, 245),
    "immature": (220, 80, 210),
}

# Stratified 3-Image Holdout Val Set (1 Grid + 1 Square + 1 Triangle = 23 beans, >=2 per class for all 7 classes)
VAL_IMAGE_NAMES = {"Mixed_grid1.jpg", "Mixed_square2.jpg", "Mixed_triangle1.jpg"}


def load_exif_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return cv2.cvtColor(np.array(im.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR)


def objects_to_yolo_seg_lines(objects: List[Dict[str, Any]], img_w: int, img_h: int) -> List[str]:
    lines: List[str] = []
    for obj in objects:
        cls_idx = CLASS_TO_IDX[obj["class"]]
        poly = obj["polygon"]
        step = max(1, len(poly) // 90)
        sampled = poly[::step]
        if len(sampled) < 3:
            sampled = poly
        coords = " ".join(
            f"{min(1.0, max(0.0, pt[0] / float(img_w))):.6f} {min(1.0, max(0.0, pt[1] / float(img_h))):.6f}"
            for pt in sampled
        )
        lines.append(f"{cls_idx} {coords}")
    return lines


def compute_active_roi_from_objects(
    objects: List[Dict[str, Any]],
    orig_w: int,
    orig_h: int,
    margin_ratio: float = 0.18,
) -> Tuple[int, int, int, int]:
    pts = [pt for o in objects for pt in o["polygon"]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    span = max(x2 - x1, y2 - y1, 500)
    side = min(orig_w, orig_h, int(round(span * (1.0 + 2.0 * margin_ratio))))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx1 = max(0, min(orig_w - side, cx - side // 2))
    ry1 = max(0, min(orig_h - side, cy - side // 2))
    return (rx1, ry1, rx1 + side, ry1 + side)


def extract_beans_from_image(
    img_bgr: np.ndarray,
    objects: List[Dict[str, Any]],
    erode_px: int = 0,
    source_tag: str = "d3",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """สร้าง uint16 seg_mask และสกัดเมล็ดเดี่ยวเข้าคลัง (รองรับการบีบขอบ erode_px สำหรับ Day 1/2 Replay)"""
    h, w = img_bgr.shape[:2]
    seg_mask = np.zeros((h, w), dtype=np.uint16)
    extracted: List[Dict[str, Any]] = []

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)) if erode_px > 0 else None

    for idx, obj in enumerate(objects, start=1):
        pts = np.array(obj["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        bx, by, bw, bh = cv2.boundingRect(pts)
        pad = 6
        x1, y1 = max(0, bx - pad), max(0, by - pad)
        x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
        local_pts = pts.copy()
        local_pts[:, 0, 0] -= x1
        local_pts[:, 0, 1] -= y1

        local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.drawContours(local_mask, [local_pts], -1, 255, thickness=-1)
        if kern is not None:
            eroded = cv2.erode(local_mask, kern, iterations=1)
            if np.count_nonzero(eroded) > 0.65 * np.count_nonzero(local_mask):
                local_mask = eroded

        cnts, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        best_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) < 200:
            continue
        clean_local = np.zeros_like(local_mask)
        cv2.drawContours(clean_local, [best_cnt], -1, 255, thickness=-1)
        seg_mask[y1:y2, x1:x2][clean_local > 0] = idx

        # Tight crop around clean_local
        ys, xs = np.where(clean_local > 0)
        cx1, cx2 = max(0, int(xs.min()) - 2), min(clean_local.shape[1], int(xs.max()) + 3)
        cy1, cy2 = max(0, int(ys.min()) - 2), min(clean_local.shape[0], int(ys.max()) + 3)
        extracted.append(
            {
                "class": obj["class"],
                "bgr": img_bgr[y1 + cy1 : y1 + cy2, x1 + cx1 : x1 + cx2].copy(),
                "mask": clean_local[cy1:cy2, cx1:cx2].copy(),
                "source": source_tag,
            }
        )

    return seg_mask, extracted


def crop_and_resize_entry(
    img_bgr: np.ndarray,
    seg_mask_u16: np.ndarray,
    id_to_cls: Dict[int, str],
    crop_box: Tuple[int, int, int, int],
    target_wh: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    rx1, ry1, rx2, ry2 = crop_box
    tw, th = target_wh
    crop_img = cv2.resize(img_bgr[ry1:ry2, rx1:rx2], (tw, th), interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(seg_mask_u16[ry1:ry2, rx1:rx2], (tw, th), interpolation=cv2.INTER_NEAREST)

    clean_mask = np.zeros((th, tw), dtype=np.uint16)
    objs: List[Dict[str, Any]] = []
    fid = 1
    for oid, cls_name in id_to_cls.items():
        sub = (crop_mask == oid).astype(np.uint8) * 255
        if int(np.count_nonzero(sub)) < 40:
            continue
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < 40 or len(best) < 3:
            continue
        single = np.zeros_like(sub)
        cv2.drawContours(single, [best], -1, 255, thickness=-1)
        clean_mask[single > 0] = fid
        objs.append(
            {
                "id": fid,
                "class": cls_name,
                "polygon": [[int(pt[0][0]), int(pt[0][1])] for pt in best],
            }
        )
        fid += 1

    return crop_img, clean_mask, objs


def make_real_wood_bg(wood_strip: np.ndarray, width: int, height: int, rng: random.Random) -> np.ndarray:
    """สร้างพื้นหลังโต๊ะไม้จริงจากแถบโต๊ะไม้ในภาพ Day 3 (Mixed_triangle3)"""
    if rng.random() < 0.5:
        strip = cv2.rotate(wood_strip, cv2.ROTATE_90_CLOCKWISE)
    else:
        strip = wood_strip.copy()
    if rng.random() < 0.5:
        strip = cv2.flip(strip, 1)
    sh, sw = strip.shape[:2]
    # Tile with mirrored reflections to avoid seams
    tile_x = np.hstack([strip, cv2.flip(strip, 1)])
    tile_xy = np.vstack([tile_x, cv2.flip(tile_x, 0)])
    th, tw = tile_xy.shape[:2]
    rep_y = (height // th) + 2
    rep_x = (width // tw) + 2
    tiled = np.tile(tile_xy, (rep_y, rep_x, 1))
    oy = rng.randint(0, max(0, tiled.shape[0] - height))
    ox = rng.randint(0, max(0, tiled.shape[1] - width))
    bg = tiled[oy : oy + height, ox : ox + width].astype(np.float32)
    gain = rng.uniform(0.95, 1.38)  # ครอบคลุมทั้งโต๊ะไม้สีเข้มและสีอ่อน
    bg = np.clip(bg * gain, 20, 230).astype(np.uint8)
    return bg


def make_synth_wood_bg(width: int, height: int, rng: random.Random) -> np.ndarray:
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
    noise = np.random.default_rng(rng.randint(0, 999999)).normal(0, 0.055, (height, width)).astype(np.float32)
    t = np.clip((stripes + 1.0) * 0.5 + noise, 0.0, 1.0)
    dark_wood = np.array([rng.uniform(45, 70), rng.uniform(80, 110), rng.uniform(115, 148)], dtype=np.float32)
    light_wood = np.array([rng.uniform(95, 120), rng.uniform(140, 168), rng.uniform(175, 208)], dtype=np.float32)
    wood = dark_wood[None, None, :] * (1.0 - t[:, :, None]) + light_wood[None, None, :] * t[:, :, None]
    bg = cv2.GaussianBlur(np.clip(wood, 0, 255).astype(np.uint8), (9, 3), 0)
    if rng.random() < 0.35:
        bg = cv2.rotate(bg, cv2.ROTATE_90_CLOCKWISE)
    return bg


def make_paper_bg(paper_patches: List[np.ndarray], width: int, height: int, rng: random.Random, add_table_border: bool = False, wood_strip: np.ndarray | None = None) -> np.ndarray:
    base = rng.choice(paper_patches)
    bg = cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    gx = np.linspace(rng.uniform(-12, 12), rng.uniform(-12, 12), width, dtype=np.float32)[None, :, None]
    gy = np.linspace(rng.uniform(-12, 12), rng.uniform(-12, 12), height, dtype=np.float32)[:, None, None]
    tint = np.array([rng.uniform(-6, 6), rng.uniform(-5, 5), rng.uniform(-6, 6)], dtype=np.float32)[None, None, :]
    bg = np.clip(bg + gx + gy + tint, 142, 238).astype(np.uint8)

    if add_table_border and wood_strip is not None:
        # ใส่ขอบโต๊ะไม้จริงด้านบนหรือด้านล่าง (Hard Negative ป้องกัน False Positive แบบ Mixed_triangle3)
        border_h = rng.randint(55, 115)
        w_bg = make_real_wood_bg(wood_strip, width, border_h, rng)
        if rng.random() < 0.5:
            bg[:border_h, :] = w_bg
        else:
            bg[-border_h:, :] = w_bg
    return bg


def transform_bean_clean(
    bgr: np.ndarray,
    mask_u8: np.ndarray,
    angle_deg: float,
    scale: float,
    flip_h: bool,
    gain: float,
    trim_px: float = 1.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """หมุน/ปรับขนาดเมล็ดพร้อม Inner-Only Alpha Feathering (ไร้ขอบขาว Halo บนพื้นไม้และกระดาษ)"""
    if flip_h:
        bgr = cv2.flip(bgr, 1)
        mask_u8 = cv2.flip(mask_u8, 1)
    h, w = mask_u8.shape[:2]
    diag = int(math.ceil(math.hypot(w, h) * scale * 1.25)) + 8
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    M[0, 2] += diag / 2.0 - cx
    M[1, 2] += diag / 2.0 - cy

    rot_bgr = cv2.warpAffine(bgr, M, (diag, diag), flags=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    rot_mask = cv2.warpAffine(mask_u8, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    _, rot_mask = cv2.threshold(rot_mask, 150, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(rot_mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty rotated bean")
    x1, x2 = max(0, int(xs.min()) - 2), min(diag, int(xs.max()) + 3)
    y1, y2 = max(0, int(ys.min()) - 2), min(diag, int(ys.max()) + 3)

    c_bgr = np.clip(rot_bgr[y1:y2, x1:x2].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    c_mask = rot_mask[y1:y2, x1:x2]

    dist = cv2.distanceTransform(c_mask, cv2.DIST_L2, 5)
    alpha = np.clip((dist - trim_px) / 1.8, 0.0, 1.0).astype(np.float32)
    hard_mask = ((dist >= trim_px) & (c_mask > 0)).astype(np.uint8) * 255
    return c_bgr, hard_mask, alpha


def apply_v2_domain_effect(img_bgr: np.ndarray, effect: str, rng: random.Random) -> Tuple[np.ndarray, int]:
    """ใส่สภาพแสง/White-Balance/ISO Noise/Blur แบบสมจริงสำหรับชุดเทรน V2 Hardened
    (คงความต่างสีเชิงสัมพัทธ์ระหว่างคลาส เพื่อให้โมเดลเรียนรู้ Color Constancy จริง)
    """
    h, w = img_bgr.shape[:2]
    f = img_bgr.astype(np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    jq = 84

    if effect == "cool_wb":
        f[:, :, 0] *= rng.uniform(1.14, 1.28)
        f[:, :, 1] *= rng.uniform(0.99, 1.05)
        f[:, :, 2] *= rng.uniform(0.74, 0.86)
    elif effect == "warm_wb":
        f[:, :, 0] *= rng.uniform(0.64, 0.78)
        f[:, :, 1] *= rng.uniform(0.94, 1.00)
        f[:, :, 2] *= rng.uniform(1.14, 1.28)
    elif effect == "dim_dark":
        f = f * rng.uniform(0.48, 0.68) + np_rng.normal(0, rng.uniform(3.0, 7.0), f.shape).astype(np.float32)
    elif effect == "split_shadow":
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        a, b = rng.uniform(0.4, 0.9), rng.uniform(0.4, 0.9)
        diag = (xx * a + yy * b) - rng.uniform(0.45, 0.80)
        m = 1.0 / (1.0 + np.exp(-diag * rng.uniform(18.0, 28.0)))
        gain = rng.uniform(0.44, 0.58) + rng.uniform(0.55, 0.72) * m
        f = f * gain[:, :, None]
    elif effect == "iso_noise":
        sigma = rng.uniform(9.0, 20.0)
        f = f + np_rng.normal(0, sigma, f.shape).astype(np.float32)
        jq = rng.randint(65, 82)
    elif effect == "cool_noise":
        f[:, :, 0] *= rng.uniform(1.12, 1.22)
        f[:, :, 2] *= rng.uniform(0.78, 0.88)
        f = f + np_rng.normal(0, rng.uniform(8.0, 15.0), f.shape).astype(np.float32)
    elif effect == "low_jpeg_blur":
        out = np.clip(f, 0, 255).astype(np.uint8)
        if rng.random() < 0.65:
            out = cv2.GaussianBlur(out, (5, 5), rng.uniform(0.9, 1.8))
        return out, rng.randint(25, 42)

    return np.clip(f, 0, 255).astype(np.uint8), jq


def make_empty_hard_negative(
    kind: str,
    paper_patches: List[np.ndarray],
    wood_strip: np.ndarray,
    rng: random.Random,
    w: int = 960,
    h: int = 960,
) -> np.ndarray:
    """สร้างภาพเปล่า 0 เม็ด (Hard Negative) เพื่อแก้ Ghost FP ตรงมุมขอบโต๊ะไม้/รอยพับ/ISO Noise"""
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    if kind == "paper_double_border_crease":
        bg = make_paper_bg(paper_patches, w, h, rng, add_table_border=False)
        bh1, bh2 = rng.randint(60, 115), rng.randint(60, 115)
        bg[:bh1, :] = make_real_wood_bg(wood_strip, w, bh1, rng)
        bg[-bh2:, :] = make_real_wood_bg(wood_strip, w, bh2, rng)
        cx = int(w * rng.uniform(0.30, 0.70))
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        dist_line = np.abs((xx - cx) + (yy - h * 0.5) * rng.uniform(-0.22, 0.22))
        crease = 1.0 - rng.uniform(0.08, 0.16) * np.exp(-(dist_line ** 2) / 150.0)
        bg = np.clip(bg.astype(np.float32) * crease[:, :, None], 0, 255).astype(np.uint8)
        return bg
    if kind == "paper_border_noisy":
        bg = make_paper_bg(paper_patches, w, h, rng, add_table_border=True, wood_strip=wood_strip)
        noisy = bg.astype(np.float32) + np_rng.normal(0, rng.uniform(12.0, 21.0), bg.shape).astype(np.float32)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "dark_slate_empty":
        base = np.full((h, w, 3), (38, 40, 45), dtype=np.float32)
        hi = np_rng.normal(0, rng.uniform(5.0, 13.0), (h, w, 3)).astype(np.float32)
        return np.clip(base + hi, 14, 90).astype(np.uint8)
    # dark wood / synth wood empty
    bg = make_real_wood_bg(wood_strip, w, h, rng) if rng.random() < 0.7 else make_synth_wood_bg(w, h, rng)
    return np.clip(bg.astype(np.float32) * rng.uniform(0.65, 1.10), 12, 230).astype(np.uint8)


def synthesize_mixed_pile(
    primary_bank: Dict[str, List[Dict[str, Any]]],
    replay_bank: Dict[str, List[Dict[str, Any]]] | None,
    primary_prob: float,
    paper_patches: List[np.ndarray],
    wood_strip: np.ndarray,
    target_classes: List[str],
    num_beans: int,
    bg_mode: str,
    rng: random.Random,
    canvas_size: Tuple[int, int] = (960, 960),
    max_overlap_ratio: float = 0.10,
    scale_range: Tuple[float, float] = (0.76, 1.06),
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    cw, ch = canvas_size
    if bg_mode == "real_wood":
        canvas = make_real_wood_bg(wood_strip, cw, ch, rng)
        trim_px = 1.8
    elif bg_mode == "synth_wood":
        canvas = make_synth_wood_bg(cw, ch, rng)
        trim_px = 1.8
    elif bg_mode == "dark_slate":
        np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        base = np.full((ch, cw, 3), (40, 42, 46), dtype=np.float32)
        canvas = np.clip(base + np_rng.normal(0, 6.0, (ch, cw, 3)).astype(np.float32), 18, 85).astype(np.uint8)
        trim_px = 2.0
    elif bg_mode == "paper_both_borders":
        canvas = make_paper_bg(paper_patches, cw, ch, rng, add_table_border=False)
        bh1, bh2 = rng.randint(55, 105), rng.randint(55, 105)
        canvas[:bh1, :] = make_real_wood_bg(wood_strip, cw, bh1, rng)
        canvas[-bh2:, :] = make_real_wood_bg(wood_strip, cw, bh2, rng)
        trim_px = 1.2
    elif bg_mode == "paper_border":
        canvas = make_paper_bg(paper_patches, cw, ch, rng, add_table_border=True, wood_strip=wood_strip)
        trim_px = 1.2
    else:
        canvas = make_paper_bg(paper_patches, cw, ch, rng, add_table_border=False)
        trim_px = 1.2

    safe_top = 120 if "border" in bg_mode else 35
    safe_bot = ch - 120 if "border" in bg_mode else ch - 35

    raw_id_mask = np.zeros((ch, cw), dtype=np.uint16)
    placed_area: Dict[int, int] = {}
    id_to_cls: Dict[int, str] = {}
    occupied_centers: List[Tuple[float, float, float]] = []

    # จัดสมดุลคลาสภายในภาพ: แจกทุกคลาสใน target_classes ให้ทั่วก่อนแล้วค่อยสุ่มเพิ่ม
    cls_seq = list(target_classes) * (num_beans // len(target_classes)) + rng.sample(target_classes, k=num_beans % len(target_classes))
    rng.shuffle(cls_seq)

    cur_id = 1
    for cls_name in cls_seq:
        use_primary = (replay_bank is None) or (not replay_bank.get(cls_name)) or (rng.random() < primary_prob)
        pool = primary_bank[cls_name] if (use_primary and primary_bank.get(cls_name)) else replay_bank[cls_name]  # type: ignore[index]
        sample = rng.choice(pool)

        # ขนาดเม็ดคละสเกลสมจริง (ปรับช่วงผ่าน scale_range ได้สำหรับภาพไกล/จ่อใกล้)
        scale = (min(cw, ch) / 1120.0) * rng.uniform(scale_range[0], scale_range[1])
        angle = rng.uniform(0.0, 360.0)
        flip_h = rng.random() < 0.5
        gain = rng.uniform(0.92, 1.08)
        b_bgr, b_mask, b_alpha = transform_bean_clean(
            sample["bgr"], sample["mask"], angle, scale, flip_h, gain, trim_px=trim_px
        )
        bh, bw = b_mask.shape[:2]
        bean_bool = b_mask > 0
        bean_area = int(np.count_nonzero(bean_bool))
        if bean_area < 180:
            continue
        rad = 0.5 * math.hypot(bw, bh) * 0.42

        placed = False
        for attempt in range(220):
            if occupied_centers and rng.random() < 0.84:
                ax, ay, ar = rng.choice(occupied_centers)
                th = rng.uniform(0.0, 2.0 * math.pi)
                dist = (ar + rad) * rng.uniform(0.80, 1.08)
                cx = ax + math.cos(th) * dist
                cy = ay + math.sin(th) * dist
            else:
                cx = rng.gauss(cw * 0.50, cw * 0.19)
                cy = rng.gauss((safe_top + safe_bot) * 0.50, (safe_bot - safe_top) * 0.20)

            px = int(round(cx - bw / 2.0))
            py = int(round(cy - bh / 2.0))
            if px < 30 or py < safe_top or (px + bw) >= (cw - 30) or (py + bh) >= safe_bot:
                continue

            roi_id = raw_id_mask[py : py + bh, px : px + bw]
            overlap_m = (roi_id > 0) & bean_bool
            overlap_px = int(np.count_nonzero(overlap_m))
            if overlap_px > int(bean_area * max_overlap_ratio):
                continue

            bad_occ = False
            for aid in np.unique(roi_id[overlap_m]):
                cut = int(np.count_nonzero((roi_id == aid) & bean_bool))
                rem = int(np.count_nonzero(raw_id_mask == aid)) - cut
                if rem < int(placed_area[int(aid)] * 0.85):
                    bad_occ = True
                    break
            if bad_occ:
                continue

            # Soft directional drop shadow
            sy1, sy2 = min(ch, py + 4), min(ch, py + bh + 4)
            sx1, sx2 = min(cw, px + 3), min(cw, px + bw + 3)
            sh_h, sh_w = sy2 - sy1, sx2 - sx1
            if sh_h > 0 and sh_w > 0:
                sh_m = cv2.GaussianBlur(b_alpha[:sh_h, :sh_w], (11, 11), 0)[:, :, None]
                sh_str = 0.28 if "wood" in bg_mode else 0.22
                canvas[sy1:sy2, sx1:sx2] = np.clip(
                    canvas[sy1:sy2, sx1:sx2].astype(np.float32) * (1.0 - sh_str * sh_m), 0, 255
                ).astype(np.uint8)

            a3 = b_alpha[:, :, None]
            roi_c = canvas[py : py + bh, px : px + bw].astype(np.float32)
            canvas[py : py + bh, px : px + bw] = np.clip(
                b_bgr.astype(np.float32) * a3 + roi_c * (1.0 - a3), 0, 255
            ).astype(np.uint8)

            raw_id_mask[py : py + bh, px : px + bw][bean_bool] = cur_id
            placed_area[cur_id] = bean_area
            id_to_cls[cur_id] = cls_name
            occupied_centers.append((px + bw / 2.0, py + bh / 2.0, rad))
            cur_id += 1
            placed = True
            break

        if not placed:
            continue

    final_mask = np.zeros((ch, cw), dtype=np.uint16)
    final_objs: List[Dict[str, Any]] = []
    fid = 1
    for rid, cls_name in id_to_cls.items():
        ys, xs = np.where(raw_id_mask == rid)
        if len(xs) < 120:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        sub = (raw_id_mask[y1:y2, x1:x2] == rid).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < 120 or len(best) < 3:
            continue
        single = np.zeros_like(sub)
        cv2.drawContours(single, [best], -1, 255, thickness=-1)
        final_mask[y1:y2, x1:x2][single > 0] = fid
        g_cnt = best.copy()
        g_cnt[:, 0, 0] += x1
        g_cnt[:, 0, 1] += y1
        final_objs.append(
            {
                "id": fid,
                "class": cls_name,
                "polygon": [[int(pt[0][0]), int(pt[0][1])] for pt in g_cnt],
            }
        )
        fid += 1

    return canvas, final_mask, final_objs


def render_preview_tile(img_bgr: np.ndarray, seg_mask: np.ndarray, objects: List[Dict[str, Any]], title: str, size: int = 400) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()
    ov = vis.copy()
    for obj in objects:
        col = CLASS_COLORS_BGR.get(obj["class"], (0, 230, 0))
        pts = np.array(obj["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        ov[seg_mask == obj["id"]] = col
        cv2.polylines(vis, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
    vis = cv2.addWeighted(ov, 0.40, vis, 0.60, 0.0)
    # Pad to square for uniform grid
    side = max(h, w)
    sq = np.full((side, side, 3), 24, dtype=np.uint8)
    sq[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = vis
    tile = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (size, 34), (20, 20, 24), -1)
    cv2.putText(tile, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 1, cv2.LINE_AA)
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Day 3 Multi-Class Segmentation Dataset")
    parser.add_argument(
        "--mode",
        choices=("full", "holdout", "v2_hardened"),
        default="v2_hardened",
        help="v2_hardened: Full 16 real images + WB/Noise/Hard-Neg/Immature hardening (warm-start from V1); full: V1 full; holdout: 13/3 val split",
    )
    args = parser.parse_args()
    v2_mode = args.mode == "v2_hardened"
    full_mode = args.mode in ("full", "v2_hardened")
    colab_zip_path = COLAB_ZIP_V2 if v2_mode else (COLAB_ZIP_FULL if full_mode else COLAB_ZIP_HOLDOUT)

    rng = random.Random(2026111909 if v2_mode else (2026111902 if full_mode else 20261119))
    val_rng = random.Random(2026111988 if v2_mode else 2026111999)

    if OUT_DS_DIR.exists():
        shutil.rmtree(OUT_DS_DIR)
    for sub in (
        OUT_DS_DIR / "images" / "train",
        OUT_DS_DIR / "images" / "val",
        OUT_DS_DIR / "labels" / "train",
        OUT_DS_DIR / "labels" / "val",
        OUT_DS_DIR / "val_gt_day3_format" / "seg_mask",
        OUT_DS_DIR / "pretrained",
        QUALITY_DIR,
    ):
        sub.mkdir(parents=True, exist_ok=True)

    # คัดลอก Pretrained Weights เข้าไปในแพ็กเพื่อให้ Colab Fine-Tune ต่อได้ทันที (อัปโหลดแค่ไฟล์ zip เดียว)
    if v2_mode and DAY3_V1_BEST_PT.exists():
        shutil.copy2(DAY3_V1_BEST_PT, OUT_DS_DIR / "pretrained" / "day3_v1_best.pt")
        print(f"[PRETRAINED] Bundled Day 3 V1 Final best.pt -> {OUT_DS_DIR / 'pretrained' / 'day3_v1_best.pt'}")
    if DAY2_BEST_PT.exists():
        shutil.copy2(DAY2_BEST_PT, OUT_DS_DIR / "pretrained" / "day2_best.pt")
        print(f"[PRETRAINED] Bundled Day 2 best.pt -> {OUT_DS_DIR / 'pretrained' / 'day2_best.pt'}")
    elif not (v2_mode and DAY3_V1_BEST_PT.exists()):
        print(f"[WARN] Pretrained weights not found at {DAY2_BEST_PT}")

    # 1. โหลดชุดข้อมูลจริง Day 3 (16 ภาพ, 124 เม็ด)
    day3_labels = json.loads((DAY3_DS_DIR / "labels.json").read_text(encoding="utf-8"))
    d3_train_bank: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    d3_val_bank: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    d3_train_paper: List[np.ndarray] = []
    d3_val_paper: List[np.ndarray] = []
    real_wood_strip: np.ndarray | None = None

    val_gt_entries: List[Dict[str, Any]] = []
    train_count = 0
    val_count = 0
    train_cls_hist = Counter()
    val_cls_hist = Counter()
    preview_real_tiles: List[np.ndarray] = []

    print("=" * 78)
    mode_lbl = "FINAL FULL-DATA (ALL 16 REAL IMAGES -> TRAIN)" if full_mode else "Strict Image-Level Holdout (13 Train / 3 Val)"
    print(f"[1/4] Processing 16 Real Day-3 Multi-Class Images [{mode_lbl}]")
    print("=" * 78)
    for entry in sorted(day3_labels, key=lambda x: x["image_name"]):
        img_name = entry["image_name"]
        stem = Path(img_name).stem
        in_val_ref = img_name in VAL_IMAGE_NAMES
        is_val_only = (not full_mode) and in_val_ref
        split = "val" if is_val_only else "train"

        img_bgr = load_exif_bgr(DAY3_DS_DIR / "Images" / img_name)
        oh, ow = img_bgr.shape[:2]

        if img_name == "Mixed_triangle3.jpg":
            real_wood_strip = img_bgr[0:145, 80 : ow - 80].copy()

        paper_patch = img_bgr[int(oh * 0.72) : int(oh * 0.88), int(ow * 0.15) : int(ow * 0.85)].copy()
        if is_val_only:
            d3_val_paper.append(paper_patch)
        else:
            d3_train_paper.append(paper_patch)
            if full_mode and in_val_ref:
                d3_val_paper.append(paper_patch)

        seg_full, extracted = extract_beans_from_image(img_bgr, entry["objects"], erode_px=0, source_tag=f"d3_{stem}")
        id_to_cls = {i + 1: o["class"] for i, o in enumerate(entry["objects"])}

        for b in extracted:
            if is_val_only:
                d3_val_bank[b["class"]].append(b)
            else:
                d3_train_bank[b["class"]].append(b)
                if full_mode:
                    d3_val_bank[b["class"]].append(b)

        # (A) Active-Pile ROI Crop 960x960
        roi_box = compute_active_roi_from_objects(entry["objects"], ow, oh, margin_ratio=0.18)
        roi_img, roi_mask, roi_objs = crop_and_resize_entry(img_bgr, seg_full, id_to_cls, roi_box, (960, 960))
        roi_fname = f"d3_{split}_roi_{stem}.jpg"
        cv2.imwrite(str(OUT_DS_DIR / "images" / split / roi_fname), roi_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        (OUT_DS_DIR / "labels" / split / f"{Path(roi_fname).stem}.txt").write_text(
            "\n".join(objects_to_yolo_seg_lines(roi_objs, 960, 960)) + "\n", encoding="utf-8"
        )

        # (B) Full-Frame 720x960 (รวมขอบโต๊ะไม้จริงด้านบน/ล่างเป็น Negative Background!)
        full_img, full_mask, full_objs = crop_and_resize_entry(img_bgr, seg_full, id_to_cls, (0, 0, ow, oh), (720, 960))
        full_fname = f"d3_{split}_full_{stem}.jpg"
        cv2.imwrite(str(OUT_DS_DIR / "images" / split / full_fname), full_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        (OUT_DS_DIR / "labels" / split / f"{Path(full_fname).stem}.txt").write_text(
            "\n".join(objects_to_yolo_seg_lines(full_objs, 720, 960)) + "\n", encoding="utf-8"
        )

        if is_val_only:
            val_count += 2
            for f_nm, f_m, f_o in ((roi_fname, roi_mask, roi_objs), (full_fname, full_mask, full_objs)):
                cv2.imwrite(str(OUT_DS_DIR / "val_gt_day3_format" / "seg_mask" / f"{Path(f_nm).stem}.png"), f_m.astype(np.uint16))
                val_gt_entries.append({"image_name": f_nm, "subset": "real_val_day3", "objects": f_o})
                for o in f_o:
                    val_cls_hist[o["class"]] += 1
        else:
            train_count += 2
            for o in roi_objs + full_objs:
                train_cls_hist[o["class"]] += 1
            # ในโหมด full ใส่ 3 ภาพอ้างอิงเข้า val ด้วยเพื่อติดตามความคมของ Mask 1px ระหว่างเทรน
            if full_mode and in_val_ref:
                ref_fname = f"d3_val_roi_{stem}.jpg"
                cv2.imwrite(str(OUT_DS_DIR / "images" / "val" / ref_fname), roi_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                (OUT_DS_DIR / "labels" / "val" / f"{Path(ref_fname).stem}.txt").write_text(
                    "\n".join(objects_to_yolo_seg_lines(roi_objs, 960, 960)) + "\n", encoding="utf-8"
                )
                cv2.imwrite(str(OUT_DS_DIR / "val_gt_day3_format" / "seg_mask" / f"{Path(ref_fname).stem}.png"), roi_mask.astype(np.uint16))
                val_gt_entries.append({"image_name": ref_fname, "subset": "real_val_day3", "objects": roi_objs})
                val_count += 1
                for o in roi_objs:
                    val_cls_hist[o["class"]] += 1

        if len(preview_real_tiles) < 8:
            preview_real_tiles.append(render_preview_tile(roi_img, roi_mask, roi_objs, f"[{split.upper()}] {stem} ({len(roi_objs)})"))
        print(f"  [{split.upper():<5}] {img_name:<20} | {len(extracted):2d} beans | ROI={roi_box[2]-roi_box[0]}x{roi_box[3]-roi_box[1]}")

    assert real_wood_strip is not None, "Failed to extract real wood strip from Mixed_triangle3.jpg"

    # 2. โหลด Day 1/2 Real Replay Buffer (21 ภาพ, 175 เม็ด — บีบขอบ 2px ให้คมเท่า Human Label)
    print("\n[2/4] Loading Day 1/2 Replay Buffer (21 images, 175 beans, 2px tightened masks -> TRAIN ONLY)")
    d2_replay_bank: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    d2_gt = json.loads((DAY2_REAL_GT_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    for entry in sorted(d2_gt, key=lambda x: x["image_name"]):
        img_name = entry["image_name"]
        stem = Path(img_name).stem
        img_bgr = load_exif_bgr(DAY2_REAL_GT_DIR / "images" / img_name)
        oh, ow = img_bgr.shape[:2]
        seg_full, extracted = extract_beans_from_image(img_bgr, entry["objects"], erode_px=2, source_tag=f"d2_{stem}")
        id_to_cls = {i + 1: o["class"] for i, o in enumerate(entry["objects"])}
        for b in extracted:
            d2_replay_bank[b["class"]].append(b)

        roi_box = compute_active_roi_from_objects(entry["objects"], ow, oh, margin_ratio=0.18)
        roi_img, _, roi_objs = crop_and_resize_entry(img_bgr, seg_full, id_to_cls, roi_box, (960, 960))
        fname = f"d2_replay_roi_{stem}.jpg"
        cv2.imwrite(str(OUT_DS_DIR / "images" / "train" / fname), roi_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        (OUT_DS_DIR / "labels" / "train" / f"{Path(fname).stem}.txt").write_text(
            "\n".join(objects_to_yolo_seg_lines(roi_objs, 960, 960)) + "\n", encoding="utf-8"
        )
        train_count += 1
        for o in roi_objs:
            train_cls_hist[o["class"]] += 1

    print("  Day-3 Train Bank :", {c: len(d3_train_bank[c]) for c in CLASSES})
    print("  Day-3 Val Bank   :", {c: len(d3_val_bank[c]) for c in CLASSES})
    print("  Day-1/2 Replay   :", {c: len(d2_replay_bank[c]) for c in CLASSES})

    # 3. สร้าง Synthetic Multi-Class Touching Piles สำหรับ TRAIN
    n_mixed = 64 if full_mode else 60
    print(f"\n[3/4] Synthesizing Train Multi-Class & Hard Confusing-Pair Touching Piles (mode={args.mode})...")
    pile_preview_tiles: List[np.ndarray] = []

    bg_schedule = (
        (["paper"] * 32 + ["real_wood"] * 13 + ["synth_wood"] * 10 + ["paper_border"] * 9)
        if full_mode
        else (["paper"] * 30 + ["real_wood"] * 12 + ["synth_wood"] * 10 + ["paper_border"] * 8)
    )
    rng.shuffle(bg_schedule)
    for i, bg_m in enumerate(bg_schedule, start=1):
        k_cls = 7 if i <= 32 else rng.randint(3, 6)
        chosen = CLASSES if k_cls == 7 else rng.sample(CLASSES, k=k_cls)
        n_beans = rng.randint(18, 30)
        canvas, smask, objs = synthesize_mixed_pile(
            primary_bank=d3_train_bank,
            replay_bank=d2_replay_bank,
            primary_prob=0.76 if full_mode else 0.72,
            paper_patches=d3_train_paper,
            wood_strip=real_wood_strip,
            target_classes=chosen,
            num_beans=n_beans,
            bg_mode=bg_m,
            rng=rng,
        )
        fname = f"synth_tr_mixed_{i:02d}_{bg_m}.jpg"
        cv2.imwrite(str(OUT_DS_DIR / "images" / "train" / fname), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
        (OUT_DS_DIR / "labels" / "train" / f"{Path(fname).stem}.txt").write_text(
            "\n".join(objects_to_yolo_seg_lines(objs, 960, 960)) + "\n", encoding="utf-8"
        )
        train_count += 1
        for o in objs:
            train_cls_hist[o["class"]] += 1
        if len(pile_preview_tiles) < 4:
            pile_preview_tiles.append(render_preview_tile(canvas, smask, objs, f"TR Mixed #{i} ({bg_m}, {len(objs)})"))

    # 3B: Hard Confusing-Pair Piles (เน้นพิเศษ immature + good + parchment ในโหมด Final!)
    hard_groups = (
        [["good", "immature", "parchment"]] * (10 if full_mode else 6)
        + [["broken", "shell", "black"]] * (6 if full_mode else 5)
        + [["husk", "black", "broken"]] * (6 if full_mode else 5)
    )
    for i, grp in enumerate(hard_groups, start=1):
        bg_m = ["paper", "paper", "real_wood", "synth_wood"][i % 4]
        n_beans = rng.randint(16, 25)
        canvas, smask, objs = synthesize_mixed_pile(
            primary_bank=d3_train_bank,
            replay_bank=d2_replay_bank,
            primary_prob=0.80 if full_mode else 0.75,
            paper_patches=d3_train_paper,
            wood_strip=real_wood_strip,
            target_classes=grp,
            num_beans=n_beans,
            bg_mode=bg_m,
            rng=rng,
        )
        fname = f"synth_tr_hard_{i:02d}_{bg_m}.jpg"
        cv2.imwrite(str(OUT_DS_DIR / "images" / "train" / fname), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
        (OUT_DS_DIR / "labels" / "train" / f"{Path(fname).stem}.txt").write_text(
            "\n".join(objects_to_yolo_seg_lines(objs, 960, 960)) + "\n", encoding="utf-8"
        )
        train_count += 1
        for o in objs:
            train_cls_hist[o["class"]] += 1
        if len(pile_preview_tiles) < 6:
            pile_preview_tiles.append(render_preview_tile(canvas, smask, objs, f"TR Hard #{i} ({'+'.join(c[:3] for c in grp)})"))

    # 3C: 14 Single-Class Replay Dense Piles (คลาสละ 2 ภาพ)
    for cls_name in CLASSES:
        for v in range(2):
            bg_m = "paper" if v == 0 else "real_wood"
            canvas, smask, objs = synthesize_mixed_pile(
                primary_bank=d3_train_bank,
                replay_bank=d2_replay_bank,
                primary_prob=0.70,
                paper_patches=d3_train_paper,
                wood_strip=real_wood_strip,
                target_classes=[cls_name],
                num_beans=rng.randint(18, 26),
                bg_mode=bg_m,
                rng=rng,
            )
            fname = f"synth_tr_single_{cls_name}_{v + 1:02d}.jpg"
            cv2.imwrite(str(OUT_DS_DIR / "images" / "train" / fname), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            (OUT_DS_DIR / "labels" / "train" / f"{Path(fname).stem}.txt").write_text(
                "\n".join(objects_to_yolo_seg_lines(objs, 960, 960)) + "\n", encoding="utf-8"
            )
            train_count += 1
            for o in objs:
                train_cls_hist[o["class"]] += 1

    # 3D: [V2 HARDENED ONLY] เสริมจุดอ่อนทั้ง 4 ข้อที่ตรวจพบจาก Brutal Stress Test (+72 ภาพเทรน)
    v2_preview_tiles: List[np.ndarray] = []
    if v2_mode:
        print("\n[3D/4] Adding V2 Hardened Slices (+72 targeted stress-cure images: WB/Dim, ISO Noise, Empty Hard-Neg, Immature/Shell/Broken, Multi-Scale)...")

        # Slice A (+24): White-Balance & Dim/Shadow Lighting Piles (แก้ cool/warm/dim ClsAcc 23-64%)
        wb_effects = ["cool_wb"] * 8 + ["warm_wb"] * 8 + ["dim_dark"] * 5 + ["split_shadow"] * 3
        rng.shuffle(wb_effects)
        for i, eff in enumerate(wb_effects, start=1):
            bg_m = ["paper", "paper", "real_wood", "paper_border"][i % 4]
            canvas, smask, objs = synthesize_mixed_pile(
                primary_bank=d3_train_bank,
                replay_bank=d2_replay_bank,
                primary_prob=0.78,
                paper_patches=d3_train_paper,
                wood_strip=real_wood_strip,
                target_classes=CLASSES,
                num_beans=rng.randint(16, 24),
                bg_mode=bg_m,
                rng=rng,
            )
            canvas, jq = apply_v2_domain_effect(canvas, eff, rng)
            fname = f"v2_tr_wb_{i:02d}_{eff}.jpg"
            cv2.imwrite(str(OUT_DS_DIR / "images" / "train" / fname), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), jq])
            (OUT_DS_DIR / "labels" / "train" / f"{Path(fname).stem}.txt").write_text(
                "\n".join(objects_to_yolo_seg_lines(objs, 960, 960)) + "\n", encoding="utf-8"
            )
            train_count += 1
            for o in objs:
                train_cls_hist[o["class"]] += 1
            if len(v2_preview_tiles) < 2 and eff in ("cool_wb", "warm_wb", "dim_dark"):
                v2_preview_tiles.append(render_preview_tile(canvas, smask, objs, f"V2 SliceA: {eff} ({len(objs)})"))

        # Slice B (+14): ISO Noise & Low-JPEG/Blur Piles (แก้ ISO noise ลั่นเป็น husk + FP=4)
        noise_effects = ["iso_noise"] * 6 + ["cool_noise"] * 2 + ["low_jpeg_blur"] * 6
        for i, eff in enumerate(noise_effects, start=1):
            bg_m = ["paper", "real_wood", "paper_both_borders", "dark_slate"][i % 4]
            canvas, smask, objs = synthesize_mixed_pile(
                primary_bank=d3_train_bank,
                replay_bank=d2_replay_bank,
                primary_prob=0.76,
                paper_patches=d3_train_paper,
                wood_strip=real_wood_strip,
                target_classes=CLASSES,
                num_beans=rng.randint(14, 22),
                bg_mode=bg_m,
                rng=rng,
            )
            canvas, jq = apply_v2_domain_effect(canvas, eff, rng)
            fname = f"v2_tr_cam_{i:02d}_{eff}.jpg"
            cv2.imwrite(str(OUT_DS_DIR / "images" / "train" / fname), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), jq])
            (OUT_DS_DIR / "labels" / "train" / f"{Path(fname).stem}.txt").write_text(
                "\n".join(objects_to_yolo_seg_lines(objs, 960, 960)) + "\n", encoding="utf-8"
            )
            train_count += 1
            for o in objs:
                train_cls_hist[o["class"]] += 1
            if len(v2_preview_tiles) < 4 and eff in ("iso_noise", "low_jpeg_blur"):
                v2_preview_tiles.append(render_preview_tile(canvas, smask, objs, f"V2 SliceB: {eff} ({len(objs)})"))

        # Slice C (+12): Hard Negative Empty Backgrounds (0 เม็ด — ฆ่า Ghost FP มุมขอบโต๊ะไม้ + Noise!)
        neg_kinds = (
            ["paper_double_border_crease"] * 4
            + ["paper_border_noisy"] * 4
            + ["dark_slate_empty"] * 2
            + ["dark_wood_empty"] * 2
        )
        for i, nk in enumerate(neg_kinds, start=1):
            canvas = make_empty_hard_negative(nk, d3_train_paper, real_wood_strip, rng, 960, 960)
            fname = f"v2_tr_empty_{i:02d}_{nk}.jpg"
            cv2.imwrite(str(OUT_DS_DIR / "images" / "train" / fname), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            # ไฟล์ label ว่างเปล่าสำหรับ Hard Negative (YOLO background-only image)
            (OUT_DS_DIR / "labels" / "train" / f"{Path(fname).stem}.txt").write_text("", encoding="utf-8")
            train_count += 1
            if len(v2_preview_tiles) < 5 and nk == "paper_border_noisy":
                v2_preview_tiles.append(render_preview_tile(canvas, np.zeros((960, 960), dtype=np.uint16), [], f"V2 SliceC: Empty HardNeg (0)"))

        # Slice D (+12): Extra Fine-Grained Contrast Piles (ดัน immature F1=70% และ shell<->broken)
        v2_hard_grps = [["immature", "good", "parchment"]] * 6 + [["shell", "broken", "good"]] * 6
        v2_hard_effs = ["none", "cool_wb", "warm_wb", "dim_dark", "iso_noise", "none"]
        for i, grp in enumerate(v2_hard_grps, start=1):
            bg_m = ["paper", "real_wood", "dark_slate", "paper_border"][i % 4]
            canvas, smask, objs = synthesize_mixed_pile(
                primary_bank=d3_train_bank,
                replay_bank=d2_replay_bank,
                primary_prob=0.82,
                paper_patches=d3_train_paper,
                wood_strip=real_wood_strip,
                target_classes=grp,
                num_beans=rng.randint(16, 24),
                bg_mode=bg_m,
                rng=rng,
            )
            eff = v2_hard_effs[(i - 1) % len(v2_hard_effs)]
            jq = 84
            if eff != "none":
                canvas, jq = apply_v2_domain_effect(canvas, eff, rng)
            fname = f"v2_tr_cls_{i:02d}_{eff}.jpg"
            cv2.imwrite(str(OUT_DS_DIR / "images" / "train" / fname), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), jq])
            (OUT_DS_DIR / "labels" / "train" / f"{Path(fname).stem}.txt").write_text(
                "\n".join(objects_to_yolo_seg_lines(objs, 960, 960)) + "\n", encoding="utf-8"
            )
            train_count += 1
            for o in objs:
                train_cls_hist[o["class"]] += 1
            if len(v2_preview_tiles) < 6 and eff != "none":
                v2_preview_tiles.append(render_preview_tile(canvas, smask, objs, f"V2 SliceD: {'+'.join(c[:3] for c in grp)} ({eff})"))

        # Slice E (+10): Multi-Scale Far/Small-Bean & Close-Up Piles (แก้ภาพไกล/ขอบ 2 ด้าน/จ่อใกล้)
        for i in range(1, 11):
            is_small = i <= 6
            bg_m = "paper_both_borders" if (is_small and i % 2 == 0) else ("real_wood" if i % 3 == 0 else "paper")
            s_rng = (0.38, 0.56) if is_small else (1.22, 1.48)
            n_b = rng.randint(18, 26) if is_small else rng.randint(5, 8)
            canvas, smask, objs = synthesize_mixed_pile(
                primary_bank=d3_train_bank,
                replay_bank=d2_replay_bank,
                primary_prob=0.80,
                paper_patches=d3_train_paper,
                wood_strip=real_wood_strip,
                target_classes=CLASSES,
                num_beans=n_b,
                bg_mode=bg_m,
                rng=rng,
                scale_range=s_rng,
            )
            tag = "small" if is_small else "large"
            fname = f"v2_tr_scale_{i:02d}_{tag}.jpg"
            cv2.imwrite(str(OUT_DS_DIR / "images" / "train" / fname), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            (OUT_DS_DIR / "labels" / "train" / f"{Path(fname).stem}.txt").write_text(
                "\n".join(objects_to_yolo_seg_lines(objs, 960, 960)) + "\n", encoding="utf-8"
            )
            train_count += 1
            for o in objs:
                train_cls_hist[o["class"]] += 1
            if len(v2_preview_tiles) < 8 and i in (2, 8):
                v2_preview_tiles.append(render_preview_tile(canvas, smask, objs, f"V2 SliceE: {tag} ({len(objs)})"))

    # 4. สร้าง Validation Mixed Touching Piles (unseen layouts/rotations/overlaps)
    val_bgs = (
        ["paper"] * 6 + ["real_wood"] * 4 + ["synth_wood"] * 2 + ["paper_border"] * 2
        if full_mode
        else ["paper"] * 6 + ["real_wood"] * 3 + ["synth_wood"] * 3
    )
    print(f"[4/4] Synthesizing {len(val_bgs)} Validation Mixed Touching Piles...")
    for i, bg_m in enumerate(val_bgs, start=1):
        if full_mode and i in (13, 14):
            chosen = ["good", "immature", "parchment"]
        else:
            k_cls = 7 if i <= 10 else val_rng.randint(4, 6)
            chosen = CLASSES if k_cls == 7 else val_rng.sample(CLASSES, k=k_cls)
        n_beans = val_rng.randint(16, 24)
        canvas, smask, objs = synthesize_mixed_pile(
            primary_bank=d3_val_bank,
            replay_bank=None,
            primary_prob=1.0,
            paper_patches=d3_val_paper,
            wood_strip=real_wood_strip,
            target_classes=chosen,
            num_beans=n_beans,
            bg_mode=bg_m,
            rng=val_rng,
        )
        fname = f"val_mixed_pile_{i:02d}_{bg_m}.jpg"
        cv2.imwrite(str(OUT_DS_DIR / "images" / "val" / fname), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        (OUT_DS_DIR / "labels" / "val" / f"{Path(fname).stem}.txt").write_text(
            "\n".join(objects_to_yolo_seg_lines(objs, 960, 960)) + "\n", encoding="utf-8"
        )
        cv2.imwrite(str(OUT_DS_DIR / "val_gt_day3_format" / "seg_mask" / f"{Path(fname).stem}.png"), smask.astype(np.uint16))
        val_gt_entries.append({"image_name": fname, "subset": "unseen_mixed_pile_val", "objects": objs})
        val_count += 1
        for o in objs:
            val_cls_hist[o["class"]] += 1
        if len(pile_preview_tiles) < 8:
            pile_preview_tiles.append(render_preview_tile(canvas, smask, objs, f"VAL Mixed #{i} ({bg_m}, {len(objs)})"))

    # 4B: [V2 HARDENED ONLY] เพิ่ม 6 Hardened Val Images (WB / Dim / ISO Noise / Small / Empty) ให้ best.pt เลือกจุดที่เก่งทั้งภาพสะอาดและสภาพโหด!
    if v2_mode:
        val_hard_specs = [
            ("cool_wb", "paper", CLASSES, 16, (0.76, 1.02)),
            ("warm_wb", "real_wood", ["immature", "good", "parchment"], 16, (0.76, 1.02)),
            ("dim_dark", "paper", CLASSES, 16, (0.76, 1.02)),
            ("iso_noise", "paper_both_borders", CLASSES, 15, (0.74, 0.98)),
            ("none", "paper_both_borders", CLASSES, 20, (0.42, 0.56)),
            ("empty", "paper_double_border_crease", CLASSES, 0, (0.8, 1.0)),
        ]
        for j, (eff, bg_m, t_cls, n_b, s_rng) in enumerate(val_hard_specs, start=1):
            if eff == "empty":
                canvas = make_empty_hard_negative(bg_m, d3_val_paper, real_wood_strip, val_rng, 960, 960)
                smask = np.zeros((960, 960), dtype=np.uint16)
                objs = []
                jq = 86
            else:
                canvas, smask, objs = synthesize_mixed_pile(
                    primary_bank=d3_val_bank,
                    replay_bank=None,
                    primary_prob=1.0,
                    paper_patches=d3_val_paper,
                    wood_strip=real_wood_strip,
                    target_classes=t_cls,
                    num_beans=n_b,
                    bg_mode=bg_m,
                    rng=val_rng,
                    scale_range=s_rng,
                )
                jq = 86
                if eff != "none":
                    canvas, jq = apply_v2_domain_effect(canvas, eff, val_rng)
            fname = f"val_v2_hard_{j:02d}_{eff}.jpg"
            cv2.imwrite(str(OUT_DS_DIR / "images" / "val" / fname), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), jq])
            lbl_txt = "\n".join(objects_to_yolo_seg_lines(objs, 960, 960)) + ("\n" if objs else "")
            (OUT_DS_DIR / "labels" / "val" / f"{Path(fname).stem}.txt").write_text(lbl_txt, encoding="utf-8")
            cv2.imwrite(str(OUT_DS_DIR / "val_gt_day3_format" / "seg_mask" / f"{Path(fname).stem}.png"), smask.astype(np.uint16))
            val_gt_entries.append({"image_name": fname, "subset": "v2_hardened_val", "objects": objs})
            val_count += 1
            for o in objs:
                val_cls_hist[o["class"]] += 1

    # บันทึก Ground Truth ของชุด Val, dataset_mode.json และ beans_seg_day3.yaml
    (OUT_DS_DIR / "val_gt_day3_format" / "ground_truth.json").write_text(
        json.dumps(val_gt_entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DS_DIR / "dataset_info.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "train_images": train_count,
                "val_images": val_count,
                "day3_train_real_beans": {c: len(d3_train_bank[c]) for c in CLASSES},
                "train_class_instances": dict(sorted(train_cls_hist.items())),
                "val_class_instances": dict(sorted(val_cls_hist.items())),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    yaml_text = (
        f"path: {OUT_DS_DIR.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
        + "\n"
    )
    (OUT_DS_DIR / "beans_seg_day3.yaml").write_text(yaml_text, encoding="utf-8")

    if len(preview_real_tiles) == 8:
        cv2.imwrite(str(QUALITY_DIR / "01_day3_real_split_preview.jpg"), np.vstack([np.hstack(preview_real_tiles[:4]), np.hstack(preview_real_tiles[4:8])]))
    if len(pile_preview_tiles) == 8:
        cv2.imwrite(str(QUALITY_DIR / "02_day3_mixed_piles_preview.jpg"), np.vstack([np.hstack(pile_preview_tiles[:4]), np.hstack(pile_preview_tiles[4:8])]))
    if len(v2_preview_tiles) >= 8:
        cv2.imwrite(str(QUALITY_DIR / "03_day3_v2_hardened_preview.jpg"), np.vstack([np.hstack(v2_preview_tiles[:4]), np.hstack(v2_preview_tiles[4:8])]))

    if colab_zip_path.exists():
        colab_zip_path.unlink()
    with zipfile.ZipFile(colab_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(OUT_DS_DIR.rglob("*")):
            if f.is_file():
                zf.write(f, Path("datasets_seg_day3") / f.relative_to(OUT_DS_DIR))

    zip_mb = colab_zip_path.stat().st_size / (1024 * 1024)
    print("=" * 78)
    print(f"[SUCCESS] Mode               : {args.mode.upper()}")
    print(f"[SUCCESS] Day 3 Train Images : {train_count} images | {sum(train_cls_hist.values()):,} instances")
    print(f"          Train Class Counts : {dict(sorted(train_cls_hist.items()))}")
    print(f"[SUCCESS] Day 3 Val Images   : {val_count} images | {sum(val_cls_hist.values()):,} instances")
    print(f"          Val Class Counts   : {dict(sorted(val_cls_hist.items()))}")
    print(f"[SUCCESS] Colab Zip Ready    : {colab_zip_path} ({zip_mb:.2f} MB)")
    print("=" * 78)


if __name__ == "__main__":
    main()
