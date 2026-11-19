#!/usr/bin/env python3
"""
generate_brutal_stress_data.py — ตัวสร้างชุดข้อมูลภาพเมล็ดกาแฟแบบโหด (Adversarial Stress-Test Suite)
สำหรับประเมินขีดจำกัดและหาจุดอ่อนของโมเดล Day 3 ก่อนซ้อมและสอบจริง

สร้างชุดทดสอบ 7 หมวดสุดโหด (ค่าเริ่มต้นหมวดละ 6 ภาพ = 42 ภาพ) พร้อม Ground Truth ระดับพิกเซล:
  1. 01_dense_touching     : กองแน่น 18-28 เม็ด ชิดติดกัน/ซ้อนทับสูง ครบทั้ง 7 คลาส
  2. 02_harsh_lighting     : แสงมืด 50%, แสงแฟลชจ้า/Glare, ไฟสีเหลือง/ฟ้าเพี้ยน, เงาพาดครึ่งกอง
  3. 03_hostile_bg         : พื้นหลังนอกกรอบ (โต๊ะดำเข้ม, กระดาษลัง, ถาดโลหะ, สมุดกริด, ผ้าทอ, ขอบไม้ 2 ด้าน)
  4. 04_highres_edge_roi   : ภาพกล้องมือถือจริง 3072x4096 & 4096x3072 (วางชิดขอบ/มุม/กระจายกว้าง/พื้นไม้เข้ม)
                             -> ทดสอบ detect_active_pile_roi() ใน infer.py โดยตรง!
  5. 05_camera_artifacts   : หลุดโฟกัส, Motion Blur, ISO Noise จัด, JPEG Quality 25, กล้องเอียง 3D Perspective
  6. 06_scale_extremes     : เม็ดเล็กจิ๋วถ่ายไกล (0.45-0.60x) และ เม็ดยักษ์ถ่ายจ่อใกล้ (1.40-1.65x)
  7. 07_confusing_and_empty: กองรวมคลาสคู่แฝดแยกยากล้วน, เม็ดเดี่ยว 1 เม็ด, และภาพเปล่า 0 เม็ด (จับ Ghost FP)

ผลลัพธ์อยู่ในรูปแบบเดียวกับ Dataset_Day-3 (Images/ + labels.json) จึงสามารถนำไปป้อนให้
  - evaluate_stress_test.py เพื่อวิเคราะห์คะแนน/จุดอ่อนทันที
  - หรือ mock_server.py --dataset track2/day-3/Task/stress_test/dataset เพื่อยิงจาก Raspberry Pi ข้าม LAN ได้เลย!
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
DAY3_DIR = TASK_DIR.parent
TRACK2_DIR = DAY3_DIR.parent

DAY3_DS_DIR = TASK_DIR / "Dataset_Day-3"
DAY2_REAL_GT_DIR = (
    TRACK2_DIR / "day-2" / "team_01_day2" / "task" / "datasets_seg" / "ground_truth_day2_format" / "real_21_full"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "dataset"

CLASSES: List[str] = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]


def load_exif_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return cv2.cvtColor(np.array(im.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR)


def extract_beans(
    img_bgr: np.ndarray,
    objects: List[Dict[str, Any]],
    erode_px: int = 0,
    source_tag: str = "d3",
) -> List[Dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)) if erode_px > 0 else None
    out: List[Dict[str, Any]] = []
    for obj in objects:
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
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < 200:
            continue
        clean = np.zeros_like(local_mask)
        cv2.drawContours(clean, [best], -1, 255, thickness=-1)
        ys, xs = np.where(clean > 0)
        cx1, cx2 = max(0, int(xs.min()) - 2), min(clean.shape[1], int(xs.max()) + 3)
        cy1, cy2 = max(0, int(ys.min()) - 2), min(clean.shape[0], int(ys.max()) + 3)
        out.append(
            {
                "class": obj["class"],
                "bgr": img_bgr[y1 + cy1 : y1 + cy2, x1 + cx1 : x1 + cx2].copy(),
                "mask": clean[cy1:cy2, cx1:cx2].copy(),
                "source": source_tag,
            }
        )
    return out


def load_source_assets() -> Tuple[Dict[str, List[Dict[str, Any]]], List[np.ndarray], np.ndarray]:
    bank: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    paper_patches: List[np.ndarray] = []
    wood_strip: Optional[np.ndarray] = None

    # 1. Day 3 real 16 images (124 beans)
    d3_labels = json.loads((DAY3_DS_DIR / "labels.json").read_text(encoding="utf-8"))
    for entry in sorted(d3_labels, key=lambda x: x["image_name"]):
        img_name = entry["image_name"]
        img_bgr = load_exif_bgr(DAY3_DS_DIR / "Images" / img_name)
        oh, ow = img_bgr.shape[:2]
        if img_name == "Mixed_triangle3.jpg":
            wood_strip = img_bgr[0:145, 80 : ow - 80].copy()
        paper_patches.append(img_bgr[int(oh * 0.72) : int(oh * 0.88), int(ow * 0.15) : int(ow * 0.85)].copy())
        for b in extract_beans(img_bgr, entry["objects"], erode_px=0, source_tag=f"d3_{Path(img_name).stem}"):
            bank[b["class"]].append(b)

    # 2. Day 1/2 real 21 images (175 beans) for maximum bean appearance diversity
    d2_gt_path = DAY2_REAL_GT_DIR / "ground_truth.json"
    if d2_gt_path.exists():
        d2_labels = json.loads(d2_gt_path.read_text(encoding="utf-8"))
        for entry in sorted(d2_labels, key=lambda x: x["image_name"]):
            ip = DAY2_REAL_GT_DIR / "images" / entry["image_name"]
            if not ip.exists():
                continue
            img_bgr = load_exif_bgr(ip)
            for b in extract_beans(img_bgr, entry["objects"], erode_px=2, source_tag=f"d2_{ip.stem}"):
                bank[b["class"]].append(b)

    if wood_strip is None:
        wood_strip = np.full((140, 800, 3), (65, 98, 138), dtype=np.uint8)
    return bank, paper_patches, wood_strip


# =====================================================================
# 1. Background Generators (Clean Paper, Real Wood, & Unseen Hostile BGs)
# =====================================================================

def make_real_wood_bg(wood_strip: np.ndarray, w: int, h: int, rng: random.Random, darken: float = 1.0) -> np.ndarray:
    strip = cv2.rotate(wood_strip, cv2.ROTATE_90_CLOCKWISE) if rng.random() < 0.5 else wood_strip.copy()
    if rng.random() < 0.5:
        strip = cv2.flip(strip, 1)
    tile_x = np.hstack([strip, cv2.flip(strip, 1)])
    tile_xy = np.vstack([tile_x, cv2.flip(tile_x, 0)])
    th, tw = tile_xy.shape[:2]
    tiled = np.tile(tile_xy, ((h // th) + 2, (w // tw) + 2, 1))
    oy = rng.randint(0, max(0, tiled.shape[0] - h))
    ox = rng.randint(0, max(0, tiled.shape[1] - w))
    bg = tiled[oy : oy + h, ox : ox + w].astype(np.float32)
    gain = rng.uniform(0.82, 1.25) * darken
    return np.clip(bg * gain, 15, 235).astype(np.uint8)


def make_paper_bg(
    paper_patches: List[np.ndarray],
    w: int,
    h: int,
    rng: random.Random,
    wood_strip: Optional[np.ndarray] = None,
    border_mode: str = "none",
) -> np.ndarray:
    base = rng.choice(paper_patches)
    bg = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    gx = np.linspace(rng.uniform(-14, 14), rng.uniform(-14, 14), w, dtype=np.float32)[None, :, None]
    gy = np.linspace(rng.uniform(-14, 14), rng.uniform(-14, 14), h, dtype=np.float32)[:, None, None]
    tint = np.array([rng.uniform(-7, 7), rng.uniform(-6, 6), rng.uniform(-7, 7)], dtype=np.float32)[None, None, :]
    bg = np.clip(bg + gx + gy + tint, 138, 242).astype(np.uint8)

    if wood_strip is not None and border_mode in ("top", "bottom", "both"):
        bh_top = int(h * rng.uniform(0.06, 0.11))
        bh_bot = int(h * rng.uniform(0.06, 0.11))
        if border_mode in ("top", "both"):
            bg[:bh_top, :] = make_real_wood_bg(wood_strip, w, bh_top, rng)
            # Subtle shadow along paper edge
            grad = np.linspace(0.75, 1.0, max(8, h // 90), dtype=np.float32)[:, None, None]
            bg[bh_top : bh_top + len(grad), :] = np.clip(bg[bh_top : bh_top + len(grad), :] * grad, 0, 255).astype(np.uint8)
        if border_mode in ("bottom", "both"):
            bg[-bh_bot:, :] = make_real_wood_bg(wood_strip, w, bh_bot, rng)
            grad = np.linspace(1.0, 0.75, max(8, h // 90), dtype=np.float32)[:, None, None]
            bg[-bh_bot - len(grad) : -bh_bot, :] = np.clip(bg[-bh_bot - len(grad) : -bh_bot, :] * grad, 0, 255).astype(np.uint8)
    return bg


def make_hostile_bg(
    kind: str,
    w: int,
    h: int,
    rng: random.Random,
    paper_patches: List[np.ndarray],
    wood_strip: np.ndarray,
) -> np.ndarray:
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))

    if kind == "dark_slate":
        # โต๊ะหินชนวน/ดำด้าน (พรางเม็ด black และ husk!)
        base = np.full((h, w, 3), (38, 40, 44), dtype=np.float32)
        low_noise = cv2.resize(np_rng.normal(0, 12, (h // 16, w // 16, 1)).astype(np.float32), (w, h))
        hi_noise = np_rng.normal(0, 4.5, (h, w, 3)).astype(np.float32)
        bg = base + low_noise[:, :, None] + hi_noise
        return np.clip(bg, 16, 85).astype(np.uint8)

    if kind == "brown_kraft":
        # กระดาษลัง/คราฟต์สีน้ำตาลอุ่น (พรางเม็ด parchment/good/husk!)
        base = np.full((h, w, 3), (95, 132, 166), dtype=np.float32)
        fibers = np_rng.normal(0, 7.0, (h, w, 1)).astype(np.float32)
        streaks = cv2.GaussianBlur(np_rng.normal(0, 18.0, (h // 4, w // 4, 1)).astype(np.float32), (15, 3), 0)
        streaks = cv2.resize(streaks, (w, h))[:, :, None]
        bg = base + fibers + streaks
        return np.clip(bg, 55, 205).astype(np.uint8)

    if kind == "metal_tray":
        # ถาดสเตนเลส/โลหะมันวาวมีแสงสะท้อนพาดเฉียง + รอยขีดข่วน
        y = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        x = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        spec = np.exp(-((x * 0.7 + y * 0.5 - 0.15) ** 2) / 0.18) * 65.0
        brushed = cv2.GaussianBlur(np_rng.normal(0, 9.0, (h, w)).astype(np.float32), (31, 1), 0)
        gray = np.clip(142.0 + spec + brushed, 70, 238)
        bg = np.stack([gray * 1.02, gray, gray * 0.98], axis=2).astype(np.uint8)
        # Subtle tray rim border
        cv2.rectangle(bg, (18, 18), (w - 18, h - 18), (110, 112, 115), 3)
        return bg

    if kind == "lined_notebook":
        # กระดาษสมุดมีเส้นกริด/เส้นบรรทัดสีฟ้า + เส้นกั้นหน้าสีแดง
        bg = make_paper_bg(paper_patches, w, h, rng)
        step = max(28, w // 24)
        for yy in range(step, h, step):
            cv2.line(bg, (0, yy), (w, yy), (218, 198, 178), 1, cv2.LINE_AA)
        for xx in range(step, w, step):
            cv2.line(bg, (xx, 0), (xx, h), (222, 208, 195), 1, cv2.LINE_AA)
        cv2.line(bg, (int(w * 0.14), 0), (int(w * 0.14), h), (155, 150, 220), 2, cv2.LINE_AA)
        return bg

    if kind == "woven_cloth":
        # ผ้าทอสีครีม/น้ำตาลอ่อนมีลวดลายตารางละเอียด
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        weave = (np.sin(xx * 0.85) * 6.0 + np.sin(yy * 0.85) * 6.0 + np.sin((xx + yy) * 0.42) * 4.0)
        noise = np_rng.normal(0, 4.0, (h, w)).astype(np.float32)
        r = np.clip(182 + weave + noise, 90, 235)
        g = np.clip(168 + weave + noise, 85, 225)
        b = np.clip(146 + weave + noise, 75, 210)
        return np.stack([b, g, r], axis=2).astype(np.uint8)

    if kind == "double_wood_border":
        return make_paper_bg(paper_patches, w, h, rng, wood_strip=wood_strip, border_mode="both")

    if kind == "paper_with_crease":
        bg = make_paper_bg(paper_patches, w, h, rng, wood_strip=wood_strip, border_mode="top")
        # รอยพับกระดาษเป็นเงายาวพาดกลางภาพ (ดัก False Positive)
        cx = int(w * rng.uniform(0.35, 0.65))
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        dist_line = np.abs((xx - cx) + (yy - h * 0.5) * 0.15)
        crease = 1.0 - 0.14 * np.exp(-(dist_line ** 2) / 140.0)
        return np.clip(bg.astype(np.float32) * crease[:, :, None], 0, 255).astype(np.uint8)

    return make_real_wood_bg(wood_strip, w, h, rng)


# =====================================================================
# 2. Bean Placement & Photometric / Geometric Torture Effects
# =====================================================================

def transform_bean(
    bgr: np.ndarray,
    mask_u8: np.ndarray,
    angle_deg: float,
    scale_x: float,
    scale_y: float,
    flip_h: bool,
    gain: float,
    trim_px: float = 1.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if flip_h:
        bgr = cv2.flip(bgr, 1)
        mask_u8 = cv2.flip(mask_u8, 1)
    h, w = mask_u8.shape[:2]
    # Apply slight non-uniform aspect ratio stretch for out-of-distribution shape diversity
    nw = max(12, int(round(w * scale_x)))
    nh = max(12, int(round(h * scale_y)))
    interp = cv2.INTER_AREA if min(scale_x, scale_y) < 1.0 else cv2.INTER_LINEAR
    s_bgr = cv2.resize(bgr, (nw, nh), interpolation=interp)
    s_mask = cv2.resize(mask_u8, (nw, nh), interpolation=cv2.INTER_LINEAR)
    _, s_mask = cv2.threshold(s_mask, 140, 255, cv2.THRESH_BINARY)

    diag = int(math.ceil(math.hypot(nw, nh) * 1.15)) + 8
    cx, cy = nw / 2.0, nh / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] += diag / 2.0 - cx
    M[1, 2] += diag / 2.0 - cy

    rot_bgr = cv2.warpAffine(s_bgr, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    rot_mask = cv2.warpAffine(s_mask, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    _, rot_mask = cv2.threshold(rot_mask, 150, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(rot_mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty bean after transform")
    x1, x2 = max(0, int(xs.min()) - 2), min(diag, int(xs.max()) + 3)
    y1, y2 = max(0, int(ys.min()) - 2), min(diag, int(ys.max()) + 3)

    c_bgr = np.clip(rot_bgr[y1:y2, x1:x2].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    c_mask = rot_mask[y1:y2, x1:x2]

    dist = cv2.distanceTransform(c_mask, cv2.DIST_L2, 5)
    t_eff = min(trim_px, max(0.4, float(dist.max()) * 0.14))
    alpha = np.clip((dist - t_eff) / 1.8, 0.0, 1.0).astype(np.float32)
    hard_mask = ((dist >= t_eff) & (c_mask > 0)).astype(np.uint8) * 255
    return c_bgr, hard_mask, alpha


def extract_objects_from_mask(
    raw_id_mask: np.ndarray,
    id_to_cls: Dict[int, str],
    min_area: int = 60,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    h, w = raw_id_mask.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint16)
    final_objs: List[Dict[str, Any]] = []
    fid = 1
    for rid, cls_name in id_to_cls.items():
        ys, xs = np.where(raw_id_mask == rid)
        if len(xs) < min_area:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        sub = (raw_id_mask[y1:y2, x1:x2] == rid).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < min_area or len(best) < 3:
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
    return final_mask, final_objs


def place_beans_on_canvas(
    canvas: np.ndarray,
    bank: Dict[str, List[Dict[str, Any]]],
    target_classes: List[str],
    num_beans: int,
    rng: random.Random,
    scale_range: Tuple[float, float] = (0.65, 0.95),
    max_overlap_ratio: float = 0.14,
    min_keep_ratio: float = 0.80,
    touch_prob: float = 0.85,
    placement_box: Optional[Tuple[int, int, int, int]] = None,
    trim_px: float = 1.6,
    shadow_strength: float = 0.28,
    shadow_offset: Tuple[int, int] = (4, 5),
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    ch, cw = canvas.shape[:2]
    canvas = canvas.copy()
    raw_id_mask = np.zeros((ch, cw), dtype=np.uint16)
    placed_area: Dict[int, int] = {}
    id_to_cls: Dict[int, str] = {}
    occupied_centers: List[Tuple[float, float, float]] = []

    if num_beans <= 0:
        return canvas, raw_id_mask, id_to_cls

    bx1, by1, bx2, by2 = placement_box if placement_box is not None else (24, 24, cw - 24, ch - 24)
    cls_seq = list(target_classes) * (num_beans // len(target_classes)) + rng.sample(
        target_classes, k=num_beans % len(target_classes)
    )
    rng.shuffle(cls_seq)

    cur_id = 1
    sh_dx, sh_dy = shadow_offset
    for cls_name in cls_seq:
        sample = rng.choice(bank[cls_name])
        s_base = rng.uniform(scale_range[0], scale_range[1])
        aspect_jit = rng.uniform(0.92, 1.08)
        angle = rng.uniform(0.0, 360.0)
        flip_h = rng.random() < 0.5
        gain = rng.uniform(0.90, 1.10)
        b_bgr, b_mask, b_alpha = transform_bean(
            sample["bgr"],
            sample["mask"],
            angle,
            s_base * aspect_jit,
            s_base / aspect_jit,
            flip_h,
            gain,
            trim_px=trim_px,
        )
        bh, bw = b_mask.shape[:2]
        bean_bool = b_mask > 0
        bean_area = int(np.count_nonzero(bean_bool))
        if bean_area < 80:
            continue
        rad = 0.5 * math.hypot(bw, bh) * 0.42

        for _ in range(280):
            if occupied_centers and rng.random() < touch_prob:
                ax, ay, ar = rng.choice(occupied_centers)
                th = rng.uniform(0.0, 2.0 * math.pi)
                dist = (ar + rad) * rng.uniform(0.72, 1.06)
                cx = ax + math.cos(th) * dist
                cy = ay + math.sin(th) * dist
            else:
                cx = rng.uniform(bx1 + bw * 0.55, max(bx1 + bw * 0.56, bx2 - bw * 0.55))
                cy = rng.uniform(by1 + bh * 0.55, max(by1 + bh * 0.56, by2 - bh * 0.55))

            px = int(round(cx - bw / 2.0))
            py = int(round(cy - bh / 2.0))
            if px < bx1 or py < by1 or (px + bw) >= bx2 or (py + bh) >= by2:
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
                if rem < int(placed_area[int(aid)] * min_keep_ratio):
                    bad_occ = True
                    break
            if bad_occ:
                continue

            # Cast drop shadow
            sy1, sy2 = max(0, py + sh_dy), min(ch, py + bh + sh_dy)
            sx1, sx2 = max(0, px + sh_dx), min(cw, px + bw + sh_dx)
            ay1, ax1 = sy1 - (py + sh_dy), sx1 - (px + sh_dx)
            sh_h, sh_w = sy2 - sy1, sx2 - sx1
            if sh_h > 0 and sh_w > 0:
                k = max(5, (min(bw, bh) // 14) | 1)
                sh_m = cv2.GaussianBlur(b_alpha[ay1 : ay1 + sh_h, ax1 : ax1 + sh_w], (k, k), 0)[:, :, None]
                canvas[sy1:sy2, sx1:sx2] = np.clip(
                    canvas[sy1:sy2, sx1:sx2].astype(np.float32) * (1.0 - shadow_strength * sh_m), 0, 255
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
            break

    return canvas, raw_id_mask, id_to_cls


def apply_harsh_lighting(img_bgr: np.ndarray, effect: str, rng: random.Random) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    f = img_bgr.astype(np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))

    if effect == "dim_underexposed":
        f = f * 0.50 + np_rng.normal(0, 5.5, f.shape).astype(np.float32)
    elif effect == "glare_overexposed":
        yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        cx, cy = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
        glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.32)[:, :, None] * 95.0
        f = f * 1.16 + glare + 12.0
    elif effect == "warm_tungsten":
        # โทนเหลืองส้มจัด (2700K) — ทดสอบการแยก immature / good / parchment
        f[:, :, 0] *= 0.68  # B down
        f[:, :, 1] *= 0.96  # G slight down
        f[:, :, 2] *= 1.24  # R up
    elif effect == "cool_fluorescent":
        # โทนฟ้าเขียวจัด (6800K)
        f[:, :, 0] *= 1.25  # B up
        f[:, :, 1] *= 1.03  # G slight up
        f[:, :, 2] *= 0.78  # R down
    elif effect == "split_shadow":
        # เงาดำพาดครึ่งกองเมล็ดกาแฟ (เหมือนคนยืนบังไฟ)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        diag = (xx * 0.75 + yy * 0.65) - 0.68
        mask = 1.0 / (1.0 + np.exp(-diag * 26.0))  # smooth hard-ish shadow edge
        gain = 0.44 + 0.70 * mask
        f = f * gain[:, :, None]
    elif effect == "vignette_high_contrast":
        yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        vig = np.clip(1.0 - 0.45 * (xx ** 2 + yy ** 2), 0.35, 1.0)[:, :, None]
        f = ((f - 128.0) * 1.32 + 128.0) * vig

    return np.clip(f, 0, 255).astype(np.uint8)


def apply_camera_artifact(
    img_bgr: np.ndarray,
    raw_mask: np.ndarray,
    effect: str,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """คืนค่า (img_bgr, raw_mask, jpeg_quality)"""
    h, w = img_bgr.shape[:2]
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))

    if effect == "defocus_blur":
        return cv2.GaussianBlur(img_bgr, (9, 9), 2.3), raw_mask, 88

    if effect == "motion_blur":
        k = 13
        kern = np.zeros((k, k), dtype=np.float32)
        kern[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2.0, k / 2.0), rng.uniform(0, 180), 1.0)
        kern = cv2.warpAffine(kern, M, (k, k))
        kern /= max(float(kern.sum()), 1e-6)
        return cv2.filter2D(img_bgr, -1, kern), raw_mask, 85

    if effect == "heavy_iso_noise":
        noisy = img_bgr.astype(np.float32) + np_rng.normal(0, 19.0, img_bgr.shape).astype(np.float32)
        return np.clip(noisy, 0, 255).astype(np.uint8), raw_mask, 80

    if effect == "low_jpeg_q25":
        return img_bgr, raw_mask, 24

    if effect in ("perspective_tilt_1", "perspective_tilt_2"):
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        if effect == "perspective_tilt_1":
            # กล้องเอียงก้มหน้าลง ~16 องศา
            dx = w * 0.11
            dy = h * 0.05
            dst = np.array([[dx, dy], [w - dx, dy], [w, h], [0, h]], dtype=np.float32)
            q = 82
        else:
            # กล้องเอียงข้าง + เบลอเล็กน้อย + JPEG ต่ำ
            dx = w * 0.05
            dy = h * 0.11
            dst = np.array([[0, 0], [w - dx, dy], [w - dx, h - dy], [0, h]], dtype=np.float32)
            q = 38
        H = cv2.getPerspectiveTransform(src, dst)
        w_img = cv2.warpPerspective(img_bgr, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        w_msk = cv2.warpPerspective(raw_mask, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if effect == "perspective_tilt_2":
            w_img = cv2.GaussianBlur(w_img, (5, 5), 1.1)
        return w_img, w_msk, q

    return img_bgr, raw_mask, 90


# =====================================================================
# 3. Main Generator Orchestrator (7 Torture Categories)
# =====================================================================

def generate_suite(out_dir: Path, seed: int = 20261119, per_category: int = 6) -> None:
    rng = random.Random(seed)
    bank, paper_patches, wood_strip = load_source_assets()
    total_bank = sum(len(v) for v in bank.values())
    print(f"[ASSETS] Loaded {total_bank} real bean sprites across {len(bank)} classes: "
          + ", ".join(f"{c}={len(bank[c])}" for c in CLASSES))

    img_out_dir = out_dir / "Images"
    mask_out_dir = out_dir / "gt_masks"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    labels_list: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []

    def save_sample(
        img_name: str,
        category: str,
        sub_case: str,
        difficulty: str,
        canvas: np.ndarray,
        raw_mask: np.ndarray,
        id_to_cls: Dict[int, str],
        jpeg_quality: int = 92,
        min_area: int = 60,
    ) -> None:
        final_mask, objs = extract_objects_from_mask(raw_mask, id_to_cls, min_area=min_area)
        cv2.imwrite(str(img_out_dir / img_name), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        cv2.imwrite(str(mask_out_dir / f"{Path(img_name).stem}.png"), final_mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        labels_list.append({"image_name": img_name, "objects": objs})
        manifest.append(
            {
                "image_name": img_name,
                "category": category,
                "sub_case": sub_case,
                "difficulty": difficulty,
                "width": int(canvas.shape[1]),
                "height": int(canvas.shape[0]),
                "num_beans": len(objs),
                "jpeg_quality": jpeg_quality,
            }
        )
        print(f"  + {img_name:<38} | {category:<22} | {sub_case:<24} | {canvas.shape[1]}x{canvas.shape[0]} | {len(objs):>2} beans")

    # ------------------------------------------------------------------
    # Category 1: 01_dense_touching (กองแน่น 18-28 เม็ด ชิดติดกัน/ซ้อนทับ)
    # ------------------------------------------------------------------
    for i in range(per_category):
        bg_type = "paper" if i % 2 == 0 else "real_wood"
        cw, ch = (960, 960)
        bg = make_paper_bg(paper_patches, cw, ch, rng) if bg_type == "paper" else make_real_wood_bg(wood_strip, cw, ch, rng)
        n_target = rng.randint(20, 28)
        canvas, rmask, id_cls = place_beans_on_canvas(
            bg, bank, CLASSES, n_target, rng,
            scale_range=(0.56, 0.78),
            max_overlap_ratio=0.22,
            min_keep_ratio=0.76,
            touch_prob=0.95,
            trim_px=1.6,
            shadow_strength=0.30,
        )
        save_sample(
            f"01_dense_{i+1:02d}_{bg_type}.jpg",
            "01_dense_touching",
            f"dense_{n_target}_{bg_type}",
            "HARD",
            canvas, rmask, id_cls, jpeg_quality=90,
        )

    # ------------------------------------------------------------------
    # Category 2: 02_harsh_lighting (แสงมืด/จ้า/WB เพี้ยน/เงาพาดครึ่งกอง)
    # ------------------------------------------------------------------
    light_effects = [
        "dim_underexposed",
        "glare_overexposed",
        "warm_tungsten",
        "cool_fluorescent",
        "split_shadow",
        "vignette_high_contrast",
    ]
    for i in range(per_category):
        eff = light_effects[i % len(light_effects)]
        bg = make_paper_bg(paper_patches, 960, 960, rng) if i % 3 != 2 else make_real_wood_bg(wood_strip, 960, 960, rng)
        canvas, rmask, id_cls = place_beans_on_canvas(
            bg, bank, CLASSES, rng.randint(11, 15), rng,
            scale_range=(0.68, 0.92),
            max_overlap_ratio=0.12,
            touch_prob=0.85,
        )
        canvas = apply_harsh_lighting(canvas, eff, rng)
        save_sample(
            f"02_light_{i+1:02d}_{eff}.jpg",
            "02_harsh_lighting",
            eff,
            "HARD",
            canvas, rmask, id_cls, jpeg_quality=90,
        )

    # ------------------------------------------------------------------
    # Category 3: 03_hostile_bg (พื้นหลังสุดโหดที่ไม่เคยอยู่ในชุดเทรน)
    # ------------------------------------------------------------------
    bg_kinds = [
        "dark_slate",
        "brown_kraft",
        "metal_tray",
        "lined_notebook",
        "woven_cloth",
        "double_wood_border",
    ]
    for i in range(per_category):
        kind = bg_kinds[i % len(bg_kinds)]
        bg = make_hostile_bg(kind, 960, 960, rng, paper_patches, wood_strip)
        pbox = (30, 120, 930, 840) if kind == "double_wood_border" else (30, 30, 930, 930)
        canvas, rmask, id_cls = place_beans_on_canvas(
            bg, bank, CLASSES, rng.randint(10, 14), rng,
            scale_range=(0.68, 0.92),
            max_overlap_ratio=0.12,
            placement_box=pbox,
            trim_px=2.0,
            shadow_strength=0.32,
        )
        save_sample(
            f"03_bg_{i+1:02d}_{kind}.jpg",
            "03_hostile_bg",
            kind,
            "BRUTAL",
            canvas, rmask, id_cls, jpeg_quality=90,
        )

    # ------------------------------------------------------------------
    # Category 4: 04_highres_edge_roi (3072x4096 & 4096x3072 จับบั๊ก ROI!)
    # ------------------------------------------------------------------
    highres_specs = [
        ("bottom_edge_beans", (3072, 4096), "paper", (200, 3150, 2920, 4020), (0.90, 1.15), 11, 0.45),
        ("top_left_corner", (3072, 4096), "paper_top", (120, 460, 1650, 1950), (0.88, 1.12), 12, 0.85),
        ("landscape_wide_spread", (4096, 3072), "paper", (180, 180, 3920, 2890), (0.90, 1.15), 14, 0.25),
        ("highres_dark_wood", (3072, 4096), "dark_wood", (550, 900, 2550, 3000), (0.92, 1.18), 12, 0.82),
        ("highres_far_with_stray", (3072, 4096), "paper", (900, 1200, 2200, 2600), (0.62, 0.80), 10, 0.88),
        ("highres_both_borders", (3072, 4096), "paper_both", (450, 620, 2650, 3420), (0.92, 1.18), 12, 0.85),
    ]
    for i in range(per_category):
        sname, (cw, ch), bg_k, pbox, s_rng, n_b, t_prob = highres_specs[i % len(highres_specs)]
        if bg_k == "paper":
            bg = make_paper_bg(paper_patches, cw, ch, rng)
        elif bg_k == "paper_top":
            bg = make_paper_bg(paper_patches, cw, ch, rng, wood_strip=wood_strip, border_mode="top")
        elif bg_k == "paper_both":
            bg = make_paper_bg(paper_patches, cw, ch, rng, wood_strip=wood_strip, border_mode="both")
        else:
            bg = make_real_wood_bg(wood_strip, cw, ch, rng, darken=0.78)

        canvas, rmask, id_cls = place_beans_on_canvas(
            bg, bank, CLASSES, n_b, rng,
            scale_range=s_rng,
            max_overlap_ratio=0.10,
            touch_prob=t_prob,
            placement_box=pbox,
            trim_px=1.5,
            shadow_offset=(12, 15),
        )
        # สำหรับ highres_far_with_stray เพิ่มเม็ดหลงเดี่ยว 2 เม็ดอยู่มุมล่างขวาไกลๆ (ดัก ROI ตัดทิ้ง!)
        if sname == "highres_far_with_stray":
            canvas, rmask2, id_cls2 = place_beans_on_canvas(
                canvas, bank, ["black", "good"], 2, rng,
                scale_range=(0.65, 0.78),
                touch_prob=0.0,
                placement_box=(2350, 3350, 2960, 3960),
                shadow_offset=(10, 12),
            )
            offset = max(id_cls.keys(), default=0)
            for rid, cname in id_cls2.items():
                rmask[rmask2 == rid] = rid + offset
                id_cls[rid + offset] = cname

        save_sample(
            f"04_hires_{i+1:02d}_{sname}.jpg",
            "04_highres_edge_roi",
            sname,
            "BRUTAL",
            canvas, rmask, id_cls, jpeg_quality=88, min_area=180,
        )

    # ------------------------------------------------------------------
    # Category 5: 05_camera_artifacts (เบลอ/สั่น/Noise/JPEG25/กล้องเอียง)
    # ------------------------------------------------------------------
    cam_effects = [
        "defocus_blur",
        "motion_blur",
        "heavy_iso_noise",
        "low_jpeg_q25",
        "perspective_tilt_1",
        "perspective_tilt_2",
    ]
    for i in range(per_category):
        eff = cam_effects[i % len(cam_effects)]
        bg = make_paper_bg(paper_patches, 1024, 1024, rng) if i % 2 == 0 else make_real_wood_bg(wood_strip, 1024, 1024, rng)
        canvas, rmask, id_cls = place_beans_on_canvas(
            bg, bank, CLASSES, rng.randint(10, 14), rng,
            scale_range=(0.68, 0.90),
            max_overlap_ratio=0.12,
            placement_box=(90, 90, 934, 934),
        )
        canvas, rmask, jq = apply_camera_artifact(canvas, rmask, eff, rng)
        save_sample(
            f"05_cam_{i+1:02d}_{eff}.jpg",
            "05_camera_artifacts",
            eff,
            "HARD",
            canvas, rmask, id_cls, jpeg_quality=jq,
        )

    # ------------------------------------------------------------------
    # Category 6: 06_scale_extremes (เม็ดเล็กจิ๋วไกล vs เม็ดยักษ์จ่อใกล้)
    # ------------------------------------------------------------------
    for i in range(per_category):
        is_tiny = i < (per_category // 2)
        bg = make_paper_bg(paper_patches, 960, 960, rng) if i % 2 == 0 else make_real_wood_bg(wood_strip, 960, 960, rng)
        if is_tiny:
            sub = f"tiny_far_{i+1}"
            canvas, rmask, id_cls = place_beans_on_canvas(
                bg, bank, CLASSES, rng.randint(22, 28), rng,
                scale_range=(0.36, 0.50),
                max_overlap_ratio=0.12,
                touch_prob=0.80,
                trim_px=1.0,
                shadow_offset=(2, 3),
            )
            min_a = 45
        else:
            sub = f"huge_closeup_{i - (per_category // 2) + 1}"
            canvas, rmask, id_cls = place_beans_on_canvas(
                bg, bank, CLASSES, rng.randint(4, 6), rng,
                scale_range=(1.35, 1.62),
                max_overlap_ratio=0.10,
                touch_prob=0.75,
                trim_px=2.0,
                shadow_offset=(8, 10),
            )
            min_a = 200
        save_sample(
            f"06_scale_{i+1:02d}_{sub}.jpg",
            "06_scale_extremes",
            sub,
            "HARD",
            canvas, rmask, id_cls, jpeg_quality=90, min_area=min_a,
        )

    # ------------------------------------------------------------------
    # Category 7: 07_confusing_and_empty (กับดักแยกคลาส + เม็ดเดี่ยว + ภาพเปล่า 0 เม็ด)
    # ------------------------------------------------------------------
    trap_specs = [
        ("trap_good_imm_parch", ["good", "immature", "parchment"], 14, "paper"),
        ("trap_broken_shell_good", ["broken", "shell", "good"], 14, "real_wood"),
        ("trap_black_husk", ["black", "husk"], 14, "paper"),
        ("sparse_single_bean", CLASSES, 1, "paper_with_crease"),
        ("empty_paper_crease_border", CLASSES, 0, "paper_with_crease"),
        ("empty_dark_wood", CLASSES, 0, "dark_wood"),
    ]
    for i in range(per_category):
        tname, tclasses, n_b, bg_k = trap_specs[i % len(trap_specs)]
        if bg_k == "paper":
            bg = make_paper_bg(paper_patches, 960, 960, rng)
        elif bg_k == "real_wood":
            bg = make_real_wood_bg(wood_strip, 960, 960, rng)
        elif bg_k == "dark_wood":
            bg = make_real_wood_bg(wood_strip, 960, 960, rng, darken=0.75)
        else:
            bg = make_hostile_bg("paper_with_crease", 960, 960, rng, paper_patches, wood_strip)

        pbox = (60, 140, 900, 900) if bg_k == "paper_with_crease" else (40, 40, 920, 920)
        canvas, rmask, id_cls = place_beans_on_canvas(
            bg, bank, tclasses, n_b, rng,
            scale_range=(0.68, 0.94),
            max_overlap_ratio=0.14,
            touch_prob=0.90,
            placement_box=pbox,
        )
        save_sample(
            f"07_trap_{i+1:02d}_{tname}.jpg",
            "07_confusing_and_empty",
            tname,
            "BRUTAL",
            canvas, rmask, id_cls, jpeg_quality=90,
        )

    # Save labels.json (compatible with mock_server.py & official evaluator) + stress_manifest.json
    (out_dir / "labels.json").write_text(json.dumps(labels_list, ensure_ascii=False), encoding="utf-8")
    (out_dir / "stress_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    cls_hist = Counter(o["class"] for e in labels_list for o in e["objects"])
    print("\n" + "=" * 78)
    print(f"[DONE] Generated {len(labels_list)} brutal stress-test images ({sum(cls_hist.values())} total beans)")
    print(f"       Output dir : {out_dir}")
    print("       Class dist : " + ", ".join(f"{c}={cls_hist[c]}" for c in CLASSES))
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Brutal Adversarial Stress-Test Coffee Bean Dataset")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=20261119, help="Random seed")
    parser.add_argument("--per-category", type=int, default=6, help="Images per torture category (default: 6 -> 42 total)")
    args = parser.parse_args()
    generate_suite(args.out, seed=args.seed, per_category=args.per_category)


if __name__ == "__main__":
    main()
