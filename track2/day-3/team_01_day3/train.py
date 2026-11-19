#!/usr/bin/env python3
"""
train.py — Coffee Bean Multi-Class Instance Segmentation Fine-Tuning Pipeline (Day 3)
Target Deployment: Raspberry Pi 5 (16GB / 8GB RAM, 64-bit OS, CPU) & Raspberry Pi 4

กลยุทธ์การเทรน Day 3 (Warm-Start Fine-Tuning จาก Day 2 best.pt):
  1. เริ่มต้นจากค่าน้ำหนัก Day 2 (pretrained/day2_best.pt) ด้วย Learning Rate ที่นุ่มนวล (lr0=5e-4, AdamW + Cosine LR)
     เพื่อคงความสามารถการแยกเมล็ดซ้อนจาก Day 2 และเรียนรู้สี/ผิว/ขอบ 1px ของชุด Day 3 คละคลาส
  2. บีบช่วง HSV Augmentation ให้สมจริง (hsv_h=0.008, hsv_s=0.22, hsv_v=0.25) และปิด mixup=0.0
     เพื่อไม่ให้สีของ parchment / immature / good / husk เพี้ยนข้ามคลาส
  3. เพิ่มน้ำหนักคลาส (cls=1.2) + close_mosaic=20 เพื่อดันทั้ง Multi-Class Accuracy และ Mask IoU
  4. ประเมินผลบนชุด Validation แบบ Zero-Leakage (3 ภาพจริงที่แยกไว้ + 12 กองคละคลาสจาก 23 เม็ด Val เท่านั้น)
     พร้อมเทียบคะแนนก่อน-หลัง (Day 2 Zero-Shot vs Day 3 Fine-Tuned) ตามสูตรกรรมการสไลด์หน้า 12:
       - P1: Average Mask IoU (เต็ม 15.00)
       - P2: Counting Accuracy (เต็ม 15.00)
       - Multi-Class Accuracy & Confusion Matrix
  5. Export โมเดลครบชุดสำหรับ Raspberry Pi 5:
       weights/best.pt, weights/best.onnx, weights/best_ncnn_model/, weights/model_meta.npz
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
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

# =====================================================================
# 1. Hyperparameters & Path Setup (รองรับทั้ง Google Colab และ Local)
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
    DATASETS_SEG_DIR = BASE_DIR / "datasets_seg_day3"
    WEIGHTS_DIR = BASE_DIR / "weights"
    ANALYSIS_OUT = BASE_DIR / "training_analysis"
else:
    BASE_DIR = Path(__file__).resolve().parent
    DATASETS_SEG_DIR = BASE_DIR.parent / "Task" / "day3_prep" / "datasets_seg_day3"
    WEIGHTS_DIR = BASE_DIR / "weights"
    ANALYSIS_OUT = BASE_DIR.parent / "Task" / "day3_prep" / "training_analysis"

# Fine-Tuning Hyperparameters สำหรับ Day 3 Multi-Class Mixed Piles
EPOCHS = int(os.environ.get("DAY3_EPOCHS", "120"))
IMG_SIZE = int(os.environ.get("DAY3_IMGSZ", "640"))
BATCH_SIZE = int(os.environ.get("DAY3_BATCH", "8"))
LR0 = float(os.environ.get("DAY3_LR0", "5e-4"))
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
# 2. เตรียม Dataset & หา Pretrained Weights (Day 2 best.pt)
# =====================================================================

def ensure_dataset_ready() -> Tuple[Path, str]:
    """
    ตรวจสอบโฟลเดอร์ datasets_seg_day3 และคืนค่า (beans_seg_day3.yaml, base_weights_path)
    รองรับการอัปโหลดไฟล์ datasets_day3_colab.zip ไฟล์เดียวจบเมื่อรันบน Google Colab
    """
    global DATASETS_SEG_DIR

    if not (DATASETS_SEG_DIR / "images" / "train").exists():
        for cand in (
            BASE_DIR / "datasets_seg_day3",
            BASE_DIR.parent / "Task" / "day3_prep" / "datasets_seg_day3",
            Path("/content/datasets_seg_day3"),
        ):
            if (cand / "images" / "train").exists():
                DATASETS_SEG_DIR = cand
                break

    if not (DATASETS_SEG_DIR / "images" / "train").exists() and IN_COLAB:
        zip_candidates = (
            sorted(Path("/content").glob("*final*.zip"))
            or sorted(Path("/content").glob("*day3*.zip"))
            or sorted(Path("/content").glob("*.zip"))
        )
        if not zip_candidates:
            try:
                from google.colab import files  # type: ignore
                print("=" * 74)
                print("กรุณาอัปโหลดไฟล์ 'datasets_day3_final_colab.zip'")
                print("(จากโฟลเดอร์ track2/day-3/Task/day3_prep/datasets_day3_final_colab.zip)")
                print("=" * 74)
                uploaded = files.upload()
                zip_candidates = [Path("/content") / k for k in uploaded.keys() if k.endswith(".zip")]
            except Exception as e:
                print(f"[WARN] Colab upload skipped: {e}")

        if zip_candidates:
            zpath = zip_candidates[0]
            if not zipfile.is_zipfile(zpath):
                raise RuntimeError(
                    f"ไฟล์ {zpath} ยังอัปโหลดไม่สมบูรณ์หรือไม่ใช่ไฟล์ .zip — กรุณาลบไฟล์นี้แล้วรออัปโหลดให้ครบ 100% ก่อนรัน Cell 2"
                )
            print(f"[DATASET] Extracting {zpath} ({zpath.stat().st_size / (1024*1024):.2f} MB) ...")
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall("/content")
            if (Path("/content/datasets_seg_day3") / "images" / "train").exists():
                DATASETS_SEG_DIR = Path("/content/datasets_seg_day3")

    if not (DATASETS_SEG_DIR / "images" / "train").exists():
        raise FileNotFoundError(
            f"ไม่พบชุดข้อมูลสำหรับเทรนที่ {DATASETS_SEG_DIR / 'images' / 'train'}\n"
            f"กรุณารัน prepare_day3_dataset.py หรืออัปโหลด datasets_day3_colab.zip"
        )

    yaml_path = DATASETS_SEG_DIR / "beans_seg_day3.yaml"
    yaml_text = (
        f"path: {DATASETS_SEG_DIR.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
        + "\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    # ค้นหา Day 2 best.pt สำหรับ Warm-Start Fine-Tuning
    day2_candidates = [
        DATASETS_SEG_DIR / "pretrained" / "day2_best.pt",
        BASE_DIR.parent.parent / "day-2" / "team_01_day2" / "weights" / "best.pt",
        Path("/content/day2_best.pt"),
    ]
    base_weights = "yolo11n-seg.pt"
    for cand in day2_candidates:
        if cand.exists() and cand.stat().st_size > 1_000_000:
            base_weights = str(cand)
            break

    return yaml_path, base_weights


# =====================================================================
# 3. Subpixel-Aligned Postprocess & Official Day 3 Scoring (Slide 12)
# =====================================================================

def smooth_closed_contour(cnt: np.ndarray, sigma: float = 1.8) -> np.ndarray:
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


def decode_result_subpixel(
    result: Any,
    orig_h: int,
    orig_w: int,
    conf_thresh: float = 0.28,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """ถอดรหัส Prototype Masks แบบ Subpixel-Aligned (ตรงกับ infer.py บน Raspberry Pi 100%)"""
    seg_mask = np.zeros((orig_h, orig_w), dtype=np.uint16)
    objects: List[Dict[str, Any]] = []

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return seg_mask, objects

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks_data = result.masks.data.detach().cpu().numpy()
    mh, mw = masks_data.shape[1:]
    sx, sy = mw / float(orig_w), mh / float(orig_h)

    final_id = 1
    for idx in np.argsort(-confs):
        if confs[idx] < conf_thresh:
            continue
        bx1 = max(0, int(math.floor(boxes_xyxy[idx][0])))
        by1 = max(0, int(math.floor(boxes_xyxy[idx][1])))
        bx2 = min(orig_w, int(math.ceil(boxes_xyxy[idx][2])))
        by2 = min(orig_h, int(math.ceil(boxes_xyxy[idx][3])))
        roi_w, roi_h = bx2 - bx1, by2 - by1
        if roi_w < 8 or roi_h < 8:
            continue

        mx1 = max(0, int(math.floor(bx1 * sx)) - 2)
        my1 = max(0, int(math.floor(by1 * sy)) - 2)
        mx2 = min(mw, int(math.ceil(bx2 * sx)) + 3)
        my2 = min(mh, int(math.ceil(by2 * sy)) + 3)
        sub_m = masks_data[idx, my1:my2, mx1:mx2].astype(np.float32)
        if sub_m.size == 0:
            continue

        tx_local = (bx1 + 0.5) * sx - 0.5 - mx1
        ty_local = (by1 + 0.5) * sy - 0.5 - my1
        M = np.array([[sx, 0.0, tx_local], [0.0, sy, ty_local]], dtype=np.float32)
        up_m = cv2.warpAffine(
            sub_m,
            M,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mask_u8 = (up_m > 0.5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        best_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) < 36.0 or len(best_cnt) < 3:
            continue

        best_cnt = smooth_closed_contour(best_cnt, sigma=1.8)
        clean_sub = np.zeros_like(mask_u8)
        cv2.drawContours(clean_sub, [best_cnt], -1, 255, thickness=-1)

        roi_seg = seg_mask[by1:by2, bx1:bx2]
        avail = (clean_sub > 0) & (roi_seg == 0)
        if int(np.count_nonzero(avail)) < 36:
            continue

        avail_u8 = avail.astype(np.uint8) * 255
        a_cnts, _ = cv2.findContours(avail_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not a_cnts:
            continue
        a_cnt = max(a_cnts, key=cv2.contourArea)
        if cv2.contourArea(a_cnt) < 36.0 or len(a_cnt) < 3:
            continue

        single_u8 = np.zeros_like(avail_u8)
        cv2.drawContours(single_u8, [a_cnt], -1, 255, thickness=-1)
        roi_seg[single_u8 > 0] = final_id

        final_cnts, _ = cv2.findContours(single_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not final_cnts:
            continue
        final_cnt = max(final_cnts, key=cv2.contourArea)
        global_cnt = final_cnt.copy()
        global_cnt[:, 0, 0] += bx1
        global_cnt[:, 0, 1] += by1

        objects.append(
            {
                "id": final_id,
                "class": CLASSES[int(cls_ids[idx])],
                "polygon": [[int(pt[0][0]), int(pt[0][1])] for pt in global_cnt],
            }
        )
        final_id += 1

    return seg_mask, objects


def evaluate_day3_image(
    pred_mask: np.ndarray,
    pred_objs: List[Dict[str, Any]],
    gt_mask: np.ndarray,
    gt_objs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    คำนวณคะแนนตามสไลด์ Day 3 หน้า 12:
      - จับคู่ mask 1-to-1 ให้ IoU สูงสุด (เม็ดจริงที่ไม่มีคู่ & เม็ดที่ทายเกินได้ IoU = 0)
      - P1 per image = mean IoU across all matched + unmatched GT + extra Pred
      - P2 per image = matched_pairs / max(n_gt, n_pred)
      - Multi-Class Accuracy on matched pairs
    """
    n_gt, n_pred = len(gt_objs), len(pred_objs)
    if n_gt == 0 and n_pred == 0:
        return {"mean_iou": 1.0, "counting_acc": 1.0, "cls_correct": 0, "n_matched": 0, "n_gt": 0, "n_pred": 0, "matched_pairs": []}

    pairs: List[Tuple[float, int, int]] = []
    for gi, gobj in enumerate(gt_objs):
        g_bin = gt_mask == int(gobj["id"])
        g_area = int(np.count_nonzero(g_bin))
        if g_area == 0:
            continue
        uids, inters = np.unique(pred_mask[g_bin], return_counts=True)
        pid_to_idx = {int(po["id"]): pi for pi, po in enumerate(pred_objs)}
        for pid, inter in zip(uids, inters):
            if pid == 0 or int(pid) not in pid_to_idx:
                continue
            pi = pid_to_idx[int(pid)]
            p_area = int(np.count_nonzero(pred_mask == int(pid)))
            iou = float(inter) / float(max(g_area + p_area - int(inter), 1))
            if iou > 0.0:
                pairs.append((iou, pi, gi))

    pairs.sort(key=lambda x: x[0], reverse=True)
    used_p, used_g = set(), set()
    matched: List[Tuple[int, int, float]] = []
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched.append((pi, gi, iou))

    n_unmatched = (n_gt - len(used_g)) + (n_pred - len(used_p))
    denom = len(matched) + n_unmatched
    mean_iou = sum(x[2] for x in matched) / float(max(denom, 1))
    counting_acc = len(matched) / float(max(n_gt, n_pred, 1))
    cls_correct = sum(1 for pi, gi, _ in matched if pred_objs[pi]["class"] == gt_objs[gi]["class"])

    return {
        "mean_iou": mean_iou,
        "counting_acc": counting_acc,
        "cls_correct": cls_correct,
        "n_matched": len(matched),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "matched_pairs": matched,
    }


def evaluate_model_on_val_bundle(
    model: Any,
    val_img_dir: Path,
    val_gt_dir: Path,
    label: str = "Model",
    save_vis_panels: bool = False,
) -> Dict[str, Any]:
    gt_entries = json.loads((val_gt_dir / "ground_truth.json").read_text(encoding="utf-8"))
    gt_map = {e["image_name"]: e for e in gt_entries}

    per_img = []
    y_true, y_pred = [], []
    panels: List[np.ndarray] = []

    for entry in gt_entries:
        name = entry["image_name"]
        img_p = val_img_dir / name
        mask_p = val_gt_dir / "seg_mask" / f"{Path(name).stem}.png"
        if not img_p.exists() or not mask_p.exists():
            continue

        bgr = cv2.imread(str(img_p))
        oh, ow = bgr.shape[:2]
        gt_mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
        gt_objs = entry["objects"]

        t0 = time.perf_counter()
        res = model.predict(bgr, imgsz=IMG_SIZE, conf=0.28, iou=0.60, verbose=False)[0]
        pred_mask, pred_objs = decode_result_subpixel(res, oh, ow, conf_thresh=0.28)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        m = evaluate_day3_image(pred_mask, pred_objs, gt_mask, gt_objs)
        m["image_name"] = name
        m["subset"] = entry.get("subset", "val")
        m["infer_ms"] = infer_ms
        per_img.append(m)

        for pi, gi, _ in m["matched_pairs"]:
            y_true.append(CLASS_TO_IDX[gt_objs[gi]["class"]])
            y_pred.append(CLASS_TO_IDX[pred_objs[pi]["class"]])

        if save_vis_panels and len(panels) < 8:
            vis = bgr.copy()
            for obj in pred_objs:
                col = CLASS_COLORS_BGR.get(obj["class"], (0, 230, 0))
                pts = np.array(obj["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                vis[pred_mask == obj["id"]] = (0.55 * vis[pred_mask == obj["id"]] + 0.45 * np.array(col)).astype(np.uint8)
                cv2.polylines(vis, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
            side = max(oh, ow)
            sq = np.full((side, side, 3), 24, dtype=np.uint8)
            sq[(side - oh) // 2 : (side - oh) // 2 + oh, (side - ow) // 2 : (side - ow) // 2 + ow] = vis
            tile = cv2.resize(sq, (440, 440), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (440, 38), (18, 18, 22), -1)
            cls_pct = (m["cls_correct"] / max(m["n_gt"], 1)) * 100.0
            cv2.putText(
                tile,
                f"{Path(name).stem[:22]} | IoU:{m['mean_iou']:.2f} Cls:{cls_pct:.0f}%",
                (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 240, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(tile)

    mean_iou = float(np.mean([x["mean_iou"] for x in per_img])) if per_img else 0.0
    mean_cnt = float(np.mean([x["counting_acc"] for x in per_img])) if per_img else 0.0
    cls_acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0

    # แยกผล Real Val (3 ภาพจริง x2) vs Unseen Mixed Val Piles (12 ภาพ)
    real_sub = [x for x in per_img if x["subset"] == "real_val_day3"]
    pile_sub = [x for x in per_img if x["subset"] != "real_val_day3"]

    def _sub_summary(items: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        if not items:
            return (0.0, 0.0, 0.0)
        siou = float(np.mean([x["mean_iou"] for x in items]))
        scnt = float(np.mean([x["counting_acc"] for x in items]))
        scls = sum(x["cls_correct"] for x in items) / float(max(sum(x["n_gt"] for x in items), 1))
        return (siou, scnt, scls)

    r_iou, r_cnt, r_cls = _sub_summary(real_sub)
    p_iou, p_cnt, p_cls = _sub_summary(pile_sub)

    print(f"\n--- [{label}] Zero-Leakage Validation Benchmark ({len(per_img)} images) ---")
    print(f"  Real Held-Out Day-3 (6 views) : IoU={r_iou:.4f} | Count={r_cnt:.4f} | ClsAcc={r_cls*100:.2f}%")
    print(f"  Unseen Mixed Val Piles (12)   : IoU={p_iou:.4f} | Count={p_cnt:.4f} | ClsAcc={p_cls*100:.2f}%")
    print(f"  OVERALL Day-3 P1 (IoU / 15)   : {15.0 * mean_iou:.2f} / 15.00  (Mean IoU = {mean_iou:.4f})")
    print(f"  OVERALL Day-3 P2 (Count / 15) : {15.0 * mean_cnt:.2f} / 15.00  (Counting = {mean_cnt:.4f})")
    print(f"  OVERALL Multi-Class Accuracy  : {cls_acc * 100:.2f}% ({sum(1 for a,b in zip(y_true,y_pred) if a==b)}/{len(y_true)})")

    return {
        "mean_iou": mean_iou,
        "mean_cnt": mean_cnt,
        "p1_score": 15.0 * mean_iou,
        "p2_score": 15.0 * mean_cnt,
        "cls_acc": cls_acc,
        "real_val": {"iou": r_iou, "count": r_cnt, "cls_acc": r_cls},
        "pile_val": {"iou": p_iou, "count": p_cnt, "cls_acc": p_cls},
        "y_true": y_true,
        "y_pred": y_pred,
        "per_img": per_img,
        "panels": panels,
    }


# =====================================================================
# 4. Main Training & Multi-Runtime Export
# =====================================================================

def main() -> None:
    seed_everything(SEED)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    device_str = "0" if torch.cuda.is_available() else "cpu"
    print("=" * 76)
    print("TESA Top Gun Rally 2026 (Day 3) — Multi-Class YOLO11n-seg Fine-Tuning")
    print(f"Environment: {'Google Colab' if IN_COLAB else 'Local'} | Device: {device_str}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 76)

    yaml_path, base_weights = ensure_dataset_ready()
    print(f"[STEP 1/5] Dataset YAML : {yaml_path}")
    print(f"[STEP 1/5] Base Weights : {base_weights}")

    val_img_dir = DATASETS_SEG_DIR / "images" / "val"
    val_gt_dir = DATASETS_SEG_DIR / "val_gt_day3_format"

    # วัด Baseline ก่อน Fine-Tune ด้วย Day 2 best.pt (ถ้ามี) เพื่อเปรียบเทียบก่อน-หลังแบบชัดเจน
    baseline_res: Optional[Dict[str, Any]] = None
    if Path(base_weights).exists() and val_gt_dir.exists():
        print("\n[STEP 1.5/5] Evaluating BEFORE Fine-Tuning (Day-2 Baseline on Day-3 Zero-Leakage Val)...")
        baseline_res = evaluate_model_on_val_bundle(
            YOLO(base_weights), val_img_dir, val_gt_dir, label="BEFORE: Day-2 Baseline"
        )

    # 2. Warm-Start Fine-Tuning YOLO11n-seg สำหรับกองเมล็ดคละคลาส Day 3
    print(f"\n[STEP 2/5] Fine-Tuning YOLO11n-seg (epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={BATCH_SIZE}, lr0={LR0})...")
    runs_dir = BASE_DIR / "runs_day3"
    model = YOLO(base_weights)
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
        # เพิ่มน้ำหนัก Classification Loss สำหรับ 7 คลาสคละกัน + ลด HSV Distortion ไม่ให้สีคลาสเพี้ยน
        cls=1.3,
        hsv_h=0.008,
        hsv_s=0.20,
        hsv_v=0.24,
        degrees=180.0,
        scale=0.35,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.60,
        mixup=0.0,
        copy_paste=0.20,
        close_mosaic=20,
        overlap_mask=True,
        mask_ratio=4,
        project=str(runs_dir),
        name="yolo11n_seg_day3_mixed",
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

    # 3. Export โมเดลเป็น ONNX และ NCNN สำหรับ Raspberry Pi 5 (64-bit CPU, < 100 ms/image)
    print("\n[STEP 3/5] Exporting Best Day-3 Model to ONNX (FP32) & NCNN for Raspberry Pi 5...")
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

    meta_path = WEIGHTS_DIR / "model_meta.npz"
    np.savez_compressed(
        meta_path,
        classes=np.array(CLASSES),
        img_size=np.array([IMG_SIZE], dtype=np.int32),
    )

    # 4. ประเมินผลหลัง Fine-Tune บน Zero-Leakage Validation Benchmark (สไลด์หน้า 12)
    print("\n[STEP 4/5] Evaluating AFTER Fine-Tuning (Day-3 Multi-Class Model)...")
    final_res = evaluate_model_on_val_bundle(
        best_model, val_img_dir, val_gt_dir, label="AFTER: Day-3 Fine-Tuned", save_vis_panels=True
    )

    # 5. สร้างกราฟวิเคราะห์และ Confusion Matrix
    print("\n[STEP 5/5] Generating Day 3 Training Dashboards & Confusion Matrix...")
    results_csv = run_save_dir / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, ANALYSIS_OUT / "training_history.csv")
        rows = list(csv.DictReader(results_csv.read_text(encoding="utf-8").splitlines()))
        ep_x = [int(float(r.get("epoch", i + 1))) for i, r in enumerate(rows)]

        def _col(substr: str) -> List[float]:
            for k in rows[0].keys():
                if substr in k.strip():
                    return [float(r[k]) for r in rows]
            return [0.0] * len(rows)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=130)
        axes[0, 0].plot(ep_x, _col("train/seg_loss"), label="Train Seg Loss", lw=2)
        axes[0, 0].plot(ep_x, _col("val/seg_loss"), label="Val Seg Loss", lw=2)
        axes[0, 0].set_title("Instance Mask Segmentation Loss", fontweight="bold")
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(ep_x, _col("train/cls_loss"), label="Train Cls Loss", lw=2)
        axes[0, 1].plot(ep_x, _col("val/cls_loss"), label="Val Cls Loss", lw=2)
        axes[0, 1].set_title("Multi-Class Classification Loss", fontweight="bold")
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        m50 = _col("metrics/mAP50(M)")
        m95 = _col("metrics/mAP50-95(M)")
        axes[1, 0].plot(ep_x, [v * 100 for v in m50], label="Mask mAP50 (%)", lw=2.2, color="#16a34a")
        axes[1, 0].plot(ep_x, [v * 100 for v in m95], label="Mask mAP50-95 (%)", lw=2.2, color="#2563eb")
        axes[1, 0].set_title(f"Mask mAP Progression (Best mAP50: {max(m50)*100:.2f}%)", fontweight="bold")
        axes[1, 0].set_ylim(0, 103); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        # กราฟเทียบ Before vs After (Day 2 Zero-Shot vs Day 3 Fine-Tuned)
        cats = ["Mask IoU (%)", "Counting (%)", "Multi-Class Acc (%)"]
        after_vals = [final_res["mean_iou"] * 100, final_res["mean_cnt"] * 100, final_res["cls_acc"] * 100]
        x_p = np.arange(len(cats))
        if baseline_res is not None:
            before_vals = [baseline_res["mean_iou"] * 100, baseline_res["mean_cnt"] * 100, baseline_res["cls_acc"] * 100]
            axes[1, 1].bar(x_p - 0.18, before_vals, width=0.34, label="Before (Day-2 Baseline)", color="#94a3b8")
            axes[1, 1].bar(x_p + 0.18, after_vals, width=0.34, label="After (Day-3 Fine-Tuned)", color="#10b981")
        else:
            axes[1, 1].bar(x_p, after_vals, width=0.45, label="Day-3 Fine-Tuned", color="#10b981")
        axes[1, 1].set_xticks(x_p); axes[1, 1].set_xticklabels(cats)
        axes[1, 1].set_ylim(0, 108)
        axes[1, 1].set_title(f"Day 3 Val Score: P1+P2 = {final_res['p1_score'] + final_res['p2_score']:.2f} / 30.00", fontweight="bold")
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(ANALYSIS_OUT / "training_curves_dashboard.png")
        if IN_COLAB:
            plt.show()
        plt.close(fig)

    # Confusion Matrix
    if final_res["y_true"]:
        cm = confusion_matrix(final_res["y_true"], final_res["y_pred"], labels=list(range(len(CLASSES))))
        cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig_cm, axes_cm = plt.subplots(1, 2, figsize=(13, 5.5), dpi=125)
        for ax, mat, title, fmt in [
            (axes_cm[0], cm, "Day-3 Zero-Leakage Val Confusion Matrix (Counts)", "d"),
            (axes_cm[1], cm_norm * 100, "Normalized Confusion Matrix (%)", ".1f"),
        ]:
            ax.imshow(mat, cmap="Blues")
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

    # ภาพตัวอย่างผลการทำนายบน Val
    if final_res["panels"]:
        panels = final_res["panels"]
        while len(panels) < 8:
            panels.append(np.zeros_like(panels[0]))
        grid = np.vstack([np.hstack(panels[:4]), np.hstack(panels[4:8])])
        cv2.imwrite(str(ANALYSIS_OUT / "day3_val_predictions_grid.png"), grid)

    summary = {
        "base_weights": Path(base_weights).name,
        "img_size": IMG_SIZE,
        "epochs_configured": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr0": LR0,
        "train_minutes": round(train_minutes, 2),
        "before_day2_baseline": {
            "mean_iou": round(baseline_res["mean_iou"], 4),
            "counting_acc": round(baseline_res["mean_cnt"], 4),
            "cls_acc": round(baseline_res["cls_acc"], 4),
        } if baseline_res else None,
        "after_day3_finetuned": {
            "mean_iou": round(final_res["mean_iou"], 4),
            "counting_acc": round(final_res["mean_cnt"], 4),
            "p1_score_out_of_15": round(final_res["p1_score"], 2),
            "p2_score_out_of_15": round(final_res["p2_score"], 2),
            "cls_acc": round(final_res["cls_acc"], 4),
            "real_val": {k: round(v, 4) for k, v in final_res["real_val"].items()},
            "pile_val": {k: round(v, 4) for k, v in final_res["pile_val"].items()},
        },
        "exported_weights": [p.name for p in sorted(WEIGHTS_DIR.iterdir()) if p.name != ".gitkeep"],
    }
    (ANALYSIS_OUT / "day3_training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[SUMMARY]", json.dumps(summary, indent=2, ensure_ascii=False))

    # หากอยู่บน Google Colab ให้แพ็ก weights/ + training_analysis/ ดาวน์โหลดทันที
    if IN_COLAB:
        info_p = DATASETS_SEG_DIR / "dataset_info.json"
        ds_mode = json.loads(info_p.read_text(encoding="utf-8")).get("mode", "final") if info_p.exists() else "final"
        zip_name = "day3_final_weights_and_analysis.zip" if ds_mode == "full" else "day3_trained_weights_and_analysis.zip"
        bundle_zip = Path("/content") / zip_name
        with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in WEIGHTS_DIR.rglob("*"):
                if f.is_file():
                    zf.write(f, Path("weights") / f.relative_to(WEIGHTS_DIR))
            for f in ANALYSIS_OUT.rglob("*"):
                if f.is_file():
                    zf.write(f, Path("training_analysis") / f.relative_to(ANALYSIS_OUT))
        print(f"\n[COLAB BUNDLE] Created {bundle_zip} ({bundle_zip.stat().st_size / (1024*1024):.2f} MB)")
        try:
            from google.colab import files  # type: ignore
            files.download(str(bundle_zip))
        except Exception:
            pass


if __name__ == "__main__":
    main()
