#!/usr/bin/env python3
"""
evaluate_stress_test.py — ตัวรันทดสอบและผ่าจุดอ่อนโมเดล Day 3 บนชุดข้อมูลแบบโหด
ใช้ไปป์ไลน์จริงจาก team_01_day3/infer.py (load_model + predict_image + detect_active_pile_roi)

คำนวณคะแนนตามเกณฑ์ Day 3 สไลด์หน้า 12:
  - P1: Segmentation IoU (/ 15)
  - P2: Counting Accuracy (/ 15) + TP / FP (Ghost) / FN (Missed)
  - Multi-Class Accuracy + 7x7 Confusion Matrix + Top Confusing Pairs
  - MQTT Defect Ratio MAE
  - ROI Clipping Check (ตรวจว่า detect_active_pile_roi() ตัดเมล็ดจริงทิ้งหรือไม่!)

พร้อมสร้างภาพวินิจฉัยผลลัพธ์ใน results/:
  - category_dashboard.png      : กราฟคะแนนแยกตาม 7 หมวดสุดโหด + Confusion Matrix 7x7
  - worst_cases_diagnostic.jpg  : รวมภาพ 6 เคสที่โมเดลพลาดหนักที่สุด พร้อมวงจุด MISSED / GHOST FP / WRONG CLASS
  - stress_report.json          : รายงานตัวเลขละเอียดทั้งหมด (ใช้เทียบก่อน/หลังเทรนเพิ่มได้)

วิธีใช้งาน:
  python evaluate_stress_test.py
  python evaluate_stress_test.py --weights ../../team_01_day3/weights --conf 0.28
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
DAY3_DIR = TASK_DIR.parent
TEAM_DIR = DAY3_DIR / "team_01_day3"

if str(TEAM_DIR) not in sys.path:
    sys.path.insert(0, str(TEAM_DIR))

from infer import (  # noqa: E402
    CLASSES,
    CLASS_COLORS_BGR,
    build_mqtt_payload,
    detect_active_pile_roi,
    load_image_exif_corrected,
    load_model,
    predict_image,
)

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_WEIGHTS_DIR = TEAM_DIR / "weights"


def compute_defect_ratio(objs: List[Dict[str, Any]]) -> float:
    total = len(objs)
    if total == 0:
        return 0.0
    good = sum(1 for o in objs if o["class"] == "good")
    return (total - good) / float(total)


def evaluate_image_detailed(
    pred_objs: List[Dict[str, Any]],
    gt_objs: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    roi_box: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    """คำนวณ IoU (P1/15), Counting (P2/15), Matched Class Accuracy, Missed, Ghost FP และ ROI-clipped beans"""
    n_gt, n_pr = len(gt_objs), len(pred_objs)

    # Check if any GT bean center/area falls outside detect_active_pile_roi
    rx1, ry1, rx2, ry2 = roi_box
    roi_clipped_gt_ids: List[int] = []
    for i, o in enumerate(gt_objs, start=1):
        pts = np.array(o["polygon"], dtype=np.int32)
        bx1, by1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        bx2, by2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        # Compute fraction of bbox inside ROI
        ix1, iy1 = max(bx1, rx1), max(by1, ry1)
        ix2, iy2 = min(bx2, rx2), min(by2, ry2)
        inter_a = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        box_a = max(1, (bx2 - bx1) * (by2 - by1))
        if (inter_a / float(box_a)) < 0.75:
            roi_clipped_gt_ids.append(i)

    if n_gt == 0 and n_pr == 0:
        return {
            "n_gt": 0,
            "n_pr": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "cls_ok": 0,
            "mean_iou": 1.0,
            "counting": 1.0,
            "p1": 15.0,
            "p2": 15.0,
            "matched_pairs": [],
            "missed_gt_ids": [],
            "ghost_pr_ids": [],
            "roi_clipped_gt_ids": [],
            "defect_ratio_err": 0.0,
        }

    # Build uint16 masks (downsample slightly only if giant 3072x4096 -> 0.5x for fast exact evaluation)
    eval_scale = 0.5 if max(img_w, img_h) > 2200 else 1.0
    ew = max(1, int(round(img_w * eval_scale)))
    eh = max(1, int(round(img_h * eval_scale)))

    gt_mask = np.zeros((eh, ew), dtype=np.uint16)
    pr_mask = np.zeros((eh, ew), dtype=np.uint16)
    gt_cls: Dict[int, str] = {}
    pr_cls: Dict[int, str] = {}

    for i, o in enumerate(gt_objs, start=1):
        pts = np.round(np.array(o["polygon"], dtype=np.float32) * eval_scale).astype(np.int32).reshape((-1, 1, 2))
        cv2.drawContours(gt_mask, [pts], -1, i, thickness=-1)
        gt_cls[i] = o["class"]
    for i, o in enumerate(pred_objs, start=1):
        pts = np.round(np.array(o["polygon"], dtype=np.float32) * eval_scale).astype(np.int32).reshape((-1, 1, 2))
        cv2.drawContours(pr_mask, [pts], -1, i, thickness=-1)
        pr_cls[i] = o["class"]

    pairs: List[Tuple[float, int, int]] = []
    for gid in gt_cls:
        gb = gt_mask == gid
        ga = int(np.count_nonzero(gb))
        if ga == 0:
            continue
        uids, inters = np.unique(pr_mask[gb], return_counts=True)
        for pid, inter in zip(uids, inters):
            if pid == 0:
                continue
            pa = int(np.count_nonzero(pr_mask == int(pid)))
            iou = float(inter) / float(max(ga + pa - int(inter), 1))
            if iou > 0.0:
                pairs.append((iou, gid, int(pid)))

    pairs.sort(reverse=True)
    ug, up = set(), set()
    matched_pairs: List[Dict[str, Any]] = []
    matched_ious: List[float] = []
    cls_ok = 0

    for iou, gid, pid in pairs:
        if gid in ug or pid in up:
            continue
        if iou < 0.15:
            continue
        ug.add(gid)
        up.add(pid)
        matched_ious.append(iou)
        ok = gt_cls[gid] == pr_cls[pid]
        if ok:
            cls_ok += 1
        matched_pairs.append({"gid": gid, "pid": pid, "gt_cls": gt_cls[gid], "pr_cls": pr_cls[pid], "iou": round(iou, 4), "cls_ok": ok})

    missed_gt_ids = [gid for gid in gt_cls if gid not in ug]
    ghost_pr_ids = [pid for pid in pr_cls if pid not in up]
    unmatched = len(missed_gt_ids) + len(ghost_pr_ids)

    avg_iou = sum(matched_ious) / float(max(len(matched_ious) + unmatched, 1))
    counting = len(ug) / float(max(n_gt, n_pr, 1))

    gt_dr = compute_defect_ratio(gt_objs)
    pr_dr = compute_defect_ratio(pred_objs)

    return {
        "n_gt": n_gt,
        "n_pr": n_pr,
        "tp": len(ug),
        "fp": len(ghost_pr_ids),
        "fn": len(missed_gt_ids),
        "cls_ok": cls_ok,
        "mean_iou": avg_iou,
        "counting": counting,
        "p1": 15.0 * avg_iou,
        "p2": 15.0 * counting,
        "matched_pairs": matched_pairs,
        "missed_gt_ids": missed_gt_ids,
        "ghost_pr_ids": ghost_pr_ids,
        "roi_clipped_gt_ids": roi_clipped_gt_ids,
        "defect_ratio_err": abs(gt_dr - pr_dr),
    }


# =====================================================================
# Visual Diagnostics Renderers (Pure OpenCV, Zero Extra Dependencies)
# =====================================================================

def render_diagnostic_tile(
    img_bgr: np.ndarray,
    gt_objs: List[Dict[str, Any]],
    pred_objs: List[Dict[str, Any]],
    eval_res: Dict[str, Any],
    roi_box: Tuple[int, int, int, int],
    title: str,
    subtitle: str,
    tile_size: int = 560,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(tile_size / float(w), (tile_size - 64) / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    vis = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    overlay = vis.copy()

    matched_by_pid = {m["pid"]: m for m in eval_res["matched_pairs"]}
    ghost_pids = set(eval_res["ghost_pr_ids"])
    missed_gids = set(eval_res["missed_gt_ids"])

    # 1. Draw ROI crop rectangle if cropped
    rx1, ry1, rx2, ry2 = roi_box
    if (rx1, ry1, rx2, ry2) != (0, 0, w, h):
        cv2.rectangle(
            vis,
            (int(round(rx1 * scale)), int(round(ry1 * scale))),
            (int(round(rx2 * scale)), int(round(ry2 * scale))),
            (255, 230, 0),
            2,
            cv2.LINE_AA,
        )

    # 2. Draw predicted masks & polygons
    for pid, obj in enumerate(pred_objs, start=1):
        pts = np.round(np.array(obj["polygon"], dtype=np.float32) * scale).astype(np.int32).reshape((-1, 1, 2))
        if pid in ghost_pids:
            # Ghost False Positive -> MAGENTA
            cv2.drawContours(overlay, [pts], -1, (220, 40, 240), -1)
            cv2.polylines(vis, [pts], True, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            m = matched_by_pid[pid]
            if m["cls_ok"]:
                col = CLASS_COLORS_BGR.get(obj["class"], (0, 220, 0))
                cv2.drawContours(overlay, [pts], -1, col, -1)
                cv2.polylines(vis, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Wrong Class -> BRIGHT YELLOW border
                cv2.drawContours(overlay, [pts], -1, (0, 220, 255), -1)
                cv2.polylines(vis, [pts], True, (0, 240, 255), 2, cv2.LINE_AA)

    vis = cv2.addWeighted(overlay, 0.36, vis, 0.64, 0.0)

    # 3. Draw MISSED GT beans -> BRIGHT RED thick outline + X marker
    for gid, obj in enumerate(gt_objs, start=1):
        if gid not in missed_gids:
            continue
        pts = np.round(np.array(obj["polygon"], dtype=np.float32) * scale).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (30, 30, 255), 3, cv2.LINE_AA)
        cx = int(np.mean(pts[:, 0, 0]))
        cy = int(np.mean(pts[:, 0, 1]))
        cv2.putText(vis, f"MISS:{obj['class'][:4]}", (max(4, cx - 32), max(14, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"MISS:{obj['class'][:4]}", (max(4, cx - 32), max(14, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 120, 255), 1, cv2.LINE_AA)

    # 4. Annotate WRONG CLASS and GHOST FP labels
    for pid, obj in enumerate(pred_objs, start=1):
        pts = np.round(np.array(obj["polygon"], dtype=np.float32) * scale).astype(np.int32)
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        if pid in ghost_pids:
            txt = f"FP:{obj['class'][:4]}"
            cv2.putText(vis, txt, (max(4, cx - 26), max(14, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (max(4, cx - 26), max(14, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 110, 255), 1, cv2.LINE_AA)
        elif not matched_by_pid[pid]["cls_ok"]:
            m = matched_by_pid[pid]
            txt = f"{m['pr_cls'][:4]}!={m['gt_cls'][:4]}"
            cv2.putText(vis, txt, (max(4, cx - 36), max(14, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (max(4, cx - 36), max(14, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 245, 255), 1, cv2.LINE_AA)

    # Compose framed card with header
    card = np.full((tile_size, tile_size, 3), (22, 24, 30), dtype=np.uint8)
    ox = (tile_size - nw) // 2
    oy = 62 + ((tile_size - 64 - nh) // 2)
    card[oy : oy + nh, ox : ox + nw] = vis
    cv2.rectangle(card, (0, 0), (tile_size, 58), (18, 20, 26), -1)
    cv2.line(card, (0, 58), (tile_size, 58), (60, 65, 78), 1)
    cv2.putText(card, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 255), 1, cv2.LINE_AA)
    cv2.putText(card, subtitle, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130, 225, 255), 1, cv2.LINE_AA)
    return card


def render_category_dashboard(
    cat_summary: List[Dict[str, Any]],
    cm: np.ndarray,
    overall: Dict[str, Any],
    model_label: str,
    out_path: Path,
) -> None:
    W, H = 1540, 760
    img = np.full((H, W, 3), (20, 23, 30), dtype=np.uint8)

    # Header banner
    cv2.rectangle(img, (0, 0), (W, 82), (28, 33, 44), -1)
    cv2.line(img, (0, 82), (W, 82), (70, 80, 100), 2)
    cv2.putText(
        img,
        "TESA Day 3 — Brutal Adversarial Stress-Test Dashboard",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (245, 248, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Model: {model_label}  |  Overall P1(IoU)={overall['p1']:.2f}/15 ({overall['mean_iou']*100:.1f}%)  |  "
        f"P2(Count)={overall['p2']:.2f}/15 ({overall['counting']*100:.1f}%)  |  "
        f"ClsAcc={overall['cls_acc']*100:.1f}% ({overall['cls_ok']}/{overall['tp']})  |  "
        f"FP={overall['fp']} FN={overall['fn']}  |  MedLat={overall['med_ms']:.0f}ms",
        (24, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (120, 230, 255),
        1,
        cv2.LINE_AA,
    )

    # Left Panel: Grouped Horizontal Bar Chart by Category
    lx, ly = 24, 118
    cv2.putText(img, "Performance by Brutal Torture Category (%)", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (230, 235, 245), 2, cv2.LINE_AA)
    # Legend
    legend_items = [("Mask IoU %", (245, 165, 55)), ("Counting %", (90, 215, 115)), ("Class Acc %", (75, 185, 255))]
    leg_x = lx + 470
    for lbl, col in legend_items:
        cv2.rectangle(img, (leg_x, ly - 13), (leg_x + 14, ly + 1), col, -1)
        cv2.putText(img, lbl, (leg_x + 20, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (215, 220, 230), 1, cv2.LINE_AA)
        leg_x += 126

    bar_x0 = 240
    bar_max_w = 540
    row_h = 80
    for idx, row in enumerate(cat_summary):
        y0 = 148 + idx * row_h
        cat_label = row["category"]
        cv2.putText(img, cat_label, (24, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 240, 250), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"FP={row['fp']} FN={row['fn']} ({row['n_imgs']} imgs)",
            (24, y0 + 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (150, 162, 185),
            1,
            cv2.LINE_AA,
        )
        metrics = [
            (row["mean_iou"] * 100.0, (245, 165, 55)),
            (row["counting"] * 100.0, (90, 215, 115)),
            (row["cls_acc"] * 100.0, (75, 185, 255)),
        ]
        for mi, (val, col) in enumerate(metrics):
            by = y0 + mi * 19
            cv2.rectangle(img, (bar_x0, by), (bar_x0 + bar_max_w, by + 13), (38, 42, 54), -1)
            bw = max(2, int(round(bar_max_w * min(100.0, max(0.0, val)) / 100.0)))
            cv2.rectangle(img, (bar_x0, by), (bar_x0 + bw, by + 13), col, -1)
            cv2.putText(img, f"{val:5.1f}%", (bar_x0 + bar_max_w + 10, by + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    # Vertical divider
    cv2.line(img, (875, 96), (875, H - 24), (55, 62, 78), 1)

    # Right Panel: 7x7 Confusion Matrix Heatmap (Matched Objects)
    rx, ry = 905, 118
    cv2.putText(img, "7x7 Confusion Matrix (Rows=GT, Cols=Pred)", (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (230, 235, 245), 2, cv2.LINE_AA)
    cell = 66
    gx0, gy0 = rx + 110, ry + 55

    for ci, cname in enumerate(CLASSES):
        # Col header (Pred)
        cv2.putText(img, cname[:6], (gx0 + ci * cell + 6, gy0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (175, 210, 255), 1, cv2.LINE_AA)
        # Row header (GT)
        cv2.putText(img, cname[:9], (rx + 4, gy0 + ci * cell + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 225, 235), 1, cv2.LINE_AA)

    for r in range(len(CLASSES)):
        row_sum = max(1, int(cm[r].sum()))
        for c in range(len(CLASSES)):
            cnt = int(cm[r, c])
            frac = cnt / float(row_sum)
            x1, y1 = gx0 + c * cell, gy0 + r * cell
            if r == c:
                # Diagonal: Green intensity
                bg = (int(28 + 40 * frac), int(45 + 150 * frac), int(35 + 45 * frac))
            elif cnt > 0:
                # Off-diagonal error: Red/Orange highlight
                bg = (int(38 + 30 * frac), int(35 + 75 * frac), int(75 + 165 * min(1.0, frac * 2.2)))
            else:
                bg = (30, 34, 44)
            cv2.rectangle(img, (x1 + 2, y1 + 2), (x1 + cell - 2, y1 + cell - 2), bg, -1)
            cv2.rectangle(img, (x1 + 2, y1 + 2), (x1 + cell - 2, y1 + cell - 2), (55, 62, 78), 1)
            txt_col = (255, 255, 255) if cnt > 0 else (95, 102, 118)
            cv2.putText(img, str(cnt), (x1 + 18, y1 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.56, txt_col, 2 if cnt > 0 else 1, cv2.LINE_AA)
            if cnt > 0:
                cv2.putText(img, f"{frac*100:4.0f}%", (x1 + 12, y1 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 228, 240), 1, cv2.LINE_AA)

    # Legend note at bottom right
    cv2.putText(
        img,
        "Green diagonal = Correct class | Red off-diagonal = Class confusion",
        (rx + 6, gy0 + len(CLASSES) * cell + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (165, 175, 195),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Day 3 Coffee Bean Model on Brutal Adversarial Stress-Test Suite")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR, help="Path to stress_test dataset folder")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_DIR, help="Path to model weights folder or file")
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR, help="Output folder for reports & visuals")
    parser.add_argument("--conf", type=float, default=0.28, help="Confidence threshold (default: 0.28 matching infer.py)")
    args = parser.parse_args()

    labels_path = args.dataset / "labels.json"
    manifest_path = args.dataset / "stress_manifest.json"
    if not labels_path.exists():
        print(f"[ERROR] ไม่พบ {labels_path} — กรุณารัน generate_brutal_stress_data.py ก่อน", file=sys.stderr)
        sys.exit(1)

    gt_entries = json.loads(labels_path.read_text(encoding="utf-8"))
    manifest_by_name: Dict[str, Dict[str, Any]] = {}
    if manifest_path.exists():
        for m in json.loads(manifest_path.read_text(encoding="utf-8")):
            manifest_by_name[m["image_name"]] = m

    args.out.mkdir(parents=True, exist_ok=True)
    per_vis_dir = args.out / "per_image_vis"
    per_vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("TESA Day 3 — Brutal Adversarial Stress-Test Evaluator")
    print("=" * 84)
    t_load = time.perf_counter()
    engine = load_model(args.weights)
    model_label = engine.get("engine_label", engine["backend"])
    print(f"[MODEL] {model_label} | imgsz={engine['img_size']} | conf={args.conf} | load={(time.perf_counter()-t_load)*1000:.1f} ms")
    print(f"[DATA]  {len(gt_entries)} stress-test images from {args.dataset}\n")

    records: List[Dict[str, Any]] = []
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int32)
    pair_errors = Counter()
    tiles_cache: Dict[str, Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Tuple[int, int, int, int]]] = {}

    for idx, entry in enumerate(gt_entries, start=1):
        img_name = entry["image_name"]
        gt_objs = entry["objects"]
        meta = manifest_by_name.get(img_name, {"category": "unknown", "sub_case": img_name, "difficulty": "HARD"})
        img_path = args.dataset / "Images" / img_name
        img_bgr = load_image_exif_corrected(img_path)
        ih, iw = img_bgr.shape[:2]

        roi_box = detect_active_pile_roi(img_bgr)
        t0 = time.perf_counter()
        _, pred_objs = predict_image(engine, img_bgr, conf_thresh=args.conf)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        ev = evaluate_image_detailed(pred_objs, gt_objs, iw, ih, roi_box)
        for m in ev["matched_pairs"]:
            gi, pi = CLASS_TO_IDX[m["gt_cls"]], CLASS_TO_IDX[m["pr_cls"]]
            cm[gi, pi] += 1
            if not m["cls_ok"]:
                pair_errors[(m["gt_cls"], m["pr_cls"])] += 1

        cls_acc = (ev["cls_ok"] / float(ev["tp"])) if ev["tp"] > 0 else (1.0 if ev["n_gt"] == 0 and ev["n_pr"] == 0 else 0.0)
        # Composite score for ranking worst failure cases
        composite = 0.35 * ev["mean_iou"] + 0.35 * ev["counting"] + 0.30 * cls_acc
        rec = {
            "image_name": img_name,
            "category": meta["category"],
            "sub_case": meta["sub_case"],
            "resolution": f"{iw}x{ih}",
            "roi_cropped": roi_box != (0, 0, iw, ih),
            "roi_clipped_beans": len(ev["roi_clipped_gt_ids"]),
            "n_gt": ev["n_gt"],
            "n_pr": ev["n_pr"],
            "tp": ev["tp"],
            "fp": ev["fp"],
            "fn": ev["fn"],
            "cls_ok": ev["cls_ok"],
            "cls_acc": round(cls_acc, 4),
            "mean_iou": round(ev["mean_iou"], 4),
            "counting": round(ev["counting"], 4),
            "p1": round(ev["p1"], 2),
            "p2": round(ev["p2"], 2),
            "defect_ratio_err": round(ev["defect_ratio_err"], 4),
            "infer_ms": round(infer_ms, 1),
            "composite": round(composite, 4),
        }
        records.append(rec)
        tiles_cache[img_name] = (img_bgr, gt_objs, pred_objs, ev, roi_box)

        roi_warn = f" [ROI-CLIP:{len(ev['roi_clipped_gt_ids'])}!]" if ev["roi_clipped_gt_ids"] else ""
        print(
            f"  [{idx:02d}/{len(gt_entries):02d}] {img_name:<38} | P1={ev['p1']:5.2f} P2={ev['p2']:5.2f} "
            f"| Cls={ev['cls_ok']:2d}/{ev['tp']:2d} ({cls_acc*100:5.1f}%) | FP={ev['fp']:2d} FN={ev['fn']:2d} "
            f"| {infer_ms:5.0f}ms{roi_warn}"
        )

    # Category Summary
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    cat_summary: List[Dict[str, Any]] = []
    print("\n" + "=" * 96)
    print(f"{'CATEGORY':<24} | {'IMGS':>4} | {'P1(IoU/15)':>10} | {'P2(Cnt/15)':>10} | {'ClsAcc':>8} | {'FP':>3} | {'FN':>3} | {'ROI-CLIP':>8} | {'MedMs':>6}")
    print("-" * 96)
    for cat in sorted(by_cat.keys()):
        rows = by_cat[cat]
        m_iou = float(np.mean([r["mean_iou"] for r in rows]))
        m_cnt = float(np.mean([r["counting"] for r in rows]))
        tot_tp = sum(r["tp"] for r in rows)
        tot_ok = sum(r["cls_ok"] for r in rows)
        c_acc = (tot_ok / float(tot_tp)) if tot_tp > 0 else 1.0
        tot_fp = sum(r["fp"] for r in rows)
        tot_fn = sum(r["fn"] for r in rows)
        tot_clip = sum(r["roi_clipped_beans"] for r in rows)
        med_ms = float(np.median([r["infer_ms"] for r in rows]))
        entry = {
            "category": cat,
            "n_imgs": len(rows),
            "mean_iou": m_iou,
            "counting": m_cnt,
            "p1": 15.0 * m_iou,
            "p2": 15.0 * m_cnt,
            "cls_acc": c_acc,
            "tp": tot_tp,
            "cls_ok": tot_ok,
            "fp": tot_fp,
            "fn": tot_fn,
            "roi_clipped": tot_clip,
            "med_ms": med_ms,
        }
        cat_summary.append(entry)
        print(
            f"{cat:<24} | {len(rows):>4} | {entry['p1']:5.2f} ({m_iou*100:4.1f}%) | "
            f"{entry['p2']:5.2f} ({m_cnt*100:4.1f}%) | {c_acc*100:6.1f}%  | {tot_fp:>3} | {tot_fn:>3} | {tot_clip:>8} | {med_ms:5.0f}"
        )

    tot_tp = sum(r["tp"] for r in records)
    tot_ok = sum(r["cls_ok"] for r in records)
    tot_fp = sum(r["fp"] for r in records)
    tot_fn = sum(r["fn"] for r in records)
    tot_gt = sum(r["n_gt"] for r in records)
    overall = {
        "n_imgs": len(records),
        "total_gt_beans": tot_gt,
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "cls_ok": tot_ok,
        "mean_iou": float(np.mean([r["mean_iou"] for r in records])),
        "counting": float(np.mean([r["counting"] for r in records])),
        "p1": 15.0 * float(np.mean([r["mean_iou"] for r in records])),
        "p2": 15.0 * float(np.mean([r["counting"] for r in records])),
        "cls_acc": (tot_ok / float(tot_tp)) if tot_tp > 0 else 1.0,
        "defect_ratio_mae": float(np.mean([r["defect_ratio_err"] for r in records])),
        "med_ms": float(np.median([r["infer_ms"] for r in records])),
        "p95_ms": float(np.percentile([r["infer_ms"] for r in records], 95)),
        "max_ms": float(np.max([r["infer_ms"] for r in records])),
        "roi_clipped_total": sum(r["roi_clipped_beans"] for r in records),
    }
    print("-" * 96)
    print(
        f"{'OVERALL BRUTAL SUITE':<24} | {overall['n_imgs']:>4} | {overall['p1']:5.2f} ({overall['mean_iou']*100:4.1f}%) | "
        f"{overall['p2']:5.2f} ({overall['counting']*100:4.1f}%) | {overall['cls_acc']*100:6.1f}%  | "
        f"{tot_fp:>3} | {tot_fn:>3} | {overall['roi_clipped_total']:>8} | {overall['med_ms']:5.0f}"
    )
    print("=" * 96)
    print(
        f"  MQTT Defect-Ratio MAE : {overall['defect_ratio_mae']:.4f} | "
        f"Latency: median={overall['med_ms']:.1f}ms, p95={overall['p95_ms']:.1f}ms, max={overall['max_ms']:.1f}ms"
    )

    # Per-Class Precision / Recall / F1
    print("\n--- Per-Class Breakdown (Matched Objects) ---")
    per_class_stats: Dict[str, Dict[str, Any]] = {}
    for ci, cname in enumerate(CLASSES):
        tp_c = int(cm[ci, ci])
        row_s = int(cm[ci, :].sum())
        col_s = int(cm[:, ci].sum())
        rec_c = tp_c / float(max(row_s, 1))
        prec_c = tp_c / float(max(col_s, 1))
        f1_c = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0
        per_class_stats[cname] = {"tp": tp_c, "gt_matched": row_s, "pred_matched": col_s, "prec": round(prec_c, 4), "rec": round(rec_c, 4), "f1": round(f1_c, 4)}
        print(f"  {cname:<10} : Recall={rec_c*100:5.1f}% ({tp_c:2d}/{row_s:2d}) | Prec={prec_c*100:5.1f}% ({tp_c:2d}/{col_s:2d}) | F1={f1_c*100:5.1f}%")

    print("\n--- Top Confusing Class Pairs (GT -> Pred) ---")
    for (g, p), cnt in pair_errors.most_common(8):
        print(f"  {g:<10} -> {p:<10} : {cnt} beans")

    # Worst 8 images
    worst = sorted(records, key=lambda r: (r["composite"], r["mean_iou"]))[:8]
    print("\n--- Top 8 Worst-Failing Brutal Cases ---")
    for r in worst:
        clip_s = f" | ROI-CLIP={r['roi_clipped_beans']}" if r["roi_clipped_beans"] else ""
        print(
            f"  * {r['image_name']:<38} | P1={r['p1']:5.2f} P2={r['p2']:5.2f} "
            f"| Cls={r['cls_acc']*100:5.1f}% | GT={r['n_gt']:2d} PR={r['n_pr']:2d} (FP={r['fp']}, FN={r['fn']}){clip_s}"
        )

    # Save visual diagnostics: Dashboard + 6 Worst Cases Grid
    dash_path = args.out / "category_dashboard.png"
    render_category_dashboard(cat_summary, cm, overall, model_label, dash_path)

    worst6 = worst[:6]
    tiles: List[np.ndarray] = []
    for r in worst6:
        img_bgr, gt_objs, pred_objs, ev, roi_box = tiles_cache[r["image_name"]]
        title = f"{r['image_name']} ({r['resolution']})"
        sub = f"P1={r['p1']:.1f}/15 P2={r['p2']:.1f}/15 Cls={r['cls_acc']*100:.0f}% | FP={r['fp']} FN={r['fn']}"
        if r["roi_clipped_beans"]:
            sub += f" ROI-CLIP:{r['roi_clipped_beans']}!"
        t = render_diagnostic_tile(img_bgr, gt_objs, pred_objs, ev, roi_box, title, sub, tile_size=540)
        tiles.append(t)
        cv2.imwrite(str(per_vis_dir / r["image_name"]), t)

    # Build 2x3 montage for worst 6 cases
    if len(tiles) == 6:
        top_row = np.hstack(tiles[:3])
        bot_row = np.hstack(tiles[3:6])
        montage = np.vstack([top_row, bot_row])
        # Legend strip on bottom
        bar = np.full((38, montage.shape[1], 3), (18, 20, 26), dtype=np.uint8)
        cv2.putText(
            bar,
            "Legend:  RED outline = MISSED GT Bean (FN)  |  MAGENTA = Ghost False Positive (FP)  |  YELLOW = Wrong Class (pred!=gt)  |  CYAN box = detect_active_pile_roi() crop",
            (18, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (225, 232, 245),
            1,
            cv2.LINE_AA,
        )
        montage = np.vstack([montage, bar])
        cv2.imwrite(str(args.out / "worst_cases_diagnostic.jpg"), montage, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # Save full JSON report
    report = {
        "model": model_label,
        "weights": str(args.weights),
        "conf_thresh": args.conf,
        "overall": overall,
        "by_category": cat_summary,
        "per_class": per_class_stats,
        "confusion_matrix_classes": CLASSES,
        "confusion_matrix": cm.tolist(),
        "top_confusing_pairs": [{"gt": g, "pred": p, "count": c} for (g, p), c in pair_errors.most_common(12)],
        "worst_cases": worst,
        "all_images": records,
    }
    (args.out / "stress_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[SAVED] Dashboard    -> {dash_path}")
    print(f"[SAVED] Worst 6 Grid -> {args.out / 'worst_cases_diagnostic.jpg'}")
    print(f"[SAVED] JSON Report  -> {args.out / 'stress_report.json'}")


if __name__ == "__main__":
    main()
