#!/usr/bin/env python3
"""
Generate unseen multi-class coffee bean test images (all 7 classes per image),
run infer.py on them, score against ground truth, and create visual proofs.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
TEAM_DIR = SCRIPT_DIR.parent
DATASETS_DIR = TEAM_DIR.parent.parent / "day-1" / "datasets"
BANK_DIR = SCRIPT_DIR / "datasets_seg" / "extracted_beans_bank"
REAL_GT_MASK_DIR = SCRIPT_DIR / "datasets_seg" / "ground_truth_day2_format" / "real_21_full" / "seg_mask"
OUT_ROOT = SCRIPT_DIR / "multi_class_test"

CLASSES = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]


def load_exif_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return cv2.cvtColor(np.array(im.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR)


def make_clean_real_paper_canvas(ref_img_path: Path, ref_mask_path: Path) -> np.ndarray:
    """Erase existing beans AND their wide shadows from a real 3072x4096 photo, preserving real paper grain & lighting."""
    img = load_exif_bgr(ref_img_path)
    seg = cv2.imread(str(ref_mask_path), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]

    bean_mask = (seg > 0).astype(np.uint8) * 255
    # Large dilation removes outer cast shadows completely (no gray ghost circles left on paper)
    dilated = cv2.dilate(bean_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (145, 145)))

    swatch = img[int(h * 0.10) : int(h * 0.28), int(w * 0.15) : int(w * 0.85)].copy()
    sh, sw = swatch.shape[:2]

    small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    small_m = cv2.resize(dilated, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
    inpainted_small = cv2.inpaint(small, small_m, 11, cv2.INPAINT_TELEA)
    bg_low = cv2.resize(inpainted_small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    swatch_low = cv2.GaussianBlur(swatch.astype(np.float32), (81, 81), 0)
    swatch_hf = swatch.astype(np.float32) - swatch_low

    rep_y = (h // sh) + 1
    rep_x = (w // sw) + 1
    tiled_hf = np.tile(swatch_hf, (rep_y, rep_x, 1))[:h, :w]

    synth_patch = np.clip(bg_low + tiled_hf, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur((dilated > 0).astype(np.float32), (51, 51), 0)[:, :, None]
    return np.clip(synth_patch.astype(np.float32) * alpha + img.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def generate_wood_bg(w: int, h: int, rng: random.Random) -> np.ndarray:
    y_grid = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x_grid = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    freq1 = rng.uniform(14.0, 22.0)
    freq2 = rng.uniform(32.0, 50.0)
    phase = rng.uniform(0.0, 6.28)
    warp = 0.025 * np.sin(x_grid * 6.0 + phase)
    stripes = (
        0.55 * np.sin((y_grid + warp) * freq1 * math.pi + phase)
        + 0.30 * np.sin((y_grid - warp * 0.5) * freq2 * math.pi)
        + 0.15 * np.cos(y_grid * 95.0 * math.pi)
    )
    noise = np.random.default_rng(rng.randint(0, 999999)).normal(0, 0.05, (h, w)).astype(np.float32)
    t = np.clip((stripes + 1.0) * 0.5 + noise, 0.0, 1.0)
    dark_wood = np.array([58, 96, 132], dtype=np.float32)
    light_wood = np.array([105, 152, 192], dtype=np.float32)
    wood = dark_wood[None, None, :] * (1.0 - t[:, :, None]) + light_wood[None, None, :] * t[:, :, None]
    return cv2.GaussianBlur(np.clip(wood, 0, 255).astype(np.uint8), (9, 3), 0)


def load_bean_bank() -> Dict[str, List[Dict[str, Any]]]:
    bank: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cls_name in CLASSES:
        for png_path in sorted((BANK_DIR / cls_name).glob("*.png")):
            rgba = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] != 4:
                continue
            bank[cls_name].append(
                {
                    "bgr": rgba[:, :, :3],
                    "mask": rgba[:, :, 3],
                    "source": png_path.stem,
                }
            )
    return bank


def transform_bean(
    bgr: np.ndarray,
    mask_u8: np.ndarray,
    angle_deg: float,
    scale: float,
    flip_h: bool,
    gain: float,
    trim_edge_px: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate/scale/flip bean and compute inner-only smooth alpha (trim white paper rim)."""
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
    _, rot_mask = cv2.threshold(rot_mask, 160, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(rot_mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty rotated bean mask")
    x1, x2 = max(0, int(xs.min()) - 2), min(diag, int(xs.max()) + 3)
    y1, y2 = max(0, int(ys.min()) - 2), min(diag, int(ys.max()) + 3)

    c_bgr = np.clip(rot_bgr[y1:y2, x1:x2].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    c_mask = rot_mask[y1:y2, x1:x2]

    dist = cv2.distanceTransform(c_mask, cv2.DIST_L2, 5)
    alpha = np.clip((dist - trim_edge_px) / 2.0, 0.0, 1.0).astype(np.float32)
    hard_mask = ((dist >= trim_edge_px) & (c_mask > 0)).astype(np.uint8) * 255
    return c_bgr, hard_mask, alpha


def composite_non_burying_pile(
    canvas: np.ndarray,
    bean_specs: List[Tuple[str, Dict[str, Any], float, float, bool, float, Tuple[float, float] | None]],
    mode: str,
    rng: random.Random,
    max_overlap_ratio: float = 0.08,
    trim_edge_px: float = 2.0,
    shadow_offset: Tuple[int, int] = (5, 7),
    shadow_blur: int = 13,
    shadow_strength: float = 0.24,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Place beans checking pixel overlap strictly so no bean is buried underneath another.
    """
    out = canvas.copy()
    ch, cw = out.shape[:2]
    raw_id_mask = np.zeros((ch, cw), dtype=np.uint16)
    placed_area: Dict[int, int] = {}
    id_to_cls: Dict[int, str] = {}
    occupied_centers: List[Tuple[float, float, float]] = []

    cur_id = 1
    for cls_name, sample, angle, scale, flip_h, gain, fixed_xy in bean_specs:
        b_bgr, b_mask, b_alpha = transform_bean(
            sample["bgr"], sample["mask"], angle, scale, flip_h, gain, trim_edge_px=trim_edge_px
        )
        bh, bw = b_mask.shape[:2]
        bean_bool = b_mask > 0
        bean_area = int(np.count_nonzero(bean_bool))
        if bean_area < 200:
            continue
        rad = 0.5 * math.hypot(bw, bh) * 0.42

        chosen_px_py = None
        if fixed_xy is not None:
            px = int(round(fixed_xy[0] - bw / 2.0))
            py = int(round(fixed_xy[1] - bh / 2.0))
            if 10 <= px and 10 <= py and px + bw < cw - 10 and py + bh < ch - 10:
                chosen_px_py = (px, py)
        else:
            for attempt in range(260):
                if mode == "dense_pile" and occupied_centers and rng.random() < 0.85:
                    ax, ay, ar = rng.choice(occupied_centers)
                    th = rng.uniform(0.0, 2.0 * math.pi)
                    dist = (ar + rad) * rng.uniform(0.84, 1.08)
                    cx = ax + math.cos(th) * dist
                    cy = ay + math.sin(th) * dist
                else:
                    cx = rng.gauss(cw * 0.50, cw * 0.17)
                    cy = rng.gauss(ch * 0.50, ch * 0.17)

                px = int(round(cx - bw / 2.0))
                py = int(round(cy - bh / 2.0))
                if px < 35 or py < 35 or px + bw >= cw - 35 or py + bh >= ch - 35:
                    continue

                roi_id = raw_id_mask[py : py + bh, px : px + bw]
                overlap_mask = (roi_id > 0) & bean_bool
                overlap_px = int(np.count_nonzero(overlap_mask))
                if overlap_px > int(bean_area * max_overlap_ratio):
                    continue

                # Check already-placed beans wouldn't lose >12% area or split
                bad_occ = False
                for aid in np.unique(roi_id[overlap_mask]):
                    cut = int(np.count_nonzero((roi_id == aid) & bean_bool))
                    rem = int(np.count_nonzero(raw_id_mask == aid)) - cut
                    if rem < int(placed_area[int(aid)] * 0.86):
                        bad_occ = True
                        break
                if bad_occ:
                    continue

                chosen_px_py = (px, py)
                break

        if chosen_px_py is None:
            continue

        px, py = chosen_px_py
        # Soft cast shadow
        sx1 = max(0, px + shadow_offset[0])
        sy1 = max(0, py + shadow_offset[1])
        sx2 = min(cw, px + bw + shadow_offset[0])
        sy2 = min(ch, py + bh + shadow_offset[1])
        sw, sh = sx2 - sx1, sy2 - sy1
        if sw > 0 and sh > 0:
            k = shadow_blur | 1
            sh_m = cv2.GaussianBlur(b_alpha[:sh, :sw], (k, k), 0)[:, :, None]
            out[sy1:sy2, sx1:sx2] = np.clip(
                out[sy1:sy2, sx1:sx2].astype(np.float32) * (1.0 - shadow_strength * sh_m),
                0,
                255,
            ).astype(np.uint8)

        a3 = b_alpha[:, :, None]
        roi = out[py : py + bh, px : px + bw].astype(np.float32)
        out[py : py + bh, px : px + bw] = np.clip(
            b_bgr.astype(np.float32) * a3 + roi * (1.0 - a3), 0, 255
        ).astype(np.uint8)

        raw_id_mask[py : py + bh, px : px + bw][bean_bool] = cur_id
        placed_area[cur_id] = bean_area
        id_to_cls[cur_id] = cls_name
        occupied_centers.append((px + bw / 2.0, py + bh / 2.0, rad))
        cur_id += 1

    final_mask = np.zeros((ch, cw), dtype=np.uint16)
    gt_objects: List[Dict[str, Any]] = []
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
        gt_objects.append(
            {
                "id": fid,
                "class": cls_name,
                "polygon": [[int(p[0][0]), int(p[0][1])] for p in g_cnt],
            }
        )
        fid += 1

    return out, final_mask, gt_objects


def build_spaced_3072_specs(bank: Dict[str, List[Dict[str, Any]]], rng: random.Random) -> List[Tuple]:
    """14 beans (2 of each of the 7 classes) spaced in ring + center at 3072x4096."""
    chosen = CLASSES * 2
    rng.shuffle(chosen)
    cx0, cy0 = 1340, 1980
    coords = []
    for i in range(10):
        ang = 2.0 * math.pi * i / 10.0 + rng.uniform(-0.07, 0.07)
        r = rng.uniform(470, 530)
        coords.append((cx0 + math.cos(ang) * r, cy0 + math.sin(ang) * r))
    for i in range(4):
        ang = 2.0 * math.pi * i / 4.0 + 0.4
        r = rng.uniform(180, 225)
        coords.append((cx0 + math.cos(ang) * r, cy0 + math.sin(ang) * r))

    specs = []
    for cls_name, (px, py) in zip(chosen, coords):
        specs.append((cls_name, rng.choice(bank[cls_name]), rng.uniform(0, 360), rng.uniform(0.96, 1.04), rng.random() < 0.5, rng.uniform(0.97, 1.03), (px, py)))
    return specs


def build_touching_pairs_3072_specs(bank: Dict[str, List[Dict[str, Any]]], rng: random.Random) -> List[Tuple]:
    """21 beans (3 of every class = 7 classes) inside a compact ~1280x1280 ROI with edge-to-edge touching pairs (no burying!)."""
    # Distance between pair centers = ~185 px -> ~175px beans touch gently edge-to-edge without heavy occlusion!
    groups = [
        (["good", "immature"], (1080, 1650)),
        (["parchment", "husk"], (1520, 1620)),
        (["broken", "shell", "black"], (1920, 1680)),
        (["black", "husk"], (1040, 2060)),
        (["shell", "good", "immature"], (1480, 2060)),
        (["broken", "parchment"], (1920, 2070)),
        (["good", "broken"], (1080, 2460)),
        (["immature", "parchment"], (1500, 2460)),
        (["black", "husk", "shell"], (1900, 2450)),
    ]
    specs = []
    for cls_list, (gx, gy) in groups:
        for idx, cls_name in enumerate(cls_list):
            ang = 2.0 * math.pi * idx / len(cls_list) + rng.uniform(-0.2, 0.2)
            r = 92.0 if len(cls_list) == 2 else 112.0
            px = gx + math.cos(ang) * r
            py = gy + math.sin(ang) * r
            specs.append((cls_name, rng.choice(bank[cls_name]), rng.uniform(0, 360), rng.uniform(0.96, 1.04), rng.random() < 0.5, rng.uniform(0.97, 1.03), (px, py)))
    return specs


def build_dense_960_specs(bank: Dict[str, List[Dict[str, Any]]], rng: random.Random, num_per_class: int = 3) -> List[Tuple]:
    """21 beans (3 of each of the 7 classes) in a dense touching 960x960 pile."""
    all_cls = CLASSES * num_per_class
    rng.shuffle(all_cls)
    specs = []
    for cls_name in all_cls:
        scale = (960.0 / 1080.0) * rng.uniform(0.84, 1.04)
        specs.append((cls_name, rng.choice(bank[cls_name]), rng.uniform(0, 360), scale, rng.random() < 0.5, rng.uniform(0.96, 1.04), None))
    return specs


def evaluate_predictions(gt_entries: List[Dict[str, Any]], pred_entries: List[Dict[str, Any]], gt_mask_dir: Path, pred_mask_dir: Path) -> Dict[str, Any]:
    pred_by_img = {p["image_name"]: p["objects"] for p in pred_entries}
    total_gt = 0
    total_matched = 0
    total_cls_correct = 0
    per_class_stats = defaultdict(lambda: [0, 0])
    ious = []
    misclassified = []

    for g_entry in gt_entries:
        name = g_entry["image_name"]
        stem = Path(name).stem
        gt_mask = cv2.imread(str(gt_mask_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        pr_mask = cv2.imread(str(pred_mask_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        gt_objs = {o["id"]: o for o in g_entry["objects"]}
        pr_objs = {o["id"]: o for o in pred_by_img.get(name, [])}

        used_pr = set()
        img_correct = 0
        for gid, gobj in gt_objs.items():
            total_gt += 1
            per_class_stats[gobj["class"]][1] += 1
            g_bin = gt_mask == gid
            overlap_ids, counts = np.unique(pr_mask[g_bin], return_counts=True)
            best_pid, best_iou = None, 0.0
            for pid, inter in zip(overlap_ids, counts):
                if pid == 0 or pid in used_pr:
                    continue
                union = np.count_nonzero(g_bin | (pr_mask == pid))
                iou = inter / float(max(union, 1))
                if iou > best_iou:
                    best_iou = iou
                    best_pid = int(pid)

            if best_pid is not None and best_iou >= 0.5:
                total_matched += 1
                used_pr.add(best_pid)
                ious.append(best_iou)
                pcls = pr_objs[best_pid]["class"]
                if pcls == gobj["class"]:
                    total_cls_correct += 1
                    img_correct += 1
                    per_class_stats[gobj["class"]][0] += 1
                else:
                    misclassified.append((name, gid, gobj["class"], pcls, round(best_iou, 3)))
            else:
                misclassified.append((name, gid, gobj["class"], "MISSED", round(best_iou, 3)))

        print(f"  [EVAL] {name:<36} | GT={len(gt_objs):2d} | Pred={len(pr_objs):2d} | Exact Cls={img_correct:2d}/{len(gt_objs):2d} ({img_correct/max(len(gt_objs),1)*100:.1f}%)")

    return {
        "total_gt": total_gt,
        "total_matched": total_matched,
        "total_cls_correct": total_cls_correct,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "per_class": dict(per_class_stats),
        "misclassified": misclassified,
    }


def make_visual_summary_grid(vis_dir: Path, names: List[str], out_path: Path) -> None:
    tiles = []
    for name in names:
        img = cv2.imread(str(vis_dir / name))
        h, w = img.shape[:2]
        if h > w:
            s_top = int(h * 0.25)
            img_sq = img[s_top : s_top + w, :]
        else:
            img_sq = img
        tile = cv2.resize(img_sq, (720, 720), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (720, 44), (24, 24, 28), -1)
        cv2.putText(tile, name, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 240, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    top = np.hstack(tiles[:2])
    bot = np.hstack(tiles[2:4])
    cv2.imwrite(str(out_path), np.vstack([top, bot]))


def main() -> None:
    rng = random.Random(20261118)
    bank = load_bean_bank()

    img_dir = OUT_ROOT / "images"
    gt_mask_dir = OUT_ROOT / "gt_seg_mask"
    pred_out_dir = OUT_ROOT / "outputs"
    for d in (img_dir, gt_mask_dir, pred_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("1. Creating clean 3072x4096 real A4 paper canvases (145px shadow cleanup)...")
    paper_3072_a = make_clean_real_paper_canvas(
        DATASETS_DIR / "good" / "circle_good.jpg",
        REAL_GT_MASK_DIR / "circle_good.png",
    )
    paper_3072_b = make_clean_real_paper_canvas(
        DATASETS_DIR / "good" / "grid_good.jpg",
        REAL_GT_MASK_DIR / "grid_good.png",
    )
    paper_960 = cv2.resize(paper_3072_a[1100:2700, 600:2200], (960, 960), interpolation=cv2.INTER_AREA)
    wood_960 = generate_wood_bg(960, 960, rng)

    specs = [
        ("01_all7_spaced_3072x4096.jpg", paper_3072_a, build_spaced_3072_specs(bank, rng), "fixed", 0.08, 1.8, (6, 8), 17, 0.24),
        ("02_all7_touching_pairs_3072x4096.jpg", paper_3072_b, build_touching_pairs_3072_specs(bank, rng), "fixed", 0.08, 1.8, (6, 8), 17, 0.24),
        ("03_all7_dense_pile_paper_960.jpg", paper_960, build_dense_960_specs(bank, rng, num_per_class=3), "dense_pile", 0.06, 2.2, (4, 5), 11, 0.22),
        ("04_all7_dense_pile_wood_960.jpg", wood_960, build_dense_960_specs(bank, rng, num_per_class=3), "dense_pile", 0.06, 2.5, (4, 5), 11, 0.28),
    ]

    gt_entries = []
    for fname, canvas, bean_specs, mode, max_ov, trim_px, sh_off, sh_blur, sh_str in specs:
        out_bgr, seg_u16, objs = composite_non_burying_pile(
            canvas, bean_specs, mode=mode, rng=rng,
            max_overlap_ratio=max_ov, trim_edge_px=trim_px,
            shadow_offset=sh_off, shadow_blur=sh_blur, shadow_strength=sh_str,
        )
        cv2.imwrite(str(img_dir / fname), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(gt_mask_dir / f"{Path(fname).stem}.png"), seg_u16.astype(np.uint16))
        gt_entries.append({"image_name": fname, "objects": objs})
        cls_cnt = Counter(o["class"] for o in objs)
        print(f"  [GEN] {fname:<36} | {len(objs):2d} beans | classes={dict(sorted(cls_cnt.items()))}")

    (OUT_ROOT / "ground_truth.json").write_text(
        json.dumps(gt_entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n2. Running infer.py on the 4 unseen multi-class images...")
    cmd = [
        sys.executable,
        str(TEAM_DIR / "infer.py"),
        "--input",
        str(img_dir),
        "--weights",
        str(TEAM_DIR / "weights"),
        "--out",
        str(pred_out_dir),
    ]
    subprocess.run(cmd, check=True)

    print("\n3. Evaluating Multi-Class Separation vs Ground Truth:")
    pred_entries = json.loads((pred_out_dir / "predictions.json").read_text(encoding="utf-8"))
    summary = evaluate_predictions(gt_entries, pred_entries, gt_mask_dir, pred_out_dir / "seg_mask")
    print(f"\n  TOTAL MATCHED: {summary['total_matched']}/{summary['total_gt']} | EXACT CLASS: {summary['total_cls_correct']}/{summary['total_gt']} ({summary['total_cls_correct']/max(summary['total_gt'],1)*100:.2f}%) | Mean IoU: {summary['mean_iou']:.4f}")
    print("  Per-class:", summary["per_class"])
    if summary["misclassified"]:
        print("  Misclassified/Missed:", summary["misclassified"])

    make_visual_summary_grid(
        pred_out_dir / "vis",
        [s[0] for s in specs],
        OUT_ROOT / "summary_multiclass_4grid.jpg",
    )
    print(f"\n[DONE] Summary image saved to: {OUT_ROOT / 'summary_multiclass_4grid.jpg'}")


if __name__ == "__main__":
    main()
