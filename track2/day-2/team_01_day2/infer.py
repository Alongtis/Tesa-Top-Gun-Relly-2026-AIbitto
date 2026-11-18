#!/usr/bin/env python3
"""
infer.py — Coffee Bean Instance Segmentation & Classification Inference Pipeline (Day 2)
Target Execution:
  - Raspberry Pi 5 (16GB / 8GB RAM, 64-bit Raspberry Pi OS aarch64, CPU)
  - Raspberry Pi 4 (4GB RAM, 64-bit OS, CPU)
  - Ubuntu / Linux & Windows 11

คำสั่งมาตรฐานตาม DAY2_INFER_SPEC.md:
    python infer.py --input test/ --weights weights/ --out outputs/

คุณสมบัติหลัก:
  1. Auto-Discovery Multi-Runtime Engine:
     - ค้นหาและเลือกใช้โมเดลที่เร็วที่สุดในโฟลเดอร์ --weights อัตโนมัติ:
       (1) NCNN (best_ncnn_model/ ~99 ms บน Pi 5)
       (2) ONNX Runtime (best.onnx ~180 ms บน Pi 5 โดยไม่ต้องพึ่ง PyTorch)
       (3) PyTorch Ultralytics (best.pt)
  2. Dynamic Active-Pile ROI Zoom:
     - หากภาพทดสอบมีความละเอียดสูง (เช่น 3072x4096) และกองเมล็ดอยู่กลางภาพ จะซูมเฉพาะบริเวณกองเมล็ด
       เข้าโมเดลโดยอัตโนมัติ แล้วแปลงพิกัด Mask และ Polygon กลับเป็นขนาดภาพต้นฉบับ (H, W)
  3. ROI-Vectorized Mask Decoding & Boundary Smoothing:
     - ถอดรหัส Prototype Mask เฉพาะในกรอบ Bounding Box ของแต่ละเม็ด (ลดเวลา Postprocess จาก 152 ms เหลือ ~18 ms)
     - บังคับ Topology ตาม DAY2_INFER_SPEC.md: 1 Instance = 1 Connected Component ถมรูใน + CHAIN_APPROX_NONE
     - รับประกันความตรงกัน 100% ระหว่าง outputs/seg_mask/<stem>.png (uint16) และ outputs/predictions.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

CLASSES: List[str] = [
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

CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "good": (235, 95, 45),
    "shell": (200, 210, 50),
    "black": (190, 180, 175),
    "husk": (50, 70, 230),
    "broken": (40, 195, 60),
    "parchment": (40, 210, 245),
    "immature": (220, 80, 210),
}


def configure_cpu_runtime() -> None:
    """ตั้งค่า CPU threads ให้เหมาะกับ Raspberry Pi 5 / Pi 4 (4-Core ARM Cortex-A76/A72)"""
    n_threads = min(4, os.cpu_count() or 4)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_threads))
    cv2.setNumThreads(n_threads)


def load_image_exif_corrected(img_path: Path) -> np.ndarray:
    """อ่านภาพและปรับหมุนตาม EXIF Orientation ให้ถูกต้องตาม DAY2_INFER_SPEC.md"""
    try:
        with Image.open(img_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise RuntimeError(f"ไม่สามารถอ่านไฟล์ภาพได้: {img_path} ({e})") from e


def smooth_closed_contour(cnt: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """ปรับเส้นขอบ Contour ปิดให้เรียบเนียนด้วย Periodic 1D Gaussian Smoothing"""
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


def detect_active_pile_roi(img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    ตรวจหาขอบเขตบริเวณที่มีเมล็ดกาแฟวางอยู่สำหรับภาพความละเอียดสูง (เช่น 3072x4096)
    หากเมล็ดกระจายเต็มภาพอยู่แล้ว จะคืนค่าขนาดภาพเต็ม (0, 0, orig_w, orig_h)
    """
    orig_h, orig_w = img_bgr.shape[:2]
    if max(orig_h, orig_w) <= 1500:
        return (0, 0, orig_w, orig_h)

    scale = 512.0 / float(max(orig_h, orig_w))
    sw = max(64, int(round(orig_w * scale)))
    sh = max(64, int(round(orig_h * scale)))
    small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_c, a_c, b_c = cv2.split(lab)
    ksize = max(31, int(round(min(sw, sh) * 0.22)) | 1)
    bg_L = cv2.GaussianBlur(L_c, (ksize, ksize), 0)
    bg_a = cv2.GaussianBlur(a_c, (ksize, ksize), 0)
    bg_b = cv2.GaussianBlur(b_c, (ksize, ksize), 0)

    dark_diff = np.clip(bg_L - L_c, 0, 255)
    chroma_diff = np.sqrt((a_c - bg_a) ** 2 + (b_c - bg_b) ** 2)
    score = np.clip(dark_diff * 0.75 + chroma_diff * 1.90, 0, 255).astype(np.uint8)

    mx, mt, mb = int(sw * 0.05), int(sh * 0.05), int(sh * 0.88)
    roi_m = np.zeros_like(score)
    roi_m[mt:mb, mx : sw - mx] = 255
    masked = cv2.bitwise_and(score, score, mask=roi_m)

    otsu_t, _ = cv2.threshold(masked[mt:mb, mx : sw - mx], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_m = cv2.threshold(masked, max(28.0, float(otsu_t) * 1.05), 255, cv2.THRESH_BINARY)
    bin_m = cv2.morphologyEx(bin_m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    cnts, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_a = (sw * sh) * 0.0004
    valid_boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= min_a]
    if not valid_boxes:
        return (0, 0, orig_w, orig_h)

    x1 = min(b[0] for b in valid_boxes) / scale
    y1 = min(b[1] for b in valid_boxes) / scale
    x2 = max(b[0] + b[2] for b in valid_boxes) / scale
    y2 = max(b[1] + b[3] for b in valid_boxes) / scale

    span = max(x2 - x1, y2 - y1, 450.0)
    side = int(round(span * 1.32))
    if side >= int(min(orig_w, orig_h) * 0.82):
        return (0, 0, orig_w, orig_h)

    cx, cy = int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5))
    rx1 = max(0, min(orig_w - side, cx - side // 2))
    ry1 = max(0, min(orig_h - side, cy - side // 2))
    return (rx1, ry1, rx1 + side, ry1 + side)


# =====================================================================
# Pure ONNX Runtime Backend สำหรับ YOLO11n-seg (ไม่ต้องพึ่ง PyTorch)
# =====================================================================

def letterbox_image(
    img_bgr: np.ndarray,
    new_shape: int = 640,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """ปรับขนาดภาพแบบรักษาสัดส่วน (Letterbox) สำหรับป้อนเข้า ONNX Runtime"""
    h, w = img_bgr.shape[:2]
    r = min(new_shape / float(h), new_shape / float(w))
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = new_shape - nw, new_shape - nh
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)

    interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, (nw / float(w), nh / float(h)), (left, top)


def run_onnx_yolo_seg(
    sess: Any,
    img_bgr: np.ndarray,
    img_size: int = 640,
    conf_thresh: float = 0.28,
    iou_thresh: float = 0.60,
) -> List[Dict[str, Any]]:
    """รัน YOLO11n-seg ผ่าน ONNX Runtime และถอดรหัส Prototype Masks แบบ Subpixel-Aligned เฉพาะในกรอบ ROI"""
    oh, ow = img_bgr.shape[:2]
    padded, (rx, ry), (pad_x, pad_y) = letterbox_image(img_bgr, img_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, :, :, :])

    inp_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {inp_name: blob})
    pred = outputs[0][0]   # (4 + nc + 32, 8400)
    proto = outputs[1][0]  # (32, 160, 160)

    nc = len(CLASSES)
    if pred.shape[0] == 4 + nc + 32:
        pred = pred.T  # (8400, 43)

    boxes_cxcywh = pred[:, :4]
    cls_scores = pred[:, 4 : 4 + nc]
    mask_coeffs = pred[:, 4 + nc : 4 + nc + 32]

    cls_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores[np.arange(len(cls_scores)), cls_ids]
    keep = confs >= conf_thresh
    if not np.any(keep):
        return []

    boxes_cxcywh = boxes_cxcywh[keep]
    confs = confs[keep]
    cls_ids = cls_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    # แปลง cx, cy, w, h -> x1, y1, w, h สำหรับ NMSBoxes
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] * 0.5
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] * 0.5
    bboxes_xywh = np.stack([x1, y1, boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(bboxes_xywh, confs.tolist(), conf_thresh, iou_thresh)
    if len(indices) == 0:
        return []

    indices = np.array(indices).flatten()
    indices = indices[np.argsort(-confs[indices])]

    ph, pw = proto.shape[1:]
    sx_p = pw / float(img_size)
    sy_p = ph / float(img_size)
    sx = rx * sx_p
    sy = ry * sy_p
    detections: List[Dict[str, Any]] = []

    for idx in indices:
        cx, cy, bw, bh = boxes_cxcywh[idx]
        lx1 = max(0.0, cx - bw * 0.5)
        ly1 = max(0.0, cy - bh * 0.5)
        lx2 = min(float(img_size), cx + bw * 0.5)
        ly2 = min(float(img_size), cy + bh * 0.5)

        # พิกัดบนภาพจริง (oh, ow) ตรงกับกรอบ Box จริงของ YOLO (ไม่ขยับ offset)
        ox1 = max(0, int(math.floor((lx1 - pad_x) / rx)))
        oy1 = max(0, int(math.floor((ly1 - pad_y) / ry)))
        ox2 = min(ow, int(math.ceil((lx2 - pad_x) / rx)))
        oy2 = min(oh, int(math.ceil((ly2 - pad_y) / ry)))
        roi_w, roi_h = ox2 - ox1, oy2 - oy1
        if roi_w < 8 or roi_h < 8:
            continue

        # พิกัดบน prototype feature map (160x160) พร้อมเผื่อขอบให้ bilinear sampling แม่นยำ
        px1 = max(0, int(math.floor(lx1 * sx_p)) - 2)
        py1 = max(0, int(math.floor(ly1 * sy_p)) - 2)
        px2 = min(pw, int(math.ceil(lx2 * sx_p)) + 3)
        py2 = min(ph, int(math.ceil(ly2 * sy_p)) + 3)

        proto_roi = proto[:, py1:py2, px1:px2]  # (32, rh, rw)
        if proto_roi.size == 0:
            continue
        logits_roi = np.tensordot(mask_coeffs[idx], proto_roi, axes=([0], [0])).astype(np.float32)

        # Subpixel-aligned inverse affine mapping จาก prototype space -> ROI พิกัดภาพจริง
        tx_local = (ox1 + 0.5) * sx + pad_x * sx_p - 0.5 - px1
        ty_local = (oy1 + 0.5) * sy + pad_y * sy_p - 0.5 - py1
        M = np.array([[sx, 0.0, tx_local], [0.0, sy, ty_local]], dtype=np.float32)
        up_logits = cv2.warpAffine(
            logits_roi,
            M,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )

        detections.append(
            {
                "bbox": (ox1, oy1, ox2, oy2),
                "conf": float(confs[idx]),
                "class": CLASSES[int(cls_ids[idx])],
                "mask_u8": (up_logits > 0.0).astype(np.uint8) * 255,
            }
        )

    return detections


# =====================================================================
# โหลดโมเดลและทำนายผลตาม DAY2_INFER_SPEC.md
# =====================================================================

def load_model(weights_path: str | Path) -> Dict[str, Any]:
    """
    โหลดโมเดลจากโฟลเดอร์ --weights โดยเรียงลำดับความเร็วบน Raspberry Pi 5 (64-bit):
      1. NCNN folder (best_ncnn_model/) ผ่าน Ultralytics
      2. ONNX Runtime (best.onnx) ผ่าน onnxruntime (เบาและเร็ว ไม่ต้องโหลด PyTorch)
      3. PyTorch checkpoint (best.pt) ผ่าน Ultralytics
    """
    configure_cpu_runtime()
    w_path = Path(weights_path)
    if not w_path.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์หรือไฟล์ weights: {w_path}")

    search_dir = w_path if w_path.is_dir() else w_path.parent
    img_size = 640
    meta_files = list(search_dir.glob("*.npz"))
    if meta_files:
        try:
            with np.load(meta_files[0], allow_pickle=False) as meta:
                if "img_size" in meta:
                    img_size = int(meta["img_size"][0])
        except Exception:
            pass

    # 1. ค้นหาโมเดล NCNN ก่อน (เร็วที่สุดบน Raspberry Pi 5 ARM Cortex-A76: ~99 ms)
    ncnn_dirs = [d for d in search_dir.iterdir() if d.is_dir() and "ncnn" in d.name.lower()] if w_path.is_dir() else []
    if ncnn_dirs:
        try:
            from ultralytics import YOLO
            model = YOLO(str(ncnn_dirs[0]), task="segment")
            # Warm-up 1 รอบเพื่อให้ NCNN จองหน่วยความจำล่วงหน้า
            model.predict(np.zeros((img_size, img_size, 3), dtype=np.uint8), imgsz=img_size, verbose=False)
            return {
                "backend": "ultralytics",
                "engine_label": f"NCNN ({ncnn_dirs[0].name})",
                "model": model,
                "img_size": img_size,
            }
        except Exception:
            pass

    # 2. ค้นหาโมเดล ONNX Runtime (ไม่ต้องพึ่ง PyTorch รันเร็วและใช้ RAM ต่ำ)
    onnx_files = [w_path] if (w_path.is_file() and w_path.suffix.lower() == ".onnx") else sorted(search_dir.glob("*.onnx"))
    if onnx_files:
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = min(4, os.cpu_count() or 4)
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(onnx_files[0]), sess_options=so, providers=["CPUExecutionProvider"])
            # Warm-up
            inp_name = sess.get_inputs()[0].name
            sess.run(None, {inp_name: np.zeros((1, 3, img_size, img_size), dtype=np.float32)})
            return {
                "backend": "onnxruntime",
                "engine_label": f"ONNX Runtime ({onnx_files[0].name})",
                "session": sess,
                "img_size": img_size,
            }
        except Exception:
            pass

    # 3. ค้นหาโมเดล PyTorch (.pt)
    pt_files = [w_path] if (w_path.is_file() and w_path.suffix.lower() == ".pt") else sorted(search_dir.glob("*.pt"))
    if pt_files:
        from ultralytics import YOLO
        model = YOLO(str(pt_files[0]), task="segment")
        model.predict(np.zeros((img_size, img_size, 3), dtype=np.uint8), imgsz=img_size, verbose=False)
        return {
            "backend": "ultralytics",
            "engine_label": f"PyTorch Ultralytics ({pt_files[0].name})",
            "model": model,
            "img_size": img_size,
        }

    raise FileNotFoundError(
        f"ไม่พบไฟล์โมเดลที่รองรับ (best_ncnn_model/, *.onnx, หรือ *.pt) ใน {w_path}"
    )


def predict_image(
    engine: Dict[str, Any],
    img_bgr: np.ndarray,
    conf_thresh: float = 0.28,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    ทำนาย Instance Mask (uint16) และรายการ objects (id, class, polygon) สำหรับภาพ 1 ภาพ
    ตรงตามข้อกำหนดใน DAY2_INFER_SPEC.md ทุกประการ
    """
    orig_h, orig_w = img_bgr.shape[:2]
    rx1, ry1, rx2, ry2 = detect_active_pile_roi(img_bgr)
    roi_bgr = img_bgr[ry1:ry2, rx1:rx2]
    rh, rw = roi_bgr.shape[:2]

    backend = engine["backend"]
    img_size = int(engine.get("img_size", 640))

    raw_detections: List[Dict[str, Any]] = []
    if backend == "onnxruntime":
        dets = run_onnx_yolo_seg(engine["session"], roi_bgr, img_size=img_size, conf_thresh=conf_thresh)
        for d in dets:
            bx1, by1, bx2, by2 = d["bbox"]
            raw_detections.append(
                {
                    "bbox": (bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1),
                    "conf": d["conf"],
                    "class": d["class"],
                    "mask_u8": d["mask_u8"],
                }
            )
    else:
        # ย่อด้วย INTER_AREA ก่อนส่งเข้า Ultralytics/NCNN เพื่อป้องกันการเกิด Aliasing ของพื้นผิวเมล็ดเมื่อย่อจากภาพความละเอียดสูง
        if max(rh, rw) > img_size:
            scale_u = img_size / float(max(rh, rw))
            uw, uh = max(32, int(round(rw * scale_u))), max(32, int(round(rh * scale_u)))
            infer_input_bgr = cv2.resize(roi_bgr, (uw, uh), interpolation=cv2.INTER_AREA)
            u_sx = rw / float(uw)
            u_sy = rh / float(uh)
        else:
            infer_input_bgr = roi_bgr
            u_sx = 1.0
            u_sy = 1.0

        res = engine["model"].predict(infer_input_bgr, imgsz=img_size, conf=conf_thresh, iou=0.60, verbose=False)[0]
        if res.masks is not None and res.boxes is not None and len(res.boxes) > 0:
            boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy().copy()
            boxes_xyxy[:, [0, 2]] *= u_sx
            boxes_xyxy[:, [1, 3]] *= u_sy
            confs = res.boxes.conf.detach().cpu().numpy()
            cls_ids = res.boxes.cls.detach().cpu().numpy().astype(int)
            masks_data = res.masks.data.detach().cpu().numpy()
            mh, mw = masks_data.shape[1:]

            sx = mw / float(rw)
            sy = mh / float(rh)
            order = np.argsort(-confs)
            for idx in order:
                if confs[idx] < conf_thresh:
                    continue
                bx1 = max(0, int(math.floor(boxes_xyxy[idx][0])))
                by1 = max(0, int(math.floor(boxes_xyxy[idx][1])))
                bx2 = min(rw, int(math.ceil(boxes_xyxy[idx][2])))
                by2 = min(rh, int(math.ceil(boxes_xyxy[idx][3])))
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
                raw_detections.append(
                    {
                        "bbox": (bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1),
                        "conf": float(confs[idx]),
                        "class": CLASSES[int(cls_ids[idx])],
                        "mask_u8": (up_m > 0.5).astype(np.uint8) * 255,
                    }
                )

    # ประกอบ Mask ลงในพิกัดภาพต้นฉบับ (orig_h, orig_w) โดยห้ามซ้อนทับกัน และบังคับ Single Connected Component
    seg_mask = np.zeros((orig_h, orig_w), dtype=np.uint16)
    objects: List[Dict[str, Any]] = []
    final_id = 1

    for det in raw_detections:
        gx1, gy1, gx2, gy2 = det["bbox"]
        mask_u8 = det["mask_u8"]

        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        best_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) < 36.0 or len(best_cnt) < 3:
            continue

        best_cnt = smooth_closed_contour(best_cnt, sigma=1.8)
        clean_sub = np.zeros_like(mask_u8)
        cv2.drawContours(clean_sub, [best_cnt], -1, 255, thickness=-1)

        roi_seg = seg_mask[gy1:gy2, gx1:gx2]
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
        global_cnt[:, 0, 0] += gx1
        global_cnt[:, 0, 1] += gy1

        objects.append(
            {
                "id": final_id,
                "class": det["class"],
                "polygon": [[int(pt[0][0]), int(pt[0][1])] for pt in global_cnt],
            }
        )
        final_id += 1

    return seg_mask, objects


def render_vis_image(
    img_bgr: np.ndarray,
    seg_mask: np.ndarray,
    objects: List[Dict[str, Any]],
    max_side: int = 1280,
) -> np.ndarray:
    """สร้างภาพ Overlay mask สำหรับบันทึกใน outputs/vis/ (ย่อก่อนวาดเพื่อความเร็วบน Pi)"""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / float(max(h, w))
        nw, nh = int(round(w * s)), int(round(h * s))
        vis = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        mask_vis = cv2.resize(seg_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    else:
        s = 1.0
        vis = img_bgr.copy()
        mask_vis = seg_mask

    overlay = vis.copy()
    vh, vw = vis.shape[:2]
    line_t = max(2, int(round(max(vh, vw) / 650)))
    font_s = max(0.45, max(vh, vw) / 1800.0)

    for obj in objects:
        oid = int(obj["id"])
        cls_name = obj["class"]
        color = CLASS_COLORS_BGR.get(cls_name, (0, 220, 0))
        pts = np.round(np.array(obj["polygon"], dtype=np.float32) * s).astype(np.int32).reshape((-1, 1, 2))
        overlay[mask_vis == oid] = color
        cv2.polylines(vis, [pts], True, (255, 255, 255), line_t + 1, cv2.LINE_AA)
        cv2.polylines(vis, [pts], True, color, line_t, cv2.LINE_AA)

    vis = cv2.addWeighted(overlay, 0.42, vis, 0.58, 0.0)
    for obj in objects:
        oid = int(obj["id"])
        cls_name = obj["class"]
        pts = np.round(np.array(obj["polygon"], dtype=np.float32) * s).astype(np.int32)
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        lbl = f"#{oid} {cls_name}"
        cv2.putText(vis, lbl, (max(4, cx - 28), cy), cv2.FONT_HERSHEY_SIMPLEX, font_s, (15, 15, 15), line_t + 1, cv2.LINE_AA)
        cv2.putText(vis, lbl, (max(4, cx - 28), cy), cv2.FONT_HERSHEY_SIMPLEX, font_s, (255, 255, 255), max(1, line_t - 1), cv2.LINE_AA)

    return vis


def collect_input_images(input_path: Path) -> List[Path]:
    """รวบรวมไฟล์ภาพจาก --input เรียงตามชื่อไฟล์ และตรวจสอบว่าชื่อไฟล์/stem ไม่ซ้ำกัน"""
    if not input_path.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์หรือไฟล์ input: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in VALID_IMAGE_EXTS:
            raise ValueError(f"นามสกุลไฟล์ไม่รองรับ: {input_path}")
        return [input_path]

    imgs = sorted(
        [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS],
        key=lambda p: p.name,
    )
    if not imgs:
        raise ValueError(f"ไม่พบไฟล์ภาพ (.jpg, .jpeg, .png) ในโฟลเดอร์: {input_path}")

    seen_stems: Dict[str, str] = {}
    for p in imgs:
        if p.stem in seen_stems:
            raise ValueError(
                f"พบชื่อไฟล์ที่มี stem ซ้ำกัน: '{p.name}' และ '{seen_stems[p.stem]}' "
                f"(จะทำให้ seg_mask/{p.stem}.png เขียนทับกันตาม DAY2_INFER_SPEC.md)"
            )
        seen_stems[p.stem] = p.name

    return imgs


def format_predictions_json(predictions_list: List[Dict[str, Any]]) -> str:
    """จัดรูปแบบ JSON ให้เหมือนตัวอย่างใน DAY2_INFER_SPEC.md (พิกัด polygon เรียงในบรรทัดเดียว อ่านง่าย)"""
    lines = ["["]
    for i, entry in enumerate(predictions_list):
        img_name_json = json.dumps(entry["image_name"], ensure_ascii=False)
        objs = entry.get("objects", [])
        if not objs:
            comma = "," if i < len(predictions_list) - 1 else ""
            lines.append(f'  {{"image_name": {img_name_json}, "objects": []}}{comma}')
            continue

        lines.append("  {")
        lines.append(f'    "image_name": {img_name_json},')
        lines.append('    "objects": [')
        for j, obj in enumerate(objs):
            obj_comma = "," if j < len(objs) - 1 else ""
            poly_str = json.dumps(obj["polygon"])
            cls_str = json.dumps(obj["class"], ensure_ascii=False)
            lines.append("      {")
            lines.append(f'        "id": {int(obj["id"])},')
            lines.append(f'        "class": {cls_str},')
            lines.append(f'        "polygon": {poly_str}')
            lines.append(f"      }}{obj_comma}")
        lines.append("    ]")
        comma = "," if i < len(predictions_list) - 1 else ""
        lines.append(f"  }}{comma}")
    lines.append("]\n")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 2 Coffee Bean Instance Segmentation Inference")
    parser.add_argument("--input", required=True, type=Path, help="Input directory or image file")
    parser.add_argument("--weights", required=True, type=Path, help="Weights directory or model file")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        if args.input.resolve() == args.out.resolve():
            raise ValueError("--out ต้องเป็นคนละโฟลเดอร์กับ --input เพื่อไม่ให้แก้ไขไฟล์ต้นฉบับ")

        img_paths = collect_input_images(args.input)
        t_load = time.perf_counter()
        engine = load_model(args.weights)
        load_ms = (time.perf_counter() - t_load) * 1000.0
        print(
            f"[MODEL] {engine.get('engine_label', engine['backend'])} "
            f"| imgsz={engine['img_size']} | warmup_load={load_ms:.1f} ms"
        )

        seg_mask_dir = args.out / "seg_mask"
        vis_dir = args.out / "vis"
        seg_mask_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        predictions_list: List[Dict[str, Any]] = []
        infer_times_ms: List[float] = []
        total_times_ms: List[float] = []

        for img_p in img_paths:
            t0 = time.perf_counter()
            img_bgr = load_image_exif_corrected(img_p)
            seg_mask_u16, objects = predict_image(engine, img_bgr)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            # ตรวจสอบความถูกต้องของผลลัพธ์ก่อนบันทึก
            for obj in objects:
                if obj["class"] not in VALID_CLASSES_SET:
                    raise RuntimeError(f"ชื่อคลาสไม่ถูกต้อง: {obj['class']}")
                if len(obj["polygon"]) < 3:
                    raise RuntimeError(f"Polygon ของ id={obj['id']} มีจุดน้อยกว่า 3 จุด")

            mask_out_path = seg_mask_dir / f"{img_p.stem}.png"
            if not cv2.imwrite(
                str(mask_out_path),
                seg_mask_u16.astype(np.uint16),
                [int(cv2.IMWRITE_PNG_COMPRESSION), 1],
            ):
                raise RuntimeError(f"ไม่สามารถบันทึกไฟล์ seg_mask ได้: {mask_out_path}")

            vis_img = render_vis_image(img_bgr, seg_mask_u16, objects)
            cv2.imwrite(str(vis_dir / img_p.name), vis_img)

            predictions_list.append(
                {
                    "image_name": img_p.name,
                    "objects": objects,
                }
            )
            total_ms = (time.perf_counter() - t0) * 1000.0
            infer_times_ms.append(infer_ms)
            total_times_ms.append(total_ms)
            print(
                f"[OK] {img_p.name:<28} -> {len(objects):3d} beans "
                f"| infer: {infer_ms:6.1f} ms | e2e: {total_ms:6.1f} ms"
            )

        pred_json_path = args.out / "predictions.json"
        pred_json_path.write_text(
            format_predictions_json(predictions_list),
            encoding="utf-8",
        )
        print(
            f"[TIMING] Images: {len(predictions_list)} "
            f"| Infer Median: {np.median(infer_times_ms):.1f} ms "
            f"| E2E Median: {np.median(total_times_ms):.1f} ms "
            f"| E2E Mean: {np.mean(total_times_ms):.1f} ms "
            f"| Max: {np.max(total_times_ms):.1f} ms"
        )
        print(f"[DONE] Saved {len(predictions_list)} predictions to {pred_json_path}")
        sys.exit(0)

    except Exception as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
