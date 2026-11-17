#!/usr/bin/env python3
"""
ตัวอย่างสคริปต์ Auto-Label & Auto-Crop เมล็ดกาแฟด้วย OpenCV
สำหรับตรวจสอบความแม่นยำของกรอบ Bounding Box และภาพ Crop รายเม็ดทั้ง 7 คลาส

ผลลัพธ์ที่ได้จะถูกบันทึกไว้ในโฟลเดอร์:
  track2/day-1/auto_label_preview/
  ├── summary_all_21_images.jpg     <-- ภาพรวมทั้ง 21 รูปพร้อมตีกรอบ Bounding Box
  ├── crops_grid_by_class/          <-- ภาพรวมเมล็ดที่ถูก Crop แยกตามคลาส (ตรวจดูง่ายในภาพเดียว)
  ├── annotated_images/             <-- ภาพเต็มแต่ละรูปที่วาดเส้น Contour (สีเขียว) + กรอบ YOLO (สีฟ้า)
  ├── crops/                        <-- ภาพเมล็ดเดี่ยวๆ ที่ตัดออกมาแยกตามโฟลเดอร์ 7 คลาส
  └── yolo_labels/                  <-- ไฟล์ .txt ฟอร์แมต YOLO (เปิดแก้ไขต่อใน LabelImg ได้ทันที)
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageOps

# ลำดับคลาสตามสเปก INFER_SPEC.md
CLASSES = [
    "good",
    "shell",
    "black",
    "husk",
    "broken",
    "parchment",
    "immature",
]


def load_image_exif_corrected(img_path: Path) -> np.ndarray:
    """อ่านภาพพร้อมปรับหมุนตาม EXIF Orientation ให้ถูกต้องตาม INFER_SPEC.md"""
    pil_img = Image.open(img_path)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_beans_opencv(
    img_bgr: np.ndarray,
    work_scale: float = 0.25,
    pad_ratio: float = 0.08,
) -> List[Dict[str, Any]]:
    """
    ตรวจจับเมล็ดกาแฟทุกเม็ดบนพื้นหลังกระดาษขาวด้วย OpenCV
    โดยใช้การลบพื้นหลังท้องถิ่น (Local Illumination Normalization) + สี CIE L*a*b*
    ทำให้ทนต่อแสงเงาไล่เฉดบนกระดาษ และตัดขอบโต๊ะด้านล่างออกอัตโนมัติ
    """
    orig_h, orig_w = img_bgr.shape[:2]
    ws_w = int(round(orig_w * work_scale))
    ws_h = int(round(orig_h * work_scale))
    small = cv2.resize(img_bgr, (ws_w, ws_h), interpolation=cv2.INTER_AREA)

    # 1. แปลงเป็น CIE L*a*b* และ HSV เพื่อจับทั้งเม็ดสีเข้ม (black/husk) และเม็ดสีเหลืองอ่อน (parchment/immature)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_chan, a_chan, b_chan = cv2.split(lab)

    # ประมาณค่าแสงพื้นหลังกระดาษด้วย GaussianBlur ขนาดใหญ่กว่าเมล็ดกาแฟ
    bg_L = cv2.GaussianBlur(L_chan, (151, 151), 0)
    bg_a = cv2.GaussianBlur(a_chan, (151, 151), 0)
    bg_b = cv2.GaussianBlur(b_chan, (151, 151), 0)

    # ความมืดกว่ากระดาษ (Darkness diff) + ความต่างของสี (Chroma diff สำหรับเม็ดสีอ่อนแต่ติดเหลือง/น้ำตาล)
    dark_diff = np.clip(bg_L - L_chan, 0, 255)
    chroma_diff = np.sqrt(np.maximum(0.0, (a_chan - bg_a) ** 2 + (b_chan - bg_b) ** 2))

    # รวมคะแนนความต่างจากพื้นหลังกระดาษ (ถ่วงน้ำหนักสีเพื่อให้เม็ดสีอ่อนเด่นชัด และไม่จับเงาจางๆ บนกระดาษ)
    fg_score = dark_diff * 0.75 + chroma_diff * 1.85
    fg_score_u8 = np.clip(fg_score, 0, 255).astype(np.uint8)
    fg_score_u8 = cv2.GaussianBlur(fg_score_u8, (5, 5), 0)

    # 2. ตัดขอบภาพด้านนอก (ป้องกันขอบโต๊ะสีดำด้านล่างและมุมภาพ)
    margin_x = int(ws_w * 0.05)
    margin_top = int(ws_h * 0.05)
    margin_bot = int(ws_h * 0.88)  # ขอบโต๊ะมักอยู่ล่างสุดของเฟรม
    roi_mask = np.zeros_like(fg_score_u8)
    roi_mask[margin_top:margin_bot, margin_x : ws_w - margin_x] = 255
    fg_masked = cv2.bitwise_and(fg_score_u8, fg_score_u8, mask=roi_mask)

    # 3. Otsu Thresholding + เกณฑ์ขั้นต่ำเพื่อตัดเงาจางๆ รอบเมล็ด
    otsu_val, _ = cv2.threshold(
        fg_masked[margin_top:margin_bot, margin_x : ws_w - margin_x],
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    thresh_val = max(28.0, float(otsu_val) * 1.08)
    _, binary = cv2.threshold(fg_masked, thresh_val, 255, cv2.THRESH_BINARY)

    # 4. Morphology ตัดจุดฝุ่นผงและปิดรูเล็กๆ โดยไม่เชื่อมเมล็ดที่วางใกล้กัน
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 5. แยกเมล็ดที่อาจวางชิดกันด้วย Distance Transform + Watershed (กรณีเงาเชื่อมกัน)
    raw_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = ws_w * ws_h
    min_bean_area = img_area * 0.00045
    max_bean_area = img_area * 0.04000

    valid_areas = [cv2.contourArea(c) for c in raw_contours if min_bean_area <= cv2.contourArea(c) <= max_bean_area]
    median_area = float(np.median(valid_areas)) if valid_areas else (img_area * 0.002)

    contours = []
    for cnt in raw_contours:
        area = cv2.contourArea(cnt)
        if not (min_bean_area <= area <= max_bean_area):
            continue
        # ถ้าก้อนไหนใหญ่เกิน 1.55 เท่าของขนาดเมล็ดปกติ แปลว่ามี 2 เม็ดเชื่อมกัน -> แยกด้วย Distance Transform
        if area > median_area * 1.55:
            mask_single = np.zeros_like(binary)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)
            dist = cv2.distanceTransform(mask_single, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
            sure_fg = sure_fg.astype(np.uint8)
            n_labels, labels = cv2.connectedComponents(sure_fg)
            if n_labels > 2:  # แยกได้มากกว่า 1 เม็ด
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
        if not (min_bean_area <= area <= max_bean_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > 3.5:
            continue  # กรองเส้นยาวผิดปกติ

        # แปลงพิกัดกลับเป็นสเกลภาพต้นฉบับ (3072x4096)
        cnt_orig = (cnt.astype(np.float32) / work_scale).astype(np.int32)
        x1 = int(round(x / work_scale))
        y1 = int(round(y / work_scale))
        bw = int(round(w / work_scale))
        bh = int(round(h / work_scale))
        x2 = x1 + bw
        y2 = y1 + bh

        # เผื่อขอบ (Padding) เล็กน้อยสำหรับภาพ Crop และกรอบ Bounding Box
        pad_x = int(round(bw * pad_ratio))
        pad_y = int(round(bh * pad_ratio))
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(orig_w, x2 + pad_x)
        py2 = min(orig_h, y2 + pad_y)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 1.0

        beans.append(
            {
                "bbox_tight": (x1, y1, x2, y2),
                "bbox_padded": (px1, py1, px2, py2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "contour": cnt_orig,
                "solidity": round(solidity, 3),
                "area_px": int(round(area / (work_scale**2))),
            }
        )

    # เรียงลำดับเมล็ดจากบนลงล่าง ซ้ายไปขวา เพื่อให้นับตรวจเช็คง่าย
    beans.sort(key=lambda b: (int(b["center"][1] // 250), b["center"][0]))
    return beans


def make_class_crops_sheet(crops: List[np.ndarray], class_name: str, cols: int = 5) -> np.ndarray:
    """สร้างภาพตารางรวมเมล็ดที่ถูก Crop ออกมาทั้งหมดของแต่ละคลาส"""
    cell_size = 200
    n = len(crops)
    rows = max(1, (n + cols - 1) // cols)
    header_h = 60
    sheet = np.full((header_h + rows * cell_size, cols * cell_size, 3), 245, dtype=np.uint8)

    cv2.putText(
        sheet,
        f"Class: {class_name} | Total Cropped Beans: {n}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    for idx, crop in enumerate(crops):
        r = idx // cols
        c = idx % cols
        ch, cw = crop.shape[:2]
        scale = min((cell_size - 24) / max(cw, 1), (cell_size - 24) / max(ch, 1))
        nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        y0 = header_h + r * cell_size + (cell_size - nh) // 2
        x0 = c * cell_size + (cell_size - nw) // 2
        sheet[y0 : y0 + nh, x0 : x0 + nw] = resized

        # วาดกรอบช่องและหมายเลขเมล็ด
        cv2.rectangle(
            sheet,
            (c * cell_size + 4, header_h + r * cell_size + 4),
            ((c + 1) * cell_size - 4, header_h + (r + 1) * cell_size - 4),
            (200, 200, 200),
            1,
        )
        cv2.putText(
            sheet,
            f"#{idx + 1}",
            (c * cell_size + 10, header_h + r * cell_size + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 100, 200),
            2,
            cv2.LINE_AA,
        )

    return sheet


def main():
    parser = argparse.ArgumentParser(description="Preview OpenCV Auto-Label & Auto-Crop on Coffee Dataset")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets",
        help="Path to datasets folder",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "auto_label_preview",
        help="Output folder for visual verification and YOLO labels",
    )
    args = parser.parse_args()

    out_annotated = args.out_dir / "annotated_images"
    out_crops = args.out_dir / "crops"
    out_crops_grid = args.out_dir / "crops_grid_by_class"
    out_yolo = args.out_dir / "yolo_labels"

    for d in (out_annotated, out_crops, out_crops_grid, out_yolo):
        d.mkdir(parents=True, exist_ok=True)

    # บันทึก classes.txt สำหรับเปิดใน LabelImg
    (out_yolo / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    summary_thumbs: List[np.ndarray] = []
    total_beans_all = 0

    print("=" * 65)
    print(f"{'CLASS':<12} | {'IMAGE':<22} | {'DETECTED BEANS':<15}")
    print("=" * 65)

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = args.dataset_dir / cls_name
        if not cls_dir.exists():
            print(f"[WARN] Missing folder: {cls_dir}")
            continue

        cls_crop_dir = out_crops / cls_name
        cls_crop_dir.mkdir(parents=True, exist_ok=True)
        for old_f in cls_crop_dir.glob("*.jpg"):
            old_f.unlink()
        (out_yolo / cls_name).mkdir(parents=True, exist_ok=True)

        cls_crops_list: List[np.ndarray] = []
        img_files = sorted(
            [p for p in cls_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )

        for img_path in img_files:
            img_bgr = load_image_exif_corrected(img_path)
            orig_h, orig_w = img_bgr.shape[:2]
            beans = detect_beans_opencv(img_bgr)
            total_beans_all += len(beans)

            print(f"{cls_name:<12} | {img_path.name:<22} | {len(beans)} beans")

            vis = img_bgr.copy()
            yolo_lines: List[str] = []

            for b_idx, b in enumerate(beans, start=1):
                px1, py1, px2, py2 = b["bbox_padded"]
                tx1, ty1, tx2, ty2 = b["bbox_tight"]

                # 1. เซฟภาพ Crop รายเม็ด
                crop_img = img_bgr[py1:py2, px1:px2]
                crop_name = f"{img_path.stem}_bean_{b_idx:02d}.jpg"
                cv2.imwrite(str(out_crops / cls_name / crop_name), crop_img)
                cls_crops_list.append(crop_img)

                # 2. คำนวณพิกัด YOLO format: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
                cx_norm = ((px1 + px2) / 2.0) / orig_w
                cy_norm = ((py1 + py2) / 2.0) / orig_h
                w_norm = (px2 - px1) / orig_w
                h_norm = (py2 - py1) / orig_h
                yolo_lines.append(
                    f"{cls_idx} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

                # 3. วาดเส้น Contour เมล็ด (สีเขียว) + กรอบ YOLO Padded Box (สีแดง/ฟ้า) ลงบนภาพตรวจสอบ
                cv2.drawContours(vis, [b["contour"]], -1, (0, 230, 0), 6)
                cv2.rectangle(vis, (px1, py1), (px2, py2), (255, 80, 0), 6)
                cv2.putText(
                    vis,
                    f"#{b_idx} {cls_name}",
                    (px1, max(45, py1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 0, 220),
                    4,
                    cv2.LINE_AA,
                )

            # เขียนป้ายสรุปมุมบนซ้ายของภาพ
            cv2.rectangle(vis, (20, 20), (1350, 150), (30, 30, 30), -1)
            cv2.putText(
                vis,
                f"{img_path.name} -> Detected: {len(beans)} beans",
                (45, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.2,
                (0, 255, 255),
                5,
                cv2.LINE_AA,
            )

            # บันทึกภาพที่ตีกรอบแล้ว (ย่อลงครึ่งหนึ่งเพื่อให้เปิดดูง่ายและไฟล์ไม่ใหญ่เกินไป)
            vis_half = cv2.resize(vis, (orig_w // 2, orig_h // 2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_annotated / f"annotated_{img_path.name}"), vis_half)

            # บันทึกไฟล์ Label .txt
            label_path = out_yolo / cls_name / f"{img_path.stem}.txt"
            label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

            # เก็บ Thumbnail สำหรับสร้างภาพรวม 21 รูป (ซูมเฉพาะโซนกลางภาพที่เมล็ดวางอยู่ให้เห็นชัดๆ)
            thumb = cv2.resize(vis, (450, 600), interpolation=cv2.INTER_AREA)
            summary_thumbs.append(thumb)

        # สร้างแผ่นรวมภาพ Crop ของคลาสนี้
        class_sheet = make_class_crops_sheet(cls_crops_list, cls_name, cols=5)
        cv2.imwrite(str(out_crops_grid / f"crops_{cls_name}.jpg"), class_sheet)

    # สร้างภาพรวมทั้ง 21 รูป (7 แถว x 3 คอลัมน์)
    if len(summary_thumbs) == 21:
        rows_imgs = []
        for r in range(7):
            row_concat = np.hstack(summary_thumbs[r * 3 : (r + 1) * 3])
            rows_imgs.append(row_concat)
        full_grid = np.vstack(rows_imgs)
        cv2.imwrite(str(args.out_dir / "summary_all_21_images.jpg"), full_grid)

    print("=" * 65)
    print(f"[SUCCESS] Total beans extracted across all 21 images: {total_beans_all}")
    print(f"[OUTPUT]  Saved visual verification files to: {args.out_dir}")


if __name__ == "__main__":
    main()
