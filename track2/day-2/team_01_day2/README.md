# Team TGR-01 — Coffee Bean Instance Segmentation & Edge Optimization (Day 2)

โซลูชัน **Instance Segmentation & Classification** เมล็ดกาแฟโรบัสต้า 7 คลาส (`good`, `shell`, `black`, `husk`, `broken`, `parchment`, `immature`) ออกแบบสำหรับ **Raspberry Pi 5 (16 GB / 8 GB RAM, Raspberry Pi OS 64-bit `aarch64`)** และ **Raspberry Pi 4 (4 GB RAM, 64-bit OS)** โดยใช้ **`YOLO11n-seg`** ร่วมกับ **Active-Pile ROI Zoom**, **Subpixel-Aligned Prototype Mask Decoding**, และ **Periodic Gaussian Contour Smoothing**

## โครงสร้างไฟล์

```text
team_01_day2/
├── README.md
├── requirements.txt
├── train.py
├── train_colab.ipynb
├── infer.py
├── weights/
│   ├── best_ncnn_model/       # โมเดล NCNN FP32 สำหรับ Raspberry Pi 5 (รันเร็วที่สุดบน ARM CPU)
│   ├── best.onnx              # โมเดล ONNX Runtime FP32 (รันได้โดยไม่ต้องพึ่ง PyTorch)
│   ├── best.pt                # โมเดล PyTorch Ultralytics YOLO11n-seg (2.86M params)
│   └── model_meta.npz         # ข้อมูล config (img_size=640) และ 28 Physical Features Metadata
└── outputs/
    ├── seg_mask/
    │   ├── img_01.png         # Single-channel uint16 PNG (0=พื้นหลัง, 1..N=ID เมล็ด)
    │   └── img_02.png
    ├── predictions.json       # พิกัด Polygon (CHAIN_APPROX_NONE) และคลาสของทุกเมล็ด
    ├── vis/                   # ภาพ Overlay mask สำหรับกรรมการตรวจด้วยสายตา
    └── training_analysis/     # กราฟการเทรน, Confusion Matrix, PCA และ Feature Maps 5 Stage
```

## การติดตั้งบน Raspberry Pi 5 (64-bit OS `aarch64`) / Ubuntu / Windows

```bash
# 1. สร้างและเปิดใช้งาน Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# 2. (แนะนำบน Raspberry Pi OS 64-bit) ติดตั้ง PyTorch CPU wheel ก่อนเพื่อให้ pip แก้ dependency เร็วขึ้น
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. ติดตั้ง dependencies สำหรับ infer.py (รองรับทั้ง NCNN และ ONNX Runtime)
pip install -r requirements.txt
```

> **หมายเหตุ (Raspberry Pi OS Lite / Headless):**
> - หากระบบไม่มี library ของหน้าจอและพบ error `libGL.so.1` จากแพ็กเกจ `opencv-python` ที่ติดมากับ `ultralytics` ให้ติดตั้งเพิ่มด้วย `sudo apt update && sudo apt install -y libgl1 libglib2.0-0`
> - หากต้องการติดตั้งแบบเบาสุด (~60 MB ไม่ลง PyTorch/Ultralytics) ให้ติดตั้งเฉพาะ `pip install opencv-python-headless numpy pillow onnxruntime` ตัว `infer.py` จะสลับไปใช้ `weights/best.onnx` ให้อัตโนมัติ

## คำสั่งรันทำนายผล (`infer.py`)

```bash
python infer.py --input test/ --weights weights/ --out outputs/
```

- `--input`: โฟลเดอร์ภาพทดสอบ หรือไฟล์ภาพเดี่ยว (`.jpg`, `.jpeg`, `.png` แบบไม่สนตัวพิมพ์เล็ก-ใหญ่) เรียงลำดับตามชื่อไฟล์ก่อนประมวลผล
- `--weights`: **โฟลเดอร์ `weights/`** โดย `infer.py` จะค้นหาและเลือก Backend ที่เร็วที่สุดให้อัตโนมัติ (`best_ncnn_model/` $\rightarrow$ `best.onnx` $\rightarrow$ `best.pt`) และสามารถระบุไฟล์เดี่ยวโดยตรงได้เช่นกัน
- `--out`: โฟลเดอร์ปลายทางสำหรับบันทึก `seg_mask/<stem>.png`, `predictions.json` และ `vis/<image_name>`

## สถาปัตยกรรมโมเดลและการ Preprocessing / Postprocessing

1. **อ่านภาพและปรับทิศทาง EXIF:** อ่านภาพด้วย Pillow พร้อมปรับหมุนด้วย `ImageOps.exif_transpose` ก่อนทุกขั้นตอน และไม่แก้ไข/ลบไฟล์ต้นฉบับใน `--input`
2. **Dynamic Active-Pile ROI Zoom:** สำหรับภาพมุมกว้างความละเอียดสูง (เช่น `3072×4096`) จะใช้ตัวกรอง CIE $L^*a^*b^*$ ตรวจหาขอบเขตกลุ่มเมล็ดแล้วซูมเฉพาะโซนกองเมล็ดเข้าโมเดล ก่อนแปลงพิกัดกลับสู่ภาพต้นฉบับ `(H, W)` เพื่อเพิ่มความละเอียดของเมล็ดขนาดเล็ก
3. **Deterministic Inference (`imgsz=640`, ไม่ใช้ TTA / Random Augmentation):**
   - ใช้โมเดล **`YOLO11n-seg`** (2.86M params, 9.9 GFLOPs) รันบน CPU ด้วยโหมด Multi-thread (`4 threads`) พร้อม Warm-up 1 รอบตอนโหลดโมเดล
   - **Subpixel-Aligned ROI Mask Decoding:** ถอดรหัส 32 Prototype Masks (`160×160`) เฉพาะในกรอบ Bounding Box ของแต่ละเม็ดด้วย `cv2.warpAffine` แทนการขยาย Mask เต็มภาพ เพื่อลดความหน่วง Postprocessing และรักษาความแม่นยำระดับซับพิกเซล
4. **Topology Enforcement ตาม `DAY2_INFER_SPEC.md`:**
   - แปลง Mask ให้เป็น **Single Connected Component ที่ถมรูด้านในแล้ว** โดยไม่มีเม็ดใดซ้อนทับกัน (`1 พิกเซล = 1 Instance ID`)
   - ปรับความเรียบของเส้นขอบด้วย **Periodic 1D Gaussian Contour Smoothing** แล้วดึงพิกัด `polygon` ด้วย `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)` จากข้อมูล `seg_mask` จริง เพื่อให้ `seg_mask/*.png` (`uint16`) และ `predictions.json` ตรงกัน `IoU = 1.000000`

## ไฟล์ประกอบใน `weights/`

- `weights/best_ncnn_model/` (~11.0 MB): โมเดล `YOLO11n-seg` ในฟอร์แมต NCNN FP32 (`model.ncnn.param`, `model.ncnn.bin`, `metadata.yaml`) _(Backend หลักที่ `infer.py` เลือกใช้บน Raspberry Pi 5)_
- `weights/best.onnx` (11.15 MB): โมเดล `YOLO11n-seg` ในฟอร์แมต ONNX FP32 (opset 17, simplified) _(รันผ่าน `onnxruntime` ได้ทันทีโดยไม่ต้องพึ่ง PyTorch)_
- `weights/best.pt` (5.77 MB): ไฟล์ checkpoint ฝั่ง PyTorch Ultralytics สำหรับเก็บอ้างอิงหรือ Fine-tune ต่อ
- `weights/model_meta.npz` (1.06 MB): ข้อมูล `img_size`, รายชื่อคลาส 7 คลาส, และสถิติ 28 Physical Features จากชุดเทรน

## พฤติกรรมโปรแกรมและรูปแบบผลลัพธ์ (`DAY2_INFER_SPEC.md`)

- **`outputs/seg_mask/<stem>.png`:** ภาพ Single-channel `uint16` ขนาดเท่าภาพต้นฉบับ (`0` = พื้นหลัง, `1..N` = ID ของแต่ละเม็ด; หากไม่พบเม็ดจะเป็นค่า `0` ทั้งภาพ)
- **`outputs/predictions.json`:** UTF-8 JSON Array ประกอบด้วย `image_name` และรายการ `objects` (`id`, `class` ตัวพิมพ์เล็ก, `polygon` อย่างน้อย 3 จุด; หากไม่พบเม็ด `objects` เป็น `[]`)
- **`outputs/vis/`:** ภาพสี Overlay mask พร้อมเส้นขอบและเลข ID/คลาสสำหรับตรวจสอบด้วยสายตา
- **Exit Code & Error Handling:** คืนค่า `0` เมื่อสำเร็จ; หากไม่พบ `--input`/`--weights`, ไม่มีภาพ, พบ `stem` ซ้ำ, `--out` ซ้ำกับ `--input`, หรืออ่านภาพไม่ได้ จะแสดงข้อความ error และจบด้วย **non-zero exit code ทันที**
