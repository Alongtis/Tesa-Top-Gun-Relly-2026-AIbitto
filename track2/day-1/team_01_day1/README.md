# Team TGR-01 — Coffee Bean Defect Classification (Day 1)

โซลูชันจำแนกคุณภาพเมล็ดกาแฟ 7 คลาส (`good`, `shell`, `black`, `husk`, `broken`, `parchment`, `immature`) รองรับการทำงานทั้งบน **Raspberry Pi 4 (32-bit `armv7l` และ 64-bit `aarch64`)**, **Ubuntu / Linux**, และ **Windows** โดยฝั่ง Inference (`infer.py`) ใช้เพียง **OpenCV DNN + NumPy + Pillow** ไม่ต้องติดตั้ง PyTorch หรือ scikit-learn

## โครงสร้างไฟล์

```text
team_01_day1/
├── README.md
├── requirements.txt
├── train.py
├── infer.py
├── weights/
│   ├── best.onnx
│   ├── model_meta.npz
│   └── best.pt
└── outputs/
    ├── predictions.csv
    ├── good/
    ├── shell/
    ├── black/
    ├── husk/
    ├── broken/
    ├── parchment/
    └── immature/
```

## การติดตั้ง (Raspberry Pi OS 32-bit / 64-bit / Ubuntu / Linux)

> ใช้เฉพาะ Dependency สำหรับ Inference ตาม `requirements.txt` เท่านั้น (มี pre-built wheels บน PyPI และ piwheels สำหรับ 32-bit `armv7l` ครบทุกตัว)

```bash
# 1. (แนะนำบน Raspberry Pi OS / Ubuntu) สร้าง Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# 2. ติดตั้ง dependency สำหรับ infer.py
pip install --upgrade pip
pip install -r requirements.txt
```

_(หมายเหตุ: หาก Raspberry Pi OS ยังขาด system library พื้นฐานของภาพ ให้รัน `sudo apt update && sudo apt install -y python3-venv libopenjp2-7 libtiff6` เพิ่มเติมก่อน)_

## คำสั่งรันทำนายผล (`infer.py`)

```bash
python infer.py --input test/ --weights weights/ --out outputs/
```

- `--input`: โฟลเดอร์ภาพทดสอบ หรือไฟล์ภาพเดี่ยว (`.jpg`, `.jpeg`, `.png` ไม่สนตัวพิมพ์เล็ก/ใหญ่)
- `--weights`: **โฟลเดอร์ `weights/`** โดย `infer.py` จะค้นหาไฟล์ weight ที่ต้องใช้ภายในโฟลเดอร์ให้อัตโนมัติไม่จำกัดชื่อไฟล์ (รองรับการเก็บไฟล์ weight หลายไฟล์ไว้ด้วยกัน และสามารถระบุชื่อไฟล์ตรงๆ ได้เช่นกัน)
- `--out`: โฟลเดอร์ปลายทางสำหรับบันทึก `predictions.csv` และคัดลอกภาพเข้าโฟลเดอร์ทั้ง 7 คลาส

## สถาปัตยกรรมโมเดลและการ Preprocessing

1. **อ่านภาพ & EXIF Orientation:** อ่านภาพผ่าน Pillow พร้อมปรับหมุนด้วย `ImageOps.exif_transpose` ก่อนส่งเข้าประมวลผลทุกครั้ง
2. **Bean Localization (OpenCV):** ค้นหาเมล็ดกาแฟในภาพด้วย Local Illumination Normalization + CIE $L^*a^*b^*$ + Watershed เพื่อแยกเมล็ดแต่ละเม็ดออกจากพื้นหลังและตัดขอบโต๊ะ (หากเป็นภาพซูมเมล็ดเดียวเต็มเฟรมจะใช้ทั้งภาพโดยอัตโนมัติ)
3. **Deterministic Preprocessing (ไม่มี TTA / Random Augmentation):** ทำ Letterbox Resize เป็น `224x224` เติมขอบด้วยสีพื้นหลังของภาพ และ Normalize ด้วยค่า ImageNet mean/std
4. **Hybrid Classification (OpenCV DNN + Pure-NumPy):**
   - **ConvNet Branch (`weights/best.onnx`):** `MobileNetV3-Small` รันผ่าน `cv2.dnn.readNetFromONNX` บน CPU
   - **Physical Branch (28 OpenCV Features):** รูปทรง (`Solidity`, `Circularity`, `Convexity Defects`), ค่าสี CIE $L^*a^*b^*$ เทียบกับพื้นหลัง, อัตราส่วนโพรง `Core-vs-Rim` (`shell`), และพื้นผิว `GLCM`/`Black-Hat`
   - **Ensemble (`weights/model_meta.npz`):** รวมคะแนนจาก ConvNet (`55%`), Cosine Class Prototypes (`20%`), และ Flat Pure-NumPy Decision Trees (`25%`) ของทุกเม็ดในภาพเพื่อสรุปผลเป็น 1 คลาสต่อภาพ

## ไฟล์ประกอบใน `weights/`

- `weights/best.onnx` (3.55 MB): น้ำหนัก MobileNetV3-Small Feature Extractor สำหรับ `cv2.dnn.readNetFromONNX` _(ไฟล์หลักที่ `infer.py` โหลดเมื่อส่ง `--weights weights/`)_
- `weights/model_meta.npz` (0.80 MB): พารามิเตอร์ Fusion MLP, Classifier, Standardization (`feat_mean`, `feat_std`), `class_prototypes` และ Decision Trees ในฟอร์แมต Pure-NumPy (`allow_pickle=False`) _(ไฟล์ประกอบบังคับคู่กับ `best.onnx`)_
- `weights/best.pt` (6.59 MB): ไฟล์ checkpoint ฝั่ง PyTorch ที่ได้จากการเทรน (เก็บไว้เป็นไฟล์อ้างอิง/สำหรับเทรนต่อ หรือใช้เฉพาะเมื่อระบุ `--weights weights/<file>.pt` บนเครื่องที่มี PyTorch)

## พฤติกรรมโปรแกรมและผลลัพธ์ (`INFER_SPEC.md`)

- **Interface หลักใน `infer.py`:** `load_model()` และ `predict_image()`
- **เอาต์พุตใน `--out`:** สร้างไฟล์ `predictions.csv` (UTF-8, คอลัมน์ `filename,class`) เรียงลำดับตามชื่อไฟล์ พร้อมคัดลอกภาพต้นฉบับไปยังโฟลเดอร์ย่อยทั้ง 7 คลาส (`good/`, `shell/`, `black/`, `husk/`, `broken/`, `parchment/`, `immature/`) โดยไม่ลบหรือย้ายภาพต้นฉบับ
- **Exit Code:** คืนค่า `0` เมื่อสำเร็จ และคืนค่าไม่ใช่ `0` พร้อมข้อความ error ทันทีหากไม่พบ input/weights ไม่มีภาพ อ่านภาพไม่ได้ หรือรูปแบบผลลัพธ์ไม่ถูกต้อง
