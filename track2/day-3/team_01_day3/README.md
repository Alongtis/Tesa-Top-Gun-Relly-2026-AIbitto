# Team TGR-01 — Coffee Bean Integration Pipeline (Day 3)

โซลูชัน **Multi-Class Instance Segmentation & Classification + HTTP/MQTT Integration** สำหรับตรวจคุณภาพเมล็ดกาแฟโรบัสต้าคละทั้ง 7 คลาส (`good`, `shell`, `black`, `husk`, `broken`, `parchment`, `immature`) บน **Raspberry Pi 5 (64-bit OS `aarch64`, CPU)** และ **Raspberry Pi 4**

- **โมเดล:** `YOLO11n-seg` (`imgsz=640`, V2 Hardened Fine-Tune ครบ 16 ภาพจริง Day 3 + 21 ภาพ Day 1/2 Replay + กองคละคลาสซ้อนทับ + เสริมความทนทานแสง/White-Balance/ISO Noise/Hard-Negative)
- **ความแม่นยำและเวลา:**
  - **ชุดจริง Day 3 (16 ภาพ, 124 เม็ด):** `P1 (IoU) = 14.24/15` (`94.9%`), `P2 (Count) = 15.00/15` (`100.0%`), Multi-Class Accuracy **`100.0%` (`124/124`)**, `MQTT Defect-Ratio MAE = 0.0000`
  - **ชุดจริง Day 1/2 Replay (21 ภาพ, 175 เม็ด):** นับถูกและจำแนกคลาสถูก **`100.0%` (`175/175`)**
  - **ชุด Validation V2 (23 ภาพ) / ชุดทดสอบโหด (42 ภาพ, 577 เม็ด):** Val `P1=14.54/15`, `P2=14.96/15`, `ClsAcc=98.69%` | Brutal Stress `P1=14.20/15`, `P2=14.90/15 (99.3%)`, `ClsAcc=87.8%`
  - **Runtime:** รัน **Native NCNN (`best_ncnn_model/`)** ตรงๆ โดยไม่ต้องลง PyTorch/Ultralytics/CUDA พร้อม **ONNX Runtime (`best.onnx`)** สำรองอัตโนมัติ (`< 100 ms` บน Pi 5)

---

## โครงสร้างโฟลเดอร์ (`team_01_day3/`)

```text
team_01_day3/
├── README.md
├── requirements.txt
├── train.py
├── infer.py
├── client.py              # HTTP + MQTT (WSS MQTTv5) ในสคริปต์เดียว
├── weights/
│   ├── best_ncnn_model/   # โมเดล NCNN FP32 สำหรับ Raspberry Pi 5 (เร็วที่สุด)
│   ├── best.onnx          # โมเดล ONNX Runtime FP32 (รันได้โดยไม่ต้องพึ่ง PyTorch)
│   ├── best.pt            # โมเดล PyTorch checkpoint
│   └── model_meta.npz     # ข้อมูล config (img_size=640, classes)
└── outputs/
    ├── seg_mask/          # <ภาพ>.png (uint16: 0=พื้นหลัง, 1..N=ID เมล็ด)
    ├── predictions.json   # รายการ image_name + objects [{id, class, polygon}]
    ├── mqtt/              # สำเนา <ภาพ>.json ที่ publish ขึ้น MQTT
    └── vis/               # ภาพ overlay mask
```

---

## 1. การติดตั้งบน Raspberry Pi 5 (64-bit OS) / Linux / Windows

เบาและเร็ว (~100 MB, **ไม่ติดตั้ง PyTorch / Ultralytics / NVIDIA CUDA**):

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **หมายเหตุ (Raspberry Pi OS Lite / Headless):**
> - หากพบ error `libGL.so.1` ให้รัน `sudo apt update && sudo apt install -y libgl1 libglib2.0-0`
> - `infer.py` เรียก **Native NCNN (`ncnn.Net`)** โดยตรงและสลับไป **ONNX Runtime (`best.onnx`)** ให้อัตโนมัติหากไม่มี `ncnn` (ส่วน `train.py` ออกแบบสำหรับรันเทรนบน Google Colab GPU)

---

## 2. การรันทำนายผลแบบ Offline (`infer.py`)

```bash
python infer.py --input <test_folder/> --weights weights/ --out outputs/
```

สร้างไฟล์ครบทั้ง 4 ส่วนใน `--out`: `seg_mask/<stem>.png` (`uint16`), `predictions.json`, `mqtt/<stem>.json`, และ `vis/<image_name>`

---

## 3. การตั้งค่าและรันระบบ Integration (`client.py` — HTTP + MQTT)

> ⚠️ **คำเตือน Credentials (ห้ามสลับกัน):**
> - **`DAY3_HTTP_TOKEN`** = **Password ล็อกอินเว็บไซต์ของทีม** (ใส่ใน `Authorization: Bearer <TOKEN>`)
> - **`DAY3_MQTT_KEY`** = **MQTT Key ที่กด Generate จากหน้า Dashboard / My Team** (ใช้เป็น MQTT Password และ Topic `/event/<MQTT_KEY>/EdgeAI/Tesa2026-EdgeAI`)

### สร้างไฟล์ `.env` ไว้ในโฟลเดอร์ `team_01_day3/` (ไม่ควร commit ขึ้น Git)

```ini
DAY3_HTTP_URL=http://<SERVER_IP>:8000
DAY3_HTTP_TOKEN=YOUR_TEAM_WEBSITE_PASSWORD
DAY3_TEAM_ID=TGR-01
DAY3_MQTT_KEY=YOUR_GENERATED_MQTT_KEY
DAY3_MQTT_CLIENT_ID=TGR-01:device-1
DAY3_MQTT_HOST=tesa-score.c333m.site
DAY3_MQTT_PORT=443
DAY3_MQTT_WS_PATH=/mqtt/
DAY3_TRACK=EdgeAI
DAY3_EVENT=Tesa2026-EdgeAI
```

### คำสั่งใช้งาน `client.py`

```bash
# 1. เช็กการเชื่อมต่อและดูโควตาทีม (GET /api/v1/stats)
python client.py --stats

# 2. รอบซ้อมระบบ (19/11 เวลา 15:00-16:00 และ 20:00-22:00) — ทดสอบ 1 ภาพก่อน
python client.py --mode train --count 1 --clean-out

# 3. รอบซ้อมระบบ — ดึงซ้อมหลายภาพต่อเนื่อง
python client.py --mode train --count 5 --clean-out

# 4. รอบสอบจริง (20/11 เวลา 12:00-12:15) — ล้าง outputs/ และรันครบตามโควตา test พร้อมสร้างไฟล์ส่งงาน
python client.py --mode test --all --clean-out --build-zip ../team_01_Day3.zip
```

### กลไกการลด Latency (`duration_seconds`) ใน `client.py`
1. **Pre-load & Warm-up โมเดล + เปิด HTTP Keep-Alive + เชื่อมต่อ MQTT WSS ล่วงหน้า** ก่อนเรียก `GET /api/v1/image`
2. **ถอดรหัสภาพใน RAM (`io.BytesIO`)** พร้อมปรับ EXIF ทันทีโดยไม่เขียนลง SD Card ก่อน
3. **ส่ง `POST /api/v1/submit` และ `MQTT publish (QoS 1)` พร้อมกันทันที** ผ่าน `ThreadPoolExecutor` แล้วค่อยบันทึกไฟล์ `seg_mask/*.png`, `vis/*.jpg`, `mqtt/*.json`, `predictions.json` ลงดิสก์หลังเซิร์ฟเวอร์ตอบ `200 OK`
