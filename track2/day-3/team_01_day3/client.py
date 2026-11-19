#!/usr/bin/env python3
"""
client.py — Day 3 Integrated HTTP + MQTT Client (ในสคริปต์เดียว)
ตรงตามข้อกำหนดใน API_DOCS.md และสไลด์ Day 3 หน้า 6–10

ลำดับการทำงานเพื่อรีด Latency (duration_seconds) ให้ต่ำที่สุดบน Raspberry Pi 5:
  0. Pre-load & Warm-up โมเดล (NCNN / ONNX) + เปิด HTTP Session (Keep-Alive) + เชื่อม MQTT WSS ล่วงหน้า
  1. GET /api/v1/image -> รับไบนารีภาพ + Header (X-Image-ID, X-Image-Name, X-Mode)
  2. ถอดรหัสภาพใน RAM (EXIF-corrected) + รัน predict_image() ทันที (ไม่รอเขียนดิสก์)
  3. ส่งพร้อมกันแบบ Concurrent (ThreadPoolExecutor):
     - Task A (HTTP): POST /api/v1/submit (ส่ง img_id + predictions {image_name, objects})
     - Task B (MQTT): publish JSON บรรทัดเดียวขึ้น wss://tesa-score.c333m.site:443/mqtt/ (QoS 1 + wait_for_publish)
  4. หลังเซิร์ฟเวอร์ตอบ 200 OK แล้ว จึงบันทึกไฟล์ลง outputs/ (seg_mask/, predictions.json, mqtt/, vis/)
     เพื่อไม่ให้เวลาเขียน SD Card ไปบวกเพิ่มใน duration_seconds ของเซิร์ฟเวอร์

คำเตือน Credentials (ห้ามสลับกัน):
  - HTTP_TOKEN (DAY3_HTTP_TOKEN) = Password ที่ใช้ล็อกอินเว็บไซต์ของทีม (Authorization: Bearer <TOKEN>)
  - MQTT_KEY   (DAY3_MQTT_KEY)   = MQTT Key ที่กด Generate จากหน้า Dashboard / My Team
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from infer import (  # noqa: E402
    VALID_CLASSES_SET,
    build_mqtt_payload,
    decode_image_bytes_exif_corrected,
    format_predictions_json,
    load_model,
    predict_image,
    render_vis_image,
)

DEFAULT_MQTT_HOST = "tesa-score.c333m.site"
DEFAULT_MQTT_PORT = 443
DEFAULT_MQTT_WS_PATH = "/mqtt/"
DEFAULT_TEAM_ID = "TGR-01"
DEFAULT_MQTT_CLIENT_ID = "TGR-01:device-1"
DEFAULT_TRACK = "EdgeAI"
DEFAULT_EVENT = "1"


# =====================================================================
# 1. โหลดค่า Config จาก .env (ไม่ต้องพึ่งแพ็กเกจเสริม)
# =====================================================================

def load_dotenv_file(env_path: Path) -> None:
    """อ่านไฟล์ .env แบบ key=value เข้า os.environ (ไม่เขียนทับค่าที่มีอยู่แล้วใน environment)"""
    if not env_path.exists() or not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
            val = val[1:-1]
        if key:
            os.environ.setdefault(key, val)


# =====================================================================
# 2. HTTP Client (ใช้ requests.Session Keep-Alive + fallback urllib)
# =====================================================================

class CompetitionHTTPClient:
    """HTTP Client แบบ Persistent Session สำหรับเชื่อมต่อ Evaluation Server ด้วย Bearer Token"""

    def __init__(self, base_url: str, bearer_token: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token
        self.timeout = timeout
        self.auth_headers = {"Authorization": f"Bearer {bearer_token}"}
        self._session: Any = None
        try:
            import requests  # type: ignore
            sess = requests.Session()
            sess.headers.update(self.auth_headers)
            self._session = sess
        except ImportError:
            self._session = None

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass

    def _urllib_request(
        self,
        method: str,
        url: str,
        body_bytes: Optional[bytes] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, str], bytes]:
        headers = dict(self.auth_headers)
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                return int(resp.status), hdrs, resp.read()
        except urllib.error.HTTPError as e:
            hdrs = {k.lower(): v for k, v in e.headers.items()} if e.headers else {}
            return int(e.code), hdrs, e.read()

    def get_stats(self) -> Tuple[int, Dict[str, Any]]:
        url = f"{self.base_url}/api/v1/stats"
        if self._session is not None:
            resp = self._session.get(url, timeout=self.timeout)
            data = self._safe_json(resp.content)
            return int(resp.status_code), data
        status, _, raw = self._urllib_request("GET", url)
        return status, self._safe_json(raw)

    def get_image(self, mode: str = "auto") -> Tuple[int, Dict[str, str], bytes]:
        url = f"{self.base_url}/api/v1/image"
        if mode in ("train", "test"):
            url = f"{url}?{urllib.parse.urlencode({'mode': mode})}"
        if self._session is not None:
            resp = self._session.get(url, timeout=self.timeout)
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            return int(resp.status_code), hdrs, resp.content
        return self._urllib_request("GET", url)

    def post_submit(self, img_id: str, predictions_obj: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        url = f"{self.base_url}/api/v1/submit"
        payload_bytes = json.dumps(
            {"img_id": img_id, "predictions": predictions_obj},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        hdrs = {"Content-Type": "application/json"}
        if self._session is not None:
            resp = self._session.post(url, data=payload_bytes, headers=hdrs, timeout=self.timeout)
            return int(resp.status_code), self._safe_json(resp.content)
        status, _, raw = self._urllib_request("POST", url, body_bytes=payload_bytes, extra_headers=hdrs)
        return status, self._safe_json(raw)

    @staticmethod
    def _safe_json(raw: bytes) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw.decode("utf-8", errors="replace"))
            return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except Exception:
            return {"raw_text": raw.decode("utf-8", errors="replace")[:400]}


def describe_http_error(status: int, detail: Any = "") -> str:
    hints = {
        401: "401 Unauthorized — Password ล็อกอินเว็บไซต์ (HTTP_TOKEN) ไม่ถูกต้อง (ห้ามเอา MQTT Key มาใส่ช่อง HTTP)",
        403: "403 Forbidden — โควตา Test เต็มแล้ว หรือเซิร์ฟเวอร์ยังไม่เปิดโหมดนี้",
        408: "408 Request Timeout — ส่งผล JSON ช้าเกินเวลาที่ผู้คุมสอบกำหนด",
        409: "409 Conflict — img_id นี้ถูกส่งไปแล้ว (ห้ามส่งภาพเดิมซ้ำ)",
        503: "503 Service Unavailable — เซิร์ฟเวอร์หยุดชั่วคราวโดยกรรมการ",
    }
    base = hints.get(status, f"HTTP {status}")
    return f"{base} | detail={detail}" if detail else base


# =====================================================================
# 3. MQTT WebSocket TLS Client (MQTTv5, Persistent Connection + QoS 1)
# =====================================================================

class CompetitionMQTTClient:
    """เชื่อมต่อ MQTT Broker แบบ WebSocket TLS (WSS) ล่วงหน้า เพื่อ Publish QoS 1 ได้ทันทีหลัง Inference"""

    def __init__(
        self,
        host: str,
        port: int,
        ws_path: str,
        team_id: str,
        mqtt_key: str,
        client_id: str = DEFAULT_MQTT_CLIENT_ID,
        track: str = DEFAULT_TRACK,
        event: str = DEFAULT_EVENT,
    ) -> None:
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError as e:
            raise RuntimeError("ยังไม่ได้ติดตั้ง paho-mqtt (กรุณารัน: pip install 'paho-mqtt>=2,<3')") from e

        self.host = host
        self.port = port
        self.ws_path = ws_path
        self.team_id = team_id
        self.mqtt_key = mqtt_key
        self.client_id = client_id
        self.topic = f"/event/{mqtt_key}/{track}/{event}"

        self._connected_evt = threading.Event()
        self._connect_rc: Optional[Any] = None

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            transport="websockets",
            protocol=mqtt.MQTTv5,
        )
        self.client.username_pw_set(team_id, mqtt_key)
        self.client.ws_set_options(path=ws_path)
        if port == 443:
            self.client.tls_set_context(ssl.create_default_context())

        def _on_connect(client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
            self._connect_rc = reason_code
            self._connected_evt.set()

        self.client.on_connect = _on_connect

    def connect(self, timeout: float = 10.0) -> None:
        self._connected_evt.clear()
        self.client.connect(self.host, self.port, keepalive=60)
        self.client.loop_start()
        if not self._connected_evt.wait(timeout=timeout):
            self.close()
            scheme = "wss" if self.port == 443 else "ws"
            raise TimeoutError(
                f"เชื่อมต่อ MQTT Broker ไม่สำเร็จภายใน {timeout:.0f}s "
                f"({scheme}://{self.host}:{self.port}{self.ws_path})"
            )
        if getattr(self._connect_rc, "is_failure", False):
            rc = self._connect_rc
            self.close()
            raise RuntimeError(
                f"MQTT Broker ปฏิเสธการเชื่อมต่อ (reason_code={rc}) — "
                f"ตรวจสอบ TEAM_ID ({self.team_id}), CLIENT_ID ({self.client_id}) และ MQTT_KEY (ต้องกด Generate จากหน้าเว็บ)"
            )

    def publish_payload(self, payload_dict: Dict[str, Any], timeout: float = 8.0) -> str:
        """Publish JSON บรรทัดเดียวด้วย QoS 1 และรอ PUBACK ยืนยันการส่ง"""
        one_line_json = json.dumps(payload_dict, ensure_ascii=False)
        info = self.client.publish(self.topic, one_line_json.encode("utf-8"), qos=1)
        info.wait_for_publish(timeout=timeout)
        if not info.is_published():
            raise RuntimeError(f"MQTT publish หมดเวลารอ PUBACK (rc={info.rc})")
        return one_line_json

    def close(self) -> None:
        try:
            self.client.loop_stop()
        except Exception:
            pass
        try:
            self.client.disconnect()
        except Exception:
            pass


# =====================================================================
# 4. บันทึกไฟล์ผลลัพธ์ลง outputs/ และสร้าง team_01_Day3.zip
# =====================================================================

def save_round_outputs(
    out_dir: Path,
    img_name: str,
    img_bgr: np.ndarray,
    seg_mask_u16: np.ndarray,
    objects: List[Dict[str, Any]],
    mqtt_payload: Dict[str, Any],
    all_predictions_by_name: Dict[str, Dict[str, Any]],
    save_vis: bool = True,
) -> None:
    """บันทึก seg_mask/<stem>.png (uint16), mqtt/<stem>.json, vis/<img_name>, และ predictions.json"""
    seg_mask_dir = out_dir / "seg_mask"
    mqtt_dir = out_dir / "mqtt"
    vis_dir = out_dir / "vis"
    seg_mask_dir.mkdir(parents=True, exist_ok=True)
    mqtt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(img_name).stem
    mask_path = seg_mask_dir / f"{stem}.png"
    if not cv2.imwrite(str(mask_path), seg_mask_u16.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 1]):
        raise RuntimeError(f"ไม่สามารถบันทึกไฟล์ seg_mask ได้: {mask_path}")

    (mqtt_dir / f"{stem}.json").write_text(
        json.dumps(mqtt_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if save_vis:
        vis_img = render_vis_image(img_bgr, seg_mask_u16, objects)
        cv2.imwrite(str(vis_dir / img_name), vis_img)

    all_predictions_by_name[img_name] = {"image_name": img_name, "objects": objects}
    ordered_list = [all_predictions_by_name[k] for k in sorted(all_predictions_by_name.keys())]
    (out_dir / "predictions.json").write_text(format_predictions_json(ordered_list), encoding="utf-8")


def build_submission_zip(team_dir: Path, zip_path: Path) -> Path:
    """แพ็กโฟลเดอร์ team_01_day3/ เป็น team_01_Day3.zip ตามโครงสร้างสไลด์หน้า 8–9 (ไม่รวม .env และ __pycache__)"""
    include_files = ["README.md", "requirements.txt", "train.py", "infer.py", "client.py"]
    include_dirs = ["weights", "outputs"]
    root_name = team_dir.name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fname in include_files:
            fp = team_dir / fname
            if fp.is_file():
                zf.write(fp, Path(root_name) / fname)
        for dname in include_dirs:
            dp = team_dir / dname
            if not dp.exists():
                continue
            for f in sorted(dp.rglob("*")):
                if not f.is_file():
                    continue
                if "__pycache__" in f.parts or f.name in (".gitkeep", ".env"):
                    continue
                zf.write(f, Path(root_name) / f.relative_to(team_dir))
    return zip_path


# =====================================================================
# 5. CLI & Main Execution Loop
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Team TGR-01 Day 3 Integrated HTTP + MQTT Client (API_DOCS.md)"
    )
    parser.add_argument("--env-file", type=Path, default=BASE_DIR / ".env", help="ไฟล์ .env สำหรับเก็บ credentials (default: .env)")
    parser.add_argument("--weights", type=Path, default=BASE_DIR / "weights", help="โฟลเดอร์ weights/ ของโมเดล Day 3")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "outputs", help="โฟลเดอร์ outputs/ สำหรับบันทึกผลลัพธ์")
    parser.add_argument("--http-url", type=str, default=None, help="URL ของ HTTP Evaluation Server (หรือ DAY3_HTTP_URL)")
    parser.add_argument("--http-token", type=str, default=None, help="Password ล็อกอินเว็บไซต์สำหรับ HTTP Bearer Token (หรือ DAY3_HTTP_TOKEN)")
    parser.add_argument("--team-id", type=str, default=None, help=f"รหัสทีมสำหรับ MQTT Username (default: {DEFAULT_TEAM_ID}, หรือ DAY3_TEAM_ID)")
    parser.add_argument("--mqtt-key", type=str, default=None, help="MQTT Key ที่ Generate จากหน้าเว็บ (หรือ DAY3_MQTT_KEY)")
    parser.add_argument("--mqtt-client-id", type=str, default=None, help=f"MQTT Client ID (default: {DEFAULT_MQTT_CLIENT_ID}, หรือ DAY3_MQTT_CLIENT_ID)")
    parser.add_argument("--mqtt-host", type=str, default=None, help=f"MQTT Broker Host (default: {DEFAULT_MQTT_HOST})")
    parser.add_argument("--mqtt-port", type=int, default=None, help=f"MQTT Broker Port (default: {DEFAULT_MQTT_PORT})")
    parser.add_argument("--mqtt-ws-path", type=str, default=None, help=f"MQTT WebSocket Path (default: {DEFAULT_MQTT_WS_PATH})")
    parser.add_argument("--track", type=str, default=None, help=f"ชื่อ Track แบบ Case-sensitive (default: {DEFAULT_TRACK})")
    parser.add_argument("--event", type=str, default=None, help=f"รหัส Event (default: {DEFAULT_EVENT})")
    parser.add_argument("--mode", choices=("auto", "train", "test"), default="auto", help="โหมดดึงภาพ: auto (อิงเซิร์ฟเวอร์), train, หรือ test")
    parser.add_argument("--count", type=int, default=1, help="จำนวนภาพที่ต้องการดึงและส่งผล (default: 1)")
    parser.add_argument("--all", action="store_true", help="ดึงจนครบโควตา test (remaining_attempts) หรือสูงสุด 16 ภาพในโหมด train")
    parser.add_argument("--delay", type=float, default=0.0, help="หน่วงเวลาระหว่างรอบ (วินาที, default: 0.0)")
    parser.add_argument("--stats", action="store_true", help="เรียก GET /api/v1/stats เพื่อดูสถานะ/โควตาอย่างเดียวแล้วจบ")
    parser.add_argument("--no-mqtt", action="store_true", help="ข้ามการ publish ขึ้น MQTT Broker (ยังบันทึก outputs/mqtt/<stem>.json)")
    parser.add_argument("--no-vis", action="store_true", help="ไม่บันทึกภาพ overlay ใน outputs/vis/")
    parser.add_argument("--clean-out", action="store_true", help="ล้างไฟล์เก่าใน outputs/ (seg_mask, mqtt, vis, predictions.json) ก่อนเริ่มรอบ")
    parser.add_argument("--save-raw-dir", type=Path, default=None, help="บันทึกภาพดิบที่โหลดจาก HTTP ไว้ในโฟลเดอร์แยก (นอก outputs/)")
    parser.add_argument("--build-zip", type=Path, default=None, help="สร้างไฟล์ submission ZIP หลังรันเสร็จ เช่น ../team_01_Day3.zip")
    return parser.parse_args()


def clean_outputs_dir(out_dir: Path) -> None:
    for sub in ("seg_mask", "mqtt", "vis"):
        dp = out_dir / sub
        if dp.exists():
            for f in dp.iterdir():
                if f.is_file() and f.name != ".gitkeep":
                    f.unlink()
    pred_f = out_dir / "predictions.json"
    if pred_f.exists():
        pred_f.write_text("[]\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    load_dotenv_file(args.env_file)

    http_url = (args.http_url or os.environ.get("DAY3_HTTP_URL") or "").strip()
    http_token = (args.http_token or os.environ.get("DAY3_HTTP_TOKEN") or "").strip()
    team_id = (args.team_id or os.environ.get("DAY3_TEAM_ID") or DEFAULT_TEAM_ID).strip()
    mqtt_key = (args.mqtt_key or os.environ.get("DAY3_MQTT_KEY") or "").strip()
    mqtt_client_id = (args.mqtt_client_id or os.environ.get("DAY3_MQTT_CLIENT_ID") or DEFAULT_MQTT_CLIENT_ID).strip()
    mqtt_host = (args.mqtt_host or os.environ.get("DAY3_MQTT_HOST") or DEFAULT_MQTT_HOST).strip()
    mqtt_port = args.mqtt_port if args.mqtt_port is not None else int(os.environ.get("DAY3_MQTT_PORT", str(DEFAULT_MQTT_PORT)))
    mqtt_ws_path = (args.mqtt_ws_path or os.environ.get("DAY3_MQTT_WS_PATH") or DEFAULT_MQTT_WS_PATH).strip()
    track = (args.track or os.environ.get("DAY3_TRACK") or DEFAULT_TRACK).strip()
    event = (args.event or os.environ.get("DAY3_EVENT") or DEFAULT_EVENT).strip()

    if not http_url:
        print("[ERROR] กรุณาระบุ --http-url หรือตั้งค่า DAY3_HTTP_URL ใน .env", file=sys.stderr)
        sys.exit(1)
    if not http_token:
        print("[ERROR] กรุณาระบุ --http-token หรือตั้งค่า DAY3_HTTP_TOKEN (Password ล็อกอินเว็บไซต์ของทีม)", file=sys.stderr)
        sys.exit(1)
    if not args.stats and not args.no_mqtt and not mqtt_key:
        print("[ERROR] กรุณาระบุ --mqtt-key หรือตั้งค่า DAY3_MQTT_KEY (Generate จากหน้าเว็บ) หรือใช้ --no-mqtt เพื่อทดสอบเฉพาะ HTTP", file=sys.stderr)
        sys.exit(1)
    if mqtt_key and http_token == mqtt_key:
        print("[WARN] HTTP_TOKEN และ MQTT_KEY มีค่าเหมือนกัน — โปรดตรวจเช็กว่าไม่ได้สลับ Password เว็บไซต์กับ MQTT Key!", file=sys.stderr)

    http_client = CompetitionHTTPClient(http_url, http_token)
    mqtt_client: Optional[CompetitionMQTTClient] = None

    try:
        # 1. ตรวจสอบ /api/v1/stats ก่อนเสมอ (ยืนยัน Token และดูโควตา)
        st_code, stats_data = http_client.get_stats()
        if st_code != 200:
            print(f"[ERROR] GET /api/v1/stats ล้มเหลว: {describe_http_error(st_code, stats_data)}", file=sys.stderr)
            sys.exit(1)

        print(f"[STATS] {json.dumps(stats_data, ensure_ascii=False)}")
        if args.stats:
            sys.exit(0)

        # คำนวณจำนวนรอบที่จะรัน
        target_rounds = max(1, int(args.count))
        if args.all:
            if args.mode == "test":
                rem = int(stats_data.get("test_mode", {}).get("remaining_attempts", 0))
                if rem <= 0:
                    print("[INFO] โควตา test_mode เหลือ 0 แล้ว — หยุดทำงานเพื่อไม่ให้ยิงเกินโควตา")
                    sys.exit(0)
                target_rounds = rem
                print(f"[MODE=TEST] จะรันทั้งหมด {target_rounds} ภาพตาม remaining_attempts")
            elif args.mode == "train":
                target_rounds = max(target_rounds, 16)
            else:
                rem = int(stats_data.get("test_mode", {}).get("remaining_attempts", 5))
                target_rounds = max(target_rounds, rem if rem > 0 else 5)

        if args.clean_out:
            clean_outputs_dir(args.out)

        # 2. Pre-load & Warm-up โมเดลล่วงหน้า (ก่อน GET /api/v1/image!)
        t_load = time.perf_counter()
        engine = load_model(args.weights)
        load_ms = (time.perf_counter() - t_load) * 1000.0
        print(
            f"[MODEL] {engine.get('engine_label', engine['backend'])} "
            f"| imgsz={engine['img_size']} | warmup_load={load_ms:.1f} ms"
        )

        # 3. Pre-connect MQTT ล่วงหน้า (ลดเวลา TLS/WebSocket handshake ออกจากช่วงประมวลผล!)
        if not args.no_mqtt:
            mqtt_client = CompetitionMQTTClient(
                host=mqtt_host,
                port=mqtt_port,
                ws_path=mqtt_ws_path,
                team_id=team_id,
                mqtt_key=mqtt_key,
                client_id=mqtt_client_id,
                track=track,
                event=event,
            )
            mqtt_client.connect()
            mqtt_scheme = "wss" if mqtt_port == 443 else "ws"
            print(f"[MQTT] Connected {mqtt_scheme}://{mqtt_host}:{mqtt_port}{mqtt_ws_path} | client_id={mqtt_client_id} | topic={mqtt_client.topic}")

        # โหลด predictions.json เดิมไว้ (ถ้าไม่ได้สั่ง --clean-out)
        all_predictions_by_name: Dict[str, Dict[str, Any]] = {}
        pred_json_path = args.out / "predictions.json"
        if pred_json_path.exists() and not args.clean_out:
            try:
                existing = json.loads(pred_json_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    for item in existing:
                        if isinstance(item, dict) and "image_name" in item:
                            all_predictions_by_name[item["image_name"]] = item
            except Exception:
                pass

        if args.save_raw_dir is not None:
            args.save_raw_dir.mkdir(parents=True, exist_ok=True)

        submitted_ids: set[str] = set()
        success_count = 0
        server_durations: List[float] = []
        local_latencies_ms: List[int] = []

        # 4. ลูปดึงภาพ -> Inference -> ส่ง HTTP + MQTT พร้อมกัน -> บันทึก outputs/
        with ThreadPoolExecutor(max_workers=2) as pool:
            for round_idx in range(1, target_rounds + 1):
                print(f"\n--- Round {round_idx}/{target_rounds} (mode={args.mode}) ---")
                g_status, g_headers, img_bytes = http_client.get_image(mode=args.mode)
                if g_status != 200:
                    err_msg = describe_http_error(g_status, CompetitionHTTPClient._safe_json(img_bytes))
                    if g_status == 403 and success_count > 0:
                        print(f"[STOP] {err_msg} (ส่งสำเร็จไปแล้ว {success_count} ภาพ)")
                        break
                    print(f"[ERROR] GET /api/v1/image ล้มเหลว: {err_msg}", file=sys.stderr)
                    sys.exit(1)

                img_id = g_headers.get("x-image-id", "").strip()
                img_name = g_headers.get("x-image-name", "").strip() or f"{img_id or f'image_{round_idx:02d}'}.jpg"
                x_mode = g_headers.get("x-mode", args.mode).strip()
                if not img_id:
                    print("[ERROR] Response Header ไม่มี X-Image-ID", file=sys.stderr)
                    sys.exit(1)
                if img_id in submitted_ids:
                    print(f"[WARN] ได้รับ img_id ซ้ำ ({img_id}) — ข้ามเพื่อป้องกัน 409 Conflict")
                    continue

                # [CRITICAL PATH] ถอดรหัสใน RAM + รันโมเดลทันที
                t0 = time.perf_counter()
                img_bgr = decode_image_bytes_exif_corrected(img_bytes, source_name=img_name)
                seg_mask_u16, objects = predict_image(engine, img_bgr)
                infer_ms = (time.perf_counter() - t0) * 1000.0

                for obj in objects:
                    if obj["class"] not in VALID_CLASSES_SET or len(obj["polygon"]) < 3:
                        raise RuntimeError(f"ผลลัพธ์ไม่ถูกต้องสำหรับ {img_name}: id={obj.get('id')} class={obj.get('class')}")

                pred_entry = {"image_name": img_name, "objects": objects}
                mqtt_payload = build_mqtt_payload(img_name, infer_ms, objects)

                # [CRITICAL PATH] ยิง HTTP POST + MQTT Publish พร้อมกันทันที (ก่อนเขียนลง SD Card!)
                fut_http = pool.submit(http_client.post_submit, img_id, pred_entry)
                fut_mqtt = (
                    pool.submit(mqtt_client.publish_payload, mqtt_payload)
                    if mqtt_client is not None
                    else None
                )

                p_status, p_data = fut_http.result()
                mqtt_ok_str = "SKIPPED (--no-mqtt)"
                if fut_mqtt is not None:
                    try:
                        fut_mqtt.result()
                        mqtt_ok_str = "OK (QoS 1 PUBACK)"
                    except Exception as me:
                        mqtt_ok_str = f"FAILED ({me})"
                        print(f"[ERROR] MQTT publish ล้มเหลวสำหรับ {img_name}: {me}", file=sys.stderr)

                if p_status != 200:
                    print(
                        f"[ERROR] POST /api/v1/submit ล้มเหลว ({img_name}, img_id={img_id}): "
                        f"{describe_http_error(p_status, p_data)}",
                        file=sys.stderr,
                    )
                    if p_status in (401, 403):
                        sys.exit(1)
                    continue

                submitted_ids.add(img_id)
                success_count += 1
                srv_dur = p_data.get("duration_seconds")
                if isinstance(srv_dur, (int, float)):
                    server_durations.append(float(srv_dur))
                local_latencies_ms.append(int(mqtt_payload["latency_ms"]))

                # [POST-SUBMIT I/O] บันทึกไฟล์ outputs/ หลังส่ง HTTP+MQTT สำเร็จแล้ว
                save_round_outputs(
                    out_dir=args.out,
                    img_name=img_name,
                    img_bgr=img_bgr,
                    seg_mask_u16=seg_mask_u16,
                    objects=objects,
                    mqtt_payload=mqtt_payload,
                    all_predictions_by_name=all_predictions_by_name,
                    save_vis=not args.no_vis,
                )
                if args.save_raw_dir is not None:
                    (args.save_raw_dir / img_name).write_bytes(img_bytes)

                print(
                    f"[OK] {img_name} (id={img_id}, mode={x_mode}) -> {len(objects)} beans "
                    f"| infer={mqtt_payload['latency_ms']} ms | server_dur={srv_dur} s "
                    f"| defect_ratio={mqtt_payload['defect_ratio']} | MQTT={mqtt_ok_str}"
                )
                print(f"     counts={mqtt_payload['counts']}")

                if args.delay > 0 and round_idx < target_rounds:
                    time.sleep(args.delay)

        # 5. สรุปผลและเช็ก /api/v1/stats ปิดท้าย
        print("\n" + "=" * 72)
        print(f"[SUMMARY] Submitted: {success_count}/{target_rounds} images -> {args.out}")
        if local_latencies_ms:
            print(
                f"  Local Infer Latency : median={np.median(local_latencies_ms):.1f} ms | "
                f"mean={np.mean(local_latencies_ms):.1f} ms | max={np.max(local_latencies_ms)} ms"
            )
        if server_durations:
            print(
                f"  Server duration_sec : median={np.median(server_durations):.3f} s | "
                f"mean={np.mean(server_durations):.3f} s | max={np.max(server_durations):.3f} s"
            )
        st_end, end_stats = http_client.get_stats()
        if st_end == 200:
            print(f"[FINAL STATS] {json.dumps(end_stats, ensure_ascii=False)}")

        if args.build_zip is not None:
            zp = build_submission_zip(BASE_DIR, args.build_zip)
            print(f"[ZIP] Created submission archive -> {zp} ({zp.stat().st_size / (1024*1024):.2f} MB)")
        print("=" * 72)

    finally:
        if mqtt_client is not None:
            mqtt_client.close()
        http_client.close()


if __name__ == "__main__":
    main()
