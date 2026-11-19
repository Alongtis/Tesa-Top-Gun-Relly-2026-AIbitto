#!/usr/bin/env python3
"""
mock_server.py — Local Competition Mock Server (HTTP Evaluation + MQTT-over-WebSocket Broker)
จำลองเซิร์ฟเวอร์กรรมการ Day 3 ตาม API_DOCS.md 100% บนเครื่อง PC สำหรับให้ Raspberry Pi ยิงทดสอบข้าม LAN

รองรับครบทั้ง 2 บริการในสคริปต์เดียว (ไม่ต้องลงแพ็กเกจเสริม ใช้ Python มาตรฐาน + OpenCV/NumPy ที่มีอยู่แล้ว):
  1. HTTP Evaluation Server (port 8000):
     - GET  /                  : ตรวจสถานะเซิร์ฟเวอร์ (เช็ก Firewall จาก Pi ได้ทันที)
     - GET  /api/v1/stats      : ดูสถิติและโควตาของทีม
     - GET  /api/v1/image      : แจกภาพจริงจาก Task/Dataset_Day-3/Images พร้อม X-Image-ID, X-Image-Name, X-Mode
     - POST /api/v1/submit     : ตรวจ Schema, จับเวลา duration_seconds จริง, และคำนวณคะแนนสด (IoU/Counting/Class/Latency)
  2. Mock MQTT-over-WebSocket Broker (ws://0.0.0.0:8083/mqtt/, รองรับ MQTTv5 + QoS 1 PUBACK):
     - ตรวจ Username (TEAM_01) + Password (MQTT_KEY) + Topic (/event/<MQTT_KEY>/EdgeAI/1)
     - ตรวจโครงสร้าง JSON บรรทัดเดียว (image, int latency_ms, counts ครบ 7 คลาส, defect_ratio)
     - เทียบ counts กับผลที่ส่งทาง HTTP POST แบบเรียลไทม์ (เหมือนเกณฑ์ MQTT 5 คะแนน หน้า 12)

วิธีรันบนเครื่อง PC:
    python track2/day-3/Task/mock_server.py
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import math
import socket
import struct
import sys
import threading
import time
import urllib.parse
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "Dataset_Day-3"
CLASSES = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]
VALID_CLASSES_SET = set(CLASSES)
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def get_lan_ips() -> List[str]:
    ips: List[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and not ip.startswith("169.254.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips or ["127.0.0.1"]


def score_latency_day3(duration_sec: float) -> int:
    """เกณฑ์หักคะแนน Latency Day 3 สไลด์หน้า 13 (เต็ม 25, เกิน 1.00s หักขั้นละ 5)"""
    n = max(0, int(math.ceil(duration_sec - 1.0))) if duration_sec > 1.0 else 0
    return max(0, 25 - 5 * n)


def evaluate_polygons_vs_gt(
    pred_objs: List[Dict[str, Any]],
    gt_objs: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> Optional[Dict[str, Any]]:
    """คำนวณ IoU (P1/15), Counting (P2/15) และ Class Accuracy เทียบกับ labels.json"""
    if not HAS_CV2:
        return None
    n_gt, n_pr = len(gt_objs), len(pred_objs)
    if n_gt == 0 and n_pr == 0:
        return {"mean_iou": 1.0, "counting": 1.0, "cls_ok": 0, "n_gt": 0, "n_pr": 0, "p1": 15.0, "p2": 15.0}

    gt_mask = np.zeros((img_h, img_w), dtype=np.uint16)
    pr_mask = np.zeros((img_h, img_w), dtype=np.uint16)
    gt_cls: Dict[int, str] = {}
    pr_cls: Dict[int, str] = {}

    for i, o in enumerate(gt_objs, start=1):
        pts = np.array(o["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(gt_mask, [pts], -1, i, thickness=-1)
        gt_cls[i] = o["class"]
    for i, o in enumerate(pred_objs, start=1):
        pts = np.array(o["polygon"], dtype=np.int32).reshape((-1, 1, 2))
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
    matched_ious: List[float] = []
    cls_ok = 0
    for iou, gid, pid in pairs:
        if gid in ug or pid in up:
            continue
        ug.add(gid)
        up.add(pid)
        matched_ious.append(iou)
        if gt_cls[gid] == pr_cls[pid]:
            cls_ok += 1

    unmatched = (n_gt - len(ug)) + (n_pr - len(up))
    avg_iou = sum(matched_ious) / float(max(len(matched_ious) + unmatched, 1))
    counting = len(ug) / float(max(n_gt, n_pr, 1))
    return {
        "mean_iou": avg_iou,
        "counting": counting,
        "cls_ok": cls_ok,
        "n_gt": n_gt,
        "n_pr": n_pr,
        "p1": 15.0 * avg_iou,
        "p2": 15.0 * counting,
    }


class MockState:
    def __init__(
        self,
        dataset_dir: Path,
        http_token: str,
        team_id: str,
        mqtt_key: str,
        track: str,
        event: str,
        active_mode: str,
        test_quota: int,
        timeout_sec: float,
    ) -> None:
        self.lock = threading.Lock()
        self.http_token = http_token
        self.team_id = team_id
        self.mqtt_key = mqtt_key
        self.track = track
        self.event = event
        self.active_mode = active_mode
        self.test_quota = test_quota
        self.timeout_sec = timeout_sec

        img_dir = dataset_dir / "Images"
        self.images: List[Path] = sorted(
            [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")],
            key=lambda p: p.name,
        )
        if not self.images:
            raise FileNotFoundError(f"ไม่พบไฟล์ภาพใน {img_dir}")

        self.gt_by_name: Dict[str, List[Dict[str, Any]]] = {}
        self.img_wh_by_name: Dict[str, Tuple[int, int]] = {}
        lbl_path = dataset_dir / "labels.json"
        if lbl_path.exists():
            for entry in json.loads(lbl_path.read_text(encoding="utf-8")):
                self.gt_by_name[entry["image_name"]] = entry["objects"]

        self.train_idx = 0
        self.test_idx = 0
        self.train_requested = 0
        self.train_submitted = 0
        self.test_requested = 0
        self.test_submitted = 0

        self.issued: Dict[str, Dict[str, Any]] = {}
        self.submitted_ids: set[str] = set()
        self.http_counts_by_image: Dict[str, Dict[str, int]] = {}
        self.mqtt_counts_by_image: Dict[str, Dict[str, int]] = {}

    def get_img_wh(self, img_path: Path) -> Tuple[int, int]:
        if img_path.name not in self.img_wh_by_name and HAS_CV2:
            im = cv2.imread(str(img_path))
            if im is not None:
                self.img_wh_by_name[img_path.name] = (im.shape[1], im.shape[0])
        return self.img_wh_by_name.get(img_path.name, (3072, 4096))

    def check_mqtt_http_match(self, image_name: str) -> Optional[bool]:
        hc = self.http_counts_by_image.get(image_name)
        mc = self.mqtt_counts_by_image.get(image_name)
        if hc is None or mc is None:
            return None
        return hc == mc


# =====================================================================
# 1. HTTP Evaluation Server (ThreadingHTTPServer)
# =====================================================================

def make_http_handler(state: MockState):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send_json(self, code: int, obj: Dict[str, Any]) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _check_auth(self) -> bool:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {state.http_token}":
                print(f"[HTTP 401] Unauthorized from {self.client_address[0]} | got Authorization={auth!r}")
                self._send_json(401, {"detail": "Invalid Bearer token"})
                return False
            return True

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            if path == "/" or path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "online",
                        "service": "TESA Day 3 Local Mock Evaluation Server",
                        "active_mode": state.active_mode,
                        "dataset_images": len(state.images),
                    },
                )
                return

            if not self._check_auth():
                return

            if path == "/api/v1/stats":
                with state.lock:
                    body = {
                        "team_name": state.team_id,
                        "train_mode": {
                            "images_requested": state.train_requested,
                            "submissions_received": state.train_submitted,
                        },
                        "test_mode": {
                            "quota_limit": state.test_quota,
                            "images_requested": state.test_requested,
                            "submissions_received": state.test_submitted,
                            "remaining_attempts": max(0, state.test_quota - state.test_requested),
                        },
                    }
                print(f"[HTTP GET /api/v1/stats] from {self.client_address[0]} -> 200 OK")
                self._send_json(200, body)
                return

            if path == "/api/v1/image":
                req_mode = (qs.get("mode", [state.active_mode])[0] or state.active_mode).lower()
                if req_mode not in ("train", "test"):
                    self._send_json(400, {"detail": f"Invalid mode: {req_mode}"})
                    return

                with state.lock:
                    if req_mode == "test":
                        if state.test_requested >= state.test_quota:
                            print(f"[HTTP 403] Test quota exceeded ({state.test_quota}/{state.test_quota}) from {self.client_address[0]}")
                            self._send_json(403, {"detail": "Test mode quota limit reached"})
                            return
                        img_p = state.images[state.test_idx % len(state.images)]
                        state.test_idx += 1
                        state.test_requested += 1
                    else:
                        img_p = state.images[state.train_idx % len(state.images)]
                        state.train_idx += 1
                        state.train_requested += 1

                    img_id = f"img_{uuid.uuid4().hex[:16]}"
                    raw = img_p.read_bytes()

                ctype = "image/png" if img_p.suffix.lower() == ".png" else "image/jpeg"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("X-Image-ID", img_id)
                self.send_header("X-Image-Name", img_p.name)
                self.send_header("X-Mode", req_mode)
                self.end_headers()
                self.wfile.write(raw)
                self.wfile.flush()

                # จับเวลาหลังส่งไฟล์ภาพเสร็จสมบูรณ์ตาม API_DOCS.md บรรทัด 118
                sent_done_t = time.perf_counter()
                with state.lock:
                    state.issued[img_id] = {
                        "image_name": img_p.name,
                        "image_path": img_p,
                        "mode": req_mode,
                        "sent_at": sent_done_t,
                    }
                print(
                    f"[HTTP GET /api/v1/image] -> {self.client_address[0]} | "
                    f"mode={req_mode} | id={img_id} | file={img_p.name} ({len(raw)/1024:.0f} KB)"
                )
                return

            self._send_json(404, {"detail": f"Unknown endpoint: {path}"})

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if not self._check_auth():
                return

            if path == "/api/v1/reset":
                with state.lock:
                    state.train_idx = state.test_idx = 0
                    state.train_requested = state.train_submitted = 0
                    state.test_requested = state.test_submitted = 0
                    state.issued.clear()
                    state.submitted_ids.clear()
                    state.http_counts_by_image.clear()
                    state.mqtt_counts_by_image.clear()
                print(f"[HTTP POST /api/v1/reset] State reset by {self.client_address[0]}")
                self._send_json(200, {"status": "reset_ok"})
                return

            if path == "/api/v1/submit":
                recv_t = time.perf_counter()
                clen = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(clen)
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception as e:
                    self._send_json(400, {"detail": f"Invalid JSON: {e}"})
                    return

                img_id = body.get("img_id", "")
                preds = body.get("predictions")
                if not isinstance(preds, dict) or "image_name" not in preds or not isinstance(preds.get("objects"), list):
                    self._send_json(422, {"detail": "predictions must be {image_name, objects: [...]}"})
                    return

                with state.lock:
                    if img_id not in state.issued:
                        self._send_json(404, {"detail": f"Unknown img_id: {img_id}"})
                        return
                    if img_id in state.submitted_ids:
                        print(f"[HTTP 409] Duplicate img_id={img_id} from {self.client_address[0]}")
                        self._send_json(409, {"detail": "Duplicate submission for img_id"})
                        return
                    rec = state.issued[img_id]
                    duration_sec = round(recv_t - rec["sent_at"], 3)
                    if state.timeout_sec > 0 and duration_sec > state.timeout_sec:
                        print(f"[HTTP 408] Timeout ({duration_sec}s > {state.timeout_sec}s) for {img_id}")
                        self._send_json(408, {"detail": f"Submission exceeded {state.timeout_sec}s"})
                        return

                    # ตรวจ schema ของ objects ตาม API_DOCS.md
                    objs = preds["objects"]
                    for o in objs:
                        if set(o.keys()) != {"id", "class", "polygon"}:
                            self._send_json(422, {"detail": f"Invalid object keys: {list(o.keys())}"})
                            return
                        if o["class"] not in VALID_CLASSES_SET:
                            self._send_json(422, {"detail": f"Invalid class: {o['class']}"})
                            return
                        if not isinstance(o["polygon"], list) or len(o["polygon"]) < 3:
                            self._send_json(422, {"detail": f"Polygon of id={o['id']} must have >= 3 points"})
                            return

                    state.submitted_ids.add(img_id)
                    if rec["mode"] == "test":
                        state.test_submitted += 1
                    else:
                        state.train_submitted += 1

                    c_map = {c: 0 for c in CLASSES}
                    for o in objs:
                        c_map[o["class"]] += 1
                    state.http_counts_by_image[rec["image_name"]] = c_map
                    match_flag = state.check_mqtt_http_match(rec["image_name"])

                # ตอบ 200 OK กลับไปให้ Pi ทันที (แล้วค่อยคำนวณ IoU โชว์บนจอ PC)
                resp_obj = {
                    "status": "success",
                    "message": "Submission recorded successfully",
                    "img_id": img_id,
                    "image_name": preds["image_name"],
                    "mode": rec["mode"],
                    "duration_seconds": duration_sec,
                    "received_at": datetime.now(timezone.utc).isoformat(),
                }
                self._send_json(200, resp_obj)

                # คำนวณคะแนนสดบนฝั่ง PC
                l3 = score_latency_day3(duration_sec)
                gt_objs = state.gt_by_name.get(rec["image_name"])
                grade_str = ""
                if gt_objs is not None:
                    iw, ih = state.get_img_wh(rec["image_path"])
                    sc = evaluate_polygons_vs_gt(objs, gt_objs, iw, ih)
                    if sc is not None:
                        cls_pct = (sc["cls_ok"] / max(sc["n_gt"], 1)) * 100.0
                        grade_str = (
                            f" | P1(IoU)={sc['p1']:.2f}/15 ({sc['mean_iou']:.4f})"
                            f" | P2(Cnt)={sc['p2']:.2f}/15 ({sc['n_pr']}/{sc['n_gt']})"
                            f" | Cls={sc['cls_ok']}/{sc['n_gt']} ({cls_pct:.1f}%)"
                        )
                mqtt_chk = "" if match_flag is None else (" | MQTT-vs-HTTP: MATCH 5/5!" if match_flag else " | MQTT-vs-HTTP: MISMATCH!")
                print(
                    f"[HTTP POST /api/v1/submit] 200 OK | {rec['image_name']} | "
                    f"duration={duration_sec:.3f}s -> L3={l3}/25{grade_str}{mqtt_chk}"
                )
                return

            self._send_json(404, {"detail": f"Unknown endpoint: {path}"})

    return Handler


# =====================================================================
# 2. Pure-Python Mock MQTT-over-WebSocket Broker (RFC 6455 + MQTTv5)
# =====================================================================

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _ws_read_frame(sock: socket.socket) -> Tuple[int, bytes]:
    hdr = _recv_exact(sock, 2)
    opcode = hdr[0] & 0x0F
    masked = (hdr[1] & 0x80) != 0
    length = hdr[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    mask_key = _recv_exact(sock, 4) if masked else b""
    payload = _recv_exact(sock, length)
    if masked:
        payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def _ws_send_frame(sock: socket.socket, payload: bytes, opcode: int = 0x2) -> None:
    length = len(payload)
    header = bytearray([0x80 | opcode])
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    sock.sendall(bytes(header) + payload)


def _decode_varint(data: bytes, pos: int) -> Tuple[int, int]:
    mult = 1
    val = 0
    while True:
        b = data[pos]
        pos += 1
        val += (b & 0x7F) * mult
        if (b & 0x80) == 0:
            break
        mult *= 128
    return val, pos


def _read_utf8_str(data: bytes, pos: int) -> Tuple[str, int]:
    slen = struct.unpack("!H", data[pos : pos + 2])[0]
    pos += 2
    return data[pos : pos + slen].decode("utf-8", errors="replace"), pos + slen


def handle_mqtt_ws_client(conn: socket.socket, addr: Tuple[str, int], state: MockState) -> None:
    proto_ver = 5
    try:
        conn.settimeout(130.0)
        # 1. HTTP WebSocket Upgrade Handshake
        req_buf = b""
        while b"\r\n\r\n" not in req_buf and len(req_buf) < 16384:
            chunk = conn.recv(4096)
            if not chunk:
                return
            req_buf += chunk

        header_raw, leftover = req_buf.split(b"\r\n\r\n", 1)
        header_lines = header_raw.decode("utf-8", errors="replace").split("\r\n")
        req_line = header_lines[0] if header_lines else ""
        hdrs: Dict[str, str] = {}
        for line in header_lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                hdrs[k.strip().lower()] = v.strip()

        ws_key = hdrs.get("sec-websocket-key", "")
        if not ws_key:
            conn.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return

        accept = base64.b64encode(hashlib.sha1((ws_key + WS_GUID).encode("ascii")).digest()).decode("ascii")
        proto_hdr = ""
        req_proto = hdrs.get("sec-websocket-protocol", "")
        if "mqtt" in req_proto.lower():
            first_p = req_proto.split(",")[0].strip()
            proto_hdr = f"Sec-WebSocket-Protocol: {first_p}\r\n"

        resp = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            f"{proto_hdr}\r\n"
        ).encode("ascii")
        conn.sendall(resp)

        # 2. MQTT Packet Loop over WebSocket Binary Frames
        stream_buf = bytearray()
        if leftover:
            # เผื่อกรณีมีเฟรมต่อท้ายมากับ handshake
            pass
        while True:
            opcode, frame_data = _ws_read_frame(conn)
            if opcode == 0x8:  # Close
                _ws_send_frame(conn, b"", opcode=0x8)
                break
            if opcode == 0x9:  # Ping
                _ws_send_frame(conn, frame_data, opcode=0xA)
                continue
            if opcode not in (0x2, 0x0):
                continue

            stream_buf.extend(frame_data)
            while len(stream_buf) >= 2:
                # ตรวจว่าได้ packet ครบตาม remaining length หรือยัง
                try:
                    rem_len, hdr_end = _decode_varint(bytes(stream_buf), 1)
                except IndexError:
                    break
                total_len = hdr_end + rem_len
                if len(stream_buf) < total_len:
                    break

                pkt = bytes(stream_buf[:total_len])
                del stream_buf[:total_len]

                pkt_type = (pkt[0] >> 4) & 0x0F
                pkt_flags = pkt[0] & 0x0F
                body = pkt[hdr_end:]

                if pkt_type == 1:  # CONNECT
                    pos = 0
                    _, pos = _read_utf8_str(body, pos)  # "MQTT"
                    proto_ver = body[pos]
                    pos += 1
                    connect_flags = body[pos]
                    pos += 1
                    pos += 2  # keepalive
                    if proto_ver == 5:
                        prop_len, pos = _decode_varint(body, pos)
                        pos += prop_len
                    client_id, pos = _read_utf8_str(body, pos)
                    if connect_flags & 0x04:  # Will flag
                        if proto_ver == 5:
                            w_prop_len, pos = _decode_varint(body, pos)
                            pos += w_prop_len
                        _, pos = _read_utf8_str(body, pos)
                        w_len = struct.unpack("!H", body[pos : pos + 2])[0]
                        pos += 2 + w_len
                    username = ""
                    password = ""
                    if connect_flags & 0x80:
                        username, pos = _read_utf8_str(body, pos)
                    if connect_flags & 0x40:
                        plen = struct.unpack("!H", body[pos : pos + 2])[0]
                        pos += 2
                        password = body[pos : pos + plen].decode("utf-8", errors="replace")

                    auth_ok = (username == state.team_id) and (password == state.mqtt_key)
                    if auth_ok:
                        connack = b"\x20\x03\x00\x00\x00" if proto_ver == 5 else b"\x20\x02\x00\x00"
                        _ws_send_frame(conn, connack)
                        print(f"[MQTT CONNECT] OK from {addr[0]} | user={username} | proto=v{proto_ver} | req='{req_line}'")
                    else:
                        connack = b"\x20\x03\x00\x86\x00" if proto_ver == 5 else b"\x20\x02\x00\x04"
                        _ws_send_frame(conn, connack)
                        print(f"[MQTT CONNECT] REJECTED from {addr[0]} | got user={username!r}, key={password!r}")
                        return

                elif pkt_type == 3:  # PUBLISH
                    qos = (pkt_flags >> 1) & 0x03
                    pos = 0
                    topic, pos = _read_utf8_str(body, pos)
                    pkt_id = 0
                    if qos > 0:
                        pkt_id = struct.unpack("!H", body[pos : pos + 2])[0]
                        pos += 2
                    if proto_ver == 5:
                        prop_len, pos = _decode_varint(body, pos)
                        pos += prop_len
                    payload_bytes = body[pos:]

                    if qos == 1:
                        puback = (
                            b"\x40\x04" + struct.pack("!H", pkt_id) + b"\x00\x00"
                            if proto_ver == 5
                            else b"\x40\x02" + struct.pack("!H", pkt_id)
                        )
                        _ws_send_frame(conn, puback)

                    expected_topic = f"/event/{state.mqtt_key}/{state.track}/{state.event}"
                    topic_ok = topic == expected_topic
                    raw_text = payload_bytes.decode("utf-8", errors="replace")
                    single_line = "\n" not in raw_text.strip()
                    try:
                        data = json.loads(raw_text)
                        img_name = data.get("image", "")
                        lat_ok = isinstance(data.get("latency_ms"), int)
                        counts = data.get("counts", {})
                        keys_ok = isinstance(counts, dict) and set(counts.keys()) == VALID_CLASSES_SET
                        with state.lock:
                            if keys_ok and img_name:
                                state.mqtt_counts_by_image[img_name] = {c: int(counts[c]) for c in CLASSES}
                            match_flag = state.check_mqtt_http_match(img_name) if img_name else None

                        match_str = "" if match_flag is None else (" | HTTP-vs-MQTT: MATCH 5/5!" if match_flag else " | HTTP-vs-MQTT: MISMATCH!")
                        print(
                            f"[MQTT PUBLISH] QoS={qos} | topic={'OK' if topic_ok else 'WRONG(' + topic + ')'} | "
                            f"1-line={'OK' if single_line else 'NO'} | int_lat={'OK' if lat_ok else 'NO'} | "
                            f"7-keys={'OK' if keys_ok else 'NO'} | image={img_name} | "
                            f"latency={data.get('latency_ms')}ms | defect_ratio={data.get('defect_ratio')}{match_str}"
                        )
                        print(f"               counts={counts}")
                    except Exception as je:
                        print(f"[MQTT PUBLISH] INVALID JSON from {addr[0]}: {je} | raw={raw_text[:200]}")

                elif pkt_type == 12:  # PINGREQ
                    _ws_send_frame(conn, b"\xd0\x00")

                elif pkt_type == 14:  # DISCONNECT
                    print(f"[MQTT DISCONNECT] Clean disconnect from {addr[0]}")
                    return
    except (ConnectionError, TimeoutError, OSError):
        pass
    except Exception as e:
        print(f"[MQTT WARN] {addr[0]}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def start_mqtt_ws_server(host: str, port: int, state: MockState) -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(16)

    def _accept_loop() -> None:
        while True:
            try:
                conn, addr = srv.accept()
                threading.Thread(target=handle_mqtt_ws_client, args=(conn, addr, state), daemon=True).start()
            except OSError:
                break

    threading.Thread(target=_accept_loop, daemon=True).start()
    return srv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 3 Local Mock HTTP + MQTT-WS Competition Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0 รับการเชื่อมต่อจาก Raspberry Pi ข้าม LAN)")
    parser.add_argument("--http-port", type=int, default=8000, help="HTTP Server Port (default: 8000)")
    parser.add_argument("--mqtt-port", type=int, default=8083, help="Mock MQTT WebSocket Port (default: 8083)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR, help="Path to Dataset_Day-3 folder")
    parser.add_argument("--http-token", default="mock_web_password_01", help="Mock Website Password (HTTP Bearer Token)")
    parser.add_argument("--team-id", default="TEAM_01", help="Team ID / MQTT Username (default: TEAM_01)")
    parser.add_argument("--mqtt-key", default="mock_mqtt_key_abcdef12", help="Mock MQTT Key (Password + Topic)")
    parser.add_argument("--track", default="EdgeAI", help="Track name (default: EdgeAI)")
    parser.add_argument("--event", default="1", help="Event number (default: 1)")
    parser.add_argument("--mode", choices=("train", "test"), default="train", help="Default active mode when client omits ?mode=")
    parser.add_argument("--test-quota", type=int, default=5, help="Quota limit in test mode (default: 5)")
    parser.add_argument("--timeout", type=float, default=0.0, help="Optional HTTP 408 timeout threshold in seconds (0 = disabled)")
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    state = MockState(
        dataset_dir=args.dataset,
        http_token=args.http_token,
        team_id=args.team_id,
        mqtt_key=args.mqtt_key,
        track=args.track,
        event=args.event,
        active_mode=args.mode,
        test_quota=args.test_quota,
        timeout_sec=args.timeout,
    )

    httpd = http.server.ThreadingHTTPServer((args.host, args.http_port), make_http_handler(state))
    mqtt_sock = start_mqtt_ws_server(args.host, args.mqtt_port, state)
    lan_ips = get_lan_ips()
    primary_ip = next((ip for ip in lan_ips if ip.startswith("192.168.0.") or ip.startswith("192.168.1.") or ip.startswith("10.")), lan_ips[0])

    print("=" * 78)
    print("TESA Top Gun 2026 (Day 3) — Local Mock HTTP + MQTT-WS Competition Server")
    print("=" * 78)
    print(f"  HTTP Server     : http://{primary_ip}:{args.http_port}  (All IPs: {', '.join(lan_ips)})")
    print(f"  MQTT-WS Broker  : ws://{primary_ip}:{args.mqtt_port}/mqtt/")
    print(f"  Dataset Loaded  : {len(state.images)} images | GT labels: {len(state.gt_by_name)} | Live IoU Grader: {'ON' if HAS_CV2 else 'OFF'}")
    print(f"  Active Mode     : {args.mode} | Test Quota: {args.test_quota}")
    print("-" * 78)
    print("  [Mock Credentials สำหรับใส่ใน .env บน Raspberry Pi]")
    print(f"    DAY3_HTTP_URL=http://{primary_ip}:{args.http_port}")
    print(f"    DAY3_HTTP_TOKEN={args.http_token}")
    print(f"    DAY3_TEAM_ID={args.team_id}")
    print(f"    DAY3_MQTT_KEY={args.mqtt_key}")
    print(f"    DAY3_MQTT_HOST={primary_ip}")
    print(f"    DAY3_MQTT_PORT={args.mqtt_port}")
    print("    DAY3_MQTT_WS_PATH=/mqtt/")
    print(f"    DAY3_TRACK={args.track}")
    print(f"    DAY3_EVENT={args.event}")
    print("=" * 78)
    print("Waiting for requests from Raspberry Pi... (Ctrl+C to stop)\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down mock server...")
    finally:
        httpd.server_close()
        mqtt_sock.close()


if __name__ == "__main__":
    main()
