"""
Asynchronous Non-Blocking MQTT Telemetry Publisher
Aligned with Node-RED Server Schema (flows.json -> yolo_predictions SQLite table)
Topics supported: 'tni/tesa/task2', 'tni/tesa/task3' (QoS 1)
"""

import json
import queue
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List
import psutil

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


class AsyncMQTTPublisher:
    """
    Background MQTT publisher using a bounded thread-safe queue.
    Publishes each detected coffee bean in the exact JSON format expected by
    Node-RED (flows.json -> function 3 / function 4 -> SQLite yolo_predictions).
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic: str = "tni/tesa/task2",
        device_id: str = "ryzen5500u-node-01",
        session_id: str = "lot_default",
        qos: int = 1,
        enabled: bool = True,
        max_queue_size: int = 200,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.device_id = device_id
        self.session_id = session_id
        self.qos = qos
        self.enabled = enabled and MQTT_AVAILABLE

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._connected = False
        self._client = None

        if self.enabled:
            self._client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"{self.device_id}_pub",
            )
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._worker = threading.Thread(target=self._publish_loop, daemon=True)
            self._worker.start()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self._connected = (reason_code == 0)
        if self._connected:
            print(
                f"[MQTT] Connected to {self.broker_host}:{self.broker_port} | "
                f"Target Topic: '{self.topic}' (QoS {self.qos})"
            )

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self._connected = False

    def _publish_loop(self):
        try:
            self._client.connect_async(self.broker_host, self.broker_port, keepalive=60)
            self._client.loop_start()
        except Exception as exc:
            print(f"[MQTT WARNING] Initial connection failed: {exc}")

        while not self._stop_event.is_set():
            try:
                msg_item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._connected and self._client is not None:
                try:
                    target_topic, payload_dict = msg_item
                    payload_json = json.dumps(payload_dict, ensure_ascii=False)
                    self._client.publish(target_topic, payload_json, qos=self.qos)
                except Exception as exc:
                    print(f"[MQTT ERROR] Publish failed: {exc}")
            self._queue.task_done()

    def publish_prediction(self, prediction_payload: Dict[str, Any], topic: str = ""):
        """
        Enqueues a single bean prediction payload for Node-RED SQLite insertion.
        Matches flows.json:
          data.source, data.image_name, data.class_id, data.class_name,
          data.confidence, data.bbox.x, data.bbox.y
        """
        if not self.enabled:
            return
        target_topic = topic or self.topic
        try:
            self._queue.put_nowait((target_topic, prediction_payload))
        except queue.Full:
            # Drop oldest item if broker is congested to protect 8GB RAM limit
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((target_topic, prediction_payload))
            except queue.Empty:
                pass

    def publish_frame_detections(self, frame_payload: Dict[str, Any], topic: str = ""):
        """Publishes every detected bean in the frame as an individual MQTT message for Node-RED."""
        if not self.enabled:
            return
        for det in frame_payload.get("detections", []):
            self.publish_prediction(det, topic=topic)

    def close(self):
        """Flushes remaining queued messages briefly and disconnects cleanly."""
        deadline = datetime.now().timestamp() + 2.0
        while not self._queue.empty() and datetime.now().timestamp() < deadline:
            threading.Event().wait(0.05)
        self._stop_event.set()
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()


def get_cpu_temperature() -> float:
    """Reads CPU package/core temperature on Ubuntu Linux (k10temp for Ryzen / cpu_thermal for RPi5)."""
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return 0.0
        for sensor_name in ("k10temp", "coretemp", "cpu_thermal", "acpitz"):
            if sensor_name in temps and temps[sensor_name]:
                return round(float(temps[sensor_name][0].current), 1)
        first_group = next(iter(temps.values()))
        if first_group:
            return round(float(first_group[0].current), 1)
    except Exception:
        pass
    return 0.0


def build_nodered_prediction_payload(
    source_name: str,
    image_name: str,
    bean_id: int,
    class_id: int,
    class_name: str,
    sca_category: str,
    confidence: float,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    cx_norm: float,
    cy_norm: float,
    w_norm: float,
    h_norm: float,
    opencv_metrics: Dict[str, Any],
    rule_validation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Constructs a single detection JSON object that is 100% compatible with
    Node-RED flows.json (function 3 / function 4) while preserving OpenCV metrics.

    Node-RED SQL extraction mapping:
      - data.source       -> source (TEXT NOT NULL)
      - data.image_name   -> image_name (TEXT)
      - data.class_id     -> class_id (INTEGER NOT NULL)
      - data.class_name   -> class_name (TEXT NOT NULL)
      - data.confidence   -> confidence (REAL NOT NULL)
      - data.bbox.x       -> bbox_x (REAL) [Center X in pixels]
      - data.bbox.y       -> bbox_y (REAL) [Center Y in pixels]
    """
    center_x_px = round((x_min + x_max) / 2.0, 2)
    center_y_px = round((y_min + y_max) / 2.0, 2)

    return {
        "source": str(source_name),
        "image_name": str(image_name),
        "bean_id": int(bean_id),
        "class_id": int(class_id),
        "class_name": str(class_name),
        "confidence": round(float(confidence), 4),
        "bbox": {
            "x": center_x_px,
            "y": center_y_px,
            "w": round(float(x_max - x_min), 2),
            "h": round(float(y_max - y_min), 2),
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
            "cx_norm": round(float(cx_norm), 4),
            "cy_norm": round(float(cy_norm), 4),
            "w_norm": round(float(w_norm), 4),
            "h_norm": round(float(h_norm), 4),
        },
        "sca_category": sca_category,
        "opencv_metrics": opencv_metrics,
        "rule_validation": rule_validation,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    }


def build_frame_payload(
    device_id: str,
    session_id: str,
    source_name: str,
    image_name: str,
    frame_id: int,
    fps: float,
    yolo_ms: float,
    opencv_ms: float,
    total_ms: float,
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Builds the complete frame telemetry object for HUD display and local JSONL logging."""
    cat1_count = 0
    cat2_count = 0
    dist: Dict[str, int] = {}

    for det in detections:
        cls_name = det["class_name"]
        dist[cls_name] = dist.get(cls_name, 0) + 1
        if det["sca_category"].startswith("Category 1"):
            cat1_count += 1
        else:
            cat2_count += 1

    return {
        "schema_version": "1.1.0",
        "device_id": device_id,
        "session_id": session_id,
        "source": source_name,
        "image_name": image_name,
        "frame_id": frame_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "performance": {
            "fps": round(fps, 2),
            "ncnn_inference_ms": round(yolo_ms, 2),
            "opencv_feature_ms": round(opencv_ms, 2),
            "total_pipeline_ms": round(total_ms, 2),
            "cpu_temp_c": get_cpu_temperature(),
        },
        "frame_summary": {
            "total_beans_detected": len(detections),
            "cat1_primary_defects": cat1_count,
            "cat2_secondary_defects": cat2_count,
            "defect_distribution": dist,
        },
        "detections": detections,
    }
