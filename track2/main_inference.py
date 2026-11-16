#!/usr/bin/env python3
"""
Main Real-Time Coffee Bean Quality Inspection Pipeline
Target System 1: Ubuntu 26.04 LTS | AMD Ryzen 5 5500U (6C/12T, AVX2) | 8GB RAM
Target System 2: Raspberry Pi 5 | ARM Cortex-A76 (4C, NEON) | 16GB RAM

Default Model: v1/coffee_best.pt
MQTT Integration: Publishes per-bean predictions to 'tni/tesa/task2' or 'tni/tesa/task3'
matching the Node-RED + SQLite server schema in flows.json.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from opencv_analyzer import BeanOpenCVAnalyzer, CLASS_NAMES, get_sca_category
from mqtt_client import (
    AsyncMQTTPublisher,
    build_nodered_prediction_payload,
    build_frame_payload,
)


def configure_cpu_environment(num_threads: int):
    """Optimizes CPU thread affinity and math libraries for Ryzen 5500U / Cortex-A76."""
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    cv2.setNumThreads(max(2, num_threads // 2))


def parse_args():
    parser = argparse.ArgumentParser(description="Coffee Bean Defect Detection Edge AI Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="v1/coffee_best.pt",
        help="Path to PyTorch .pt model (default: v1/coffee_best.pt), .onnx, or NCNN folder",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g., 0), video path, image file, or directory of images",
    )
    parser.add_argument(
        "--source-label",
        type=str,
        default="edge_camera",
        help="Value for 'source' field in Node-RED SQLite table (e.g., 'webcam', 'task2_cam')",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference input resolution (default: 640)")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="CPU threads (Recommended: 6 for Ryzen 5500U physical cores, 4 for RPi 5)",
    )
    parser.add_argument("--device-id", type=str, default="ryzen5500u-ubuntu-01", help="Edge Node Device ID")
    parser.add_argument("--session-id", type=str, default="lot_2026_001", help="Batch / Lot Session ID")
    parser.add_argument("--mqtt-host", type=str, default="localhost", help="MQTT Broker hostname/IP (e.g. localhost or 10.10.10.20)")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT Broker port (default: 1883)")
    parser.add_argument(
        "--mqtt-topic",
        type=str,
        default="tni/tesa/task2",
        help="MQTT Topic matching Node-RED flow ('tni/tesa/task2' or 'tni/tesa/task3')",
    )
    parser.add_argument(
        "--mqtt-interval",
        type=float,
        default=0.0,
        help="Minimum seconds between MQTT publishes in live video/webcam mode (0 = every frame)",
    )
    parser.add_argument("--enable-mqtt", action="store_true", help="Enable live MQTT publishing to Node-RED")
    parser.add_argument("--save-json-log", type=str, default="", help="Optional path to append JSONL telemetry")
    parser.add_argument("--no-display", action="store_true", help="Run headless without OpenCV GUI window")
    return parser.parse_args()


def draw_hud(frame: np.ndarray, payload: Dict[str, Any], mqtt_topic: str, mqtt_enabled: bool) -> np.ndarray:
    """Draws bounding boxes, center coordinates (bbox.x, bbox.y), SCA badges, and telemetry HUD."""
    vis = frame.copy()
    perf = payload["performance"]
    summary = payload["frame_summary"]

    for det in payload["detections"]:
        b = det["bbox"]
        x1, y1, x2, y2 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
        cx, cy = int(round(b["x"])), int(round(b["y"]))
        is_cat1 = det["sca_category"].startswith("Category 1")
        color = (0, 0, 255) if is_cat1 else (0, 180, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)

        lab = det["opencv_metrics"]["color_lab"]
        morph = det["opencv_metrics"]["morphology"]
        label_top = f"[{det['class_id']}] {det['class_name']} ({det['confidence']*100:.0f}%)"
        label_bot = f"xy:({b['x']:.0f},{b['y']:.0f}) L*:{lab['L_mean']:.0f} Sol:{morph['solidity']:.2f}"

        cv2.putText(vis, label_top, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(vis, label_bot, (x1, min(vis.shape[0] - 6, y2 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # Top-left status bar
    overlay = vis.copy()
    cv2.rectangle(overlay, (8, 8), (430, 122), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    mqtt_status = f"ON ({mqtt_topic})" if mqtt_enabled else "OFF"
    hud_lines = [
        f"FPS: {perf['fps']:.1f} (Total: {perf['total_pipeline_ms']:.1f} ms)",
        f"YOLO (.pt): {perf['ncnn_inference_ms']:.1f} ms | OpenCV: {perf['opencv_feature_ms']:.1f} ms",
        f"Beans: {summary['total_beans_detected']} | Cat1: {summary['cat1_primary_defects']} | Cat2: {summary['cat2_secondary_defects']}",
        f"MQTT: {mqtt_status} | CPU Temp: {perf['cpu_temp_c']:.1f} C",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(vis, line, (18, 32 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180), 1, cv2.LINE_AA)

    return vis


def process_single_frame(
    frame: np.ndarray,
    frame_id: int,
    image_name: str,
    model: YOLO,
    analyzer: BeanOpenCVAnalyzer,
    args,
    fps_ema: float,
) -> Dict[str, Any]:
    """Executes Hybrid YOLO Detection (.pt) + OpenCV Feature Extraction on one frame."""
    t_start = time.perf_counter()
    h, w = frame.shape[:2]

    # 1. Primary Detector: YOLO11n (.pt)
    t_yolo_start = time.perf_counter()
    results = model.predict(
        source=frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
        device="cpu",
    )
    yolo_ms = (time.perf_counter() - t_yolo_start) * 1000.0

    # 2. Secondary Feature Extractor: OpenCV per-bean ROI analysis
    t_cv_start = time.perf_counter()
    detections: List[Dict[str, Any]] = []
    r = results[0]
    model_names = r.names if hasattr(r, "names") and r.names else {i: n for i, n in enumerate(CLASS_NAMES)}

    if r.boxes is not None and len(r.boxes) > 0:
        xyxy_arr = r.boxes.xyxy.cpu().numpy()
        xywhn_arr = r.boxes.xywhn.cpu().numpy()
        conf_arr = r.boxes.conf.cpu().numpy()
        cls_arr = r.boxes.cls.cpu().numpy().astype(int)

        for idx in range(len(xyxy_arr)):
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy_arr[idx]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cx_n, cy_n, bw_n, bh_n = [float(v) for v in xywhn_arr[idx]]
            conf_val = float(conf_arr[idx])
            cls_id = int(cls_arr[idx])
            cls_name = model_names.get(cls_id, f"class_{cls_id}")

            roi = frame[y1:y2, x1:x2]
            cv_res = analyzer.analyze_roi(roi, cls_name)

            det_payload = build_nodered_prediction_payload(
                source_name=args.source_label,
                image_name=image_name,
                bean_id=idx + 1,
                class_id=cls_id,
                class_name=cls_name,
                sca_category=get_sca_category(cls_name),
                confidence=conf_val,
                x_min=x1,
                y_min=y1,
                x_max=x2,
                y_max=y2,
                cx_norm=cx_n,
                cy_norm=cy_n,
                w_norm=bw_n,
                h_norm=bh_n,
                opencv_metrics=cv_res["opencv_metrics"],
                rule_validation=cv_res["rule_validation"],
            )
            detections.append(det_payload)

    opencv_ms = (time.perf_counter() - t_cv_start) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    return build_frame_payload(
        device_id=args.device_id,
        session_id=args.session_id,
        source_name=args.source_label,
        image_name=image_name,
        frame_id=frame_id,
        fps=fps_ema,
        yolo_ms=yolo_ms,
        opencv_ms=opencv_ms,
        total_ms=total_ms,
        detections=detections,
    )


def main():
    args = parse_args()
    configure_cpu_environment(args.threads)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Please check the path (e.g. v1/coffee_best.pt)."
        )

    print(f"[INIT] Loading model: {model_path} (CPU Threads: {args.threads})")
    model = YOLO(str(model_path), task="detect")
    analyzer = BeanOpenCVAnalyzer()

    mqtt_pub = AsyncMQTTPublisher(
        broker_host=args.mqtt_host,
        broker_port=args.mqtt_port,
        topic=args.mqtt_topic,
        device_id=args.device_id,
        session_id=args.session_id,
        qos=1,
        enabled=args.enable_mqtt,
    )

    # Warmup inference with dummy tensor to fuse Conv+BN layers in .pt model
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model.predict(source=dummy, imgsz=args.imgsz, verbose=False, device="cpu")
    print("[INIT] Model warmup complete. Starting inspection pipeline...")

    source_path = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    is_image_dir = source_path.is_dir()
    is_single_image = source_path.is_file() and source_path.suffix.lower() in exts

    if is_image_dir or is_single_image:
        image_files = (
            [source_path]
            if is_single_image
            else sorted([p for p in source_path.rglob("*") if p.suffix.lower() in exts])
        )
        print(f"[INFO] Running inspection on {len(image_files)} image(s)...")
        fps_ema = 30.0
        for frame_id, img_p in enumerate(image_files, start=1):
            frame = cv2.imread(str(img_p))
            if frame is None:
                continue
            payload = process_single_frame(
                frame=frame,
                frame_id=frame_id,
                image_name=img_p.name,
                model=model,
                analyzer=analyzer,
                args=args,
                fps_ema=fps_ema,
            )
            inst_fps = 1000.0 / max(payload["performance"]["total_pipeline_ms"], 1e-3)
            fps_ema = 0.85 * fps_ema + 0.15 * inst_fps
            payload["performance"]["fps"] = round(fps_ema, 2)

            # Publish each detected bean to Node-RED MQTT topic (tni/tesa/task2 or task3)
            mqtt_pub.publish_frame_detections(payload)

            if args.save_json_log:
                with open(args.save_json_log, "a", encoding="utf-8") as f:
                    for det in payload["detections"]:
                        f.write(json.dumps(det, ensure_ascii=False) + "\n")

            print(
                f"[FRAME {frame_id}/{len(image_files)}] {img_p.name} -> "
                f"Detected {len(payload['detections'])} bean(s) in {payload['performance']['total_pipeline_ms']:.1f} ms"
            )

            if not args.no_display:
                vis = draw_hud(frame, payload, args.mqtt_topic, args.enable_mqtt)
                cv2.imshow("Coffee Bean Inspection (Press Q to quit)", vis)
                wait_ms = 0 if is_single_image else 1
                if cv2.waitKey(wait_ms) & 0xFF in (ord("q"), 27):
                    break
    else:
        cam_src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(cam_src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.imgsz)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.imgsz)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video/camera source: {args.source}")

        frame_id = 0
        fps_ema = 0.0
        last_mqtt_time = 0.0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
                img_name = f"frame_{frame_id:06d}.jpg"
                payload = process_single_frame(
                    frame=frame,
                    frame_id=frame_id,
                    image_name=img_name,
                    model=model,
                    analyzer=analyzer,
                    args=args,
                    fps_ema=fps_ema,
                )
                inst_fps = 1000.0 / max(payload["performance"]["total_pipeline_ms"], 1e-3)
                fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)
                payload["performance"]["fps"] = round(fps_ema, 2)

                now = time.monotonic()
                if now - last_mqtt_time >= args.mqtt_interval and payload["detections"]:
                    mqtt_pub.publish_frame_detections(payload)
                    last_mqtt_time = now

                if args.save_json_log and payload["detections"]:
                    with open(args.save_json_log, "a", encoding="utf-8") as f:
                        for det in payload["detections"]:
                            f.write(json.dumps(det, ensure_ascii=False) + "\n")

                if not args.no_display:
                    vis = draw_hud(frame, payload, args.mqtt_topic, args.enable_mqtt)
                    cv2.imshow("Coffee Bean Inspection (Press Q to quit)", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        finally:
            cap.release()

    cv2.destroyAllWindows()
    mqtt_pub.close()
    print("[INFO] Pipeline shut down cleanly.")


if __name__ == "__main__":
    main()
