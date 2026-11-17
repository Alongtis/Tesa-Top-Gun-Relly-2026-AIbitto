#!/usr/bin/env python3
"""
Quick End-to-End Verification Script for Local Mockup Server (Mosquitto + Node-RED + SQLite)
1. Publishes a sample coffee bean prediction to 'tni/tesa/task2'
2. Queries 'http://localhost:1880/api/data' to verify SQLite insertion
"""

import json
import time
import urllib.request
import paho.mqtt.client as mqtt


def main():
    topic = "tni/tesa/task2"
    sample_payload = {
        "source": "ryzen5500u_test",
        "image_name": "verify_001.jpg",
        "class_id": 5,
        "class_name": "Full Black",
        "confidence": 0.965,
        "bbox": {
            "x": 320.5,
            "y": 240.0,
        },
    }

    print(f"[1/2] Publishing sample payload to localhost:1883 -> topic '{topic}'...")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="verify_tester")
    client.connect("localhost", 1883, keepalive=10)
    client.loop_start()
    info = client.publish(topic, json.dumps(sample_payload), qos=1)
    info.wait_for_publish(timeout=5.0)
    client.loop_stop()
    client.disconnect()
    print("      Published successfully!")

    time.sleep(1.0)
    api_url = "http://localhost:1880/api/data"
    print(f"[2/2] Checking Node-RED SQLite API at {api_url}...")
    try:
        with urllib.request.urlopen(api_url, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"      Success! Total rows in 'yolo_predictions': {len(data)}")
            if data:
                print("      Latest row:", json.dumps(data[-1], ensure_ascii=False))
    except Exception as exc:
        print(f"      Could not fetch from {api_url} yet: {exc}")
        print("      (Make sure flows.json is deployed and 'Create Table' was clicked in Node-RED)")


if __name__ == "__main__":
    main()
