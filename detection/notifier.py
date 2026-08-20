# detection/notifier.py
import requests
import json
import time

BACKEND_URL = "http://localhost:5000/alert"  # change if backend remote

def send_alert(driver="Unknown", status="Drowsy", extra=None):
    payload = {
        "driver": driver,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "extra": extra or {}
    }
    try:
        res = requests.post(BACKEND_URL, json=payload, timeout=3)
        return res.ok
    except Exception as e:
        # print locally if server unavailable
        print("Notifier error:", e)
        return False
