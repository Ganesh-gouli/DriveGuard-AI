# backend/app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
from datetime import datetime
from database import add_alert, read_alerts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALERT_FILE = os.path.join(BASE_DIR, "static", "alerts.json")
LIVE_DATA_FILE = os.path.join(BASE_DIR, "static", "live_data.json")

# Ensure live data file exists
if not os.path.exists(LIVE_DATA_FILE):
    with open(LIVE_DATA_FILE, "w") as f:
        json.dump({"eyes": 0, "yawn": 0, "head_tilt": 0, "overall": 0, "timestamp": ""}, f)

app = Flask(__name__, template_folder="templates", static_folder="static")

def read_alerts():
    if not os.path.exists(ALERT_FILE):
        return []
    with open(ALERT_FILE, "r") as f:
        try:
            return json.load(f)
        except:
            return []

def write_alert(alert):
    alerts = read_alerts()
    alerts.insert(0, alert)  # newest first
    # keep last 200 records
    alerts = alerts[:200]
    with open(ALERT_FILE, "w") as f:
        json.dump(alerts, f, indent=2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/alerts", methods=["GET"])
def get_alerts():
    return jsonify(read_alerts())

@app.route("/live", methods=["GET"])
def get_live_data():
    try:
        with open(LIVE_DATA_FILE, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({"eyes": 0, "yawn": 0, "head_tilt": 0, "overall": 0, "timestamp": ""})

@app.route("/alert", methods=["POST"])
def post_alert():
    data = request.get_json(force=True)
    # expected keys: driver, status, timestamp, extra(optional)
    if "timestamp" not in data:
        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_alert(data)
    print("Received alert:", data)
    return jsonify({"ok": True}), 200

# serve demo static files if needed
@app.route("/static/<path:p>")
def static_files(p):
    return send_from_directory(os.path.join(BASE_DIR, "static"), p)

if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    # ensure alerts.json exists
    if not os.path.exists(ALERT_FILE):
        with open(ALERT_FILE, "w") as f:
            json.dump([], f)
    app.run(host="0.0.0.0", port=5000, debug=True)
