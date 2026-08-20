import json
from datetime import datetime
from pathlib import Path

ALERT_FILE = Path("backend/static/alerts.json")


def load_alerts():
    """Load existing alert logs from alerts.json."""
    if not ALERT_FILE.exists():
        return []
    try:
        with open(ALERT_FILE, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []


def save_alerts(alerts):
    """Save alert list back to alerts.json."""
    with open(ALERT_FILE, "w") as file:
        json.dump(alerts, file, indent=4)


def log_event(driver, status):
    """
    Add a new log entry to alerts.json.
    
    Example entry:
    {
        "driver": "Ganesh",
        "status": "Drowsy",
        "time": "2025-11-26 14:05:32"
    }
    """
    alerts = load_alerts()
    
    new_entry = {
        "driver": driver,
        "status": status,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    alerts.append(new_entry)
    save_alerts(alerts)

    return new_entry
