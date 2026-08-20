import json
from pathlib import Path
from datetime import datetime

# Path to alerts.json inside backend/static
ALERT_FILE = Path("backend/static/alerts.json")


def init_db():
    """Create alerts.json file if it doesn't exist."""
    ALERT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not ALERT_FILE.exists():
        with open(ALERT_FILE, "w") as file:
            json.dump([], file, indent=4)


def read_alerts():
    """Read all alerts from alerts.json."""
    init_db()
    try:
        with open(ALERT_FILE, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []


def add_alert(driver, status):
    """
    Add a new alert log entry.

    Returns the created entry.
    """
    alerts = read_alerts()

    entry = {
        "driver": driver,
        "status": status,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    alerts.append(entry)

    with open(ALERT_FILE, "w") as file:
        json.dump(alerts, file, indent=4)

    return entry
