import time

class StateManager:
    def __init__(self):
        self.start_time = time.time()
        self.break_alert_triggered = False
        self.BREAK_THRESHOLD_SECONDS = 1.5 * 60 * 60  # 1.5 hours
        # self.BREAK_THRESHOLD_SECONDS = 10 # Debug: 10 seconds

    def get_driving_stats(self):
        current_time = time.time()
        duration_seconds = int(current_time - self.start_time)
        
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        
        remaining_seconds = max(0, self.BREAK_THRESHOLD_SECONDS - duration_seconds)
        remaining_minutes = int(remaining_seconds // 60)
        
        should_trigger_break = duration_seconds >= self.BREAK_THRESHOLD_SECONDS
        
        return {
            "driving_time_str": f"{hours}h {minutes}m",
            "driving_minutes": minutes + (hours * 60),
            "remaining_break_min": remaining_minutes,
            "break_recommended": should_trigger_break
        }

    def reset_timer(self):
        self.start_time = time.time()
        self.break_alert_triggered = False
