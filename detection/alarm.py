# detection/alarm.py
import os
import threading
import time

ALARM_FILE = os.path.join(os.path.dirname(__file__), "alarm.mp3")

# Global flag for continuous alarm
_alarm_active = False
_alarm_thread = None

def _play_sound_fallback():
    # simple fallback beep (may not work in all terminals)
    print("\a", end="", flush=True)

def _continuous_loop():
    global _alarm_active
    while _alarm_active:
        try:
            from playsound import playsound
            if os.path.exists(ALARM_FILE):
                playsound(ALARM_FILE)
            else:
                import winsound
                winsound.Beep(2500,2500) # 500ms beep
                time.sleep(0.1)
        except Exception:
            # Fallback
            import winsound
            try:
                winsound.Beep(2500, 500)
            except:
                _play_sound_fallback()
            time.sleep(0.1)

def start_continuous_alarm():
    """Starts the alarm in a non-blocking continuous loop if not already running."""
    global _alarm_active, _alarm_thread
    if _alarm_active:
        return # Already running

    _alarm_active = True
    _alarm_thread = threading.Thread(target=_continuous_loop, daemon=True)
    _alarm_thread.start()

def stop_continuous_alarm():
    """Stops the continuous alarm loop."""
    global _alarm_active, _alarm_thread
    _alarm_active = False
    if _alarm_thread:
        _alarm_thread.join(timeout=1.0)
        _alarm_thread = None

def play_alarm():
    """Legacy single-shot alarm (kept for backward compatibility if needed)"""
    try:
        from playsound import playsound
        if os.path.exists(ALARM_FILE):
            t = threading.Thread(target=playsound, args=(ALARM_FILE,), daemon=True)
            t.start()
        else:
            import winsound
            winsound.Beep(2500, 1000)
    except Exception:
        pass
def play_tts_alert(message):
    """Plays a TTS alert."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
    except ImportError:
        print(f"TTS Alert: {message} (pyttsx3 not installed)")
    except Exception as e:
        print(f"TTS Error: {e}")

def buzz_for_duration(seconds):
    """Buzzes for a specific duration."""
    try:
        import winsound
        # Buzz for 'seconds' duration (approx)
        # winsound.Beep blocks, so we might want to run this in a thread if blocking is bad.
        # But for 2 seconds it might be acceptable or we use a loop of short beeps.
        end_time = time.time() + seconds
        while time.time() < end_time:
            winsound.Beep(2500, 200) # 200ms beep
            time.sleep(0.1)
    except Exception:
        _play_sound_fallback()
