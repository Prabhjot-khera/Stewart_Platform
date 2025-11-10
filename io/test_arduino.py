#!/usr/bin/env python3
"""Simple test script for Arduino servo communication."""
import json
import os
import random
import time
from io.arduino import ServoBus

# Load config
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_path = os.path.join(project_root, "config", "config.json")

with open(config_path, "r") as f:
    cfg = json.load(f)

# Initialize servo
servo = ServoBus(
    port=cfg["servo"]["port"],
    baud=cfg["servo"]["baud"],
    neutral_deg=cfg["servo"]["theta0_deg"]
)

try:
    print("Connecting to Arduino...")
    servo.open()
    print("Connected! Sending random angles (Ctrl+C to stop)...\n")
    
    for i in range(100):
        # Random angles between 0 and 30 degrees
        a1 = random.randint(0, 30)
        a2 = random.randint(0, 30)
        a3 = random.randint(0, 30)
        
        print(f"[{i+1}] Sending: ({a1}°, {a2}°, {a3}°)")
        servo.send_angles(a1, a2, a3)
        time.sleep(0.5)
        
except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("Returning to neutral...")
    servo.level()
    time.sleep(0.5)
    servo.close()
    print("Done.")
