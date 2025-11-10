#!/usr/bin/env python3
"""Simple test script for Arduino servo communication."""
import json
import os
import random
import time
from arduino import ServoBus

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
    print(f"Port: {cfg['servo']['port']}, Baud: {cfg['servo']['baud']}")
    servo.open()
    print("✅ Connected! Sending test angles (Ctrl+C to stop)...\n")
    print("Waiting 2 seconds for Arduino to initialize...")
    time.sleep(2)
    
    # Test with known angles first
    print("Testing with fixed angles first...")
    test_angles = [
        (20, 15, 15),  # Servo 1 up
        (15, 20, 15),  # Servo 2 up
        (15, 15, 20),  # Servo 3 up
        (10, 15, 15),  # Servo 1 down
        (15, 10, 15),  # Servo 2 down
        (15, 15, 10),  # Servo 3 down
    ]
    
    for a1, a2, a3 in test_angles:
        print(f"Sending: ({a1}°, {a2}°, {a3}°)")
        servo.send_angles(a1, a2, a3)
        time.sleep(1.0)
        input("Did servos move? Press Enter to continue...")
    
    print("\nNow sending random angles...")
    for i in range(100):
        # Random angles between 0 and 30 degrees
        a1 = random.randint(0, 30)
        a2 = random.randint(0, 30)
        a3 = random.randint(0, 30)
        
        print(f"[{i+1}] Sending: ({a1}°, {a2}°, {a3}°)")
        servo.send_angles(a1, a2, a3)
        time.sleep(0.5)
        
except KeyboardInterrupt:
    print("\n\nStopped by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\nCleaning up...")
    try:
        print("Returning to neutral...")
        servo.level()
        time.sleep(0.5)
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")
    finally:
        servo.close()
        print("✅ Port closed. Done.")
