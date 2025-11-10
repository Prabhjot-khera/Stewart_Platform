#!/usr/bin/env python3
"""
Stewart Platform Alignment Calibration Script

This script helps you align the 3 servos to the correct azimuth angles (0°, 120°, 240°).
It tests tilting in different directions to verify which servo corresponds to which azimuth.
"""
import json
import os
import math
import time
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io.arduino import ServoBus
from kinematics.ik_small import ThreeServoTiltIKSmall

def load_config():
    """Load config.json."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "config", "config.json")
    
    with open(config_path, "r") as f:
        return json.load(f)

def test_tilt_direction(servo, ik, direction_name, pitch_rad, roll_rad, hold_time=2.0):
    """Test tilting in a specific direction."""
    print(f"\n{'='*60}")
    print(f"Testing: {direction_name}")
    print(f"  Pitch: {math.degrees(pitch_rad):.2f}°  Roll: {math.degrees(roll_rad):.2f}°")
    
    # Calculate servo angles
    theta1, theta2, theta3 = ik.solve(pitch_rad, roll_rad)
    
    print(f"  Servo angles: ({theta1:.1f}°, {theta2:.1f}°, {theta3:.1f}°)")
    print(f"  Changes from neutral: ({theta1-15:.1f}°, {theta2-15:.1f}°, {theta3-15:.1f}°)")
    
    # Send command
    servo.send_angles(theta1, theta2, theta3)
    print(f"  → Plate should tilt {direction_name}")
    print(f"  → Observe which servo moves the most")
    time.sleep(hold_time)
    
    # Return to neutral
    servo.level()
    time.sleep(0.5)

def main():
    """Main alignment procedure."""
    print("Stewart Platform Alignment Calibration")
    print("="*60)
    
    # Load config
    cfg = load_config()
    
    # Initialize servo
    servo = ServoBus(
        port=cfg["servo"]["port"],
        baud=cfg["servo"]["baud"],
        neutral_deg=cfg["servo"]["theta0_deg"]
    )
    
    # Initialize IK
    ik = ThreeServoTiltIKSmall(
        plate_radius_m=cfg["plate"]["radius_m"],
        servo_azimuth_deg=cfg["plate"]["servo_azimuth_deg"],
        theta0_deg=cfg["servo"]["theta0_deg"],
        crank_length_m=cfg["ik"]["crank_length_m"]
    )
    
    print(f"\nExpected azimuth angles: {cfg['plate']['servo_azimuth_deg']}°")
    print(f"  Servo 1 should be at 0° (typically 'forward' or 'right')")
    print(f"  Servo 2 should be at 120° (rotated 120° from servo 1)")
    print(f"  Servo 3 should be at 240° (rotated 240° from servo 1)")
    
    try:
        print("\nConnecting to Arduino...")
        servo.open()
        print("Connected! Starting alignment tests...\n")
        time.sleep(1)
        
        # Move to neutral first
        print("Moving to neutral position...")
        servo.level()
        time.sleep(2)
        
        # Test 1: Tilt in +X direction (should primarily affect servo at 0°)
        # Small tilt: 2 degrees pitch (about y-axis = tilt in x direction)
        test_tilt_direction(
            servo, ik, 
            "Tilt RIGHT (+X direction)",
            pitch_rad=math.radians(2.0),  # Positive pitch = tilt right
            roll_rad=0.0,
            hold_time=3.0
        )
        input("\nPress Enter to continue...")
        
        # Test 2: Tilt in -X direction
        test_tilt_direction(
            servo, ik,
            "Tilt LEFT (-X direction)",
            pitch_rad=math.radians(-2.0),  # Negative pitch = tilt left
            roll_rad=0.0,
            hold_time=3.0
        )
        input("\nPress Enter to continue...")
        
        # Test 3: Tilt in +Y direction (should primarily affect servo at 90°, but with 120° spacing)
        test_tilt_direction(
            servo, ik,
            "Tilt FORWARD (+Y direction)",
            pitch_rad=0.0,
            roll_rad=math.radians(2.0),  # Positive roll = tilt forward
            hold_time=3.0
        )
        input("\nPress Enter to continue...")
        
        # Test 4: Tilt in -Y direction
        test_tilt_direction(
            servo, ik,
            "Tilt BACKWARD (-Y direction)",
            pitch_rad=0.0,
            roll_rad=math.radians(-2.0),  # Negative roll = tilt backward
            hold_time=3.0
        )
        input("\nPress Enter to continue...")
        
        # Test 5: Tilt in diagonal directions
        test_tilt_direction(
            servo, ik,
            "Tilt RIGHT-FORWARD (diagonal)",
            pitch_rad=math.radians(1.5),
            roll_rad=math.radians(1.5),
            hold_time=3.0
        )
        
        print("\n" + "="*60)
        print("Alignment Test Complete!")
        print("\nVerification:")
        print("1. When tilting RIGHT (+X): Servo at 0° should move most")
        print("2. When tilting FORWARD (+Y): Servo at 120° should move most")
        print("3. When tilting LEFT-BACKWARD: Servo at 240° should move most")
        print("\nIf the servos don't match expected directions:")
        print("  - Physically rotate the servos to match the expected positions")
        print("  - OR update 'servo_azimuth_deg' in config.json to match your physical setup")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nReturning to neutral position...")
        servo.level()
        time.sleep(1)
        servo.close()
        print("Done.")

if __name__ == "__main__":
    main()

