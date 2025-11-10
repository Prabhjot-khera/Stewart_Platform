#!/usr/bin/env python3
"""
Main control loop for Stewart Platform ball balancing.

Integrates:
- Vision (ball detection + coordinate mapping)
- Control (PID controllers)
- Kinematics (inverse kinematics)
- Servo communication
"""
import cv2
import json
import os
import time
import math
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.detector2 import BallDetector2D
from vision.circle_mapper import CircleMapper
from control.pid import PID1D
from kinematics.ik_small import ThreeServoTiltIKSmall
from io.arduino import ServoBus

def load_config():
    """Load config.json with proper path resolution."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "config.json")
    
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    """Main ball balancing control loop."""
    print("="*60)
    print("Stewart Platform Ball Balancing System")
    print("="*60)
    
    # Load configuration
    cfg = load_config()
    
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(cfg["camera"]["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["camera"]["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    
    # Initialize vision components
    print("Initializing vision...")
    detector = BallDetector2D(cfg["vision"])
    mapper = CircleMapper(cfg["plate"]["radius_m"], cfg["mapping"])
    
    # Initialize control components
    print("Initializing control...")
    loop_hz = cfg["control"]["loop_hz"]
    dt = 1.0 / loop_hz
    tilt_limit_rad = math.radians(cfg["control"]["tilt_deg_limit"])
    
    pid_x = PID1D(
        cfg["control"]["pid_x"],
        out_clip=tilt_limit_rad
    )
    pid_y = PID1D(
        cfg["control"]["pid_y"],
        out_clip=tilt_limit_rad
    )
    
    # Initialize inverse kinematics
    print("Initializing kinematics...")
    ik = ThreeServoTiltIKSmall(
        plate_radius_m=cfg["plate"]["radius_m"],
        servo_azimuth_deg=cfg["plate"]["servo_azimuth_deg"],
        theta0_deg=cfg["servo"]["theta0_deg"],
        crank_length_m=cfg["ik"]["crank_length_m"]
    )
    
    # Initialize servo communication
    print("Initializing servos...")
    servo = ServoBus(
        port=cfg["servo"]["port"],
        baud=cfg["servo"]["baud"],
        neutral_deg=cfg["servo"]["theta0_deg"]
    )
    
    try:
        print("Connecting to Arduino...")
        servo.open()
        print("✅ Connected!")
        time.sleep(1)  # Wait for Arduino to stabilize
        
        print("\n" + "="*60)
        print("Starting control loop...")
        print(f"Loop rate: {loop_hz} Hz")
        print("Press 'q' to quit, 'r' to reset PID integrals")
        print("Use trackbars in the window to tune PID gains in real-time")
        print("="*60 + "\n")
        
        # Control loop variables
        last_time = time.time()
        frame_count = 0
        last_ball_found = False
        
        # Create display window
        cv2.namedWindow("Ball Balancing", cv2.WINDOW_NORMAL)
        
        # Create trackbars for PID tuning
        # Trackbar ranges: Kp (0-200, scale 0.1), Ki (0-100, scale 0.01), Kd (0-100, scale 0.1)
        def nothing(x):
            pass  # Placeholder callback
        
        # X-axis PID trackbars
        cv2.createTrackbar("X Kp", "Ball Balancing", 
                          int(cfg["control"]["pid_x"]["kp"] * 10), 200, nothing)
        cv2.createTrackbar("X Ki", "Ball Balancing", 
                          int(cfg["control"]["pid_x"]["ki"] * 100), 100, nothing)
        cv2.createTrackbar("X Kd", "Ball Balancing", 
                          int(cfg["control"]["pid_x"]["kd"] * 10), 100, nothing)
        
        # Y-axis PID trackbars
        cv2.createTrackbar("Y Kp", "Ball Balancing", 
                          int(cfg["control"]["pid_y"]["kp"] * 10), 200, nothing)
        cv2.createTrackbar("Y Ki", "Ball Balancing", 
                          int(cfg["control"]["pid_y"]["ki"] * 100), 100, nothing)
        cv2.createTrackbar("Y Kd", "Ball Balancing", 
                          int(cfg["control"]["pid_y"]["kd"] * 10), 100, nothing)
        
        while True:
            loop_start = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to grab frame")
                continue
            
            # Resize if needed (optional, for consistent processing)
            # frame = cv2.resize(frame, (640, 480))
            
            # Vision pipeline
            det_res = detector.update(frame, return_mask=False)
            rim_fit = mapper.update_from_frame(frame)
            
            # Calculate ball position in meters
            if det_res["found"] and rim_fit["valid"]:
                x_m, y_m = mapper.px_to_m(det_res["xy_px"])
                last_ball_found = True
            else:
                # Ball not found or rim not detected - use last known position or zero
                if last_ball_found:
                    # Hold last position (could also use (0, 0) to return to center)
                    x_m, y_m = 0.0, 0.0  # Return to center if ball lost
                else:
                    x_m, y_m = 0.0, 0.0
                last_ball_found = False
            
            # Calculate errors (setpoint = center = 0, 0)
            error_x = 0.0 - x_m
            error_y = 0.0 - y_m
            
            # Update PID gains from trackbars (real-time tuning)
            kp_x = cv2.getTrackbarPos("X Kp", "Ball Balancing") / 10.0
            ki_x = cv2.getTrackbarPos("X Ki", "Ball Balancing") / 100.0
            kd_x = cv2.getTrackbarPos("X Kd", "Ball Balancing") / 10.0
            pid_x.set_gains(kp=kp_x, ki=ki_x, kd=kd_x)
            
            kp_y = cv2.getTrackbarPos("Y Kp", "Ball Balancing") / 10.0
            ki_y = cv2.getTrackbarPos("Y Ki", "Ball Balancing") / 100.0
            kd_y = cv2.getTrackbarPos("Y Kd", "Ball Balancing") / 10.0
            pid_y.set_gains(kp=kp_y, ki=ki_y, kd=kd_y)
            
            # PID control
            pitch_rad = pid_x.step(error_x, dt)
            roll_rad = pid_y.step(error_y, dt)
            
            # Inverse kinematics
            theta1, theta2, theta3 = ik.solve(pitch_rad, roll_rad)
            
            # Send to servos
            servo.send_angles(theta1, theta2, theta3)
            
            # Visualization
            vis = frame.copy()
            
            # Draw rim if detected
            if rim_fit["valid"]:
                u0, v0 = map(int, rim_fit["center"])
                a, b = int(rim_fit["a"]), int(rim_fit["b"])
                angle_deg = int(rim_fit["psi"] * 180.0 / math.pi)
                cv2.ellipse(vis, (u0, v0), (a, b), angle_deg, 0, 360, (255, 0, 0), 2)
                cv2.circle(vis, (u0, v0), 4, (0, 255, 0), -1)
            
            # Draw ball if detected
            if det_res["found"]:
                u, v = det_res["xy_px"]
                r = det_res["radius_px"]
                cv2.circle(vis, (u, v), r, (0, 255, 255), 2)
                cv2.circle(vis, (u, v), 3, (0, 255, 255), -1)
                
                # Show position in meters
                if rim_fit["valid"]:
                    cv2.putText(vis, f"x={x_m:+.3f}m y={y_m:+.3f}m", 
                               (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 255), 1)
            
            # Status text
            status_y = 30
            if rim_fit["valid"]:
                cv2.putText(vis, "Rim: OK", (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "Rim: NOT DETECTED", (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            status_y += 25
            if det_res["found"]:
                cv2.putText(vis, f"Ball: x={x_m:+.3f}m y={y_m:+.3f}m", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 255), 2)
            else:
                cv2.putText(vis, "Ball: NOT FOUND", (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            status_y += 25
            cv2.putText(vis, f"Pitch: {math.degrees(pitch_rad):+.2f}deg Roll: {math.degrees(roll_rad):+.2f}deg", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            status_y += 20
            cv2.putText(vis, f"Servos: ({theta1:.1f}deg, {theta2:.1f}deg, {theta3:.1f}deg)", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            status_y += 20
            cv2.putText(vis, f"PID X: Kp={kp_x:.2f} Ki={ki_x:.3f} Kd={kd_x:.2f}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 0), 1)
            
            status_y += 20
            cv2.putText(vis, f"PID Y: Kp={kp_y:.2f} Ki={ki_y:.3f} Kd={kd_y:.2f}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 0), 1)
            
            # FPS counter
            frame_count += 1
            if frame_count % 60 == 0:
                elapsed = time.time() - last_time
                fps = 60.0 / elapsed
                last_time = time.time()
                print(f"FPS: {fps:.1f} | Ball: ({x_m:+.3f}, {y_m:+.3f})m | "
                      f"Tilt: ({math.degrees(pitch_rad):+.2f}, {math.degrees(roll_rad):+.2f})deg")
            
            cv2.imshow("Ball Balancing", vis)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                pid_x.reset()
                pid_y.reset()
                print("PID integrals reset")
            
            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")
        print("Returning servos to neutral...")
        servo.level()
        time.sleep(0.5)
        servo.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
