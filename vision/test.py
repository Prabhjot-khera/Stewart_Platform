"""
import cv2, json
import numpy as np
from detector import BallDetector2D

cfg = {
    "ball_hsv_lower": [10, 161, 214],
    "ball_hsv_upper": [25, 255, 255],
    "min_radius_px": 5,
    "max_radius_px": 200,
    "deglare": False
}

cap = cv2.VideoCapture(1)
det = BallDetector2D(cfg)

while True:
    ok, frame = cap.read()
    if not ok: break
    res = det.update(frame, return_mask=True)

    vis = frame.copy()
    if res["found"]:
        (u, v) = res["xy_px"]
        r = int(res["radius_px"])
        cv2.circle(vis, (u, v), r, (0, 255, 0), 2)
        cv2.circle(vis, (u, v), 2, (0, 0, 255), -1)
        cv2.putText(vis, f"({u},{v}) r={r:.1f}", (u+8, v-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 220, 20), 1, cv2.LINE_AA)
    else:
        cv2.putText(vis, "NO BALL", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    if res["mask"] is not None:
        mask3 = cv2.cvtColor(res["mask"], cv2.COLOR_GRAY2BGR)
        vis = np.hstack([vis, mask3])

    cv2.imshow("Ball detector (left) | mask (right)", vis)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
"""

# tests/test_ball_detection_main.py
import cv2, json, os, time
from detector2 import BallDetector2D
from circle_mapper import CircleMapper

def load_cfg():
    # Get path relative to script location, not current working directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    path = os.path.join(PROJECT_ROOT, "config", "config.json")
    
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    # Fallback defaults
    return {
        "camera": {"index": 1},
        "vision": {
            "ball_hsv_lower": [10, 120, 120],
            "ball_hsv_upper": [25, 255, 255],
            "erode_iter": 2, "dilate_iter": 2, "morph_kernel": 5,
            "min_radius_px": 8, "max_radius_px": 80, "min_area_frac": 0.0003
        },
        "plate": {"radius_m": 0.15},
        "mapping": { "edge_canny": {"low": 60, "high": 160},
                     "ellipse_refresh_interval": 5 }
    }

def draw_detection(frame, fit, det_res, mapper):
    """Returns (vis_frame, found_bool, pos_m_tuple or None)."""
    vis = frame.copy()

    # Draw rim/ellipse if valid
    if fit["valid"]:
        u0, v0 = map(int, fit["center"])
        a, b = int(fit["a"]), int(fit["b"])
        ang = int(fit["psi"] * 180.0 / 3.14159265)
        cv2.circle(vis, (u0, v0), 4, (0,255,0), -1)
        cv2.ellipse(vis, (u0, v0), (a, b), ang, 0, 360, (255,0,0), 2)
    else:
        cv2.putText(vis, "Rim not detected", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Draw ball & compute meters
    if det_res["found"]:
        (u, v) = det_res["xy_px"]
        r = det_res["radius_px"]
        cv2.circle(vis, (u, v), r, (0,255,255), 2)
        cv2.circle(vis, (u, v), 3, (0,255,255), -1)

        if fit["valid"]:
            x_m, y_m = mapper.px_to_m((u, v))
            cv2.putText(vis, f"x={x_m:+.3f} m  y={y_m:+.3f} m", (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            return vis, True, (x_m, y_m)
        else:
            cv2.putText(vis, "Rim not ready → no meters", (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            return vis, True, None
    else:
        cv2.putText(vis, "Ball not found", (12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return vis, False, None

def main():
    """Test ball detection + circle mapping with current config."""
    cfg = load_cfg()

    detector = BallDetector2D(cfg.get("vision", {}))
    mapper   = CircleMapper(cfg.get("plate", {}).get("radius_m", 0.15), cfg.get("mapping", {}))

    cam = cfg.get("camera", {})
    cap = cv2.VideoCapture(int(cam.get("index", 1)))
    if not cap.isOpened():
        print("❌ Failed to open camera"); return

    print("Ball Detection Test (Detector + CircleMapper)")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize for consistent processing (optional)
        frame = cv2.resize(frame, (640, 480))

        # Update modules
        fit = mapper.update_from_frame(frame)
        det_res = detector.update(frame, return_mask=False)

        # Build visualization & report
        vis_frame, found, pos_m = draw_detection(frame, fit, det_res, mapper)

        if found and pos_m is not None:
            x_m, y_m = pos_m
            print(f"Ball detected at x={x_m:+.4f} m, y={y_m:+.4f} m from center")
        elif found:
            print("Ball detected (rim not ready yet → meters unavailable)")

        cv2.imshow("Ball Detection Test", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

