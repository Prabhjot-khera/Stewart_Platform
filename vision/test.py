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
