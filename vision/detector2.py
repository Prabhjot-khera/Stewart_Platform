# vision/detector.py
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Any
import time
import json
import os

class BallDetector2D:
    """
    Orange-ball detector using:
      1) HSV threshold (orange)
      2) Morphology (erode/dilate)
      3) Contours -> minEnclosingCircle on the largest blob

    Returns ONLY pixel-space outputs (top-left image origin):
      {
        "found": bool,
        "xy_px": (u, v),       # pixel center
        "radius_px": r,        # pixel radius
        "mask": <np.ndarray>   # only if return_mask=True
      }
    """

    def __init__(self, cfg: Dict[str, Any] | None):
        vcfg = cfg or {}
        # HSV bounds for orange (tune as needed)
        self.lo = np.array(vcfg.get("ball_hsv_lower", [10, 120, 120]), dtype=np.uint8)
        self.hi = np.array(vcfg.get("ball_hsv_upper", [25, 255, 255]), dtype=np.uint8)

        # Optional pre-blur on HSV V to reduce glare
        self.deglare = bool(vcfg.get("deglare", False))

        # Morphology
        self.erode_iter  = int(vcfg.get("erode_iter", 2))
        self.dilate_iter = int(vcfg.get("dilate_iter", 2))
        k = int(vcfg.get("morph_kernel", 3))
        k = max(3, k | 1)  # ensure odd >=3
        self.kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # Radius sanity (pixels)
        self.rmin = int(vcfg.get("min_radius_px", 6))
        self.rmax = int(vcfg.get("max_radius_px", 120))

        # Min contour area (optional guard to avoid tiny specks)
        self.min_area = float(vcfg.get("min_area_frac", 0.0002))  # 0.02% of image by default

    def update(self, frame_bgr, return_mask: bool = False) -> Dict[str, Any]:
        """Process a BGR frame and return pixel location/radius of the orange ball if found."""
        h, w = frame_bgr.shape[:2]

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        if self.deglare:
            # clip extreme highlights in V channel
            H, S, V = cv2.split(hsv)
            V = np.minimum(V, np.uint8(240))
            hsv = cv2.merge((H, S, V))

        # 1) Threshold to orange
        mask = cv2.inRange(hsv, self.lo, self.hi)

        # 2) Morphological cleanup: erode -> dilate
        if self.erode_iter > 0:
            mask = cv2.erode(mask, self.kern, iterations=self.erode_iter)
        if self.dilate_iter > 0:
            mask = cv2.dilate(mask, self.kern, iterations=self.dilate_iter)

        # 3) Contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Optional: filter by area fraction to remove tiny specks
        if cnts and self.min_area > 0:
            minA = self.min_area * (w * h)
            cnts = [c for c in cnts if cv2.contourArea(c) >= minA]

        if not cnts:
            out = {"found": False, "xy_px": (0, 0), "radius_px": 0}
            if return_mask: out["mask"] = mask
            return out

        # Largest contour
        c = max(cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        u, v, r = int(round(x)), int(round(y)), float(radius)

        # Radius sanity check
        if r < self.rmin or r > self.rmax:
            out = {"found": False, "xy_px": (0, 0), "radius_px": 0}
            if return_mask: out["mask"] = mask
            return out

        out = {"found": True, "xy_px": (u, v), "radius_px": int(round(r))}
        if return_mask: out["mask"] = mask
        return out

if __name__ == "__main__":
    # Get path relative to script location, not current working directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    CFG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")
    
    # Load config from file, or use defaults
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH, "r") as f:
            cfg = json.load(f)
    else:
        # Fallback defaults
        cfg = {
            "camera": {"index": 1, "width": 640, "height": 480, "fps": 30},
            "vision": {
                "ball_hsv_lower": [10, 120, 120],
                "ball_hsv_upper": [25, 255, 255],
                "erode_iter": 2,
                "dilate_iter": 2,
                "morph_kernel": 5,
                "min_radius_px": 8,
                "max_radius_px": 200,
                "min_area_frac": 0.0003,
                "deglare": False
            }
        }

    cam = cfg.get("camera", {})
    cap = cv2.VideoCapture(int(cam.get("index", 1)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 480))
    cap.set(cv2.CAP_PROP_FPS,          cam.get("fps", 30))
    if not cap.isOpened():
        print("❌ Could not open camera"); raise SystemExit

    det = BallDetector2D(cfg["vision"])

    t0 = time.time()
    MAX_RUNTIME = 60  # seconds

    while True:
        if time.time() - t0 > MAX_RUNTIME:
            print("⏰ Auto-terminating after 20s")
            break

        ok, frame = cap.read()
        if not ok:
            print("⚠️ Failed to read frame")
            break

        res = det.update(frame, return_mask=True)
        vis = frame.copy()

        # draw detection
        if res["found"]:
            (u, v) = res["xy_px"]
            r = res["radius_px"]
            cv2.circle(vis, (u, v), r, (0, 255, 255), 2)
            cv2.circle(vis, (u, v), 3, (0, 255, 255), -1)
            cv2.putText(vis, f"Ball px: ({u}, {v})  r={r}px",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            # console log
            print(f"Ball @ pixels: ({u}, {v}), r={r}")
        else:
            cv2.putText(vis, "Ball not found", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # (optional) show mask in a corner
        if "mask" in res:
            mask_bgr = cv2.cvtColor(res["mask"], cv2.COLOR_GRAY2BGR)
            h, w = vis.shape[:2]
            mh, mw = mask_bgr.shape[:2]
            scale = 0.3
            mask_small = cv2.resize(mask_bgr, (int(mw*scale), int(mh*scale)))
            vis[0:mask_small.shape[0], 0:mask_small.shape[1]] = mask_small

        cv2.imshow("Detector Only", vis)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()