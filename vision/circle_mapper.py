# vision/circle_mapper.py
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import math
from dataclasses import dataclass
import cv2, json, time


@dataclass
class RimFit:
    valid: bool
    center_px: Tuple[float, float] = (0.0, 0.0)
    a_px: float = 0.0   # ellipse major semi-axis (pixels)
    b_px: float = 0.0   # ellipse minor semi-axis (pixels)
    psi_rad: float = 0.0  # rotation of ellipse (radians)

class CircleMapper:
    
    def __init__(self, plate_radius_m: float, cfg: Dict[str, Any]):
        self.R = float(plate_radius_m)
        mcfg = cfg or {}
        ecfg = mcfg.get("edge_canny", {"low": 60, "high": 160})
        self.canny_low = int(ecfg.get("low", 60))
        self.canny_high = int(ecfg.get("high", 160))
        self.refresh_N = int(mcfg.get("ellipse_refresh_interval", 5))
        self.frame_count = 0

        self.rim = RimFit(False)
        # Cached transforms
        self._cos = 1.0
        self._sin = 0.0
        self._sx = 1.0  # meters per pixel along ellipse major-aligned x
        self._sy = 1.0  # meters per pixel along ellipse major-aligned y

    # ---------- public API ----------
    def update_from_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        
        self.frame_count += 1
        if (self.frame_count == 1) or (self.frame_count % self.refresh_N == 0) or (not self.rim.valid):
            self._fit_rim(frame_bgr)
            self._update_affine()

        return {
            "valid": self.rim.valid,
            "center": self.rim.center_px,
            "a": self.rim.a_px,
            "b": self.rim.b_px,
            "psi": self.rim.psi_rad
        }

    def px_to_m(self, uv: Tuple[float, float]) -> Tuple[float, float]:
       
        if not self.rim.valid:
            return (0.0, 0.0)

        u, v = uv
        u0, v0 = self.rim.center_px
        # translate to ellipse center
        du = u - u0
        dv = v - v0
        # rotate by -psi (align to ellipse axes)
        x_al =  self._cos * du + self._sin * dv
        y_al = -self._sin * du + self._cos * dv
        # scale to meters using anisotropic scale
        x_m = x_al * self._sx
        y_m = y_al * self._sy
        return (x_m, y_m)

    def m_to_px(self, xy_m: Tuple[float, float]) -> Tuple[int, int]:
        
        if not self.rim.valid:
            return (0, 0)

        x_m, y_m = xy_m
        # inverse scale
        x_al = x_m / (self._sx + 1e-12)
        y_al = y_m / (self._sy + 1e-12)
        # rotate by +psi and translate back
        du =  self._cos * x_al - self._sin * y_al
        dv =  self._sin * x_al + self._cos * y_al
        u0, v0 = self.rim.center_px
        u = int(round(u0 + du))
        v = int(round(v0 + dv))
        return (u, v)

    def _fit_rim(self, frame_bgr) -> None:
       
        

        h, w = frame_bgr.shape[:2]
        min_area = 0.05 * (w * h)  # ignore small contours (<5% of image); tune as needed

        # 1) HSV green mask
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Default green band; adjust S/V mins if lighting is dim
        green_lo = np.array([35,  60,  60], np.uint8)
        green_hi = np.array([85, 255, 255], np.uint8)
        mask = cv2.inRange(hsv, green_lo, green_hi)

        # 2) Clean up mask (open/close) and get edges
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

        edges = cv2.Canny(mask, self.canny_low, self.canny_high)
        edges = cv2.dilate(edges, k3, iterations=1)  # fuse small gaps

        # 3) Largest valid contour
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = [c for c in cnts if cv2.contourArea(cv2.convexHull(c)) > min_area]
        if not cnts:
            self.rim = RimFit(False)
            return

        best = max(cnts, key=lambda c: cv2.contourArea(cv2.convexHull(c)))
        if len(best) < 5:
            self.rim = RimFit(False)
            return

        # 4) Ellipse fit + normalize angle to major axis
        (u0, v0), (W, H), angle_deg = cv2.fitEllipse(best)
        if H > W:
            a = 0.5 * H; b = 0.5 * W
            psi = math.radians(angle_deg + 90.0)
        else:
            a = 0.5 * W; b = 0.5 * H
            psi = math.radians(angle_deg)

        self.rim = RimFit(True, (float(u0), float(v0)), float(a), float(b), float(psi))

    def _update_affine(self) -> None:
        if not self.rim.valid or self.rim.a_px <= 1 or self.rim.b_px <= 1:
            self._cos, self._sin = 1.0, 0.0
            self._sx, self._sy = 1.0, 1.0
            return

        self._cos = math.cos(self.rim.psi_rad)
        self._sin = math.sin(self.rim.psi_rad)

        self._sx = self.R / self.rim.a_px
        self._sy = self.R / self.rim.b_px


if __name__ == "__main__":
    # --- Load config ---
    cfg = {
        "plate": {"radius_m": 0.15},  # your real plate radius in meters
        "mapping": {
            "edge_canny": {"low": 60, "high": 160},
            "ellipse_refresh_interval": 5
        }
    }

    # --- Initialize mapper ---
    mapper = CircleMapper(cfg["plate"]["radius_m"], cfg["mapping"])

    # --- Open camera ---
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Could not open camera")
        exit()

    start = time.time()
    MAX_RUNTIME = 40  # seconds

    while True:
        if time.time() - start > MAX_RUNTIME:
            print("⏰ Auto-terminating after 20 seconds.")
            break

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame.")
            break

        fit = mapper.update_from_frame(frame)
        vis = frame.copy()

        if fit["valid"]:
            u0, v0 = map(int, fit["center"])
            cv2.circle(vis, (u0, v0), 4, (0, 255, 0), -1)

            # Draw ellipse
            a, b = int(fit["a"]), int(fit["b"])
            angle_deg = int(fit["psi"] * 180.0 / 3.14159)
            cv2.ellipse(vis, (u0, v0), (a, b), angle_deg, 0, 360, (255, 0, 0), 2)

            # Draw a 2 cm tick (for scale verification)
            u_tick, v_tick = mapper.m_to_px((0.02, 0))
            cv2.line(vis, (u0, v0), (u_tick, v_tick), (0, 255, 255), 2)
            cv2.putText(vis, "Rim fit OK", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            cv2.putText(vis, "No rim detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Circle Mapper", vis)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
