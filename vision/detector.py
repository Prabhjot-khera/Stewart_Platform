# vision/detector.py
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

@dataclass
class DetectResult:
    found: bool
    xy_px: Optional[Tuple[int, int]] = None
    radius_px: Optional[float] = None
    mask: Optional[np.ndarray] = None

class BallDetector2D:
   
    def __init__(self, cfg: Dict[str, Any]):
        lo = cfg.get("ball_hsv_lower", [25, 80, 80])
        hi = cfg.get("ball_hsv_upper", [40, 255, 255])
        self.lower = np.array(lo, dtype=np.uint8)
        self.upper = np.array(hi, dtype=np.uint8)

        self.rmin = int(cfg.get("min_radius_px", 5))
        self.rmax = int(cfg.get("max_radius_px", 80))
        self.deglare = bool(cfg.get("deglare", True))

        # Morphology kernels
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Pre-allocs
        self._hsv = None
        self._mask = None

    # ---------- public API ----------
    def update(self, frame_bgr: np.ndarray, return_mask: bool = False) -> Dict[str, Any]:
       
        if frame_bgr is None or frame_bgr.size == 0:
            return {"found": False, "xy_px": None, "radius_px": None, "mask": None}

        img = frame_bgr

        if self.deglare:
            img = self._suppress_glare(img)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self._hsv = hsv

        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Clean up
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        self._mask = mask

        # Find best circular blob
        uvr = self._find_best_blob(mask)
        if uvr is None:
            return {"found": False, "xy_px": None, "radius_px": None, "mask": mask if return_mask else None}

        (u, v, r) = uvr
        if r < self.rmin or r > self.rmax:
            return {"found": False, "xy_px": None, "radius_px": None, "mask": mask if return_mask else None}

        return {"found": True, "xy_px": (int(u), int(v)), "radius_px": float(r), "mask": mask if return_mask else None}

    # ---------- internals ----------
    def _suppress_glare(self, img_bgr: np.ndarray) -> np.ndarray:
        
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_soft = cv2.bilateralFilter(v, d=5, sigmaColor=30, sigmaSpace=5)
        v_clamped = np.minimum(v, v_soft + 25).astype(np.uint8)
        hsv2 = cv2.merge([h, s, v_clamped])
        return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    def _find_best_blob(self, mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        best = None
        best_score = -1.0

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue

            circularity = 4.0 * np.pi * (area / (peri * peri))  
            (x, y), r = cv2.minEnclosingCircle(c)

            # Prefer round, medium-sized blobs
            score = circularity * (r / (self.rmax + 1e-6))

            if score > best_score:
                best_score = score
                best = (x, y, r)

        return best
