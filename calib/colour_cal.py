# colour_cal.py
import cv2, json, os, time
import numpy as np

# Get path relative to script location, not current working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CFG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")

def _ensure_cfg(path=CFG_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            try: return json.load(f)
            except: pass
    # defaults if no config present
    return {
        "camera": {"index": 0, "width": 640, "height": 480, "fps": 30},
        "vision": {
            "ball_hsv_lower": [10, 120, 120],
            "ball_hsv_upper": [25, 255, 255],
            "erode_iter": 2, "dilate_iter": 2, "morph_kernel": 5,
            "min_radius_px": 8, "max_radius_px": 80, "min_area_frac": 0.0003,
            "deglare": False
        }
    }

def _save_cfg(cfg, path=CFG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n✅ Saved HSV & params to {path}")

def _get(v, key, default):  # tiny helper
    return v.get(key, default)

def main():
    cfg = _ensure_cfg()
    v = cfg.setdefault("vision", {})

    # starting values from config (or defaults)
    loH, loS, loV = _get(v, "ball_hsv_lower", [10,120,120])
    hiH, hiS, hiV = _get(v, "ball_hsv_upper", [25,255,255])
    erode_i  = int(_get(v, "erode_iter", 2))
    dilate_i = int(_get(v, "dilate_iter", 2))
    morph_k  = int(_get(v, "morph_kernel", 5))
    rmin     = int(_get(v, "min_radius_px", 8))
    rmax     = int(_get(v, "max_radius_px", 80))
    deglare  = bool(_get(v, "deglare", False))

    # camera
    cam = cfg.get("camera", {"index": 1})
    cap = cv2.VideoCapture(int(cam.get("index", 1)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 480))
    cap.set(cv2.CAP_PROP_FPS,          cam.get("fps", 30))

    if not cap.isOpened():
        print("❌ Could not open camera"); return

    # UI
    cv2.namedWindow("cal_mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cal_mask", 640, 480)
    cv2.namedWindow("cal_frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cal_frame", 640, 480)

    # Trackbars
    cv2.createTrackbar("loH", "cal_mask", loH, 179, lambda v: None)
    cv2.createTrackbar("hiH", "cal_mask", hiH, 179, lambda v: None)
    cv2.createTrackbar("loS", "cal_mask", loS, 255, lambda v: None)
    cv2.createTrackbar("hiS", "cal_mask", hiS, 255, lambda v: None)
    cv2.createTrackbar("loV", "cal_mask", loV, 255, lambda v: None)
    cv2.createTrackbar("hiV", "cal_mask", hiV, 255, lambda v: None)

    cv2.createTrackbar("erode",  "cal_mask", erode_i, 5, lambda v: None)
    cv2.createTrackbar("dilate", "cal_mask", dilate_i, 5, lambda v: None)
    cv2.createTrackbar("kernel", "cal_mask", morph_k, 15, lambda v: None)

    cv2.createTrackbar("rMIN", "cal_mask", rmin, 300, lambda v: None)
    cv2.createTrackbar("rMAX", "cal_mask", rmax, 400, lambda v: None)

    cv2.createTrackbar("deglare(0/1)", "cal_mask", 1 if deglare else 0, 1, lambda v: None)

    last_print = 0.0
    print("🎯 Colour calibration running")
    print("• Move the sliders until the MASK isolates ONLY the ball")
    print("• Press 's' to save, 'q' to quit")

    while True:
        ok, frame = cap.read()
        if not ok: print("⚠️ frame grab failed"); break

        # read UI
        loH = cv2.getTrackbarPos("loH", "cal_mask")
        hiH = cv2.getTrackbarPos("hiH", "cal_mask")
        loS = cv2.getTrackbarPos("loS", "cal_mask")
        hiS = cv2.getTrackbarPos("hiS", "cal_mask")
        loV = cv2.getTrackbarPos("loV", "cal_mask")
        hiV = cv2.getTrackbarPos("hiV", "cal_mask")
        erode_i  = cv2.getTrackbarPos("erode",  "cal_mask")
        dilate_i = cv2.getTrackbarPos("dilate", "cal_mask")
        morph_k  = max(3, cv2.getTrackbarPos("kernel", "cal_mask") | 1)  # odd >=3
        rmin     = cv2.getTrackbarPos("rMIN", "cal_mask")
        rmax     = max(rmin+1, cv2.getTrackbarPos("rMAX", "cal_mask"))
        deglare  = bool(cv2.getTrackbarPos("deglare(0/1)", "cal_mask"))

        # HSV + optional deglare
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if deglare:
            H, S, V = cv2.split(hsv)
            V = np.minimum(V, np.uint8(240))
            hsv = cv2.merge((H, S, V))

        lo = np.array([loH, loS, loV], np.uint8)
        hi = np.array([hiH, hiS, hiV], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)

        # morphology
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        if erode_i  > 0: mask = cv2.erode(mask, k, iterations=erode_i)
        if dilate_i > 0: mask = cv2.dilate(mask, k, iterations=dilate_i)

        # find contours & draw largest circle (like your detector)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = frame.copy()
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            (x, y), R = cv2.minEnclosingCircle(c)
            u, v, r = int(round(x)), int(round(y)), int(round(R))
            if rmin <= r <= rmax:
                cv2.circle(out, (u, v), r, (0,255,255), 2)
                cv2.circle(out, (u, v), 2, (0,255,255), -1)
                cv2.putText(out, f"px:({u},{v}) r={r}", (12,24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            else:
                cv2.putText(out, f"radius {r}px out of [{rmin},{rmax}]", (12,24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cv2.putText(out, "no contours", (12,24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # show
        cv2.imshow("cal_frame", out)
        cv2.imshow("cal_mask", mask)

        # HUD console each second
        now = time.time()
        if now - last_print > 1.0:
            last_print = now
            print(f"HSV lo={lo.tolist()} hi={hi.tolist()}  morph(k={morph_k}, e={erode_i}, d={dilate_i})  r=[{rmin},{rmax}] deglare={deglare}")

        kkey = cv2.waitKey(1) & 0xFF
        if kkey == ord('q'):
            break
        elif kkey == ord('s'):
            # persist into config
            cfg["vision"]["ball_hsv_lower"] = lo.tolist()
            cfg["vision"]["ball_hsv_upper"] = hi.tolist()
            cfg["vision"]["erode_iter"] = int(erode_i)
            cfg["vision"]["dilate_iter"] = int(dilate_i)
            cfg["vision"]["morph_kernel"] = int(morph_k)
            cfg["vision"]["min_radius_px"] = int(rmin)
            cfg["vision"]["max_radius_px"] = int(rmax)
            cfg["vision"]["deglare"] = bool(deglare)
            _save_cfg(cfg)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
