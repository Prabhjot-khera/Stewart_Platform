# io/arduino.py
from __future__ import annotations
import time
import serial
from typing import Sequence, Optional

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

class ServoBus:
    """
    Minimal serial driver for the Arduino sketch that reads CSV lines:
        "angle1,angle2,angle3\\n"
    Angles are in DEGREES. This class clamps to [min_deg, max_deg] before sending.
    """

    def __init__(
        self,
        port: str,
        baud: int = 9600,
        timeout_s: float = 0.02,
        neutral_deg: Sequence[float] = (15.0, 15.0, 15.0),
        limits_deg: Optional[dict] = None,  # {"min": 0, "max": 30}
    ):
        self.port = port
        self.baud = baud
        self.timeout = timeout_s
        self.neutral = tuple(neutral_deg)
        lim = limits_deg or {"min": 0.0, "max": 30.0}
        self.min_deg = float(lim.get("min", 0.0))
        self.max_deg = float(lim.get("max", 30.0))
        self.ser: Optional[serial.Serial] = None

    # ----- lifecycle -----
    def open(self):
        self.close()
        # Wait a bit longer to ensure port is fully released
        time.sleep(0.2)
        
        # Try to open with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
                # many Arduinos auto-reset on serial open; give it a moment
                time.sleep(0.5)
                self.flush()
                # move to neutral once connected
                self.level()
                return self  # allow chaining
            except serial.SerialException as e:
                if attempt < max_retries - 1:
                    print(f"Port open attempt {attempt + 1} failed: {e}")
                    print("Retrying in 0.5 seconds...")
                    time.sleep(0.5)
                    self.close()  # Ensure clean state
                else:
                    raise

    def close(self):
        if self.ser:
            try:
                # Flush buffers before closing
                if self.ser.is_open:
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                self.ser.close()
            except Exception:
                pass
            finally:
                self.ser = None
            # Give the OS time to release the port
            time.sleep(0.1)

    def flush(self):
        if not self.ser:
            return
        try:
            if self.ser.in_waiting:
                _ = self.ser.read(self.ser.in_waiting)
            self.ser.reset_output_buffer()
        except Exception:
            pass

    # ----- commands -----
    def send_angles(self, t1: float, t2: float, t3: float):
        """
        Send three angles (deg) as 3 bytes, matching ballBalance copy approach.
        Values are rounded to integers and clamped to [min_deg, max_deg].
        """
        if not self.ser or not self.ser.is_open:
            print("Warning: Serial port not open!")
            return
        a = int(round(_clip(t1, self.min_deg, self.max_deg)))
        b = int(round(_clip(t2, self.min_deg, self.max_deg)))
        c = int(round(_clip(t3, self.min_deg, self.max_deg)))
        # Send 3 bytes: angle1, angle2, angle3 (matching ballBalance copy method)
        data = bytes([a, b, c])
        self.ser.write(data)
        self.ser.flush()  # Ensure data is sent immediately
        # Debug: uncomment to verify bytes being sent
        # print(f"Sent bytes: {[a, b, c]} = {data.hex()}")

    def level(self):
        """Send neutral angles."""
        self.send_angles(*self.neutral)

    # context manager sugar
    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
