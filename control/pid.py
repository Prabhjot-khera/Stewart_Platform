# control/pid.py
from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass
class PIDGains:
    kp: float = 6.0
    ki: float = 0.0
    kd: float = 0.8
    d_lp_hz: float = 20.0   # derivative low-pass cutoff (Hz)
    i_clamp: float = 0.2    # integral state clamp (abs units of integral term)
    out_clip: float | None = None  # absolute clamp for output (same units as output)

class PID1D:
    """
    Simple PID controller for one axis.

    - 'step(error, dt)' returns the control output (e.g., radians of tilt).
    - Derivative is computed on error and low-passed with a 1st-order filter.
    - Integral term is clamped (anti-windup).
    - Output can be clamped via 'out_clip'.

    Typical use in your loop:
        pid = PID1D(PIDGains(kp=6.0, ki=0.0, kd=0.8, d_lp_hz=20.0, i_clamp=0.2, out_clip=tilt_cap))
        u = pid.step(error, dt)
    """

    def __init__(self, gains: dict | PIDGains | None = None, **kwargs):
        if isinstance(gains, dict):
            self.g = PIDGains(**gains)
        elif isinstance(gains, PIDGains):
            self.g = gains
        else:
            # allow direct kwargs too, e.g., PID1D(kp=..., kd=...)
            self.g = PIDGains(**kwargs)

        self._i = 0.0
        self._e_prev = 0.0
        self._d_prev = 0.0
        self._init = False

    def reset(self):
        """Clear integral and derivative memory."""
        self._i = 0.0
        self._e_prev = 0.0
        self._d_prev = 0.0
        self._init = False

    def set_gains(self, **kwargs):
        """Update gains at runtime, e.g., set_gains(kp=7.5)."""
        for k, v in kwargs.items():
            if hasattr(self.g, k):
                setattr(self.g, k, v)

    def step(self, error: float, dt: float) -> float:
        """Advance one control step; returns control output."""
        if dt <= 0.0:
            # if timing hiccup, return pure P as safe fallback
            return self.g.kp * error

        # --- Proportional
        P = self.g.kp * error

        # --- Derivative on error, 1st-order low-pass
        if not self._init:
            de = 0.0
            self._d_prev = 0.0
            self._init = True
        else:
            de = (error - self._e_prev) / dt

        # LP alpha
        rc = 1.0 / (2.0 * math.pi * max(self.g.d_lp_hz, 1e-6))
        alpha = dt / (rc + dt)
        d_filt = (1.0 - alpha) * self._d_prev + alpha * de
        self._d_prev = d_filt
        self._e_prev = error

        D = self.g.kd * d_filt

        # --- Integral with clamp (anti-windup via state clamp)
        self._i += error * dt
        # clamp integral state
        ic = max(-self.g.i_clamp, min(self._i, self.g.i_clamp))
        # (optional: only assign if clamped; this is equivalent here)
        self._i = ic
        I = self.g.ki * self._i

        # --- Output
        u = P + I + D

        # hard output clamp (e.g., radians or degrees)
        if self.g.out_clip is not None:
            u = max(-self.g.out_clip, min(u, self.g.out_clip))

        return u
