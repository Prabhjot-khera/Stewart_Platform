# kinematics/ik_small.py
from __future__ import annotations
import math
from typing import Sequence, Tuple

class ThreeServoTiltIKSmall:
    """
    Small-angle IK for a 3-leg tilt plate.
    Inputs: pitch θ (about +y), roll φ (about +x) in radians
    Each leg at azimuth α_i (deg) around the plate center, radius R (m).
    Height at leg i: Δz_i = φ * (R*sin α_i) - θ * (R*cos α_i)
    Servo rotation (rad) ≈ Δθ_i = Δz_i / L_crank_i
    Command (deg): t_i = θ0_i + deg(Δθ_i)
    
    Uses actual physical dimensions:
    - L_crank: length of crank arm (from servo axis to crank-link joint) in meters
    - L_link: length of connecting link (from crank-link joint to plate attachment) in meters
    For small angles, the effective lever arm ≈ L_crank
    """
    def __init__(self,
                 plate_radius_m: float,
                 servo_azimuth_deg: Sequence[float],   # e.g. [0,120,240]
                 theta0_deg: Sequence[float],          # neutral servo angles
                 crank_length_m: Sequence[float],       # crank arm length per leg (m)
                 link_length_m: Sequence[float] | None = None,  # link length per leg (m), optional
                 leg_sign: Sequence[int] | None = None # optional +1/-1 if a leg's sense is inverted
                 ):
        self.R = float(plate_radius_m)
        self.alphas = [math.radians(a) for a in servo_azimuth_deg]
        self.theta0 = list(theta0_deg)
        self.L_crank = list(crank_length_m)
        
        # For small-angle approximation, effective lever arm ≈ crank length
        # If link_length is provided, we could use it for more accurate calculation,
        # but for small angles, crank length is sufficient
        if link_length_m is not None:
            self.L_link = list(link_length_m)
        else:
            self.L_link = None
        
        if leg_sign is None:
            self.sign = [1, 1, 1]
        else:
            self.sign = [1 if s>=0 else -1 for s in leg_sign]

    def solve(self, pitch_rad: float, roll_rad: float) -> Tuple[float, float, float]:
        """
        Solve inverse kinematics: convert (pitch, roll) to servo angles.
        
        Args:
            pitch_rad: Pitch angle (rotation about y-axis) in radians
            roll_rad: Roll angle (rotation about x-axis) in radians
            
        Returns:
            Tuple of 3 servo angles in degrees: (theta1, theta2, theta3)
        """
        out = []
        for a, L_c, t0, s in zip(self.alphas, self.L_crank, self.theta0, self.sign):
            # Calculate vertical displacement at this leg's attachment point
            dz = roll_rad * (self.R * math.sin(a)) - pitch_rad * (self.R * math.cos(a))
            
            # For small angles: dz ≈ L_crank * dtheta_servo
            # Therefore: dtheta_servo ≈ dz / L_crank
            dtheta_rad = s * dz / max(L_c, 1e-9)  # servo rotation in radians
            
            # Convert to degrees and add neutral angle
            out.append(t0 + math.degrees(dtheta_rad))
        return tuple(out)

    def level(self) -> Tuple[float, float, float]:
        return tuple(self.theta0)
