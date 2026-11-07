"""Venue landing pad: a blue rounded-square mat ("H" pad), detected in a COLOUR image. PURE (numpy + cv2).

Venue facts (photos IMG_9593/9594, 2026-11-07): saturated sky blue (about RGB 60,168,240, hue ~205 deg)
on an orange-red floor print (hue ~5-10 deg); tan H, tan corner bars, white arcs/dashed circle; a few mm
thick (no depth signature). Side ~0.64 m is an ESTIMATE (FPVtosky family sells 0.51 and 0.64 m): treated
as unknown within 0.45-0.75 m, accepted by metric size only. It replaces the CAD's 5 ft ring pad.

The skybeta drone streams only the OV9282 MONO pair (oakd_vslam.yaml: i_pipeline_type stereo). Hue does
not exist there: `detect` returns None for a grey image (all channels equal) and says so. See
docs/SAFE_POLICY.md "Landing pad".

Output of `detect` (when a pad is found):
  centre_px, quad_px (4x2), area_px, partial (touches the border or is not convex enough: a centre from a
  partial blob is biased, use it to step toward the pad, never to descend), solidity.
  With a camera model (K, R_level_cam, t_level_cam, height above the floor) also, in the LEVEL body frame
  (x forward, y left, metres, from the body origin): centre_xy, side_m (2), yaw_rad (pad side direction,
  modulo 90 deg), size_ok.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except ImportError:                                   # the detector needs OpenCV; callers check
    cv2 = None

HUE_LO, HUE_HI = 84, 114         # OpenCV hue (deg / 2): 168-228 deg; pad ~205, ~185 when highlights clip; floor ~5-10
SAT_MIN, VAL_MIN = 70, 60         # saturated and not in deep shadow; hue, not brightness, does the work
SIDE_MIN_M, SIDE_MAX_M = 0.45, 0.75
MIN_AREA_FRAC = 0.002             # blob area as a fraction of the image


def is_colour(img):
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        return False
    s = img[::8, ::8, :3].astype(np.int16)
    return bool(np.abs(s[..., 0] - s[..., 1]).mean() > 2 or np.abs(s[..., 1] - s[..., 2]).mean() > 2)


def pad_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (HUE_LO, SAT_MIN, VAL_MIN), (HUE_HI, 255, 255))
    # close the H, bars, arcs and fold creases into the pad; open away floor speckle
    k = max(3, int(round(min(img_bgr.shape[:2]) / 150)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5 * k, 5 * k)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m


def _level_from_floor_normal(n_cam, R_level_cam):
    """Smallest rotation correction of R_level_cam (level <- camera) that maps the measured floor
    normal (camera frame) onto level +z; keeps the camera's yaw about the vertical from R_level_cam."""
    z = np.array([0.0, 0.0, 1.0])
    m = R_level_cam @ n_cam
    v = np.cross(m, z)
    s, c = np.linalg.norm(v), float(m @ z)
    if s < 1e-12:
        return R_level_cam
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rc = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return Rc @ R_level_cam


def _ground(px, K, R_level_cam, t_level_cam, height_m):
    """Pixels -> level-body floor points (x fwd, y left), z = -height (floor below the body origin)."""
    uv1 = np.column_stack([px[:, 0], px[:, 1], np.ones(len(px))])
    rays = (R_level_cam @ (np.linalg.inv(K) @ uv1.T)).T              # level body frame
    cz = t_level_cam[2] + height_m                                    # camera above the floor
    down = -rays[:, 2]
    ok = down > 1e-3
    s = np.where(ok, cz / np.maximum(down, 1e-9), np.nan)
    return t_level_cam[:2] + rays[:, :2] * s[:, None], ok


def _side_lines(contour, quad):
    """Homogeneous image lines of the 4 sides, fitted to the contour points between the quad's corners
    (corner 20 % trimmed: the mat's corners are rounded). Falls back to the quad's chords."""
    P = contour.reshape(-1, 2).astype(float)
    lines = []
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        ab = b - a
        L = float(np.linalg.norm(ab))
        t = (P - a) @ ab / max(L * L, 1e-9)
        d = np.abs((P - a) @ np.array([-ab[1], ab[0]])) / max(L, 1e-9)
        sel = (t > 0.2) & (t < 0.8) & (d < 0.08 * L)
        if sel.sum() >= 6:
            vx, vy, x0, y0 = cv2.fitLine(P[sel].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            p, q = np.array([x0, y0, 1.0]), np.array([x0 + vx, y0 + vy, 1.0])
        else:
            p, q = np.array([*a, 1.0]), np.array([*b, 1.0])
        lines.append(np.cross(p, q))
    return lines


def floor_normal_from_square(contour, quad, K):
    """The floor's up-normal in the CAMERA optical frame from the square's two vanishing directions
    (opposite sides are parallel on the floor). Independent of the camera mount pitch and the vehicle's
    attitude estimate. None when the geometry is too weak (a side pair nearly parallel in the image is
    fine: its vanishing point is at infinity and stays a valid direction)."""
    l0, l1, l2, l3 = _side_lines(contour, quad)
    Kinv = np.linalg.inv(K)
    d1 = Kinv @ np.cross(l0, l2)            # sides 0 and 2 are opposite
    d2 = Kinv @ np.cross(l1, l3)
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    d1, d2 = d1 / n1, d2 / n2
    if abs(d1 @ d2) > 0.35:                 # the two floor directions should be ~perpendicular
        return None
    n = np.cross(d1, d2)
    n /= np.linalg.norm(n)
    # optical frame: y is DOWN; the floor normal pointing up toward the camera has negative y
    return -n if n[1] > 0 else n


def detect(img, K=None, R_level_cam=None, t_level_cam=None, height_m=None):
    if cv2 is None or not is_colour(img):
        return None
    h, w = img.shape[:2]
    m = pad_mask(img)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < MIN_AREA_FRAC * h * w:
        return None
    blob = (lab == i).astype(np.uint8)
    cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    solidity = float(cv2.contourArea(c) / max(cv2.contourArea(hull), 1.0))
    x, y, bw, bh = cv2.boundingRect(c)
    border = x <= 2 or y <= 2 or x + bw >= w - 3 or y + bh >= h - 3
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            quad = a.reshape(4, 2).astype(float)
            break
    if quad is None:                                       # rounded corners / occlusion: rotated box
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)
    M = cv2.moments(blob, True)
    centre_px = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    out = dict(centre_px=centre_px, quad_px=quad, area_px=area, solidity=solidity,
               partial=bool(border or solidity < 0.90), border=bool(border))
    if K is None:
        return out
    # metric: project the blob outline onto the floor and fit the square there (perspective removed).
    # The floor plane comes from the square itself when its vanishing points are usable, so a camera
    # mount pitch error or attitude error does not move the answer; else from the level rotation.
    K = np.asarray(K, float)
    R = np.asarray(R_level_cam, float)
    out["floor_from"] = "attitude"
    if not out["partial"]:
        n_cam = floor_normal_from_square(c, quad, K)
        if n_cam is not None:
            R = _level_from_floor_normal(n_cam, R)
            out["floor_from"] = "square"
            out["pitch_correction_deg"] = float(np.degrees(np.arccos(np.clip(
                (R @ np.array([0, 0, 1.0])) @ (np.asarray(R_level_cam, float) @ np.array([0, 0, 1.0])), -1, 1))))
    pts, ok = _ground(c.reshape(-1, 2).astype(float), K, R, np.asarray(t_level_cam, float), float(height_m))
    pts = pts[ok]
    if len(pts) < 8:
        out.update(size_ok=False)
        return out
    (cx, cy), (s1, s2), ang = cv2.minAreaRect(pts.astype(np.float32))
    out.update(centre_xy=np.array([cx, cy], float), side_m=(float(s1), float(s2)),
               yaw_rad=math.radians(ang) % (math.pi / 2),
               size_ok=bool(SIDE_MIN_M <= min(s1, s2) and max(s1, s2) <= SIDE_MAX_M))
    return out


def draw(img, det, path):
    vis = img.copy()
    m = pad_mask(img)
    vis[m > 0] = (0.5 * vis[m > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    if det:
        cv2.polylines(vis, [det["quad_px"].astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 6)
        c = tuple(int(v) for v in det["centre_px"])
        cv2.drawMarker(vis, c, (255, 255, 255), cv2.MARKER_CROSS, 80, 8)
        cv2.putText(vis, f"partial={det['partial']} solidity={det['solidity']:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
    cv2.imwrite(path, vis)


# Measured on synthetic frames with the drone's intrinsics (tests/test_square_pad.py): bearing (lateral)
# error <= 1.5 cm at 1.9-2.6 m even with 3 deg of pitch error; range error 6-9 cm with the floor from the
# square (pitch-immune) and 10-32 cm per 1-3 deg of pitch error with the floor from the attitude.
SIGMA_LATERAL_M = 0.02
SIGMA_RANGE_M = {"square": 0.08, "attitude": 0.25}
# Footprint (7-inch props on the fleet rotor layout): x from -0.239 to +0.169 m (0.408 long),
# y +-0.249 m (0.498 wide); its centre is 0.035 m behind the body origin.
FOOTPRINT_CENTRE_X_M = -0.035


def fix_from_detection(det, cam_xy_body):
    """-> dict(xy, cov, yaw) in the LEVEL BODY frame (from the body origin), or None if not usable.
    The covariance is long along the line of sight from the camera, short across it."""
    if not det or det.get("partial") or not det.get("size_ok") or "centre_xy" not in det:
        return None
    xy = np.asarray(det["centre_xy"], float)
    los = xy - np.asarray(cam_xy_body, float)
    u = los / max(np.linalg.norm(los), 1e-9)
    t = np.array([-u[1], u[0]])
    sr = SIGMA_RANGE_M.get(det.get("floor_from"), 0.25)
    cov = sr ** 2 * np.outer(u, u) + SIGMA_LATERAL_M ** 2 * np.outer(t, t)
    yaw = det.get("yaw_rad") if det.get("floor_from") == "square" else None
    return dict(xy=xy, cov=cov, yaw=yaw, floor_from=det.get("floor_from"))


class PadFuser:
    """Information-weighted fusion of pad fixes (course frame) with the CAD position as a weak prior.
    Bearing lines from different viewpoints intersect; one viewpoint gives an accurate line and a weak range."""

    def __init__(self, prior_xy, prior_sigma_m=0.40):
        self.prior = np.asarray(prior_xy, float)
        self.info = np.eye(2) / prior_sigma_m ** 2
        self.vec = self.info @ self.prior
        self.n = 0
        self.yaw_c = self.yaw_s = 0.0
        self.n_yaw = 0

    def add(self, xy, cov, yaw=None):
        I = np.linalg.inv(np.asarray(cov, float))
        self.info = self.info + I
        self.vec = self.vec + I @ np.asarray(xy, float)
        self.n += 1
        if yaw is not None:                       # a square's side direction is defined modulo 90 deg
            self.yaw_c += math.cos(4 * yaw)
            self.yaw_s += math.sin(4 * yaw)
            self.n_yaw += 1

    @property
    def xy(self):
        return np.linalg.solve(self.info, self.vec)

    @property
    def sigma(self):
        return float(np.sqrt(np.max(np.linalg.eigvalsh(np.linalg.inv(self.info)))))

    @property
    def yaw(self):
        """Pad side direction modulo 90 deg (course frame), or None."""
        if self.n_yaw == 0:
            return None
        return math.atan2(self.yaw_s, self.yaw_c) / 4.0


def footprint_inside(x, y, yaw, pad_xy, side_m, pad_yaw=0.0, rotors=((0.08, 0.16), (-0.15, -0.14), (0.08, -0.16), (-0.15, 0.14)),
                     prop_d=7 * 0.0254):
    """Grader rule (local Gazebo sim): all four prop discs inside the square pad. -> (inside, min margin m)."""
    c, s = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(-pad_yaw), math.sin(-pad_yaw)
    margin = math.inf
    for rx, ry in rotors:
        wx, wy = x + c * rx - s * ry - pad_xy[0], y + s * rx + c * ry - pad_xy[1]
        px, py = cp * wx - sp * wy, sp * wx + cp * wy
        margin = min(margin, side_m / 2 - (abs(px) + prop_d / 2), side_m / 2 - (abs(py) + prop_d / 2))
    return margin >= 0, margin
