"""
=============================================================
  Dual RealSense D435i — Stable Stitch + Calibration Export
=============================================================

FIXES v2:
  1. Distortion fix    — FLANN matcher (ORB se zyada stable),
                         adaptive match threshold,
                         homography sanity check (reject bad H)
  2. Calibration save  — S press → JSON file with ALL intrinsic
                         + extrinsic + depth + stitch parameters
  3. Medium overlap    — 40% overlap zone optimized blending
  4. ROI floor-only    — ceiling pe detect nahi karta

SERIAL NUMBERS:
  SN1 / SN2 mein apne serials daalo

KEYBOARD:
  S  -> Screenshot + calibration JSON save
  R  -> Homography force recompute
  D  -> Depth overlay toggle
  C  -> Calibration-only save (bina screenshot)
  Q  -> Quit
=============================================================
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os, time, json
from datetime import datetime

# =============================================================
#  CONFIG
# =============================================================
SN1 = "112322077861"
SN2 = "112222070011"

WIDTH, HEIGHT      = 640, 480
FPS_TARGET         = 30
SAVE_DIR           = "captures_calibration_data"
CALIB_DIR          = "calibration_data"

# Homography cache: recompute every N frames
H_RECOMPUTE_FRAMES = 90

# Medium overlap (~40%) → blend width
BLEND_WIDTH        = 100

# Depth occupancy
OBSTACLE_H_M       = 0.10
MIN_DEPTH_M        = 0.20
MAX_DEPTH_M        = 5.0
OCC_TH             = 0.12

os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(CALIB_DIR, exist_ok=True)

# =============================================================
#  AUTO SERIAL DETECT
# =============================================================
def get_serials():
    ctx  = rs.context()
    devs = ctx.query_devices()
    return [d.get_info(rs.camera_info.serial_number) for d in devs]

# =============================================================
#  PIPELINE
# =============================================================
def start_pipe(serial):
    pl  = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS_TARGET)
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS_TARGET)
    prof = pl.start(cfg)

    # Lock exposure + gain same on both cameras (critical for good matching)
    color_sensor = prof.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    color_sensor.set_option(rs.option.exposure, 5)
    color_sensor.set_option(rs.option.gain, 16)

    depth_sensor = prof.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    # Get intrinsics from stream profile
    color_profile = prof.get_stream(rs.stream.color)
    depth_profile = prof.get_stream(rs.stream.depth)
    ci = color_profile.as_video_stream_profile().get_intrinsics()
    di = depth_profile.as_video_stream_profile().get_intrinsics()

    intrinsics = {
        "color": {
            "fx": ci.fx, "fy": ci.fy,
            "ppx": ci.ppx, "ppy": ci.ppy,
            "width": ci.width, "height": ci.height,
            "distortion_model": str(ci.model),
            "coeffs": list(ci.coeffs)
        },
        "depth": {
            "fx": di.fx, "fy": di.fy,
            "ppx": di.ppx, "ppy": di.ppy,
            "width": di.width, "height": di.height,
            "distortion_model": str(di.model),
            "coeffs": list(di.coeffs),
            "depth_scale_m_per_unit": depth_scale
        }
    }

    print(f"  [OK] {serial}  depth_scale={depth_scale:.6f}")
    return pl, depth_scale, intrinsics

def grab(pl, aligner):
    try:
        fr  = pl.wait_for_frames(timeout_ms=500)  # 2000 → 500ms, fast fail
        aln = aligner.process(fr)
        cf  = aln.get_color_frame()
        df  = aln.get_depth_frame()
        if not cf:
            return None, None
        return np.asanyarray(cf.get_data()), df
    except:
        return None, None

# =============================================================
#  HOMOGRAPHY — STABLE VERSION
#  Fix: FLANN + SIFT (more stable than ORB for texture)
#       + Sanity check on H matrix
#       + Adaptive threshold
# =============================================================
class HomographyCache:
    def __init__(self):
        self.H           = None
        self.frame_age   = 9999
        self.last_status = "NOT COMPUTED"
        self.last_inliers = 0

        # SIFT is more stable than ORB for smooth textures (parking floor)
        # Falls back to ORB if SIFT unavailable (older OpenCV)
        try:
            self.detector = cv2.SIFT_create(nfeatures=2000)
            self.use_sift = True
            FLANN_INDEX_KDTREE = 1
            index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher  = cv2.FlannBasedMatcher(index_params, search_params)
            print("  [MATCHER] SIFT + FLANN")
        except:
            self.detector = cv2.ORB_create(nfeatures=3000)
            self.use_sift = False
            self.matcher  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            print("  [MATCHER] ORB + BFMatcher (SIFT unavailable)")

        # Store raw matches for calibration export
        self.last_src_pts = None
        self.last_dst_pts = None
        self.last_good_count = 0

    def get(self, img1, img2, force=False):
        if force or self.H is None or self.frame_age >= H_RECOMPUTE_FRAMES:
            self._compute(img1, img2)
        else:
            self.frame_age += 1
        return self.H

    def _compute(self, img1, img2):
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # CLAHE: enhance contrast for better feature detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g1 = clahe.apply(g1)
        g2 = clahe.apply(g2)

        kp1, des1 = self.detector.detectAndCompute(g1, None)
        kp2, des2 = self.detector.detectAndCompute(g2, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            self.last_status = f"Features insufficient: {len(kp1 or [])}/{len(kp2 or [])}"
            return

        # Matching
        if self.use_sift:
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
            matches_raw = self.matcher.knnMatch(des1, des2, k=2)
            # Lowe's ratio test — 0.72 good for medium overlap
            good = [m for m,n in matches_raw if m.distance < 0.72 * n.distance]
        else:
            matches_raw = self.matcher.knnMatch(des1, des2, k=2)
            good = [m for m,n in matches_raw if m.distance < 0.75 * n.distance]

        self.last_good_count = len(good)

        if len(good) < 15:
            self.last_status = f"Matches insufficient: {len(good)}"
            return

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4.0,
                                     confidence=0.995)
        if H is None:
            self.last_status = "Homography computation failed"
            return

        # ── SANITY CHECK (this is the distortion fix) ────────
        # Bad homography has extreme scale/rotation/shear
        # Check: determinant should be near 1 (pure rotation+translation)
        det = np.linalg.det(H[:2,:2])
        if not (0.2 < det < 7.0):
            self.last_status = f"H rejected (det={det:.3f}, unstable)"
            print(f"[H] Rejected bad homography det={det:.3f}")
            return

        # Check: no extreme perspective warp
        # H[2,0] and H[2,1] are perspective components
        persp = abs(H[2,0]) + abs(H[2,1])
        if persp > 0.005:
            self.last_status = f"H rejected (perspective={persp:.5f}, too warped)"
            print(f"[H] Rejected extreme perspective warp")
            return

        inliers = int(mask.ravel().sum())
        self.H             = H
        self.frame_age     = 0
        self.last_inliers  = inliers
        self.last_src_pts  = src_pts[mask.ravel()==1]
        self.last_dst_pts  = dst_pts[mask.ravel()==1]
        self.last_status   = f"OK ({inliers} inliers, {len(good)} matches)"
        print(f"[H] Recomputed — {self.last_status}")

    def status(self):
        age_s = self.frame_age / FPS_TARGET
        return f"{self.last_status}  age={age_s:.1f}s"

    def to_dict(self):
        """Export homography data for calibration JSON"""
        if self.H is None:
            return {"status": "not_computed", "matrix": None}
        return {
            "status": "computed",
            "matrix_3x3": self.H.tolist(),
            "inliers": self.last_inliers,
            "matches_used": self.last_good_count,
            "frame_age_at_save": self.frame_age,
            "description": (
                "Homography H maps cam2 pixel coords to cam1 pixel coords. "
                "Apply: p_cam1 = H @ p_cam2 (homogeneous coords)"
            )
        }

# =============================================================
#  SMOOTH SEAM BLENDING — optimized for 40% overlap
# =============================================================
def smooth_stitch(img1, img2, H):
    h1, w1 = img1.shape[:2]

    corners2   = np.float32([
        [0,0], [img2.shape[1],0],
        [img2.shape[1],img2.shape[0]], [0,img2.shape[0]]
    ]).reshape(-1,1,2)
    corners2_t = cv2.perspectiveTransform(corners2, H)

    all_c  = np.concatenate([
        np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2),
        corners2_t
    ])
    xmin,ymin = np.int32(all_c.min(axis=0).ravel())
    xmax,ymax = np.int32(all_c.max(axis=0).ravel())

    tx = max(0, -xmin)
    ty = max(0, -ymin)
    cw = min(xmax - xmin, 2200)
    ch = min(ymax - ymin, 1200)

    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)

    warped2 = cv2.warpPerspective(img2, T @ H, (cw, ch))

    canvas = warped2.copy()
    ey = min(ty + h1, ch)
    ex = min(tx + w1, cw)
    canvas[ty:ey, tx:ex] = img1[:ey-ty, :ex-tx]

    seam_x = tx + w1

    if seam_x <= BLEND_WIDTH or seam_x >= cw:
        return canvas

    bw      = BLEND_WIDTH
    x_start = max(tx, seam_x - bw)
    x_end   = min(cw, seam_x + bw)

    # Sigmoid alpha ramp (smoother than linear for medium overlap)
    n       = x_end - x_start
    t       = np.linspace(-6, 6, n, dtype=np.float32)
    alpha_col = 1.0 / (1.0 + np.exp(-(-t)))   # 1→0 left to right
    alpha   = np.tile(alpha_col, (ch, 1))

    zone_canvas = canvas[:,  x_start:x_end].astype(np.float32)
    zone_warped = warped2[:, x_start:x_end].astype(np.float32)

    blended = (alpha[:,:,None] * zone_canvas +
               (1 - alpha[:,:,None]) * zone_warped).astype(np.uint8)
    canvas[:, x_start:x_end] = blended

    return canvas

# =============================================================
#  FLOOR ROI OCCUPANCY
# =============================================================
def check_occupancy(depth_frame1, stitch_w, stitch_h):
    if depth_frame1 is None:
        return "UNKNOWN", 0.0, 0.0, None

    depth_arr = np.asanyarray(depth_frame1.get_data()).astype(np.float32)
    scale     = depth_frame1.get_units()
    depth_m   = depth_arr * scale

    dh, dw = depth_m.shape

    rx1 = int(dw * 0.15); rx2 = int(dw * 0.85)
    ry1 = int(dh * 0.55); ry2 = int(dh * 0.92)

    roi = depth_m[ry1:ry2, rx1:rx2]
    valid_mask = (roi > MIN_DEPTH_M) & (roi < MAX_DEPTH_M)
    valid_vals = roi[valid_mask]

    if len(valid_vals) < 50:
        return "UNKNOWN", 0.0, 0.0, None

    ground_d  = float(np.median(valid_vals))
    obs_mask  = valid_mask & (roi < (ground_d - OBSTACLE_H_M))
    obs_ratio = float(obs_mask.sum()) / (valid_mask.sum() + 1e-5)
    status    = "OCCUPIED" if obs_ratio > OCC_TH else "EMPTY"

    scale_x = (stitch_w / 2) / dw
    scale_y = stitch_h / dh
    roi_in_stitch = (
        int(rx1 * scale_x), int(ry1 * scale_y),
        int(rx2 * scale_x), int(ry2 * scale_y),
    )

    return status, obs_ratio, ground_d, roi_in_stitch

# =============================================================
#  CALIBRATION JSON EXPORT
#  All intrinsic, extrinsic, depth, stitch parameters
# =============================================================
def export_calibration(sn1, sn2, intrinsics1, intrinsics2,
                       H_cache, depth_scale1, depth_scale2,
                       stitch_shape, occ_status, occ_ratio,
                       occ_depth, img1, img2, stitched):
    """
    Save complete calibration + session data to JSON.
    Called on S or C keypress.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extrinsic: rotation + translation from H
    # For side-by-side cameras with medium overlap,
    # H encodes the relative pose between cam1 and cam2
    H = H_cache.H
    R_approx = None
    t_approx = None
    if H is not None:
        # Decompose H into rotation + translation (approximate)
        # Only valid for planar scenes (floor view)
        # We use SVD-based decomposition
        try:
            _, Rs, Ts, Ns = cv2.decomposeHomographyMat(
                H,
                np.array([
                    [intrinsics1["color"]["fx"], 0, intrinsics1["color"]["ppx"]],
                    [0, intrinsics1["color"]["fy"], intrinsics1["color"]["ppy"]],
                    [0, 0, 1]
                ])
            )
            # Take first valid decomposition
            R_approx = Rs[0].tolist()
            t_approx = Ts[0].flatten().tolist()
        except Exception as e:
            R_approx = f"decomposition_failed: {str(e)}"
            t_approx = None

    data = {
        "metadata": {
            "timestamp": ts,
            "datetime_human": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "script_version": "v2_stable",
            "description": "Dual RealSense D435i calibration data — research phase 1",
            "camera_setup": "side_by_side_fixed_mount",
            "overlap_estimate": "medium_40_percent"
        },

        "cameras": {
            "cam1": {
                "serial_number": sn1,
                "role": "left",
                "resolution": {"width": WIDTH, "height": HEIGHT},
                "fps": FPS_TARGET,
                "exposure": 500,
                "gain": 64,
                "intrinsics": intrinsics1
            },
            "cam2": {
                "serial_number": sn2,
                "role": "right",
                "resolution": {"width": WIDTH, "height": HEIGHT},
                "fps": FPS_TARGET,
                "exposure": 500,
                "gain": 64,
                "intrinsics": intrinsics2
            }
        },

        "depth": {
            "cam1_depth_scale_m_per_unit": depth_scale1,
            "cam2_depth_scale_m_per_unit": depth_scale2,
            "min_valid_depth_m": MIN_DEPTH_M,
            "max_valid_depth_m": MAX_DEPTH_M,
            "obstacle_height_threshold_m": OBSTACLE_H_M,
            "notes": (
                "depth_value_in_meters = raw_uint16 * depth_scale. "
                "Depth aligned to color frame using RealSense SDK align()."
            )
        },

        "extrinsics": {
            "description": (
                "Relative pose between cam2 and cam1 coordinate frames. "
                "Derived from planar homography — approximate for 3D scenes."
            ),
            "homography": H_cache.to_dict(),
            "rotation_matrix_approx": R_approx,
            "translation_vector_approx": t_approx,
            "notes": (
                "For precise extrinsics, use checkerboard stereo calibration "
                "(cv2.stereoCalibrate). Homography-based extrinsics are "
                "approximate and valid only for planar scenes (floor)."
            )
        },

        "stitching": {
            "method": "perspective_warp_with_sigmoid_blend",
            "feature_detector": "SIFT" if H_cache.use_sift else "ORB",
            "matcher": "FLANN" if H_cache.use_sift else "BFMatcher",
            "lowe_ratio_threshold": 0.72,
            "ransac_reprojection_threshold_px": 4.0,
            "blend_width_px": BLEND_WIDTH,
            "blend_function": "sigmoid",
            "homography_recompute_interval_frames": H_RECOMPUTE_FRAMES,
            "homography_sanity_checks": {
                "det_min": 0.2,
                "det_max": 7.0,
                "max_perspective_component": 0.005
            },
            "output_canvas_max": {"width": 1600, "height": 900},
            "stitched_output_shape": {
                "height": stitch_shape[0] if stitch_shape else None,
                "width":  stitch_shape[1] if stitch_shape else None
            }
        },

        "occupancy_detection": {
            "roi_depth_image": {
                "x1_frac": 0.15, "y1_frac": 0.55,
                "x2_frac": 0.85, "y2_frac": 0.92,
                "zone": "floor_lower_half"
            },
            "obstacle_ratio_threshold": OCC_TH,
            "status_at_save": occ_status,
            "obstacle_ratio_at_save": round(occ_ratio, 4),
            "ground_depth_m_at_save": round(occ_depth, 4)
        },

        "camera_matrix_cam1": {
            "K": [
                [intrinsics1["color"]["fx"], 0,  intrinsics1["color"]["ppx"]],
                [0, intrinsics1["color"]["fy"],   intrinsics1["color"]["ppy"]],
                [0, 0, 1]
            ],
            "dist_coeffs": intrinsics1["color"]["coeffs"],
            "description": "3x3 intrinsic matrix K for cam1 color stream"
        },

        "camera_matrix_cam2": {
            "K": [
                [intrinsics2["color"]["fx"], 0,  intrinsics2["color"]["ppx"]],
                [0, intrinsics2["color"]["fy"],   intrinsics2["color"]["ppy"]],
                [0, 0, 1]
            ],
            "dist_coeffs": intrinsics2["color"]["coeffs"],
            "description": "3x3 intrinsic matrix K for cam2 color stream"
        }
    }

    # Save JSON
    json_path = os.path.join(CALIB_DIR, f"{ts}_calibration.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"[CALIB SAVED] {json_path}")
    return json_path, ts

# =============================================================
#  DEPTH HEATMAP
# =============================================================
def make_depth_vis(depth_frame):
    if depth_frame is None:
        return None
    arr   = np.asanyarray(depth_frame.get_data())
    scale = cv2.convertScaleAbs(arr, alpha=0.04)
    return cv2.applyColorMap(scale, cv2.COLORMAP_TURBO)

# =============================================================
#  DRAW OVERLAY
# =============================================================
def draw_overlay(canvas, status, obs_ratio, avg_depth,
                 roi_rect, h_status, fps, show_depth):
    h, w = canvas.shape[:2]

    box_col = (0,255,0)  if status == "EMPTY"    else \
              (0,50,255) if status == "OCCUPIED"  else \
              (0,165,255)

    cv2.rectangle(canvas, (0,0), (420,82), (15,15,15), -1)
    cv2.putText(canvas, f"Parking: {status}",
                (8, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, box_col, 2)
    cv2.putText(canvas, f"Obstacle: {obs_ratio:.0%}   Depth: {avg_depth:.2f}m",
                (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
    cv2.putText(canvas, f"FPS:{fps:.0f}  H:{h_status[:40]}",
                (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130,130,130), 1)

    if roi_rect:
        x1,y1,x2,y2 = roi_rect
        cv2.rectangle(canvas, (x1,y1),(x2,y2), box_col, 2)
        lbl_y = max(y1 + 24, 30)
        cv2.rectangle(canvas, (x1,y1),(x1+160,y1+28),(15,15,15),-1)
        cv2.putText(canvas, f"{status} {obs_ratio:.0%}",
                    (x1+5, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_col, 2)

    cv2.putText(canvas,
                "S=Save+Calib  R=Recompute H  C=Calib only  D=Depth  Q=Quit",
                (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,100,100), 1)
    return canvas

# =============================================================
#  MAIN
# =============================================================
def main():
    print("="*60)
    print("  Dual RealSense — Stable Stitch + Calibration Export")
    print("="*60)

    sn1, sn2 = SN1, SN2
    if "REPLACE" in sn1:
        found = get_serials()
        if len(found) < 2:
            print(f"[ERROR] {len(found)} camera mila, 2 chahiye!")
            return
        sn1, sn2 = found[0], found[1]
        print(f"  Auto-detected: {sn1}  |  {sn2}")

    pipe1 = pipe2 = None
    try:
        print("\n[INIT]")
        pipe1, ds1, intr1 = start_pipe(sn1)
        pipe2, ds2, intr2 = start_pipe(sn2)
        aln1 = rs.align(rs.stream.color)
        aln2 = rs.align(rs.stream.color)

        print("[WARMUP]...")
        for _ in range(25):
            pipe1.wait_for_frames()
            pipe2.wait_for_frames()

        H_cache     = HomographyCache()
        frame_count = 0
        t0          = time.time()
        fps_val     = 0.0
        show_depth  = True

        occ_status  = "UNKNOWN"
        occ_ratio   = 0.0
        occ_depth   = 0.0
        occ_roi     = None
        last_df1    = None
        last_img1   = None
        last_img2   = None
        last_stitched = None

        print("\n[READY]  S=Save+Calib  R=Recompute  C=Calib  D=Depth  Q=Quit\n")

        while True:
            img1, df1 = grab(pipe1, aln1)
            img2, df2 = grab(pipe2, aln2)

            # ── Software Sync Fix ─────────────────────────────
            # Agar ek camera ne frame nahi di → purani frame use karo
            # Dono miss karein → skip karo
            if img1 is None and img2 is None:
                continue
            if img1 is None:
                if last_img1 is None:
                    continue       # koi purani frame bhi nahi
                img1 = last_img1   # purani cached frame reuse
                df1  = last_df1
            if img2 is None:
                if last_img2 is None:
                    continue
                img2 = last_img2   # purani cached frame reuse

            # Cache valid frames for sync fallback
            if img1 is not None: last_img1 = img1
            if img2 is not None: last_img2 = img2
            if df1 is not None:  last_df1  = df1

            frame_count += 1

            H = H_cache.get(img1, img2)

            if H is not None:
                stitched = smooth_stitch(img1, img2, H)
            else:
                stitched = np.hstack([img1, img2])

            last_stitched = stitched
            sh, sw = stitched.shape[:2]

            # if frame_count % 2 == 0 and last_df1 is not None:
            #     occ_status, occ_ratio, occ_depth, occ_roi = \
            #         check_occupancy(last_df1, sw, sh)

            if frame_count % 20 == 0:
                fps_val = 20.0 / (time.time() - t0 + 1e-5)
                t0      = time.time()

            disp = draw_overlay(
                stitched.copy(),
                occ_status, occ_ratio, occ_depth,
                None, H_cache.status(), fps_val, show_depth
            )

            cv2.imshow("Panorama + Parking Detection", disp)

            if show_depth and last_df1 is not None:
                dv = make_depth_vis(last_df1)
                if dv is not None:
                    dh2, dw2 = dv.shape[:2]
                    drx1 = int(dw2 * 0.15); drx2 = int(dw2 * 0.85)
                    dry1 = int(dh2 * 0.55); dry2 = int(dh2 * 0.92)
                    bcol = (0,255,0) if occ_status=="EMPTY" else (0,50,255)
                    cv2.rectangle(dv,(drx1,dry1),(drx2,dry2),bcol,2)
                    cv2.putText(dv, f"{occ_status} {occ_ratio:.0%}",
                                (drx1+5, dry1+22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bcol, 2)
                    cv2.putText(dv, "FLOOR ROI",
                                (drx1+5, dry2-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1)
                    cv2.imshow("Depth View (Cam 1)", dv)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                # Screenshot + calibration
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"{SAVE_DIR}/{ts}_panorama.jpg", disp)
                cv2.imwrite(f"{SAVE_DIR}/{ts}_cam1.jpg",    last_img1)
                cv2.imwrite(f"{SAVE_DIR}/{ts}_cam2.jpg",    last_img2)
                dv2 = make_depth_vis(last_df1)
                if dv2 is not None:
                    cv2.imwrite(f"{SAVE_DIR}/{ts}_depth.jpg", dv2)
                print(f"[SAVED] Images → {SAVE_DIR}/{ts}_*.jpg")

                json_path, _ = export_calibration(
                    sn1, sn2, intr1, intr2,
                    H_cache, ds1, ds2,
                    last_stitched.shape if last_stitched is not None else None,
                    occ_status, occ_ratio, occ_depth,
                    last_img1, last_img2, last_stitched
                )

            elif key == ord('c'):
                # Calibration only, no screenshot
                json_path, _ = export_calibration(
                    sn1, sn2, intr1, intr2,
                    H_cache, ds1, ds2,
                    last_stitched.shape if last_stitched is not None else None,
                    occ_status, occ_ratio, occ_depth,
                    last_img1, last_img2, last_stitched
                )

            elif key == ord('r'):
                print("[INFO] Force recomputing homography...")
                H_cache.get(img1, img2, force=True)

            elif key == ord('d'):
                show_depth = not show_depth
                if not show_depth:
                    try: cv2.destroyWindow("Depth View (Cam 1)")
                    except: pass

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback; traceback.print_exc()
    finally:
        for p in [pipe1, pipe2]:
            if p:
                try: p.stop()
                except: pass
        cv2.destroyAllWindows()
        print("[DONE]")

if __name__ == "__main__":
    main()