## TODO 
## sprawdz docu co doimplementwoac np elipsy
## policz meany i wariancje i per blok etc, dla wszystkich filmikow, zunifikowane dla poprzednich danych
"""
elbow_metric_example_integrated.py

Integrated overlay that combines the original elbow/shoulder angle metrics
with full fingertip measures from the performance spec. It uses the
end-of-finger keypoint (index 124) and assumes the movement window is the full
trial: from frame 0 to the last frame of the video.

Usage:
  python elbow_metric_example_integrated.py \
      --json /path/to/results_<name>.json \
      --video /path/to/<name>.avi \
      --out_video /path/to/<name>_overlay.mp4 \
      --out_csv /path/to/<name>_metrics.csv \
      --px_per_cm 40 \
      --target_csv /path/to/target_xy.csv  # optional (t,x,y)

Notes:
- If --target_csv is not provided, TE and PoL are skipped (NaN/0), the rest compute.
- If you want speeds in cm/s, pass --px_per_cm.
- Uses RIGHT-side joints from kinematic_metrics.RIGHT mapping.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))  # allow importing sibling module

import csv
import argparse
from typing import Optional
import numpy as np
import cv2
import json

from kinematic_metrics import (
    RIGHT,
    TrialEventTimes, Trial2DSeries, KeypointSeries,
    movement_time, touching_error_px, predicting_or_lagging,
    fingertip_speed, fingertip_path_length, fingertip_PV_AV_P2PV,
    D2T_metrics,
    shoulder_elevation_series, elbow_flexion_series, angle_velocity_metrics,
    time_lag_peak_velocity, cc_joint_angles, average_continuous_relative_phase,
    time_normalized_metrics, time_normalize_angle_phase_samples, sample_entropy, norm_rows
)

# ------------------------ I/O helpers ------------------------

def load_keypoints(json_path: str) -> np.ndarray:
    """Load MMPose whole-body JSON (one subject per frame) -> [T, J, 3]."""
    data = json.loads(Path(json_path).read_text())
    frames = []
    for f in data.get("instance_info", []):
        if "keypoints_2d" in f and f["keypoints_2d"].get("__ndarray__"):
            kp = np.array(f["keypoints_2d"]["__ndarray__"][0], dtype=float)
            frames.append(kp)
        else:
            # If missing, hold previous
            frames.append(frames[-1] if frames else None)
    frames = [f for f in frames if f is not None]
    if not frames:
        raise ValueError("No keypoints found in JSON.")
    return np.stack(frames, axis=0)  # [T, J, 3]


def load_target_csv(path: Optional[str], t_ref: np.ndarray) -> np.ndarray:
    """
    Load target center trajectory CSV with header t,x,y and resample to t_ref.
    Returns array [T,2]. If path is None, returns zeros.
    """
    if path is None:
        return np.zeros((t_ref.size, 2), dtype=float)
    import pandas as pd
    df = pd.read_csv(path)
    if not set(["t", "x", "y"]).issubset(df.columns):
        raise ValueError("target_csv must have columns: t,x,y")
    tx = np.interp(t_ref, df["t"].to_numpy(), df["x"].to_numpy())
    ty = np.interp(t_ref, df["t"].to_numpy(), df["y"].to_numpy())
    return np.column_stack([tx, ty])


def fingertip_xy_series(kp_TxJxK: np.ndarray, end_of_finger_index: int = 124) -> np.ndarray:
    return kp_TxJxK[:, end_of_finger_index, :2]


def make_events_from_frames(T: int, fps: float) -> TrialEventTimes:
    """
    Movement window is from frame 0 to frame T-1 (per integration spec).
    """
    t0 = 0.0
    t_end = (T - 1) / fps if fps > 0 else T - 1
    return TrialEventTimes(key_release=t0, touch=t_end)


# ------------------------ Drawing helpers ------------------------

def put_multiline_text(img, lines, org=(10, 30), line_h=22, font=cv2.FONT_HERSHEY_SIMPLEX,
                       scale=0.6, thickness=1, color=(255,255,255), bg=None):
    x, y = org
    if bg is not None:
        wbox = max([cv2.getTextSize(s, font, scale, thickness)[0][0] for s in lines]) + 10
        hbox = line_h * len(lines) + 10
        cv2.rectangle(img, (x-5, y-20), (x-5+wbox, y-20+hbox), bg, -1)
    for i, s in enumerate(lines):
        yy = y + i*line_h
        cv2.putText(img, s, (x, yy), font, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(
    frame,
    t_s: float,
    fps: float,
    w: int, h: int,
    left_lines,
    right_lines,
    events: TrialEventTimes
):
    # Highlight movement window
    if events.key_release <= t_s <= events.touch:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 120, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Left column
    put_multiline_text(frame, left_lines, org=(10, 28), bg=(0,0,0))

    # Right column
    sizes = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for s in right_lines]
    block_w = max(sizes) + 20 if sizes else 250
    put_multiline_text(frame, right_lines, org=(w - block_w, 28), bg=(0,0,0))

    return frame


# ------------------------ Main ------------------------
def pixel_distance(point1, point2):
    """Calculate Euclidean distance between two points in pixels."""
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def main(write_video=False,json_path=None, video_path=None, out_video_path=None, out_csv_path=None, px_per_cm=None):
    """
    Main function to overlay elbow/shoulder + fingertip kinematics and export trial metrics.

    Parameters (optional):
        json_path (str): Path to MMPose JSON (one subject per frame).
        video_path (str): Corresponding raw video path.
        out_video_path (str): Output overlay video path (.mp4).
        out_csv_path (str): Output CSV with trial-level metrics.
        px_per_cm (float): Pixels per cm (optional). If given, speeds are in cm/s.
    """

    # Only parse CLI arguments if not passed directly
    if all(v is None for v in [json_path, video_path, out_video_path, out_csv_path]):
        ap = argparse.ArgumentParser(description="Overlay elbow/shoulder + fingertip kinematics and export trial metrics.")
        ap.add_argument("--json", required=True, help="Path to MMPose JSON (one subject per frame).")
        ap.add_argument("--video", required=True, help="Corresponding raw video path.")
        ap.add_argument("--out_video", required=True, help="Output overlay video path (.mp4).")
        ap.add_argument("--out_csv", required=True, help="Output CSV with trial-level metrics.")
        ap.add_argument(
            "--px_per_cm",
            type=float,
            default=pixel_distance([735, 506], [945, 506]) / 23.5,
            help="Pixels per cm. If given, speeds are in cm/s."
        )
        args = ap.parse_args()
        json_path = args.json
        video_path = args.video
        out_video_path = args.out_video
        out_csv_path = args.out_csv
        px_per_cm = args.px_per_cm
    else:
        # If px_per_cm not given in function call, set default
        if px_per_cm is None:
            px_per_cm = pixel_distance([735, 506], [945, 506]) / 23.5

    # --- Load data and basic timebase ---
    kp = load_keypoints(json_path)  # [T, J, 3]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T = min(kp.shape[0], n_frames_vid)
    t = np.arange(T) / fps

    # --- Movement events: frame 0 to last frame ---
    events = make_events_from_frames(T, fps)

    # --- Fingertip series and target ---
    ft_xy = fingertip_xy_series(kp, RIGHT["end_of_finger"])[:T]
    touch_xy = ft_xy[-1]

    trial = Trial2DSeries(
        t=t,
        fingertip_xy=ft_xy,
        touch_xy=touch_xy,
        px_per_cm=px_per_cm
    )
    
    # --- Fingertip metrics ---
    MT = movement_time(events)

    fpv = fingertip_PV_AV_P2PV(trial, events)
    PV, AV, P2PV = fpv["PV"], fpv["AV"], fpv["P2PV"]
    #distance_to_screen_cm = np.linalg.norm(ft_xy - touch_xy, axis=1) / args.px_per_cm
    seg = np.diff(trial.fingertip_xy, axis=0)
    d = norm_rows(seg) / trial.px_per_cm
    cumulative = np.cumsum(d)
    total_dist = np.sum(d)
    cumulative = np.append(0,cumulative)
    distance_to_screen_cm = cumulative - total_dist
    
    d2_metrics = D2T_metrics(trial, events, distance_to_screen_cm=distance_to_screen_cm, initial_distance_cm=None, threshold_frac=0.10)
    D2TPV, D2TEM = d2_metrics["D2TPV"], d2_metrics["D2TEM"]
    XYPV_x, XYPV_y = d2_metrics["XYPV_x"], d2_metrics["XYPV_y"]
    XYEM_x, XYEM_y = d2_metrics["XYEM_x"], d2_metrics["XYEM_y"]
    path_len = fingertip_path_length(trial)

    spd, _ = fingertip_speed(trial)

    # --- Joint angles and derived metrics ---
    kps = KeypointSeries(
        t=t,
        shoulder=kp[:T, RIGHT["shoulder"], :2],
        elbow=kp[:T, RIGHT["elbow"], :2],
        wrist=kp[:T, RIGHT["wrist"], :2],
        hip=kp[:T, RIGHT["hip"], :2]
    )
    shoulder_deg, _ = shoulder_elevation_series(kps)
    elbow_deg, _ = elbow_flexion_series(kps)

    sh = angle_velocity_metrics(shoulder_deg, t, events)
    el = angle_velocity_metrics(elbow_deg, t, events)

    # Coordination
    w_sh = np.gradient(shoulder_deg, t)
    w_el = np.gradient(elbow_deg, t)
    TLPV = time_lag_peak_velocity(t, w_sh, t, w_el, events)
    CCJA = cc_joint_angles(shoulder_deg, elbow_deg)
    ACRP, arcp_series = average_continuous_relative_phase(shoulder_deg, t, elbow_deg, t)

    # Variability example
    SaEn_elbow = sample_entropy(elbow_deg)
    SaEn_shoulder = sample_entropy(shoulder_deg)
    SaEn_arcp = sample_entropy(arcp_series)

    # --- Prepare overlay writer ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if write_video:
        out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    # Identify PV and EM indices on movement window
    mask = (t >= events.key_release) & (t <= events.touch)
    t_m = t[mask]; s_m = spd[mask]
    pv_idx_local = int(np.nanargmax(s_m)) if np.any(np.isfinite(s_m)) else 0
    pv_t = t_m[pv_idx_local] if len(t_m) else 0.0
    pv_global_idx = int(np.searchsorted(t, pv_t))
    # EM: first time speed <= 10% PV after PV
    below = np.where(s_m[pv_idx_local:] <= 0.10 * np.nanmax(s_m))[0] if np.any(np.isfinite(s_m)) else []
    em_idx_local = pv_idx_local + int(below[0]) if len(below) else len(s_m) - 1
    em_t = t_m[em_idx_local] if len(t_m) else t[-1]
    em_global_idx = int(np.searchsorted(t, em_t))
    
    # Per-trial TNA/TNP at fixed normalized times (e.g., 5 points → 20% steps)
    sh_bins = time_normalize_angle_phase_samples(shoulder_deg[mask], t_m, n_points=5)
    el_bins = time_normalize_angle_phase_samples(elbow_deg[mask],    t_m, n_points=5)

    # --- Overlay loop ---
    for i in range(T):
        if write_video:
            ok, frame = cap.read()
            if not ok:
                break
    #+315
        # Draw fingertip trajectory so far
        pts = ft_xy[:i+1].astype(int)
        offset_y = 285
        pts[:,1] += offset_y
        if write_video:
            for j in range(1, len(pts)):
                cv2.line(frame, tuple(pts[j-1]), tuple(pts[j]), (0, 255, 255), 2)

        
            # Draw PV and EM markers if time passed
            if pv_global_idx <= i: 
                cv2.circle(frame, tuple(ft_xy[pv_global_idx].astype(int) + [0, offset_y]), 6, (0, 0, 255), -1)
            if em_global_idx <= i:
                cv2.circle(frame, tuple(ft_xy[em_global_idx].astype(int) + [0, offset_y]), 6, (255, 0, 0), -1)

            # Draw final touch point
            cv2.circle(frame, tuple(ft_xy[-1].astype(int) + [0, offset_y]), 6, (0, 255, 0), -1)
        
        # Compose left/right text blocks
        t_i = t[i]
        # per-frame text (left)
        left_lines = [
            f"t = {t_i:6.3f}s  (fps={fps:.1f})",
            f"Finger speed: {spd[i]:6.2f}" + (" cm/s" if px_per_cm else " px/s"),
            f"Shoulder: {shoulder_deg[i]:7.2f} deg   |w|={abs(w_sh[i]):6.2f} deg/s",
            f"Elbow:    {elbow_deg[i]:7.2f} deg   |w|={abs(w_el[i]):6.2f} deg/s",
        ]

        # trial metrics (right)
        right_lines = [
            "Trial metrics (movement window):",
            f"MT={MT:.3f}s",
            f"PV={PV:.2f}  AV={AV:.2f}  P2PV={P2PV:.1f}%",
            f"D2TPV={D2TPV:.1f}%  D2TEM={D2TEM:.1f}%",
            f"XYPV=({XYPV_x:.1f},{XYPV_y:.1f})  XYEM=({XYEM_x:.1f},{XYEM_y:.1f})",
            f"PathLen={path_len:.1f}",
            f"SH: PV={sh['PV']:.2f} deg/s  AV={sh['AV']:.2f}  ROM={sh['ROM']:.2f}",
            f"    P2PV={sh['P2PV']:.1f}%  AA={sh['AA']:.2f}  SA={sh['SA']:.2f}",
            f"    AUMC={sh['AUMC']:.2f}  AvgPh={sh['AvgPh']:.1f} deg",
            f"EL: PV={el['PV']:.2f} deg/s  AV={el['AV']:.2f}  ROM={el['ROM']:.2f}",
            f"    P2PV={el['P2PV']:.1f}%  AA={el['AA']:.2f}  SA={el['SA']:.2f}",
            f"    AUMC={el['AUMC']:.2f}  AvgPh={el['AvgPh']:.1f} deg",
            f"Coord: TLPV={TLPV:.1f}%  CCJA={CCJA:.3f}  ACRP={ACRP:.1f} deg",
            f"SaEn(elbow)={SaEn_elbow:.3f} SaEn(shoulder)={SaEn_shoulder:.3f} SaEn(ACRP)={SaEn_arcp:.3f}",
        ]


        if write_video:
            frame = draw_overlay(frame, t_i, fps, w, h, left_lines, right_lines, events)
            out.write(frame)

    cap.release()
    if write_video:
        out.release()

    # --- Save trial-level CSV ---
    rows = {
        "MT_s": MT,
        "PV": PV, "AV": AV, "P2PV_pct": P2PV,
        "D2TPV_pct": D2TPV, "D2TEM_pct": D2TEM,
        "XYPV_x": XYPV_x, "XYPV_y": XYPV_y,
        "XYEM_x": XYEM_x, "XYEM_y": XYEM_y,
        "PathLen": path_len,
        **{f"sh_{k}": v for k, v in sh.items()},
        **{f"el_{k}": v for k, v in el.items()},
        "coord_TLPV_pct": TLPV, "coord_CCJA": CCJA, "coord_ACRP_deg": ACRP,
        "SaEn_elbow": SaEn_elbow,
        "SaEn_shoulder": SaEn_shoulder,
        "SaEn_ACRP": SaEn_arcp,
    }
    for i, v in enumerate(sh_bins["tau"], 1):
        rows[f"tau_t{i}"] = v
    for i, v in enumerate(sh_bins["TNA"], 1):
        rows[f"sh_TNA_t{i}"] = v
    for i, v in enumerate(sh_bins["TNP"], 1):
        rows[f"sh_TNP_t{i}"] = v
    for i, v in enumerate(el_bins["TNA"], 1):
        rows[f"el_TNA_t{i}"] = v
    for i, v in enumerate(el_bins["TNP"], 1):
        rows[f"el_TNP_t{i}"] = v

    with open(out_csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(rows.keys())
        wcsv.writerow(rows.values())

    print(f"Overlay saved: {out_video_path}")
    print(f"Metrics CSV:   {out_csv_path}")


if __name__ == "__main__":
    main()
