
"""
kinematic_metrics.py

Utilities to compute kinematic metrics for fingertip reaching and joint-angle dynamics
(shoulder elevation and elbow flexion) from keypoints and touch-task logs.

This module is designed to plug into an existing MMPose-based pipeline (see notes in
`KeypointSeries`) and to complement/extend the example in elbow_metric.py.

Author: ChatGPT
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Iterable
import numpy as np

# =========================
# General helpers
# =========================

def _as_np(a):
    return np.asarray(a, dtype=float)

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Simple causal moving average filter (w must be >= 1)."""
    if w <= 1:
        return x
    c = np.convolve(x, np.ones(w)/w, mode='same')
    # Fix edges to avoid attenuation
    c[:w//2] = c[w//2]
    c[-w//2:] = c[-w//2-1]
    return c

def finite_diff(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central finite difference with edge-safe np.gradient."""
    x = _as_np(x)
    t = _as_np(t)
    return np.gradient(x, t, edge_order=2)

def trapz_integral(y: np.ndarray, t: np.ndarray) -> float:
    """Trapezoidal integral (area under curve)."""
    return float(np.trapz(_as_np(y), _as_np(t)))

def norm_rows(xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xy, axis=-1)

def euclid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return norm_rows(_as_np(a) - _as_np(b))

def normalize_series(x: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] by dividing by its max absolute value (avoid divide-by-zero)."""
    x = _as_np(x)
    m = np.nanmax(np.abs(x))
    if m == 0 or np.isnan(m):
        return np.zeros_like(x)
    return x / m

def wrap_angle_deg(a: np.ndarray) -> np.ndarray:
    """Wrap degrees to (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0

# =========================
# Data containers
# =========================

@dataclass
class TrialEventTimes:
    """
    Event timestamps (sec) for one trial.
    key_release: time the key is released (movement onset t0).
    touch: time of touchscreen contact (movement end t_end).
    """
    key_release: float
    touch: float

@dataclass
class Trial2DSeries:
    """
    Time-aligned 2D trajectories (in px or cm) for a trial.
    t: shape (T,)
    fingertip_xy: shape (T,2) fingertip positions
    touch_xy:    shape (2,) final touch coordinate on the touchscreen
    px_per_cm:   pixels per cm scale. If None, speeds will be in px/s.
    """
    t: np.ndarray
    fingertip_xy: np.ndarray
    touch_xy: np.ndarray
    px_per_cm: Optional[float] = None

@dataclass
class KeypointSeries:
    """
    Time-aligned 3D keypoints (or 2D with z=None) for shoulder, elbow, wrist, and hip.
    Arrays must have shape (T,3) in (x,y,z) or (T,2) if z is not available.
    Indices follow the same side as in your pipeline (right side in elbow_metric.py).

    If you work with MMPose 2D keypoints, pass z=None and angles will be 2D.
    """
    t: np.ndarray
    shoulder: np.ndarray  # (T,3) or (T,2)
    elbow: np.ndarray     # (T,3) or (T,2)
    wrist: np.ndarray     # (T,3) or (T,2)
    hip: np.ndarray       # (T,3) or (T,2)

# =========================
# Fingertip / target metrics
# =========================

def movement_time(events: TrialEventTimes) -> float:
    """MT (s): from key release to screen touch."""
    return float(events.touch - events.key_release)

def touching_error_px(trial: Trial2DSeries, events: TrialEventTimes) -> float:
    """
    TE (px): distance between the moving circle center at touch time and the touch point.
    """
    # Interpolate target to the touch timestamp
    t = trial.t
    tx = np.interp(events.touch, t, trial.target_xy[:,0])
    ty = np.interp(events.touch, t, trial.target_xy[:,1])
    return float(np.hypot(trial.touch_xy[0]-tx, trial.touch_xy[1]-ty))

def predicting_or_lagging(trial: Trial2DSeries, events: TrialEventTimes) -> int:
    """
    PoL: +1 if touch is ahead in the direction of the target's motion at touch,
         -1 if lagging behind.
    """
    t = trial.t
    tgt = trial.target_xy
    # Velocity at touch time
    vx = np.gradient(tgt[:,0], t)
    vy = np.gradient(tgt[:,1], t)
    vxt = np.interp(events.touch, t, vx)
    vyt = np.interp(events.touch, t, vy)
    v = np.array([vxt, vyt])
    if np.allclose(v, 0):
        return 0  # undefined motion, neutral
    tgt_at_touch = np.array([np.interp(events.touch, t, tgt[:,0]),
                             np.interp(events.touch, t, tgt[:,1])])
    disp = _as_np(trial.touch_xy) - tgt_at_touch
    return 1 if float(np.dot(v, disp)) > 0 else -1

def fingertip_speed(trial: Trial2DSeries) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (speed, t) where speed is in cm/s if px_per_cm is given, else px/s.
    """
    t = trial.t
    ft = trial.fingertip_xy
    vx = finite_diff(ft[:,0], t)
    vy = finite_diff(ft[:,1], t)
    spd = np.sqrt(vx**2 + vy**2)
    if trial.px_per_cm and trial.px_per_cm > 0:
        spd = spd / trial.px_per_cm  # convert px/s -> cm/s
    return spd, t

def fingertip_path_length(trial: Trial2DSeries) -> float:
    """Total path length (cm if scale known, else px)."""
    seg = np.diff(trial.fingertip_xy, axis=0)
    d = norm_rows(seg).sum()
    if trial.px_per_cm and trial.px_per_cm > 0:
        d = d / trial.px_per_cm
    return float(d)

def fingertip_PV_AV_P2PV(trial: Trial2DSeries, events: TrialEventTimes) -> Dict[str, float]:
    """
    PV (peak velocity, cm/s or px/s), AV (average velocity), P2PV (%).
    All computed on the movement window [key_release, touch].
    """
    spd, t = fingertip_speed(trial)
    # Restrict to movement window
    mask = (t >= events.key_release) & (t <= events.touch)
    
    t_m = t[mask]
    s_m = spd[mask]
    if len(t_m) < 2:
        return {"PV": np.nan, "AV": np.nan, "P2PV": np.nan}
    pv_idx = int(np.nanargmax(s_m))
    pv = float(s_m[pv_idx])
    mt = movement_time(events)
    av = float(fingertip_path_length(Trial2DSeries(t_m, trial.fingertip_xy[mask], trial.touch_xy, trial.px_per_cm)) / mt)
    p2pv = float(100.0 * (t_m[pv_idx] - events.key_release) / mt)
    return {"PV": pv, "AV": av, "P2PV": p2pv, "t_at_PV": float(t_m[pv_idx])}

def _distance_to_screen_series(distance_to_screen: Optional[np.ndarray],
                               t: np.ndarray,
                               default_initial: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (d(t), t) for distance-to-screen in cm.
    If distance_to_screen is None, fallback to a constant series (no change),
    using default_initial if provided, else zeros.
    """
    if distance_to_screen is None:
        if default_initial is None:
            return np.zeros_like(t), t
        else:
            return np.full_like(t, float(default_initial)), t
    return _as_np(distance_to_screen), t

def D2T_metrics(trial: Trial2DSeries,
                events: TrialEventTimes,
                distance_to_screen_cm: Optional[np.ndarray] = None,
                initial_distance_cm: Optional[float] = None,
                threshold_frac: float = 0.10) -> Dict[str, float]:
    """
    D2TPV (%), D2TEM (%), XYPV (cm,cm), XYEM (cm,cm).

    distance_to_screen_cm: optional series for fingertip distance to the touchscreen plane (cm).
    If not given, you can pass initial_distance_cm and a constant distance will be assumed.
    """
    spd, t = fingertip_speed(trial)
    mask = (t >= events.key_release) & (t <= events.touch)
    t_m = t[mask]
    s_m = spd[mask]
    ft = trial.fingertip_xy[mask]

    if len(t_m) < 3 or np.all(~np.isfinite(s_m)):
        return {k: np.nan for k in ["D2TPV","D2TEM","XYPV_x","XYPV_y","XYEM_x","XYEM_y"]}

    pv_idx = int(np.nanargmax(s_m))
    pv_val = float(s_m[pv_idx])

    # Distance-to-screen timeline
    d2s, _ = _distance_to_screen_series(distance_to_screen_cm, t, initial_distance_cm)
    d2s_m = d2s[mask]
    total_dist = float(d2s_m[0]) if len(d2s_m) > 0 else np.nan
    d2t_at_pv = float(d2s_m[pv_idx]) if len(d2s_m) > pv_idx else np.nan
    d2tpv = 100.0 * d2t_at_pv / total_dist if total_dist not in (0, np.nan) else np.nan

    # End-of-movement threshold (<= 10% PV)
    below = np.where(s_m[pv_idx:] <= threshold_frac * pv_val)[0]
    if len(below) == 0:
        em_idx = len(s_m) - 1
    else:
        em_idx = pv_idx + int(below[0])

    d2t_em = float(d2s_m[em_idx]) if len(d2s_m) > em_idx else np.nan
    d2tem = 100.0 * d2t_em / total_dist if total_dist not in (0, np.nan) else np.nan

    # Coordinates at PV and EM, convert to cm if scale known
    scale = trial.px_per_cm
    origin = ft[0]

    xypv = [ft[pv_idx][0] - origin[0], origin[1] - ft[pv_idx][1]]
    xyem = [ft[em_idx][0] - origin[0], origin[1] - ft[em_idx][1]]

    if scale and scale > 0:
        xypv = xypv / scale
        xyem = xyem / scale
    return {
        "D2TPV": d2tpv,
        "D2TEM": d2tem,
        "XYPV_x": float(xypv[0]), "XYPV_y": float(xypv[1]),
        "XYEM_x": float(xyem[0]), "XYEM_y": float(xyem[1]),
    }

# =========================
# Success coding (MOT / MIT)
# =========================

def score_MOT(is_target_selected: Iterable[bool]) -> np.ndarray:
    """
    For MOT: success is simply whether the indicated item after tracking is the target.
    Returns array of 1/0.
    """
    return np.array([1 if b else 0 for b in is_target_selected], dtype=int)

def score_MIT(clicked: Iterable[str], target_label: Iterable[str], distractor_label: Iterable[str]) -> np.ndarray:
    """
    For MIT: correct only if (clicked == target and selected 'target' on the shapes wreath)
    OR (clicked == distractor and selected 'distractor' on the shapes wreath).
    Here we assume 'clicked' is either 'target' or 'distractor', and the two label lists give the
    symbols chosen on the wreath (strings). This can be adapted to your data schema.
    """
    clicked = list(clicked)
    target_label = list(target_label)
    distractor_label = list(distractor_label)
    out = []
    for c, t, d in zip(clicked, target_label, distractor_label):
        if c == 'target' and t == 'target':
            out.append(1)
        elif c == 'distractor' and d == 'distractor':
            out.append(1)
        else:
            out.append(0)
    return np.array(out, dtype=int)

# =========================
# Touch cloud / confidence ellipse (2D)
# =========================

def confidence_ellipse_area(points_xy: np.ndarray) -> Dict[str, float]:
    """
    Compute the area of the confidence ellipse fitted to 2D points.
    Returns center (cx, cy), semi-axes (a, b), orientation (theta, deg), and area (pi*a*b).
    The semi-axes are sqrt of eigenvalues of the covariance matrix.
    """
    P = _as_np(points_xy)
    if P.ndim != 2 or P.shape[1] != 2 or len(P) < 2:
        return {"cx": np.nan, "cy": np.nan, "a": np.nan, "b": np.nan, "theta_deg": np.nan, "area": np.nan}
    mu = P.mean(axis=0)
    C = np.cov(P - mu, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    a = float(np.sqrt(max(evals[0], 0.0)))
    b = float(np.sqrt(max(evals[1], 0.0)))
    theta = float(np.degrees(np.arctan2(evecs[1,0], evecs[0,0])))
    area = float(np.pi * a * b)
    return {"cx": float(mu[0]), "cy": float(mu[1]), "a": a, "b": b, "theta_deg": theta, "area": area}

# =========================
# Joint angles (shoulder elevation, elbow flexion)
# =========================

def _angle(a, b, c, normal=None, y_down=False) -> float:
    """
    Signed angle ABC (at point B), in degrees.

    • 2D: uses atan2(cross_z, dot). If y_down=True (OpenCV image coords),
      flips the sign so positive is CCW in a math-style y-up frame.
    • 3D: needs a plane normal. If not provided, assumes camera normal [0,0,1].
      Returns angle in (-180, 180].

    Returns np.nan if vectors are degenerate.
    """
    a = _as_np(a); b = _as_np(b); c = _as_np(c)
    ba = a - b
    bc = c - b
    n_ba = np.linalg.norm(ba)
    n_bc = np.linalg.norm(bc)
    if n_ba == 0 or n_bc == 0:
        return np.nan

    # 2D case
    if ba.shape[-1] == 2 or bc.shape[-1] == 2:
        ba2 = ba[:2]; bc2 = bc[:2]
        dot = float(np.dot(ba2, bc2))
        cross_z = float(ba2[0]*bc2[1] - ba2[1]*bc2[0])
        if y_down:
            cross_z = -cross_z  # flip sign for image coords (y increases downward)
        ang = np.degrees(np.arctan2(cross_z, dot))  # (-180, 180]
        return float(ang)

    return np.nan

def shoulder_elevation_series(kps: KeypointSeries) -> Tuple[np.ndarray, np.ndarray]:
    """
    Angle between torso (shoulder-hip) and upper arm (shoulder-elbow).
    Defined to be ~0 deg when the arm lies along the torso.
    Returns (angle_deg[T], t[T]).
    """
    T = len(kps.t)
    ang = np.full(T, np.nan)
    for i in range(T):
        ang[i] = _angle(kps.elbow[i], kps.shoulder[i], kps.hip[i])
    return ang, kps.t

def elbow_flexion_series(kps: KeypointSeries) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal angle at the elbow between upper arm (shoulder-elbow) and forearm (wrist-elbow).
    Returns (angle_deg[T], t[T]).
    """
    T = len(kps.t)
    ang = np.full(T, np.nan)
    for i in range(T):
        ang[i] = _angle(kps.shoulder[i], kps.elbow[i], kps.wrist[i])
    return ang, kps.t

def angle_velocity_metrics(angle_deg: np.ndarray, t: np.ndarray,
                           events: TrialEventTimes,
                           smooth_win: int = 1) -> Dict[str, float]:
    """
    For an angle time series compute:
    PV (deg/s), P2PV (%), AV (deg/s), ROM (deg), AA (deg), SA (deg),
    AUMC (deg*s), AvgPh (deg).
    """
    a = _as_np(angle_deg)
    if smooth_win > 1:
        a = moving_average(a, smooth_win)
    # Movement window
    mask = (t >= events.key_release) & (t <= events.touch)
    if not np.any(mask):
        return {k: np.nan for k in ["PV","P2PV","AV","ROM","AA","SA","AUMC","AvgPh"]}
    a_m = a[mask]
    t_m = _as_np(t)[mask]
    # Angular velocity
    w = finite_diff(a, t)
    w_m = w[mask]

    # PV (deg/s)
    pv_idx = int(np.nanargmax(np.abs(w_m)))
    pv = float(np.abs(w_m[pv_idx]))

    # P2PV (%)
    mt = float(t_m[-1] - t_m[0])
    p2pv = float(100.0 * (t_m[pv_idx] - t_m[0]) / mt) if mt > 0 else np.nan

    # AV (deg/s): path length of angle divided by time
    ang_path = float(np.sum(np.abs(np.diff(a_m))))
    av = float(ang_path / mt) if mt > 0 else np.nan

    # ROM (deg)
    rom = float(np.nanmax(a_m) - np.nanmin(a_m))

    # AA (deg) and SA (deg)
    aa = float(np.nanmean(a_m))
    sa = float(a_m[0])

    # AUMC (deg*s): area under angle-time curve (absolute value to avoid cancellation)
    a_abs = np.abs(a_m)
    aumc = trapz_integral(a_abs, t_m)

    # AvgPh (deg): atan2 of normalized w and angle
    a_norm = normalize_series(a_m)
    w_norm = normalize_series(w_m)
    ph = np.degrees(np.arctan2(w_norm, a_norm))
    avgph = float(np.nanmean(ph))

    return {"PV": pv, "P2PV": p2pv, "AV": av, "ROM": rom, "AA": aa, "SA": sa, "AUMC": aumc, "AvgPh": avgph}

# =========================
# Time-normalized angle/phase
# =========================

def resample_to_bins(y: np.ndarray, t: np.ndarray, n_bins: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a time series to n_bins equally spaced in time (percentage domain).
    Returns (y_bins[n_bins], tau[n_bins]) where tau in [0,1].
    """
    y = _as_np(y); t = _as_np(t)
    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0:
        return np.full(n_bins, np.nan), np.linspace(0, 1, n_bins)
    tt = np.linspace(t0, t1, n_bins)
    yb = np.interp(tt, t, y)
    tau = (tt - t0) / (t1 - t0)
    return yb, tau

def time_normalized_metrics(trials_angles: List[np.ndarray],
                            trials_times: List[np.ndarray],
                            n_bins: int = 5) -> Dict[str, np.ndarray]:
    """
    Compute time-normalized angle (TNA) and phase (TNP) across trials.
    Returns dict with means and stds per bin and the std of the per-bin means.
    """
    binned_angles = []
    binned_phases = []
    for a, t in zip(trials_angles, trials_times):
        yb, tau = resample_to_bins(a, t, n_bins=n_bins)
        # Phase for this trial
        w = finite_diff(a, t)
        mask = slice(None)  # already full
        a_norm = normalize_series(a)
        w_norm = normalize_series(w)
        ph = np.degrees(np.arctan2(w_norm, a_norm))
        phb, _ = resample_to_bins(ph, t, n_bins=n_bins)
        binned_angles.append(yb)
        binned_phases.append(phb)

    A = np.vstack(binned_angles) if binned_angles else np.empty((0, n_bins))
    P = np.vstack(binned_phases) if binned_phases else np.empty((0, n_bins))
    TNA_mean = np.nanmean(A, axis=0) if A.size else np.full(n_bins, np.nan)
    TNA_std = np.nanstd(A, axis=0, ddof=1) if A.size else np.full(n_bins, np.nan)
    TNP_mean = np.nanmean(P, axis=0) if P.size else np.full(n_bins, np.nan)
    TNP_std = np.nanstd(P, axis=0, ddof=1) if P.size else np.full(n_bins, np.nan)
    # Std of the per-bin means (variability across bins)
    TNA_std_of_means = float(np.nanstd(TNA_mean, ddof=1)) if np.all(np.isfinite(TNA_mean)) else np.nan
    TNP_std_of_means = float(np.nanstd(TNP_mean, ddof=1)) if np.all(np.isfinite(TNP_mean)) else np.nan
    return {
        "tau": tau,
        "TNA_mean": TNA_mean, "TNA_std": TNA_std, "TNA_std_of_means": TNA_std_of_means,
        "TNP_mean": TNP_mean, "TNP_std": TNP_std, "TNP_std_of_means": TNP_std_of_means,
    }

def time_normalize_angle_phase_samples(angle_deg: np.ndarray,
                                       t: np.ndarray,
                                       n_points: int = 5) -> Dict[str, np.ndarray]:
    # TNA: sample angle at n_points normalized times
    TNA, tau = resample_to_bins(angle_deg, t, n_bins=n_points)
    # TNP: build phase(t) then sample it at the same normalized times
    w = finite_diff(angle_deg, t)
    ph = np.degrees(np.arctan2(normalize_series(w), normalize_series(angle_deg)))
    TNP, _ = resample_to_bins(ph, t, n_bins=n_points)
    return {"tau": tau, "TNA": TNA, "TNP": TNP}

# =========================
# Coordination measures
# =========================

def time_lag_peak_velocity(t_sh: np.ndarray, w_sh: np.ndarray,
                           t_el: np.ndarray, w_el: np.ndarray,
                           events: TrialEventTimes) -> float:
    """
    TLPV (%): difference in times when shoulder and elbow reach their (absolute) peak
    angular velocities (over the movement window), expressed as % of movement time.
    """
    mask_sh = (t_sh >= events.key_release) & (t_sh <= events.touch)
    mask_el = (t_el >= events.key_release) & (t_el <= events.touch)
    if not np.any(mask_sh) or not np.any(mask_el):
        return np.nan
    idx_sh = int(np.nanargmax(np.abs(w_sh[mask_sh])))
    idx_el = int(np.nanargmax(np.abs(w_el[mask_el])))
    t_peak_sh = float(t_sh[mask_sh][idx_sh])
    t_peak_el = float(t_el[mask_el][idx_el])
    mt = movement_time(events)
    return float(100.0 * (t_peak_el - t_peak_sh) / mt) if mt > 0 else np.nan

def cc_joint_angles(a_sh: np.ndarray, a_el: np.ndarray) -> float:
    """
    CCJA: Pearson correlation (zero lag) between shoulder and elbow angle series.
    Arrays must be same length; NaNs are removed pairwise.
    """
    a_sh = _as_np(a_sh); a_el = _as_np(a_el)
    mask = np.isfinite(a_sh) & np.isfinite(a_el)
    if mask.sum() < 2:
        return np.nan
    r = np.corrcoef(a_sh[mask], a_el[mask])[0,1]
    return float(r)

def average_continuous_relative_phase(a_sh: np.ndarray, t_sh: np.ndarray,
                                      a_el: np.ndarray, t_el: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    ACRP: mean of CRP(t) where CRP = phase(shoulder) - phase(elbow) and
    phase(x) = atan2(ω_norm, x_norm) in degrees.
    Returns (ACRP, CRP_series_aligned).
    """
    # Resample elbow to shoulder timebase (or vice versa)
    # Use shoulder timebase:
    a_el_rs = np.interp(t_sh, t_el, a_el)
    w_sh = finite_diff(a_sh, t_sh)
    w_el = finite_diff(a_el_rs, t_sh)
    ph_sh = np.degrees(np.arctan2(normalize_series(w_sh), normalize_series(a_sh)))
    ph_el = np.degrees(np.arctan2(normalize_series(w_el), normalize_series(a_el_rs)))
    crp = wrap_angle_deg(ph_sh - ph_el)
    return float(np.nanmean(crp)), crp

# =========================
# Sample Entropy
# =========================

def sample_entropy(x: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """
    SampEn(m, r): negative log of conditional probability that two sequences similar
    for m points remain similar at m+1, excluding self matches.
    r defaults to 0.2 * std(x).
    """
    x = _as_np(x)
    x = x[np.isfinite(x)]
    N = len(x)
    if N <= m + 1:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x, ddof=1)
    # Build template vectors
    def _phi(m):
        count = 0
        templates = np.array([x[i:i+m] for i in range(N - m + 1)])
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            # exclude self-match by removing i-th element
            count += np.sum(d <= r) - 1
        denom = (N - m + 1) * (N - m)
        return count / denom if denom > 0 else np.nan
    
    B = _phi(m)
    A = _phi(m+1)

    # minimal, robust handling
    if np.isnan(A) or np.isnan(B):
        return np.nan
    if B == 0:
        return np.nan  # undefined conditional probability
    if A == 0:
        A = np.finfo(float).tiny  # avoid +inf, keep value large but finite

    return float(-np.log(A / B))


# =========================
# MMPose convenience helpers
# =========================

RIGHT = dict(shoulder=6, elbow=8, wrist=10, hip=12, end_of_finger=124)

def compute_joint_angles_from_right_side(
    t: np.ndarray,
    keypoints_TxJxK: np.ndarray,
    *,
    shoulder=RIGHT["shoulder"],
    elbow=RIGHT["elbow"],
    wrist=RIGHT["wrist"],
    hip=RIGHT["hip"],
    y_down=True,   # set False if your coords are y-up
) -> dict:
    """
    Compute shoulder elevation and elbow flexion directly from the full keypoint array.
    Expects keypoints shaped [T, J, K>=2]; only x,y are used.
    Returns: {"shoulder_deg": [T], "elbow_deg": [T], "t": [T]}
    """
    T = keypoints_TxJxK.shape[0]
    xy = keypoints_TxJxK[..., :2]

    s = xy[:, shoulder, :]
    e = xy[:, elbow,   :]
    w = xy[:, wrist,   :]
    h = xy[:, hip,     :]

    shoulder_deg = np.empty(T, dtype=float)
    elbow_deg    = np.empty(T, dtype=float)

    for i in range(T):
        # Shoulder elevation: angle(E, S, H)  (upper-arm vs torso at the shoulder)
        shoulder_deg[i] = _angle(e[i], s[i], h[i], y_down=y_down)
        # Elbow flexion:     angle(S, E, W)  (upper-arm vs forearm at the elbow)
        elbow_deg[i]    = _angle(s[i], e[i], w[i], y_down=y_down)

    return {"shoulder_deg": shoulder_deg, "elbow_deg": elbow_deg, "t": np.asarray(t)}

# =========================
# Aggregation utilities
# =========================

def per_condition_std(values: Iterable[float]) -> float:
    """Standard deviation for a set of values from a condition (ddof=1)."""
    x = _as_np(list(values))
    if len(x) < 2:
        return np.nan
    return float(np.nanstd(x, ddof=1))

# =========================
# End of module
# =========================
