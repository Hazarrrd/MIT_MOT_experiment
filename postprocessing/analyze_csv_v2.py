# analyze_experiment.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.formula.api as smf

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "/media/janek/T7/results_real/all_trials_concat.csv"      # <- change me
EXCLUDE_TRAINING = True
OUTPUT_DIR = Path("/media/janek/T7/results_real/analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

KEY_METRICS = [
    "Age", "Task_time_guess", "Task_time_motoric", "Movement_start","Movement_duration", 
    "MT_s","PV","TE","PoL","AV","P2PV_pct","D2TPV_pct","D2TEM_pct","XYPV_x","XYPV_y","XYEM_x","XYEM_y","PathLen","sh_PV","sh_P2PV","sh_AV","sh_ROM","sh_AA","sh_SA","sh_AUMC","sh_AvgPh","el_PV","el_P2PV","el_AV","el_ROM","el_AA","el_SA","el_AUMC","el_AvgPh","coord_TLPV_pct","coord_CCJA","coord_ACRP_deg","SaEn_elbow","SaEn_shoulder","SaEn_ACRP","tau_t1","tau_t2","tau_t3","tau_t4","tau_t5","sh_TNA_t1","sh_TNA_t2","sh_TNA_t3","sh_TNA_t4","sh_TNA_t5","sh_TNP_t1","sh_TNP_t2","sh_TNP_t3","sh_TNP_t4","sh_TNP_t5","el_TNA_t1","el_TNA_t2","el_TNA_t3","el_TNA_t4","el_TNA_t5","el_TNP_t1","el_TNP_t2","el_TNP_t3","el_TNP_t4","el_TNP_t5"
]
OUTLIER_MODES = [
    ("none", "mode-none"),
    ("radius", "mode-radius"),
    ("std", "mode-std"),
]
DEFAULT_OUTLIER_MODE = "radius"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def save_relative_cov_ellipse_from_center_cov(
    center: np.ndarray,
    cov_matrix: np.ndarray,
    relative_points: np.ndarray | None = None,
    out_path: str = "ellipse.png",
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 200,
    show: bool = False,
) -> None:
    """
    Save a visualization of the covariance-based ellipse defined by (center, cov_matrix).
    This draws the same std-ellipse used by your area function (semi-axes = sqrt(eigenvalues)).

    Parameters
    ----------
    center : array-like, shape (2,)
        Ellipse center (mean of relative points).
    cov_matrix : array-like, shape (2, 2)
        Covariance matrix of mean-centered relative points.
    relative_points : array-like, shape (n_samples, 2), optional
        Points to scatter for context (e.g., click_points - object_points).
    out_path : str, default "ellipse.png"
        File path to save the figure.
    figsize : tuple, default (6, 6)
        Matplotlib figure size.
    dpi : int, default 200
        Output DPI.
    show : bool, default True
        If True, displays the figure after saving; otherwise closes it.
    """
    center = np.asarray(center, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)

    # Eigen-decomposition (eigh -> symmetric)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort by descending eigenvalue (major axis first)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)  # guard tiny negatives
    eigvecs = eigvecs[:, order]

    # Semi-axes (std-ellipse)
    a, b = float(np.sqrt(eigvals[0])), float(np.sqrt(eigvals[1]))

    # Orientation (degrees) from major-axis eigenvector
    vx, vy = eigvecs[0, 0], eigvecs[1, 0]
    angle_deg = float(np.degrees(np.arctan2(vy, vx)))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    if relative_points is not None:
        rel = np.asarray(relative_points, dtype=float)
        ax.scatter(rel[:, 0], rel[:, 1], s=12, alpha=0.75)

    # Mark origin and center
    ax.scatter([0], [0], marker="x", s=60)
    ax.annotate("(0, 0)", (0, 0), xytext=(5, 5), textcoords="offset points")
    ax.scatter([center[0]], [center[1]], s=30)
    ax.annotate("center", (center[0], center[1]), xytext=(5, 5), textcoords="offset points")

    # Ellipse patch: matplotlib expects diameters
    ell = Ellipse(
        xy=(center[0], center[1]),
        width=2 * a,
        height=2 * b,
        angle=angle_deg,
        fill=False,
        linewidth=2,
    )
    ax.add_patch(ell)

    # Cosmetics
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X (relative)")
    ax.set_ylabel("Y (relative)")
    ax.set_title("Covariance ellipse (std)")

    # Limits with padding
    pad = 0.2 * max(2 * a, 2 * b, 1.0)
    xmin = center[0] - a - pad
    xmax = center[0] + a + pad
    ymin = center[1] - b - pad
    ymax = center[1] + b + pad
    # Expand to include points if provided
    if relative_points is not None:
        xmin = min(xmin, np.min(relative_points[:, 0]) - pad)
        xmax = max(xmax, np.max(relative_points[:, 0]) + pad)
        ymin = min(ymin, np.min(relative_points[:, 1]) - pad)
        ymax = max(ymax, np.max(relative_points[:, 1]) + pad)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Save and show/close
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)

# -----------------------------
# Confidence ellipse function (as provided)
# -----------------------------
def confidence_ellipse_area_relative(
    click_points,
    object_points,
    metadata="",
    MOTORIC_RADIUS=98.30400000000003,
    outlier_mode: str = "radius",
) -> float:
    """
    Calculates the area of a confidence ellipse for clicks relative to object positions.

    Parameters
    ----------
    click_points : array-like, shape (n_samples, 2)
        Coordinates of the finger touch points.
    object_points : array-like, shape (n_samples, 2)
        Coordinates of the object positions.

    outlier_mode : {"none", "radius", "std"}, default "radius"
        Mode for filtering outliers in radial distance. "none" keeps all points,
        "radius" drops distances greater than 2 × MOTORIC_RADIUS, and "std"
        drops distances greater than 3 × distance standard deviation.

    Returns
    -------
    float
        Area of the confidence ellipse.
    """
    click_points = np.asarray(click_points, dtype=float)
    object_points = np.asarray(object_points, dtype=float)

    if click_points.shape != object_points.shape or click_points.shape[1] != 2:
        raise ValueError("Both inputs must have shape (n_samples, 2) and match in size.")

    # Compute relative positions so that object is at (0,0)
    relative_points = click_points - object_points

    dists = np.linalg.norm(relative_points, axis=1)
    keep_mask = np.ones_like(dists, dtype=bool)
    outlier_threshold = None

    if outlier_mode == "radius":
        outlier_threshold = 2 * MOTORIC_RADIUS
        keep_mask = dists <= outlier_threshold
    elif outlier_mode == "std":
        dist_std = np.std(dists)
        if dist_std > 0:
            outlier_threshold = 3 * dist_std
            keep_mask = dists <= outlier_threshold
    elif outlier_mode == "none":
        pass
    else:
        raise ValueError("outlier_mode must be one of {'none','radius','std'}")

    relative_points = relative_points[keep_mask]
    if len(relative_points) < 3:
        return np.nan
    
    # Mean center of relative positions
    center = relative_points.mean(axis=0)

    # Shift to center at (0,0) for covariance calculation
    shifted = relative_points - center

    # Covariance matrix
    cov_matrix = np.cov(shifted, rowvar=False)

    # Eigenvalues → variances along ellipse axes
    eigenvalues, _ = np.linalg.eigh(cov_matrix)

    # Semi-axes lengths = sqrt of eigenvalues
    a, b = np.sqrt(np.maximum(eigenvalues, 0))  # guard tiny negatives
    
    # Area of ellipse
    area = np.pi * a * b

    removed = int((~keep_mask).sum())
    outlier_info = f"mode{outlier_mode}"
    if outlier_threshold is not None:
        outlier_info += f"_thr{int(outlier_threshold)}"
    save_relative_cov_ellipse_from_center_cov(
        center,
        cov_matrix,
        relative_points,
        out_path=f"ellipses/ellipse_{metadata}_{outlier_info}_removed{removed}_area{int(area)}.png",
    )
    return area

# -----------------------------
# Helpers
# -----------------------------
def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    from math import sqrt
    z = 1.959963984540054 if alpha == 0.05 else 1.0
    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2/(2*n)
    adj = z * sqrt((phat*(1-phat) + z**2/(4*n))/n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return low, high

def coerce_numeric(df, skip_cols=None):
    skip = set(skip_cols or [])
    for c in df.columns:
        if c in skip:
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def safe_mean(x): 
    return np.nanmean(x) if len(x) else np.nan

def safe_std(x):
    return np.nanstd(x, ddof=1) if np.isfinite(x).sum() > 1 else np.nan

# -----------------------------
# Load & clean
# -----------------------------
df = pd.read_csv(CSV_PATH, engine="python")
df_dropped_task_1 = df[df['Guess_success'] == -1]
df = df[df['Guess_success'] != -1]
df_dropped_motoric = df[df['Movement_duration'] == -1]
df = df[df['Movement_duration'] != -1]
df_dropped_task_2 = df[df['MIT_obj_identified'] == -1]
df = df[df['MIT_obj_identified'] != -1]
df = df[df['Is_training'] == 0]

def clean_mit_obj_identified(x):
        if x in [1, 2]:
            return 1
        elif x == 'MOT' or pd.isna(x):
            return np.nan
        else:
            return 0

def Pol_to_numbers(x):
        if x in ["before"]:
            return 1
        else:
            return -1
    
df['MIT_obj_identified'] = pd.to_numeric(df['MIT_obj_identified'], errors='coerce')
df['MIT_obj_identified'] = df['MIT_obj_identified'].apply(clean_mit_obj_identified)
#df['PoL'] = df['PoL'].apply(Pol_to_numbers)

df.columns = [c.strip().replace(" ", "_") for c in df.columns]
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

for cat_col in ["Experiment","ID","Sex","Type","Ground_truth_guess","Guess",
                "Indicated_img","Img_to_guess"]:
    if cat_col in df.columns:
        df[cat_col] = df[cat_col].astype(str)

skip = {"Experiment","ID","Sex","Type","Ground_truth_guess","Guess",
        "Timestamp","Indicated_img","Img_to_guess"}
df = coerce_numeric(df, skip_cols=skip)

if "Type" in df.columns:
    df["Type"] = df["Type"].str.upper().str.strip()

if "Is_training" in df.columns and EXCLUDE_TRAINING:
    if df["Is_training"].dtype.kind in "biufc":
        df = df[df["Is_training"] == 0].copy()

if "N_targets" in df.columns:
    df["N_targets"] = pd.to_numeric(df["N_targets"], errors="coerce")

# -----------------------------
# Success (MOT vs MIT)
# -----------------------------
df["Success"] = np.nan
if {"Type","Guess_success"}.issubset(df.columns):
    is_mot = df["Type"] == "MOT"
    df.loc[is_mot, "Success"] = (df.loc[is_mot, "Guess_success"] == 1).astype(float)

if {"Type","Guess_success","MIT_obj_identified"}.issubset(df.columns):
    is_mit = df["Type"] == "MIT"
    df.loc[is_mit, "Success"] = (
        (df.loc[is_mit, "Guess_success"] == 1) &
        (df.loc[is_mit, "MIT_obj_identified"] == 1)
    ).astype(float)

# -----------------------------
# Per-Type & N_targets summary
# -----------------------------
group_cols = ["Type", "N_targets"]
present_metrics = [m for m in KEY_METRICS if m in df.columns]

def summarize_group(g):
    out = {}
    n = len(g)
    k = np.nansum(g["Success"].values) if "Success" in g else np.nan
    out["trials"] = n
    out["success_n"] = k
    out["success_rate"] = k / n if n else np.nan
    low, high = wilson_ci(k, n) if n else (np.nan, np.nan)
    out["success_ci_low"] = low
    out["success_ci_high"] = high
   # descriptive stats ONLY among successful trials
    g_succ = g[g["Success"] == 1]
    for m in present_metrics:
        out[f"{m}_mean"] = safe_mean(g_succ[m].values) if m in g_succ else np.nan
        out[f"{m}_sd"]   = safe_std(g_succ[m].values)  if m in g_succ else np.nan

    # participant counts (both total and successful-only for clarity)
    if "ID" in g.columns:
        out["participants_total"]   = g["ID"].nunique()
    return pd.Series(out) 

by_type_targets = (
    df.groupby(group_cols, dropna=False)
      .apply(summarize_group)
      .reset_index()  # <-- this restores 'Type' and 'N_targets' as columns
)

by_type_targets.sort_values(group_cols, inplace=True)


# -----------------------------
# NEW: Confidence-ellipse area per Type × N_targets
# -----------------------------
def ellipse_area_in_group(g, outlier_mode: str = DEFAULT_OUTLIER_MODE):
    """Return ellipse area (pixel^2) of Click relative to Target in group."""
    needed = {"ClickX","ClickY","TargetX","TargetY"}
    if not needed.issubset(g.columns):
        return pd.Series({"ellipse_area_px2": np.nan, "ellipse_points": 0})
    arr = g[["ClickX","ClickY","TargetX","TargetY"]].astype(float)
    arr = arr.replace([np.inf, -np.inf], np.nan).dropna()
    if len(arr) < 3:  # too few for a stable covariance
        return pd.Series({"ellipse_area_px2": np.nan, "ellipse_points": len(arr)})
    click = arr[["ClickX","ClickY"]].to_numpy()
    obj   = arr[["TargetX","TargetY"]].to_numpy()
    metadata = "_".join([str(x) for x in [group_cols, g.name, "points:", len(arr), f"mode:{outlier_mode}"]])
    try:
        area = confidence_ellipse_area_relative(click, obj, metadata=metadata, outlier_mode=outlier_mode)
    except Exception:
        area = np.nan
    return pd.Series({"ellipse_area_px2": area, "ellipse_points": len(arr)})

ellipse_summaries: dict[str, pd.DataFrame] = {}
OUTPUT_DIR.mkdir(exist_ok=True)
for mode, tag in OUTLIER_MODES:
    ellipse_by_group = (
        df[df["Success"] == 1]  # only successes
        .groupby(group_cols, dropna=False)
        .apply(lambda g, m=mode: ellipse_area_in_group(g, outlier_mode=m))
        .reset_index()
    )
    merged = by_type_targets.merge(ellipse_by_group, on=group_cols, how="left")
    ellipse_summaries[mode] = merged
    merged.to_csv(OUTPUT_DIR / f"summary_by_type_n_targets_{tag}.csv", index=False)

# Keep default mode for downstream compatibility and legacy filename
by_type_targets = ellipse_summaries.get(DEFAULT_OUTLIER_MODE, by_type_targets.copy())
by_type_targets.to_csv(OUTPUT_DIR / "summary_by_type_n_targets.csv", index=False)

# -----------------------------
# Participant-level summaries
# -----------------------------

# Successful trials only
df_succ = df[df["Success"] == 1].copy()
df_not_only_succ = df.copy()
#df_all = df.copy()
def ellipse_area_for_participant(g, outlier_mode: str = DEFAULT_OUTLIER_MODE):
    needed = {"ClickX","ClickY","TargetX","TargetY"}
    if not needed.issubset(g.columns):
        return np.nan
    arr = g[["ClickX","ClickY","TargetX","TargetY"]].astype(float).dropna()
    if len(arr) < 3:
        return np.nan
    click = arr[["ClickX","ClickY"]].to_numpy()
    obj   = arr[["TargetX","TargetY"]].to_numpy()
    metadata = "_".join([str(x) for x in [group_cols, g.name, "points:", len(arr), f"mode:{outlier_mode}"]])
    return confidence_ellipse_area_relative(click, obj, metadata=metadata, outlier_mode=outlier_mode)

participant_ellipses_by_mode: dict[str, pd.DataFrame] = {}
participant_ellipses_all_trials_by_mode: dict[str, pd.DataFrame] = {}
for mode, tag in OUTLIER_MODES:
    part_succ = (
        df_succ.groupby(["ID", "Type", "N_targets"], dropna=False)
               .apply(lambda g, m=mode: ellipse_area_for_participant(g, outlier_mode=m))
               .reset_index(name="ellipse_area_px2")
    )
    part_all = (
        df_not_only_succ.groupby(["ID", "Type", "N_targets"], dropna=False)
               .apply(lambda g, m=mode: ellipse_area_for_participant(g, outlier_mode=m))
               .reset_index(name="ellipse_area_px2")
    )
    part_succ.to_csv(OUTPUT_DIR / f"ellipse_area_per_participant_success_{tag}.csv", index=False)
    part_all.to_csv(OUTPUT_DIR / f"ellipse_area_per_participant_{tag}.csv", index=False)
    participant_ellipses_by_mode[mode] = part_succ
    participant_ellipses_all_trials_by_mode[mode] = part_all

# Keep default mode for downstream steps and legacy filenames
participant_ellipses = participant_ellipses_by_mode.get(DEFAULT_OUTLIER_MODE)
participant_ellipses_not_only_succ = participant_ellipses_all_trials_by_mode.get(DEFAULT_OUTLIER_MODE)
if participant_ellipses is not None:
    participant_ellipses.to_csv(OUTPUT_DIR / "ellipse_area_per_participant_success.csv", index=False)
if participant_ellipses_not_only_succ is not None:
    participant_ellipses_not_only_succ.to_csv(OUTPUT_DIR / "ellipse_area_per_participant.csv", index=False)

# Base per-participant (all trials): trials and success_rate
per_participant_base = (df
    .groupby(["ID","Experiment"] + group_cols, dropna=False)
    .agg(trials=("Success","count"),
         success_rate=("Success","mean"))
    .reset_index())

# Metrics per-participant from successful trials only
df_succ = df[df["Success"] == 1].copy()
metric_aggs = {f"{m}_mean": (m, "mean") for m in present_metrics}
metric_aggs.update({f"{m}_sd": (m, "std") for m in present_metrics})

per_participant_metrics = (df_succ
    .groupby(["ID","Experiment"] + group_cols, dropna=False)
    .agg(**metric_aggs)
    .reset_index())

# Merge
per_participant = per_participant_base.merge(
    per_participant_metrics,
    on=["ID","Experiment","Type","N_targets"],
    how="left"
)
per_participant = per_participant.merge(participant_ellipses, on=["ID", "Type", "N_targets"], how="left")

per_participant.to_csv(OUTPUT_DIR / "participant_level_summary.csv", index=False)

# Aggregated
pp_agg = (per_participant
          .groupby(["Experiment"] + group_cols, dropna=False)
          .agg(
              participants=("ID","nunique"),
              trials_total=("trials","sum"),
              success_rate_mean=("success_rate","mean"),
              success_rate_sd=("success_rate","std"),
          ).reset_index())
pp_agg.to_csv(OUTPUT_DIR / "participant_level_aggregated.csv", index=False)


# -----------------------------
# Logistic regression
# -----------------------------
logit_df = df.dropna(subset=["Success","Type","N_targets"]).copy()
logit_df["Success"] = logit_df["Success"].astype(int)
if not logit_df.empty:
    model = smf.logit("Success ~ C(Type) * N_targets", data=logit_df).fit(disp=False)
    with open(OUTPUT_DIR / "logit_summary.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

# -----------------------------
# Plots
# -----------------------------
def _plot_ellipse_area(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df is None or "ellipse_area_px2" not in summary_df.columns:
        return
    plt.figure()
    for t, sub in summary_df.dropna(subset=["Type"]).groupby("Type"):
        plt.plot(sub["N_targets"].values, sub["ellipse_area_px2"].values, "o-", label=t)
    plt.xlabel("N_targets")
    plt.ylabel("Confidence-ellipse area (pixel²)")
    plt.title("Click dispersion (relative to object) by Type and N_targets")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

# 1) Success rate vs N_targets with 95% CI (mode-independent)
plt.figure()
for t, sub in by_type_targets.dropna(subset=["Type"]).groupby("Type"):
    x = sub["N_targets"].values
    y = sub["success_rate"].values
    yerr_low = y - sub["success_ci_low"].values
    yerr_high = sub["success_ci_high"].values - y
    plt.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o-", capsize=4, label=t)
plt.xlabel("N_targets")
plt.ylabel("Success rate")
plt.title("Success rate by Type and N_targets (95% CI)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "success_rate_by_type_n_targets.png", dpi=200)

# 2) Ellipse area vs N_targets (pixel^2), per mode
_plot_ellipse_area(by_type_targets, OUTPUT_DIR / "ellipse_area_by_type_n_targets.png")
for mode, tag in OUTLIER_MODES:
    summary_mode = ellipse_summaries.get(mode)
    if summary_mode is None:
        continue
    filename = "ellipse_area_by_type_n_targets.png" if mode == DEFAULT_OUTLIER_MODE else f"ellipse_area_by_type_n_targets_{tag}.png"
    _plot_ellipse_area(summary_mode, OUTPUT_DIR / filename)

# 3) Task_time_guess vs N_targets (mean ± SD)
if "Task_time_guess_mean" in by_type_targets.columns:
    plt.figure()
    for t, sub in by_type_targets.dropna(subset=["Type"]).groupby("Type"):
        x = sub["N_targets"].values
        y = sub["Task_time_guess_mean"].values
        sd = sub["Task_time_guess_sd"].values
        plt.errorbar(x, y, yerr=sd, fmt="o-", capsize=4, label=t)
    plt.xlabel("N_targets")
    plt.ylabel("Task_time_guess (mean ± SD)")
    plt.title("Task_time_guess by Type and N_targets")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task_time_guess_by_type_n_targets.png", dpi=200)

# -----------------------------
# Console preview
# -----------------------------
#print("\n=== Summary by Type & N_targets (top 20 rows) ===")
#print(by_type_targets.head(20).to_string(index=False))
print("\nSaved files in:", OUTPUT_DIR.resolve())

# -------------------------------------------------------------------------
# STATISTICAL TESTS BETWEEN GROUPS
# -------------------------------------------------------------------------
from itertools import combinations
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind

# ------------- helpers ---------------------------------------------------
def fdr(pvals, alpha=0.05, method="fdr_bh"):
    """Return boolean mask of significant results after FDR correction."""
    reject, adj_p, _, _ = multipletests(pvals, alpha=alpha, method=method)
    return reject, adj_p

# ------------- set-up -----------------------------------------------------
grp_keys  = ["Type", "N_targets"]
summary   = by_type_targets.copy()
summary["group"] = summary[grp_keys].astype(str).agg("_".join, axis=1)

pairs = list(combinations(summary["group"], 2))

# ------------- SUCCESS-RATE tests ----------------------------------------
# Participant success rates from your per_participant_base
spr = per_participant_base.copy()
spr["group"] = spr["Type"].astype(str) + "_" + spr["N_targets"].astype(str)

# Welch t-tests on participant-level success_rate (optionally logit-transform)
from scipy.special import logit
eps = 1e-3
spr["sr_logit"] = logit(np.clip(spr["success_rate"], eps, 1 - eps))

succ_tests = []
for g1, g2 in combinations(spr["group"].unique(), 2):
    x = spr.loc[spr["group"] == g1, "sr_logit"].dropna()
    y = spr.loc[spr["group"] == g2, "sr_logit"].dropna()
    if len(x) < 3 or len(y) < 3:
        continue
    t, p = ttest_ind(x, y, equal_var=False)
    succ_tests.append({"group1": g1, "group2": g2, "t": t, "p_raw": p})

succ_df = pd.DataFrame(succ_tests)
if not succ_df.empty:
    rej, p_adj, _, _ = multipletests(succ_df["p_raw"], alpha=0.05, method="fdr_bh")
    succ_df["p_adj"] = p_adj
    succ_df["signif"] = rej
    succ_df.to_csv(OUTPUT_DIR / "pairwise_success_rate_tests_participant.csv", index=False)


# ------------- CONTINUOUS metrics ----------------------------------------
# === Participant-level means for metrics among successful trials ===
pp_means = (
    df[df["Success"] == 1]
    .groupby(["ID", "Type", "N_targets"], dropna=False)[present_metrics]
    .mean()
    .reset_index()
)
pp_means["group"] = pp_means["Type"].astype(str) + "_" + pp_means["N_targets"].astype(str)

# === Pairwise Welch t-tests on participant means (per metric) ===
from itertools import combinations
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

cont_tests = []
groups = pp_means["group"].unique()
for metric in present_metrics:
    for g1, g2 in combinations(groups, 2):
        x = pp_means.loc[pp_means["group"] == g1, metric].dropna()
        y = pp_means.loc[pp_means["group"] == g2, metric].dropna()
        if len(x) < 3 or len(y) < 3:
            continue
        t, p = ttest_ind(x, y, equal_var=False)
        cont_tests.append({"metric": metric, "group1": g1, "group2": g2, "t": t, "p_raw": p})

cont_df = pd.DataFrame(cont_tests)
if not cont_df.empty:
    cont_df["p_adj"] = np.nan
    cont_df["signif"] = False
    for m in cont_df["metric"].unique():
        mask = cont_df["metric"] == m
        reject, p_adj, _, _ = multipletests(cont_df.loc[mask, "p_raw"], alpha=0.05, method="fdr_bh")
        cont_df.loc[mask, "p_adj"] = p_adj
        cont_df.loc[mask, "signif"] = reject
    cont_df.to_csv(OUTPUT_DIR / "pairwise_continuous_tests.csv", index=False)



# -------------------------------------------------------------------------
# 4) Heat-map of success-rate (Type × N_targets)
# -------------------------------------------------------------------------
if not by_type_targets.empty:
    pivot = by_type_targets.pivot(index="Type", columns="N_targets", values="success_rate")
    plt.figure()
    im = plt.imshow(pivot, aspect="auto", origin="lower")
    plt.colorbar(im, label="Success rate")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)),   pivot.index)
    plt.title("Success-rate heat-map")
    plt.xlabel("N_targets"); plt.ylabel("Type")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_success_rate.png", dpi=200)

# -------------------------------------------------------------------------
# 5) Violin plots of Task_time_guess by Type (all N_targets pooled)
# -------------------------------------------------------------------------
if "Task_time_guess" in df.columns:
    import seaborn as sns
    plt.figure()
    sns.violinplot(data=df, x="Type", y="Task_time_guess", inner="quartile", cut=0)
    plt.title("Distribution of Task_time_guess by Type")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "violin_task_time_guess_by_type.png", dpi=200)

# -------------------------------------------------------------------------
# 6) Scatter + regression lines: ellipse area vs. Task_time_motoric
# -------------------------------------------------------------------------
if {"ellipse_area_px2", "Task_time_motoric"}.issubset(df.columns):
    plt.figure()
    sns.scatterplot(data=df, x="Task_time_motoric", y="ellipse_area_px2", hue="Type",
                    alpha=0.4, edgecolor=None)
    sns.regplot(data=df, x="Task_time_motoric", y="ellipse_area_px2", scatter=False,
                lowess=True, color="k", line_kws=dict(lw=2, ls="--"))
    plt.title("Dispersion (ellipse area) vs. motoric time")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_ellipse_vs_motoric.png", dpi=200)

# -------------------------------------------------------------------------
# 7) Correlation heat-map of *all* key metrics (pooled)
# -------------------------------------------------------------------------
metrics_present = [m for m in KEY_METRICS if m in df.columns]
if len(metrics_present) >= 3:
    corr = df[metrics_present].corr(method="spearman", min_periods=30)
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                cbar_kws=dict(shrink=0.8, label="Spearman ρ"))
    plt.title("Correlation matrix of key metrics")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_correlations.png", dpi=200)

# -------------------------------------------------------------------------
#ELIPSES

# -------------------------------------------------------------------------
# Significance testing on ellipse areas (Welch's t-tests)
# -------------------------------------------------------------------------
from itertools import combinations
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd

# Create group label
participant_ellipses["group"] = (
    participant_ellipses["Type"].astype(str) + "_" + participant_ellipses["N_targets"].astype(str)
)
groups = participant_ellipses["group"].unique()

tests = []
for g1, g2 in combinations(groups, 2):
    x = participant_ellipses.loc[participant_ellipses["group"] == g1, "ellipse_area_px2"].dropna()
    y = participant_ellipses.loc[participant_ellipses["group"] == g2, "ellipse_area_px2"].dropna()
    if len(x) < 3 or len(y) < 3:
        continue
    stat, p = ttest_ind(x, y, equal_var=False)
    tests.append({"group1": g1, "group2": g2, "t": stat, "p_raw": p})

tests_df = pd.DataFrame(tests)

# FDR correction
if not tests_df.empty:
    reject, p_adj, _, _ = multipletests(tests_df["p_raw"], alpha=0.05, method="fdr_bh")
    tests_df["p_adj"] = p_adj
    tests_df["signif"] = reject
    tests_df.to_csv(OUTPUT_DIR / "pairwise_ellipse_area_tests.csv", index=False)

# -------------------------------------------------------------------------
# Boxplot/violin plot for per-participant ellipse areas
# -------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.boxplot(data=participant_ellipses, x="N_targets", y="ellipse_area_px2", hue="Type")
plt.yscale("log")  # ellipse areas tend to be skewed
plt.ylabel("Ellipse area (px², log scale)")
plt.xlabel("N_targets")
plt.title("Per-participant click dispersion (ellipse area)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ellipse_area_per_participant_boxplot.png", dpi=200)


from matplotlib.patches import Ellipse

def draw_confidence_ellipse(ax, click_points, object_points, n_std=2.0, **kwargs):
    """
    Draws a confidence ellipse (n_std ≈ 2 → ~95% ellipse) for click positions relative to target.
    """
    click_points = np.asarray(click_points, dtype=float)
    object_points = np.asarray(object_points, dtype=float)
    
    # Compute relative clicks (center object at (0,0))
    rel = click_points - object_points
    if len(rel) < 3:
        return  # not enough points for a stable ellipse
    
    cov = np.cov(rel, rowvar=False)
    if np.linalg.det(cov) <= 0:
        return  # degenerate covariance
    
    mean_rel = rel.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues/eigenvectors
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    # Convert to angles in degrees
    angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
    
    # Width/height = 2 * n_std * sqrt(eigenvalue)
    width, height = 2 * n_std * np.sqrt(eigvals)
    
    ellipse = Ellipse(xy=mean_rel, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


# Successful trials only
df_succ = df[df["Success"] == 1].copy()

# Plot one figure per N_targets
for nt in sorted(df_succ["N_targets"].dropna().unique()):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for t, sub in df_succ[df_succ["N_targets"] == nt].groupby("Type"):
        # Click and target positions
        clicks  = sub[["ClickX", "ClickY"]].to_numpy(dtype=float)
        targets = sub[["TargetX", "TargetY"]].to_numpy(dtype=float)
        
        # Draw scatter of relative positions
        rel = clicks - targets
        ax.scatter(rel[:,0], rel[:,1], alpha=0.3, s=10, label=f"{t} clicks")
        
        # Draw the ellipse
        draw_confidence_ellipse(ax, clicks, targets, n_std=2.0,
                                facecolor='none', edgecolor='black', lw=2)
    
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("Relative X (px)")
    ax.set_ylabel("Relative Y (px)")
    ax.set_title(f"Click dispersion (relative to target) – N_targets={nt}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"click_dispersion_ellipses_nt{nt}.png", dpi=200)
