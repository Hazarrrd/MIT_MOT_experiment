import os
import pandas as pd
import numpy as np
import re
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt

LOAD_MAPPING = {
    "LOW": ["MIT_1", "MOT_2"],
    "MID": ["MIT_2", "MOT_3"],
    "HIGH": ["MIT_3", "MOT_4"],
}
CONDITION_TO_LOAD = {
    condition: load for load, conditions in LOAD_MAPPING.items() for condition in conditions
}
LOAD_ORDER_DEFAULT = list(LOAD_MAPPING.keys())
TASK_ORDER_DEFAULT = ["MIT", "MOT"]
LOAD_COLOR_MAP = {"LOW": "tab:green", "MID": "tab:orange", "HIGH": "tab:purple"}
TASK_COLOR_MAP = {"MIT": "tab:blue", "MOT": "tab:orange"}
SIGNIFICANCE_EFFECTS = ["Load", "TaskType", "TaskType*Load"]
ELLIPSE_OUTLIER_VARIANTS = [
    "_mode-none",
    "_mode-radius",
    "_mode-std",
]

def coerce_numeric(df, skip_cols=None):
    skip = set(skip_cols or [])
    for c in df.columns:
        if c in skip:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def process_experiment_data(input_csv: str, output_excel: str, csv_output_folder="data/subsheets", success=True):
    os.makedirs(csv_output_folder, exist_ok=True)
    csv_output_folder_std = f"{csv_output_folder}_std"
    os.makedirs(csv_output_folder_std, exist_ok=True)
    excel_base, excel_ext = os.path.splitext(output_excel)
    output_excel_std = f"{excel_base}_std{excel_ext or '.xlsx'}"
    rms_groups = {
        "sh_TNA_rms": ["sh_TNA_t1", "sh_TNA_t2", "sh_TNA_t3", "sh_TNA_t4", "sh_TNA_t5"],
        "sh_TNP_rms": ["sh_TNP_t1", "sh_TNP_t2", "sh_TNP_t3", "sh_TNP_t4", "sh_TNP_t5"],
        "el_TNA_rms": ["el_TNA_t1", "el_TNA_t2", "el_TNA_t3", "el_TNA_t4", "el_TNA_t5"],
        "el_TNP_rms": ["el_TNP_t1", "el_TNP_t2", "el_TNP_t3", "el_TNP_t4", "el_TNP_t5"],
    }
    exclude_cols = [
        "Experiment", "ID", "Sex", "Age", "Block", "Trial", "Type", "Is_training",
        "N_targets", "N_distractors", "N_circles", "Timestamp", "Ground_truth_guess",
        "Guess", "Indicated_img", "Img_to_guess",
        "10a.png","10b.png","11a.png","11b.png",
        "1a.png","1b.png","2a.png","2b.png","3a.png","3b.png","4a.png","4b.png",
        "5a.png","5b.png","6a.png","6b.png","7a.png","7b.png","8a.png","8b.png",
        "9a.png","9b.png", "tau_t1","tau_t2","tau_t3","tau_t4","tau_t5"
    ]
    df = pd.read_csv(input_csv)


    df = df[df["Is_training"] == 0]  
    df = df[df["Guess_success"] != -1]
    df = df[df['Movement_duration'] != -1]
    df = df[df['MIT_obj_identified'] != -1]
    def Pol_to_numbers(x):
        if x in ["before"]:
            return 1
        else:
            return -1
    df['PoL'] = df['PoL'].apply(Pol_to_numbers)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    def clean_mit_obj_identified(x):
        if x in [1, 2]:
            return 1
        elif x == 'MOT' or pd.isna(x):
            return np.nan
        else:
            return 0
    
    df['MIT_obj_identified'] = pd.to_numeric(df['MIT_obj_identified'], errors='coerce')
    df['MIT_obj_identified'] = df['MIT_obj_identified'].apply(clean_mit_obj_identified)
    skip = {"Experiment","ID","Sex","Type","Ground_truth_guess","Guess",
        "Timestamp","Indicated_img","Img_to_guess"}
    
    df = coerce_numeric(df, skip_cols=skip)

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
    if success:
        df = df[df["Success"] == 1]
    
    target_columns = [c for c in df.columns if c not in exclude_cols]

    with pd.ExcelWriter(output_excel, engine="openpyxl") as mean_writer, \
         pd.ExcelWriter(output_excel_std, engine="openpyxl") as std_writer:
        for col in target_columns:
            tmp = df[["Experiment", "Type", "N_targets", col]].copy()
            grouped = tmp.groupby(["Experiment", "Type", "N_targets"])[col]

            pivot_mean = grouped.mean().unstack(level=["Type", "N_targets"])
            pivot_std = grouped.std().unstack(level=["Type", "N_targets"])

            # flatten MultiIndex to one row: MIT_1, MIT_2, MOT_3...
            pivot_mean.columns = [f"{t}_{n}" for (t, n) in pivot_mean.columns]
            pivot_std.columns = [f"{t}_{n}" for (t, n) in pivot_std.columns]

            sheet_name = str(col)[:31]
            pivot_mean.to_excel(mean_writer, sheet_name=sheet_name)
            pivot_std.to_excel(std_writer, sheet_name=sheet_name)

            pivot_mean.to_csv(os.path.join(csv_output_folder, f"{col}.csv"), index=True)
            pivot_std.to_csv(os.path.join(csv_output_folder_std, f"{col}.csv"), index=True)

    csv_output_folder_rms = f"{csv_output_folder}_rms"
    os.makedirs(csv_output_folder_rms, exist_ok=True)
    output_excel_rms = f"{excel_base}_rms{excel_ext or '.xlsx'}"

    with pd.ExcelWriter(output_excel_rms, engine="openpyxl") as rms_writer:
        rms_count = 0
        for label, cols in rms_groups.items():
            present_cols = [c for c in cols if c in df.columns]
            if not present_cols:
                continue
            tmp = df[["Experiment", "Type", "N_targets"] + present_cols].copy()
            grouped = tmp.groupby(["Experiment", "Type", "N_targets"])
            var_values = grouped[present_cols].var()
            if var_values.empty:
                continue
            mean_var = var_values.mean(axis=1, skipna=True)
            rms_series = np.sqrt(mean_var)
            pivot_rms = rms_series.unstack(level=["Type", "N_targets"])
            pivot_rms.columns = [f"{t}_{n}" for (t, n) in pivot_rms.columns]

            sheet_name = label[:31]
            pivot_rms.to_excel(rms_writer, sheet_name=sheet_name)
            pivot_rms.to_csv(os.path.join(csv_output_folder_rms, f"{label}.csv"), index=True)
            rms_count += 1

    print(
        f"✅ Exported {len(target_columns)} sheets to {output_excel} (mean) "
        f"and {output_excel_std} (std); "
        f"{rms_count} RMS sheets to {output_excel_rms}"
    )

def transform_for_anova(input_csv: str, output_csv: str):
    """
    Transform a long-format CSV (Experiment, Type, N_targets, ellipse_area_px2)
    into wide format for ANOVA: MIT_1, MIT_2, MIT_3, MOT_2, MOT_3, MOT_4
    """
    df = pd.read_csv(input_csv)
    df["Experiment"] = df["ID"]
    # Standardize fields
    df["N_targets"] = df["N_targets"].astype(int)
    df["Type"] = df["Type"].str.strip().str.upper()
    df["Condition"] = df["Type"] + "_" + df["N_targets"].astype(str)

    # Pivot to wide format
    wide = df.pivot_table(
        index="Experiment",
        columns="Condition",
        values="ellipse_area_px2",
        aggfunc="mean"
    ).reset_index()

    # Keep columns in desired order if present
    desired_cols = ["Experiment", "MIT_1", "MIT_2", "MIT_3", "MOT_2", "MOT_3", "MOT_4"]
    cols_present = [c for c in desired_cols if c in wide.columns]
    wide = wide[cols_present]

    wide.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv} ({len(wide)} experiments)")
    return wide

def _values_by_order(frame: pd.DataFrame, key_col: str, value_col: str, order: list[str], order_index: dict[str, int]):
    if not order:
        return []
    values = [np.nan] * len(order)
    for key, value in zip(frame[key_col], frame[value_col]):
        if pd.isna(key) or pd.isna(value):
            continue
        idx = order_index.get(str(key))
        if idx is None or idx >= len(values):
            continue
        values[idx] = value
    return values


def _extract_anova_significance(melted: pd.DataFrame) -> dict[str, tuple[bool | None, float | None]]:
    result: dict[str, tuple[bool | None, float | None]] = {
        effect: (None, None) for effect in SIGNIFICANCE_EFFECTS
    }
    if melted.empty:
        return result

    try:
        aov = pg.rm_anova(
            dv="Value",
            within=["TaskType", "Load"],
            subject="Experiment",
            data=melted,
            detailed=True,
            effsize="np2"
        )
    except Exception:
        return result

    p_col = "p-GG-corr" if "p-GG-corr" in aov.columns else "p-unc" if "p-unc" in aov.columns else None
    if p_col is None:
        return result

    for _, row in aov.iterrows():
        raw_source = row.get("Source")
        if raw_source is None or (isinstance(raw_source, float) and pd.isna(raw_source)):
            continue
        source = str(raw_source).replace(" ", "")
        if source not in result:
            continue
        p_val = row.get(p_col)
        if pd.isna(p_val):
            result[source] = (None, None)
            continue
        result[source] = (bool(p_val < 0.05), float(p_val))

    return result


def _significance_suffix(sig: bool | None) -> str:
    return "_sig" if sig else "_ns"


def _significance_title_fragment(sig: bool | None, p_value: float | None) -> str:
    if p_value is None or pd.isna(p_value):
        fragment = "p=n/a"
    else:
        fragment = f"p={p_value:.3f}"
    if sig:
        fragment += " *"
    return fragment


def _plot_load_effect(summary: pd.DataFrame, metric: str, output_dir: str, tasks_order: list[str],
                      loads_order: list[str], load_index: dict[str, int],
                      sig_info: tuple[bool | None, float | None], display_metric: str | None = None) -> None:
    if len(loads_order) < 2 or not tasks_order:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    has_data = False
    x_positions = np.arange(len(loads_order))

    for task in tasks_order:
        subset = summary[summary["TaskType"] == task]
        y_values = _values_by_order(subset, "Load", "Value", loads_order, load_index)
        y_array = np.asarray(y_values, dtype=float)

        if np.isnan(y_array).all():
            continue

        has_data = True
        color = TASK_COLOR_MAP.get(task, "tab:gray")
        ax.plot(x_positions, y_array, marker="o", linewidth=2, color=color, label=task)

    if has_data:
        sig, p_val = sig_info
        suffix = _significance_suffix(sig)
        sig_fragment = _significance_title_fragment(sig, p_val)
        metric_label = display_metric or metric
        if sig_fragment:
            title = f"{metric_label} - Load comparison ({sig_fragment})"
        else:
            title = f"{metric_label} - Load comparison"
        ax.set_xticks(x_positions)
        ax.set_xticklabels(loads_order)
        ax.set_xlabel("Load")
        ax.set_ylabel(metric_label)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{display_metric}_load{suffix}.png"), dpi=300)
    plt.close(fig)


def _plot_tasktype_effect(summary: pd.DataFrame, metric: str, output_dir: str, tasks_order: list[str],
                          loads_order: list[str], task_index: dict[str, int],
                          sig_info: tuple[bool | None, float | None], display_metric: str | None = None) -> None:
    if len(tasks_order) < 2 or not loads_order:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    has_data = False
    x_positions = np.arange(len(tasks_order))

    for load in loads_order:
        subset = summary[summary["Load"] == load]
        y_values = _values_by_order(subset, "TaskType", "Value", tasks_order, task_index)
        y_array = np.asarray(y_values, dtype=float)

        if np.isnan(y_array).all():
            continue

        has_data = True
        color = LOAD_COLOR_MAP.get(load, "tab:gray")
        ax.plot(x_positions, y_array, marker="o", linewidth=2, color=color, label=load)

    if has_data:
        sig, p_val = sig_info
        suffix = _significance_suffix(sig)
        sig_fragment = _significance_title_fragment(sig, p_val)
        metric_label = display_metric or metric
        if sig_fragment:
            title = f"{metric_label} - TaskType comparison ({sig_fragment})"
        else:
            title = f"{metric_label} - TaskType comparison"
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tasks_order)
        ax.set_xlabel("TaskType")
        ax.set_ylabel(metric_label)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{display_metric}_tasktype{suffix}.png"), dpi=300)
    plt.close(fig)


def _plot_interaction(summary: pd.DataFrame, melted: pd.DataFrame, metric: str, output_dir: str,
                      tasks_order: list[str], loads_order: list[str], load_index: dict[str, int],
                      sig_info: tuple[bool | None, float | None], display_metric: str | None = None) -> None:
    if len(loads_order) < 2 or not tasks_order:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    has_data = False
    x_positions = np.arange(len(loads_order))

    for task in tasks_order:
        task_data = melted[melted["TaskType"] == task]
        if task_data.empty:
            continue

        base_color = TASK_COLOR_MAP.get(task, "tab:gray")
        plotted_task = False

        for experiment, group in task_data.groupby("Experiment"):
            values = [np.nan] * len(loads_order)
            for load, val in zip(group["Load"], group["Value"]):
                if pd.isna(load) or pd.isna(val):
                    continue
                idx_pos = load_index.get(str(load))
                if idx_pos is None:
                    continue
                values[idx_pos] = val
            arr = np.asarray(values, dtype=float)
            if np.isnan(arr).all():
                continue
            ax.plot(x_positions, arr, color=base_color, alpha=0.2, linewidth=1)
            plotted_task = True

        mean_values = _values_by_order(
            summary[summary["TaskType"] == task],
            "Load",
            "Value",
            loads_order,
            load_index,
        )
        mean_arr = np.asarray(mean_values, dtype=float)
        if not np.isnan(mean_arr).all():
            ax.plot(
                x_positions,
                mean_arr,
                marker="o",
                linewidth=2.5,
                color=base_color,
                label=f"{task} mean",
            )
            plotted_task = True

        if plotted_task:
            has_data = True

    if has_data:
        sig, p_val = sig_info
        suffix = _significance_suffix(sig)
        sig_fragment = _significance_title_fragment(sig, p_val)
        metric_label = display_metric or metric
        if sig_fragment:
            title = f"{metric_label} - TaskType × Load ({sig_fragment})"
        else:
            title = f"{metric_label} - TaskType × Load"
        ax.set_xticks(x_positions)
        ax.set_xticklabels(loads_order)
        ax.set_xlabel("Load")
        ax.set_ylabel(metric_label)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{display_metric}_tasktype_load{suffix}.png"), dpi=300)
    plt.close(fig)


def generate_metric_plots(csv_folder: str, output_folder: str, label_suffix: str = "") -> None:
    os.makedirs(output_folder, exist_ok=True)

    for fname in sorted(os.listdir(csv_folder)):
        if not fname.endswith(".csv"):
            continue

        fpath = os.path.join(csv_folder, fname)
        metric = os.path.splitext(fname)[0]
        display_metric = f"{metric}{label_suffix}" if label_suffix else metric

        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue

        if df.empty or "Experiment" not in df.columns:
            continue

        melted = df.melt(id_vars=["Experiment"], var_name="Condition", value_name="Value")
        melted["TaskType"] = melted["Condition"].str.extract(r"^(MIT|MOT)", expand=False)
        melted["Load"] = melted["Condition"].map(CONDITION_TO_LOAD)
        melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
        melted = melted.dropna(subset=["TaskType", "Load", "Value"])

        if melted.empty:
            continue

        melted["TaskType"] = melted["TaskType"].astype(str)
        melted["Load"] = melted["Load"].astype(str)

        unique_tasks = list(dict.fromkeys(melted["TaskType"]))
        unique_loads = list(dict.fromkeys(melted["Load"]))

        tasks_order = [task for task in TASK_ORDER_DEFAULT if task in unique_tasks]
        tasks_order.extend(task for task in unique_tasks if task not in tasks_order)

        loads_order = [load for load in LOAD_ORDER_DEFAULT if load in unique_loads]
        loads_order.extend(load for load in unique_loads if load not in loads_order)

        task_index = {label: idx for idx, label in enumerate(tasks_order)}
        load_index = {label: idx for idx, label in enumerate(loads_order)}

        summary = (
            melted.groupby(["TaskType", "Load"], as_index=False)["Value"]
            .mean()
        )

        sig_info = _extract_anova_significance(melted)

        _plot_load_effect(
            summary,
            metric,
            output_folder,
            tasks_order,
            loads_order,
            load_index,
            sig_info.get("Load", (None, None)),
            display_metric,
        )
        _plot_tasktype_effect(
            summary,
            metric,
            output_folder,
            tasks_order,
            loads_order,
            task_index,
            sig_info.get("TaskType", (None, None)),
            display_metric,
        )
        _plot_interaction(
            summary,
            melted,
            metric,
            output_folder,
            tasks_order,
            loads_order,
            load_index,
            sig_info.get("TaskType*Load", (None, None)),
            display_metric,
        )

def run_repeated_anova_for_csvs(csv_folder: str, summary_output: str):
    """
    Runs two-way repeated-measures ANOVA (TaskType × Load)
    for each metric CSV file and saves summarized results with significance info.
    Also logs and prints skipped files with reasons.
    """
    results = []
    skipped = []

    for fname in os.listdir(csv_folder):
        if not fname.endswith(".csv"):
            continue

        fpath = os.path.join(csv_folder, fname)
        metric = os.path.splitext(fname)[0]

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            reason = f"cannot read file: {e}"
            print(f"⚠️ Skipped {fname} ({reason})")
            skipped.append({"File": fname, "Reason": reason})
            continue

        if df.empty or "Experiment" not in df.columns:
            reason = "empty or invalid format"
            print(f"⚠️ Skipped {fname} ({reason})")
            skipped.append({"File": fname, "Reason": reason})
            continue

        # Melt to long format
        melted = df.melt(id_vars=["Experiment"], var_name="Condition", value_name="Value")
        melted["TaskType"] = melted["Condition"].str.extract(r"^(MIT|MOT)")
        melted["Load"] = melted["Condition"].map(CONDITION_TO_LOAD)
        melted = melted.dropna(subset=["Load", "Value"])

        if melted["TaskType"].nunique() < 2 or melted["Load"].nunique() < 2:
            reason = "not enough conditions for ANOVA"
            print(f"⚠️ Skipped {metric} ({reason})")
            skipped.append({"File": fname, "Reason": reason})
            continue

        try:
            aov = pg.rm_anova(
                dv="Value",
                within=["TaskType", "Load"],
                subject="Experiment",
                data=melted,
                detailed=True,
                effsize="np2"
            )
            aov["Metric"] = metric

            # Add significance flag
            if "p-GG-corr" in aov.columns:
                aov["p_value"] = aov["p-GG-corr"]
            elif "p-unc" in aov.columns:
                aov["p_value"] = aov["p-unc"]
            else:
                aov["p_value"] = 1.0

            aov["Significant"] = aov["p_value"].apply(lambda p: "YES" if p < 0.05 else "NO")
            results.append(aov)
        except Exception as e:
            reason = f"error during ANOVA: {e}"
            print(f"⚠️ Error running ANOVA for {metric}: {e}")
            skipped.append({"File": fname, "Reason": reason})

    # Merge results
    if results:
        summary_df = pd.concat(results, ignore_index=True)
        summary_df.to_csv(summary_output, index=False)
        print(f"\n✅ Repeated-measures ANOVA summary saved to {summary_output}")
    else:
        print("\n⚠️ No valid ANOVA results computed (check input CSVs).")
        summary_df = pd.DataFrame()

    # Save skipped info
    skipped_path = summary_output.replace(".csv", "_skipped.csv")
    if skipped:
        pd.DataFrame(skipped).to_csv(skipped_path, index=False)

    # Print summary info at the end
    processed_count = len(results)
    skipped_count = len(skipped)

    print(f"\nSummary:")
    print(f"  • Processed files: {processed_count}")
    print(f"  • Skipped files:   {skipped_count}")

    if skipped_count > 0:
        print(f"  • Skipped details saved to: {skipped_path}")
        print("\n⚠️ Skipped files:")
        for s in skipped:
            print(f"    - {s['File']}: {s['Reason']}")

    # Print significant results
    if not summary_df.empty and "Significant" in summary_df.columns:
        sig = summary_df[summary_df["Significant"] == "YES"]
        cols_to_show = [c for c in ["Metric", "Source", "F", "p_value", "np2", "Significant"] if c in summary_df.columns]

        if not sig.empty:
            print("\n🔥 Significant effects found:")
            print(sig[cols_to_show])
        else:
            print("\nNo significant main or interaction effects (p < 0.05).")
        
if __name__ == "__main__":
    # Example usage

    for success in [True, False]:
        if success:
            sufix = "_success"
        else:
            sufix = ""
        csv_output_folder = f"/media/janek/T7/results_real/data/subsheets{sufix}"
        os.makedirs(csv_output_folder, exist_ok=True)
        ellipse_base = f"/media/janek/T7/results_real/analysis_outputs/ellipse_area_per_participant{sufix}"
        for ellipse_variant in ELLIPSE_OUTLIER_VARIANTS:
            ellipse_path = f"{ellipse_base}{ellipse_variant}.csv"
            if not os.path.exists(ellipse_path):
                continue
            ellipse_out = os.path.join(
                csv_output_folder,
                f"ellipse_area_per_participant{sufix}{ellipse_variant}_wide.csv",
            )
            transform_for_anova(ellipse_path, ellipse_out)
        input_csv="/media/janek/T7/results_real/all_trials_concat.csv"
        output_excel=f"/media/janek/T7/results_real/data/output_formated{sufix}.xlsx"
        
        process_experiment_data(
            input_csv=input_csv,
            output_excel=output_excel,
            csv_output_folder=csv_output_folder,
            success=success
        )
        stats_variants = [
            ("", csv_output_folder, ""),
            ("_std", f"{csv_output_folder}_std", " (std dev)"),
            ("_rms", f"{csv_output_folder}_rms", " (RMS)"),
        ]
        for variant_suffix, folder, label_suffix in stats_variants:
            if not os.path.isdir(folder):
                continue
            variant_plots_dir = os.path.join("/media/janek/T7/results_real/plots", f"metrics{sufix}{variant_suffix}")
            generate_metric_plots(
                csv_folder=folder,
                output_folder=variant_plots_dir,
                label_suffix=label_suffix
            )
            run_repeated_anova_for_csvs(
                csv_folder=folder,
                summary_output=f"/media/janek/T7/results_real/data/anova_summary{sufix}{variant_suffix}.csv"
            )

    
