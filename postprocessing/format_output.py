import os
import pandas as pd
import numpy as np
import re
from scipy import stats
import pingouin as pg

def coerce_numeric(df, skip_cols=None):
    skip = set(skip_cols or [])
    for c in df.columns:
        if c in skip:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def process_experiment_data(input_csv: str, output_excel: str, csv_output_folder="data/subsheets", success=True):
    os.makedirs(csv_output_folder, exist_ok=True)
    exclude_cols = [
        "Experiment", "ID", "Sex", "Age", "Block", "Trial", "Type", "Is_training",
        "N_targets", "N_distractors", "N_circles", "Timestamp", "Ground_truth_guess",
        "Guess", "Indicated_img", "Img_to_guess",
        "10a.png","10b.png","11a.png","11b.png",
        "1a.png","1b.png","2a.png","2b.png","3a.png","3b.png","4a.png","4b.png",
        "5a.png","5b.png","6a.png","6b.png","7a.png","7b.png","8a.png","8b.png",
        "9a.png","9b.png"
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

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for col in target_columns:
            tmp = df[["Experiment", "Type", "N_targets", col]].copy()

            # convert to numeric, invalid -> NaN
          #  tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
           # tmp[col] = tmp[col].replace(-1, np.nan)

            pivot = (
                tmp.groupby(["Experiment", "Type", "N_targets"])[col]
                .mean()
                .unstack(level=["Type", "N_targets"])
            )

            # flatten MultiIndex to one row: MIT_1, MIT_2, MOT_3...
            pivot.columns = [f"{t}_{n}" for (t, n) in pivot.columns]

            pivot.to_excel(writer, sheet_name=str(col)[:31])
            csv_path = os.path.join(csv_output_folder, f"{col}.csv")
            pivot.to_csv(csv_path, index=True)

    print(f"✅ Exported {len(target_columns)} sheets to {output_excel}")


def run_repeated_anova_for_csvs(csv_folder: str, summary_output: str):
    """
    Runs two-way repeated-measures ANOVA (TaskType × Load)
    for each metric CSV file and saves summarized results with significance info.
    Also logs and prints skipped files with reasons.
    """
    results = []
    skipped = []

    load_mapping = {
        "LOW": ["MIT_1", "MOT_2"],
        "MID": ["MIT_2", "MOT_3"],
        "HIGH": ["MIT_3", "MOT_4"]
    }

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
        melted["Load"] = melted["Condition"].apply(
            lambda x: next((k for k, v in load_mapping.items() if x in v), None)
        )
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
            
        input_csv="/home/janek/Downloads/all_trials_concat.csv"
        output_excel=f"output_formated{sufix}.xlsx"
        csv_output_folder="data/subsheets"+sufix
        
        process_experiment_data(
            input_csv=input_csv,
            output_excel=output_excel,
            csv_output_folder=csv_output_folder,
            success=success
        )
        run_repeated_anova_for_csvs(
            csv_folder=csv_output_folder,
            summary_output=f"data/anova_summary{sufix}.csv"
        )
