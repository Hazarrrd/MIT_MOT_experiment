import pandas as pd
import numpy as np

def coerce_numeric(df, skip_cols=None):
    skip = set(skip_cols or [])
    for c in df.columns:
        if c in skip:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def process_experiment_data(input_csv: str, output_excel: str):
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
    #df = df[df["Success"] == 1]
    
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

    print(f"✅ Exported {len(target_columns)} sheets to {output_excel}")

if __name__ == "__main__":
    # Example usage
    process_experiment_data("/home/janek/Downloads/all_trials_concat.csv", "output_formated.xlsx")
