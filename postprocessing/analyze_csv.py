import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import re

from scipy import stats

result_path = "/media/janek/T7/results_real/"
experiments_dirs = glob.glob(os.path.join(result_path, "*"))

import numpy as np

def analyze_single_experiment(csv_file_path):
    df = pd.read_csv(csv_file_path)
    if 'is_anomaly' in df.columns:
        n_anomalies = df['is_anomaly'].sum()
        if n_anomalies > 0:
            print(f"  ⚠️ Odfiltrowano {int(n_anomalies)} triali z anomaliami kinematycznymi")
        df = df[df['is_anomaly'] != 1]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H-%M-%S', errors='coerce')
    df_whole = df.copy()
    df_training = df[df['Is_training'] != 0]
    df = df[df['Is_training'] == 0]
    
    if df['Task_time_motoric'].isnull().all():
        df['Task_time_motoric'] = np.where(
            df['Movement_start'] != -1,
            df['Movement_start'] + df['Movement_duration'],
            -1
        )
        
    df_dropped_task_1 = df[df['Guess_success'] == -1]
    df = df[df['Guess_success'] != -1]
    df_dropped_motoric = df[df['Movement_duration'] == -1]
    df = df[df['Movement_duration'] != -1]
    df_dropped_task_2 = df[df['MIT_obj_identified'] == -1]
    df = df[df['MIT_obj_identified'] != -1]

    # Filter click-distance outliers
    MOTORIC_RADIUS = 98.30400000000003
    if {'ClickX', 'ClickY', 'TargetX', 'TargetY'}.issubset(df.columns):
        click_xy = df[['ClickX', 'ClickY']].to_numpy(dtype=float)
        target_xy = df[['TargetX', 'TargetY']].to_numpy(dtype=float)
        click_dist = np.linalg.norm(click_xy - target_xy, axis=1)
        radius_outlier = click_dist > 2 * MOTORIC_RADIUS
        dist_std = np.nanstd(click_dist)
        std_outlier = click_dist > 3 * dist_std if dist_std > 0 else np.zeros(len(df), dtype=bool)
        combined = radius_outlier | std_outlier
        n_click_outliers = int(combined.sum())
        if n_click_outliers > 0:
            print(f"  ⚠️ Odfiltrowano {n_click_outliers} triali z outlierami kliknięć")
        df = df[~combined]

    def clean_mit_obj_identified(x):
        if x in [1, 2]:
            return 1
        elif x == 'MOT' or pd.isna(x):
            return np.nan
        else:
            return 0
    
    df['MIT_obj_identified'] = pd.to_numeric(df['MIT_obj_identified'], errors='coerce')
    df['MIT_obj_identified'] = df['MIT_obj_identified'].apply(clean_mit_obj_identified)

    experiment_name = os.path.basename(csv_file_path).replace(".csv", "")
    
    # First: global means
    num_good_trials = len(df)
    num_dropped_task1 = len(df_dropped_task_1)
    num_dropped_task2 = len(df_dropped_task_2)
    num_dropped_motoric = len(df_dropped_motoric)
    
    
    exp_id = df['ID'].iloc[0]
    exp_sex = df['Sex'].iloc[0]
    exp_age = df['Age'].iloc[0] #.seconds
    time_diff_first_last_trial_training = (df_training['Timestamp'].iloc[-1] - df_training['Timestamp'].iloc[0]).total_seconds()
    time_diff_first_last_trial_research = (df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds()
    time_diff_first_last_trial_whole = (df_whole['Timestamp'].iloc[-1] - df_whole['Timestamp'].iloc[0]).total_seconds()
    sum_target_guess = df[df['Guess']=='target']['Guess'].count()
    sum_distractor_guess = df[df['Guess']=='distractor']['Guess'].count()
    sum_GT_target = df[df['Ground_truth_guess']=='target']['Ground_truth_guess'].count()
    sum_GT_distractor = df[df['Ground_truth_guess']=='distractor']['Ground_truth_guess'].count()
    mean_guess_success_distractor = df[df['Guess']=='distractor']['Guess_success'].mean()
    mean_guess_success_target = df[df['Guess']=='target']['Guess_success'].mean()
    mean_guess_success = df['Guess_success'].mean()
    mean_identified = df['MIT_obj_identified'].mean()
    mean_task_time_guess = df['Task_time_guess'].mean()
    mean_movement_duration = df['Movement_duration'].mean()
    mean_movement_start = df['Movement_start'].mean()
    mean_task_time_motoric = df['Task_time_motoric'].mean()
    mean_norm_dist = df['Norm_Euc_Dist'].mean()
    mean_clickX = df['ClickX'].mean()
    mean_clickY = df['ClickY'].mean()
    mean_TargetX = df['TargetX'].mean()
    mean_TargetY = df['TargetY'].mean()  
    mean_diff_X = abs(mean_clickX - mean_TargetX)
    mean_diff_Y = abs(mean_clickY - mean_TargetY)
    mean_motoric_obj_Vx = df['Motoric_obj_Vx'].mean() if 'Motoric_obj_Vx' in df.columns else np.nan
    mean_motoric_obj_Vy = df['Motoric_obj_Vy'].mean() if 'Motoric_obj_Vy' in df.columns else np.nan
    mean_motoric_obj_V1_magnitude = df['Motoric_obj_V1_magnitude'].mean() if 'Motoric_obj_V1_magnitude' in df.columns else np.nan
    mean_motoric_click_V2_magnitude = df['Motoric_click_V2_magnitude'].mean() if 'Motoric_click_V2_magnitude' in df.columns else np.nan
    mean_angle_objV_click = df['Angle_objV_click'].mean() if 'Angle_objV_click' in df.columns else np.nan

    # Start summary dict
    summary = {
        'Experiment': experiment_name,
        'ID': exp_id,
        'Sex': exp_sex,
        'Age': exp_age,
        'T_diff_first_last_trial_training': time_diff_first_last_trial_training,
        'T_diff_first_last_trial_research': time_diff_first_last_trial_research,
        'T_diff_first_last_trial_whole': time_diff_first_last_trial_whole,
        'Good_trials': num_good_trials,
        'Dropped_Task1': num_dropped_task1,
        'Dropped_Task2': num_dropped_task2,
        'Dropped_Motoric': num_dropped_motoric,
        'Mean_Guess_Success_dist': mean_guess_success_distractor,
        'Mean_Guess_Success_targ': mean_guess_success_target,
        'Mean_Guess_Success': mean_guess_success,
        'Sum_Target_Guess': sum_target_guess,
        'Sum_Distractor_Guess': sum_distractor_guess,
        'Sum_GT_Target': sum_GT_target,
        'Sum_GT_Distractor': sum_GT_distractor,
        'Mean_MIT_Identified': mean_identified,
        'Mean_Task_Time_Guess': mean_task_time_guess,
        'Mean_Movement_Duration': mean_movement_duration,
        'Mean_Movement_Start': mean_movement_start,
        'Mean_Task_Time_Motoric': mean_task_time_motoric,
        'Mean_Norm_Euc_Dist': mean_norm_dist,
        'Mean_Diff_X': mean_diff_X,
        'Mean_Diff_Y': mean_diff_Y,
        'Mean_ClickX': mean_clickX,
        'Mean_ClickY': mean_clickY,
        'Mean_TargetX': mean_TargetX,
        'Mean_TargetY': mean_TargetY,
        'Mean_motoric_obj_Vx': mean_motoric_obj_Vx,
        'Mean_motoric_obj_Vy': mean_motoric_obj_Vy,
        'Mean_motoric_obj_V1_magnitude': mean_motoric_obj_V1_magnitude,
        'Mean_motoric_click_V2_magnitude': mean_motoric_click_V2_magnitude,
        'Mean_angle_objV_click': mean_angle_objV_click,
    }
    
    # Second: Accuracy by Type
    accuracy_by_type = df.groupby(['Type'])[['Guess_success', 'MIT_obj_identified', 'Norm_Euc_Dist', 
         'Task_time_guess', 'Movement_duration', 'Movement_start']].mean()
    for type_val, row in accuracy_by_type.iterrows():
        for col_name, value in row.items():
            summary[f'{col_name}_type_{type_val}'] = value
        
    # Second: Accuracy by Type
    accuracy_by_type_and_GT = df.groupby(['Type', 'Ground_truth_guess'])[['Guess_success', 'MIT_obj_identified', 'Norm_Euc_Dist', 
         'Task_time_guess', 'Movement_duration', 'Movement_start']].mean()
    for (type_val, GT), row in accuracy_by_type_and_GT.iterrows():
        for col_name, value in row.items():
            flat_col_name = f'{col_name}_GT_{GT}_{type_val}'
            summary[flat_col_name] = value
    
    # Third: Accuracy by Block and Type
    accuracy_by_block = df.groupby(['Type', 'Block'])[['Guess_success', 'MIT_obj_identified', 'Norm_Euc_Dist', 
         'Task_time_guess', 'Movement_duration', 'Movement_start']].mean()
    for (type_val, block_val), row in accuracy_by_block.iterrows():
        for col_name, value in row.items():
            summary[f'{col_name}_block_{block_val}_{type_val}'] = value

    # Fourth: Stats per N_targets and Type
    stats_per_targets_type = df.groupby(['Type', 'N_targets'])[
        ['Guess_success', 'MIT_obj_identified', 'Norm_Euc_Dist', 
         'Task_time_guess', 'Movement_duration', 'Movement_start']
    ].mean()
    
    for (type_val, n_targets_val), row in stats_per_targets_type.iterrows():
        for col_name, value in row.items():
            flat_col_name = f'{col_name}_Ntargets_{n_targets_val}_{type_val}'
            summary[flat_col_name] = value
    
    summary_df = pd.DataFrame([summary])
    stats_csv_output = os.path.join(os.path.dirname(csv_file_path), "summary_stats.csv")
    summary_df.to_csv(stats_csv_output)

    return summary

def final_analyse(summary_all_path, save_dir):
    # Load your summary_all_experiments
    summary_all_df = pd.read_csv(summary_all_path)

    # 1. Numeric statistics
    summary_numeric = summary_all_df.select_dtypes(include=[np.number])

    # Calculate various statistics
    summary_mean = summary_numeric.mean(axis=0)
    summary_std = summary_numeric.std(axis=0)
    summary_median = summary_numeric.median(axis=0)
    summary_min = summary_numeric.min(axis=0)
    summary_max = summary_numeric.max(axis=0)

    # Create DataFrames
    summary_mean_df = pd.DataFrame([summary_mean])
    summary_std_df = pd.DataFrame([summary_std])
    summary_median_df = pd.DataFrame([summary_median])
    summary_min_df = pd.DataFrame([summary_min])
    summary_max_df = pd.DataFrame([summary_max])

    # Add type labels
    summary_mean_df.insert(0, 'Summary_Type', 'Global_Mean')
    summary_std_df.insert(0, 'Summary_Type', 'Global_Std')
    summary_median_df.insert(0, 'Summary_Type', 'Global_Median')
    summary_min_df.insert(0, 'Summary_Type', 'Global_Min')
    summary_max_df.insert(0, 'Summary_Type', 'Global_Max')

    # 2. Categorical statistics
    summary_categorical = summary_all_df.select_dtypes(include=['object'])

    # Handle Sex counts
    sex_counts_dict = {}
    if 'Sex' in summary_categorical.columns:
        sex_counts = summary_categorical['Sex'].value_counts(dropna=False)  # count NaNs too
        for sex, count in sex_counts.items():
            if pd.isna(sex):
                sex_label = 'nan'
            else:
                sex_label = str(sex)
            sex_counts_dict[f"Sex_count_{sex_label}"] = count

    # 3. Merge everything
    # Start with mean
    full_summary = summary_mean_df.copy()

    # Add Sex counts into mean row
    for key, value in sex_counts_dict.items():
        full_summary[key] = value

    # Stack all statistics together
    full_summary = pd.concat([
        full_summary,
        summary_std_df,
        summary_median_df,
        summary_min_df,
        summary_max_df
    ], ignore_index=True)

    # 4. Reorder columns: put Sex_count_* right after Summary_Type
    cols = full_summary.columns.tolist()
    sex_cols = [col for col in cols if col.startswith('Sex_count_')]
    other_cols = [col for col in cols if col not in sex_cols and col != 'Summary_Type']
    new_order = sex_cols + ['Summary_Type'] + other_cols
    full_summary = full_summary[new_order]

    # 5. Save to CSV
    summary_means_output_path = os.path.join(save_dir, "summary_all_experiments_summary.csv")
    full_summary.to_csv(summary_means_output_path, index=False)
    return summary_means_output_path

    print(f"Saved full summarized statistics CSV to {summary_means_output_path}")

def final_analyse_plots(summary_all_path, save_dir):
    """
    Generate dynamic plots: how key metrics change over N_targets and Block number.
    """

    plots_dir = os.path.join(save_dir, "summary_plots_dynamics")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(summary_all_path)
    df_mean = df[df['Summary_Type'] == 'Global_Mean']
    if df_mean.empty:
        print("No Global_Mean found, aborting plots.")
        return

    df_numeric = df_mean.select_dtypes(include=[float, int]).drop(
        columns=['Sex_count_Kobieta', 'Sex_count_Nie chcę podawać', 'ID', 'Age'], errors='ignore'
    )

    # Two separate extractors
    def extract_n_targets(colname):
        match = re.search(r'_Ntargets_(\d+)', colname)
        if match:
            return int(match.group(1))
        else:
            return None

    def extract_block(colname):
        match = re.search(r'_block_(\d+)', colname)
        if match:
            return int(match.group(1))
        else:
            return None
        
    def extract_type(colname):
        match = re.search(r'_type_(\w+)', colname)
        if match:
            return match.group(1)
        else:
            return None
    
    def extract_GT(colname):
        match = re.search(r'_GT_(\w+)', colname)
        if match:
            return match.group(1)
        else:
            return None

    # Dynamic plot generator
    def plot_dynamic(metric_keyword, pattern_type, xlabel, title_prefix, filename_prefix):
        if pattern_type == 'Ntargets':
            extractor = extract_n_targets
        elif pattern_type == 'block':
            extractor = extract_block
        elif pattern_type == 'type':
            extractor = extract_type
        elif pattern_type == 'GT':
            extractor = extract_GT
        else:
            raise ValueError("pattern_type must be 'Ntargets' or 'block'")

        metric_cols = [col for col in df_numeric.columns if pattern_type in col and metric_keyword in col]
        if not metric_cols:
            return
        
        plt.figure(figsize=(12, 6))
        all_x_clean = []
        for task in ['MIT', 'MOT']:
            task_cols = [col for col in metric_cols if task in col]
            x = []
            y = []
            for col in task_cols:
                x_val = extractor(col)
                if x_val is not None:
                    x.append(x_val)
                    y.append(df_numeric[col].values[0])

            # Clean NaNs
            x_clean = []
            y_clean = []
            for xi, yi in zip(x, y):
                if not pd.isna(yi):
                    x_clean.append(xi)
                    y_clean.append(yi)

            if x_clean and y_clean:
                sorted_pairs = sorted(zip(x_clean, y_clean))
                x_sorted, y_sorted = zip(*sorted_pairs)
                plt.plot(x_sorted, y_sorted, marker='o', label=task)
                all_x_clean.extend(x_sorted)

        plt.title(f"{title_prefix} ({metric_keyword.replace('_', ' ')})")
        plt.xlabel(xlabel)
        plt.ylabel(metric_keyword.replace('_', ' '))
        plt.legend()
        plt.grid(True)

        # Force integer ticks
        if all_x_clean and type(max(all_x_clean)) is not str:
            min_x = min(all_x_clean)
            max_x = max(all_x_clean)
            plt.xticks(range(min_x, max_x + 1))

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{filename_prefix}_{metric_keyword}.png"))
        plt.close()

    # Metrics you want to track
    metrics = [
        'Guess_success',
        'MIT_obj_identified',
        'Norm_Euc_Dist',
        'Task_time_guess',
        'Movement_duration',
        'Movement_start'
    ]

    # Plot dynamics for N_targets
    for metric in metrics:
        plot_dynamic(
            metric_keyword=metric,
            pattern_type='Ntargets',
            xlabel='Number of Targets',
            title_prefix='Dynamics over N_targets',
            filename_prefix='dynamics_Ntargets'
        )

    # Plot dynamics for Blocks
    for metric in metrics:
        plot_dynamic(
            metric_keyword=metric,
            pattern_type='block',
            xlabel='Block Number',
            title_prefix='Dynamics over Blocks',
            filename_prefix='dynamics_Blocks'
        )
    
    for metric in metrics:
        plot_dynamic(
            metric_keyword=metric,
            pattern_type='GT',
            xlabel='Distractor/Target GT',
            title_prefix='Depending from Ground Truth',
            filename_prefix='dynamics_GT'
        )

    print(f"✅ Generated dynamic plots correctly into {plots_dir}")
    
import os
import pandas as pd

def analyze_concat_file(df, save_dir):
    """
    Analyze the concatenated DataFrame and save the results to a specific folder.
    Focuses only on creating readable agreement tables.
    """
    #print(df.columns)
    df = df[df['Type'] == 'MIT']
    df = df[df['Ground_truth_guess'] == 'target']
    
    # Check if the filtered dataframe is empty
    if df.empty:
        print("No MIT data found after filtering.")
        return
    
    # Create a specific subfolder for the outputs
    output_folder = os.path.join(save_dir, "analysis_shapes")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Created output folder: {output_folder}")
    
    # --- Agreement Table (original names) with improved formatting ---
    if 'Img_to_guess' in df.columns and 'Indicated_img' in df.columns:
        # Create the cross-tabulation
        agreement_table = pd.crosstab(
            df['Img_to_guess'], 
            df['Indicated_img'], 
            margins=True,  # Add row and column totals
            margins_name='Total'
        )
        
        # Calculate percentages
        percentage_table = pd.crosstab(
            df['Img_to_guess'], 
            df['Indicated_img'], 
            normalize='index',  # Get row percentages
            values=df['Img_to_guess'],
            aggfunc=lambda x: len(x) / len(x)  # This just counts
        ) * 100  # Convert to percentage
        
        # Round percentages to 1 decimal place
        percentage_table = percentage_table.round(1)
        
        # Create a combination table with counts and percentages
        combined_table = pd.DataFrame()
        
        for img_to_guess in agreement_table.index:
            if img_to_guess == 'Total':
                continue
            for indicated_img in agreement_table.columns:
                if indicated_img == 'Total':
                    continue
                count = agreement_table.loc[img_to_guess, indicated_img]
                percentage = percentage_table.loc[img_to_guess, indicated_img] if img_to_guess in percentage_table.index and indicated_img in percentage_table.columns else 0
                combined_table.loc[img_to_guess, indicated_img] = f"{count} ({percentage}%)"
        
        # Save both tables
        agreement_table_path = os.path.join(output_folder, "img_to_guess_vs_indicated_img_counts.csv")
        agreement_table.to_csv(agreement_table_path)
        print(f"Agreement table (counts) saved to {agreement_table_path}")
        
        percentage_table_path = os.path.join(output_folder, "img_to_guess_vs_indicated_img_percentages.csv")
        percentage_table.to_csv(percentage_table_path)
        print(f"Agreement table (percentages) saved to {percentage_table_path}")
        
        combined_table_path = os.path.join(output_folder, "img_to_guess_vs_indicated_img_combined.csv")
        combined_table.to_csv(combined_table_path)
        print(f"Agreement table (counts with percentages) saved to {combined_table_path}")
        
        # Generate a more readable HTML version with styling
        html_path = os.path.join(output_folder, "img_to_guess_vs_indicated_img.html")
        
        # Create a styled HTML table
        html_content = """
        <html>
        <head>
            <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                    font-family: Arial, sans-serif;
                }
                th, td {
                    border: 1px solid #dddddd;
                    text-align: center;
                    padding: 8px;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .total-row, .total-col {
                    font-weight: bold;
                    background-color: #e6e6e6;
                }
                .highlight {
                    background-color: #e6ffe6;
                }
            </style>
        </head>
        <body>
            <h2>Image Selection Analysis</h2>
            <p>This table shows the relationship between the image to guess and what was indicated by participants.</p>
            <h3>Counts with Percentages</h3>
        """
        
        # Add the table content
        html_content += "<table><tr><th>Image to Guess / Indicated</th>"
        for col in agreement_table.columns:
            css_class = " class='total-col'" if col == 'Total' else ""
            html_content += f"<th{css_class}>{col}</th>"
        html_content += "</tr>"
        
        for idx, row in enumerate(agreement_table.index):
            css_class = " class='total-row'" if row == 'Total' else ""
            html_content += f"<tr{css_class}><th{css_class}>{row}</th>"
            for col in agreement_table.columns:
                cell_class = ""
                if row == col and row != 'Total' and col != 'Total':
                    cell_class = " class='highlight'"  # Highlight diagonal (correct guesses)
                elif row == 'Total' or col == 'Total':
                    cell_class = " class='total-col'" if col == 'Total' and row == 'Total' else " class='total-col'" if col == 'Total' else " class='total-row'"
                
                value = agreement_table.loc[row, col]
                percentage = ""
                if row != 'Total' and col != 'Total':
                    percentage = f" ({percentage_table.loc[row, col]:.1f}%)" if row in percentage_table.index and col in percentage_table.columns else ""
                
                html_content += f"<td{cell_class}>{value}{percentage}</td>"
            html_content += "</tr>"
        
        html_content += """
            </table>
            <p><em>Note: Numbers in parentheses represent row percentages (% of each "Image to guess" that was indicated as each option).</em></p>
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        print(f"Interactive HTML agreement table saved to {html_path}")
    else:
        print("Cannot create agreement table: 'Img_to_guess' or 'Indicated_img' columns are missing.")
    
def generate_anomaly_report(result_path, experiments_dirs):
    anomaly_rows = []
    total_rows = []

    for exp_path in experiments_dirs:
        file_name = os.path.basename(exp_path)
        csv_file_path = os.path.join(exp_path, f"{file_name}_joined.csv")
        if not os.path.exists(csv_file_path):
            continue
        df = pd.read_csv(csv_file_path)
        if 'is_anomaly' not in df.columns:
            continue
        df['Experiment'] = file_name
        df['is_anomaly'] = pd.to_numeric(df['is_anomaly'], errors='coerce').fillna(0)
        total_rows.append(df)
        anom = df[df['is_anomaly'] == 1]
        if not anom.empty:
            anomaly_rows.append(anom)

    if not total_rows:
        print("⚠️ Brak danych do raportu anomalii")
        return

    df_all = pd.concat(total_rows, ignore_index=True)
    df_anom = pd.concat(anomaly_rows, ignore_index=True) if anomaly_rows else pd.DataFrame()

    report_dir = os.path.join(result_path, "anomaly_report")
    os.makedirs(report_dir, exist_ok=True)

    n_total = len(df_all)
    n_anom = len(df_anom)
    n_clean = n_total - n_anom

    # --- 1. Summary ---
    summary_rows = []
    summary_rows.append({
        'Category': 'OVERALL',
        'Group': 'all',
        'Total_trials': n_total,
        'Anomalies': n_anom,
        'Clean': n_clean,
        'Anomaly_pct': round(100 * n_anom / n_total, 2) if n_total else 0,
    })

    for task_type in ['MIT', 'MOT']:
        mask_all = df_all['Type'] == task_type
        mask_anom = df_anom['Type'] == task_type if not df_anom.empty else pd.Series(dtype=bool)
        t = int(mask_all.sum())
        a = int(mask_anom.sum())
        summary_rows.append({
            'Category': 'Type',
            'Group': task_type,
            'Total_trials': t,
            'Anomalies': a,
            'Clean': t - a,
            'Anomaly_pct': round(100 * a / t, 2) if t else 0,
        })

    for nt in sorted(df_all['N_targets'].dropna().unique()):
        mask_all = df_all['N_targets'] == nt
        mask_anom = df_anom['N_targets'] == nt if not df_anom.empty else pd.Series(dtype=bool)
        t = int(mask_all.sum())
        a = int(mask_anom.sum())
        summary_rows.append({
            'Category': 'N_targets',
            'Group': int(nt),
            'Total_trials': t,
            'Anomalies': a,
            'Clean': t - a,
            'Anomaly_pct': round(100 * a / t, 2) if t else 0,
        })

    if 'anomaly_reasons' in df_anom.columns and not df_anom.empty:
        reasons = df_anom['anomaly_reasons'].dropna().astype(str)
        all_reasons = []
        for r in reasons:
            all_reasons.extend([x.strip() for x in r.split(';') if x.strip()])
        from collections import Counter
        reason_counts = Counter(all_reasons)
        for reason, count in reason_counts.most_common():
            summary_rows.append({
                'Category': 'Reason',
                'Group': reason,
                'Total_trials': n_total,
                'Anomalies': count,
                'Clean': '',
                'Anomaly_pct': round(100 * count / n_total, 2) if n_total else 0,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(report_dir, "anomaly_report_summary.csv"), index=False)

    # --- 2. Per participant ---
    participant_rows = []
    id_col = 'ID' if 'ID' in df_all.columns else 'Experiment'
    for exp_name, grp in df_all.groupby('Experiment'):
        exp_id = grp[id_col].iloc[0] if id_col in grp.columns else exp_name
        t = len(grp)
        a = int(grp['is_anomaly'].sum())
        a_mit = int(grp[(grp['is_anomaly'] == 1) & (grp['Type'] == 'MIT')].shape[0])
        a_mot = int(grp[(grp['is_anomaly'] == 1) & (grp['Type'] == 'MOT')].shape[0])
        row = {
            'Experiment': exp_name,
            'ID': exp_id,
            'Total_trials': t,
            'Anomalies': a,
            'Anomaly_pct': round(100 * a / t, 2) if t else 0,
            'Anomalies_MIT': a_mit,
            'Anomalies_MOT': a_mot,
        }
        for nt in sorted(df_all['N_targets'].dropna().unique()):
            sub = grp[(grp['is_anomaly'] == 1) & (grp['N_targets'] == nt)]
            row[f'Anomalies_Nt{int(nt)}'] = len(sub)
        if 'anomaly_reasons' in grp.columns:
            anom_reasons = grp.loc[grp['is_anomaly'] == 1, 'anomaly_reasons'].dropna().astype(str)
            row['Reasons'] = '; '.join(anom_reasons.unique()) if not anom_reasons.empty else ''
        participant_rows.append(row)

    part_df = pd.DataFrame(participant_rows).sort_values('Anomaly_pct', ascending=False)
    part_df.to_csv(os.path.join(report_dir, "anomaly_report_per_participant.csv"), index=False)

    # --- 3. Cross-tab Type × N_targets ---
    if not df_anom.empty:
        ct = pd.crosstab(
            df_anom['Type'],
            df_anom['N_targets'],
            margins=True,
            margins_name='Total'
        )
        ct.to_csv(os.path.join(report_dir, "anomaly_report_type_x_ntargets.csv"))

        ct_all = pd.crosstab(df_all['Type'], df_all['N_targets'], margins=True, margins_name='Total')
        ct_pct = (ct / ct_all * 100).round(2)
        ct_pct.to_csv(os.path.join(report_dir, "anomaly_report_type_x_ntargets_pct.csv"))

    print(f"\n📊 Anomaly Report:")
    print(f"   Total trials: {n_total}")
    print(f"   Anomalies:    {n_anom} ({100*n_anom/n_total:.1f}%)")
    print(f"   Clean:        {n_clean} ({100*n_clean/n_total:.1f}%)")
    print(f"   Saved to:     {report_dir}/")


if __name__ == "__main__":
    all_experiments_summary = []
    all_trials = [] 
    
    for exp_path in experiments_dirs:
        file_name = os.path.basename(exp_path)
        print(f"Processing {file_name}")
        
        csv_file_path = os.path.join(exp_path, f"{file_name}_joined.csv")
        if os.path.exists(csv_file_path):
            summary = analyze_single_experiment(csv_file_path)
            all_experiments_summary.append(summary)
            
            df_trials = pd.read_csv(csv_file_path)
            df_trials = df_trials.loc[:, ~df_trials.columns.str.contains('^Unnamed')]
            if 'is_anomaly' in df_trials.columns:
                df_trials = df_trials[df_trials['is_anomaly'] != 1]
            df_trials['Experiment'] = file_name
            all_trials.append(df_trials)
        else:
            print(f"CSV file not found for {file_name}. Skipping.")

    if all_experiments_summary:
        summary_df = pd.DataFrame(all_experiments_summary)
        summary_output_path = os.path.join(result_path, "summary_all_experiments.csv")
        summary_df.to_csv(summary_output_path, index=False)
        print(f"Global summary CSV saved to {summary_output_path}")
    else:
        print("No data collected.")
        
    if all_trials:
        all_trials_df = pd.concat(all_trials, ignore_index=True)
        cols = all_trials_df.columns.tolist()
        cols = ['Experiment'] + [col for col in cols if col != 'Experiment']
        all_trials_df = all_trials_df[cols]
        analyze_concat_file(all_trials_df, result_path)
        all_trials_output_path = os.path.join(result_path, "all_trials_concat.csv")
        all_trials_df.to_csv(all_trials_output_path, index=False)
        print(f"All trials concatenated CSV saved to {all_trials_output_path}")
    else:
        print("No trial data collected.")
        
    summary_output_full_path = final_analyse(summary_output_path, result_path)
    final_analyse_plots(summary_output_full_path, result_path)

    generate_anomaly_report(result_path, experiments_dirs)
