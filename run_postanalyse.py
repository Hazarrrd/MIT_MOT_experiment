import os 
import subprocess
import glob
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from postprocessing.elbow_metric import get_metrics
from postprocessing.elbow_metric_comphresive import main
#from pose_estimation.inferencer_demo import get_predictions
def model_inference(path_to_experiment):
    # Build ffmpeg command
    #/home/janek/miniconda3/envs/bundesliga/bin/python3 /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/inferencer_demo.py --input webcam --output x.mp4
    ### ffmpeg -i MOT2.mp4 -c:v libx264 -preset ultrafast -crf 30 -c:a aac -b:a 128k MOT22.mp4 COMPRESS
    
    input_path = Path(os.path.join(path_to_experiment,'videos'))
    if input_path.is_dir():
        inputs = input_path.glob("*.*")
    else:
        inputs = [input_path]

    for input_file in inputs:
        get_predictions(input_file,
                        Path(os.path.join(path_to_experiment,'inference_results','2d')))
    #process = subprocess.Popen(command)
    #process.wait()

def model_inference3D(path_to_experiment):
    # Build ffmpeg command
    #/home/janek/miniconda3/envs/bundesliga/bin/python3 /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/inferencer_demo.py --input webcam --output x.mp4
    #for video in glob.glob(os.path.join(path_to_experiment,'videos',"*")):
    video = os.path.join(path_to_experiment,'videos')
    print(video)
    command = f"/home/janek/miniconda3/envs/bundesliga/bin/python pose_estimation/body3d_img2pose_demo.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_640-8xb32_coco-person.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth --input {video} --save-predictions --output-root {os.path.join(path_to_experiment,'inference_results/3d')} --show-interval 0".split(" ")
    process = subprocess.Popen(command)
    process.wait()

def model_inference3D_win(path_to_experiment):
    path_to_experiment = Path(path_to_experiment)  # Convert to Path object for cross-platform compatibility
    
    # Ensure output directory exists
    output_root = path_to_experiment / "inference_results" / "3d"
    output_root.mkdir(parents=True, exist_ok=True)

    # Loop through video files
    for video in glob.glob(str(path_to_experiment / "videos" / "*")):
        video = Path(video)  # Convert to Path object
        print(f"Processing: {video}")

        # Adjust paths for Windows
        python_exec = sys.executable  # Use the correct Python environment
        pose_script = Path(r"C:\Users\janns\Desktop\psychologia\MIT_MOT_experiment\pose_estimation\body3d_img2pose_demo.py")  # Update this path!
        model_config_1 = Path(r"C:\path\to\models\det_models\rtmdet_m_640-8xb32_coco-person.py")
        model_weights_1 = Path(r"C:\path\to\models\det_models\rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")
        model_config_2 = Path(r"C:\path\to\models\rtmpose\rtmw3d-l_8xb64_cocktail14-384x288.py")
        model_weights_2 = Path(r"C:\path\to\models\rtmpose\rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth")

        # Construct the command as a list (important for Windows compatibility)
        command = [
            python_exec, str(pose_script),
            str(model_config_1), str(model_weights_1),
            str(model_config_2), str(model_weights_2),
            "--input", str(video),
            "--save-predictions",
            "--output-root", str(output_root),
            "--show-interval", "0"
        ]

        # Run the process and wait for it to complete
        process = subprocess.Popen(command, shell=True)
        process.wait()


def sumamrize_results(path_experiment):
    # Path to your folder with CSV files
    # List to store all DataFrames
    base_folder = os.path.join(path_experiment, "inference_results", "3d")
    if not os.path.exists(base_folder):
        print(f"Folder {base_folder} does not exist.")
        return
    df_list = []

    # Loop over all CSV files in the folder
    for filename in os.listdir(base_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(base_folder, filename)

            # Extract MIT/MOT, block, and trial from filename
            match = re.match(r"(MIT|MOT)_block_(\d+)_trial(\d+)", filename)
            if match:
                exp_type = match.group(1)
                block = int(match.group(2))
                trial = int(match.group(3))
            else:
                exp_type, block, trial = None, None, None

            # Read CSV file
            df = pd.read_csv(filepath)

            # Insert the extracted values as new columns at the beginning
            df.insert(0, "Type", exp_type)
            df.insert(1, "Block", block)
            df.insert(2, "Trial", trial)

            df_list.append(df)

    # Concatenate all DataFrames row-wise
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df = final_df.sort_values(by=["Block", "Trial"]).reset_index(drop=True)
        output_path = os.path.join(path_experiment, "merged_results_kinematic.csv")
        final_df.to_csv(output_path, index=False)
        print(f"✅ Merged CSV saved to: {output_path}")
        
        # Exclude Type, Block, Trial from statistics
        exclude_cols = {"Type", "Block", "Trial"}
        numeric_df = final_df[[c for c in final_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(final_df[c])]]

        # Calculate mean and std
        stats_df = pd.DataFrame({
            "mean": numeric_df.mean(),
            "std": numeric_df.std()
        })

        # Save stats CSV
        stats_path = os.path.join(path_experiment, "merged_results_stats.csv")
        stats_df.to_csv(stats_path)
        print(f"✅ Stats CSV saved to: {stats_path}")
        
        other_df = pd.read_csv(os.path.join(path_experiment, path_experiment.name + ".csv"))
        # Merge on Type, Block, Trial (inner join so only matching rows remain)
        joined_df = pd.merge(other_df, final_df, on=["Type", "Block", "Trial"], how="left")
        joined_df.insert(57, "TE", joined_df["Motoric_click_V2_magnitude"])
        joined_df.insert(58, "PoL", joined_df["Angle_objV_click"].apply(lambda x: 1 if x < 90 else -1))
        joined_path = os.path.join(path_experiment, path_experiment.name + "_joined.csv")
        joined_df.to_csv(joined_path, index=False)
        
    else:
        print("No CSV files found in the folder.")
    
def do_kinematic_analyze(file_dir, filename):
    name = filename
    input_json = f"{file_dir}/inference_results/3d/results_{name}.json"  # Replace with actual path
    input_video = f"{file_dir}/inference_results/3d/{name}.avi"
    output_video = f"{file_dir}/inference_results/3d/{name}_kintematic.mp4"
    output_csv = f"{file_dir}/inference_results/3d/{name}_metrics.csv"
    print(input_json)
    if os.path.exists(output_csv):
        print(f"JSON file {output_csv} already exists. Skipping analysis.")
        return
    if not os.path.exists(input_video):
        print(f"Video file {input_video} does not exist. Skipping analysis.")
        return
    main(
        json_path=input_json,
        video_path=input_video,
        out_video_path=output_video,
        out_csv_path=output_csv
    )
    #print(input_json)
    #print(input_video)
    #print(output_video)
    #with open(input_json, 'r') as f:
    #    data = json.load(f)
    #angles,dists = extract_elbow_angles(data)
    #overlay_angles_on_video(angles,dists, input_video, output_video)
    
if __name__ == "__main__":
    do_for_each_video = True
    results_dir = Path("/home/janek/psychologia/MIT_MOT_experiment/results_real/")
    if do_for_each_video:
        # Iterate over all subdirectories starting with "experiment_"
        for experiment_path in results_dir.glob("experiment_*"):
            if experiment_path.is_dir():
                print(f"Running model inference for: {experiment_path}")
               # model_inference3D(str(experiment_path))
                for filename in glob.glob(str(experiment_path / "videos" / "*")):
                    continue
                   # get_metrics(experiment_path, Path(filename).stem)
                #do_kinematic_analyze(experiment_path, Path(filename).stem)
                sumamrize_results(experiment_path)
        
        # Run the process and wait for it to complete
        process = subprocess.Popen("python3 postprocessing/analyze_csv.py", shell=True)
        process.wait()
        process = subprocess.Popen("python3 postprocessing/analyze_csv_v2.py", shell=True)
        process.wait()
    else:
        experiment = "/home/janek/psychologia/MIT_MOT_experiment/results/experiment_2025-03-08-18-00-32_TEST/"
        experiment = "/home/janek/Downloads/k2/"
        experiment = "/home/janek/psychologia/MIT_MOT_experiment/results_real/experiment_2025-06-07-10-05-45_test/"
        experiment_path = Path(experiment)
        
        #model_inference3D(experiment)
        for i in ['MIT_block_2_trial1']:
            get_metrics(experiment, i)
            