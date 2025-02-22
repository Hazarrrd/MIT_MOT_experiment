import os 
import subprocess
import glob
from pathlib import Path
from postprocessing.elbow_metric import get_metrics
from pose_estimation.inferencer_demo import get_predictions
def model_inference(path_to_experiment):
    # Build ffmpeg command
    #/home/janek/miniconda3/envs/bundesliga/bin/python3 /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/inferencer_demo.py --input webcam --output x.mp4
    ### ffmpeg -i MOT2.mp4 -c:v libx264 -preset ultrafast -crf 30 -c:a aac -b:a 128k MOT22.mp4 COMPRESS
    input_path = Path(os.path.join(path_to_experiment,'videos', 'MIT_block_1_trial1.avi'))
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
    for video in glob.glob(os.path.join(path_to_experiment,'videos',"*")):
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

experiment = r"C:\Users\janns\Desktop\psychologia\MIT_MOT_experiment\results\experiment_2025-02-16-15-23-31"
experiment_path = Path(experiment)

model_inference(experiment)
#model_inference3D(experiment)
# Get all video files in the "videos" folder
#for filename in glob.glob(str(experiment_path / "videos" / "*")):
#    get_metrics(experiment, Path(filename).stem)