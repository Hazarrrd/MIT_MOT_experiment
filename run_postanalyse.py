import os 
import subprocess
import glob
from postprocessing.elbow_metric import get_metrics
def model_inference(path_to_experiment):
    # Build ffmpeg command
    #/home/janek/miniconda3/envs/bundesliga/bin/python3 /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/inferencer_demo.py --input webcam --output x.mp4
    command = [
        "/home/janek/miniconda3/envs/bundesliga/bin/python3",
        "pose_estimation/inferencer_demo.py",  # Overwrite output file if it exists
        "--input", f"{os.path.join(path_to_experiment,'videos')}",  # Use the Video4Linux2 driver
        "--output", f"{os.path.join(path_to_experiment,'inference_results/2d')}"
    ]
    ### ffmpeg -i MOT2.mp4 -c:v libx264 -preset ultrafast -crf 30 -c:a aac -b:a 128k MOT22.mp4 COMPRESS
    
    process = subprocess.Popen(command)
    process.wait()

def model_inference3D(path_to_experiment):
    # Build ffmpeg command
    #/home/janek/miniconda3/envs/bundesliga/bin/python3 /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/inferencer_demo.py --input webcam --output x.mp4
    for video in glob.glob(os.path.join(path_to_experiment,'videos',"*")):
        print(video)
        command = f"/home/janek/miniconda3/envs/bundesliga/bin/python pose_estimation/body3d_img2pose_demo.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_640-8xb32_coco-person.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth --input {video} --save-predictions --output-root {os.path.join(path_to_experiment,'inference_results/3d')} --show-interval 0".split(" ")
        process = subprocess.Popen(command)
        process.wait()
        
experiment = "results/experiment_2025-02-08-20:52:49/"
#model_inference(experiment)
model_inference3D(experiment)
for filename in glob.glob(f"{experiment}videos/*"):
    get_metrics(experiment, filename.split("/")[-1].split(".")[0])