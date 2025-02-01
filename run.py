from psychopy import visual, core, event, gui
from mitmotexperiment.MOT import MOT
from mitmotexperiment.MIT import MIT
from mitmotexperiment.Experiment import Experiment
from datetime import datetime
import os
import numpy as np
import subprocess
import glob

##TODO
## ogarnij zapisywanie analizy posea
## ogarnij kalibracje i miary ppsea

##DO OBGADANIA!
## od kiedy kazac im trzymac palec?
## FPSy - kamera zmiana/próba ogarnienica ustawien swiatla - cel 60
## Modele, czy 2d starczy, jak 3D to raczej ptorzbuję 2 takich samych kamer i statywów
## ogarnij instrukcje 
## Wstaw śnieżki od Olgi

##### CONFIGURATION:
#win_size = [2560, 1440] 
#win_size = [1280, 1024]
win_size = [800, 800] ##should be set accordingly to monitor resolution even when FULL_SIZE!!!
full_size = True
show_circles = False
show_trial_results = True
do_inference_after = False
fps = 60
  # Set desired FPS


circle_radius = win_size[1]*0.28            #340
small_circle_radius = circle_radius/2.5
obj_radius = small_circle_radius/3.5
motoric_radius = obj_radius *3
motoric_circle_radius = small_circle_radius * 2

n_targets = 4
hz_target = 0.45 # cycles per second
hz_circle = 0.1 # cycles per second
hz_motoric = 0.45

observation_time = 1
tracking_time = 1  # seconds
guessing_time = 1  # seconds  ##Marking time

#[[t1,t2],[t3,t4]...] - each circle will change direction two times - in time sampled from t1 <= t <= t2 and in time sampled from t3 <= t <= t4
#pass direction_changes = [] in order to skip directions changes
direction_changes = [[1,tracking_time]]   
change_big_direction = False

answer_1_time_limit = 3
answer_MIT_time_limit = 5 
motor_task_time_limit = 10

#[[t1,t2],[t3,t4]...] - each circle will change direction two times - in time sampled from t1 <= t <= t2 and in time sampled from t3 <= t <= t4
#pass direction_changes_motoric = [] in order to skip directions changes
direction_changes_motoric = [[1,motor_task_time_limit]]


instruction_1 = "Keep track of the blue targets. Press 'left' key to start and hold it until end of the trial."
instruction_2 = "The object was {ground_truth}. The user chose {choice}."
instruction_2_MIT = "The object was {ground_truth}. The user chose {choice}. \n {identificiation}"

random_direction_small_circles = True
random_direction_big_circle = True
random_offset_target_distractor = True
random_offset_circles = True
random_distractor_target_orientation = True

path_for_mit_icons = "/home/janek/psychologia/MIT_MOT_experiment/icons_mit/icons"
results_dir = "/home/janek/psychologia/MIT_MOT_experiment/results/"
img_mode = True

def model_inference(path_to_experiment):
    #frame_width, frame_height, FPS_DESIRE, format_cam = 640, 480, 30, "yuyv422"
    frame_width, frame_height, FPS_DESIRE, format_cam = 1280, 720, 60, "mjpeg"
    # Build ffmpeg command
    #/home/janek/miniconda3/envs/bundesliga/bin/python3 /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/inferencer_demo.py --input webcam --output x.mp4
    command = [
        "/home/janek/miniconda3/envs/bundesliga/bin/python3",
        "pose_estimation/inferencer_demo.py",  # Overwrite output file if it exists
        "--input", f"{os.path.join(path_to_experiment,'videos')}",  # Use the Video4Linux2 driver
        "--output", f"{os.path.join(path_to_experiment,'inference_results')}"
    ]
    ### ffmpeg -i MOT2.mp4 -c:v libx264 -preset ultrafast -crf 30 -c:a aac -b:a 128k MOT22.mp4 COMPRESS
    
    process = subprocess.Popen(command)
    process.wait()

def model_inference3D(path_to_experiment):
    #frame_width, frame_height, FPS_DESIRE, format_cam = 640, 480, 30, "yuyv422"
    frame_width, frame_height, FPS_DESIRE, format_cam = 1280, 720, 60, "mjpeg"
    # Build ffmpeg command
    #/home/janek/miniconda3/envs/bundesliga/bin/python3 /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/inferencer_demo.py --input webcam --output x.mp4
    for video in glob.glob(os.path.join(path_to_experiment,'videos',"*")):
        print(video)
        command = f"/home/janek/miniconda3/envs/bundesliga/bin/python pose_estimation/body3d_img2pose_demo.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_640-8xb32_coco-person.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth --input {video} --save-predictions --output-root {os.path.join(path_to_experiment,'inference_results')} --show-interval 0".split(" ")
        process = subprocess.Popen(command)
        process.wait()

if __name__ == '__main__':
    experiment = Experiment(win_size = win_size, full_size = full_size, results_dir = results_dir)
    experiment.upload_param(
    show_circles=show_circles, 
    circle_radius=circle_radius, 
    small_circle_radius=small_circle_radius, 
    obj_radius=obj_radius, 
    n_targets=n_targets, 
    hz_target=hz_target, 
    hz_circle=hz_circle, 
    instruction_1=instruction_1, 
    instruction_2=instruction_2, 
    instruction_2_MIT=instruction_2_MIT, 
    fps=fps, 
    random_direction_small_circles=random_direction_small_circles, 
    random_direction_big_circle=random_direction_big_circle, 
    random_offset_target_distractor=random_offset_target_distractor, 
    random_offset_circles=random_offset_circles, 
    random_distractor_target_orientation=random_distractor_target_orientation, 
    observation_time=observation_time, 
    tracking_time=tracking_time, 
    guessing_time=guessing_time, 
    direction_changes=direction_changes, 
    direction_changes_motoric=direction_changes_motoric, 
    change_big_direction=change_big_direction, 
    show_trial_results=show_trial_results, 
    path_for_mit_icons=path_for_mit_icons, 
    img_mode=img_mode, 
    motoric_radius=motoric_radius, 
    motoric_circle_radius=motoric_circle_radius, 
    hz_motoric=hz_motoric, 
    answer_1_time_limit=answer_1_time_limit, 
    answer_MIT_time_limit=answer_MIT_time_limit, 
    motor_task_time_limit=motor_task_time_limit
)
    experiment.initialize_mot()
    experiment.initialize_mit()

    experiment.run_MOT_block(2)
    experiment.run_MIT_block(1)
    experiment.close()
    if do_inference_after:
        model_inference(experiment.dir_name)
        model_inference3D(experiment.dir_name)
    
