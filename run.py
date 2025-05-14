from psychopy import visual, core, event, gui
from mitmotexperiment.MOT import MOT
from mitmotexperiment.MIT import MIT
from mitmotexperiment.Experiment import Experiment
from datetime import datetime
import os
import numpy as np
import subprocess
import glob
import random




## Distance to target: w kazdej klatce dystans palca wskazujacego do celu
## W każdej klatce V cm na sekunda
## Kąty okciowe/ramienia
## Prędkość zmiany kąta
## peak velocity pozycje palca
## 10% pierwszy raz przekroczone 

# ##DO OBGADANIA!
# dalej monitor

##### CONFIGURATION:
#win_size = [2560, 1440] 
win_size = [1280, 1024]
#win_size = [1920, 1080] ##should be set accordingly to monitor resolution even when FULL_SIZE!!!
is_windows_OS = True
full_size = True
show_circles = False
show_trial_results = True
do_inference_after = False
mouse_visible = False
fps = 60

training_trials_per_block = 12 #12
trials_per_block = 54 #54
block_pairs_number = 2 #2
target_circles_ammount_settups = {"MIT" : [[1,5],[2,5],[3,5]] , ##[X,Y] X - number of targets, Y - number of circles
                                  "MOT" : [[2,5],[3,5],[4,5]]} 

circle_radius = win_size[1]*0.28            #340
small_circle_radius = circle_radius/2.5
obj_radius = small_circle_radius/3.5
motoric_radius = obj_radius *3
motoric_circle_radius = small_circle_radius * 2

hz_target = 0.45 # cycles per second #270/360  #0.45
hz_circle = 0.1 # cycles per second # 
hz_motoric = 0.35

observation_time = 1.5
tracking_time = 6  # seconds
guessing_time = 1.5  # seconds  ##Marking time

#[[t1,t2],[t3,t4]...] - each circle will change direction two times - in time sampled from t1 <= t <= t2 and in time sampled from t3 <= t <= t4
#pass direction_changes = [] in order to skip directions changes
direction_changes = [[1,tracking_time]]   
change_big_direction = False
mit_objects_folder = "icons_shapes_c"
snowflakes_id_to_use = [str(i+1)+"a" for i in range(0,11)]
snowflakes_id_to_use.extend([str(i+1)+"b" for i in range(0,11)])
#print(snowflakes_id_to_use)
#snowflakes_id_to_use.extend([f"{i+1}_negative" for i in range(0,16)]) 
#snowflakes_id_to_use = ['1','3','5','25','2','9','7','8','18','19','11','28'] 
#snowflakes_id_to_use = ['brazil','canada','china','france','germany','japan','south-korea','switzerland','turkey','united-kingdom']

answer_1_time_limit = 4
answer_MIT_time_limit = 8 
motor_task_time_limit = 10

#[[t1,t2],[t3,t4]...] - each circle will change direction two times - in time sampled from t1 <= t <= t2 and in time sampled from t3 <= t <= t4
#pass direction_changes_motoric = [] in order to skip directions changes
direction_changes_motoric = [[1,motor_task_time_limit]]


instruction_main_mot = f"INSTRUKCJA MOT \n \n \n Za chwilę zostanie Ci przedstawiona gra która polega na uwagowym śledzeniu poruszających się obiektów. Każda próba będzie zaczynała się gdy naciśniesz oznaczony klawisz – trzymaj na nim palec wskazujący do momentu, który zostanie opisany w dalszej części instrukcji.\n \n \
Na początku każdej próby zobaczysz na ekranie losowo rozmieszczone czarne koła. Część z nich zostanie oznaczona zieloną obwódką jako cele. W różnych wariantach trudności liczba oznaczonych celów może się różnić. Należy zapamiętać, które obiekty są celami, ponieważ oznaczenie po krótkiej chwili zniknie i wszystkie czarne koła zaczną się poruszać w dość losowy sposób. Podczas gdy czarne koła poruszają się, staraj się śledzić te, które zostały oznaczone jako cele. Po kilku sekundach wszystkie koła zatrzymają się i jedno losowo wybrane koło zostanie oznaczone niebieską obwódką. Po krótkiej chwili koła znikną. \
Następnie pojawi się zadanie w którym należy: zwolnić naciskany klawisz i tym palcem dotknąć środka czarnego koła, które będzie się poruszało po ekranie. Środek koła został oznaczony białym znakiem +. Nie zwlekaj zbyt długo z odpowiedzią.  Gdy wykonujesz ruch staraj się, by był jak najbardziej pewny, prosty.  \
Następie na ekranie zobaczysz dwa koła: zielone i czerwone. Jeżeli uważasz, że oznaczone niebieską obwódką czarne koło było celem – należy wybrać zielone koło; jeżeli uważasz, że oznaczone niebieską obwódką czarne koło nie było celem, wybierz czerwone koło. Żeby wybrać koło (zielone lub czerwone) należy dotknąć koła na ekranie palcem, którym wcześniej naciskało się oznaczony klawisz. Gdy udzielisz odpowiedzi i ponownie naciśniesz oznaczony klawisz rozpocznie się kolejna próba. \n \n \
Będzie {training_trials_per_block} prób, podczas których nauczysz się wykonywać zadanie. Po treningu rozpoczną się próby właściwe. \n \
Jeśli masz jakieś pytania lub wątpliwości, możesz zgłosić je teraz osobie prowadzącej. Jeżeli nie, wciśnij oznaczony klawisz, aby przejść dalej."

instruction_main_mit = f"INSTRUKCJA MIT \n \n \n  Za chwilę zostanie Ci przedstawiona gra która polega na uwagowym śledzeniu poruszających się obiektów. Każda próba będzie zaczynała się gdy naciśniesz oznaczony klawisz – przy każdej próbie trzymaj na nim palec wskazujący do momentu, który zostanie opisany w dalszej części instrukcji. \n \n \
Na początku każdej próby zobaczysz losowo rozmieszczone obiekty. Część z nich zostanie oznaczona zieloną obwódką jako cele. W różnych wariantach trudności liczba oznaczonych celów może się różnić. Należy zapamiętać, które obiekty są celami, ponieważ oznaczenie po krótkiej chwili zniknie i wszystkie obiekty zaczną się poruszać w dość losowy sposób. Podczas gdy obiekty poruszają się, staraj się śledzić te, które zostały oznaczone jako cele.  Po kilku sekundach wszystkie obiekty zatrzymają się i zostaną ukryte pod czarnymi kołami, a jeden z nich zostanie oznaczony niebieską obwódką.  Po krótkiej chwili koła znikną. \
Następnie pojawi się zadanie w którym należy: zwolnić naciskany klawisz i tym palcem dotknąć środka czarnego koła, które będzie się poruszało po ekranie. Środek koła został oznaczony białym znakiem +. Nie zwlekaj zbyt długo z odpowiedzią.  Gdy wykonujesz ruch staraj się, by był jak najbardziej pewny, prosty. \
Następie na ekranie zobaczysz dwa koła: zielone i czerwone. Jeżeli uważasz, że oznaczony niebieską obwódką obiekt ukryty pod czarnym kołem był celem – należy wybrać zielone koło; jeżeli uważasz, że oznaczony niebieską obwódką obiekt ukryty pod czarnym kołem nie był celem, wybierz czerwone koło.  Żeby wybrać koło (zielone lub czerwone) należy dotknąć koła na ekranie palcem, którym wcześniej naciskało się oznaczony klawisz. \
Jako ostatnie zadanie na ekranie pojawią się rozmieszczone w okręgu obiekty, które wcześniej poruszały się po ekranie. Jeżeli uważasz, że oznaczone niebieską obwódką koło kryło obiekt będący celem, dotknij palcem jego odpowiednik na ekranie. Jeżeli uważasz, że oznaczone niebieską obwódką czarne koło kryło obiekt, który nie był celem, wybierz czarne koło znajdujące się na górnej części okręgu złożonego z obiektów. Gdy udzielisz odpowiedzi i ponownie naciśniesz oznaczony klawisz rozpocznie się kolejna próba. \n \n \
Będzie {training_trials_per_block} prób, podczas których nauczysz się wykonywać zadanie. Po treningu rozpoczną się próby właściwe. \n \
Jeśli masz jakieś pytania lub wątpliwości, możesz zgłosić je teraz osobie prowadzącej. Jeżeli nie, wciśnij oznaczony klawisz', aby przejść dalej."

instructions_mit_mot = {"MIT" : instruction_main_mit, 
                        "MOT" : instruction_main_mot} 


info_about_experiment_start = f"Koniec bloku treningowego. Rozpoczynamy badanie. Wciśnij oznaczony klawisz, aby przejść dalej do bloku zadania "
info_experiment_end = f"Koniec badania. Dziękujemy za udział."
instruction_1 = "Wciśnij przycisk, aby rozpocząć następną próbę i trzymaj go naciśnięty do momentu zadania z ruchomym czarnym kołem."
instruction_2 = "Oznaczony był {ground_truth}. Badany wybrał {choice}."
instruction_2_MIT = "Zadanie I) Oznaczony był {ground_truth}. Badany wybrał {choice}. \n\n Zadanie II) {identificiation}"
instruction_before_block = "Blok został zakończony. Możesz odpocząć, jeśli potrzebujesz. Wciśnij przycisk, aby rozpocząć blok zadania "
random_direction_small_circles = True
random_direction_big_circle = True
random_offset_target_distractor = True
random_offset_circles = True
random_distractor_target_orientation = True

key_to_click = '1'
camera_pause = 2.3

if is_windows_OS:
    path_for_mit_icons = r"D:\Desktop\MIT_MOT_experiment/icons_mit/" + mit_objects_folder
    results_dir = r"D:\Desktop\MIT_MOT_experiment/results/"
else:
    path_for_mit_icons = "/home/janek/psychologia/MIT_MOT_experiment/icons_mit/" + mit_objects_folder
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
    experiment = Experiment(win_size = win_size, full_size = full_size, results_dir = results_dir, path_for_mit_icons = path_for_mit_icons, snowflakes_id_to_use = snowflakes_id_to_use, key_to_click = key_to_click)
    experiment.set_mouse_visibility(mouse_visible)
    experiment.upload_param(
    show_circles=show_circles, 
    circle_radius=circle_radius, 
    small_circle_radius=small_circle_radius, 
    obj_radius=obj_radius, 
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
    snowflakes_id_to_use=snowflakes_id_to_use,
    img_mode=img_mode, 
    motoric_radius=motoric_radius, 
    motoric_circle_radius=motoric_circle_radius, 
    hz_motoric=hz_motoric, 
    answer_1_time_limit=answer_1_time_limit, 
    answer_MIT_time_limit=answer_MIT_time_limit, 
    motor_task_time_limit=motor_task_time_limit,
    key_to_click = key_to_click,
    camera_pause = camera_pause,
    is_windows_OS=is_windows_OS
)
    experiment.initialize_mot()
    experiment.initialize_mit()
    
    block_type_list = ["MIT", "MOT"]
    random.shuffle(block_type_list)
    
    if training_trials_per_block > 0:
        experiment.show_text(instructions_mit_mot[block_type_list[0]])
        experiment.run_block(block_type_list[0],training_trials_per_block,target_circles_ammount_settups[block_type_list[0]], is_training=True)
        experiment.show_text(instructions_mit_mot[block_type_list[1]])
        experiment.run_block(block_type_list[1],training_trials_per_block,target_circles_ammount_settups[block_type_list[1]], is_training=True)
    
    experiment.show_text(info_about_experiment_start + block_type_list[0])
    for i in range(block_pairs_number):
        if i > 0:
            experiment.show_text(instruction_before_block + block_type_list[0])
        experiment.run_block(block_type_list[0],trials_per_block,target_circles_ammount_settups[block_type_list[0]], is_training=False)
        experiment.show_text(instruction_before_block + block_type_list[1])
        experiment.run_block(block_type_list[1],trials_per_block,target_circles_ammount_settups[block_type_list[1]], is_training=False)
    experiment.show_text(info_experiment_end)
    experiment.close()
    if do_inference_after:
        model_inference(experiment.dir_name)
        model_inference3D(experiment.dir_name)
    
