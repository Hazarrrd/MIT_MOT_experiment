from psychopy import visual, core, event, gui
import numpy as np
import cv2
import os
import time
import threading
import math
import subprocess
from psychopy.hardware import keyboard
import ctypes
#xlib = ctypes.cdll.LoadLibrary("libX11.so")
#xlib.XInitThreads()
## Uncomment it if you want reproduce same randomness each program execution
#np.random.seed(12201)


class Trial():
    def __init__(self, win, show_circles, circle_radius, small_circle_radius, obj_radius, hz_target, hz_circle, instruction_1, instruction_2, instruction_2_MIT, fps,
                random_direction_small_circles, random_direction_big_circle, random_offset_target_distractor, random_offset_circles, random_distractor_target_orientation,
                observation_time,tracking_time, guessing_time, direction_changes, direction_changes_motoric, change_big_direction, show_trial_results, path_for_mit_icons, snowflakes_id_to_use, img_mode, 
                motoric_radius, motoric_circle_radius, hz_motoric, answer_1_time_limit, answer_MIT_time_limit, motor_task_time_limit, is_windows_OS, dir_name, df, form):
        self.is_windows_OS = is_windows_OS
        self.form = form
        self.motoric_movement_start = -1
        self.win = win
        self.df = df
        self.show_circles = show_circles
        self.circle_radius = circle_radius
        self.small_circle_radius = small_circle_radius
        self.obj_radius = obj_radius
        self.instruction_1 = instruction_1
        self.instruction_2 = instruction_2
        self.instruction_2_MIT = instruction_2_MIT
        self.observation_time = observation_time
        self.tracking_time = tracking_time
        self.guessing_time = guessing_time
        self.direction_changes = direction_changes
        self.direction_changes_motoric = direction_changes_motoric
        self.change_big_direction = change_big_direction
        self.show_trial_results = show_trial_results

        self.random_direction_small_circles = random_direction_small_circles
        self.random_direction_big_circle = random_direction_big_circle
        self.random_offset_target_distractor = random_offset_target_distractor
        self.random_offset_circles = random_offset_circles
        self.random_distractor_target_orientation = random_distractor_target_orientation
        self.fps = fps
        ## setting variables
        self.speed_objects = (hz_target/fps)*2*np.pi 
        self.speed_circle = (hz_circle/fps)*2*np.pi
        self.frame_duration = 1.0 / fps  # Duration of each frame in seconds
        self.img_mode = img_mode
        self.path_for_mit_icons = path_for_mit_icons
        self.snowflakes_id_to_use = snowflakes_id_to_use
        self.dir_name = dir_name

        self.motoric_radius = motoric_radius
        self.motoric_circle_radius = motoric_circle_radius
        self.speed_motoric = (hz_motoric/fps)*2*np.pi 

        self.answer_1_time_limit = answer_1_time_limit
        self.motor_task_time_limit = motor_task_time_limit
        self.answer_MIT_time_limit = answer_MIT_time_limit
        self.camera_is_recording = False
        self.kb = keyboard.Keyboard()

    def set_params_for_block(self, n_targets, n_circles):
        self.n_targets = n_targets
        self.n_circles = n_circles
        self.n_distractors = 2*self.n_circles - self.n_targets
        self.n_objects = self.n_targets + self.n_distractors
        
    def randomize_variables(self):
        # Adding demanded randomness
        if self.random_direction_big_circle:
            self.big_circle_direction = np.random.choice([1,-1])
        else:
            self.big_circle_direction = 1

        if self.random_direction_small_circles:
            self.small_circle_direction = np.random.choice([1,-1], size=self.n_circles)
        else:
            self.small_circle_direction = np.ones(self.n_circles)

        if self.random_offset_target_distractor:
            objects_offset = np.random.uniform(-np.pi/2, np.pi/2, size=self.n_circles)
        else:
            objects_offset = np.zeros(self.n_circles) + np.pi/2

        circles_offset = np.pi/2 + np.pi/self.n_circles
        if self.random_offset_circles:
            circles_offset += np.random.uniform(-(np.pi/self.n_circles)/3,(np.pi/self.n_circles)/3) 

        self.time_of_mov_changes = []
        if self.direction_changes:
            for time_frame in self.direction_changes:
                self.time_of_mov_changes.append(np.random.uniform(time_frame[0], time_frame[1], size=self.n_circles).tolist())
                if self.change_big_direction:
                    self.time_of_mov_changes[-1].append(np.random.uniform(time_frame[0], time_frame[1]))
        self.time_of_mov_changes = sorted(list(zip(*self.time_of_mov_changes)))
        self.time_of_mov_changes = [sorted(list(a)) for a in self.time_of_mov_changes]

        self.time_of_mov_changes_motoric = []
        if self.direction_changes_motoric:
            for time_frame in self.direction_changes_motoric:
                self.time_of_mov_changes_motoric.append(np.random.uniform(time_frame[0], time_frame[1]))
        self.time_of_mov_changes_motoric = sorted(self.time_of_mov_changes_motoric)

        # Assign random directions for each object
        self.angles_pair_1 = np.array([])
        self.angles_pair_2 = np.array([])
        for i in range(self.n_circles):
            if self.random_distractor_target_orientation:
                rnd = np.random.choice([np.pi,0]) 
            else:
                rnd = np.pi
            self.angles_pair_1 = np.append(self.angles_pair_1, np.pi-rnd)
            self.angles_pair_2 = np.append(self.angles_pair_2,rnd)
        self.angles_pair_1 += objects_offset
        self.angles_pair_2 += objects_offset
        self.angles_circle = np.linspace(0, 2 * np.pi, self.n_circles, endpoint=False) + circles_offset

    ## abstract method 
    def do_single_trial(self, block_id, trial_id, is_training):
        self.block_id = block_id
        self.trial_id = trial_id
        self.is_training = is_training

    ## abstract method 
    def create_objects(self):
        pass

    # Function to update targets and distractors positions
    def update_positions(self, objects, angles, speed_object, small_circles, small_circle_direction):
        for i, obj in enumerate(objects):
            center_positions = small_circles[i].pos
            angles[i] = (angles[i]+ speed_object*small_circle_direction[i]) % (2*np.pi) ##care for modulo
            obj.pos = (center_positions[0] + self.small_circle_radius * np.cos(angles[i]), 
                        center_positions[1] + self.small_circle_radius * np.sin(angles[i]))

    # Function to update circles positions
    def update_circles(self):
        for i, obj in enumerate(self.small_circles):
            center_positions = (0, 0)
            self.angles_circle[i] = (self.angles_circle[i]+ self.big_circle_direction*self.speed_circle) % (2*np.pi) ##care for modulo
            obj.pos = (center_positions[0] + self.circle_radius * np.cos(self.angles_circle[i]), 
                        center_positions[1] + self.circle_radius * np.sin(self.angles_circle[i]))

    def observation_phase(self):
        # Start the Tracking Phase
        self.update_circles()
        self.update_positions(self.pair_1, self.angles_pair_1, self.speed_objects, self.small_circles, self.small_circle_direction)
        self.update_positions(self.pair_2, self.angles_pair_2, self.speed_objects, self.small_circles, self.small_circle_direction)
        
        for target in self.targets:
            visual.Circle(self.win, radius=self.obj_radius + 20, pos=(target.pos[0], target.pos[1]), fillColor=None, lineColor='green', lineWidth=10).draw()

        for obj in self.all_objects:
            obj.draw()

        self.win.flip()
        core.wait(self.observation_time)

    def update_directions(self, track_time):
        for idx, move_change in enumerate(self.time_of_mov_changes):
            if move_change:
                if move_change[0] < track_time:
                    if idx < len(self.small_circle_direction):
                        self.small_circle_direction[idx] *= -1
                    else:
                        self.big_circle_direction *= -1
                    del move_change[0]
                        

    def tracking_phase(self):
        # Start the Tracking Phase
        start_time = core.getTime()
        actual_time = start_time
        while actual_time - start_time < self.tracking_time:
            self.update_directions(actual_time - start_time)
            self.update_circles()
            self.update_positions(self.pair_1, self.angles_pair_1, self.speed_objects, self.small_circles, self.small_circle_direction)
            self.update_positions(self.pair_2, self.angles_pair_2, self.speed_objects, self.small_circles, self.small_circle_direction)

            for obj in self.all_objects:
                obj.draw()
            self.win.flip()
            frame_elapsed = core.getTime()-actual_time
            if frame_elapsed < self.frame_duration:
                core.wait(self.frame_duration - frame_elapsed)
            else:
                print(f"WARRNING: FPS may be too big {self.fps}, what may cause speed errors")
            actual_time = core.getTime()

    def revealing_guess_object(self):   
        for obj in self.all_objects:
            obj.draw()
        to_guess_obj_or_dist = self.targets if np.random.rand() < 0.5 else self.distractors
        to_guess = np.random.choice(to_guess_obj_or_dist)
        visual.Circle(self.win, radius=self.obj_radius + 20, pos=(to_guess.pos[0], to_guess.pos[1]), fillColor=None, lineColor='blue', lineWidth=10).draw()
        self.win.flip()
        core.wait(self.guessing_time)
        return to_guess

    def motoric_task(self):
        objectibe_circle = visual.Circle(self.win, radius=self.motoric_circle_radius, lineColor="white", fillColor=None, pos=(0,0))
        objective = visual.Circle(self.win, radius=self.motoric_radius, fillColor="black", pos=None)
        task_data = {'TargetX': None,
                    'TargetY': None,
                    'ClickX': None,
                    'ClickY': None,
                    'Norm_Euc_Dist': None,
                    'Task_time_motoric': None,
                    'Movement_start': None,
                    'Movement_duration': None,
                    }
        # Define the cross using ShapeStim
        cross = [visual.ShapeStim(
            self.win,
            vertices=None,
            lineWidth=8,
            closeShape=True,
            lineColor="white"
        ) for i in range(2)]
        how_long = self.motoric_radius//5

        objective_position = np.random.uniform(-np.pi/2, np.pi/2)
        objective_direction = np.random.choice([1,-1])

        video_filename = os.path.join(os.path.join(self.dir_name,"videos"), f"{self.__class__.__name__}_block_{self.block_id}_trial{self.trial_id}")
        self.motoric_movement_start = -1
        if self.is_windows_OS:
            cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap=cv2.VideoCapture(0)
            
        self.camera_is_recording = False
        if cap and cap.isOpened():
            self.camera_is_recording = True
           # cap.release()
        if self.camera_is_recording:
            cam_thread = threading.Thread(target=self.camera_recording_opencv, args=(video_filename,cap))
           # cam_thread = threading.Thread(target=self.camera_recording_ffmpg, args=(video_filename,))
            cam_thread.start()
        
        mouse = event.Mouse(win=self.win)
        self.win.flip()
        time.sleep(2) ## camera starting time
       
        keys = []
        last_key_time = 0
        
        self.kb.clock.reset()  # when you want to start the timer from
        keys = self.kb.getKeys( waitRelease=False)
        if len(keys)==1 and (keys[0].value == "down" or keys[0].value == "4"):
            task_time = 0
            start_time = core.getTime()
            actual_time = start_time
            change_direction = 1
            while not mouse.getPressed()[0] and task_time < self.motor_task_time_limit:
                self.win.flip()
                if self.motoric_movement_start == -1 and not self.kb.getState(keys):
                    self.motoric_movement_start = core.getTime()
                    print(f"Hand movement started {self.motoric_movement_start-start_time}")
                if self.time_of_mov_changes_motoric:
                    for move_change in self.time_of_mov_changes_motoric:
                        if move_change < task_time:
                            change_direction *= -1
                            del self.time_of_mov_changes_motoric[0]
                objective_position = (objective_position + self.speed_motoric*objective_direction*change_direction) % (2*np.pi) ##care for modulo
                objective.pos = (self.motoric_circle_radius * np.cos(objective_position), 
                                self.motoric_circle_radius * np.sin(objective_position))
                
                cross[0].setVertices([
                    [objective.pos[0], objective.pos[1] + how_long], [objective.pos[0], objective.pos[1] - how_long]   # Vertical line
                ])
                cross[1].vertices = (
                    (objective.pos[0] - how_long, objective.pos[1]), (objective.pos[0] + how_long , objective.pos[1])  # Horizontal line
                
                )
                
                objective.draw()
                cross[0].draw()
                cross[1].draw()

                frame_elapsed = core.getTime()-actual_time
                if frame_elapsed < self.frame_duration:
                    core.wait(self.frame_duration - frame_elapsed)
                else:
                    print(f"WARRNING: FPS may be too big {self.fps}, what may cause speed errors")
                actual_time = core.getTime()
                task_time = actual_time-start_time
                click_pos = mouse.getPos()
            task_data['TargetX'] = objective.pos[0]
            task_data['TargetY'] = objective.pos[1]
            task_data['Task_time_motoric'] = task_time
            if task_time <= self.motor_task_time_limit:
                task_data['ClickX'] = click_pos[0]
                task_data['ClickY'] = click_pos[1]
                task_data['Movement_start'] = self.motoric_movement_start-start_time
                task_data['Norm_Euc_Dist'] = math.sqrt(pow((objective.pos[0]- click_pos[0])/self.win.size[0],2)+pow((objective.pos[1]- click_pos[1])/self.win.size[1],2))
                task_data['Movement_duration'] = task_time-(self.motoric_movement_start-start_time)
                for key, val in task_data.items():
                    print(f"{key}: {val}")
            else:
                print("Task time out")
                task_data['ClickX'] = -1
                task_data['ClickY'] = -1
                task_data['Movement_start'] = -1
                task_data['Norm_Euc_Dist'] = -1
                task_data['Movement_duration'] = -1
        else:
            print("No key 'left' pressed - failed trial")
            task_data['TargetX'] = -1
            task_data['TargetY'] = -1
            task_data['Task_time_motoric'] = -1
            task_data['ClickX'] = -1
            task_data['ClickY'] = -1
            task_data['Movement_start'] = -1
            task_data['Norm_Euc_Dist'] = -1
            task_data['Movement_duration'] = -1
        self.win.flip()
        if self.camera_is_recording:
            self.camera_is_recording = False
            cam_thread.join()
            if cap.isOpened():
                cv2.destroyAllWindows()
                cap.release()
        event.clearEvents(eventType='keyboard')
        return task_data
        #model_inference_thread = threading.Thread(target=self.model_inference, args=(video_filename,))
        #model_inference_thread.start()
        #if cap.isOpened():
        #    cam_thread.join()
        #    cv2.destroyAllWindows()
       # cap.release()

    def camera_recording_ffmpg(self, video_filename):
        #frame_width, frame_height, FPS_DESIRE, format_cam = 640, 480, 30, "yuyv422"
        frame_width, frame_height, FPS_DESIRE, format_cam = 1280, 720, 60, "mjpeg"
        # Build ffmpeg command
        command = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f", "v4l2",  # Use the Video4Linux2 driver
            "-input_format", format_cam,  # Set input format
            "-video_size", f"{frame_width}x{frame_height}",  # Set frame size
            "-framerate", str(FPS_DESIRE),  # Set frame rate
            "-i", "/dev/video0",  # Input device
          #  "-vsync","0",
            "-c:v", "libx264",  # Codec for AVI (MJPEG is a common choice)
            "-f", "mp4",
            "-r", f"{FPS_DESIRE}",
            #"-pix_fmt", "yuvj422p",  # Ensure a compatible pixel format for MJPEG (yuvj422p)
          #  "-q:v", "5",  # Set video quality (lower is better quality, higher is worse)
          
            video_filename + ".mp4"  # Output file
        ]
        print(" ".join(command))
        process = subprocess.Popen(command, stdout=None, stderr=None)
        while self.camera_is_recording:
            # Your main application can perform other tasks here
         #   print("Recording in progress...")
            time.sleep(0.1)  # You can perform other tasks instead of sleeping
        # Terminate the ffmpeg subprocess after 10 seconds
        process.terminate()

        # Optionally, wait for the process to terminate cleanly
        process.wait()

    def camera_recording_opencv(self, video_filename, cap):

       # frame_width, frame_height, FPS_DESIRE, format_cam = 640, 480, 30, "YUYV"
        frame_width, frame_height, FPS_DESIRE, format_cam = 1280, 720, 60, "MJPG"
        # Set the capture properties (resolution and FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cap.set(cv2.CAP_PROP_FPS, FPS_DESIRE)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format_cam))  # MJPEG codec
#MP4V
        # Create VideoWriter object to save the video
        # Using 'XVID' or 'MJPG' codec for .avi or .mp4
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or use 'XVID' for .avi - MP4V for ubuntu
        out = cv2.VideoWriter(video_filename + ".avi", fourcc, FPS_DESIRE, (frame_width, frame_height))

        
        while self.camera_is_recording:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and self.motoric_movement_start > -1:
                    out.write(frame)
        out.release()

    def answer_if_distractor_or_target(self, guess_object):
        mouse = event.Mouse(win=self.win)
        rnd = np.random.choice([-1,1])
        green_dot = visual.Circle(self.win, radius=self.obj_radius*2.5, pos=(0 + rnd*2*self.obj_radius*2, 0), fillColor='green', lineColor='green')
        red_dot = visual.Circle(self.win, radius=self.obj_radius*2.5, pos=(0 - rnd*2*self.obj_radius*2, 0), fillColor='red', lineColor='red')
        red_dot.draw()
        green_dot.draw()
        self.win.flip()

        if guess_object in self.targets:
            ground_truth = "target"
        else:
            ground_truth = "distractor"
        
        task_time = 0
        start_time = core.getTime()
        actual_time = start_time
        while task_time <= self.answer_1_time_limit:
            mouse_pos = mouse.getPos()
         #   print(f"{mouse.getPressed()}   {mouse_pos}")  # Will return a tuple (x, y) position of the mouse
            if mouse.getPressed()[0]:  # Left mouse click
                click_pos = mouse.getPos()
                if red_dot.contains(click_pos):
                    # participant thinks guessing object is distractor
                    choice = "distractor"
                    break
                elif green_dot.contains(click_pos):
                    # participant thinks guessing object is target
                    choice = "target"
                    break
            actual_time = core.getTime()
            task_time = actual_time-start_time
        if task_time > self.answer_1_time_limit:
            choice = -1
        return ground_truth, choice, task_time

    ## abstracion function
    def show_results_window(self, ground_truth, choice):
        pass


