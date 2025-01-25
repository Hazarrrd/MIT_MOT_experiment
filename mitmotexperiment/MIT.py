from mitmotexperiment.Trial import Trial
from psychopy import visual, core, event, gui
import numpy as np
import math
import os
import random

class MIT(Trial):
    
    color_dict = {
        1: "orange",
        2: "purple",
        3: "blue",
        4: "yellow",
        5: "gold",
        6: "white",
        7: "brown",
        8: "pink",
        9: "cyan",
        10: "magenta",
        11: "lime",
        12: "indigo"
    }
    color_list = []

    def __init__(self, *args):
        super().__init__(*args)
        self.black_path, self.png_files = MIT.get_png_files(self.path_for_mit_icons)

    # Function to get list of PNG file paths in a folder
    @staticmethod
    def get_png_files(folder_path):
        # List to store full paths of PNG files
        png_files = []
        
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is a PNG file
            if filename.lower().endswith(".png"):
                # Create full path and append to the list
                full_path = os.path.join(folder_path, filename)
                if 'black' in filename:
                    black_path = full_path
                else:
                    png_files.append(full_path)
        random.shuffle(png_files)
        return black_path, png_files

    def do_single_trial(self, block_id, tral_id):
        super().do_single_trial(block_id, tral_id)
        self.randomize_variables()
        if self.img_mode:
            self.create_objects()
        else:
            self.create_objects_circle()

        # Display Instructions
        instr = visual.TextStim(self.win, text=self.instruction_1, color="black")
        instr.draw()
        self.win.flip()
        event.waitKeys()

        # Start the Observation Phase
        self.observation_phase()

        # Start the Tracking Phase
        self.tracking_phase()

        # Hide Targets (all objects look the same)
        if self.img_mode:
            self.hide_objects()
        else:
            self.hide_objects_circle()

        # Choose object for guessing
        guess_object = self.revealing_guess_object()

        self.motoric_task()

        # Answering window
        ground_truth, choice = self.answer_if_distractor_or_target(guess_object)
        if choice != -1:
            # Indicate which target, for MIT
            success = self.indicate_target(guess_object)

            # Results window
            if self.show_trial_results and success != -1:
                self.show_results_window(ground_truth, choice, success)

    def create_objects(self):
        self.targets = [visual.ImageStim(self.win, size=(self.obj_radius*2, self.obj_radius*2), image=self.png_files[i], pos=None) for i in range(self.n_targets)]
        self.distractors = [visual.ImageStim(self.win, size=(self.obj_radius*2, self.obj_radius*2), image=self.png_files[self.n_targets + i], pos=None) for i in range(self.n_distractors)]
        self.small_circles = [visual.Circle(self.win, radius=self.small_circle_radius, lineColor="white", fillColor=None, pos=None) for i in range(self.n_circles)]

        # Combine targets and distractors for tracking
        if self.show_circles:
            # Create big circle object
            main_circle = visual.Circle(self.win, radius=self.circle_radius, pos=(0, 0), lineColor="white", fillColor=None)
            self.all_objects = [main_circle] + self.small_circles + self.targets + self.distractors
        else:
            self.all_objects = self.targets + self.distractors

    def create_objects_circle(self):
        # Create target and distractor objects
        self.color_list = np.random.permutation(self.n_targets*2) + 1
        self.targets = [visual.Circle(self.win, radius=self.obj_radius, fillColor=self.color_dict[self.color_list[i]], pos=None) for i in range(self.n_targets)]
        self.distractors = [visual.Circle(self.win, radius=self.obj_radius, fillColor=self.color_dict[self.color_list[self.n_targets + i]], pos=None) for i in range(self.n_distractors)]
        self.small_circles = [visual.Circle(self.win, radius=self.small_circle_radius, lineColor="white", fillColor=None, pos=None) for i in range(self.n_circles)]

        # Combine targets and distractors for tracking
        if self.show_circles:
            # Create big circle object
            main_circle = visual.Circle(self.win, radius=self.circle_radius, pos=(0, 0), lineColor="white", fillColor=None)
            self.all_objects = [main_circle] + self.small_circles + self.targets + self.distractors
        else:
            self.all_objects = self.targets + self.distractors
    
    def hide_objects(self):
        for target in self.targets:
            target.image = self.black_path
        for distractor in self.distractors:
            distractor.image = self.black_path
    
    def uncover_objects(self):
        for i, target in enumerate(self.targets):
            target.image = self.png_files[i]
        for i, distractor in enumerate(self.distractors):
            distractor.image = self.png_files[self.n_targets + i]

    def hide_objects_circle(self):
        for target in self.targets:
            target.fillColor = "black"

        for distractor in self.distractors:
            distractor.fillColor = "black"
    
    def uncover_objects_circle(self):
        for i, target in enumerate(self.targets):
            target.fillColor = self.color_dict[self.color_list[i]]
        for i, distractor in enumerate(self.distractors):
            distractor.fillColor = self.color_dict[self.color_list[i+ self.n_targets]]

    @staticmethod
    def _place_objects_in_circle(radius, objects):
        n = len(objects)
        angle_step = 2 * math.pi / n
        for i, obj in enumerate(objects):
            # Calculate the angle for the current circle
            angle = angle_step * i + math.pi/2
            # Convert polar coordinates (radius, angle) to Cartesian coordinates (x, y)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            obj.pos=(x, y)

    def indicate_target(self, guess_object):
        ## get proper colors back
        if self.img_mode:
            self.uncover_objects()
        else:
            self.uncover_objects_circle()
        multipl_radius = 2
        distr = visual.Circle(self.win, radius=self.obj_radius*multipl_radius, pos=(0, 0), fillColor='red', lineColor='red')
        random_order_obj = [distr] + np.random.permutation(self.targets + self.distractors).tolist()
        ## increasing radius for choosing
        for obj in self.targets + self.distractors:
            obj.size *= multipl_radius
        
        ## set new positions for objects
        MIT._place_objects_in_circle(self.obj_radius*(len(self.targets)+1) * multipl_radius, random_order_obj)
        for obj in random_order_obj:
            obj.draw()
        self.win.flip()

        # Capture Mouse Clicks and Identify Targets
        mouse = event.Mouse(win=self.win)
        success = None

        task_time = 0
        start_time = core.getTime()
        actual_time = start_time
        while task_time <= self.answer_MIT_time_limit:
            if mouse.getPressed()[0]:  # Left mouse click
                click_pos = mouse.getPos()
                for obj in self.targets + self.distractors:
                    if obj.contains(click_pos):
                        if obj == guess_object:
                            success = 1
                        else:
                            success = 0
                        return success
                if distr.contains(click_pos):
                    if guess_object in self.distractors:
                        success = 2
                    else:
                        success = 3
                    return success
            actual_time = core.getTime()
            task_time = actual_time-start_time
        if task_time > self.answer_MIT_time_limit:
            success = -1
        return success

    # Show Results and Exit
    def show_results_window(self, ground_truth, choice, success):
        text_success = ""
        if success==1:
            text_success = "The object has been identified correctly."
        elif success==0:
            text_success = "The object has been NOT identified correctly."
        elif success==2:
            text_success = "The object has been identified as distractor."
        elif success==3:
            text_success = "The object has been WRONGLY identified as distractor."
        result_text = self.instruction_2_MIT.format(ground_truth=ground_truth, choice=choice, identificiation = text_success)
        result_msg = visual.TextStim(self.win, text=result_text, color="black")
        result_msg.draw()
        self.win.flip()
        #core.wait(3)
        event.waitKeys()