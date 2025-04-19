from psychopy import visual, core, event, gui
from mitmotexperiment.MOT import MOT
from mitmotexperiment.MIT import MIT
from mitmotexperiment.ParticipantForm import ParticipantForm
from datetime import datetime
import os
import sys
import pandas as pd
import numpy as np
import json
import random

class Experiment():
    def __init__(self, win_size = (800,800), full_size = True, results_dir = None, path_for_mit_icons = None, snowflakes_id_to_use = None):
        # Create the Window
        self.win = visual.Window(win_size, fullscr=False, color="gray", units="pix")
    #    event.globalKeys.add(key='q', func=self.escape_handler)
       # self.win.winHandle.set_fullscreen(False)
       # self.win.winHandle.minimize()
        self.win.flip()
        self.form = ParticipantForm(self.win)
        self.form.show_form()
        self.win.close()
        self.win = visual.Window(win_size, fullscr=full_size, color="gray", units="pix")
        event.globalKeys.add(key='q', func=self.escape_handler)
      #  self.win.winHandle.maximize()
      #  self.win.winHandle.activate()
      #  self.win.fullscr = full_size
      #  self.win.winHandle.set_fullscreen(full_size)
        self.win.flip()
        self.block_id = 0
        
        # Generate a unique directory name with the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        dir_name = os.path.join(results_dir, f"experiment_{timestamp}")

        # Create the directory if it doesn't exist
        if not os.path.exists(dir_name):
            os.makedirs(os.path.join(dir_name,"videos"))
            print(f"Directory '{dir_name}' created.")
        else:
            print(f"Directory '{dir_name}' already exists.")
            sys.exit()
        self.dir_name = dir_name
     
        # Define data
        data = {
            'ID': [],
            'Sex': [],
            'Age': [],
            'Block': [],
            'Trial': [],
            'Type': [],
            'Is_training': [],
            'N_targets': [],
            'N_distractors': [],
            'N_circles': [],
            'Timestamp': [],
            'Ground_truth_guess': [],
            'Guess': [],
            'Guess_success': [],
            'Task_time_guess': [],
            'MIT_obj_identified': [],
            'Task_time_identification': [],
            'TargetX': [],
            'TargetY': [],
            'ClickX': [],
            'ClickY': [],
            'Motoric_obj_Vx': [],
            'Motoric_obj_Vy': [],
            'Motoric_obj_V1_magnitude': [],
            'Motoric_click_V2_magnitude': [],
            'Angle_objV_click': [],
            'Norm_Euc_Dist': [],
            'Task_time_motoric': [],
            'Movement_start': [],
            'Movement_duration': [],
            "Indicated_img": [],
            "Img_to_guess": [],
        }
        _, png_files = MIT.get_png_files(path_for_mit_icons, snowflakes_id_to_use)
        png_files = sorted(png_files)
        for i in range(len(png_files)):
            data[os.path.basename(png_files[i])] = []
                
        # Create a DataFrame
        df = pd.DataFrame(data)
        # Save DataFrame to CSV
        df.to_csv(os.path.join(self.dir_name, f"{os.path.basename(os.path.normpath(self.dir_name))}.csv"), index=True)
        self.df = df
        #create general dataframe
     #   if not os.path.exists(os.path.join(results_dir, "results.csv")):
     #       df.to_csv(os.path.join(results_dir, "results.csv"), index=True)
        
    def set_mouse_visibility(self,mouse_visible):
        mouse = event.Mouse(win=self.win)
        mouse.setVisible(mouse_visible)
        
    def show_text(self, text):
        """
        Displays the given text in a PsychoPy window for the specified duration.
        
        Parameters:
        text (str): The text to display.
        duration (float): The time in seconds to display the text. Default is 2 seconds.
        """    
        # Create a text stimulus
        text_stim = visual.TextStim(self.win, wrapWidth=int(self.win.size[0]*0.9),  text=text, color="black")
        
        # Draw the text stimulus and flip the window
        text_stim.draw()
        self.win.flip()   
        # Wait for the duration or until a key is pressed
        while True:
            keys = event.getKeys()
            if 'down' in keys or '4' in keys:
                break
    
    def upload_param(self, **kwargs):
        self.params = list(kwargs.values())
        # Save parameters to a JSON file in the created directory
        json_file_path = os.path.join(self.dir_name, "experiment_params.json")
        kwargs['win_size'] = [int(self.win.size[0]),int(self.win.size[1])]
        with open(json_file_path, 'w') as json_file:
            json.dump(kwargs, json_file, indent=4)
        print(f"Parameters saved to {json_file_path}")

    def escape_handler(self):
        self.win.close()
        core.quit()

    def initialize_mit(self):
        self.mit_obj = MIT(self.win, *self.params, self.dir_name,self.df, self.form)
    
    def initialize_mot(self):
        self.mot_obj = MOT(self.win, *self.params, self.dir_name,self.df, self.form)
    
    def run_block(self, block_type, block_size, target_circles_ammount_settups, is_training = False):
        if block_type == "MOT":
            block_obj = self.mot_obj
        elif block_type == "MIT":
            block_obj = self.mit_obj
            
        if block_size % len(target_circles_ammount_settups) != 0:
            raise ValueError('ERROR: block_size must be a multiple of target_circles_ammount_settups list length to show each setup with equal frequency')
            self.close()
        expanded_list = []
        for i in range(len(target_circles_ammount_settups)):
            expanded_list.extend([target_circles_ammount_settups[i]] * (block_size//len(target_circles_ammount_settups)))
        if not is_training:
            random.shuffle(expanded_list)
        self.block_id +=1
        trial_id = 0
        for n_targets, n_circles in expanded_list:
            trial_id +=1
            block_obj.set_params_for_block(n_targets, n_circles)
            block_obj.do_single_trial(self.block_id, trial_id, is_training)

    def close(self):
        """Closes the experiment window and releases resources."""
        if hasattr(self, 'win') and self.win:
           # core.quit()
            self.win.close()
            print("Window closed.")
        else:
            print("Window was already closed.")
