from psychopy import visual, core, event, gui
from mitmotexperiment.MOT import MOT
from mitmotexperiment.MIT import MIT
from datetime import datetime
import os
import sys
import pandas as pd
import numpy as np

class Experiment():
    def __init__(self, win_size = (800,800), full_size = True, results_dir = None):
        # Create the Window
        self.win = visual.Window(win_size, fullscr=full_size, color="gray", units="pix")
        event.globalKeys.add(key='escape', func=self.escape_handler)
        self.block_id = 0
        
        # Generate a unique directory name with the current timestamp
        timestamp = datetime.now().strftime('%Y%-m-%d-%H:%M:%S')
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
            'Blok': [],
            'Trial': [],
            'Score': []
        }
        # Create a DataFrame
        df = pd.DataFrame(data)
        # Save DataFrame to CSV
        df.to_csv(os.path.join(self.dir_name, "results.csv"), index=False)
        self.df = df
        #create general dataframe
     #   if not os.path.exists(os.path.join(results_dir, "results.csv")):
     #       df.to_csv(os.path.join(results_dir, "results.csv"), index=True)
        


    def upload_param(self, *params):
        self.params = params

    def escape_handler(self):
        core.quit()

    def initialize_mit(self):
        self.mit_obj = MIT(self.win, *self.params, self.dir_name)
    
    def initialize_mot(self):
        self.mot_obj = MOT(self.win, *self.params, self.dir_name)
    
    def run_MIT_block(self, block_size):
        self.block_id +=1
        for i in range(block_size):
            self.mit_obj.do_single_trial(self.block_id, i)
    
    def run_MOT_block(self, block_size):
        self.block_id +=1
        for i in range(block_size):
            self.mot_obj.do_single_trial(self.block_id, i)

    def close(self):
        """Closes the experiment window and releases resources."""
        if hasattr(self, 'win') and self.win:
           # core.quit()
            self.win.close()
            print("Window closed.")
        else:
            print("Window was already closed.")
