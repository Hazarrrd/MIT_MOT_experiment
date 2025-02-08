from mitmotexperiment.Trial import Trial
from psychopy import visual, core, event, gui
from datetime import datetime
import pandas as pd
import os

class MOT(Trial):
    
    def do_single_trial(self, block_id, tral_id):
        super().do_single_trial(block_id, tral_id)
        current_datetime = datetime.now()
        self.new_row_df = {'Block': block_id, 
                           'Trial': tral_id, 
                           'Type': 'MOT', 
                           'Timestamp': current_datetime.strftime('%Y-%m-%d %H:%M:%S'), 
                           'Ground_truth_guess': None, 
                           'Guess': None, 
                           'Guess_success': None, 
                           'MIT_obj_identified': 'MOT', 
                           'TargetX': None, 
                           'TargetY': None, 
                           'ClickX': None, 
                           'ClickY': None, 
                           'Norm_Euc_Dist': None, 
                           'Task_time': None, 
                           'Movement_start': None, 
                           'Movement_duration': None}
        self.randomize_variables()
        self.create_objects()

        # Display Instructions
        instr = visual.TextStim(self.win, text=self.instruction_1, color="black")
        instr.draw()
        self.win.flip()
        event.waitKeys()

        # Start the Observation Phase
        self.observation_phase()

        # Start the Tracking Phase
        self.tracking_phase()

        # Choose object for guessing
        guess_object = self.revealing_guess_object()
        task_data = self.motoric_task()
        self.new_row_df.update({k: task_data[k] for k in task_data if k in self.new_row_df})
        # Answering window
        ground_truth, choice = self.answer_if_distractor_or_target(guess_object)
        if choice != -1:
            # Results window
            if self.show_trial_results:
                self.show_results_window(ground_truth, choice)
                
        self.new_row_df['Ground_truth_guess'] = ground_truth
        self.new_row_df['Guess'] = choice
        self.new_row_df['Guess_success'] = int(choice == ground_truth) if choice != -1 else choice
        
        self.df.loc[len(self.df)] = self.new_row_df
        self.df.to_csv(os.path.join(self.dir_name, f"{os.path.basename(os.path.normpath(self.dir_name))}.csv"), index=True)
        
    def create_objects(self):
        # Create target and distractor objects
        self.targets = [visual.Circle(self.win, radius=self.obj_radius, fillColor="black", pos=None) for i in range(self.n_targets)]
        self.distractors = [visual.Circle(self.win, radius=self.obj_radius, fillColor="black", pos=None) for i in range(self.n_distractors)]
        self.small_circles = [visual.Circle(self.win, radius=self.small_circle_radius, lineColor="white", fillColor=None, pos=None) for i in range(self.n_circles)]

        # Combine targets and distractors for tracking
        if self.show_circles:
            # Create big circle object
            main_circle = visual.Circle(self.win, radius=self.circle_radius, pos=(0, 0), lineColor="white", fillColor=None)
            self.all_objects = [main_circle] + self.small_circles + self.targets + self.distractors
        else:
            self.all_objects = self.targets + self.distractors

    # Show Results and Exit
    def show_results_window(self, ground_truth, choice):
        result_text = self.instruction_2.format(ground_truth=ground_truth, choice=choice)
        result_msg = visual.TextStim(self.win, text=result_text, color="black")
        result_msg.draw()
        self.win.flip()
        #core.wait(3)
        event.waitKeys()