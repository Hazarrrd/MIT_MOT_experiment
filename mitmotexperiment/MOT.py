from mitmotexperiment.Trial import Trial
from psychopy import visual, core, event, gui
from datetime import datetime
import pandas as pd
import os
import random

class MOT(Trial):
    
    def do_single_trial(self, block_id, tral_id, is_training):
        super().do_single_trial(block_id, tral_id, is_training)
        current_datetime = datetime.now()
        self.new_row_df = {
                            'First_name': self.form.form_data["Imię:"],
                            'Last_name': self.form.form_data["Nazwisko:"],
                            'Gender': self.form.form_data["Płeć:"],
                            'Email': self.form.form_data["Email:"],
                            'Block': block_id, 
                            'Trial': tral_id, 
                            'Type': 'MOT', 
                            'Is_training': int(is_training),
                            'N_targets': self.n_targets,
                            'N_distractors': self.n_distractors,
                            'N_circles': self.n_circles,
                            'Timestamp': current_datetime.strftime('%Y-%m-%d %H-%M-%S'), 
                            'Ground_truth_guess': None, 
                            'Guess': None, 
                            'Guess_success': None, 
                            'Task_time_guess': None,
                            'MIT_obj_identified': 'MOT',
                            'Task_time_identification': 'MOT', 
                            'TargetX': None, 
                            'TargetY': None, 
                            'ClickX': None, 
                            'ClickY': None, 
                            'Norm_Euc_Dist': None, 
                            'Task_time_motoric': None, 
                            'Movement_start': None, 
                            'Movement_duration': None,
                            "Indicated_img": 'MOT',
                            "Img_to_guess": 'MOT'}
        self.randomize_variables()
        self.create_objects()
        
        # Display Instructions
        instr = visual.TextStim(self.win, text=self.instruction_1, color="black")
        instr.draw()
        self.win.flip()
        while True:
            keys = event.getKeys()
            if 'down' in keys:
                break

        # Start the Observation Phase
        self.observation_phase()

        # Start the Tracking Phase
        self.tracking_phase()

        # Choose object for guessing
        guess_object = self.revealing_guess_object()
        task_data = self.motoric_task()
        core.wait(0.2)
        self.new_row_df.update({k: task_data[k] for k in task_data if k in self.new_row_df})
        # Answering window
        ground_truth, choice, task_time_guess = self.answer_if_distractor_or_target(guess_object)
        self.new_row_df['Task_time_guess']= task_time_guess
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
        self.pair_1 = [target for target in self.targets]
        self.pair_2 = [distr for distr in self.distractors]

        while len(self.pair_1) != len(self.pair_2):
            if len(self.pair_1) > len(self.pair_2):
                self.pair_2.append(self.pair_1.pop())  # Move from pair1 → pair2
            else:
                self.pair_1.append(self.pair_2.pop())  # Move from pair2 → pair1
        random.shuffle(self.pair_1)
        random.shuffle(self.pair_2)
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
        while True:
            keys = event.getKeys()
            if 'down' in keys:
                break