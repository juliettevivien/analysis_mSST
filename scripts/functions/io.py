"""
In this file should be stored import-export functions.

"""

import os
from os.path import join
import json
import pandas as pd


def save_epochs(epochs, session_ID, filename, saving_path):
    """Save MNE Epochs object as a .fif file and return the filename."""
    epochs_file = os.path.join(saving_path, f"{session_ID}-{filename}.fif")
    epochs.save(epochs_file, overwrite=True)  # Save in MNE's preferred format
    return epochs_file  # Return file path to store in JSON


# Define a function to load JSON files and extract relevant data
def load_data_from_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
    

def load_behav_data(
        included_subjects,
        onedrive_path
):
    # load data by looping through the subjects
    data = {}
    for subject in included_subjects:
        print(subject)
        if subject.startswith('sub'):
            subject_ID, na, condition, task = subject.split(' ')
            sub_onedrive_path = join(onedrive_path, subject_ID)
            sub_onedrive_path_raw = join(sub_onedrive_path, 'raw_data')
            sub_onedrive_path_behav = join(sub_onedrive_path_raw, 'BEHAVIOR')
            sub_onedrive_path_condition = join(sub_onedrive_path_behav, (na + ' ' + condition))
            sub_onedrive_path_task = join(sub_onedrive_path_condition, task)
            behav_filename = [f for f in os.listdir(sub_onedrive_path_task) if f.endswith('.csv')]
            filepath = join(sub_onedrive_path_task, behav_filename[0])
            df = pd.read_csv(filepath)
            
        elif subject.startswith('C'):
            subject_ID, task = subject.split(' ')
            sub_onedrive_path = join(onedrive_path, subject_ID)
            sub_onedrive_path_raw = join(sub_onedrive_path, 'raw_data')
            sub_onedrive_path_behav = join(sub_onedrive_path_raw, 'BEHAVIOR')
            sub_onedrive_path_task = join(sub_onedrive_path_behav, task)
            behav_filename = [f for f in os.listdir(sub_onedrive_path_task) if f.endswith('.csv')]
            filepath = join(sub_onedrive_path_task, behav_filename[0])
            df = pd.read_csv(filepath)
        
        elif subject.startswith('preop'):
            subject_ID, task = subject.split(' ')
            sub_onedrive_path = join(onedrive_path, subject_ID)
            sub_onedrive_path_raw = join(sub_onedrive_path, 'raw_data')
            sub_onedrive_path_behav = join(sub_onedrive_path_raw, 'BEHAVIOR')
            sub_onedrive_path_behav_preop = join(sub_onedrive_path_behav, 'PRE OP')
            sub_onedrive_path_task = join(sub_onedrive_path_behav_preop, task)
            behav_filename = [f for f in os.listdir(sub_onedrive_path_task) if f.endswith('.csv')]
            filepath = join(sub_onedrive_path_task, behav_filename[0])
            df = pd.read_csv(filepath)        

        data[subject] = df

    return data