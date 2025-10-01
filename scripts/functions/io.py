"""
In this file should be stored import-export functions.

"""

import os
from os.path import join
import json
import pandas as pd
from pyxdf import resolve_streams
from numpy.core.records import fromarrays
from scipy.io import savemat

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



def find_EEG_stream(fpath_external, stream_name):
    # Determine which stream contains the EEG data:
    xdf_datas = resolve_streams(fpath_external)
    streams_dict = {}

    for streams in range(0, len(xdf_datas), 1):
        streams_dict[xdf_datas[streams]['name']] = xdf_datas[streams]['stream_id']
    
    # in streams_dict, find the stream_id corresponding to the EEG stream:
    stream_id = streams_dict[stream_name]

    return stream_id


def write_set(fname, raw, annotations_onset):
    """Export raw to EEGLAB .set file."""
    data = raw.get_data() * 1e6  # convert to microvolts
    fs = raw.info["sfreq"]
    times = raw.times
    ch_names = raw.info["ch_names"]
    chanlocs = fromarrays([ch_names], names=["labels"])
    events = fromarrays([raw.annotations.description,
                         annotations_onset * fs + 1,
                         raw.annotations.duration * fs],
                        names=["type", "latency", "duration"])
    savemat(fname, dict(EEG=dict(data=data,
                                 setname=fname,
                                 nbchan=data.shape[0],
                                 pnts=data.shape[1],
                                 trials=1,
                                 srate=fs,
                                 xmin=times[0],
                                 xmax=times[-1],
                                 chanlocs=chanlocs,
                                 event=events,
                                 icawinv=[],
                                 icasphere=[],
                                 icaweights=[])),
            appendmat=False)
    