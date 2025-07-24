"""
In this file should be stored utility functions.

"""

import os
import numpy as np
import json
import seaborn as sns
from os.path import join
import pandas as pd
from collections import defaultdict


def create_inhibition_df(
        included_subjects,
        stats
):
    # Create a DataFrame for the inhibition functions:
    rt_inhibition_df = pd.DataFrame(columns=[
        "subject", # only the first part of the subject name, e.g. "C001", "sub001"
        "condition", # "control", "preop", "DBS ON", "DBS OFF"
        "SSD",
        "RT"
        ])
    # loop through the stats and fill the inhibition_df
    for sub in included_subjects:
        ssd_counts = {}  # Dictionary to store counts of SSDs
        if "DBS OFF" in sub:
            condition = "DBS OFF"
        elif "DBS ON" in sub:
            condition = "DBS ON"
        elif "C" in sub:
            condition = "control"
        elif "preop" in sub:
            condition = "preop"
        df_unsuccess = pd.DataFrame({
            'subject': sub.split()[0],  # Extract the subject ID (e.g., "C001")
            'condition': condition,  # Extract the condition (e.g., "DBS ON", "DBS OFF", "control", "preop")
            'SSD': stats[sub]['unsuccessful stop SSD (ms)'], 
            'RT': stats[sub]['stop_trial RTs (ms)']
            })

        rt_inhibition_df = pd.concat([rt_inhibition_df, df_unsuccess], ignore_index=True)

    return rt_inhibition_df


def create_grouped_df_for_inhibitory_functions(
        included_subjects,
        stats,
):
    # Create a DataFrame for the inhibition functions:
    inhibition_df = pd.DataFrame(columns=[
        "subject", # only the first part of the subject name, e.g. "C001", "sub001"
        "condition", # "control", "preop", "DBS ON", "DBS OFF"
        "SSD",
        "p"
        ])
    # loop through the stats and fill the inhibition_df
    for sub in included_subjects:
        ssd_counts = {}  # Dictionary to store counts of SSDs
        if "DBS OFF" in sub:
            condition = "DBS OFF"
        elif "DBS ON" in sub:
            condition = "DBS ON"
        elif "C" in sub:
            condition = "control"
        elif "preop" in sub:
            condition = "preop"
        df_success = pd.DataFrame({
            'subject': sub.split()[0],  # Extract the subject ID (e.g., "C001")
            'condition': condition,  # Extract the condition (e.g., "DBS ON", "DBS OFF", "control", "preop")
            'SSD': stats[sub]['successful stop SSD (ms)'], 
            'p': 0
            })
        df_unsuccess = pd.DataFrame({
            'subject': sub.split()[0],  # Extract the subject ID (e.g., "C001")
            'condition': condition,  # Extract the condition (e.g., "DBS ON", "DBS OFF", "control", "preop")
            'SSD': stats[sub]['unsuccessful stop SSD (ms)'], 
            'p': 1
            })

        df = pd.concat([df_success, df_unsuccess], ignore_index=True)
        df.sort_values(by='SSD', inplace=True)
        
        unique_SSD = df['SSD'].unique()
        inhibition_df = pd.concat([inhibition_df, df], ignore_index=True)
        p_values = []
        for ssd in unique_SSD:
            subset = df[df['SSD'] == ssd]
            p_value = subset['p'].mean()
            ssd_counts[ssd] = len(subset)
            p_values.append(p_value)

        ssd_p_df = pd.DataFrame({'SSD': unique_SSD, 'p': p_values})

    # Group by subject, condition, and SSD, then calculate mean of p
    grouped_df = inhibition_df.groupby(['subject', 'condition', 'SSD'], as_index=False).agg(
        p=('p', 'mean'),
        count=('p', 'count')
    )

    return grouped_df


def filter_subjects(subject_list):
    dbs_conditions = defaultdict(set)
    keep = []

    # First pass: find DBS subjects and store their ON/OFF status
    for entry in subject_list:
        if "DBS" in entry:
            parts = entry.split()
            subj_id = parts[0]
            condition = parts[1]  # 'DBS'
            on_off = parts[2]     # 'ON' or 'OFF'
            dbs_conditions[subj_id].add(on_off)

    # Second pass: collect entries to keep
    for entry in subject_list:
        if entry.startswith("C") or entry.startswith("preop"):
            keep.append(entry)
        elif "DBS" in entry:
            parts = entry.split()
            subj_id = parts[0]
            on_off = parts[2]
            if {"ON", "OFF"}.issubset(dbs_conditions[subj_id]):
                keep.append(entry)

    return keep


def create_1_Hz_wide_bands(lowest, highest):
    """Generate a list of 1 Hz-wide frequency bands from lowest to highest."""
    return [(f, f + 1) for f in range(lowest, highest)]


def _convert_numpy(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, dict):
        return {key: _convert_numpy(value) for key, value in obj.items()}  # Recursively convert dictionary
    elif isinstance(obj, list):
        return [_convert_numpy(item) for item in obj]  # Convert lists containing ndarrays
    elif isinstance(obj, np.integer):  # Convert int64, int32, etc.
        return int(obj)
    elif isinstance(obj, np.floating):  # Convert float64, float32, etc.
        return float(obj)
    elif isinstance(obj, np.ndarray):  # Convert numpy arrays to lists
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")    
    


def _update_and_save_multiple_params(dictionary, session_ID, saving_path):
    """
    Update parameters dictionary, convert non-serializable objects, and save as JSON.

    Inputs:
        - dictionary: dict, contains multiple keys and their values
        - session_ID: str, the session identifier
        - saving_path: str, the path where to save/find the json file
    """
    parameters = _convert_numpy(dictionary)  # Ensure everything is JSON-serializable

    json_file_path = os.path.join(saving_path, f"{session_ID}.json")
    with open(json_file_path, "w") as json_file:
        json.dump(parameters, json_file, indent=4)

    print(f"JSON saved: {json_file_path}")



def update_or_create_json(json_path, subject_ID, new_key, new_value):
    """
    Finds or creates a JSON file for the given subject and condition,
    adds a new key-value pair, and saves it back.

    Parameters:
        json_path (str): Path to the folder containing the JSON files.
        subject_ID (str): Subject & session identifier (e.g., 'sub006 DBS OFF mSST').
        new_key (str): The key to add or update in the JSON file.
        new_value (any): The value associated with the new key.
    """
    # Ensure the folder exists
    os.makedirs(json_path, exist_ok=True)

    # Look for an existing JSON file
    for filename in os.listdir(json_path):
        if filename.endswith(".json") and subject_ID in filename:
            file_path = os.path.join(json_path, filename)
            
            # Load the existing JSON data
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            break
    else:
        # If no file was found, create a new JSON file
        filename = f"{subject_ID}.json"
        file_path = os.path.join(json_path, filename)
        data = {}  # Start with an empty dictionary
        print(f"Creating new JSON file: {filename}")

    # Add or update the key-value pair
    data[new_key] = new_value

    # Save the updated or new JSON file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"Updated {filename} with {new_key} = {new_value}")



def extract_stats(data):
    stats = {}
    for subject in data:
        df = data[subject]
        sub_dict = {}
        # return the index of the first row which is not filled by a Nan value:
        start_task_index = df['blocks.thisRepN'].first_valid_index()
        # Crop dataframe in 2 parts: before and after the task:
        df_maintask = df.iloc[start_task_index:-1]

        total_trials = 0
        # count each trial_type and total number:
        for trial_type in df_maintask['trial_type'].unique():
            total_trials += len(df_maintask[df_maintask['trial_type'] == trial_type])
            sub_dict[trial_type] = len(df_maintask[df_maintask['trial_type'] == trial_type])
            sub_dict['total_trials'] = total_trials

        # calculate how many trials of each type are correct based on the column 'key_resp_experiment.corr':
        correct_trials = 0
        for trial_type in df_maintask['trial_type'].unique():
            correct_trials += len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['key_resp_experiment.corr'] == 1) & (df_maintask['early_press_resp.corr'] == 0)])
            #print(subject, trial_type, 'correct :', len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['key_resp_experiment.corr'] == 1)]), '(', (len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['key_resp_experiment.corr'] == 1)])/len(df_maintask[df_maintask['trial_type'] == trial_type])*100), ')')
            sub_dict['percent correct ' + trial_type] = (len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['key_resp_experiment.corr'] == 1) & (df_maintask['early_press_resp.corr'] == 0)])/len(df_maintask[df_maintask['trial_type'] == trial_type])*100)
            sub_dict['correct ' + trial_type] = len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['key_resp_experiment.corr'] == 1) & (df_maintask['early_press_resp.corr'] == 0)])
            sub_dict['incorrect early ' + trial_type] = len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['early_press_resp.corr'] == 1)])
            sub_dict['incorrect wrong ' + trial_type] = len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['key_resp_experiment.corr'] == 0) & (df_maintask['early_press_resp.corr'] == 0)])
            sub_dict['total correct trials'] = correct_trials

        early_presses = len(df_maintask[df_maintask['early_press_resp.corr'] == 1])
        early_press_go = len(df_maintask[(df_maintask['early_press_resp.corr'] == 1) & (df_maintask['trial_type'] == 'go_trial')])
        if early_press_go > 0:
            early_press_go_rt = [0] * early_press_go
        else: early_press_go_rt = []
        early_presses_trials = df_maintask[df_maintask['early_press_resp.corr'] == 1].index
        sub_dict['early presses'] = early_presses
 
        # remove trials with early presses from the dataframe:
        df_maintask_copy = df_maintask.drop(early_presses_trials)

        # calculate the reaction time for each trial type:
        for trial_type in df_maintask_copy['trial_type'].unique():
            sub_dict[trial_type + ' RTs (ms)'] = (df_maintask_copy[(df_maintask_copy['trial_type'] == trial_type)]['key_resp_experiment.rt'].dropna() *1000).tolist()
            sub_dict[trial_type + ' mean RT (ms)'] = (df_maintask_copy[(df_maintask_copy['trial_type'] == trial_type)]['key_resp_experiment.rt'].mean())*1000
    
        # Extract mean SSD, and compute SSRT:
        go_rt = (df_maintask[df_maintask['trial_type'] == 'go_trial']['key_resp_experiment.rt'].dropna() *1000).tolist()
        total_go_trials = len(df_maintask[df_maintask['trial_type'] == 'go_trial'])
        early_and_correct_go = early_press_go + len(go_rt)
        no_response_go = total_go_trials - early_and_correct_go
        if no_response_go > 0:
            go_rt.extend([1000]*no_response_go)
        ordered_go_rt = np.sort(go_rt + early_press_go_rt)
        #ordered_go_rt = np.sort((df_maintask[df_maintask['trial_type'] == 'go_trial']['key_resp_experiment.rt'].dropna() *1000).tolist())
        percent_corr_stop = len(
            df_maintask[
                (df_maintask['trial_type'] == 'stop_trial') &
                  (df_maintask['key_resp_experiment.corr'] == 1) &
                    (df_maintask['early_press_resp.corr'] == 0)
                    ] 
                    ) / len(df_maintask[df_maintask['trial_type'] == 'stop_trial'])*100
        n = round((1 - (percent_corr_stop / 100)) * len(ordered_go_rt))
        nth_GO_RT = ordered_go_rt[n+1]
        stop_trials = df_maintask[df_maintask['trial_type'] == 'stop_trial']
        mean_ssd = (stop_trials['stop_signal_time'].mean())*1000
        sub_dict['mean SSD (ms)'] = mean_ssd
        ssrt_value = nth_GO_RT - mean_ssd   
        sub_dict['SSRT (ms)'] = ssrt_value

        # get the SSD values for all unsuccessful stop trials:
        u_stop_trials = df_maintask_copy[
            (df_maintask_copy['trial_type'] == 'stop_trial') &
            (df_maintask_copy['key_resp_experiment.corr'] == 0)
            ]
        ssd_u_stop_trials = (u_stop_trials['stop_signal_time']*1000).tolist()
        sub_dict['unsuccessful stop SSD (ms)'] = ssd_u_stop_trials


        # get the SSD values for all successful stop trials:
        s_stop_trials = df_maintask_copy[
            (df_maintask_copy['trial_type'] == 'stop_trial') &
            (df_maintask_copy['key_resp_experiment.corr'] == 1)
            ]
        ssd_s_stop_trials = (s_stop_trials['stop_signal_time']*1000).tolist()
        sub_dict['successful stop SSD (ms)'] = ssd_s_stop_trials

        # get the columns with trial_type :
        trials = df_maintask_copy['trial_type']
        # get all RT values : 
        rts = df_maintask_copy['key_resp_experiment.rt']
        sub_dict['trial IDs'] = trials.tolist()
        sub_dict['RTs (ms)'] = (rts * 1000).tolist()

        # also keep information about the block number:
        sub_dict['block number'] = df_maintask_copy['blocks.thisN'].tolist()
        
        stats[subject] = sub_dict
            
    return stats


def _get_onedrive_path(
    folder: str = 'onedrive', 
    sub: str = None
):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    Folder has to be in ['onedrive', 'DATA']
    """

    # from your cwd get the path and stop at 'Users'
    path = os.getcwd()

    while os.path.dirname(path)[-5:] != 'Users':
        path = os.path.dirname(path) # path is now leading to Users/username

    # get the onedrive folder containing "onedrive" and "charit" and add it to the path
    onedrive_f = [
        f for f in os.listdir(path) if np.logical_and(
            'onedrive' in f.lower(),
            'charit' in f.lower())
            ]

    path = os.path.join(path, onedrive_f[0]) # path is now leading to Onedrive folder


    # add the folder DATA to the path and from there open the folders depending on input folder
    path = os.path.join(path, 'DATA')

    return path


# Helper function for plotting spread
def stat_fun(x):
    """Return sum of squares."""
    return np.sum(x**2, axis=0)



def create_color_palette(
        included_subjects
):
    # associate a specific color to each subject present in included_subjects, but keep the same color if the first 6 characters of a subject is not unique:
    n=0
    included_subjects_colors = {}
    unique_sub = []
    for sub in included_subjects:
        if "C" in sub:
            if sub[:4] not in unique_sub:
                unique_sub.append(sub[:4])
        if "preop" in sub:
            if sub[:8] not in unique_sub:
                unique_sub.append(sub[:8])
        if "sub" in sub:
            if sub[:6] not in unique_sub:
                unique_sub.append(sub[:6])
        
    print(unique_sub)

    #colors = sns.color_palette("husl", len(unique_sub))
    #subject_palette = sns.color_palette("husl", len(unique_sub))  # Ensure distinct colors
    #subject_colors = dict(zip(unique_sub, subject_palette))

    subject_palette1 = sns.color_palette("husl", len(unique_sub))  # Original
    subject_palette2 = sns.husl_palette(len(unique_sub), s=0.9, l=0.5)  # Adjust s (saturation) and l (lightness)

    # Create a new palette by alternating colors
    interleaved_palette = [
        subject_palette1[i] if i % 2 == 0 else subject_palette2[i]
        for i in range(len(unique_sub))
    ]

    subject_colors = dict(zip(unique_sub, interleaved_palette))
    
    
    return subject_colors



def prepare_merged_dataframe(
    df_proactive_all,
    df_reactive_all,
    stats_OFF,
    stats_ON,
    stats_CONTROL,
    stats_PREOP,
    behav_results_saving_path    
):
    
    df_reactive_all['Subject'] = df_reactive_all['Subject'] + ' ' + df_reactive_all['Condition']
    df_reactive_all = df_reactive_all.drop(columns=['Condition'])
    # merge columns Subject and Condition in df_proactive:
    df_proactive_all['Subject'] = df_proactive_all['Subject'] + ' ' + df_proactive_all['Condition']
    df_proactive_all = df_proactive_all.drop(columns=['Condition'])

    # Define conditions and corresponding dictionaries
    conditions = {
        'control': stats_CONTROL,
        'DBS OFF': stats_OFF,
        'DBS ON': stats_ON,
        'preop': stats_PREOP
    }

    # Initialize dictionaries to hold results for each condition
    results = {condition: {} for condition in conditions.keys()}

    # Loop through each condition and subject
    for condition, data_dict in conditions.items():
        for subject_id, metrics in data_dict.items():
            # Extract the subject ID (first part of subject_id before the first space)
            sub_id = subject_id.split()[0]
            # Retrieve the required metrics and store them in the result dictionary
            SSD = metrics['mean SSD (ms)']
            results[condition][sub_id] = SSD

    # Prepare data for DataFrame
    data = []
    for condition, subject_dict in results.items():
        for subject_id, SSD in subject_dict.items():
            data.append({'Subject': subject_id, 'Condition': condition, 'mean SSD (ms)': SSD})

    # Create DataFrame
    df_SSD_all = pd.DataFrame(data)

    # same for df_SSD:
    df_SSD_all['Subject'] = df_SSD_all['Subject'] + ' ' + df_SSD_all['Condition']
    df_SSD_all = df_SSD_all.drop(columns=['Condition'])

    # merge the two dataframes:
    df_merged = pd.merge(df_proactive_all, df_reactive_all, on='Subject', how='outer')

    # remove lines containing NaNs:
    df_merged = df_merged.dropna()

    df_merged = pd.merge(df_merged, df_SSD_all, on='Subject', how='outer')

    # remove lines containing NaNs:
    df_merged = df_merged.dropna()

    df_merged.to_excel(join(behav_results_saving_path,'proactive_reactive_inhibition2.xlsx'), index=False)

    return df_merged

'''
import os
from os.path import join
from mne.io import read_raw

from functions.io import (
    find_EEG_stream,
    write_set
)


def convert_xdf_to_set(
    session_ID="sub008 DBS ON mSST",
    fname_external="sub-008_ses-mSST_ON_task-Default_run-001_eeg.xdf",
):

    """
    convert_xdf_to_set can be used to convert .xdf file to .set files, which
    are more suitable for EEG data analysis.

    Parameters
    ----------
    session_ID: string, name of the session to be analyzed

    fname_external: string, name of the external file to be analyzed (.xdf file)

    .................................................................................

    Results
    -------
    The resulting file will be saved in the results folder, in a sub-folder named after the 
    session_ID parameter. It will be a .set file, suitable for EEG data analysis.
    """
    working_path = os.getcwd()

    #  Set saving path
    results_path = join(working_path, "results")
    saving_path = join(results_path, session_ID)
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)

    #  Set source path
    source_path = join(working_path, "sourcedata")


    #  1. LOADING DATASETS AND EXTRACT CHANNEL CONTAINING ARTIFACTS:

        ##  External data from XDF
        # load external dataset into mne
    fpath_external = join(source_path, fname_external)  
    print(f"Loading external data from {fpath_external}")
    stream_id = find_EEG_stream(fpath_external, stream_name = 'SAGA')  # find the stream_id of the EEG data, which is called 'SAGA' in our setup
    TMSi_rec = read_raw(fpath_external, stream_ids = [stream_id], preload=True)
    print(f"External data loaded from {fpath_external}")
    external_title = ("SYNCHRONIZED_EXTERNAL_" + str(fname_external[:-4]) + ".set")
    fname_external_out=join(saving_path, external_title)

    write_set(fname_external_out, TMSi_rec, TMSi_rec.annotations.onset)
    print(f"External data saved as .set file in {fname_external_out}")
'''