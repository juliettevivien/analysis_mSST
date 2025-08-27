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
from scipy.stats import wilcoxon, ttest_rel



def compute_band_metrics(psd, freqs, bands=None):
    """
    Compute band-limited power (µV²) and RMS amplitude (µV) from PSD.
    
    Parameters
    ----------
    psd : ndarray, shape (..., n_freqs)
        Power spectral density in V²/Hz.
    freqs : ndarray, shape (n_freqs,)
        Frequencies corresponding to the PSD.
    bands : dict or None
        Dictionary of band names and their frequency ranges in Hz.
        Example: {'theta': (2,7), 'alpha': (8,12)}
        
    Returns
    -------
    results : dict
        Dictionary with structure:
        {
          'theta': {'power_uV2': ..., 'rms_uV': ...},
          'alpha': {'power_uV2': ..., 'rms_uV': ...},
          ...
        }
    """
    if bands is None:
        bands = {
            'theta': (2, 7),
            'alpha': (8, 12),
            'low-beta': (13, 19),
            'high-beta': (20, 35)
        }

    results = {}
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            results[name] = {'power_uV2': np.nan, 'rms_uV': np.nan}
            continue
        
        # Integrate PSD over the band (trapezoidal integration)
        band_power_V2 = np.trapz(psd[..., mask], freqs[mask], axis=-1)
        
        # Convert to µV² and µV
        band_power_uV2 = band_power_V2 * 1e12
        band_rms_uV = np.sqrt(band_power_uV2)
        
        results[name] = {'power_uV2': band_power_uV2, 'rms_uV': band_rms_uV}

    return results



def add_significance_bars(ax, comparisons, positions_base, conditions, condition_colors, box_width):
    """Add significance bars and p-values to the plot"""
    
    # Group comparisons by condition
    condition_comparisons = {}
    for comp in comparisons:
        condition = comp['condition']
        if condition not in condition_comparisons:
            condition_comparisons[condition] = []
        condition_comparisons[condition].append(comp)
    
    # Add significance bars for each condition
    y_offset_base = ax.get_ylim()[1] * 0.08  # Start bars 8% above the highest point
    
    for cond_idx, condition in enumerate(conditions):
        if condition not in condition_comparisons:
            continue
            
        comps = condition_comparisons[condition]
        color = condition_colors[condition]
        
        # Sort comparisons by block distance (closer blocks first) and filter significant ones
        significant_comps = [comp for comp in comps if comp['p_value'] < 0.05]
        significant_comps.sort(key=lambda x: abs(x['block2'] - x['block1']))
        
        for comp_idx, comp in enumerate(significant_comps):
            block1, block2 = comp['block1'], comp['block2']
            
            # Calculate x positions for the bars
            x1 = positions_base[block1] + (cond_idx - len(conditions)/2 + 0.5) * box_width
            x2 = positions_base[block2] + (cond_idx - len(conditions)/2 + 0.5) * box_width
            
            # Calculate y position for this comparison bar - stack them based on block distance
            block_distance = abs(block2 - block1)
            y_pos = ax.get_ylim()[1] + y_offset_base * (1 + block_distance * 0.7) + (comp_idx * y_offset_base * 0.3)
            
            # Draw the significance bar
            ax.plot([x1, x2], [y_pos, y_pos], color=color, linewidth=1.5, alpha=0.8)
            ax.plot([x1, x1], [y_pos - y_offset_base*0.1, y_pos + y_offset_base*0.1], color=color, linewidth=1.5, alpha=0.8)
            ax.plot([x2, x2], [y_pos - y_offset_base*0.1, y_pos + y_offset_base*0.1], color=color, linewidth=1.5, alpha=0.8)
            
            # Add p-value text
            p_text = f"p={comp['p_value']:.3f}" if comp['p_value'] >= 0.001 else "p<0.001"
            ax.text((x1 + x2) / 2, y_pos + y_offset_base*0.15, p_text, 
                   ha='center', va='bottom', fontsize=7, color=color, weight='bold')




def perform_block_comparisons(df_plot, condition):
    """Perform pairwise comparisons between all blocks within a condition"""
    comparisons = []
    condition_data = df_plot[df_plot['Condition'] == condition]
    
    # Get unique participants for this condition
    participants = condition_data['Participant'].unique()
    
    # Compare all blocks with all other blocks (0 vs 1, 0 vs 2, 0 vs 3, 1 vs 2, 1 vs 3, 2 vs 3)
    all_block_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for block1, block2 in all_block_pairs:
        # Get data for both blocks
        block1_data = []
        block2_data = []
        
        for participant in participants:
            participant_data = condition_data[condition_data['Participant'] == participant]
            
            block1_values = participant_data[participant_data['Block_num'] == block1]['Preparation_Cost'].values
            block2_values = participant_data[participant_data['Block_num'] == block2]['Preparation_Cost'].values
            
            if len(block1_values) > 0 and len(block2_values) > 0:
                block1_data.append(block1_values[0])
                block2_data.append(block2_values[0])
        
        # Perform statistical test if we have paired data
        if len(block1_data) >= 2 and len(block2_data) >= 2 and len(block1_data) == len(block2_data):
            try:
                # Use Wilcoxon signed-rank test for paired data
                statistic, p_value = wilcoxon(block1_data, block2_data)
                
                comparisons.append({
                    'condition': condition,
                    'comparison': f'Block {block1} vs Block {block2}',
                    'block1': block1,
                    'block2': block2,
                    'n_pairs': len(block1_data),
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            except:
                # If Wilcoxon fails, use paired t-test
                try:
                    statistic, p_value = ttest_rel(block1_data, block2_data)
                    comparisons.append({
                        'condition': condition,
                        'comparison': f'Block {block1} vs Block {block2}',
                        'block1': block1,
                        'block2': block2,
                        'n_pairs': len(block1_data),
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
                except:
                    pass
    
    return comparisons


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
            sub_dict['percent correct all trials even early ' + trial_type] = (len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['key_resp_experiment.corr'] == 1) & (df_maintask['early_press_resp.corr'] == 0)])/len(df_maintask[df_maintask['trial_type'] == trial_type])*100)
            sub_dict['correct ' + trial_type] = len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['key_resp_experiment.corr'] == 1) & (df_maintask['early_press_resp.corr'] == 0)])
            sub_dict['incorrect early ' + trial_type] = len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['early_press_resp.corr'] == 1)])
            sub_dict['incorrect wrong ' + trial_type] = len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['key_resp_experiment.corr'] == 0) & (df_maintask['early_press_resp.corr'] == 0)])
            sub_dict['total correct trials'] = correct_trials
            sub_dict['percent correct ' + trial_type] = (len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['key_resp_experiment.corr'] == 1) & (df_maintask['early_press_resp.corr'] == 0)])/len(df_maintask[(df_maintask['trial_type'] == trial_type) & (df_maintask['early_press_resp.corr'] == 0)])*100)

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

        # calculate preparation cost mean GO RT - mean GF RT for each block:
        all_prep_cost = []
        for block in df_maintask_copy['blocks.thisN'].unique():
            block_df = df_maintask_copy[df_maintask_copy['blocks.thisN'] == block]
            go_RTs = (block_df[block_df['trial_type'] == 'go_trial']['key_resp_experiment.rt'].dropna() * 1000).tolist()
            gf_RTs = (block_df[block_df['trial_type'] == 'go_fast_trial']['key_resp_experiment.rt'].dropna() * 1000).tolist()
            if go_RTs and gf_RTs:
                mean_go_RT = np.nanmean(go_RTs)
                mean_gf_RT = np.nanmean(gf_RTs)
                prep_cost = mean_go_RT - mean_gf_RT
            else:
                prep_cost = None
            all_prep_cost.append(prep_cost)
        sub_dict['all preparation costs (ms)'] = all_prep_cost
        
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