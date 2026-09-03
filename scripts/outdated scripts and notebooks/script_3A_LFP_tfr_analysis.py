"""
Included subjects for group analysis (including subjects who had ecg artifacts,
cleaned by ReSync):
[
#'sub006 DBS ON mSST', 'sub006 DBS OFF mSST, # not well cleaned yet
#'sub007 DBS OFF mSST',
'sub008 DBS OFF mSST', 'sub008 DBS ON mSST', 
#'sub009 DBS OFF mSST', 'sub009 DBS ON mSST',
'sub011 DBS OFF mSST', 'sub011 DBS ON mSST',  
#'sub013 DBS OFF mSST', 
#'sub014 DBS ON mSST', 
'sub015 DBS OFF mSST', 'sub015 DBS ON mSST', 
#'sub017 DBS OFF mSST',
#'sub019 DBS OFF mSST', 'sub019 DBS ON mSST', # not well cleaned yet
#'sub020 DBS ON mSST', 
'sub021 DBS OFF mSST', 'sub021 DBS ON mSST', 
#'sub022 DBS ON mSST', 
'sub023 DBS OFF mSST', 'sub023 DBS ON mSST'
]



OLD COMMENTS:
[
    #'sub006 DBS OFF mSST', # I contains NaNs + (ECG artifacts) but seems like pressing with other hand... (desync ipsilateral)
    'sub006 DBS ON mSST',  # I data quality weird but ok?
    'sub007 DBS OFF mSST', # I data quality ok
    #'sub008 DBS OFF mSST', #  broadband weird rythmic pattern in low freq <30
    #'sub008 DBS ON mSST', # I data quality ok 
    #'sub009 DBS OFF mSST',  # I behavior too bad BUT DATA VERY GOOD
    #'sub009 DBS ON mSST', # I very good data
    #'sub011 DBS OFF mSST', # broadband weird rythmic pattern in low freq <30
    #'sub011 DBS ON mSST',  # broadband weird rythmic pattern in low freq <30, even stronger dbs on
    #'sub012 DBS ON mSST', # extremely noisy LFP and only Left STN
    #'sub013 DBS OFF mSST', # I data looks good but double beta peak which have opposite variations
    'sub014 DBS ON mSST', #  broadband weird rythmic pattern in low freq <30,
    #'sub015 DBS OFF mSST', # I data looks great
    #'sub015 DBS ON mSST', # I data looks great
    #'sub017 DBS ON mSST', #  broadband weird rythmic pattern in low freq <30
    'sub017 DBS OFF mSST', # I data quality ok. behavior too bad? + seems like pressing with other hand... (desync ipsilateral)
    #'sub019 DBS OFF mSST', # I data looks fine, but desync also seems stronger in ipsi
    #'sub019 DBS ON mSST',  # I contains NaNs and stim channel weird? + desync also seems stronger in ipsi
    'sub020 DBS ON mSST', # I data might be ok? behavior too bad + + desync also seems stronger in ipsi
    #'sub021 DBS OFF mSST', # I data quality ok, contains NaNs
    'sub021 DBS ON mSST', # I data quality ok
    #'sub022 DBS ON mSST', # I behavior ok except GC, lfp data bof
    #'sub023 DBS OFF mSST', # I behavior ok, lfp data fine (but reversed?)
    #'sub023 DBS ON mSST' # 
    ],

"""



from mne.io import read_raw
import numpy as np
from os.path import join
import os
import pandas as pd
from collections import defaultdict
import json


from functions import utils
from functions import ephy_plotting
from functions import preprocessing
from functions import analysis
from functions import io


def main_preprocess_LFP(
        included_subjects = 
        [
        'sub006 DBS ON mSST', 'sub006 DBS OFF mSST',
        #'sub007 DBS OFF mSST',
        #'sub008 DBS OFF mSST', 'sub008 DBS ON mSST', 
        #'sub009 DBS OFF mSST', 'sub009 DBS ON mSST',
        'sub011 DBS OFF mSST', 'sub011 DBS ON mSST',  
        #'sub013 DBS OFF mSST', 
        #'sub014 DBS ON mSST', 
        'sub015 DBS OFF mSST', 'sub015 DBS ON mSST', 
        #'sub017 DBS OFF mSST', 'sub017 DBS ON mSST',
        'sub019 DBS OFF mSST', 'sub019 DBS ON mSST', 
        #'sub020 DBS ON mSST', 
        #'sub021 DBS OFF mSST', 'sub021 DBS ON mSST', 
        #'sub022 DBS ON mSST', 
        'sub023 DBS OFF mSST', 'sub023 DBS ON mSST'
        #'sub024 DBS ON mSST'
        ],  
        INDIV_PLOTS = True,
        FULL_SESSION_PLOTS = True,
        GROUP_PLOTS = False,
        ON_VS_OFF_PLOTS = False
):

    onedrive_path = utils._get_onedrive_path()
    working_path = os.getcwd()
    #  Set saving paths
    results_path = join(working_path, "results")
    print(results_path)
    saving_path_group = join(results_path, 'group_level', 'lfp_perc_sig_change')  
    os.makedirs(saving_path_group, exist_ok=True)  # Create the directory if it doesn't exist
    saving_path_on_off = join(results_path, 'ON_vs_OFF')
    os.makedirs(saving_path_on_off, exist_ok=True)

    # Dictionary to store subject epochs in
    sub_dict_epochs_subsets = {}
    sub_dict_lm_GO = {}
    sub_dict_RT = {}
    sub_dict_stats = {}

    cluster_results_dict = {}
    cluster_results_dict = defaultdict(dict)  # Each missing key gets an empty dictionary
    cluster_results_dict['All subjects'] = included_subjects

    # Load all data for all included subjects
    data = io.load_behav_data(included_subjects, onedrive_path)

    # Compute statistics for each loaded subject
    stats = {}
    stats = utils.extract_stats(data)
    # If no file was found, create a new JSON file
    filename = "stats.json"
    file_path = os.path.join(results_path, filename)
    #if not os.path.isfile(file_path):
    #    with open(file_path, "w", encoding="utf-8") as file:
    #            json.dump({}, file, indent=4)

    # Save the updated or new JSON file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=4)

    # Start a loop through subjects
    for session_ID in included_subjects:
        print(f"Now processing {session_ID}")
        session_dict = {}
        sub = session_ID[:6]
        subject_ID = session_ID.split(' ') [0]
        condition = session_ID.split(' ') [1] + ' ' + session_ID.split(' ') [2]
        sub_onedrive_path = join(onedrive_path, subject_ID)
        sub_onedrive_path_task = join(onedrive_path, subject_ID, 'synced_data', 'no ecg cleaning','sub011 DBS ON mSST')
        filename = [f for f in os.listdir(sub_onedrive_path_task) if (
            f.endswith('.set') and f.startswith('SYNCHRONIZED_INTRACRANIAL'))]
        file = join(sub_onedrive_path_task, filename[0])
        raw = read_raw(file, preload=True)

        saving_path_single = join(results_path, 'single_sub', f'{sub} mSST','lfp_perc_sig_change','no ecg cleaning') 
        os.makedirs(saving_path_single, exist_ok=True)  # Create the directory if it doesn't exist

        if FULL_SESSION_PLOTS:
            ephy_plotting.plot_raw_stim(session_ID, raw, saving_path_single)
            psd_left, freqs_left, psd_right, freqs_right = analysis.compute_psd_welch(raw)
            session_dict['psd_left'] = psd_left
            session_dict['freqs_left'] = freqs_left
            session_dict['psd_right'] = psd_right
            session_dict['freqs_right'] = freqs_right
            ephy_plotting.plot_psd_log(
                session_ID, raw, freqs_left, psd_left, 
                freqs_right, psd_right, saving_path_single, is_filt=False
                )
            ephy_plotting.plot_stft_stim(
                session_ID, raw, saving_path=saving_path_single, is_filt=False, 
                vmin = -3, vmax = 1, fmin=0, fmax=100
                )

        session_dict['CHANNELS'] = raw.ch_names

        # Rename channels to be consistent across subjects:
        new_channel_names = [
            "Left_STN",
            "Right_STN",
            "left_peak_STN",
            "right_peak_STN",
            "STIM_Left_STN",
            "STIM_Right_STN"
        ]

        # Get the existing channel names
        old_channel_names = raw.ch_names

        # Create a mapping from old to new names
        rename_dict = {old: new for old, new in zip(old_channel_names, new_channel_names)}

        # Rename the channels
        raw.rename_channels(rename_dict)

        session_dict['RENAMED_CHANNELS'] = raw.ch_names

        # Filter & resample:
        filtered_data = raw.copy().filter(l_freq=1, h_freq=95)
        #filtered_data_resampled = filtered_data.copy().resample(sfreq=200) # DO NOT WORK WITH NaNs
        # DOWNSAMPLED DATA SHOULD NOT BE EPOCHED BECAUSE IT INTRODUCES JITTER IN THE EVENTS

        # Extract events and create epochs
        # only keep lfp channels
        filtered_data_lfp = filtered_data.copy().pick_channels([filtered_data.ch_names[0], filtered_data.ch_names[1]])

        epochs, filtered_event_dict = preprocessing.create_epochs(filtered_data_lfp, session_ID)

        mSST_raw_behav_session_data_path = join(
             onedrive_path, subject_ID, "raw_data", 'BEHAVIOR', condition, 'mSST'
             )
        for filename in os.listdir(mSST_raw_behav_session_data_path):
                if filename.endswith(".csv"):
                    fname = filename
        filepath_behav = join(mSST_raw_behav_session_data_path, fname)
        df = pd.read_csv(filepath_behav)

        # return the index of the first row which is not filled by a Nan value:
        start_task_index = df['blocks.thisRepN'].first_valid_index()
        # Crop dataframe in 2 parts: before and after the task:
        #df_training = df.iloc[:start_task_index]
        df_maintask = df.iloc[start_task_index:-1]

        # remove the trials with early presses, as in these trials the cues were not presented
        early_presses = df_maintask[df_maintask['early_press_resp.corr'] == 1]
        early_presses_trials = list(early_presses.index)
        number_early_presses = len(early_presses_trials)

        # remove trials with early presses from the dataframe:
        df_maintask_copy = df_maintask.drop(early_presses_trials)

        # Filter successful and unsuccessful trials:
        (epochs_subsets, epochs_lm, mean_RT_dict) = preprocessing.create_epochs_subsets_from_behav(
                df_maintask_copy, 
                epochs, 
                filtered_event_dict
                )

        sub_dict_epochs_subsets[session_ID] = epochs_subsets
        sub_dict_lm_GO[session_ID] = epochs_lm
        sub_dict_RT[session_ID] = mean_RT_dict
        sub_dict_stats[session_ID] = stats[session_ID]


    ######################
    ### TFR PARAMETERS ###
    ######################

    decim = 1 
    freqs = np.arange(1, 90, 1) 
    n_cycles = np.minimum(np.maximum(freqs / 2.0, 3), 20)
    tfr_args = dict(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        decim=decim,
        return_itc=False,
        average=False
    )        


    #############################################
    ### Plot power change for single subjects ###
    #############################################

    tmin_tmax = [-500, 1500]
    vmin_vmax = [-70, 70]

    sub_nums = []

    for sub in included_subjects:
        sub = sub[:6]
        if sub not in sub_nums:  # Check if sub is already in sub_nums
            sub_nums.append(sub)


    if INDIV_PLOTS:
        for sub in sub_nums:
            print(f"Now processing sub: {sub}")
            single_sub_dict_subsets = {key: value for key, value in sub_dict_epochs_subsets.items() if sub in key}
            print(single_sub_dict_subsets.keys())
            single_sub_dict_lm_GO = {key: value for key, value in sub_dict_lm_GO.items() if sub in key}
            single_sub_RT_dict = {key: value for key, value in sub_dict_RT.items() if sub in key}
            single_sub_stats_dict = {key: value for key, value in sub_dict_stats.items() if sub in key}
            saving_path_single = join(results_path, 'single_sub', f'{sub} mSST','lfp_perc_sig_change','no ecg cleaning')
            #saving_path_single = join(results_path, 'single_sub', f'{sub} mSST','lfp_perc_sig_change') 
            #os.makedirs(saving_path_single, exist_ok=True)  # Create the directory if it doesn't exist

            ## single condition            
            for dbs_status in ['DBS OFF', 'DBS ON']:
                if any(dbs_status in key for key in single_sub_dict_subsets.keys()):
                    for cond in [
                        'GO_successful', 
                        #'GO_unsuccessful', 
                        'GF_successful', 
                        #'GF_unsuccessful',
                        'GC_successful', 
                        #'GC_unsuccessful',
                        'GS_successful', 
                        'GS_unsuccessful',
                        'STOP_successful',
                        'STOP_unsuccessful',
                        'CONTINUE_successful'
                        ]:
                        ephy_plotting.tfr_pow_change_cond(
                            sub_dict = single_sub_dict_subsets, 
                            RT_dict = single_sub_RT_dict,   
                            stats_dict = single_sub_stats_dict, 
                            dbs_status = dbs_status, 
                            epoch_cond = cond, 
                            tfr_args = tfr_args, 
                            t_min_max = tmin_tmax, 
                            vmin_vmax = vmin_vmax,
                            baseline_correction=True,
                            saving_path=saving_path_single
                            )
                      
                    ephy_plotting.tfr_pow_change_cond(
                        single_sub_dict_lm_GO, 
                        single_sub_RT_dict,
                        single_sub_stats_dict,
                        dbs_status, 
                        "lm_GO", 
                        tfr_args, 
                        tmin_tmax,
                        vmin_vmax, 
                        baseline_correction=True,
                        saving_path=saving_path_single
                        )
                    
                    condition = f"{dbs_status} GS_successful - lm_GO {sub}"
                    cluster_results_dict = ephy_plotting.perc_pow_diff_cond2(single_sub_dict_subsets,
                                    single_sub_dict_lm_GO,
                                    single_sub_RT_dict,
                                    single_sub_stats_dict,
                                    dbs_status,  
                                    tfr_args, 
                                    tmin_tmax, 
                                    vmin_vmax,
                                    "GS_successful",
                                    "lm_GO",
                                    cluster_results_dict,
                                    condition,
                                    saving_path=saving_path_single)
                        
            ## difference conditions
                    condition = f"{dbs_status} GO_successful - GF_successful {sub}"
                    cluster_results_dict = ephy_plotting.perc_pow_diff_cond(single_sub_dict_subsets,  
                                    single_sub_RT_dict, 
                                    single_sub_stats_dict,
                                    dbs_status,  
                                    tfr_args, 
                                    tmin_tmax, 
                                    vmin_vmax,
                                    "GO_successful",
                                    "GF_successful",
                                    cluster_results_dict,
                                    condition,
                                    saving_path=saving_path_single)
                    condition = f"{dbs_status} GS_successful - GS_unsuccessful {sub}"
                    cluster_results_dict = ephy_plotting.perc_pow_diff_cond(single_sub_dict_subsets,  
                                    single_sub_RT_dict,
                                    single_sub_stats_dict,
                                    dbs_status,  
                                    tfr_args, 
                                    tmin_tmax, 
                                    vmin_vmax,
                                    "GS_successful",
                                    "GS_unsuccessful",
                                    cluster_results_dict,
                                    condition,
                                    saving_path=saving_path_single)
                    condition = f"{dbs_status} STOP_successful - STOP_unsuccessful {sub}"
                    cluster_results_dict = ephy_plotting.perc_pow_diff_cond(single_sub_dict_subsets,  
                                    single_sub_RT_dict,
                                    single_sub_stats_dict,
                                    dbs_status,  
                                    tfr_args, 
                                    tmin_tmax, 
                                    vmin_vmax,
                                    "STOP_successful",
                                    "STOP_unsuccessful",
                                    cluster_results_dict,
                                    condition,
                                    saving_path=saving_path_single,
                                    )    
                    condition = f"{dbs_status} STOP_successful - CONTINUE_successful {sub}"
                    cluster_results_dict = ephy_plotting.perc_pow_diff_cond(single_sub_dict_subsets,  
                                    single_sub_RT_dict,
                                    single_sub_stats_dict,
                                    dbs_status,  
                                    tfr_args, 
                                    tmin_tmax, 
                                    vmin_vmax,
                                    "STOP_successful",
                                    "CONTINUE_successful",
                                    cluster_results_dict,
                                    condition,
                                    saving_path=saving_path_single,
                    )

    ### group level ###
    if GROUP_PLOTS:
        for dbs_status in [
            #'DBS OFF', 
                           'DBS ON']:
            '''
            condition = f"{dbs_status} GS_successful - lm_GO"
            print(f"Now processing: {condition}")
            cluster_results_dict = ephy_plotting.perc_pow_diff_cond2(sub_dict_epochs_subsets,
                        sub_dict_lm_GO,
                        sub_dict_RT,
                        sub_dict_stats,
                        dbs_status,  
                        tfr_args, 
                        tmin_tmax, 
                        vmin_vmax,
                        "GS_successful",
                        "lm_GO",
                        cluster_results_dict,
                        condition = condition,
                        saving_path=saving_path_group
                        )
            '''
                     
            for cond in [
                #'GO_successful', 
                #'GO_unsuccessful', 
                #'GF_successful', 
                #'GF_unsuccessful',
                #'GC_successful', 
                #'GC_unsuccessful',
                'GS_successful', 
                'GS_unsuccessful',
                #'STOP_successful',
                #'STOP_unsuccessful',
                #'CONTINUE_successful'
                ]:
                print(f"Now processing: {dbs_status} - {cond} ")
                ephy_plotting.tfr_pow_change_cond(sub_dict_epochs_subsets, 
                            sub_dict_RT,
                            sub_dict_stats,
                            dbs_status, 
                            cond, 
                            tfr_args, 
                            tmin_tmax, 
                            vmin_vmax,
                            saving_path=saving_path_group)
            '''
            print(f"Now processing: {dbs_status} - lm_GO ")    
            ephy_plotting.tfr_pow_change_cond(sub_dict_lm_GO,
                            sub_dict_RT,
                            sub_dict_stats,
                            dbs_status, 
                            "lm_GO", 
                            tfr_args, 
                            tmin_tmax,
                            vmin_vmax, 
                            saving_path=saving_path_group)
            
            
        ## difference conditions
            condition = f"{dbs_status} GO - GF"
            print(f"Now processing: {condition}")
            cluster_results_dict = ephy_plotting.perc_pow_diff_cond(sub_dict_epochs_subsets, 
                        sub_dict_RT,
                        sub_dict_stats,
                        dbs_status,  
                        tfr_args, 
                        tmin_tmax, 
                        vmin_vmax,
                        "GO_successful",
                        "GF_successful",
                        cluster_results_dict,
                        condition,
                        saving_path=saving_path_group)
            '''
            condition = f"{dbs_status} GS successful - GS unsuccessful"
            print(f"Now processing: {condition}")
            cluster_results_dict = ephy_plotting.perc_pow_diff_cond(sub_dict_epochs_subsets, 
                        sub_dict_RT,
                        sub_dict_stats,
                        dbs_status,  
                        tfr_args, 
                        tmin_tmax, 
                        vmin_vmax,
                        "GS_successful",
                        "GS_unsuccessful",
                        cluster_results_dict,
                        condition,
                        saving_path=saving_path_group)
            '''
            condition = f"{dbs_status} STOP successful - STOP unsuccessful"
            print(f"Now processing: {condition}")
            cluster_results_dict = ephy_plotting.perc_pow_diff_cond(sub_dict_epochs_subsets,
                        sub_dict_RT,
                        sub_dict_stats,
                        dbs_status,  
                        tfr_args, 
                        tmin_tmax, 
                        vmin_vmax,
                        "STOP_successful",
                        "STOP_unsuccessful",
                        cluster_results_dict,
                        condition,
                        saving_path=saving_path_group
                        )
            
            condition = f"{dbs_status} STOP unsuccessful - STOP successful"
            print(f"Now processing: {condition}")
            cluster_results_dict = ephy_plotting.perc_pow_diff_cond(sub_dict_epochs_subsets,
                        sub_dict_RT,
                        sub_dict_stats,
                        dbs_status,  
                        tfr_args, 
                        tmin_tmax, 
                        vmin_vmax,
                        "STOP_unsuccessful",
                        "STOP_successful",
                        cluster_results_dict,
                        condition,
                        saving_path=saving_path_group
                        )
            

            condition = f"{dbs_status} STOP successful - CONTINUE successful"
            print(f"Now processing: {condition}")
            cluster_results_dict = ephy_plotting.perc_pow_diff_cond(sub_dict_epochs_subsets,
                        sub_dict_RT,
                        sub_dict_stats,
                        dbs_status,  
                        tfr_args, 
                        tmin_tmax, 
                        vmin_vmax,
                        "STOP_successful",
                        "CONTINUE_successful",
                        cluster_results_dict,
                        condition,
                        saving_path=saving_path_group
                        )
            
            condition = f"{dbs_status} GS successful - GC successful"
            print(f"Now processing: {condition}")
            cluster_results_dict = ephy_plotting.perc_pow_diff_cond(sub_dict_epochs_subsets,
                        sub_dict_RT,
                        sub_dict_stats,
                        dbs_status,  
                        tfr_args, 
                        tmin_tmax, 
                        vmin_vmax,
                        "GS_successful",
                        "GC_successful",
                        cluster_results_dict,
                        condition,
                        saving_path=saving_path_group
                        )
            

            condition = f"{dbs_status} STOP unsuccessful - CONTINUE successful"
            print(f"Now processing: {condition}")
            cluster_results_dict = ephy_plotting.perc_pow_diff_cond(sub_dict_epochs_subsets,
                        sub_dict_RT,
                        sub_dict_stats,
                        dbs_status,  
                        tfr_args, 
                        tmin_tmax, 
                        vmin_vmax,
                        "STOP_unsuccessful",
                        "CONTINUE_successful",
                        cluster_results_dict,
                        condition,
                        saving_path=saving_path_group
                        )
            
            condition = f"{dbs_status} GS unsuccessful - GC successful"
            print(f"Now processing: {condition}")
            cluster_results_dict = ephy_plotting.perc_pow_diff_cond(sub_dict_epochs_subsets,
                        sub_dict_RT,
                        sub_dict_stats,
                        dbs_status,  
                        tfr_args, 
                        tmin_tmax, 
                        vmin_vmax,
                        "GS_unsuccessful",
                        "GC_successful",
                        cluster_results_dict,
                        condition,
                        saving_path=saving_path_group
                        )            
            '''            

    if ON_VS_OFF_PLOTS:
        sub_both_cond = []

        sub_dict_subsets_ON_OFF = {}
        sub_dict_lm_GO_ON_OFF = {}
        sub_dict_RT_ON_OFF = {}
        sub_dict_stats_ON_OFF = {}

        for sub in sub_nums:
            if (sub + " DBS ON mSST" in included_subjects) and (sub + " DBS OFF mSST" in included_subjects):
                sub_both_cond.append(sub)
    
        # extract the data for the subjects that have both conditions
        for sub in sub_both_cond:
            single_sub_dict_subsets_ON_OFF = {key: value for key, value in sub_dict_epochs_subsets.items() if sub in key}
            sub_dict_subsets_ON_OFF[sub] = single_sub_dict_subsets_ON_OFF
            single_sub_dict_lm_GO_ON_OFF = {key: value for key, value in sub_dict_lm_GO.items() if sub in key}
            sub_dict_lm_GO_ON_OFF[sub] = single_sub_dict_lm_GO_ON_OFF
            single_sub_RT_dict_ON_OFF = {key: value for key, value in sub_dict_RT.items() if sub in key}
            sub_dict_RT_ON_OFF[sub] = single_sub_RT_dict_ON_OFF
            single_sub_stats_dict_ON_OFF = {key: value for key, value in sub_dict_stats.items() if sub in key}
            sub_dict_stats_ON_OFF[sub] = single_sub_stats_dict_ON_OFF
        
        # Collect epoch data for each condition
        for cond in [         
            'GO_successful', 
            #'GO_unsuccessful', 
            'GF_successful', 
            #'GF_unsuccessful',
            'GC_successful', 
            #'GC_unsuccessful',
            'GS_successful', 
            'GS_unsuccessful',
            'STOP_successful',
            'STOP_unsuccessful',
            'CONTINUE_successful'
            ]:
            print(f"Now processing: {cond}")
            condition = f"ON vs OFF - {cond}"
            cluster_results_dict = ephy_plotting.perc_pow_diff_on_off(
                sub_dict_ON_OFF = sub_dict_subsets_ON_OFF, 
                RT_dict_ON_OFF = sub_dict_RT_ON_OFF, 
                tfr_args = tfr_args, 
                cond = cond,
                t_min_max = tmin_tmax, 
                vmin_vmax = vmin_vmax,  
                cluster_results_dict = cluster_results_dict,
                condition = condition,
                saving_path = saving_path_on_off
            )

            
        print(f"Now processing: lm_GO")
        condition = f"ON vs OFF - lm_GO"
        cluster_results_dict = ephy_plotting.perc_pow_diff_on_off(
            sub_dict_ON_OFF = sub_dict_lm_GO_ON_OFF, 
            RT_dict_ON_OFF = sub_dict_RT_ON_OFF, 
            tfr_args = tfr_args, 
            cond = "lm_GO",
            t_min_max = tmin_tmax, 
            vmin_vmax = vmin_vmax,  
            cluster_results_dict = cluster_results_dict,
            condition = condition,
            saving_path = saving_path_on_off
        )

        print(f"Now processing contrast: GS_successful - lm_GO")
        condition = f"GS_successful - lm_GO"
        cluster_results_dict = ephy_plotting.perc_pow_diff_on_off_contrast(
            sub_dict_ON_OFF = sub_dict_subsets_ON_OFF,
            sub_dict_ON_OFF_cond2 = sub_dict_lm_GO_ON_OFF, 
            RT_dict_ON_OFF = sub_dict_RT_ON_OFF, 
            tfr_args = tfr_args, 
            cond1 = "GS_successful",
            cond2 = "lm_GO",
            t_min_max = tmin_tmax, 
            vmin_vmax = vmin_vmax,  
            cluster_results_dict = cluster_results_dict,
            condition = condition,
            saving_path = saving_path_on_off            
        )

        print(f"Now processing contrast: GS_successful - GS_unsuccessful")
        condition = f"GS_successful - GS_unsuccessful"
        cluster_results_dict = ephy_plotting.perc_pow_diff_on_off_contrast(
            sub_dict_ON_OFF = sub_dict_subsets_ON_OFF,
            sub_dict_ON_OFF_cond2 = sub_dict_subsets_ON_OFF, 
            RT_dict_ON_OFF = sub_dict_RT_ON_OFF, 
            tfr_args = tfr_args, 
            cond1 = "GS_successful",
            cond2 = "GS_unsuccessful",
            t_min_max = tmin_tmax, 
            vmin_vmax = vmin_vmax,  
            cluster_results_dict = cluster_results_dict,
            condition = condition,
            saving_path = saving_path_on_off            
        )

        print(f"Now processing contrast: GO_successful - GF_successful")
        condition = f"GO_successful - GF_successful"
        cluster_results_dict = ephy_plotting.perc_pow_diff_on_off_contrast(
            sub_dict_ON_OFF = sub_dict_subsets_ON_OFF,
            sub_dict_ON_OFF_cond2 = sub_dict_subsets_ON_OFF, 
            RT_dict_ON_OFF = sub_dict_RT_ON_OFF, 
            tfr_args = tfr_args, 
            cond1 = "GO_successful",
            cond2 = "GF_successful",
            t_min_max = tmin_tmax, 
            vmin_vmax = vmin_vmax,  
            cluster_results_dict = cluster_results_dict,
            condition = condition,
            saving_path = saving_path_on_off            
        )

        print(f"Now processing contrast: STOP_successful - STOP_unsuccessful")
        condition = f"STOP_successful - STOP_unsuccessful"
        cluster_results_dict = ephy_plotting.perc_pow_diff_on_off_contrast(
            sub_dict_ON_OFF = sub_dict_subsets_ON_OFF,
            sub_dict_ON_OFF_cond2 = sub_dict_subsets_ON_OFF, 
            RT_dict_ON_OFF = sub_dict_RT_ON_OFF, 
            tfr_args = tfr_args, 
            cond1 = "STOP_successful",
            cond2 = "STOP_unsuccessful",
            t_min_max = tmin_tmax, 
            vmin_vmax = vmin_vmax,  
            cluster_results_dict = cluster_results_dict,
            condition = condition,
            saving_path = saving_path_on_off            
        )

        print(f"Now processing contrast: STOP_unsuccessful - STOP_successful")
        condition = f"STOP_unsuccessful - STOP_successful"
        cluster_results_dict = ephy_plotting.perc_pow_diff_on_off_contrast(
            sub_dict_ON_OFF = sub_dict_subsets_ON_OFF,
            sub_dict_ON_OFF_cond2 = sub_dict_subsets_ON_OFF, 
            RT_dict_ON_OFF = sub_dict_RT_ON_OFF, 
            tfr_args = tfr_args, 
            cond1 = "STOP_unsuccessful",
            cond2 = "STOP_successful",
            t_min_max = tmin_tmax, 
            vmin_vmax = vmin_vmax,  
            cluster_results_dict = cluster_results_dict,
            condition = condition,
            saving_path = saving_path_on_off            
        )
        
        print(f"Now processing contrast: GS_unsuccessful - GC_successful")
        condition = f"GS_unsuccessful - GC_successful"
        cluster_results_dict = ephy_plotting.perc_pow_diff_on_off_contrast(
            sub_dict_ON_OFF = sub_dict_subsets_ON_OFF,
            sub_dict_ON_OFF_cond2 = sub_dict_subsets_ON_OFF, 
            RT_dict_ON_OFF = sub_dict_RT_ON_OFF, 
            tfr_args = tfr_args, 
            cond1 = "GS_unsuccessful",
            cond2 = "GC_successful",
            t_min_max = tmin_tmax, 
            vmin_vmax = vmin_vmax,  
            cluster_results_dict = cluster_results_dict,
            condition = condition,
            saving_path = saving_path_on_off            
        )

        print(f"Now processing contrast: STOP_unsuccessful - CONTINUE_successful")
        condition = f"STOP_unsuccessful - CONTINUE_successful"
        cluster_results_dict = ephy_plotting.perc_pow_diff_on_off_contrast(
            sub_dict_ON_OFF = sub_dict_subsets_ON_OFF,
            sub_dict_ON_OFF_cond2 = sub_dict_subsets_ON_OFF, 
            RT_dict_ON_OFF = sub_dict_RT_ON_OFF, 
            tfr_args = tfr_args, 
            cond1 = "STOP_unsuccessful",
            cond2 = "CONTINUE_successful",
            t_min_max = tmin_tmax, 
            vmin_vmax = vmin_vmax,  
            cluster_results_dict = cluster_results_dict,
            condition = condition,
            saving_path = saving_path_on_off            
        )
        


        print(f"Now processing contrast: GS_successful - GC_successful")
        condition = f"GS_successful - GC_successful"
        cluster_results_dict = ephy_plotting.perc_pow_diff_on_off_contrast(
            sub_dict_ON_OFF = sub_dict_subsets_ON_OFF,
            sub_dict_ON_OFF_cond2 = sub_dict_subsets_ON_OFF, 
            RT_dict_ON_OFF = sub_dict_RT_ON_OFF, 
            tfr_args = tfr_args, 
            cond1 = "GS_successful",
            cond2 = "GC_successful",
            t_min_max = tmin_tmax, 
            vmin_vmax = vmin_vmax,  
            cluster_results_dict = cluster_results_dict,
            condition = condition,
            saving_path = saving_path_on_off            
        )

        print(f"Now processing contrast: STOP_successful - CONTINUE_successful")
        condition = f"STOP_successful - CONTINUE_successful"
        cluster_results_dict = ephy_plotting.perc_pow_diff_on_off_contrast(
            sub_dict_ON_OFF = sub_dict_subsets_ON_OFF,
            sub_dict_ON_OFF_cond2 = sub_dict_subsets_ON_OFF, 
            RT_dict_ON_OFF = sub_dict_RT_ON_OFF, 
            tfr_args = tfr_args, 
            cond1 = "STOP_successful",
            cond2 = "CONTINUE_successful",
            t_min_max = tmin_tmax, 
            vmin_vmax = vmin_vmax,  
            cluster_results_dict = cluster_results_dict,
            condition = condition,
            saving_path = saving_path_on_off            
        )


    print(cluster_results_dict)
    #_update_and_save_multiple_params(cluster_results_dict, "All subjects", results_path)

if "__main__":
    main_preprocess_LFP()