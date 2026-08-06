#import mne
from mne.io import read_raw
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
from os.path import join
import os
import pandas as pd
#from scipy.signal import hilbert

from functions import utils
from functions import io
from functions import preprocessing
from functions import freq_response

"""
Included subjects for analyzing gamma-entrainment amplitude response, with entrainment at 62.5Hz:
['sub006 DBS ON mSST', 'sub009 DBS ON mSST', 'sub015 DBS ON mSST', 'sub021 DBS ON mSST', 'sub023 DBS ON mSST']


BEFORE ECG CLEANING: Included subjects for beta-band amplitude response:
[
'sub007 DBS OFF mSST', 
'sub009 DBS OFF mSST', 'sub009 DBS ON mSST', 
'sub013 DBS OFF mSST', 
'sub014 DBS ON mSST', 
'sub015 DBS OFF mSST', 'sub015 DBS ON mSST', 
'sub019 DBS OFF mSST', 
'sub022 DBS ON mSST', 
'sub023 DBS OFF mSST', 'sub023 DBS ON mSST'
]

WITH ECG CLEANING: Included subjects for beta-band amplitude response:
        [
            'sub006 DBS ON mSST',
            'sub007 DBS OFF mSST',
            'sub008 DBS ON mSST', 
            'sub009 DBS OFF mSST', 'sub009 DBS ON mSST',
            'sub011 DBS OFF mSST', 'sub011 DBS ON mSST',  
            'sub013 DBS OFF mSST', 
            'sub014 DBS ON mSST', 
            'sub015 DBS OFF mSST', 'sub015 DBS ON mSST', 
            'sub017 DBS OFF mSST',
            'sub019 DBS OFF mSST', 
            'sub020 DBS ON mSST', 
            'sub021 DBS ON mSST', 
            'sub022 DBS ON mSST', 
            'sub023 DBS OFF mSST', 'sub023 DBS ON mSST'
            ]

All subjects with notes:
[
            #'sub006 DBS OFF mSST', # contains NaNs + (ECG artifacts) but seems like pressing with other hand... (desync ipsilateral)
            'sub006 DBS ON mSST',  # 
            #'sub007 DBS OFF mSST', # data quality ok
            #'sub008 DBS OFF mSST', #  broadband weird rythmic pattern in low freq <30
            #'sub008 DBS ON mSST', # data quality ok
            #'sub009 DBS OFF mSST',  # behavior too bad BUT DATA VERY GOOD
            'sub009 DBS ON mSST', # very good data
            #'sub011 DBS OFF mSST', # broadband weird rythmic pattern in low freq <30
            #'sub011 DBS ON mSST',  # broadband weird rythmic pattern in low freq <30, even stronger dbs on
            #'sub012 DBS ON mSST', # extremely noisy LFP and only Left STN
            #'sub013 DBS OFF mSST', # data looks good but double beta peak which have opposite variations
            #'sub014 DBS ON mSST', # data quality ok
            #'sub015 DBS OFF mSST', # data looks great
            'sub015 DBS ON mSST', # data looks great
            # 'sub016 DBS OFF mSST', # tremor dominant, too challenging to analyze behavior
            #'sub017 DBS ON mSST', #  broadband weird rythmic pattern in low freq <30
            #'sub017 DBS OFF mSST', # behavior too bad? + seems like pressing with other hand... (desync ipsilateral)
            'sub019 DBS OFF mSST', # data looks fine, but desync also seems stronger in ipsi
            #'sub019 DBS ON mSST',  # contains NaNs and stim channel weird? + desync also seems stronger in ipsi
            #'sub020 DBS ON mSST', # behavior too bad + + desync also seems stronger in ipsi
            #'sub021 DBS OFF mSST', # data quality ok
            'sub021 DBS ON mSST',
            #'sub022 DBS ON mSST', # behavior ok except GC, lfp data bof
            #'sub023 DBS OFF mSST', # behavior ok, lfp data fine (but reversed?)
            'sub023 DBS ON mSST'
            ]

"""

def main_freq_response_LFP(
        included_subjects = [
            'sub006 DBS ON mSST',
            'sub006 DBS OFF mSST',
            #'sub007 DBS OFF mSST',
            #'sub008 DBS ON mSST', 
            #'sub009 DBS OFF mSST', 
            #'sub009 DBS ON mSST',
            'sub011 DBS OFF mSST', 
            'sub011 DBS ON mSST',  
            #'sub013 DBS OFF mSST', 
            #'sub014 DBS ON mSST', 
            'sub015 DBS OFF mSST', 
            'sub015 DBS ON mSST', 
            #'sub017 DBS OFF mSST',
            'sub019 DBS OFF mSST', 
            'sub019 DBS ON mSST',
            #'sub020 DBS ON mSST', 
            #'sub021 DBS ON mSST', 
            #'sub022 DBS ON mSST', 
            #'sub023 DBS OFF mSST', 
            #'sub023 DBS ON mSST'
            ],
        INDIV_PLOTS = True,
        GROUP_PLOTS = True,
        ON_VS_OFF_PLOTS = False
):

    onedrive_path = utils._get_onedrive_path()
    working_path = os.getcwd()
    #  Set saving paths
    results_path = join(working_path, "results")
    print(results_path)
    saving_path_group = join(results_path, 'group_level', 'freq_response')
    #saving_path_group = join(results_path, 'group_level', 'poster') 
    if not os.path.isdir(saving_path_group):  
        os.makedirs(saving_path_group, exist_ok=True)  # Create the directory if it doesn't exist

    # Dictionary to store subject epochs in
    sub_dict_freq_response = {}
    sub_dict_RT = {}
    sub_dict_stats = {}

    # Load all data for all included subjects
    data = io.load_behav_data(included_subjects, onedrive_path)

    # Compute statistics for each loaded subject
    stats = {}
    stats = utils.extract_stats(data)



    # Start a loop through subjects
    for session_ID in included_subjects:
        print(f"Now processing {session_ID}")
        session_dict = {}
        freq_responses = {}
        sub = session_ID[:6]
        subject_ID = session_ID.split(' ') [0]
        condition = session_ID.split(' ') [1] + ' ' + session_ID.split(' ') [2]
        sub_onedrive_path = join(onedrive_path, subject_ID)
        sub_onedrive_path_task = join(onedrive_path, subject_ID, 'synced_data', session_ID)
        filename = [f for f in os.listdir(sub_onedrive_path_task) if (
            f.endswith('.set') and f.startswith('SYNCHRONIZED_INTRACRANIAL'))]
        file = join(sub_onedrive_path_task, filename[0])
        raw = read_raw(file, preload=True)

        saving_path_single = join(results_path, 'single_sub', f'{sub} mSST','freq_response') 
        os.makedirs(saving_path_single, exist_ok=True)  # Create the directory if it doesn't exist


        #  Set saving paths
        #results_path = join(working_path, "results")
        #saving_path = join(results_path, session_ID)
        #if not os.path.isdir(saving_path):
        #    os.makedirs(saving_path)
        #epochs_saving_path = join(results_path, "epochs")
        #if not os.path.isdir(epochs_saving_path):
        #    os.makedirs(epochs_saving_path)
        #json_saving_path = join(results_path, "JSON")
        #if not os.path.isdir(json_saving_path):
        #    os.makedirs(json_saving_path)

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
        #filtered_data = raw.copy().filter(l_freq=1, h_freq=95)

        #resampled_data = filtered_data.copy().resample(sfreq=200)

        #resampled_data = raw.copy().resample(sfreq=200)

        #filtered_data = resampled_data.copy().filter(l_freq=1, h_freq=95)

        # Extract events and create epochs
        # only keep lfp channels
        #filtered_data_lfp = filtered_data.copy().pick_channels([filtered_data.ch_names[0], filtered_data.ch_names[1]])

        #epochs, filtered_event_dict = create_epochs(filtered_data_lfp, session_ID)

        # save raw epochs
        #epochs_file = save_epochs(
        #    epochs, session_ID, 'unfiltered_epoch-epo', epochs_saving_path
        #    )
        #session_dict['unfiltered_epochs_file'] = epochs_file

        # Load behavior file
        #raw_data_path = join(sub_onedrive_path, "raw_data")
        #raw_behav_data_path = join(raw_data_path, 'BEHAVIOR')
        #raw_behav_session_data_path = join(raw_behav_data_path, condition)
        
        mSST_raw_behav_session_data_path = join(
             sub_onedrive_path, "raw_data", 'BEHAVIOR', condition, 'mSST'
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
        #number_early_presses = len(early_presses_trials)

        # remove trials with early presses from the dataframe:
        df_maintask_copy = df_maintask.drop(early_presses_trials)

        # Filter successful and unsuccessful trials:
        #(epochs_subsets, epochs_lm, mean_RT_dict) = create_epochs_subsets_from_behav(
        #        df_maintask_copy, 
        #        epochs, 
        #        filtered_event_dict
        #        )

        #epochs_subsets_file = save_epochs(
        #     epochs_subsets, session_ID, 'epochs_subsets-epo', epochs_saving_path
        #     )
        #epochs_lm_file = save_epochs(
        #     epochs_lm, session_ID, 'epochs_lm-epo', epochs_saving_path
        #     )
        #session_dict["epochs_subsets_file"] = epochs_subsets_file
        #session_dict["epochs_lm_file"] = epochs_lm_file
        #session_dict['mean_RT_dict'] = mean_RT_dict        


        # timecourse of beta activity
        """
        session_dict = create_epochs_specific_freq_band(
                resampled_data,
                "beta",
                session_ID,
                df_maintask_copy,
                epochs_saving_path,
                session_dict
        )
        """
        
        bands_of_interest = [
            'delta', 'theta', 'alpha', 
            #'beta', 
            #'custom',
            #'low-beta'
             #'high-beta', 
            #'gamma', 'low-gamma', 'medium-gamma', 'high-gamma'
            ]
        # timecourse in all different frequency bands
        #baseline = (-0.5, 0)
        for freq_band in bands_of_interest:
            freq_responses = preprocessing.create_epochs_specific_freq_band(
                raw.copy(),
                freq_band, # either a name for common bands, or a tuple for a custom band (i.e. (57, 69))
                session_ID,
                df_maintask_copy,
                #epochs_saving_path,
                freq_responses,
                #baseline,
            )

        sub_dict_freq_response[session_ID] = freq_responses
        #print(sub_dict_freq_response[session_ID].keys())
        #sub_dict_RT[session_ID] = mean_RT_dict
        sub_dict_stats[session_ID] = stats[session_ID]

        #_update_and_save_multiple_params(
        #        dictionary = session_dict,
        #        session_ID = session_ID, 
        #        saving_path = json_saving_path
        #        )            

    sub_nums = []

    for sub in included_subjects:
        sub = sub[:6]
        if sub not in sub_nums:  # Check if sub is already in sub_nums
            sub_nums.append(sub)

    if INDIV_PLOTS:
        for sub in sub_nums:
            print(f"Now processing sub: {sub}")
            single_sub_dict_freq_response = {key: value for key, value in sub_dict_freq_response.items() if sub in key}
            print(single_sub_dict_freq_response.keys())
            #single_sub_dict_lm_GO = {key: value for key, value in sub_dict_lm_GO.items() if sub in key}
            single_sub_RT_dict = {key: value for key, value in sub_dict_RT.items() if sub in key}
            single_sub_stats_dict = {key: value for key, value in sub_dict_stats.items() if sub in key}
            saving_path_single = join(results_path, 'single_sub', f'{sub} mSST','freq_response')
        
            os.makedirs(saving_path_single, exist_ok=True)  # Create the directory if it doesn't exist

            #print(single_sub_dict_freq_response.keys())

    if GROUP_PLOTS:
        """
        for dbs_status in ['DBS OFF', 'DBS ON']:
            for freq_band in bands_of_interest:
                condition = f"{dbs_status}_{freq_band}_GS successful - lm_GO"
                print(f"Now processing: {condition}") 
                #print("Now processing: GS_successful - lm_GO")
                freq_response.compare_freq_response_2_cond(
                    cond1_label = "GS_successful",
                    cond2_label = "lm_GO",
                    freq_prefix = freq_band,
                    dbs_status = dbs_status,
                    behav_keys = ["mean SSD (ms)", "SSRT (ms)"],
                    sub_dict_freq_response = sub_dict_freq_response, 
                    sub_dict_stats = sub_dict_stats,
                    saving_path = saving_path_group
                )
        """
        for dbs_status in ['DBS ON', 
                           'DBS OFF'
                           ]:
            for freq_band in bands_of_interest:
                condition = f"{dbs_status} - {freq_band} - GS_unsuccessful - GS_successful"
                print(f"Now processing: {condition}") 
                #print("Now processing: GS_successful - lm_GO")
                freq_response.compare_freq_response_2_cond(
                    cond1_label = "GS_unsuccessful",
                    cond2_label = "GS_successful",
                    freq_prefix = freq_band,
                    dbs_status = dbs_status,
                    behav_keys = ["mean SSD (ms)", "SSRT (ms)"],
                    sub_dict_freq_response = sub_dict_freq_response,
                    sub_dict_stats = sub_dict_stats,
                    saving_path = saving_path_group
                )
                '''
                condition = f"{dbs_status} - {freq_band} - GS_successful - lm_GO"
                print(f"Now processing: {condition}") 
                #print("Now processing: GS_successful - lm_GO")
                freq_response.compare_freq_response_2_cond(
                    cond1_label = "GS_successful",
                    cond2_label = "lm_GO",
                    freq_prefix = freq_band,
                    dbs_status = dbs_status,
                    behav_keys = ["mean SSD (ms)", "SSRT (ms)"],
                    sub_dict_freq_response = sub_dict_freq_response,
                    sub_dict_stats = sub_dict_stats,
                    saving_path = saving_path_group
                )
                '''
       


    if ON_VS_OFF_PLOTS:
        saving_path_on_off = join(results_path, 'ON_vs_OFF')
        os.makedirs(saving_path_on_off, exist_ok=True)
        sub_both_cond = []

        sub_dict_freq_response_ON_OFF = {}
        sub_dict_stats_ON_OFF = {}

        for sub in sub_nums:
            if (sub + " DBS ON mSST" in included_subjects) and (sub + " DBS OFF mSST" in included_subjects):
                sub_both_cond.append(sub)
    
        # extract the data for the subjects that have both conditions
        for sub in sub_both_cond:
            single_sub_dict_freq_response_ON_OFF = {key: value for key, value in sub_dict_freq_response.items() if sub in key}
            sub_dict_freq_response_ON_OFF[sub] = single_sub_dict_freq_response_ON_OFF
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
            'GS_unsuccessful'
            ]:
            print(f"Now processing: {cond}")
            condition = f"ON vs OFF - {cond}"
            freq_response.compare_freq_response_on_off(
                cond_label = cond,
                freq_prefix = 'beta',
                sub_dict_freq_response = sub_dict_freq_response_ON_OFF,
                sub_dict_stats = sub_dict_stats_ON_OFF,
                saving_path = saving_path_on_off
            )




        ############################  FIGURES  ##############################
        """
        if FIGURES_FULL:
            # Plot the raw full session
            plot_raw_stim(session_ID, raw, saving_path)

            psd_left, freqs_left, psd_right, freqs_right = compute_psd_welch(raw)
            session_dict['psd_left'] = psd_left
            session_dict['freqs_left'] = freqs_left
            session_dict['psd_right'] = psd_right
            session_dict['freqs_right'] = freqs_right

            plot_psd_log(
                session_ID, raw, freqs_left, psd_left, 
                freqs_right, psd_right, saving_path, is_filt=False
                )

            plot_stft_stim(
                session_ID, raw, saving_path=saving_path, is_filt=False, 
                vmin = -3, vmax = 3, fmin=0, fmax=95
                )

            # Plot the resampled/filtered full session
            (
                psd_left_filt, freqs_left_filt, psd_right_filt, freqs_right_filt
                ) = compute_psd_welch(filtered_data)
            session_dict['psd_left_filt'] = psd_left_filt
            session_dict['freqs_left_filt'] = freqs_left_filt
            session_dict['psd_right_filt'] = psd_right_filt
            session_dict['freqs_right_filt'] = freqs_right_filt
            
            plot_psd_log(
                session_ID, filtered_data, freqs_left_filt, psd_left_filt, 
                freqs_right_filt, psd_right_filt, saving_path, is_filt=True
                )
            
            plot_stft_stim(
                session_ID, filtered_data, saving_path=saving_path, 
                is_filt=True, vmin = -3, vmax=3, fmin=1, fmax=95)
            

        if FIGURES_EPOCHS:
            baseline = (-0.5, -0.1) # use the same baseline for all figures

            # plot the mean signal per channel and trial type:
            freqs = np.arange(1, 90, 1)  # define frequencies of interest
            n_cycles = np.clip(freqs / 2, 3, 20)  # Ensure n_cycles is between 3 and 20
            tfr_kwargs = dict(
                method="morlet",
                freqs=freqs,
                n_cycles=n_cycles,
                decim=2,
                return_itc=False,
                average=False,
            )
            my_keys = ['GO_successful', 'GO_unsuccessful', 'GC_successful', 'GC_unsuccessful', 'GF_successful', 'GF_unsuccessful','GS_successful', 'GS_unsuccessful']

            available_keys = []
            for key in my_keys:
                if len(epochs_subsets[key]) > 0:
                    available_keys.append(key)

            plot_av_freq_power_by_trial(
                session_ID=session_ID,
                epochs_subsets=epochs_subsets,
                raw = raw,
                available_keys=available_keys,
                tfr_kwargs=tfr_kwargs,
                saving_path=saving_path,
                mean_RT_dict=mean_RT_dict,
                vmin = -100,
                vmax = 100,
                apply_baseline = True,
                baseline=baseline
            )

            
            # plot the mean contrast between success/unsuccess per channel and 
            # trial type, and performs a permutation cluster test to test for
            # significance
            plot_tfr_success_vs_unsuccess(
                epochs_subsets=epochs_subsets,
                session_ID=session_ID,
                filtered_data_lfp=filtered_data_lfp,
                mean_RT_dict=mean_RT_dict,
                saving_path = saving_path,
                vmin=-100,
                vmax=100,
                fmax = 90   
                )

            # plot comparison between different trial types to see the differences
            # between proactive / reactive inhibition processes
            plot_power_comparison_between_conditions(
                epochs_subsets=epochs_subsets,
                epochs_lm=epochs_lm,
                session_ID=session_ID,
                filtered_data_lfp=filtered_data_lfp,
                mean_RT_dict=mean_RT_dict,
                saving_path=saving_path,
                vmin=-100,
                vmax=100,
                fmax= 90    
                )
        

        if FIGURES_FREQS:
            for freq_band in [
                    'delta', 'theta', 'alpha', 'beta', 'low-beta', 'high-beta', 'gamma'
                    'low-gamma', 'medium-gamma', 'high-gamma'
                    ]:
                    plot_amplitude_and_difference_from_json(
                            json_saving_path,
                            session_ID,
                            freq_band = freq_band,
                            saving_path= saving_path)
        """

if "__main__":
    main_freq_response_LFP()