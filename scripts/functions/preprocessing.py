"""
In this file should be stored preprocessing functions.

"""

import numpy as np
import mne
import pandas as pd
from mne.io import read_raw
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from functions.utils import create_1_Hz_wide_bands
from functions.io import save_epochs


def create_epochs(
        file_to_epoch,
        session_ID
):
    events, event_dict = mne.events_from_annotations(file_to_epoch)

    # List of keys to keep
    keys_to_keep = ['GC', 'GF', 'GO', 'GS', 'continue', 'stop']

    # Create the new dictionary by filtering the original one
    filtered_event_dict = {key: event_dict[key] for key in keys_to_keep}

    # Get the event codes (values) from the filtered dictionary
    valid_event_codes = list(filtered_event_dict.values())

    # Filter the events array where the event code is in the valid_event_codes list
    filtered_events = np.array([event for event in events if event[2] in valid_event_codes])

    tmin = -2.5 # previously -0.5
    tmax = 2.5 # previously 1
    baseline=None
    epochs = mne.Epochs(file_to_epoch, filtered_events, event_id=filtered_event_dict, tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
    metadata = pd.DataFrame({'subject':[session_ID] * len(epochs)})
    epochs.metadata = metadata

    return epochs, filtered_event_dict



def create_epochs_subsets_from_behav(
        df_maintask, epochs, event_dict
        ):
    
    (global_idx_dict, mean_RT_dict) = find_idx_success_unsuccess(
        df_maintask, epochs)


    epochs_subsets = create_epochs_subsets(
        epochs, 
        event_dict, 
        global_idx_dict)
    
    (
        indexes_slowest_trials, 
        global_idx_dict,
        mean_RT_dict
        ) = find_slowest_GO_trials(
            df_maintask, 
            epochs, 
            global_idx_dict,
            mean_RT_dict
            )
    
    epochs_lm = epochs.copy()

    event_dict['lm_GO'] = 18

    epochs_lm.event_id = event_dict

    # Manually set the events based on your indices
    epochs_lm.events[global_idx_dict['lm_GO'], 2] = 18  # Update successful GO trials
        
    return (epochs_subsets, epochs_lm, mean_RT_dict)



def create_epochs_specific_freq_band(
        resampled_data,
        freq_band,
        session_ID, 
        df_maintask_copy,
        #epochs_saving_path,
        freq_responses,
        #baseline,
        custom:bool=False,
):
    
    if custom:
        freq_band_range = freq_band
    
    else:
        assert freq_band in [
            "delta", "theta", "alpha", "beta", "low-beta", "high-beta", "gamma", "low-gamma", "medium-gamma", "high-gamma", "custom"
        ], "Invalid frequency band: must be one of 'delta', 'theta', 'alpha', 'beta', 'low-beta', 'high-beta', 'gamma', 'low-gamma', 'medium-gamma', 'high-gamma'"

        bands_range_dict = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 35),
            "low-beta": (13, 20),
            "high-beta": (20, 35),
            "gamma": (35, 90),
            "low-gamma": (35, 40),
            "medium-gamma": (40, 65),
            "high-gamma": (65, 90),
            "custom": (57, 67) #(20, 25) # (58, 65) #(57, 69)
        }

        freq_band_range = bands_range_dict[freq_band]
    
    bands = create_1_Hz_wide_bands(freq_band_range[0], freq_band_range[1])

    # List of channels to process
    channels_to_process = resampled_data.ch_names[:2]  # Adjust if necessary

    # Initialize an empty Raw object to store the βA signals
    band_data = resampled_data.copy()

    channels_to_drop = ["left_peak_STN",
                "right_peak_STN",
                "STIM_Left_STN",
                "STIM_Right_STN"]

    # Drop the specified channels
    band_data.drop_channels(channels_to_drop)

    # Iterate over each channel
    for channel in channels_to_process:
        print(f"Processing channel: {channel}")

        # Create a copy of the original data for filtering
        single_channel_data = resampled_data.copy().pick([channel])  # Use the new `pick()` method

        # Initialize an array to store envelopes for each band
        envelopes = []

        for low, high in bands:
            # Apply bandpass filtering
            filtered = mne.filter.filter_data(
                single_channel_data.get_data().flatten(),
                sfreq=resampled_data.info['sfreq'],
                l_freq=low,
                h_freq=high,
                method='fir',  # Zero-phase FIR filter
                verbose=False,
                fir_design = 'firwin',
                l_trans_bandwidth=0.5,
                h_trans_bandwidth=0.5
            )

            # Compute the analytic signal (Hilbert transform)
            analytic_signal = hilbert(filtered)
            envelope = np.abs(analytic_signal)

            # Normalize the envelope
            normalized_envelope = (envelope - np.nanmean(envelope)) * 100
            #normalized_envelope = envelope
            envelopes.append(normalized_envelope)

            #plt.plot(single_channel_data.get_data())
            #plt.plot(filtered, color="grey")
            #plt.plot(envelope, color="black")
            #plt.plot(normalized_envelope, color="red")
            #plt.legend()
            #plt.show(block=True)

        # Average the envelopes across bands
        freq_resp = np.nanmean(envelopes, axis=0)
        #plt.plot(freq_response)

        # Replace the corresponding channel's data in `band_data`
        band_data._data[resampled_data.ch_names.index(channel), :] = freq_resp

    epochs_band, filtered_event_dict = create_epochs(band_data, session_ID)

    (epochs_band_subsets, 
    epochs_band_lm,
    mean_RT_dict
    ) = create_epochs_subsets_from_behav(
        df_maintask_copy, 
        epochs_band, 
        filtered_event_dict
        )


    for event in [
        'GO_successful', 'GO_unsuccessful',
        'GF_successful', 'GF_unsuccessful',
        'GC_successful', 'GC_unsuccessful',
        'GS_successful', 'GS_unsuccessful',
        'lm_GO'
        ]:
        ##print(event)
        if event == 'lm_GO':
            epochs = epochs_band_lm[event]

        else:
            epochs = epochs_band_subsets[event]
            if len(epochs) == 0:
                print(f"No epochs for event '{event}', skipping.")
                continue
        
        #epochs.plot(block=True)

        # Loop through first two channels
        channels = epochs.info['ch_names'][:2]
        for channel in channels:
            epochs_ch = epochs.copy().pick([channel])

            # Compute averages
            evoked = epochs_ch.average()
            #plt.plot(evoked.times, evoked.data[0], color="black")
            #plt.show(block=True)

            # Define baseline indices
            #baseline_indices = np.where((evoked.times >= baseline[0]) & (evoked.times <= baseline[1]))[0]
            #print(baseline_indices)

            # Compute baseline mean and std for normalization
            #baseline_mean1 = np.mean(evoked.data[:, baseline_indices], axis=1)
            #print(baseline_mean1)
            #baseline_std1 = np.std(evoked.data[:, baseline_indices], axis=1)

            # Normalize responses
            #normalized = (evoked.data - baseline_mean1[:, np.newaxis]) / baseline_std1[:, np.newaxis]
            #normalized = ((evoked.data - baseline_mean1[:, np.newaxis]) / baseline_mean1[:, np.newaxis]) * 100   # calculates percent change from the baseline--> better?
            #plt.plot(normalized[0, :])
            #print(normalized.shape())

            # Store results in session_dict
            #session_dict[f"{channel}_{freq_band}_amp_{event}"] = normalized[0, :]
            key = f"{channel}_{freq_band}_amp_{event}"
            freq_responses[key] = evoked.data[0, :]

        # Store time array only once
        if f"amp_times" not in freq_responses:
            freq_responses[f"amp_times"] = evoked.times

    return freq_responses


def find_idx_success_unsuccess(
    df_maintask: pd.DataFrame,
    epochs: mne.Epochs,
    GLOBAL: bool = True
):
    
    global_idx_dict = {}
    mean_RT_dict = {}

    # In df_maintask, filter the rows where the column 'trial_type' is equal to 
    # 'go_trial' and keep only the columns 'trial_type', 'key_resp_experiment.corr',
    # 'key_resp_experiment.rt'
    df_maintask_go_trials = df_maintask[
        df_maintask['trial_type'] == 'go_trial'
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt']
             ]
    # now separate successful from unsuccessful trials based on the value of 
    # 'key_resp_experiment.corr' (0 = unsuccessful, 1 = successful)
    df_maintask_go_trials.reset_index(drop=True, inplace=True)
    idx_go_trials_successful = df_maintask_go_trials[
        df_maintask_go_trials['key_resp_experiment.corr'] == 1
        ].index
    
    idx_go_trials_unsuccessful = df_maintask_go_trials[
        df_maintask_go_trials['key_resp_experiment.corr'] == 0
        ].index


    # do the same for GS_trials:
    df_maintask_stop_trials = df_maintask[
        df_maintask['trial_type'] == 'stop_trial'
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt']
             ]
    df_maintask_stop_trials.reset_index(drop=True, inplace=True)
    idx_stop_trials_successful = df_maintask_stop_trials[
        df_maintask_stop_trials['key_resp_experiment.corr'] == 1
        ].index
    idx_stop_trials_unsuccessful = df_maintask_stop_trials[
        df_maintask_stop_trials['key_resp_experiment.corr'] == 0
        ].index

    # now do the same for the GF_trials:
    df_maintask_go_fast_trials = df_maintask[#
        df_maintask['trial_type'] == 'go_fast_trial'
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt']
             ]
    df_maintask_go_fast_trials.reset_index(drop=True, inplace=True)
    idx_go_fast_trials_successful = df_maintask_go_fast_trials[
        df_maintask_go_fast_trials['key_resp_experiment.corr'] == 1].index
    idx_go_fast_trials_unsuccessful = df_maintask_go_fast_trials[
        df_maintask_go_fast_trials['key_resp_experiment.corr'] == 0
        ].index

    # now do the same for the GC_trials:
    df_maintask_go_continue_trials = df_maintask[
        df_maintask['trial_type'] == 'go_continue_trial'
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt']
             ]
    df_maintask_go_continue_trials.reset_index(drop=True, inplace=True)
    idx_go_continue_trials_successful = df_maintask_go_continue_trials[
        df_maintask_go_continue_trials['key_resp_experiment.corr'] == 1
        ].index
    idx_go_continue_trials_unsuccessful = df_maintask_go_continue_trials[
        df_maintask_go_continue_trials['key_resp_experiment.corr'] == 0
        ].index

    # extract the indexes based on continue and stop markers:
    df_maintask_stop_trials_markers = df_maintask[
        (df_maintask['trial_type'] == 'stop_trial') & (df_maintask['stop_signal_triangle.started'].notna())
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt']
             ]
    df_maintask_stop_trials_markers.reset_index(drop=True, inplace=True)
    idx_stop_markers_successful = df_maintask_stop_trials_markers[
        df_maintask_stop_trials_markers['key_resp_experiment.corr'] == 1
        ].index
    idx_stop_markers_unsuccessful = df_maintask_stop_trials_markers[
        df_maintask_stop_trials_markers['key_resp_experiment.corr'] == 0
        ].index    
    

    df_maintask_continue_trials_markers = df_maintask[
        (df_maintask['trial_type'] == 'go_continue_trial') & (df_maintask['stop_signal_triangle.started'].notna())
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt']
             ]
    df_maintask_continue_trials_markers.reset_index(drop=True, inplace=True)
    idx_continue_markers_successful = df_maintask_continue_trials_markers[
        df_maintask_continue_trials_markers['key_resp_experiment.corr'] == 1
        ].index
    idx_continue_markers_unsuccessful = df_maintask_continue_trials_markers[
        df_maintask_continue_trials_markers['key_resp_experiment.corr'] == 0
        ].index 

    if GLOBAL:
        global_idx_dict['GO_successful'] = epochs['GO'][idx_go_trials_successful].selection
        global_idx_dict['GO_unsuccessful'] = epochs['GO'][idx_go_trials_unsuccessful].selection
        global_idx_dict['GS_successful'] = epochs['GS'][idx_stop_trials_successful].selection
        global_idx_dict['GS_unsuccessful'] = epochs['GS'][idx_stop_trials_unsuccessful].selection
        global_idx_dict['GF_successful'] = epochs['GF'][idx_go_fast_trials_successful].selection
        global_idx_dict['GF_unsuccessful'] = epochs['GF'][idx_go_fast_trials_unsuccessful].selection
        global_idx_dict['GC_successful'] = epochs['GC'][idx_go_continue_trials_successful].selection
        global_idx_dict['GC_unsuccessful'] = epochs['GC'][idx_go_continue_trials_unsuccessful].selection
        global_idx_dict['STOP_successful'] = epochs['stop'][idx_stop_markers_successful].selection
        global_idx_dict['STOP_unsuccessful'] = epochs['stop'][idx_stop_markers_unsuccessful].selection
        global_idx_dict['CONTINUE_successful'] = epochs['continue'][idx_continue_markers_successful].selection
        global_idx_dict['CONTINUE_unsuccessful'] = epochs['continue'][idx_continue_markers_unsuccessful].selection

        mean_RT_dict['GO_successful'] = (df_maintask_go_trials[
            df_maintask_go_trials['key_resp_experiment.corr'] == 1
            ][
            'key_resp_experiment.rt'].mean())*1000
        mean_RT_dict['GS_unsuccessful'] = (df_maintask_stop_trials[
            df_maintask_stop_trials['key_resp_experiment.corr'] == 0
            ][
            'key_resp_experiment.rt'].mean())*1000
        mean_RT_dict['GF_successful'] = (df_maintask_go_fast_trials[
            df_maintask_go_fast_trials['key_resp_experiment.corr'] == 1][
            'key_resp_experiment.rt'].mean())*1000
        mean_RT_dict['GC_successful'] = (df_maintask_go_continue_trials[
            df_maintask_go_continue_trials['key_resp_experiment.corr'] == 1
            ][
            'key_resp_experiment.rt'].mean())*1000
        
        # Extract mean SSD, and compute SSRT:
        ordered_go_rt = np.sort((df_maintask[df_maintask['trial_type'] == 'go_trial']['key_resp_experiment.rt'].dropna() *1000).tolist())
        percent_corr_stop = len(
            df_maintask[
                (df_maintask['trial_type'] == 'stop_trial') &
                  (df_maintask['key_resp_experiment.corr'] == 1) &
                    (df_maintask['early_press_resp.corr'] == 0)
                    ] 
                    ) / len(df_maintask[df_maintask['trial_type'] == 'stop_trial'])*100
        n = round((1 - (percent_corr_stop / 100)) * len(ordered_go_rt))
        nth_GO_RT = ordered_go_rt[n]
        stop_trials = df_maintask[df_maintask['trial_type'] == 'stop_trial']
        mean_ssd = (stop_trials['stop_signal_time'].mean())*1000

        go_continue_trials = df_maintask[df_maintask['trial_type'] == 'go_continue_trial']
        mean_csd = (go_continue_trials['continue_signal_time'].mean())*1000

        #ssrt_value = nth_GO_RT - mean_ssd   

        #mean_RT_dict['mean SSD (ms)'] = mean_ssd
        #mean_RT_dict['SSRT (ms)'] = ssrt_value
        mean_RT_dict['STOP_unsuccessful'] = ((df_maintask_stop_trials[
            df_maintask_stop_trials['key_resp_experiment.corr'] == 0
            ][
            'key_resp_experiment.rt'].mean())*1000) - mean_ssd
        
        mean_RT_dict['CONTINUE_successful'] = ((df_maintask_go_continue_trials[
            df_maintask_go_continue_trials['key_resp_experiment.corr'] == 1
            ][
            'key_resp_experiment.rt'].mean())*1000) - mean_csd

        print(global_idx_dict.keys())
        return global_idx_dict, mean_RT_dict
    
    
    else:
        return (idx_go_trials_successful, idx_go_trials_unsuccessful, 
                idx_stop_trials_successful, idx_stop_trials_unsuccessful, 
                idx_go_fast_trials_successful, idx_go_fast_trials_unsuccessful, 
                idx_go_continue_trials_successful, idx_go_continue_trials_unsuccessful,
                idx_stop_markers_successful, idx_stop_markers_unsuccessful,
                idx_continue_markers_successful, idx_continue_markers_unsuccessful
                )




def find_slowest_GO_trials(
        df_maintask: pd.DataFrame, 
        epochs, 
        global_idx_dict,
        mean_RT_dict
        ):
    """
    This function finds the epochs corresponding to the 50% slowest GO trials with correct responses.

    Parameters
    ----------
    df_maintask : pd.DataFrame
        The dataframe containing the behavioral data of the main task.
    epochs : mne.Epochs
        The epochs object containing GO trial data.

    Returns
    -------
    indexes_slowest_trials : np.ndarray
        The indexes of the 50% slowest GO trials with correct responses in the original dataframe.
    global_idx_lm_go_trials : np.ndarray
        The corresponding global indexes of these trials in the `epochs['GO']` object.
    mean_slowest_GO_RT : float
        The mean reaction time (in milliseconds) of the 50% slowest GO trials.
    """
    df_maintask_go_trials = df_maintask[
        df_maintask['trial_type'] == 'go_trial'
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt', 
             'early_press_resp.corr']
             ]

    df_maintask_go_trials.reset_index(drop=True, inplace=True)
  
    # Remove trials with missing reaction times (NaN)
    df_go_correct_trials = df_maintask_go_trials.dropna(subset=['key_resp_experiment.rt'])

    # Determine the number of slowest trials (50%)
    n_slowest_trials = int(len(df_go_correct_trials) / 2)

    # Sort by reaction time in descending order to find the slowest trials
    slowest_trials = df_go_correct_trials.sort_values(
        by='key_resp_experiment.rt', ascending=False
    ).head(n_slowest_trials)

    # Get the original indices of the slowest trials
    indexes_slowest_trials = slowest_trials.index.values

    # Use these indices to map to the global indexes in epochs['GO']
    global_idx_dict['lm_GO'] = epochs['GO'][indexes_slowest_trials].selection

    # Calculate the mean reaction time of the 50% slowest trials in milliseconds
    mean_RT_dict['lm_GO'] = slowest_trials['key_resp_experiment.rt'].mean() * 1000
     

    return indexes_slowest_trials, global_idx_dict, mean_RT_dict


def create_epochs_subsets(
        epochs, 
        filtered_event_dict, 
        global_idx_dict):
    my_keys = ['GO_successful', 'GO_unsuccessful', 'GC_successful', 'GC_unsuccessful', 'GF_successful', 'GF_unsuccessful','GS_successful', 'GS_unsuccessful', 'STOP_successful', 'STOP_unsuccessful', 'CONTINUE_successful', 'CONTINUE_unsuccessful']
    my_values = [10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23]
    for key,value in zip(my_keys, my_values):
        filtered_event_dict[key] = value
    
    epochs_subsets = epochs.copy()
    epochs_subsets.event_id = filtered_event_dict

    for key,value in zip(my_keys, my_values):
        filtered_event_dict[key] = value
        epochs_subsets.events[global_idx_dict[key], 2] = value

    return epochs_subsets







def compute_frequency_maps (
        event_dict,
        idx_go_trials_successful, 
        idx_go_trials_unsuccessful, 
        idx_stop_trials_successful, 
        idx_stop_trials_unsuccessful, 
        idx_gf_trials_successful, 
        idx_gf_trials_unsuccessful, 
        idx_gc_trials_successful, 
        idx_gc_trials_unsuccessful, 
        iter_freqs, 
        tmin, tmax, 
        baseline, 
        file, 
        events, 
        session_ID, 
        frequency_maps_all_dict,
        ):
    frequency_maps_dict = {}

    for event_ID in ['GO', 'GS', 'GF', 'GC']:
        # set epoching parameters
        event_id = event_dict[event_ID]
        print(f"Now analyzing event {event_ID}")
        if event_ID == 'GO':
            idx_success = idx_go_trials_successful
            idx_unsuccess = idx_go_trials_unsuccessful
        elif event_ID == 'GS':
            idx_success = idx_stop_trials_successful
            idx_unsuccess = idx_stop_trials_unsuccessful
        elif event_ID == 'GF':
            idx_success = idx_gf_trials_successful
            idx_unsuccess = idx_gf_trials_unsuccessful
        elif event_ID == 'GC':
            idx_success = idx_gc_trials_successful
            idx_unsuccess = idx_gc_trials_unsuccessful
        

        for ch in (0,1):
            print(f'Processing channel {ch}')
            frequency_map_success = list()
            frequency_map_unsuccess = list()
            for band, fmin, fmax in iter_freqs:
            # (re)load the data to save memory
                raw = read_raw(file, preload=True)
                raw.pick([ch])

                # bandpass filter
                raw.filter(
                    fmin,
                    fmax,
                    n_jobs=None, 
                    l_trans_bandwidth=1, 
                    h_trans_bandwidth=1,
                )

                # epoch
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=baseline,
                    preload=True,
                )
                
                # Create separate Epochs objects
                epochs_successful = epochs[idx_success]
                epochs_unsuccessful = epochs[idx_unsuccess]
                
                # Remove evoked response
                epochs_successful.subtract_evoked()
                if len(epochs_unsuccessful) > 0:
                    epochs_unsuccessful.subtract_evoked()
                
                # Apply Hilbert transform to get the envelope
                epochs_successful.apply_hilbert(envelope=True)
                if len(epochs_unsuccessful) > 0:
                    epochs_unsuccessful.apply_hilbert(envelope=True)
                
                # Store the frequency map for averaging
                frequency_map_success.append(((band, fmin, fmax), epochs_successful.average()))
                if len(epochs_unsuccessful) > 0:
                    frequency_map_unsuccess.append(((band, fmin, fmax), epochs_unsuccessful.average()))

                # add to the frequency maps dictionnary of this session
                side = 'left' if ch == 0 else 'right'
                dic_key_success = f'{event_ID} successful {side} STN'
                dic_key_unsuccess = f'{event_ID} unsuccessful {side} STN'
                frequency_maps_dict[dic_key_success] = frequency_map_success
                if len(epochs_unsuccessful) > 0:
                    frequency_maps_dict[dic_key_unsuccess] = frequency_map_unsuccess

                # add the frequency map dictionnary of this session to the mother dictionnary
                frequency_maps_all_dict[session_ID] = frequency_maps_dict
                
                # Cleanup
                del epochs, epochs_successful, epochs_unsuccessful
        del raw

    return frequency_maps_all_dict, frequency_maps_dict








################################ OLD FUNCTIONS #################################
'''
def separate_success_unsucess(
        df_maintask: pd.DataFrame, 
        GO_trials: np.ndarray, 
        GS_trials: np.ndarray, 
        GF_trials: np.ndarray, 
        GC_trials: np.ndarray, 
        trials_dict: dict
        ):
    """
    This function separates successful from unsuccessful trials in the GO, GS, GF and GC trials.
    It returns a dictionary containing the successful and unsuccessful trials for each condition.
    It also returns the mean reaction time for successful GO trials, unsuccessful STOP trials, successful GF trials and successful GC trials.

    Parameters
    ----------
    df_maintask : pd.DataFrame
        The dataframe containing the behavioral data of the main task.
    GO_trials : np.ndarray
        The array containing the epochs of all GO trials.
    GS_trials : np.ndarray
        The array containing the epochs of all STOP trials.
    GF_trials : np.ndarray
        The array containing the epochs of all GF trials.
    GC_trials : np.ndarray
        The array containing the epochs of all GC trials.
    trials_dict : dict
        The dictionary containing the arrays of trials for each condition.
    
    Returns
    -------
    trials_dict : dict
        The dictionary containing the arrays of trials for each condition.
    GO_trials_successful : np.ndarray
        The array containing the epochs of successful GO trials.
    GO_trials_unsuccessful : np.ndarray
        The array containing the epochs of unsuccessful GO trials.
    stop_trials_successful : np.ndarray
        The array containing the epochs of successful STOP trials.
    stop_trials_unsuccessful : np.ndarray
        The array containing the epochs of unsuccessful STOP trials.
    go_fast_trials_successful : np.ndarray
        The array containing the epochs of successful GF trials.
    go_fast_trials_unsuccessful : np.ndarray
        The array containing the epochs of unsuccessful GF trials.
    go_continue_trials_successful : np.ndarray
        The array containing the epochs of successful GC trials.
    go_continue_trials_unsuccessful : np.ndarray
        The array containing the epochs of unsuccessful GC trials.
    mean_GO_RT_successful : float
        The mean reaction time of successful GO trials.
    mean_RT_unsuccessful_STOP : float
        The mean reaction time of unsuccessful STOP trials.
    mean_GF_RT_successful : float
        The mean reaction time of successful GF trials.
    mean_GC_RT_successful : float
        The mean reaction time of successful GC trials.
    """


    # In df_maintask, filter the rows where the column 'trial_type' is equal to 'go_trial' and keep only the columns 'trial_type', ''key_resp_experiment.corr','key_resp_experiment.rt', 'early_press_resp.corr', 'late_key_resp1.corr'
    df_maintask_go_trials = df_maintask[
        df_maintask['trial_type'] == 'go_trial'
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt',
              'early_press_resp.corr']
              ]
    df_maintask_go_trials.reset_index(drop=True, inplace=True)
    # now separate successful from unsuccessful trials based on the value of 'key_resp_experiment.corr' (0 = unsuccessful, 1 = successful)
    df_maintask_go_trials_successful = df_maintask_go_trials[df_maintask_go_trials['key_resp_experiment.corr'] == 1]
    df_maintask_go_trials_unsuccessful = df_maintask_go_trials[df_maintask_go_trials['key_resp_experiment.corr'] == 0]
    # now extract the indexes of the successful and unsuccessful trials
    successful_trials_indexes = df_maintask_go_trials_successful.index
    unsuccessful_trials_indexes = df_maintask_go_trials_unsuccessful.index
    # separate the array GO_trials containing epochs of all GO trials into 2 arrays containing successful and unsuccessful trials
    GO_trials_successful = np.array([GO_trials[i] for i in successful_trials_indexes])
    GO_trials_unsuccessful = np.array([GO_trials[i] for i in unsuccessful_trials_indexes])
    mean_GO_RT_successful = df_maintask_go_trials_successful['key_resp_experiment.rt'].mean()

    # do the same for GS_trials:
    df_maintask_stop_trials = df_maintask[df_maintask['trial_type'] == 'stop_trial'][['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt', 'early_press_resp.corr']]
    df_maintask_stop_trials.reset_index(drop=True, inplace=True)
    df_maintask_stop_trials_successful = df_maintask_stop_trials[df_maintask_stop_trials['key_resp_experiment.corr'] == 1]
    df_maintask_stop_trials_unsuccessful = df_maintask_stop_trials[df_maintask_stop_trials['key_resp_experiment.corr'] == 0]
    successful_stop_trials_indexes = df_maintask_stop_trials_successful.index
    unsuccessful_stop_trials_indexes = df_maintask_stop_trials_unsuccessful.index
    stop_trials_successful = np.array([GS_trials[i] for i in successful_stop_trials_indexes])
    stop_trials_unsuccessful = np.array([GS_trials[i] for i in unsuccessful_stop_trials_indexes])
    mean_RT_unsuccessful_STOP = df_maintask_stop_trials_unsuccessful['key_resp_experiment.rt'].mean()

    # now do the same for the GF_trials:
    df_maintask_go_fast_trials = df_maintask[df_maintask['trial_type'] == 'go_fast_trial'][['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt', 'early_press_resp.corr']]
    df_maintask_go_fast_trials.reset_index(drop=True, inplace=True)
    df_maintask_go_fast_trials_successful = df_maintask_go_fast_trials[df_maintask_go_fast_trials['key_resp_experiment.corr'] == 1]
    df_maintask_go_fast_trials_unsuccessful = df_maintask_go_fast_trials[df_maintask_go_fast_trials['key_resp_experiment.corr'] == 0]
    successful_go_fast_trials_indexes = df_maintask_go_fast_trials_successful.index
    unsuccessful_go_fast_trials_indexes = df_maintask_go_fast_trials_unsuccessful.index
    go_fast_trials_successful = np.array([GF_trials[i] for i in successful_go_fast_trials_indexes])
    go_fast_trials_unsuccessful = np.array([GF_trials[i] for i in unsuccessful_go_fast_trials_indexes])
    mean_GF_RT_successful = df_maintask_go_fast_trials_successful['key_resp_experiment.rt'].mean()

    # now do the same for the GC_trials:
    df_maintask_go_continue_trials = df_maintask[df_maintask['trial_type'] == 'go_continue_trial'][['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt', 'early_press_resp.corr']]
    df_maintask_go_continue_trials.reset_index(drop=True, inplace=True)
    df_maintask_go_continue_trials_successful = df_maintask_go_continue_trials[df_maintask_go_continue_trials['key_resp_experiment.corr'] == 1]
    df_maintask_go_continue_trials_unsuccessful = df_maintask_go_continue_trials[df_maintask_go_continue_trials['key_resp_experiment.corr'] == 0]
    successful_go_continue_trials_indexes = df_maintask_go_continue_trials_successful.index
    unsuccessful_go_continue_trials_indexes = df_maintask_go_continue_trials_unsuccessful.index
    go_continue_trials_successful = np.array([GC_trials[i] for i in successful_go_continue_trials_indexes])
    go_continue_trials_unsuccessful = np.array([GC_trials[i] for i in unsuccessful_go_continue_trials_indexes])
    mean_GC_RT_successful = df_maintask_go_continue_trials_successful['key_resp_experiment.rt'].mean()

    trials_dict['GO_successful'] = GO_trials_successful
    trials_dict['GO_unsuccessful'] = GO_trials_unsuccessful
    trials_dict['stop_successful'] = stop_trials_successful
    trials_dict['stop_unsuccessful'] = stop_trials_unsuccessful
    trials_dict['GF_successful'] = go_fast_trials_successful
    trials_dict['GF_unsuccessful'] = go_fast_trials_unsuccessful
    trials_dict['GC_successful'] = go_continue_trials_successful
    trials_dict['GC_unsuccessful'] = go_continue_trials_unsuccessful

    return (trials_dict, GO_trials_successful, GO_trials_unsuccessful, 
            stop_trials_successful, stop_trials_unsuccessful, 
            go_fast_trials_successful, go_fast_trials_unsuccessful, 
            go_continue_trials_successful, go_continue_trials_unsuccessful, 
            mean_GO_RT_successful, mean_RT_unsuccessful_STOP, 
            mean_GF_RT_successful, mean_GC_RT_successful)





def create_epochs(
        raw: mne.io.Raw, 
        events: np.ndarray, 
        event_dict: dict, 
        tmin: float, 
        tmax: float, 
        picks: list, 
        session_ID: str
        ):
    """
    This function creates epochs for each condition of the main task.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw object containing the EEG data.
    events : np.ndarray
        The array containing the events of the main task.
    event_dict : dict
        The dictionary containing the event codes for each condition.
    tmin : float
        The start time of the epochs.
    tmax : float
        The end time of the epochs.
    picks : list
        The list of channels to keep.
    session_ID : str
        The ID of the session.
    
    Returns
    -------
    epochs_condition_GO : mne.Epochs
        The epochs of the GO condition.
    epochs_condition_STOP : mne.Epochs
        The epochs of the STOP condition.
    epochs_condition_GF : mne.Epochs
        The epochs of the Go Fast condition.
    epochs_condition_GC : mne.Epochs
        The epochs of the Go Continue condition.
    """

    epochs_condition_GO = mne.Epochs(
        raw, events, event_dict['GO'], tmin, tmax, picks=picks, 
        baseline=(None, 0), preload=True
        )
    metadata_epochs_condition_GO = pd.DataFrame(
        {'subject':[session_ID] * len(epochs_condition_GO)}
        )
    epochs_condition_GO.metadata = metadata_epochs_condition_GO
    epochs_condition_STOP = mne.Epochs(
        raw, events, event_dict['GS'], tmin, tmax, picks=picks, 
        baseline=(None, 0), preload=True
        )
    metadata_epochs_condition_STOP = pd.DataFrame(
        {'subject':[session_ID] * len(epochs_condition_STOP)}
        )
    epochs_condition_STOP.metadata = metadata_epochs_condition_STOP
    epochs_condition_GF = mne.Epochs(
        raw, events, event_dict['GF'], tmin, tmax, picks=picks, 
        baseline=(None, 0), preload=True
        )
    metadata_epochs_condition_GF = pd.DataFrame(
        {'subject':[session_ID] * len(epochs_condition_GF)}
        )
    epochs_condition_GF.metadata = metadata_epochs_condition_GF
    epochs_condition_GC = mne.Epochs(
        raw, events, event_dict['GC'], tmin, tmax, picks=picks, 
        baseline=(None, 0), preload=True
        )
    metadata_epochs_condition_GC = pd.DataFrame(
        {'subject':[session_ID] * len(epochs_condition_GC)}
        )
    epochs_condition_GC.metadata = metadata_epochs_condition_GC
    
    return (epochs_condition_GO, epochs_condition_STOP, 
            epochs_condition_GF, epochs_condition_GC)

            
            
            
            

            
            
            
def epochs_separate_success_unsuccess(
    df_maintask: pd.DataFrame, 
    epochs_condition_GO: mne.Epochs, 
    epochs_condition_STOP: mne.Epochs, 
    epochs_condition_GF: mne.Epochs, 
    epochs_condition_GC: mne.Epochs
):
    """
    This function separates successful from unsuccessful trials in the GO, GS, GF and GC trials.

    Parameters
    ----------
    df_maintask : pd.DataFrame
        The dataframe containing the behavioral data of the main task.
    epochs_condition_GO : mne.Epochs
        The epochs of the GO condition.
    epochs_condition_STOP : mne.Epochs
        The epochs of the STOP condition.
    epochs_condition_GF : mne.Epochs
        The epochs of the Go Fast condition.
    epochs_condition_GC : mne.Epochs
        The epochs of the Go Continue condition.
    
    Returns
    -------
    epochs_GO_trials_successful : mne.Epochs
        The epochs of successful GO trials.
    epochs_GO_trials_unsuccessful : mne.Epochs
        The epochs of unsuccessful GO trials.
    epochs_stop_trials_successful : mne.Epochs
        The epochs of successful STOP trials.
    epochs_stop_trials_unsuccessful : mne.Epochs
        The epochs of unsuccessful STOP trials.
    epochs_go_fast_trials_successful : mne.Epochs
        The epochs of successful GF trials.
    epochs_go_fast_trials_unsuccessful : mne.Epochs
        The epochs of unsuccessful GF trials.
    epochs_go_continue_trials_successful : mne.Epochs
        The epochs of successful GC trials.
    epochs_go_continue_trials_unsuccessful : mne.Epochs
        The epochs of unsuccessful GC trials.
    mean_GO_RT_successful : float
        The mean reaction time of successful GO trials.
    mean_RT_unsuccessful_STOP : float
        The mean reaction time of unsuccessful STOP trials.
    mean_GF_RT_successful : float
        The mean reaction time of successful GF trials.
    mean_GC_RT_successful : float
        The mean reaction time of successful GC trials.
    """

    # In df_maintask, filter the rows where the column 'trial_type' is equal to 
    # 'go_trial' and keep only the columns 'trial_type', 'key_resp_experiment.corr',
    # 'key_resp_experiment.rt', 'early_press_resp.corr', 'late_key_resp1.corr'
    df_maintask_go_trials = df_maintask[
        df_maintask['trial_type'] == 'go_trial'
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt', 
             'early_press_resp.corr']
             ]
    df_maintask_go_trials.reset_index(drop=True, inplace=True)
    # now separate successful from unsuccessful trials based on the value of 
    # 'key_resp_experiment.corr' (0 = unsuccessful, 1 = successful)
    df_maintask_go_trials_successful = df_maintask_go_trials[
        df_maintask_go_trials['key_resp_experiment.corr'] == 1
        ]
    df_maintask_go_trials_unsuccessful = df_maintask_go_trials[
        df_maintask_go_trials['key_resp_experiment.corr'] == 0
        ]
    # now extract the indexes of the successful and unsuccessful trials
    successful_trials_indexes = df_maintask_go_trials_successful.index
    unsuccessful_trials_indexes = df_maintask_go_trials_unsuccessful.index
    # separate the array GO_trials containing epochs of all GO trials into 2 arrays 
    # containing successful and unsuccessful trials
    epochs_GO_trials_successful = epochs_condition_GO[successful_trials_indexes]
    epochs_GO_trials_unsuccessful =epochs_condition_GO[unsuccessful_trials_indexes]
    mean_GO_RT_successful = df_maintask_go_trials_successful[
        'key_resp_experiment.rt'].mean()

    # do the same for GS_trials:
    df_maintask_stop_trials = df_maintask[
        df_maintask['trial_type'] == 'stop_trial'
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt', 
             'early_press_resp.corr']
             ]
    df_maintask_stop_trials.reset_index(drop=True, inplace=True)
    df_maintask_stop_trials_successful = df_maintask_stop_trials[
        df_maintask_stop_trials['key_resp_experiment.corr'] == 1
        ]
    df_maintask_stop_trials_unsuccessful = df_maintask_stop_trials[
        df_maintask_stop_trials['key_resp_experiment.corr'] == 0
        ]
    successful_stop_trials_indexes = df_maintask_stop_trials_successful.index
    unsuccessful_stop_trials_indexes = df_maintask_stop_trials_unsuccessful.index
    epochs_stop_trials_successful = epochs_condition_STOP[successful_stop_trials_indexes]
    epochs_stop_trials_unsuccessful = epochs_condition_STOP[unsuccessful_stop_trials_indexes]
    mean_RT_unsuccessful_STOP = df_maintask_stop_trials_unsuccessful[
        'key_resp_experiment.rt'].mean()

    # now do the same for the GF_trials:
    df_maintask_go_fast_trials = df_maintask[
        df_maintask['trial_type'] == 'go_fast_trial'
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt', 
             'early_press_resp.corr']
             ]
    df_maintask_go_fast_trials.reset_index(drop=True, inplace=True)
    df_maintask_go_fast_trials_successful = df_maintask_go_fast_trials[
        df_maintask_go_fast_trials['key_resp_experiment.corr'] == 1]
    df_maintask_go_fast_trials_unsuccessful = df_maintask_go_fast_trials[
        df_maintask_go_fast_trials['key_resp_experiment.corr'] == 0
        ]
    successful_go_fast_trials_indexes = df_maintask_go_fast_trials_successful.index
    unsuccessful_go_fast_trials_indexes = df_maintask_go_fast_trials_unsuccessful.index
    epochs_go_fast_trials_successful = epochs_condition_GF[successful_go_fast_trials_indexes]
    epochs_go_fast_trials_unsuccessful = epochs_condition_GF[unsuccessful_go_fast_trials_indexes]
    mean_GF_RT_successful = df_maintask_go_fast_trials_successful[
        'key_resp_experiment.rt'].mean()
    
    # now do the same for the GC_trials:
    df_maintask_go_continue_trials = df_maintask[
        df_maintask['trial_type'] == 'go_continue_trial'
        ][
            ['trial_type', 'key_resp_experiment.corr', 'key_resp_experiment.rt', 
             'early_press_resp.corr']
             ]
    df_maintask_go_continue_trials.reset_index(drop=True, inplace=True)
    df_maintask_go_continue_trials_successful = df_maintask_go_continue_trials[
        df_maintask_go_continue_trials['key_resp_experiment.corr'] == 1
        ]
    df_maintask_go_continue_trials_unsuccessful = df_maintask_go_continue_trials[
        df_maintask_go_continue_trials['key_resp_experiment.corr'] == 0
        ]
    successful_go_continue_trials_indexes = df_maintask_go_continue_trials_successful.index
    unsuccessful_go_continue_trials_indexes = df_maintask_go_continue_trials_unsuccessful.index
    epochs_go_continue_trials_successful = epochs_condition_GC[successful_go_continue_trials_indexes]
    epochs_go_continue_trials_unsuccessful = epochs_condition_GC[unsuccessful_go_continue_trials_indexes]
    mean_GC_RT_successful = df_maintask_go_continue_trials_successful[
        'key_resp_experiment.rt'].mean()

    return (epochs_GO_trials_successful, epochs_GO_trials_unsuccessful,
            epochs_stop_trials_successful, epochs_stop_trials_unsuccessful,
            epochs_go_fast_trials_successful, epochs_go_fast_trials_unsuccessful,
            epochs_go_continue_trials_successful, epochs_go_continue_trials_unsuccessful,
            mean_GO_RT_successful, mean_RT_unsuccessful_STOP,
            mean_GF_RT_successful, mean_GC_RT_successful)            


def create_epoch_category(
        epoch_object: mne.Epochs, 
        indexes: np.ndarray
        ):
    """
    This function creates a new epoch object containing only the epochs at the indexes specified.

    Parameters
    ----------
    epoch_object : mne.Epochs
        The epoch object to filter.
    indexes : np.ndarray
        The indexes of the epochs to keep.

    Returns
    -------
    new_epoch_object : mne.Epochs
        The new epoch object containing only the epochs at the indexes specified.
    """

    new_epoch_object = epoch_object[indexes]
    return new_epoch_object


'''
