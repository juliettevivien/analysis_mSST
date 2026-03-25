import mne
import numpy as np
import pandas as pd
import os






def get_tfr_decomposition(
        epochs, 
        cond_of_interest, 
        ch_names, 
        tfr_args, 
        baseline_correction, 
        baseline_correction_method, 
        tmin_tmax
        ):
    latency_matched = False
    
    if cond_of_interest.split('_')[0] == 'lmGO':
        latency_matched = True
        epoch_type = 'GO'
    else:
        # Parse epoch condition
        epoch_type = cond_of_interest.split('_')[0]

    outcome_str, aligned_str = cond_of_interest.split('_')[1], cond_of_interest.split('_')[2]
    outcome = 1.0 if outcome_str == "successful" else 0.0     

    # Select the appropriate epochs
    type_mask = epochs.metadata["event"] == epoch_type
    outcome_mask = epochs.metadata["key_resp_experiment.corr"] == outcome
    data = epochs[type_mask & outcome_mask]    

    if latency_matched:
        rt = np.asarray(data.metadata['key_resp_experiment.rt'])
        threshold = np.percentile(rt, 50)
        slow_mask = rt >= threshold
        data = data[slow_mask] 

    # Select only desired channels
    epochs = data.copy().pick(ch_names)

    # Compute TFR
    power = epochs.compute_tfr(**tfr_args)  # shape: (n_epochs, n_channels, n_freqs, n_times)      
    power.data *= 1e12  # V² → (µV)²
    
    # Average across channels if multiple channels are specified
    power_mean = np.nanmean(power.data, axis=1)  # (n_epochs, n_freqs, n_times)

    times = power.times * 1000
    dt_ms = 1/round(epochs.info['sfreq'])
    #n_times_new = int(np.round(((tmin_tmax[1]- tmin_tmax[0])*1000) / dt_ms))
    freqs = power.freqs

    # crop to keep only times of interest               
    # determine t0 et t1 indices based on tmin_tmax and aligned_str
    if aligned_str != 'square':
        if aligned_str == 'response':
            t0_per_trial = np.array(data.metadata['key_resp_experiment.rt']) * 1000  # convert to ms
        elif aligned_str == 'feedback':
            if epoch_type == 'GS' and outcome_str == 'successful':
                t0_per_trial = (np.array(data.metadata['stop_signal_time']) + 1.5) * 1000  # feedback is at 1000 ms after response
            else:
                t0_per_trial = (np.array(data.metadata['key_resp_experiment.rt']) + 0.5) * 1000  # feedback is at 500 ms after response
        elif aligned_str == 'triangle':
            if epoch_type == 'GS': 
                ssd_column = 'stop_signal_time'
            elif epoch_type == 'GC':
                ssd_column = 'continue_signal_time'                        
            t0_per_trial = np.array(data.metadata[ssd_column]) * 1000
    
        new_epochs = []
        if baseline_correction and baseline_correction_method == 'group_average':
            baseline_power = np.empty((power_mean.shape[0], power_mean.shape[1], 1))  # (n_trials, n_freqs, 1)
        
        for i in range(power_mean.shape[0]):
            if baseline_correction and baseline_correction_method == 'group_average':
                # get the baseline values for this trial
                bl_start = -500  # in ms
                bl_end = -200  # in ms
                bl_idx = (times >= bl_start) & (times <= bl_end)
                # Compute mean power in this window for all frequencies
                bl_mean = np.nanmean(power_mean[i][ :, bl_idx], axis=1, keepdims=True)

                # Store baseline power
                baseline_power[i] = bl_mean                    
            
            t0_idx = t0_per_trial[i] + tmin_tmax[0]*1000
            t1_idx = t0_per_trial[i] + tmin_tmax[1]*1000
            time_idx = (times >= t0_idx) & (times <= t1_idx)
            new_epochs.append(power_mean[i][:, time_idx])  # shape: (n_freqs, n_times_new)

        min_len = min(e.shape[1] for e in new_epochs)
        new_epochs = [e[:, :min_len] for e in new_epochs]
        new_epochs = np.stack(new_epochs, axis=0)
        #new_epochs = np.stack(new_epochs, axis=0)  # shape: (n_epochs, n_freqs, n_times_new)
        new_times = np.arange(tmin_tmax[0]*1000, tmin_tmax[1]*1000, dt_ms)
        mean_power = np.nanmean(new_epochs, axis=0)
        times = new_times

        if baseline_correction and baseline_correction_method == 'group_average':
            baseline_power = np.nanmean(baseline_power, axis=0)  # shape: (n_freqs, 1 time)
            change = (mean_power - baseline_power) / baseline_power * 100  # percent change  # shape: (n_freqs, n_times)
            mean_power = change
    
    else:
        mean_power = np.nanmean(power_mean, axis=0)  # (n_freqs, n_times)

        if baseline_correction and baseline_correction_method == 'group_average':
            # Define baseline period for change calculation
            baseline_indices = (times >= -500) & (times <= -200)
            baseline_power = np.nanmean(mean_power[:, baseline_indices], axis=1, keepdims=True)  # shape: (n_freqs, 1 time)
            change = (mean_power - baseline_power) / baseline_power * 100  # percent change  # shape: (n_freqs, n_times)
            mean_power = change
        
        time_idx = (times >= tmin_tmax[0]*1000) & (times <= tmin_tmax[1]*1000)
        mean_power = mean_power[:, time_idx]
        times = times[time_idx]

    return mean_power, times, freqs        



def eeg_get_change_from_baseline(
        epochs,
        cond,
        ch_of_interest,
        tfr_args,
        baseline_correction=True,
        baseline_correction_method='group_average'
):
    """
    Compute change from baseline (in dB) for one or multiple EEG channels.
    
    If multiple channels are provided, returns the average power/change across channels.
    """
    # import numpy as np
    latency_matched = False

    if cond == 'lm_GO_successful':
        latency_matched = True
        epoch_type = 'GO'
        outcome_str = 'successful'
        outcome = 1.0
    else:
        # Parse epoch condition
        epoch_type, outcome_str = cond.split('_')
        outcome = 1.0 if outcome_str == "successful" else 0.0

    # Select the appropriate epochs
    type_mask = epochs.metadata["event"] == epoch_type
    outcome_mask = epochs.metadata["key_resp_experiment.corr"] == outcome
    data = epochs[type_mask & outcome_mask]

    if latency_matched:
        print(f"{epoch_type}: before latency filter = {len(data)} epochs")
        rt = np.asarray(data.metadata['key_resp_experiment.rt'])
        print(f"RT shape = {rt.shape}, first few RTs = {rt[:5]}")
        threshold = np.percentile(rt, 50)
        print(f"Median RT = {threshold:.3f}s")
        slow_mask = rt >= threshold
        print(f"{slow_mask.sum()} epochs slower than median")
        # Ensure mask alignment to metadata
        data = data[slow_mask] 

    # ---- CHANGE 1: allow multiple channels ----
    # ch_of_interest can now be a string (single channel) or list of channels
    if isinstance(ch_of_interest, str):
        ch_names = [ch_of_interest]
    else:
        ch_names = ch_of_interest

    # Select only desired channels
    epochs = data.copy().pick(ch_names)

    # Compute TFR
    power = epochs.compute_tfr(**tfr_args)  # shape: (n_epochs, n_channels, n_freqs, n_times)
    print(f'power shape: {power.data.shape}')

    power.data *= 1e12  # V² → (µV)²

    # ---- CHANGE 2: average across channels ----
    # mean over channel axis (axis=1)
    power_mean = np.nanmean(power.data, axis=1)  # (n_epochs, n_freqs, n_times)
    print(f'power_mean shape: {power_mean.shape}')

    times = power.times * 1000
    freqs = power.freqs

    print(f"Epochs time range: {epochs.times.min()} to {epochs.times.max()}")
    print(f"TFR time range: {power.times.min()} to {power.times.max()}")
    print(f"Baseline indices count: {np.sum((times >= -500) & (times <= -200))}")
    print(f"Number of epochs after filtering: {len(data)}")

    if not baseline_correction:
        mean_power = np.nanmean(power_mean, axis=0)  # (n_freqs, n_times)
    
        return mean_power, times, freqs

    # ---- CHANGE 3: same baseline logic but applied to channel-averaged power ----
    if baseline_correction:
        if baseline_correction_method == 'single_trial':
            print("Using single_trial baseline correction method")
            if epoch_type.startswith('G'):
                # fixed baseline
                baseline_indices = (times >= -500) & (times <= -200)
                baseline_power = np.nanmean(power_mean[:, :, baseline_indices], axis=2, keepdims=True)
                change_single_trial = 10.0 * np.log10(power_mean / baseline_power)
                change = np.nanmean(change_single_trial, axis=0)  # (n_freqs, n_times)
            else:
                # variable baseline per trial
                if epoch_type == 'stop':
                    ssd_column = 'stop_signal_time'
                elif epoch_type == 'continue':
                    ssd_column = 'continue_signal_time'
                else:
                    raise ValueError(f"Unknown epoch type: {epoch_type}")

                baseline_start_per_trial = -500 - (np.array(data.metadata[ssd_column]) * 1000)
                baseline_end_per_trial = -200 - (np.array(data.metadata[ssd_column]) * 1000)

                change_single_trial = np.empty_like(power_mean)
                baseline_power = np.empty((power_mean.shape[0], power_mean.shape[1], 1))

                for i in range(power_mean.shape[0]):
                    bl_start = baseline_start_per_trial[i]
                    bl_end = baseline_end_per_trial[i]
                    bl_idx = (times >= bl_start) & (times <= bl_end)
                    bl_mean = np.nanmean(power_mean[i][:, bl_idx], axis=1, keepdims=True)
                    baseline_power[i] = bl_mean
                    change_single_trial[i] = 10.0 * np.log10(power_mean[i] / bl_mean)

                change = np.nanmean(change_single_trial, axis=0)

            print(f'change shape: {change.shape}')
            
            return change, times, freqs
        
        elif baseline_correction_method == 'group_average':
            print("Using group_average baseline correction method")
            mean_power = np.nanmean(power_mean, axis=0)  # (n_freqs, n_times)

            if epoch_type.startswith('G'):
                # Define baseline period for change calculation
                baseline_indices = (times >= -500) & (times <= -200)

                baseline_power = np.nanmean(mean_power[:, baseline_indices], axis=1, keepdims=True)  # shape: (n_freqs, 1 time)

                change = (mean_power - baseline_power) / baseline_power * 100  # percent change  # shape: (n_freqs, n_times)

            else:
                if epoch_type == 'stop': 
                    ssd_column = 'stop_signal_time'
                elif epoch_type == 'continue':
                    ssd_column = 'continue_signal_time'          

                baseline_start_per_trial = - 500 - (np.array(data.metadata[ssd_column]) * 1000)
                baseline_end_per_trial = - 200 - (np.array(data.metadata[ssd_column]) * 1000)

                change_single_trial = np.empty_like(power_mean)  # same shape : (n_trials, n_freqs, n_times)
                baseline_power = np.empty((power_mean.shape[0], power_mean.shape[1], 1))  # (n_trials, n_freqs, 1)

                for i in range(power_mean.shape[0]):  # loop over trials
                    # Get trial-specific baseline window
                    bl_start = baseline_start_per_trial[i]
                    bl_end   = baseline_end_per_trial[i]

                    # Find baseline indices in the common time axis
                    bl_idx = (times >= bl_start) & (times <= bl_end)

                    # Compute mean power in this window for all frequencies
                    bl_mean = np.nanmean(power_mean[i][ :, bl_idx], axis=1, keepdims=True)

                    # Store baseline and change
                    baseline_power[i] = bl_mean
                    #change_single_trial[i] =  (power_mean[i] - bl_mean) / bl_mean * 100
                
                baseline_power = np.nanmean(baseline_power, axis=0)  # shape: (n_freqs, 1 time)
                change = (mean_power - baseline_power) / baseline_power * 100  # percent change  # shape: (n_freqs, n_times)
                #change = np.nanmean(change_single_trial, axis=0)  # shape: (n_freqs, n_times)

            return change, times, freqs                  

# def eeg_get_change_from_baseline(
#         epochs,
#         cond,
#         ch_of_interest,
#         tfr_args,
#         baseline_correction: True,
# ):
#     epoch_type = cond.split('_')[0]
#     outcome_str = cond.split('_')[1]
#     outcome = 1.0 if outcome_str == 'successful' else 0.0

#     type_mask = epochs.metadata["event"] == epoch_type
#     outcome_mask = epochs.metadata["key_resp_experiment.corr"] == outcome
#     data = epochs[type_mask & outcome_mask] 

#     epochs = data.copy().pick([ch_of_interest])
    
#     power = epochs.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
#     print(f'power shape: {power.data.shape}')

#     power.data *= 1e12 # V² -> (µV)²

#     power_squeeze = power.data.squeeze() # shape: (n_trials, n_freqs, n_times)
#     print(f'power_squeeze shape: {power_squeeze.shape}')

#     times = power.times * 1000
#     freqs = power.freqs

#     print(f"Epochs time range: {epochs.times.min()} to {epochs.times.max()}")
#     print(f"TFR time range: {power.times.min()} to {power.times.max()}")
#     print(f"Baseline indices count: {np.sum((times >= -500) & (times <= -200))}")
#     print(f"Number of epochs after filtering: {len(data)}")

#     if not baseline_correction:
#         mean_power = np.nanmean(power_squeeze, axis=0).squeeze() # shape: (n freqs, n times)
#         return mean_power, times, freqs

#     elif baseline_correction:
#         if epoch_type.startswith('G'):
#             # Define baseline period for change calculation
#             baseline_indices = (times >= -500) & (times <= -200)

#             baseline_power = np.nanmean(power_squeeze[:, :, baseline_indices], axis=2, keepdims=True)  # shape: (n_trials, n_freqs, 1 time)
#             change_single_trial = 10.0 * np.log10(power_squeeze / baseline_power)

#             change = np.nanmean(change_single_trial, axis=0)  # shape: (n_freqs, n_times)

#         else: 
#             if epoch_type == 'stop': 
#                 ssd_column = 'stop_signal_time'
#             elif epoch_type == 'continue':
#                 ssd_column = 'continue_signal_time'

#             baseline_start_per_trial = - 500 - (np.array(data.metadata[ssd_column]) * 1000)
#             baseline_end_per_trial = - 200 - (np.array(data.metadata[ssd_column]) * 1000)

#             change_single_trial = np.empty_like(power_squeeze)  # same shape
#             baseline_power = np.empty((power_squeeze.shape[0], power_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

#             for i in range(power_squeeze.shape[0]):  # loop over trials
#                 # Get trial-specific baseline window
#                 bl_start = baseline_start_per_trial[i]
#                 bl_end   = baseline_end_per_trial[i]

#                 # Find baseline indices in the common time axis
#                 bl_idx = (times >= bl_start) & (times <= bl_end)

#                 # Compute mean power in this window for all frequencies
#                 bl_mean = np.nanmean(power_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

#                 # Store baseline and change
#                 baseline_power[i] = bl_mean
#                 change_single_trial[i] = 10.0 * np.log10(power_squeeze[i] / bl_mean)

#             change = np.nanmean(change_single_trial, axis=0)  # shape: (n_freqs, n_times)
#         print(f'change shape: {change.shape}')

#         return change, times, freqs


def get_change_from_baseline(
        epochs,
        cond,
        tfr_args,
        baseline_correction = True,
        baseline_correction_method = 'single_trial'
):
    latency_matched = False
            
    if cond == 'lm_GO_successful':
        latency_matched = True
        epoch_type = 'GO'
        outcome_str = 'successful'
        outcome = 1.0
    else:
        # Parse epoch condition
        epoch_type, outcome_str = cond.split('_')
        outcome = 1.0 if outcome_str == "successful" else 0.0

    type_mask = epochs.metadata["event"] == epoch_type
    outcome_mask = epochs.metadata["key_resp_experiment.corr"] == outcome
    data = epochs[type_mask & outcome_mask] 

    if latency_matched:
        print(f"{epoch_type}: before latency filter = {len(data)} epochs")
        rt = np.asarray(data.metadata['key_resp_experiment.rt'])
        print(f"RT shape = {rt.shape}, first few RTs = {rt[:5]}")
        threshold = np.percentile(rt, 50)
        print(f"Median RT = {threshold:.3f}s")
        slow_mask = rt >= threshold
        print(f"{slow_mask.sum()} epochs slower than median")
        # Ensure mask alignment to metadata
        data = data[slow_mask] 
    print("Channels in epochs:", data.ch_names)
    print("Left_STN in epochs?", 'Left_STN' in data.ch_names)
    print("Right_STN in epochs?", 'Right_STN' in data.ch_names)
    print("Number of epochs:", len(data))
    
    left_epochs, right_epochs = data.copy().pick(['Left_STN']), data.copy().pick(['Right_STN'])
    
    power_left = left_epochs.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
    power_right = right_epochs.compute_tfr(**tfr_args)

    power_left.data *= 1e12 # V² -> (µV)²
    power_right.data *= 1e12 # V² -> (µV)²

    power_left_squeeze = power_left.data.squeeze() # shape: (n_trials, n_freqs, n_times)
    power_right_squeeze = power_right.data.squeeze()

    times = power_left.times * 1000
    freqs = power_left.freqs

    print(f"Epochs time range: {epochs.times.min()} to {epochs.times.max()}")
    print(f"TFR time range: {power_left.times.min()} to {power_left.times.max()}")
    print(f"Baseline indices count: {np.sum((times >= -500) & (times <= -200))}")
    print(f"Number of epochs after filtering: {len(data)}")

    if not baseline_correction:
        mean_power_left = np.nanmean(power_left.data, axis=0).squeeze() # shape: (n freqs, n times)
        mean_power_right = np.nanmean(power_right.data, axis=0).squeeze()

        return mean_power_left, mean_power_right, times, freqs

    elif baseline_correction:
        if baseline_correction_method == 'single_trial':
            print("Using single_trial baseline correction method")
            if epoch_type.startswith('G'):
                # Define baseline period for change calculation
                baseline_indices = (times >= -500) & (times <= -200)

                baseline_power_left = np.nanmean(power_left_squeeze[:, :, baseline_indices], axis=2, keepdims=True)  # shape: (n_trials, n_freqs, 1 time)
                change_left_single_trial = 10.0 * np.log10(power_left_squeeze / baseline_power_left)

                baseline_power_right = np.nanmean(power_right_squeeze[:, :, baseline_indices], axis=2, keepdims=True)  # shape: (n_trials, n_freqs, 1 time)
                change_right_single_trial = 10.0 * np.log10(power_right_squeeze / baseline_power_right)

                change_left = np.nanmean(change_left_single_trial, axis=0)  # shape: (n_freqs, n_times)
                change_right = np.nanmean(change_right_single_trial, axis=0)  # shape: (n_freqs, n_times)

            else: 
                if epoch_type == 'stop': 
                    ssd_column = 'stop_signal_time'
                elif epoch_type == 'continue':
                    ssd_column = 'continue_signal_time'

                baseline_start_per_trial = - 500 - (np.array(data.metadata[ssd_column]) * 1000)
                baseline_end_per_trial = - 200 - (np.array(data.metadata[ssd_column]) * 1000)

                change_left_single_trial = np.empty_like(power_left_squeeze)  # same shape
                baseline_power_left = np.empty((power_left_squeeze.shape[0], power_left_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

                for i in range(power_left_squeeze.shape[0]):  # loop over trials
                    # Get trial-specific baseline window
                    bl_start = baseline_start_per_trial[i]
                    bl_end   = baseline_end_per_trial[i]

                    # Find baseline indices in the common time axis
                    bl_idx = (times >= bl_start) & (times <= bl_end)

                    # Compute mean power in this window for all frequencies
                    bl_mean = np.nanmean(power_left_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

                    # Store baseline and change
                    baseline_power_left[i] = bl_mean
                    change_left_single_trial[i] = 10.0 * np.log10(power_left_squeeze[i] / bl_mean)

                change_right_single_trial = np.empty_like(power_right_squeeze)  # same shape
                baseline_power_right = np.empty((power_right_squeeze.shape[0], power_right_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

                for i in range(power_right_squeeze.shape[0]):  # loop over trials
                    # Get trial-specific baseline window
                    bl_start = baseline_start_per_trial[i]
                    bl_end   = baseline_end_per_trial[i]

                    # Find baseline indices in the common time axis
                    bl_idx = (times >= bl_start) & (times <= bl_end)

                    # Compute mean power in this window for all frequencies
                    bl_mean = np.nanmean(power_right_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

                    # Store baseline and change
                    baseline_power_right[i] = bl_mean
                    change_right_single_trial[i] = 10.0 * np.log10(power_right_squeeze[i] / bl_mean)

                change_left = np.nanmean(change_left_single_trial, axis=0)  # shape: (n_freqs, n_times)
                change_right = np.nanmean(change_right_single_trial, axis=0)  # shape: (n_freqs, n_times)


            return change_left, change_right, times, freqs

        elif baseline_correction_method == 'group_average':
            print("Using group_average baseline correction method")
            mean_power_left = np.nanmean(power_left.data, axis=0).squeeze() # shape: (n freqs, n times)
            mean_power_right = np.nanmean(power_right.data, axis=0).squeeze()

            if epoch_type.startswith('G'):
                # Define baseline period for change calculation
                baseline_indices = (times >= -500) & (times <= -200)

                baseline_power_left = np.nanmean(mean_power_left[:, baseline_indices], axis=1, keepdims=True)  # shape: (n_freqs, 1 time)
                baseline_power_right = np.nanmean(mean_power_right[:, baseline_indices], axis=1, keepdims=True)  # shape: (n_freqs, 1 time)

                change_left = (mean_power_left - baseline_power_left) / baseline_power_left * 100  # percent change  # shape: (n_freqs, n_times)
                change_right = (mean_power_right - baseline_power_right) / baseline_power_right * 100  # percent change  # shape: (n_freqs, n_times)

            else: 
                if epoch_type == 'stop': 
                    ssd_column = 'stop_signal_time'
                elif epoch_type == 'continue':
                    ssd_column = 'continue_signal_time'

                baseline_start_per_trial = - 500 - (np.array(data.metadata[ssd_column]) * 1000)
                baseline_end_per_trial = - 200 - (np.array(data.metadata[ssd_column]) * 1000)

                change_left_single_trial = np.empty_like(power_left_squeeze)  # same shape
                baseline_power_left = np.empty((power_left_squeeze.shape[0], power_left_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

                for i in range(power_left_squeeze.shape[0]):  # loop over trials
                    # Get trial-specific baseline window
                    bl_start = baseline_start_per_trial[i]
                    bl_end   = baseline_end_per_trial[i]

                    # Find baseline indices in the common time axis
                    bl_idx = (times >= bl_start) & (times <= bl_end)

                    # Compute mean power in this window for all frequencies
                    bl_mean = np.nanmean(power_left_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

                    # Store baseline and change
                    baseline_power_left[i] = bl_mean
                    #change_left_single_trial[i] = 10.0 * np.log10(power_left_squeeze[i] / bl_mean)
                
                baseline_power_left = np.nanmean(baseline_power_left, axis=0)  # shape: (n_freqs, 1 time)
                change_left = (mean_power_left - baseline_power_left) / baseline_power_left * 100  # percent change  # shape: (n_freqs, n_times)

                change_right_single_trial = np.empty_like(power_right_squeeze)  # same shape
                baseline_power_right = np.empty((power_right_squeeze.shape[0], power_right_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

                for i in range(power_right_squeeze.shape[0]):  # loop over trials
                    # Get trial-specific baseline window
                    bl_start = baseline_start_per_trial[i]
                    bl_end   = baseline_end_per_trial[i]

                    # Find baseline indices in the common time axis
                    bl_idx = (times >= bl_start) & (times <= bl_end)

                    # Compute mean power in this window for all frequencies
                    bl_mean = np.nanmean(power_right_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

                    # Store baseline and change
                    baseline_power_right[i] = bl_mean
                    #change_right_single_trial[i] = 10.0 * np.log10(power_right_squeeze[i] / bl_mean)

                #change_left = np.nanmean(change_left_single_trial, axis=0)  # shape: (n_freqs, n_times)
                #change_right = np.nanmean(change_right_single_trial, axis=0)  # shape: (n_freqs, n_times)
                baseline_power_right = np.nanmean(baseline_power_right, axis=0)  # shape: (n_freqs, 1 time)
                change_right = (mean_power_right - baseline_power_right) / baseline_power_right * 100  # percent change  # shape: (n_freqs, n_times)

            return change_left, change_right, times, freqs

def compare_band_power(all_sub_session_dict, condition_a='DBS OFF', condition_b='DBS ON', metric='power_uV2'):
    """
    Compare band power between two conditions for each subject and hemisphere.
    
    Parameters
    ----------
    all_sub_session_dict : dict
        Nested dict: all_sub_session_dict[sub][condition][hemisphere].
    condition_a : str
        First condition (baseline).
    condition_b : str
        Second condition (comparison).
    metric : str
        Metric to compare ('power_uV2' or 'rms_uV').
        
    Returns
    -------
    df : pandas.DataFrame
        Columns: ['subject', 'hemisphere', 'band', 'cond_a', 'cond_b', 'diff', 'percent_change']
    """
    rows = []
    for sub, cond_dict in all_sub_session_dict.items():
        if condition_a not in cond_dict or condition_b not in cond_dict:
            continue
        for hemi in ['left', 'right']:
            if hemi not in cond_dict[condition_a] or hemi not in cond_dict[condition_b]:
                continue
            
            metrics_a = cond_dict[condition_a][hemi]
            metrics_b = cond_dict[condition_b][hemi]
            
            for band in metrics_a.keys():
                val_a = metrics_a[band][metric]
                val_b = metrics_b[band][metric]
                
                diff = val_b - val_a
                percent_change = (diff / val_a * 100) if val_a != 0 else float('nan')
                
                rows.append({
                    'subject': sub,
                    'hemisphere': hemi,
                    'band': band,
                    f'{condition_a}_{metric}': val_a,
                    f'{condition_b}_{metric}': val_b,
                    'diff': diff,
                    'percent_change': percent_change
                })
    
    df = pd.DataFrame(rows)
    return df



def compute_psd_welch(
        raw: mne.io.Raw
):
    n_fft = int(round(raw.info['sfreq']))
    n_overlap=int(round(raw.info['sfreq'])/2)

    L_chan = raw.get_data(picks=raw.ch_names[0])[0]
    R_chan = raw.get_data(picks=raw.ch_names[1])[0]

    #start = raw.info['sfreq'] * 200
    #end = raw.info['sfreq'] * 300

    #L_chan = L_chan_i[int(start):int(end)]
    #R_chan = R_chan_i[int(start):int(end)]

    psd_left, freqs_left = mne.time_frequency.psd_array_welch(
        L_chan,raw.info['sfreq'],fmin=0,
        fmax=125,n_fft=n_fft,
        n_overlap=n_overlap)
    psd_right, freqs_right = mne.time_frequency.psd_array_welch(
        R_chan,raw.info['sfreq'],fmin=0,
        fmax=125,n_fft=n_fft,
        n_overlap=n_overlap)
    # Calculate the frequency and time resolution possible based on the n_fft and noverlap parameters: 
    # freq_res = 1/(n_fft/sf) in Hz
    # Here we have a sf=250 so n_fft = 250 samples.
    # freq_res = 1/(250/250) = 1 Hz.
    # Then if there are overlapping segment (noverlap parameter), 
    # the time resolution corresponds to the nfft - noverlap size. 
    # So here, we have noverlap = 125 samples which is then 250-125 = 125 samples = 0.5 seconds.

    return psd_left, freqs_left, psd_right, freqs_right



def compute_percent_change(condition_epochs, ch, baseline, **tfr_kwargs):
    # ch = 0 for left STN, ch = 1 for right STN
    cond_epochs = condition_epochs.copy().pick([ch])

    # Compute TFR for both conditions
    tfr_epochs = cond_epochs.compute_tfr(**tfr_kwargs)

    # Apply baseline correction
    percent_change = tfr_epochs.apply_baseline(mode="percent", baseline=baseline)*100

    # Compute average power for each condition
    avg_power = np.nanmean(percent_change.data, axis=0)[0]

    times = tfr_epochs.times * 1000
    time_indices = (times >= -500) & (times <= 1500)
    times = times[time_indices]
    avg_power = avg_power[:, time_indices]    

    return percent_change, avg_power, times


def identify_significant_clusters(
        cluster_p_values,
        clusters,
        times,
        T_obs,
        pval, 
        tfr_args,
        condition,
        roi,
        saving_path
        ):
    results = []
    # create key to store the results dynamically:
    key = f"{condition}_{roi}"
    approach_sig_idx = np.array([])
    # Identify significant clusters
    significant_cluster_idx = np.where(cluster_p_values < pval)[0]        
    approach_sig_idx = np.where((cluster_p_values >= pval) & (cluster_p_values <= 0.1))[0]
    if approach_sig_idx.size > 0:
        print(f"Clusters approaching signifiance: {len(approach_sig_idx)}\n")
        for cluster_idx in approach_sig_idx:
            # cluster_mask = clusters[cluster_idx]
            # freq_indices = np.where(cluster_mask.sum(axis=1) > 0)[0]
            # sig_freqs = tfr_args["freqs"][freq_indices]
            # print(f"Cluster approaching significance: cluster {cluster_idx + 1}: Frequencies involved - {sig_freqs}, p-value = {cluster_p_values[cluster_idx]:.4f}")
            
            # results.append({
            #     "Cluster": cluster_idx + 1,
            #     "Frequencies involved": sig_freqs,
            #     "Peak Freq (Hz)": peak_freq,
            #     "Peak Time (ms)": peak_time,
            #     "Peak T-Stat": peak_T_stat,
            #     "Pixel Size": cluster_size,
            #     "P_value": cluster_p_values[cluster_idx]
            # })
            cluster_mask = clusters[cluster_idx]  # Boolean mask (freq, time)

            # Get indices of significant points
            sig_freqs, sig_times = np.where(cluster_mask)
            
            # Map indices to actual frequency and time values
            sig_freq_values = tfr_args["freqs"][sig_freqs]  # 'freqs' should be your array of frequency values
            sig_time_values = times[sig_times]  # 'times' should be your time array in ms
            
            # Extract T-stat values within the cluster
            T_vals_in_cluster = T_obs[cluster_mask]  
            
            # Find the peak T-stat (largest absolute value)
            peak_T_stat = np.max(np.abs(T_vals_in_cluster))
            
            # Find index of this peak in cluster
            peak_idx = np.argmax(np.abs(T_vals_in_cluster))
            
            # Get corresponding peak frequency and time
            peak_freq = sig_freq_values[peak_idx]
            peak_time = sig_time_values[peak_idx]

            # Calculate cluster size (number of significant pixels)
            cluster_size = np.sum(cluster_mask)
            
            results.append({
                "Condition": f'{condition} {roi} STN',
                "Cluster": cluster_idx + 1,
                "Frequencies involved": np.unique(sig_freq_values),
                "Peak Freq (Hz)": peak_freq,
                "Peak Time (ms)": peak_time,
                "Peak T-Stat": peak_T_stat,
                "Pixel Size": cluster_size,
                "P_value": cluster_p_values[cluster_idx]
            })
        # Print results
        for res in results:
            print(f"Cluster {res['Cluster']}: Peak Freq = {res['Peak Freq (Hz)']} Hz, "
                f"Peak Time = {res['Peak Time (ms)']} ms, Peak T-Stat = {res['Peak T-Stat']:.2f}, "
                f"Pixel Size = {res['Pixel Size']} pixels, P_value = {res['P_value']:.4f}")
            
    if significant_cluster_idx.size > 0:
        print(f"\nSignificant clusters found: {len(significant_cluster_idx)}\n")
        
        for cluster_idx in significant_cluster_idx:
            cluster_mask = clusters[cluster_idx]  # Boolean mask (freq, time)

            # Get indices of significant points
            sig_freqs, sig_times = np.where(cluster_mask)
            
            # Map indices to actual frequency and time values
            sig_freq_values = tfr_args["freqs"][sig_freqs]  # 'freqs' should be your array of frequency values
            sig_time_values = times[sig_times]  # 'times' should be your time array in ms
            
            # Extract T-stat values within the cluster
            T_vals_in_cluster = T_obs[cluster_mask]  
            
            # Find the peak T-stat (largest absolute value)
            peak_T_stat = np.max(np.abs(T_vals_in_cluster))
            
            # Find index of this peak in cluster
            peak_idx = np.argmax(np.abs(T_vals_in_cluster))
            
            # Get corresponding peak frequency and time
            peak_freq = sig_freq_values[peak_idx]
            peak_time = sig_time_values[peak_idx]

            # Calculate cluster size (number of significant pixels)
            cluster_size = np.sum(cluster_mask)
            
            results.append({
                "Condition": f'{condition} {roi}',
                "Cluster": cluster_idx + 1,
                "Frequencies involved": np.unique(sig_freq_values),
                "Peak Freq (Hz)": peak_freq,
                "Peak Time (ms)": peak_time,
                "Peak T-Stat": peak_T_stat,
                "Pixel Size": cluster_size,
                "P_value": cluster_p_values[cluster_idx]
            })

        # Print results
        for res in results:
            print(f"Cluster {res['Cluster']}: Peak Freq = {res['Peak Freq (Hz)']} Hz, "
                f"Peak Time = {res['Peak Time (ms)']} ms, Peak T-Stat = {res['Peak T-Stat']:.2f}, "
                f"Pixel Size = {res['Pixel Size']} pixels, P_value = {res['P_value']:.4f}")

    else:
        print("\nNo significant clusters found\n")
        results.append({
            "Condition": f'{condition} {roi}',
            "Cluster": None,
            "Frequencies involved": None,
            "Peak Freq (Hz)": None,
            "Peak Time (ms)": None,
            "Peak T-Stat": None,
            "Pixel Size": None,
            "P_value": None
        })
    

    # save results to Excel and csv
    df = pd.DataFrame(results)
    name = f"cluster_results_{condition}_{roi}"

    excel_path_main = os.path.join(saving_path, 'Clusters excel')
    if not os.path.exists(excel_path_main):
        os.makedirs(excel_path_main)
    excel_path = os.path.join(excel_path_main, f"{name}.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Cluster results saved to {excel_path}")

    csv_path_main = os.path.join(saving_path, 'Clusters csv')
    if not os.path.exists(csv_path_main):
        os.makedirs(csv_path_main)
    csv_path = os.path.join(csv_path_main, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Cluster results saved to {csv_path}")