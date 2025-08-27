import mne
import numpy as np
import pandas as pd

def get_change_from_baseline(
        epochs,
        cond,
        tfr_args,
        baseline_correction: True,
):
    epoch_type = cond.split('_')[0]
    outcome_str = cond.split('_')[1]
    outcome = 1.0 if outcome_str == 'successful' else 0.0

    type_mask = epochs.metadata["event"] == epoch_type
    outcome_mask = epochs.metadata["key_resp_experiment.corr"] == outcome
    data = epochs[type_mask & outcome_mask] 

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
        #cluster_results_dict,
        condition,
        side
        ):
    
    # create key to store the results dynamically:
    key = f"{condition}_{side}"
    # Identify significant clusters
    significant_cluster_idx = np.where(cluster_p_values < pval)[0]
    if not significant_cluster_idx.size > 0:
        print("\nNo significant clusters found\n")
        #cluster_results_dict[key] = "No significant cluster"
        
        approach_sig_idx = np.where(cluster_p_values < 0.075)[0]
        if approach_sig_idx.size > 0:
            print(f"Clusters approaching signifiance: {len(approach_sig_idx)}\n")
            results_clust = f"No significant cluster but {len(approach_sig_idx)} clusters approaching significance"
            #cluster_results_dict[key] = results_clust
    else:
        print(f"\nSignificant clusters found: {len(significant_cluster_idx)}\n")

        for cluster_idx in significant_cluster_idx:
            cluster_mask = clusters[cluster_idx]  # Boolean mask (freq, time)
            
            # Extract the frequency indices where the cluster is present
            freq_indices = np.where(cluster_mask.sum(axis=1) > 0)[0]
            
            # Map indices to actual frequency values
            sig_freqs = tfr_args["freqs"][freq_indices]  # 'freqs' should be your frequency array
            
            print(f"Cluster {cluster_idx + 1}: Frequencies involved - {sig_freqs}")

        results = []
    
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
                "Cluster": cluster_idx + 1,
                "Peak Freq (Hz)": peak_freq,
                "Peak Time (ms)": peak_time,
                "Peak T-Stat": peak_T_stat,
                "Pixel Size": cluster_size
            })

            # cluster_results_dict[key] = {
            #     "Cluster": cluster_idx + 1,
            #     "Frequencies involved": sig_freq_values,
            #     "Peak Freq (Hz)": peak_freq,
            #     "Peak Time (ms)": peak_time,
            #     "Peak T-Stat": peak_T_stat,
            #     "Pixel Size": cluster_size
            # }
        # Print results
        for res in results:
            print(f"Cluster {res['Cluster']}: Peak Freq = {res['Peak Freq (Hz)']} Hz, "
                f"Peak Time = {res['Peak Time (ms)']} ms, Peak T-Stat = {res['Peak T-Stat']:.2f}, "
                f"Pixel Size = {res['Pixel Size']} pixels")    
            
    #return cluster_results_dict