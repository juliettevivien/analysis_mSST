import mne
import numpy as np


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
        cluster_results_dict,
        condition,
        side
        ):
    
    # create key to store the results dynamically:
    key = f"{condition}_{side}"
    # Identify significant clusters
    significant_cluster_idx = np.where(cluster_p_values < pval)[0]
    if not significant_cluster_idx.size > 0:
        print("\nNo significant clusters found\n")
        cluster_results_dict[key] = "No significant cluster"
        
        approach_sig_idx = np.where(cluster_p_values < 0.075)[0]
        if approach_sig_idx.size > 0:
            print(f"Clusters approaching signifiance: {len(approach_sig_idx)}\n")
            results_clust = f"No significant cluster but {len(approach_sig_idx)} clusters approaching significance"
            cluster_results_dict[key] = results_clust
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

            cluster_results_dict[key] = {
                "Cluster": cluster_idx + 1,
                "Frequencies involved": sig_freq_values,
                "Peak Freq (Hz)": peak_freq,
                "Peak Time (ms)": peak_time,
                "Peak T-Stat": peak_T_stat,
                "Pixel Size": cluster_size
            }
        # Print results
        for res in results:
            print(f"Cluster {res['Cluster']}: Peak Freq = {res['Peak Freq (Hz)']} Hz, "
                f"Peak Time = {res['Peak Time (ms)']} ms, Peak T-Stat = {res['Peak T-Stat']:.2f}, "
                f"Pixel Size = {res['Pixel Size']} pixels")    
            
    return cluster_results_dict