"""
This file should contain all the functions that are used to plot the data.

"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import mne
import scipy
import json
from matplotlib.gridspec import GridSpec
from os.path import join
#from mne.stats import permutation_cluster_test

from mne.baseline import rescale
from mne.stats import bootstrap_confidence_interval
import os
from functions.utils import stat_fun
from functions.stats_tests import perform_permutation_cluster_test
from functions.analysis import compute_percent_change, identify_significant_clusters, get_change_from_baseline



def perc_pow_diff_on_off_contrast(
        sub_dict_ON_OFF,
        tfr_args,
        cond1,
        cond2,
        t_min_max: list,
        vmin_vmax: list,
        condition: str,
        saving_path: str = None,
        show_fig: bool = None,
        add_rt: bool = True,
        save_as: str = 'png'
        ):
    
    all_sub_RT_ON_cond1 = []
    all_sub_RT_ON_cond2 = []
    all_sub_RT_OFF_cond1 = []
    all_sub_RT_OFF_cond2 = []
    all_diff_left = []
    all_diff_right = []
    all_diff_both = []
    subs_included = []

    epoch_type1 = cond1.split('_')[0]
    epoch_type2 = cond2.split('_')[0]
    outcome_str1 = cond1.split('_')[1] 
    outcome_str2 = cond2.split('_')[1]

    outcome1 = 1.0 if outcome_str1 == 'successful' else 0.0
    outcome2 = 1.0 if outcome_str2 == 'successful' else 0.0

    # Collect epoch data for each condition
    for sub in sub_dict_ON_OFF.keys():
        print("Now processing: ", sub, condition)
        for subject, epochs in sub_dict_ON_OFF[sub].items():
            print(subject)
            if 'DBS ON' in subject:
                type_mask1 = epochs.metadata["event"] == epoch_type1
                outcome_mask1 = epochs.metadata["key_resp_experiment.corr"] == outcome1
                data_ON_cond1 = epochs[type_mask1 & outcome_mask1]   
            
                type_mask2 = epochs.metadata["event"] == epoch_type2
                outcome_mask2 = epochs.metadata["key_resp_experiment.corr"] == outcome2
                data_ON_cond2 = epochs[type_mask2 & outcome_mask2]

                if not (cond1 == 'GS_successful' or cond1 == 'stop_successful'):
                    RT_ON_cond1 = data_ON_cond1.metadata['key_resp_experiment.rt'].mean() * 1000
                    all_sub_RT_ON_cond1.append(RT_ON_cond1)
                    print(RT_ON_cond1)
                if not (cond2 == 'GS_successful' or cond2 == 'stop_successful'):
                    RT_ON_cond2 = data_ON_cond2.metadata['key_resp_experiment.rt'].mean() * 1000
                    all_sub_RT_ON_cond2.append(RT_ON_cond2)
                    print(RT_ON_cond2)

                (percentage_change_left_ON_cond1, 
                 percentage_change_right_ON_cond1,
                    times, freqs) = get_change_from_baseline(
                        epochs = epochs,
                        cond = cond1,
                        tfr_args = tfr_args,
                        baseline_correction = True
                    )

                (percentage_change_left_ON_cond2, 
                 percentage_change_right_ON_cond2,
                    times, freqs) = get_change_from_baseline(
                        epochs = epochs,
                        cond = cond2,
                        tfr_args = tfr_args,
                        baseline_correction = True
                    )

                # left_epochs_ON_cond1, right_epochs_ON_cond1 = data_ON_cond1.copy().pick(['Left_STN']), data_ON_cond1.copy().pick(['Right_STN'])
                # power_left_ON_cond1 = left_epochs_ON_cond1.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
                # power_right_ON_cond1 = right_epochs_ON_cond1.compute_tfr(**tfr_args)

                # mean_power_left_ON_cond1 = np.nanmean(power_left_ON_cond1.data, axis=0).squeeze() # shape: (n freqs, n times)
                # mean_power_right_ON_cond1 = np.nanmean(power_right_ON_cond1.data, axis=0).squeeze()
                
                # left_epochs_ON_cond2, right_epochs_ON_cond2 = data_ON_cond2.copy().pick(['Left_STN']), data_ON_cond2.copy().pick(['Right_STN'])
                # power_left_ON_cond2 = left_epochs_ON_cond2.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
                # power_right_ON_cond2 = right_epochs_ON_cond2.compute_tfr(**tfr_args)

                # mean_power_left_ON_cond2 = np.nanmean(power_left_ON_cond2.data, axis=0).squeeze() # shape: (n freqs, n times)
                # mean_power_right_ON_cond2 = np.nanmean(power_right_ON_cond2.data, axis=0).squeeze()

                # times = power_left_ON_cond1.times * 1000
                # freqs = power_left_ON_cond1.freqs

                # # Define baseline period for percentage change calculation
                # baseline_indices = (times >= -500) & (times <= -200)

                # # Percentage change for condition 1
                # baseline_power_left_ON_cond1 = np.nanmean(mean_power_left_ON_cond1[:, baseline_indices], axis=1, keepdims=True)
                # percentage_change_left_ON_cond1 = (mean_power_left_ON_cond1 - baseline_power_left_ON_cond1) / baseline_power_left_ON_cond1 * 100
                # baseline_power_right_ON_cond1 = np.nanmean(mean_power_right_ON_cond1[:, baseline_indices], axis=1, keepdims=True)
                # percentage_change_right_ON_cond1 = (mean_power_right_ON_cond1 - baseline_power_right_ON_cond1) / baseline_power_right_ON_cond1 * 100
        
                # baseline_power_left_ON_cond2 = np.nanmean(mean_power_left_ON_cond2[:, baseline_indices], axis=1, keepdims=True)
                # percentage_change_left_ON_cond2 = (mean_power_left_ON_cond2 - baseline_power_left_ON_cond2) / baseline_power_left_ON_cond2 * 100
                # baseline_power_right_ON_cond2 = np.nanmean(mean_power_right_ON_cond2[:, baseline_indices], axis=1, keepdims=True)
                # percentage_change_right_ON_cond2 = (mean_power_right_ON_cond2 - baseline_power_right_ON_cond2) / baseline_power_right_ON_cond2 * 100
        
                percentage_change_left_ON_contrast = percentage_change_left_ON_cond1 - percentage_change_left_ON_cond2
                percentage_change_right_ON_contrast = percentage_change_right_ON_cond1 - percentage_change_right_ON_cond2

            if 'DBS OFF' in subject:
                type_mask1 = epochs.metadata["event"] == epoch_type1
                outcome_mask1 = epochs.metadata["key_resp_experiment.corr"] == outcome1
                data_OFF_cond1 = epochs[type_mask1 & outcome_mask1]   
            
                type_mask2 = epochs.metadata["event"] == epoch_type2
                outcome_mask2 = epochs.metadata["key_resp_experiment.corr"] == outcome2
                data_OFF_cond2 = epochs[type_mask2 & outcome_mask2]          

                if not (cond1 == 'GS_successful' or cond1 == 'stop_successful'):
                    RT_OFF_cond1 = data_OFF_cond1.metadata['key_resp_experiment.rt'].mean() * 1000
                    all_sub_RT_OFF_cond1.append(RT_OFF_cond1)
                    print(RT_OFF_cond1)

                if not (cond1 == 'GS_successful' or cond2 == 'stop_successful'):
                    RT_OFF_cond2 = data_OFF_cond2.metadata['key_resp_experiment.rt'].mean() * 1000
                    all_sub_RT_OFF_cond2.append(RT_OFF_cond2)

                (percentage_change_left_OFF_cond1,
                percentage_change_right_OFF_cond1,
                times, freqs) = get_change_from_baseline(
                        epochs = epochs,
                        cond = cond1,
                        tfr_args = tfr_args,
                        baseline_correction = True
                    )

                (percentage_change_left_OFF_cond2, 
                 percentage_change_right_OFF_cond2,
                    times, freqs) = get_change_from_baseline(
                        epochs = epochs,
                        cond = cond2,
                        tfr_args = tfr_args,
                        baseline_correction = True
                    )

                # left_epochs_OFF_cond1, right_epochs_OFF_cond1 = data_OFF_cond1.copy().pick(['Left_STN']), data_OFF_cond1.copy().pick(['Right_STN'])
                # power_left_OFF_cond1 = left_epochs_OFF_cond1.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
                # power_right_OFF_cond1 = right_epochs_OFF_cond1.compute_tfr(**tfr_args)

                # mean_power_left_OFF_cond1 = np.nanmean(power_left_OFF_cond1.data, axis=0).squeeze() # shape: (n freqs, n times)
                # mean_power_right_OFF_cond1 = np.nanmean(power_right_OFF_cond1.data, axis=0).squeeze()
                
                # left_epochs_OFF_cond2, right_epochs_OFF_cond2 = data_OFF_cond2.copy().pick(['Left_STN']), data_OFF_cond2.copy().pick(['Right_STN'])
                # power_left_OFF_cond2 = left_epochs_OFF_cond2.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
                # power_right_OFF_cond2 = right_epochs_OFF_cond2.compute_tfr(**tfr_args)

                # mean_power_left_OFF_cond2 = np.nanmean(power_left_OFF_cond2.data, axis=0).squeeze() # shape: (n freqs, n times)
                # mean_power_right_OFF_cond2 = np.nanmean(power_right_OFF_cond2.data, axis=0).squeeze()

                # times = power_left_OFF_cond1.times * 1000
                # freqs = power_left_OFF_cond1.freqs

                # # Define baseline period for percentage change calculation
                # baseline_indices = (times >= -500) & (times <= -200)

                # # Percentage change for condition 1
                # baseline_power_left_OFF_cond1 = np.nanmean(mean_power_left_OFF_cond1[:, baseline_indices], axis=1, keepdims=True)
                # percentage_change_left_OFF_cond1 = (mean_power_left_OFF_cond1 - baseline_power_left_OFF_cond1) / baseline_power_left_OFF_cond1 * 100

                # baseline_power_right_OFF_cond1 = np.nanmean(mean_power_right_OFF_cond1[:, baseline_indices], axis=1, keepdims=True)
                # percentage_change_right_OFF_cond1 = (mean_power_right_OFF_cond1 - baseline_power_right_OFF_cond1) / baseline_power_right_OFF_cond1 * 100
        
                # # Percentage change for condition 2
                # baseline_power_left_OFF_cond2 = np.nanmean(mean_power_left_OFF_cond2[:, baseline_indices], axis=1, keepdims=True)
                # percentage_change_left_OFF_cond2 = (mean_power_left_OFF_cond2 - baseline_power_left_OFF_cond2) / baseline_power_left_OFF_cond2 * 100

                # baseline_power_right_OFF_cond2 = np.nanmean(mean_power_right_OFF_cond2[:, baseline_indices], axis=1, keepdims=True)
                # percentage_change_right_OFF_cond2 = (mean_power_right_OFF_cond2 - baseline_power_right_OFF_cond2) / baseline_power_right_OFF_cond2 * 100
    
                percentage_change_left_OFF_contrast = percentage_change_left_OFF_cond1 - percentage_change_left_OFF_cond2
                percentage_change_right_OFF_contrast = percentage_change_right_OFF_cond1 - percentage_change_right_OFF_cond2


        diff_left = percentage_change_left_ON_contrast - percentage_change_left_OFF_contrast
        diff_right = percentage_change_right_ON_contrast - percentage_change_right_OFF_contrast
        all_diff_left.append(diff_left)
        all_diff_right.append(diff_right)
        all_diff_both.extend([diff_left, diff_right])

        subs_included.append(sub)

    print(f'Subs included in analyses: \n {subs_included}')

    all_diff_left_array = np.array(all_diff_left)  # shape: (n sub, n freqs, n times)
    all_diff_right_array = np.array(all_diff_right)
    all_diff_both_array = np.array(all_diff_both) # shape: (n sub x 2, n freqs, n times)

    time_indices = (times >= t_min_max[0]) & (times <= t_min_max[1])
    sliced_times = times[time_indices]
    all_diff_left_array_sliced = all_diff_left_array[:,:,time_indices]
    all_diff_right_array_sliced = all_diff_right_array[:,:,time_indices]
    all_diff_both_array_sliced = all_diff_both_array[:,:,time_indices]


    if len(subs_included) > 1:
        n_obs = all_diff_left_array.shape[0]
        print(n_obs)
        pval = 0.05
        df = n_obs - 1
        threshold = scipy.stats.t.ppf(1-pval / 2, df) # two-tailed distribution
        #threshold = None
        n_permutations = 1000


        # Compute permutation cluster test for the left stn
        T_obs_left, clusters_left, cluster_p_values_left, H0_left = mne.stats.permutation_cluster_1samp_test(
        all_diff_left_array_sliced, n_permutations=n_permutations,
        threshold=threshold, tail=0,
        out_type= "mask", seed=11111, verbose=True)
        print(f"p_values: {cluster_p_values_left}")
        print(f"P_values shape: {cluster_p_values_left.shape}")

        print("Clusters for Left STN")
        identify_significant_clusters(
            cluster_p_values_left, 
            clusters_left,
            sliced_times,
            T_obs_left,
            pval,
            tfr_args,
            condition,
            'Left'
            )
        
        # Compute permutation cluster test for the right stn
        T_obs_right, clusters_right, cluster_p_values_right, H0_right = mne.stats.permutation_cluster_1samp_test(
        all_diff_right_array_sliced, n_permutations=n_permutations,
        threshold=threshold, tail=0,
        out_type= "mask", seed=11111, verbose=True)
        print(f"p_values: {cluster_p_values_right}")
        print(f"P_values shape: {cluster_p_values_right.shape}")

        print("Clusters for Right STN")
        identify_significant_clusters(
            cluster_p_values_right, 
            clusters_right,
            sliced_times,
            T_obs_right,
            pval,
            tfr_args,
            condition,
            'Right'
            )

        # Compute permutation cluster test for the left + right stn
        T_obs_both, clusters_both, cluster_p_values_both, H0_both = mne.stats.permutation_cluster_1samp_test(
        all_diff_both_array_sliced, n_permutations=n_permutations,
        threshold=threshold, tail=0,
        out_type= "mask", seed=11111, verbose=True)
        print(f"p_values: {cluster_p_values_both}")
        print(f"P_values shape: {cluster_p_values_both.shape}")

        print("Clusters for Both STN")
        identify_significant_clusters(
            cluster_p_values_both, 
            clusters_both,
            sliced_times,
            T_obs_both,
            pval,
            tfr_args,
            condition,
            'Both'
            )


    # Average the percentage signal changes across subjects for left STN and for right STN
    avg_diff_left = np.nanmean(all_diff_left_array_sliced, axis=0)  # shape: (n freqs, n times)
    avg_diff_right = np.nanmean(all_diff_right_array_sliced, axis=0)
    avg_diff_both = np.nanmean(all_diff_both_array_sliced, axis=0)


    ################
    ### PLOTTING ###
    ################    

    # Create a figure with two subplots for Left and Right STN
    fig, (ax_left, ax_right, ax_both) = plt.subplots(1, 3, figsize=(20, 8))

    # Figure title for n_subjects
    sub_num = len(all_diff_left)

    if sub_num > 1:
        fig.suptitle(f"Power difference DBS ON - DBS OFF {condition}, nSub = {sub_num}")
    else:
        fig.suptitle(f"Power difference DBS ON - DBS OFF {condition}, {subject[:6]}")


    # Plot the percentage change difference for Left STN
    im_left = ax_left.imshow(avg_diff_left, aspect='auto', origin='lower', 
                            extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    ax_left.set_xlabel('Time from GO cue (ms)')
    ax_left.set_ylabel('Frequency (Hz)')
    ax_left.set_title(f'Left STN - {condition}')
    fig.colorbar(im_left, ax=ax_left, label='Mean % Change (from baseline)')

    # Plot the percentage change difference for Right STN
    im_right = ax_right.imshow(avg_diff_right, aspect='auto', origin='lower', 
                            extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    ax_right.set_xlabel('Time from GO cue (ms)')
    ax_right.set_ylabel('Frequency (Hz)')
    ax_right.set_title(f'Right STN - {condition}')
    fig.colorbar(im_right, ax=ax_right, label='Mean % Change (from baseline)')

    # Plot the percentage change difference for Left + Right STN
    im_both = ax_both.imshow(avg_diff_both, aspect='auto', origin='lower', 
                            extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    ax_both.set_xlabel('Time from GO cue (ms)')
    ax_both.set_ylabel('Frequency (Hz)')
    ax_both.set_title(f'Left + Right STN - {condition}')
    fig.colorbar(im_both, ax=ax_both, label='Mean % Change (from baseline)')

    print(f" {cond1} RT ON: {all_sub_RT_ON_cond1}")
    print(f" {cond1} RT OFF: {all_sub_RT_OFF_cond1}")
    print(f" {cond2} RT ON: {all_sub_RT_ON_cond2}")
    print(f" {cond2} RT OFF: {all_sub_RT_OFF_cond2}")

    if len(subs_included) > 1:
        for c, p_val in zip(clusters_left, cluster_p_values_left):
            if p_val <= pval:
                mask_L = np.zeros_like(T_obs_left, dtype=bool) 
                mask_L[c] = True
                ax_left.contour(mask_L, levels=[0.5], colors='black', linewidths=1.5,
                                extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
        
        
        for c, p_val in zip(clusters_right, cluster_p_values_right):
            if p_val <= pval:
                mask_R = np.zeros_like(T_obs_right, dtype=bool)
                mask_R[c] = True
                ax_right.contour(mask_R, levels=[0.5], colors='black', linewidths=1.5,
                                extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
        
        
        for c, p_val in zip(clusters_both, cluster_p_values_both):
            if p_val <= pval:
                mask_R = np.zeros_like(T_obs_both, dtype=bool)
                mask_R[c] = True
                ax_both.contour(mask_R, levels=[0.5], colors='black', linewidths=1.5,
                                extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
        
        mean_RT_ON_cond1 = np.mean(all_sub_RT_ON_cond1)
        mean_RT_ON_cond2 = np.mean(all_sub_RT_ON_cond2)
        mean_RT_OFF_cond1 = np.mean(all_sub_RT_OFF_cond1)
        mean_RT_OFF_cond2 = np.mean(all_sub_RT_OFF_cond2)
        #mean_SSD = np.mean(all_sub_SSD)
        #mean_SSRT = np.mean(all_sub_SSRT)
    #else:
        # mean_RT_ON = all_sub_RT_ON[0]
        # mean_RT_OFF = all_sub_RT_OFF[0]
        #mean_SSD = all_sub_SSD[0]
        #mean_SSRT = all_sub_SSRT[0]
    
    ax_left.axvline(mean_RT_ON_cond1, color='black', linestyle='--')
    ax_right.axvline(mean_RT_ON_cond1, color='black', linestyle='--')
    ax_both.axvline(mean_RT_ON_cond1, color='black', linestyle='--', label=f'Mean RT ON {cond1}')
    ax_left.axvline(mean_RT_OFF_cond1, color='grey', linestyle='--')
    ax_right.axvline(mean_RT_OFF_cond1, color='grey', linestyle='--')
    ax_both.axvline(mean_RT_OFF_cond1, color='grey', linestyle='--', label=f'Mean RT OFF {cond1}')

    ax_left.axvline(mean_RT_ON_cond2, color='blue', linestyle='--')
    ax_right.axvline(mean_RT_ON_cond2, color='blue', linestyle='--')
    ax_both.axvline(mean_RT_ON_cond2, color='blue', linestyle='--', label=f'Mean RT ON {cond2}')
    ax_left.axvline(mean_RT_OFF_cond2, color='green', linestyle='--')
    ax_right.axvline(mean_RT_OFF_cond2, color='green', linestyle='--')
    ax_both.axvline(mean_RT_OFF_cond2, color='green', linestyle='--', label=f'Mean RT OFF {cond2}')

    #ax_left.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')
    #ax_right.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')    
    #ax_both.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')    
    #ax_left.axvline(mean_SSRT, color='blue', linestyle='--', label='Mean SSRT')
    #ax_right.axvline(mean_SSRT, color='blue', linestyle='--', label='Mean SSRT')
    #ax_both.axvline(mean_SSRT, color='blue', linestyle='--', label='Mean SSRT')    

    fig.legend() 

    if sub_num > 1:
        figtitle = f"Power_diff_DBS_ON-DBS_OFF_{condition}.png"
        figtitle_pdf = f"Power_diff_DBS_ON-DBS_OFF_{condition}.pdf"
    else:
        figtitle = f"{subject[:6]}_Power_diff_DBS_ON-DBS_OFF_{condition}.png"
        figtitle_pdf = f"{subject[:6]}_Power_diff_DBS_ON-DBS_OFF_{condition}.pdf"

    if saving_path is not None:
        if save_as == 'png':
            plt.savefig(join(saving_path, figtitle), transparent=False)
        else:
            plt.savefig(join(saving_path, figtitle_pdf), transparent=False)

    if show_fig == True:
        plt.show()
    else:
        plt.close('all')



def perc_pow_diff_on_off(
        sub_dict_ON_OFF,
        tfr_args,
        cond,
        t_min_max: list,
        vmin_vmax: list,
        condition: str,
        saving_path: str = None,
        show_fig: bool = None,
        save_as: str = 'png',
        baseline_correction: bool=True
        ):
    
    epoch_type = cond.split('_')[0]
    outcome_str = cond.split('_')[1]    
    outcome = 1.0 if outcome_str == "successful" else 0.0

    all_sub_RT_ON = []
    all_sub_RT_OFF = []
    all_diff_left = []
    all_diff_right = []
    all_diff_both = []
    subs_included = []

    RT_plot = False

    # Collect epoch data for each condition
    for sub in sub_dict_ON_OFF.keys(): 
        print("Now processing: ", sub, cond)
        for subject, epochs in sub_dict_ON_OFF[sub].items():
            print(subject)
            if 'DBS ON' in subject:
                type_mask = epochs.metadata["event"] == epoch_type
                outcome_mask = epochs.metadata["key_resp_experiment.corr"] == outcome
                data_ON = epochs[type_mask & outcome_mask]                

                if not (cond == 'GS_successful' or cond == 'stop_successful'):
                    rt_on = data_ON.metadata['key_resp_experiment.rt'].mean() * 1000 
                    all_sub_RT_ON.append(rt_on)
                    RT_plot = True
                    print(rt_on)

                (percentage_change_left_ON, 
                 percentage_change_right_ON, 
                 times, freqs) = get_change_from_baseline(
                    epochs = epochs,
                    cond = cond,
                    tfr_args = tfr_args,
                    baseline_correction = baseline_correction
                 )

                # left_epochs_ON, right_epochs_ON = data_ON.copy().pick(['Left_STN']), data_ON.copy().pick(['Right_STN'])
                
                # power_left_ON = left_epochs_ON.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
                # power_right_ON = right_epochs_ON.compute_tfr(**tfr_args)

                # power_left_ON.data *= 1e12 # V² -> (µV)²
                # power_right_ON.data *= 1e12 # V² -> (µV)²

                # power_left_ON_squeeze = power_left_ON.data.squeeze() # shape: (n_trials, n_freqs, n_times)
                # power_right_ON_squeeze = power_right_ON.data.squeeze()

                # mean_power_left_ON = np.nanmean(power_left_ON.data, axis=0).squeeze() # shape: (n freqs, n times)
                # mean_power_right_ON = np.nanmean(power_right_ON.data, axis=0).squeeze()
                
                # times = power_left_ON.times * 1000
                # freqs = power_left_ON.freqs

                # # # Define baseline period for percentage change calculation
                # # baseline_indices = (times >= -500) & (times <= -200)

                # # # Percentage change for condition ON
                # # baseline_power_left_ON = np.nanmean(mean_power_left_ON[:, baseline_indices], axis=1, keepdims=True)
                # # percentage_change_left_ON = (mean_power_left_ON - baseline_power_left_ON) / baseline_power_left_ON * 100

                # # baseline_power_right_ON = np.nanmean(mean_power_right_ON[:, baseline_indices], axis=1, keepdims=True)
                # # percentage_change_right_ON = (mean_power_right_ON - baseline_power_right_ON) / baseline_power_right_ON * 100

                # if epoch_type.startswith('G'):
                #     # Define baseline period for percentage change calculation
                #     baseline_indices = (times >= -500) & (times <= -200)

                #     # # Calculate baseline power and percentage change for left STN and for right STN
                #     # baseline_power_left = np.nanmean(mean_power_left[:, baseline_indices], axis=1, keepdims=True)
                #     # percentage_change_left = (mean_power_left - baseline_power_left) / baseline_power_left * 100

                #     # baseline_power_right = np.nanmean(mean_power_right[:, baseline_indices], axis=1, keepdims=True)
                #     # percentage_change_right = (mean_power_right - baseline_power_right) / baseline_power_right * 100

                #     baseline_power_left_ON = np.nanmean(power_left_ON_squeeze[:, :, baseline_indices], axis=2, keepdims=True)  # shape: (n_trials, n_freqs, 1 time)
                #     #percentage_change_left_single_trial = ((power_left_squeeze - baseline_power_left) / baseline_power_left) * 100  # shape: (n_trials, n_freqs, n_times)
                #     #percentage_change_left_single_trial = power_left_squeeze - baseline_power_left # shape: (n_trials, n_freqs, n_times)
                #     percentage_change_left_ON_single_trial = 10.0 * np.log10(power_left_ON_squeeze / baseline_power_left_ON)

                #     baseline_power_right_ON = np.nanmean(power_right_ON_squeeze[:, :, baseline_indices], axis=2, keepdims=True)  # shape: (n_trials, n_freqs, 1 time)
                #     #percentage_change_right_single_trial = ((power_right_squeeze - baseline_power_right) / baseline_power_right) * 100  # shape: (n_trials, n_freqs, n_times)
                #     #percentage_change_right_single_trial = power_right_squeeze - baseline_power_right
                #     percentage_change_right_ON_single_trial = 10.0 * np.log10(power_right_ON_squeeze / baseline_power_right_ON)

                #     percentage_change_left_ON = np.nanmean(percentage_change_left_ON_single_trial, axis=0)  # shape: (n_freqs, n_times)
                #     percentage_change_right_ON = np.nanmean(percentage_change_right_ON_single_trial, axis=0)  # shape: (n_freqs, n_times)

                # else: 
                #     if epoch_type == 'stop': 
                #         ssd_column = 'stop_signal_time'
                #     elif epoch_type == 'continue':
                #         ssd_column = 'continue_signal_time'

                #     baseline_start_per_trial = - 500 - (np.array(data_ON.metadata[ssd_column]) * 1000)
                #     baseline_end_per_trial = - 200 - (np.array(data_ON.metadata[ssd_column]) * 1000)

                #     percentage_change_left_ON_single_trial = np.empty_like(power_left_ON_squeeze)  # same shape
                #     baseline_power_left_ON = np.empty((power_left_ON_squeeze.shape[0], power_left_ON_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

                #     for i in range(power_left_ON_squeeze.shape[0]):  # loop over trials
                #         # Get trial-specific baseline window
                #         bl_start = baseline_start_per_trial[i]
                #         bl_end   = baseline_end_per_trial[i]

                #         # Find baseline indices in the common time axis
                #         bl_idx = (times >= bl_start) & (times <= bl_end)

                #         # Compute mean power in this window for all frequencies
                #         bl_mean = np.nanmean(power_left_ON_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

                #         # Store baseline and percent change
                #         baseline_power_left_ON[i] = bl_mean
                #         #percentage_change_left_single_trial[i] = ((power_left_squeeze[i] - bl_mean) / bl_mean) * 100
                #         #percentage_change_left_single_trial[i] = power_left_squeeze[i] - bl_mean
                #         percentage_change_left_ON_single_trial[i] = 10.0 * np.log10(power_left_ON_squeeze[i] / bl_mean)

                #     percentage_change_right_ON_single_trial = np.empty_like(power_right_ON_squeeze)  # same shape
                #     baseline_power_right_ON = np.empty((power_right_ON_squeeze.shape[0], power_right_ON_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

                #     for i in range(power_right_ON_squeeze.shape[0]):  # loop over trials
                #         # Get trial-specific baseline window
                #         bl_start = baseline_start_per_trial[i]
                #         bl_end   = baseline_end_per_trial[i]

                #         # Find baseline indices in the common time axis
                #         bl_idx = (times >= bl_start) & (times <= bl_end)

                #         # Compute mean power in this window for all frequencies
                #         bl_mean = np.nanmean(power_right_ON_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

                #         # Store baseline and percent change
                #         baseline_power_right_ON[i] = bl_mean
                #         #percentage_change_right_single_trial[i] = ((power_right_squeeze[i] - bl_mean) / bl_mean) * 100
                #         #percentage_change_right_single_trial[i] = power_right_squeeze[i] - bl_mean
                #         percentage_change_right_ON_single_trial[i] = 10.0 * np.log10(power_right_ON_squeeze[i] / bl_mean)

                #         percentage_change_left_ON = np.nanmean(percentage_change_left_ON_single_trial, axis=0)  # shape: (n_freqs, n_times)
                #         percentage_change_right_ON = np.nanmean(percentage_change_right_ON_single_trial, axis=0)  # shape: (n_freqs, n_times)

    
            if 'DBS OFF' in subject:
                type_mask = epochs.metadata["event"] == epoch_type
                outcome_mask = epochs.metadata["key_resp_experiment.corr"] == outcome
                data_OFF = epochs[type_mask & outcome_mask]                

                if not (cond == 'GS_successful' or cond == 'stop_successful'):
                    rt_off = data_OFF.metadata['key_resp_experiment.rt'].mean() * 1000
                    all_sub_RT_OFF.append(rt_off)
                    print(rt_off)

                (percentage_change_left_OFF, 
                 percentage_change_right_OFF, 
                 times, freqs) = get_change_from_baseline(
                    epochs = epochs,
                    cond = cond,
                    tfr_args = tfr_args,
                    baseline_correction = baseline_correction
                 )

                # left_epochs_OFF, right_epochs_OFF = data_OFF.copy().pick(['Left_STN']), data_OFF.copy().pick(['Right_STN'])
            
                # power_left_OFF = left_epochs_OFF.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
                # power_right_OFF = right_epochs_OFF.compute_tfr(**tfr_args)

                # power_left_OFF.data *= 1e12 # V² -> (µV)²
                # power_right_OFF.data *= 1e12 # V² -> (µV)²

                # power_left_OFF_squeeze = power_left_OFF.data.squeeze() # shape: (n_trials, n_freqs, n_times)
                # power_right_OFF_squeeze = power_right_OFF.data.squeeze()            

                # mean_power_left_OFF = np.nanmean(power_left_OFF.data, axis=0).squeeze() # shape: (n freqs, n times)
                # mean_power_right_OFF = np.nanmean(power_right_OFF.data, axis=0).squeeze()
                
                # times = power_left_OFF.times * 1000
                # freqs = power_left_OFF.freqs

                # # # Define baseline period for percentage change calculation
                # # baseline_indices = (times >= -500) & (times <= -200)

                # # # Percentage change for condition OFF
                # # baseline_power_left_OFF = np.nanmean(mean_power_left_OFF[:, baseline_indices], axis=1, keepdims=True)
                # # percentage_change_left_OFF = (mean_power_left_OFF - baseline_power_left_OFF) / baseline_power_left_OFF * 100

                # # baseline_power_right_OFF = np.nanmean(mean_power_right_OFF[:, baseline_indices], axis=1, keepdims=True)
                # # percentage_change_right_OFF = (mean_power_right_OFF - baseline_power_right_OFF) / baseline_power_right_OFF * 100
                # if epoch_type.startswith('G'):
                #     # Define baseline period for percentage change calculation
                #     baseline_indices = (times >= -500) & (times <= -200)

                #     # # Calculate baseline power and percentage change for left STN and for right STN
                #     # baseline_power_left = np.nanmean(mean_power_left[:, baseline_indices], axis=1, keepdims=True)
                #     # percentage_change_left = (mean_power_left - baseline_power_left) / baseline_power_left * 100

                #     # baseline_power_right = np.nanmean(mean_power_right[:, baseline_indices], axis=1, keepdims=True)
                #     # percentage_change_right = (mean_power_right - baseline_power_right) / baseline_power_right * 100

                #     baseline_power_left_OFF = np.nanmean(power_left_OFF_squeeze[:, :, baseline_indices], axis=2, keepdims=True)  # shape: (n_trials, n_freqs, 1 time)
                #     #percentage_change_left_single_trial = ((power_left_squeeze - baseline_power_left) / baseline_power_left) * 100  # shape: (n_trials, n_freqs, n_times)
                #     #percentage_change_left_single_trial = power_left_squeeze - baseline_power_left # shape: (n_trials, n_freqs, n_times)
                #     percentage_change_left_OFF_single_trial = 10.0 * np.log10(power_left_OFF_squeeze / baseline_power_left_OFF)

                #     baseline_power_right_OFF = np.nanmean(power_right_OFF_squeeze[:, :, baseline_indices], axis=2, keepdims=True)  # shape: (n_trials, n_freqs, 1 time)
                #     #percentage_change_right_single_trial = ((power_right_squeeze - baseline_power_right) / baseline_power_right) * 100  # shape: (n_trials, n_freqs, n_times)
                #     #percentage_change_right_single_trial = power_right_squeeze - baseline_power_right
                #     percentage_change_right_OFF_single_trial = 10.0 * np.log10(power_right_OFF_squeeze / baseline_power_right_OFF)

                #     percentage_change_left_OFF = np.nanmean(percentage_change_left_OFF_single_trial, axis=0)  # shape: (n_freqs, n_times)
                #     percentage_change_right_OFF = np.nanmean(percentage_change_right_OFF_single_trial, axis=0)  # shape: (n_freqs, n_times)

                # else: 
                #     if epoch_type == 'stop': 
                #         ssd_column = 'stop_signal_time'
                #     elif epoch_type == 'continue':
                #         ssd_column = 'continue_signal_time'

                #     baseline_start_per_trial = - 500 - (np.array(data_OFF.metadata[ssd_column]) * 1000)
                #     baseline_end_per_trial = - 200 - (np.array(data_OFF.metadata[ssd_column]) * 1000)

                #     percentage_change_left_OFF_single_trial = np.empty_like(power_left_OFF_squeeze)  # same shape
                #     baseline_power_left_OFF = np.empty((power_left_OFF_squeeze.shape[0], power_left_OFF_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

                #     for i in range(power_left_OFF_squeeze.shape[0]):  # loop over trials
                #         # Get trial-specific baseline window
                #         bl_start = baseline_start_per_trial[i]
                #         bl_end   = baseline_end_per_trial[i]

                #         # Find baseline indices in the common time axis
                #         bl_idx = (times >= bl_start) & (times <= bl_end)

                #         # Compute mean power in this window for all frequencies
                #         bl_mean = np.nanmean(power_left_OFF_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

                #         # Store baseline and percent change
                #         baseline_power_left_OFF[i] = bl_mean
                #         #percentage_change_left_single_trial[i] = ((power_left_squeeze[i] - bl_mean) / bl_mean) * 100
                #         #percentage_change_left_single_trial[i] = power_left_squeeze[i] - bl_mean
                #         percentage_change_left_OFF_single_trial[i] = 10.0 * np.log10(power_left_OFF_squeeze[i] / bl_mean)

                #     percentage_change_right_OFF_single_trial = np.empty_like(power_right_OFF_squeeze)  # same shape
                #     baseline_power_right_OFF = np.empty((power_right_OFF_squeeze.shape[0], power_right_OFF_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

                #     for i in range(power_right_OFF_squeeze.shape[0]):  # loop over trials
                #         # Get trial-specific baseline window
                #         bl_start = baseline_start_per_trial[i]
                #         bl_end   = baseline_end_per_trial[i]

                #         # Find baseline indices in the common time axis
                #         bl_idx = (times >= bl_start) & (times <= bl_end)

                #         # Compute mean power in this window for all frequencies
                #         bl_mean = np.nanmean(power_right_OFF_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

                #         # Store baseline and percent change
                #         baseline_power_right_OFF[i] = bl_mean
                #         #percentage_change_right_single_trial[i] = ((power_right_squeeze[i] - bl_mean) / bl_mean) * 100
                #         #percentage_change_right_single_trial[i] = power_right_squeeze[i] - bl_mean
                #         percentage_change_right_OFF_single_trial[i] = 10.0 * np.log10(power_right_OFF_squeeze[i] / bl_mean)

                #         percentage_change_left_OFF = np.nanmean(percentage_change_left_OFF_single_trial, axis=0)  # shape: (n_freqs, n_times)
                #         percentage_change_right_OFF = np.nanmean(percentage_change_right_OFF_single_trial, axis=0)  # shape: (n_freqs, n_times)

        diff_left = percentage_change_left_ON - percentage_change_left_OFF
        diff_right = percentage_change_right_ON - percentage_change_right_OFF

        """
        # Slicing TFR data to include only the t_min, t_max time range
        time_indices = (times >= t_min_max[0]) & (times <= t_min_max[1])
        #time_indices = np.logical_and(times >= t_min_max[0], times <= t_min_max[1])
        print(time_indices)
        sliced_times = times[time_indices]
        print(sliced_times.shape)

        sliced_diff_left = diff_left[:, time_indices]
        print(sliced_diff_left.shape)
        sliced_diff_right = diff_right[:, time_indices]
        """

        all_diff_left.append(diff_left)
        all_diff_right.append(diff_right)
        all_diff_both.extend([diff_left, diff_right])

        subs_included.append(sub)

    print(f'Subs included in analyses: \n {subs_included}')

    all_diff_left_array = np.array(all_diff_left)  # shape: (n sub, n freqs, n times)
    all_diff_right_array = np.array(all_diff_right)
    all_diff_both_array = np.array(all_diff_both) # shape: (n sub x 2, n freqs, n times)

    time_indices = (times >= t_min_max[0]) & (times <= t_min_max[1])
    sliced_times = times[time_indices]
    all_diff_left_array_sliced = all_diff_left_array[:,:,time_indices]
    all_diff_right_array_sliced = all_diff_right_array[:,:,time_indices]
    all_diff_both_array_sliced = all_diff_both_array[:,:,time_indices]

    if len(subs_included) > 1:
        n_obs = all_diff_left_array_sliced.shape[0]
        print(n_obs)
        pval = 0.05
        df = n_obs - 1
        threshold = scipy.stats.t.ppf(1-pval / 2, df) # two-tailed distribution
        #threshold = None
        n_permutations = 1000

        # Compute permutation cluster test for the left stn
        T_obs_left, clusters_left, cluster_p_values_left, H0_left = mne.stats.permutation_cluster_1samp_test(
        all_diff_left_array_sliced, n_permutations=n_permutations,
        threshold=threshold, tail=0,
        out_type= "mask", seed=11111, verbose=True)
        print(f"p_values: {cluster_p_values_left}")
        print(f"P_values shape: {cluster_p_values_left.shape}")

        print("Clusters for Left STN")
        identify_significant_clusters(
            cluster_p_values_left, 
            clusters_left,
            sliced_times,
            T_obs_left,
            pval,
            tfr_args,
            #cluster_results_dict,
            condition,
            'Left'
            )
        
        # Compute permutation cluster test for the right stn
        T_obs_right, clusters_right, cluster_p_values_right, H0_right = mne.stats.permutation_cluster_1samp_test(
        all_diff_right_array_sliced, n_permutations=n_permutations,
        threshold=threshold, tail=0,
        out_type= "mask", seed=11111, verbose=True)
        print(f"p_values: {cluster_p_values_right}")
        print(f"P_values shape: {cluster_p_values_right.shape}")

        print("Clusters for Right STN")
        identify_significant_clusters(
            cluster_p_values_right, 
            clusters_right,
            sliced_times,
            T_obs_right,
            pval,
            tfr_args,
            #cluster_results_dict,
            condition,
            'Right'
            )

        # Compute permutation cluster test for the left + right stn
        T_obs_both, clusters_both, cluster_p_values_both, H0_both = mne.stats.permutation_cluster_1samp_test(
        all_diff_both_array_sliced, n_permutations=n_permutations,
        threshold=threshold, tail=0,
        out_type= "mask", seed=11111, verbose=True)
        print(f"p_values: {cluster_p_values_both}")
        print(f"P_values shape: {cluster_p_values_both.shape}")

        print("Clusters for Both STN")
        identify_significant_clusters(
            cluster_p_values_both, 
            clusters_both,
            sliced_times,
            T_obs_both,
            pval,
            tfr_args,
            #cluster_results_dict,
            condition,
            'Both'
            )


    # Average the percentage signal changes across subjects for left STN and for right STN
    avg_diff_left = np.nanmean(all_diff_left_array_sliced, axis=0)  # shape: (n freqs, n times)
    print(f"avg diff left shape: {avg_diff_left.shape}")
    avg_diff_right = np.nanmean(all_diff_right_array_sliced, axis=0)
    avg_diff_both = np.nanmean(all_diff_both_array_sliced, axis=0)


    ################
    ### PLOTTING ###
    ################    

    # Create a figure with two subplots for Left and Right STN
    fig, (ax_left, ax_right, ax_both) = plt.subplots(1, 3, figsize=(20, 8))

    # Figure title for n_subjects
    sub_num = len(all_diff_left)

    if sub_num > 1:
        fig.suptitle(f"Power difference DBS ON - DBS OFF {cond}, nSub = {sub_num}")
    else:
        fig.suptitle(f"Power difference DBS ON - DBS OFF {cond}, {subject[:6]}")

    if epoch_type == 'stop':
        xlabel = 'Time from STOP cue (ms)'
    elif epoch_type == 'continue':
        xlabel = 'Time from CONTINUE cue (ms)'
    else:
        xlabel = 'Time from GO cue (ms)'
    

    # Plot the percentage change difference for Left STN
    im_left = ax_left.imshow(avg_diff_left, aspect='auto', origin='lower', 
                            extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel('Frequency (Hz)')
    ax_left.set_title(f'Left STN - {cond}')
    #fig.colorbar(im_left, ax=ax_left, label='Mean % Change (from baseline)')
    fig.colorbar(im_left, ax=ax_left, label='Change from baseline (dB)')

    # Plot the percentage change difference for Right STN
    im_right = ax_right.imshow(avg_diff_right, aspect='auto', origin='lower', 
                            extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    ax_right.set_xlabel(xlabel)
    ax_right.set_ylabel('Frequency (Hz)')
    ax_right.set_title(f'Right STN - {cond}')
    #fig.colorbar(im_right, ax=ax_right, label='Mean % Change (from baseline)')
    fig.colorbar(im_right, ax=ax_right, label='Change from baseline (dB)')

    # Plot the percentage change difference for Left + Right STN
    im_both = ax_both.imshow(avg_diff_both, aspect='auto', origin='lower', 
                            extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    ax_both.set_xlabel(xlabel)
    ax_both.set_ylabel('Frequency (Hz)')
    ax_both.set_title(f'Left + Right STN - {cond}')
    #fig.colorbar(im_both, ax=ax_both, label='Mean % Change (from baseline)')
    fig.colorbar(im_both, ax=ax_both, label='Change from baseline (dB)')

    print(f" {cond} RT ON: {all_sub_RT_ON}")
    print(f" {cond} RT OFF: {all_sub_RT_OFF}")

    if len(subs_included) > 1:
        mean_RT_ON = np.mean(all_sub_RT_ON)
        mean_RT_OFF = np.mean(all_sub_RT_OFF)
    
        for c, p_val in zip(clusters_left, cluster_p_values_left):
            if p_val <= pval:
                mask_L = np.zeros_like(T_obs_left, dtype=bool) 
                mask_L[c] = True
                ax_left.contour(mask_L, levels=[0.5], colors='black', linewidths=1.5,
                                extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])    
        for c, p_val in zip(clusters_right, cluster_p_values_right):
            if p_val <= pval:
                mask_R = np.zeros_like(T_obs_right, dtype=bool)
                mask_R[c] = True
                ax_right.contour(mask_R, levels=[0.5], colors='black', linewidths=1.5,
                                extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])    
        for c, p_val in zip(clusters_both, cluster_p_values_both):
            if p_val <= pval:
                mask_R = np.zeros_like(T_obs_both, dtype=bool)
                mask_R[c] = True
                ax_both.contour(mask_R, levels=[0.5], colors='black', linewidths=1.5,
                                extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])

        #mean_SSD = np.mean(all_sub_SSD)
        #mean_SSRT = np.mean(all_sub_SSRT)
    else:
        mean_RT_ON = all_sub_RT_ON[0]
        mean_RT_OFF = all_sub_RT_OFF[0]
        #mean_SSD = all_sub_SSD[0]
        #mean_SSRT = all_sub_SSRT[0]

    if RT_plot:     
        ax_left.axvline(mean_RT_ON, color='black', linestyle='--')
        ax_right.axvline(mean_RT_ON, color='black', linestyle='--')
        ax_both.axvline(mean_RT_ON, color='black', linestyle='--', label='Mean RT ON')
        ax_left.axvline(mean_RT_OFF, color='grey', linestyle='--')
        ax_right.axvline(mean_RT_OFF, color='grey', linestyle='--')
        ax_both.axvline(mean_RT_OFF, color='grey', linestyle='--', label='Mean RT OFF')

    #ax_left.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')
    #ax_right.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')    
    #ax_both.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')    
    #ax_left.axvline(mean_SSRT, color='blue', linestyle='--', label='Mean SSRT')
    #ax_right.axvline(mean_SSRT, color='blue', linestyle='--', label='Mean SSRT')
    #ax_both.axvline(mean_SSRT, color='blue', linestyle='--', label='Mean SSRT')    

    fig.legend() 

    if sub_num > 1:
        # Save the figure if a saving path is provided
        figtitle = f"Power_diff_DBS_ON-DBS_OFF_{cond}.png"
        figtitle_pdf = f"Power_diff_DBS_ON-DBS_OFF_{cond}.pdf"
    else:
        # Save the figure if a saving path is provided
        figtitle = f"{subject[:6]}_Power_diff_DBS_ON-DBS_OFF_{cond}.png"
        figtitle_pdf = f"{subject[:6]}_Power_diff_DBS_ON-DBS_OFF_{cond}.pdf"

    if saving_path is not None:
        if save_as == 'png':
            plt.savefig(join(saving_path, figtitle), transparent=False)
            # plt.savefig(join(saving_path, figtitle_pdf), transparent=False)
        else:
            plt.savefig(join(saving_path, figtitle_pdf), transparent=False)

    if show_fig == True:
        plt.show()
    else:
        plt.close('all')


def perc_pow_diff_cond(
        sub_dict, 
        dbs_status:str, 
        tfr_args, 
        t_min_max:list, 
        vmin_vmax:list,
        epoch_cond1:str, 
        epoch_cond2:str, 
        condition: str,
        baseline_correction: bool = True,
        saving_path: str=None, 
        show_fig: bool = None,
        add_rt: bool = True,
        save_as: str = 'png'
        ):
        
    """
    Calculates % power change for the two specified conditions and subtracts epoch_cond2 from epoch_cond1, giving percentage change unique for epoch_cond1. 
    Loops through all subs in sub_dict. 

    Input:
    - sub_dict: dict. containing all epochs (cue or feedback)
    - dbs_status: "DBS ON" or "DBS OFF" 
    - tfr_args: TFR parameters
    - tmin, tmax: epoch slicing
    - epoch_cond1: Epoch of interest "Win_cue", "Loss_cue"
    - epoch_cond2: Baseline epoch, e.g., "Neutral_cue" to subtract from main epoch. 
    - saving_path: Path where plots will be saved. If None figures are not saved. 
    - show_fig: Defaults to None and figure isn't shown, if True figure is shown.
    """
    
    all_diff_left = []
    all_diff_right = []
    all_diff_both = []
    all_sub_RT1 = []
    all_sub_RT2 = []
    all_sub_SSD = []

    RT_plot1 = True
    SSD_plot1 = False
    RT_plot2 = True
    SSD_plot2 = False

    epoch_type1 = epoch_cond1.split('_')[0]
    outcome_str1 = epoch_cond1.split('_')[1]
    epoch_type2 = epoch_cond2.split('_')[0]
    outcome_str2 = epoch_cond2.split('_')[1]

    outcome1 = 1.0 if outcome_str1 == "successful" else 0.0
    outcome2 = 1.0 if outcome_str2 == "successful" else 0.0

    if epoch_cond1 == 'GS_successful':
        RT_plot1 = False
        SSD_plot1 = True

    if epoch_cond1 == 'GS_unsuccessful':
        SSD_plot1 = True

    if epoch_cond2 == 'GS_successful':
        RT_plot2 = False
        SSD_plot2 = True

    if epoch_cond2 == 'GS_unsuccessful':
        SSD_plot2 = True    

    if epoch_cond1 == 'stop_successful':
        RT_plot1 = False
    
    if epoch_cond2 == 'stop_successful':
        RT_plot2 = False

    subs_included = []

    # Collect epoch data for each condition
    for subject, epochs in sub_dict.items():
        if dbs_status in subject:            
            # Epoch condition 1
            type_mask1 = epochs.metadata["event"] == epoch_type1  # 'GO'
            outcome_mask1 = epochs.metadata["key_resp_experiment.corr"] == outcome1   # '1.0'
            data1 = epochs[type_mask1 & outcome_mask1]

            if RT_plot1 and add_rt: 
                sub_RT1 = data1.metadata["key_resp_experiment.rt"].mean() * 1000
                all_sub_RT1.append(sub_RT1)
            if SSD_plot1:
                sub_SSD = data1.metadata["stop_signal_time"].mean() * 1000
                all_sub_SSD.append(sub_SSD)

            # Epoch condition 2
            type_mask2 = epochs.metadata["event"] == epoch_type2
            outcome_mask2 = epochs.metadata["key_resp_experiment.corr"] == outcome2
            data2 = epochs[type_mask2 & outcome_mask2]

            if RT_plot2 and add_rt: 
                sub_RT2 = data2.metadata["key_resp_experiment.rt"].mean() * 1000
                all_sub_RT2.append(sub_RT2)
            if SSD_plot2:
                sub_SSD = data2.metadata["stop_signal_time"].mean() * 1000
                all_sub_SSD.append(sub_SSD)                

            (percentage_change_left_ep1, 
                percentage_change_right_ep1, 
                times, 
                freqs) = get_change_from_baseline(
                    epochs = epochs,
                    cond = epoch_cond1,
                    tfr_args = tfr_args,
                    baseline_correction = baseline_correction
                )

            (percentage_change_left_ep2, 
                percentage_change_right_ep2, 
                times, 
                freqs) = get_change_from_baseline(
                    epochs = epochs,
                    cond = epoch_cond2,
                    tfr_args = tfr_args,
                    baseline_correction = baseline_correction
                )


            # Differences between Cond1 and Cond2 left and right STN
            diff_left = percentage_change_left_ep1 - percentage_change_left_ep2
            diff_right = percentage_change_right_ep1 - percentage_change_right_ep2

            all_diff_left.append(diff_left)
            all_diff_right.append(diff_right)  
            # all_diff_both.append(diff_left)
            # all_diff_both.append(diff_right)
            all_diff_both.extend([diff_left, diff_right])

            subs_included.append(subject)


    print(f'Subs included in analyses: \n {subs_included}')

    all_diff_left_array = np.array(all_diff_left)  # shape: (n sub, n freqs, n times)
    all_diff_right_array = np.array(all_diff_right)
    all_diff_both_array = np.array(all_diff_both)

    time_indices = (times >= t_min_max[0]) & (times <= t_min_max[1])
    sliced_times = times[time_indices]
    all_diff_left_array_sliced = all_diff_left_array[:,:,time_indices]
    all_diff_right_array_sliced = all_diff_right_array[:,:,time_indices]
    all_diff_both_array_sliced = all_diff_both_array[:,:,time_indices]

    if len(subs_included) > 1:
        n_obs = all_diff_left_array_sliced.shape[0]
        print(n_obs)
        pval = 0.05
        df = n_obs - 1
        threshold = scipy.stats.t.ppf(1-pval / 2, df) # two-tailed distribution
        #threshold = None
        n_permutations = 1000

        # Compute permutation cluster test for the left stn
        T_obs_left, clusters_left, cluster_p_values_left, H0_left = mne.stats.permutation_cluster_1samp_test(
        all_diff_left_array_sliced, n_permutations=n_permutations,
        threshold=threshold, tail=0,
        out_type= "mask", seed=11111, verbose=True)
        print(f"p_values: {cluster_p_values_left}")
        print(f"P_values shape: {cluster_p_values_left.shape}")

        print("Clusters for Left STN")
        identify_significant_clusters(
            cluster_p_values_left, 
            clusters_left,
            sliced_times,
            T_obs_left,
            pval,
            tfr_args,
            condition,
            "Left"
            )

        # Compute permutation cluster test for the right stn
        T_obs_right, clusters_right, cluster_p_values_right, H0_right = mne.stats.permutation_cluster_1samp_test(
        all_diff_right_array_sliced, n_permutations=n_permutations,
        threshold=threshold, tail=0,
        out_type= "mask", seed=11111, verbose=True)
        print(f"p_values: {cluster_p_values_right}")
        print(f"P_values shape: {cluster_p_values_right.shape}")

        print("Clusters for Right STN")
        identify_significant_clusters(
            cluster_p_values_right, 
            clusters_right,
            sliced_times,
            T_obs_right,
            pval,
            tfr_args,
            condition,
            "Right"
            )

        # Compute permutation cluster test for the left + right stn
        T_obs_both, clusters_both, cluster_p_values_both, H0_both = mne.stats.permutation_cluster_1samp_test(
        all_diff_both_array_sliced, n_permutations=n_permutations,
        threshold=threshold, tail=0,
        out_type= "mask", seed=11111, verbose=True)
        print(f"p_values: {cluster_p_values_both}")
        print(f"P_values shape: {cluster_p_values_both.shape}")

        print("Clusters for Both STN")
        identify_significant_clusters(
            cluster_p_values_both, 
            clusters_both,
            sliced_times,
            T_obs_both,
            pval,
            tfr_args,
            condition,
            "Both"
            )


    # Average the percentage signal changes across subjects for left STN and for right STN
    avg_diff_left = np.nanmean(all_diff_left_array_sliced, axis=0)
    avg_diff_right = np.nanmean(all_diff_right_array_sliced, axis=0)
    avg_diff_both = np.nanmean(all_diff_both_array_sliced, axis=0)


    ################
    ### PLOTTING ###
    ################    

    # Create a figure with two subplots for Left and Right STN
    fig, (ax_left, ax_right, ax_both) = plt.subplots(1, 3, figsize=(20, 8))

    # Figure title for n_subjects
    sub_num = len(all_diff_left)

    if sub_num > 1:
        fig.suptitle(f"Power difference {epoch_cond1} - {epoch_cond2}, nSub = {sub_num}")
    else:
        fig.suptitle(f"Power difference {epoch_cond1} - {epoch_cond2}, {subject[:6]}")


    # Set the x label based on what the epochs are centered on:
    if epoch_type1 == 'stop' or epoch_type1 == 'continue':
        xlabel = 'Time from STOP cue (ms)'
    # elif epoch_type1 == 'continue':
    #     xlabel = 'Time from CONTINUE cue (ms)'
    else:
        xlabel = 'Time from GO cue (ms)'
        

    # Plot the percentage change difference for Left STN
    im_left = ax_left.imshow(avg_diff_left, aspect='auto', origin='lower', 
                            extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    

    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel('Frequency (Hz)')
    ax_left.set_title(f'Left STN - {dbs_status}')
    #fig.colorbar(im_left, ax=ax_left, label='Mean % Change (from baseline)')
    fig.colorbar(im_left, ax=ax_left, label='Change from baseline (dB)')

    # Plot the percentage change difference for Right STN
    im_right = ax_right.imshow(avg_diff_right, aspect='auto', origin='lower', 
                            extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])

    ax_right.set_xlabel(xlabel)
    ax_right.set_ylabel('Frequency (Hz)')
    ax_right.set_title(f'Right STN - {dbs_status}')
    #fig.colorbar(im_right, ax=ax_right, label='Mean % Change (from baseline)')
    fig.colorbar(im_right, ax=ax_right, label='Change from baseline (dB)')

    # Plot the percentage change difference for Left + Right STN
    im_both = ax_both.imshow(avg_diff_both, aspect='auto', origin='lower', 
                            extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])

    ax_both.set_xlabel(xlabel)
    ax_both.set_ylabel('Frequency (Hz)')
    ax_both.set_title(f'Left + Right STN - {dbs_status}')
    #fig.colorbar(im_both, ax=ax_both, label='Mean % Change (from baseline)')
    fig.colorbar(im_both, ax=ax_both, label='Change from baseline (dB)')

    # add significant clusters on the plot if group-level analysis:
    if sub_num > 1:
        for c, p_val in zip(clusters_left, cluster_p_values_left):
            if p_val <= pval:
                mask = np.zeros_like(T_obs_left, dtype=bool)  # Explicitly match dimensions
                mask[c] = True
                ax_left.contour(mask, levels=[0.5], colors='black', linewidths=1.5,
                                extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
        
        for c, p_val in zip(clusters_right, cluster_p_values_right):
            if p_val <= pval:
                mask = np.zeros_like(T_obs_right, dtype=bool)  # Explicitly match dimensions
                mask[c] = True
                ax_right.contour(mask, levels=[0.5], colors='black', linewidths=1.5,
                                extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
            
        for c, p_val in zip(clusters_both, cluster_p_values_both):
            if p_val <= pval:
                mask = np.zeros_like(T_obs_both, dtype=bool)  # Explicitly match dimensions
                mask[c] = True
                ax_both.contour(mask, levels=[0.5], colors='black', linewidths=1.5,
                                extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
        


    if RT_plot1 and add_rt:
        # Average mean RT across subjects
        if len(subs_included) > 1:
            mean_RT1 = np.mean(all_sub_RT1)
        else:
            mean_RT1 = all_sub_RT1[0]
        ax_left.axvline(mean_RT1, color='black', linestyle='--')
        ax_right.axvline(mean_RT1, color='black', linestyle='--')
        ax_both.axvline(mean_RT1, color='black', linestyle='--', label=f'Mean RT {epoch_cond1}')

    if SSD_plot1:
        # Average mean RT across subjects
        if len(subs_included) > 1:
            mean_SSD = np.mean(all_sub_SSD)
        else:
            mean_SSD = all_sub_SSD[0]
        ax_left.axvline(mean_SSD, color='grey', linestyle='--')
        ax_right.axvline(mean_SSD, color='grey', linestyle='--')
        ax_both.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')

    if RT_plot2 and add_rt:
        # Average mean RT across subjects
        if len(subs_included) > 1:
            mean_RT2 = np.mean(all_sub_RT2)
        else:
            mean_RT2 = all_sub_RT2[0]
        ax_left.axvline(mean_RT2, color='blue', linestyle='--')
        ax_right.axvline(mean_RT2, color='blue', linestyle='--')
        ax_both.axvline(mean_RT2, color='blue', linestyle='--', label=f'Mean RT {epoch_cond2}')

    if SSD_plot2:
        # Average mean RT across subjects
        if len(subs_included) > 1:
            mean_SSD = np.mean(all_sub_SSD)
        else:
            mean_SSD = all_sub_SSD[0]
        ax_left.axvline(mean_SSD, color='grey', linestyle='--')
        ax_right.axvline(mean_SSD, color='grey', linestyle='--')    
        ax_both.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')    

    fig.legend()

    if sub_num > 1:
        # Save the figure if a saving path is provided
        figtitle = f"Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.png"
        figtitle_pdf = f"Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.pdf"
    else:
        # Save the figure if a saving path is provided
        figtitle = f"{subject[:6]}_Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.png"
        figtitle_pdf = f"{subject[:6]}_Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.pdf"

    if saving_path is not None:
        if save_as == 'png':
            plt.savefig(join(saving_path, figtitle), transparent=False)
        else:
            plt.savefig(join(saving_path, figtitle_pdf), transparent=False)

    if show_fig == True:
        plt.show()
    else:
        plt.close('all')




def tfr_pow_change_cond(
        sub_dict, 
        dbs_status:str, 
        epoch_cond:str,
        tfr_args, 
        t_min_max:list, 
        vmin_vmax:list,
        baseline_correction:bool=True,
        saving_path: str=None, 
        show_fig: bool = False,
        add_rt: bool = True, 
        save_as: str = 'png'
        ):

    
    """
    Runs TFR on the chosen epoch condition, averages raw signal across epochs before performing TFR,
    and plots TFR plots showing percentage power change (or absolute power) in DBS ON or DBS OFF condition.
    Loops through all subjects in sub_dict.

    Parameters
    ----------
    sub_dict : dict
        Dictionary containing all epochs per subject.
    dbs_status : str
        "DBS ON" or "DBS OFF".
    epoch_cond : str
        Condition name (e.g., "Win_cue", "Loss_cue").
    tfr_args : dict
        TFR computation parameters (freqs, n_cycles, etc.).
    t_min_max : list
        Time range [tmin, tmax] for plotting (in ms).
    vmin_vmax : list
        Color scale range [vmin, vmax] for plots.
    baseline_correction : bool, optional
        Whether to apply baseline correction.
    saving_path : str, optional
        Directory where plots will be saved. If None, figures are not saved.
    show_fig : bool, optional
        If True, display the figure.
    add_rt : bool, optional
        If True, plot mean RT as a vertical line.
    save_as : str, optional
        File format for saving ('png' or 'pdf').
    """
  
    # Prepare containers
    all_percentage_change_left = []
    all_percentage_change_right = []
    all_percentage_change_both = []
    all_sub_RT = []
    all_sub_SSD = []
    subs_included = []

    # Parse epoch condition
    epoch_type, outcome_str = epoch_cond.split('_')
    outcome = 1.0 if outcome_str == "successful" else 0.0

    # Decide which plots to show RT/SSD lines
    RT_plot = True
    SSD_plot = False
    if epoch_type == 'GS':
        SSD_plot = True
        if outcome == 1.0:
            RT_plot = False
    if epoch_cond == 'stop_successful':
        RT_plot = False


    for subject, epochs in sub_dict.items():
        if dbs_status in subject:
            type_mask = epochs.metadata["event"] == epoch_type
            outcome_mask = epochs.metadata["key_resp_experiment.corr"] == outcome
            data = epochs[type_mask & outcome_mask]

            if RT_plot and add_rt: 
                sub_RT = data.metadata["key_resp_experiment.rt"].mean() * 1000
                all_sub_RT.append(sub_RT)
            if SSD_plot:
                sub_SSD = data.metadata["stop_signal_time"].mean() * 1000
                all_sub_SSD.append(sub_SSD)
            
            print(f"Data found: {len(data)} epochs loaded for {epoch_cond}")

            (percentage_change_left, 
                percentage_change_right, 
                times, 
                freqs) = get_change_from_baseline(
                    epochs = epochs,
                    cond = epoch_cond,
                    tfr_args = tfr_args,
                    baseline_correction = baseline_correction
                )

            # Append each subject's percentage change to the lists
            all_percentage_change_left.append(percentage_change_left)
            all_percentage_change_right.append(percentage_change_right)
            all_percentage_change_both.extend([percentage_change_left, percentage_change_right])

            subs_included.append(subject)

    print(f'Subjects included in analyses: \n {subs_included}')

    # Convert to arrays (shape: (n_subs, n_freqs, n_times))
    all_percentage_change_left = np.stack(all_percentage_change_left)
    all_percentage_change_right = np.stack(all_percentage_change_right)
    all_percentage_change_both = np.stack(all_percentage_change_both)

    # Compute grand averages
    avg_percentage_change_left = np.nanmean(all_percentage_change_left, axis=0)
    avg_percentage_change_right = np.nanmean(all_percentage_change_right, axis=0)
    avg_percentage_change_both = np.nanmean(all_percentage_change_both, axis=0)

    # Slice time window for plotting
    time_mask = (times >= t_min_max[0]) & (times <= t_min_max[1])
    sliced_data_left = avg_percentage_change_left[:, time_mask]
    sliced_data_right = avg_percentage_change_right[:, time_mask]
    sliced_data_both = avg_percentage_change_both[:, time_mask]


    # all_percentage_change_left = np.array(all_percentage_change_left)  # shape: (n sub, n freqs, n times)
    # all_percentage_change_right = np.array(all_percentage_change_right)
    # all_percentage_change_both = np.array(all_percentage_change_both)

    # # Average the percentage signal changes across subjects for left STN and for right STN
    # avg_percentage_change_left = np.nanmean(all_percentage_change_left, axis=0)
    # avg_percentage_change_right = np.nanmean(all_percentage_change_right, axis=0)
    # avg_percentage_change_both = np.nanmean(all_percentage_change_both, axis=0)

    # # ################################################################################################################################

    # # Slicing TFR data to include only the t_min, t_max time range
    # time_indices = np.logical_and(times >= t_min_max[0], times <= t_min_max[1])
    
    # # selects only timepoints where time_indices = True, then removes dimensions with 1 (i.e., first dimension which is 1 channel)
    # # first dimension is n_channels (which is already as we do left and right separately)
    # sliced_data_left = avg_percentage_change_left[:, time_indices].squeeze()
    # sliced_data_right = avg_percentage_change_right[:, time_indices].squeeze()
    # sliced_data_both = avg_percentage_change_both[:, time_indices].squeeze()

    # # Stack the scores for Left and Right STN across all subjects, so the output variable can be used for permutation cluster test
    # group_all_change_left = np.stack(all_percentage_change_left)  # Shape: (n_subjects, n_frequencies, n_times)
    # group_all_change_right = np.stack(all_percentage_change_right)  # Shape: (n_subjects, n_frequencies, n_times)
    # group_all_change_both = np.stack(all_percentage_change_both)

    # # Get the indices where the condition is True
    # freq_indices = np.where((freqs >= 5) & (freqs <= 20))[0]
    # time_indices = np.where((times >= -500) & (times <= 0))[0]

    # # Use integer-based indexing
    # filtered_data_left = group_all_change_left[:, freq_indices, :][:, :, time_indices]
    # filtered_data_right = group_all_change_right[:, freq_indices, :][:, :, time_indices]
    # filtered_data_both = group_all_change_both[:, freq_indices, :][:, :, time_indices]

    # Compute min and max along the frequency axis
    if not baseline_correction:
        min_values_left = np.min(sliced_data_left)  # Shape: (n_subjects, n_times)
        max_values_left = np.max(sliced_data_left)
        min_values_right = np.min(sliced_data_right)
        max_values_right = np.max(sliced_data_right)
        min_values_both = np.min(sliced_data_both)
        max_values_both = np.max(sliced_data_both)


    ################
    ### PLOTTING ###
    ################

    # Create a figure with two subplots for Left and Right STN
    fig, (ax_left, ax_right, ax_both) = plt.subplots(1, 3, figsize=(20, 8))

    # Figure title
    sub_num = len(subs_included)
    title_prefix = "Power change" if baseline_correction else "Power"
    bc_note = "" if baseline_correction else ", no baseline correction"
    subject_info = f"nSub = {sub_num}" if sub_num > 1 else subject[:6]
    fig.suptitle(f"{title_prefix} - {outcome_str} {epoch_type}, {subject_info}{bc_note}")


    # if baseline_correction:
    #     if sub_num > 1:
    #         fig.suptitle(f"Power change - {outcome_str} {epoch_type}, nSub = {sub_num}")
    #     else:
    #         fig.suptitle(f"Power change - {outcome_str} {epoch_type}, {subject[:6]}")
    # else:
    #     if sub_num > 1:
    #         fig.suptitle(f"Power - {outcome_str} {epoch_type}, nSub = {sub_num}, no baseline correction")
    #     else:
    #         fig.suptitle(f"Power - {outcome_str} {epoch_type}, {subject[:6]}, no baseline correction")

    # Set the x label based on what the epochs are centered on:
    if epoch_type == 'stop':
        xlabel = 'Time from STOP cue (ms)'
    elif epoch_type == 'continue':
        xlabel = 'Time from CONTINUE cue (ms)'
    else:
        xlabel = 'Time from GO cue (ms)'
        
    # Plot Left STN
    vmin, vmax = (vmin_vmax if baseline_correction else (min_values_left, max_values_left))
    im_left = ax_left.imshow(sliced_data_left, aspect='auto', origin='lower', 
                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                             cmap='jet', vmin=vmin, vmax=vmax)
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel('Frequency (Hz)')
    ax_left.set_title(f'Left STN - {dbs_status}')
    fig.colorbar(im_left, ax=ax_left, label='Change from baseline (dB)' if baseline_correction else 'Mean Power (µV²)')


    # Plot Right STN
    vmin, vmax = (vmin_vmax if baseline_correction else (min_values_right, max_values_right))
    im_right = ax_right.imshow(sliced_data_right, aspect='auto', origin='lower', 
                                extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                                cmap='jet', vmin=vmin, vmax=vmax)
    ax_right.set_xlabel(xlabel)
    ax_right.set_ylabel('Frequency (Hz)')
    ax_right.set_title(f'Right STN - {dbs_status}')
    fig.colorbar(im_right, ax=ax_right, label='Change from baseline (dB)' if baseline_correction else 'Mean Power (µV²)')


    # Plot Both STN
    vmin, vmax = (vmin_vmax if baseline_correction else (min_values_both, max_values_both))
    im_both = ax_both.imshow(sliced_data_both, aspect='auto', origin='lower', 
                              extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                              cmap='jet', vmin=vmin, vmax=vmax)
    ax_both.set_xlabel(xlabel)
    ax_both.set_ylabel('Frequency (Hz)')
    ax_both.set_title(f'Left + Right STN - {dbs_status}')
    fig.colorbar(im_both, ax=ax_both, label='Change from baseline (dB)' if baseline_correction else 'Mean Power (µV²)')



    # Add RT or SSD lines
    if RT_plot and add_rt and all_sub_RT:
        mean_RT = np.mean(all_sub_RT)
        for ax in (ax_left, ax_right, ax_both):
            ax.axvline(mean_RT, color='black', linestyle='--', label='Mean RT')
    if SSD_plot and all_sub_SSD:
        mean_SSD = np.mean(all_sub_SSD)
        for ax in (ax_left, ax_right, ax_both):
            ax.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')


    # Add legend if any labeled lines exist
    handles, labels = ax_both.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels)

    # Save figure if needed
    if saving_path:
        figtitle_base = f"{outcome_str}_{epoch_type}_{dbs_status}"
        if sub_num == 1:
            figtitle_base = f"{subject[:6]}_" + figtitle_base
        if baseline_correction:
            figtitle = f"Power_change_{figtitle_base}"
        else:
            figtitle = f"Power_{figtitle_base}_no_baseline_correction"

        file_path = join(saving_path, f"{figtitle}.{save_as}")
        plt.savefig(file_path, transparent=False)
        print(f"Figure saved as {file_path}")

    # Show or close figure
    if show_fig:
        plt.show()
    else:
        plt.close(fig)





def plot_raw_stim(
        session_ID: str, 
        raw: mne.io.Raw,
        saving_path: str
        ):
    """
    This function plots the raw data and the stimulation data for the left and right channel.

    session_ID: session ID
    raw: raw data (loaded through mne.io.read_raw)
    saving_path: path to save the figure
    """

    sf = raw.info['sfreq']
    L_chan = raw.get_data(picks=raw.ch_names[0])[0]
    R_chan = raw.get_data(picks=raw.ch_names[1])[0]
    stim_L_chan = raw.get_data(picks=raw.ch_names[4])[0]
    stim_R_chan = raw.get_data(picks=raw.ch_names[5])[0]
    #timescale = np.arange(0, len(L_chan)/sf, 1/sf)
    timescale = raw.times

    plt.figure(figsize=(15,5))
    fig, ax = plt.subplots(2,2, figsize=(15,5))
    ax[0,0].plot(timescale, L_chan*1e6, label=raw.ch_names[0])
    ax[0,0].set_title(raw.ch_names[0])
    ax[0,0].set_ylabel('LFP amplitude (µV)')
    ax[1,0].plot(timescale, stim_L_chan, color = 'orange', label='Stim left')
    ax[1,0].set_ylabel('Stimulation amplitude (mA)')
    ax[1,0].set_xlabel('Time (s)')

    #ax[0,1].legend()
    ax[0,1].plot(timescale, R_chan*1e6, label=raw.ch_names[1])
    ax[0,1].set_title(raw.ch_names[1])
    ax[1,1].plot(timescale, stim_R_chan, color= 'orange', label='Stim right')
    
    fig.suptitle(f'Raw data and stimulation data - {session_ID}')
    figtitle = f'Raw data and stimulation data - {session_ID}.png'
    plt.savefig(join(saving_path,figtitle), transparent=False)




def plot_psd_log(
        session_ID: str,
        raw: mne.io.Raw,
        freqs_left: np.ndarray,
        psd_left: np.ndarray,
        freqs_right: np.ndarray,
        psd_right: np.ndarray,
        saving_path: str,
        is_filt: bool = False
        ):
    """
    This function plots the power spectral density of the left and right channel
    in a logarithmic scale.

    session_ID: session ID
    raw: raw data (loaded through mne.io.read_raw)
    saving_path: path to save the figure
    """
    
    fig, ax = plt.subplots(2,2,figsize=(10,10))
    #ax[0,0].plot(freqs_left,np.log(psd_left))
    ax[0,0].plot(freqs_left, psd_left*1e12)
    ax[0,0].set_xlabel('Frequency (Hz)')
    #ax[0,0].set_ylabel('log(Power)')
    ax[0,0].set_ylabel('Power (µV²/Hz)')
    ax[0,0].set_title(raw.ch_names[0])
    #ax[0,1].plot(freqs_right,np.log(psd_right))
    ax[0,1].plot(freqs_right, psd_right*1e12)
    ax[0,1].set_xlabel('Frequency (Hz)')
    ax[0,1].set_ylabel('Power (µV²/Hz)')
    ax[0,1].set_title(raw.ch_names[1])
    #ax[1,0].plot(freqs_left,np.log(psd_left))
    ax[1,0].plot(freqs_left, psd_left*1e12)
    ax[1,0].set_xlabel('Frequency (Hz)')
    #ax[1,0].set_ylabel('log(Power)')
    ax[1,0].set_ylabel('Power (µV²/Hz)')
    ax[1,0].set_xlim([0, 50])
    ax[1,0].set_title(raw.ch_names[0])
    #ax[1,1].plot(freqs_right,np.log(psd_right))
    ax[1,1].plot(freqs_right, psd_right*1e12)
    ax[1,1].set_xlabel('Frequency (Hz)')
    ax[1,1].set_ylabel('Power (µV²/Hz)')
    ax[1,1].set_xlim([0, 50])
    ax[1,1].set_title(raw.ch_names[1])
    

    if is_filt:
        fig.suptitle(f'Power Spectral Density - Filtered - {session_ID}')
        figtitle = f'Filtered Power Spectral Density - {session_ID}.png'
    else:
        fig.suptitle(f'Power Spectral Density - Raw - {session_ID}')
        figtitle = f'Raw Power Spectral Density - {session_ID}.png'
    plt.savefig(join(saving_path,figtitle), transparent = False)
    plt.tight_layout()




def plot_stft(
        session_ID: str,
        raw: mne.io.Raw, 
        vmin: int, 
        vmax: int,
        saving_path: str
        ):
    """
    This function plots the short-time Fourier transform of the left and right channel.

    raw: raw data (loaded through mne.io.read_raw)
    vmin: minimum value for the colorbar
    vmax: maximum value for the color
    """

    L_chan = raw.get_data(picks=raw.ch_names[0])[0]
    R_chan = raw.get_data(picks=raw.ch_names[1])[0]

    f_left, t_left, Zxx_left = scipy.signal.stft(
        L_chan, raw.info['sfreq'], nperseg=int(raw.info['sfreq']), noverlap = (int(raw.info['sfreq'])/2), nfft=int(raw.info['sfreq'])
        )
    f_right, t_right, Zxx_right = scipy.signal.stft(
        R_chan, raw.info['sfreq'], nperseg=int(raw.info['sfreq']), noverlap = (int(raw.info['sfreq'])/2), nfft=int(raw.info['sfreq'])
        )
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].imshow(np.log(np.abs(Zxx_left)), aspect='auto', origin='lower', 
                  vmin=vmin, vmax= vmax, extent=[t_left[0], t_left[-1], 
                                                 f_left[0], f_left[-1]])
    axs[0].set_ylim(0,100)
    axs[0].set_title(f'STFT {raw.ch_names[0]}')
    axs[0].set_ylabel('Frequency [Hz]')
    axs[0].set_xlabel('Time [sec]')
    axs[1].imshow(np.log(np.abs(Zxx_right)), aspect='auto', origin='lower', 
                  vmin = vmin, vmax= vmax, extent=[t_right[0], t_right[-1], 
                                                   f_right[0], f_right[-1]])
    axs[1].set_ylim(0,100)
    axs[1].set_title(f'STFT {raw.ch_names[1]}')
    axs[1].set_ylabel('Frequency [Hz]')
    axs[1].set_xlabel('Time [sec]')
    # add the colorbar
    axs[1].figure.colorbar(axs[1].images[0], ax=axs[1], orientation='vertical')
    plt.tight_layout()
    figtitle = f'STFT - {session_ID}.png'
    plt.savefig(join(saving_path, figtitle), transparent=False)




# Time-frequency plot (Short-Time Fourier Transform) with stimulation apmlitude
def plot_stft_stim(session_ID,
                   raw, 
                   is_filt: bool = False,
                   saving_path: str = None, vmin= None, vmax= None, fmin= 0, fmax=90):
        """
        Function performs a Short Time Fourier Transformation to data and plots the spectrograms (TFR plots) 
        with stimulation amplitude on top

        Input:
        - raw: raw EEG/LFP data
        - L_chan (np.array): left channel data
        - R_chan (np.array): right channel data
        - sf: sampling frequency
        - session_id: session id, e.g., sub005 OFF MID
        """

        L_chan = raw.get_data(picks=raw.ch_names[0])[0]
        R_chan = raw.get_data(picks=raw.ch_names[1])[0]

        f_left, t_left, Zxx_left = scipy.signal.stft(
            L_chan, raw.info['sfreq'], nperseg=int(round(raw.info['sfreq'])), noverlap = int(round(raw.info['sfreq'])/2), nfft=int(round(raw.info['sfreq']))
            )
        f_right, t_right, Zxx_right = scipy.signal.stft(
            R_chan, raw.info['sfreq'], nperseg=int(round(raw.info['sfreq'])), noverlap = int(round(raw.info['sfreq'])/2), nfft=int(round(raw.info['sfreq']))
            )
        
        #Plot Spectrograms of both STNs
        fig, axes = plt.subplots(1,2, figsize = (14,6)) 

        ax_c = 0
        stim = 4
        
        for kj in np.array([0,1]):
                ax2 = axes[kj].twinx() # make right axis linked to the left one
                if kj == 1:
                        stim_data = raw.get_data(picks = raw.ch_names[stim])[0] # define stim channel
                        stim_data = stim_data*1e6  # to get milliampers
                        max_stim = np.nanmax(stim_data)
                elif kj == 0:
                        stim_data = raw.get_data(picks = raw.ch_names[stim])[0]
                        stim_data = stim_data*1e6
                        max_stim = np.nanmax(stim_data)

                #Plot STFT data
                if kj == 0:
                        Pxx = np.abs(Zxx_left)
                        t = t_left
                        f = f_left
                else:
                        Pxx = np.abs(Zxx_right)
                        t = t_right
                        f = f_right

                if vmin:
                    im = axes[ax_c].imshow(np.log(Pxx), aspect='auto', origin='lower',
                                            extent=[t[0], t[-1], f[0], f[-1]], cmap='viridis', vmin=vmin, vmax=vmax)
                else:
                    im = axes[ax_c].imshow(np.log(Pxx), aspect='auto', origin='lower',
                                            extent=[t[0], t[-1], f[0], f[-1]], cmap='viridis')
                axes[ax_c].set_ylim(bottom=fmin, top=fmax)
                axes[ax_c].set_xlim(0, raw.n_times / 250)

                # Plot stim channel on top
                ax2.plot(raw.times, stim_data, 'white', linewidth = 3, linestyle = ':')
                ax2.set_yticks(np.arange(0,4.5,0.5))

                # Right y axis label only for second plot
                if kj == 1:
                        ax2.set_ylabel('Stimulation Amplitude [mA]')
                # Left y-axis label only for first plot
                if kj == 0:
                        axes[ax_c].set_ylabel('Frequency [Hz]')
                
                axes[ax_c].set_xlabel('Time [sec]')
                axes[ax_c].set_title(f'{raw.ch_names[kj]} \n {raw.ch_names[kj+4]} \n {max_stim}mA')

                ax_c += 1
                stim += 1
                if kj == 1: 
                        plt.colorbar(im, ax=axes, orientation='horizontal', fraction=0.046, pad=0.07)

                # Add a main title for the entire figure
                if not is_filt:
                    fig.suptitle(f"Raw - {session_ID}", fontsize=16, y=0.99)  # 'y' controls vertical positioning
                elif is_filt:
                    fig.suptitle(f"Filtered - {session_ID}", fontsize=16, y=0.99)  # 'y' controls vertical positioning

        # Allows for text in figure to be modified as text, when saved as PDF!
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        
        if not is_filt:
                figtitle = f'RAW_STFT_stim - {session_ID}.png'
                figtitle_pdf = f'RAW_STFT_stim - {session_ID}.pdf'
        elif is_filt:
                figtitle = f'filtered_STFT_stim - {session_ID}.png'
                figtitle_pdf = f'filtered_STFT_stim - {session_ID}.pdf'
        
        if saving_path is not None:
            #plt.savefig(join(saving_path, figtitle_pdf))
            plt.savefig(join(saving_path, figtitle))





def plot_mean_stft_trials(
        session_ID: str, 
        epochs: mne.Epochs, 
        trials_dict: dict, 
        trial_type: np.ndarray, 
        sf: int, 
        nperseg: int, 
        noverlap: int, 
        vmin: int,
        vmax: int, 
        saving_path: str,
        mean_RT = None
        ):
    """
    This function plots the mean short-time Fourier transform of the left and right channel for a specific trial type.

    session_ID: session ID
    epochs: mne.Epochs object
    trials_dict: dictionary containing the different trial types and their data
    trial_type: array containing the trials of interest
    sf: sampling frequency
    nperseg: number of samples per segment
    noverlap: number of overlapping samples
    vmin: minimum value for the colorbar
    vmax: maximum value for the colorbar
    mean_RT: mean reaction time
    saving_path: path to save the figure
    """

    Zxx1_values_left = []
    Zxx1_values_right = []
    for i in range(0, len(trial_type)):
        f1, t1, Zxx1_left = scipy.signal.stft(
            trial_type[i][0], sf, nperseg=nperseg, noverlap = noverlap, nfft=sf*2
            )
        Zxx1_values_left.append(Zxx1_left)
        f1, t1, Zxx1_right = scipy.signal.stft(
            trial_type[i][1], sf, nperseg=nperseg, noverlap = noverlap, nfft=sf*2
            )
        Zxx1_values_right.append(Zxx1_right)

    for trial_name, trial_data in trials_dict.items():
        if np.array_equal(trial_type, trial_data):  # Compare the arrays
            T_name = trial_name

    mean_Zxx1_left = np.mean(Zxx1_values_left, axis=0)
    mean_Zxx1_right = np.mean(Zxx1_values_right, axis=0)

    fig = plt.figure(figsize=(15, 5))

    # Create a GridSpec for better control over layout
    gs = GridSpec(2, 2, height_ratios=[5, 0.2])  # 5:0.2 ratio to allocate space for colorbar

    # Create the two main subplots in the first row
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Plot on the left
    im0 = ax0.imshow(np.log(np.abs(mean_Zxx1_left)), aspect='auto', origin='lower', 
                     vmin=vmin, vmax=vmax, extent=[epochs.times[0], epochs.times[-1], 
                                                   f1[0], f1[-1]])
    ax0.set_ylim(0, 50)
    ax0.set_title('Left STN')
    ax0.set_ylabel('Frequency [Hz]')
    ax0.set_xlabel('Time [sec]')
    ax0.axvline(0, color='black', linestyle='--', label = 'GO signal')
    if mean_RT is not None:
        ax0.axvline(mean_RT, color='red', linestyle='--', label = 'mean RT')

    # Plot on the right
    im1 = ax1.imshow(np.log(np.abs(mean_Zxx1_right)), aspect='auto', origin='lower', 
                     vmin=vmin, vmax=vmax, extent=[epochs.times[0], epochs.times[-1], 
                                                   f1[0], f1[-1]])
    ax1.set_ylim(0, 50)
    ax1.set_title('Right STN')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_xlabel('Time [sec]')
    ax1.axvline(0, color='black', linestyle='--')
    if mean_RT is not None:
        ax1.axvline(mean_RT, color='red', linestyle='--')

    # Add a colorbar below the two plots (2nd row of GridSpec)
    cax = fig.add_subplot(gs[1, :])  # Span across both columns
    fig.colorbar(
        im1, cax=cax, orientation='horizontal', label='Log(Power) Magnitude', 
        fraction=0.05, pad = 0.2
        )
    fig.legend()
    fig.suptitle(f'Mean STFT Magnitude of {str(T_name)} trials (n={len(trial_type)}) - {session_ID}')
    figtitle = f'Mean STFT Magnitude of {str(T_name)} trials - {session_ID}.png'
    plt.subplots_adjust(hspace=0.3)  # Adjust the space between the plots and colorbar
    plt.savefig(join(saving_path, figtitle), transparent=False)







def plot_power_comparison_between_2_conditions(
        session_ID: str, 
        raw: mne.io.Raw,
        condition1_epochs: mne.Epochs, 
        condition1_name: str, 
        mean_RT_condition1: float, 
        condition2_epochs: mne.Epochs, 
        condition2_name: str,
        mean_RT_condition2: float, 
        channel_n: list, 
        centered_around: str, 
        saving_path: str,
        fmax: int = 50,
        vmin = None,
        vmax = None
        ):
    """
    This function plots the power comparison between two conditions for specific channels of interest.

    session_ID: session ID
    raw: raw data (loaded through mne.io.read_raw)
    condition1_epochs: mne.Epochs object for condition 1
    condition1_name: name of condition 1
    mean_RT_condition1: mean reaction time for condition 1 (in milliseconds)
    condition2_epochs: mne.Epochs object for condition 2
    condition2_name: name of condition 2
    mean_RT_condition2: mean reaction time for condition 2 (in milliseconds)
    channel_n: list of channel indices to plot
    centered_around: event to center the plot around (e.g. 'GO signal')
    saving_path: path to save the figure
    """

    # parameters for tfr computation
    freqs = np.arange(1, fmax, 1)  # define frequencies of interest
    tfr_kwargs = dict(
        method="morlet",
        freqs=freqs,
        n_cycles=freqs/2.0, # different number of cycle per frequency
        decim=2,
        return_itc=False,
        average=False,
    )
    baseline = (-0.5, 0)

    for ch in channel_n:
        ch_name = raw.ch_names[ch]

        percent_change_1, avg_power_1, times = compute_percent_change(condition1_epochs, ch, baseline, **tfr_kwargs)
        percent_change_2, avg_power_2, times = compute_percent_change(condition2_epochs, ch, baseline, **tfr_kwargs)

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax_diff) = plt.subplots(3, 1, figsize=(15, 15))

        # Plot average power for condition 1
        if vmin:
            im1 = ax1.imshow(avg_power_1, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                            aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        else:
            im1 = ax1.imshow(avg_power_1, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                            aspect="auto", origin="lower", cmap="coolwarm")             
        ax1.set_title(f"Average Time-Frequency Power for {condition1_name} trials ({ch_name} - {len(condition1_epochs)} trials)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Frequency (Hz)")
        ax1.axvline(0, color="white", linestyle="--")
        ax1.axvline(mean_RT_condition1/1000, color="green", linestyle="--")
        fig.colorbar(im1, ax=ax1)

        # Plot average power for condition 2
        if vmin:
            im2 = ax2.imshow(avg_power_2, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                            aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        else:
            im2 = ax2.imshow(avg_power_2, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                            aspect="auto", origin="lower", cmap="coolwarm")
        ax2.set_title(f"Average Time-Frequency Power for {condition2_name} trials ({ch_name} - {len(condition2_epochs)} trials)")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.axvline(0, color="white", linestyle="--")
        ax2.axvline(mean_RT_condition2/1000, color="green", linestyle="--")
        fig.colorbar(im2, ax=ax2)
    
        F_obs_plot, F_obs = perform_permutation_cluster_test(percent_change_1, percent_change_2)
        max_F = np.nanmax(abs(F_obs_plot))

        # Plot the cluster-corrected power contrast
        ax_diff.imshow(F_obs, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    aspect="auto", origin="lower", cmap="gray")
        im_diff = ax_diff.imshow(F_obs_plot, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                                aspect="auto", origin="lower", cmap="coolwarm", vmin=-max_F, vmax=max_F)

        ax_diff.set_title(f"{condition1_name} - {condition2_name} trials ({ch_name})")
        ax_diff.set_xlabel("Time (s)")
        ax_diff.set_ylabel("Frequency (Hz)")
        label1 = f"Mean RT {condition1_name}"
        label2 = f"Mean RT {condition2_name}"
        ax_diff.axvline(mean_RT_condition1/1000, color="green", linestyle="--", label=label1)
        ax_diff.axvline(mean_RT_condition2/1000, color="green", linestyle="--", label=label2)
        fig.colorbar(im_diff, ax=ax_diff)

        ax_diff.axvline(0, color="white", linestyle="--", label=centered_around)
        fig.legend(loc='center right')

        plt.tight_layout()
        figtitle = (f"Power Comparison between {condition1_name} and {condition2_name} trials ({ch_name} - {session_ID}).png")
        plt.savefig(join(saving_path,figtitle), transparent=False)


def plot_power_comparison_between_conditions(
    epochs_subsets,
    epochs_lm,
    session_ID,
    filtered_data_lfp,
    mean_RT_dict,
    saving_path,
    vmin,
    vmax,
    fmax: int = 50    
):
    if len(epochs_subsets['GO_successful']) > 0 and len(epochs_subsets['GS_successful']) > 0:
        #plot_power_comparison_between_2_conditions(session_ID, filtered_data_lfp, epochs_subsets['GO_successful'], 'successful GO', mean_RT_dict['GO_successful'], epochs_subsets['GS_successful'], 'successful STOP', mean_RT_dict['GS_unsuccessful'], (0, 1), 'GO signal', saving_path, vmin=vmin, vmax=vmax, fmax=fmax)
        plot_power_comparison_between_2_conditions(session_ID, filtered_data_lfp, epochs_subsets['GS_successful'], 'successful STOP', mean_RT_dict['GS_unsuccessful'], epochs_subsets['GO_successful'], 'successful GO', mean_RT_dict['GO_successful'], (0, 1), 'GO signal', saving_path, vmin=vmin, vmax=vmax, fmax=fmax) 
    if len(epochs_lm['lm_GO']) > 0 and len(epochs_subsets['GS_successful']) > 0:
        plot_power_comparison_between_2_conditions(session_ID, filtered_data_lfp, epochs_subsets['GS_successful'], 'STOP (s)', mean_RT_dict['GS_unsuccessful'], epochs_lm['lm_GO'], 'GO (lm)', mean_RT_dict['lm_GO'], (0, 1), 'GO signal', saving_path, vmin=vmin, vmax=vmax, fmax=fmax) 
    if len(epochs_subsets['GO_successful']) > 0 and len(epochs_subsets['GF_successful']) > 0:
        #plot_power_comparison_between_2_conditions(session_ID, filtered_data_lfp, epochs_subsets['GO_successful'], 'successful GO', mean_RT_dict['GO_successful'], epochs_subsets['GF_successful'], 'successful GO FAST', mean_RT_dict['GF_successful'], (0, 1), 'GO signal', saving_path, vmin=vmin, vmax=vmax, fmax=fmax)
        plot_power_comparison_between_2_conditions(session_ID, filtered_data_lfp, epochs_subsets['GF_successful'], 'successful GO FAST', mean_RT_dict['GF_successful'], epochs_subsets['GO_successful'], 'successful GO', mean_RT_dict['GO_successful'], (0, 1), 'GO signal', saving_path, vmin=vmin, vmax=vmax, fmax=fmax)
    if len(epochs_subsets['GO_successful']) > 0 and len(epochs_subsets['GC_successful']) > 0:
        plot_power_comparison_between_2_conditions(session_ID, filtered_data_lfp, epochs_subsets['GO_successful'], 'successful GO', mean_RT_dict['GO_successful'], epochs_subsets['GC_successful'], 'successful GO CONTINUE', mean_RT_dict['GC_successful'], (0, 1), 'GO signal', saving_path, vmin=vmin, vmax=vmax, fmax=fmax)    
    if len(epochs_subsets['GS_successful']) > 0 and len(epochs_subsets['GC_successful']) > 0:
        plot_power_comparison_between_2_conditions(session_ID, filtered_data_lfp, epochs_subsets['GS_successful'], 'successful STOP', mean_RT_dict['GS_unsuccessful'], epochs_subsets['GC_successful'], 'successful GO CONTINUE', mean_RT_dict['GC_successful'], (0, 1), 'GO signal', saving_path, vmin=vmin, vmax=vmax, fmax=fmax)    



def plot_tfr_success_vs_unsuccess(
        epochs_subsets,
        session_ID,
        filtered_data_lfp,
        mean_RT_dict,
        saving_path,
        vmin,
        vmax, 
        fmax
        ):
    if len(epochs_subsets['GO_successful']) > 0 and len(epochs_subsets['GO_unsuccessful']) > 0:
        plot_power_comparison_between_2_conditions(
            session_ID, filtered_data_lfp, epochs_subsets['GO_successful'], 
            'successful GO', mean_RT_dict['GO_successful'], 
            epochs_subsets['GO_unsuccessful'], 'unsuccessful GO', mean_RT_dict['GO_successful'], (0, 1), 
            'GO signal', saving_path, fmax=fmax, vmin=vmin, vmax=vmax
            )
    if len(epochs_subsets['GF_unsuccessful']) > 0 and len(epochs_subsets['GF_successful']) > 0:
        plot_power_comparison_between_2_conditions(
            session_ID, filtered_data_lfp, epochs_subsets['GF_successful'], 
            'successful GF', mean_RT_dict['GF_successful'], 
            epochs_subsets['GF_unsuccessful'], 'unsuccessful GF', mean_RT_dict['GF_successful'], (0, 1), 
            'GO signal', saving_path, fmax=fmax, vmin=vmin, vmax=vmax
            )
    if len(epochs_subsets['GC_unsuccessful']) > 0 and len(epochs_subsets['GC_successful']) > 0:
        plot_power_comparison_between_2_conditions(
            session_ID, filtered_data_lfp, epochs_subsets['GC_successful'], 
            'successful GC', mean_RT_dict['GC_successful'], 
            epochs_subsets['GC_unsuccessful'], 'unsuccessful GC',  mean_RT_dict['GC_successful'],  (0, 1), 
            'GO signal', saving_path, fmax=fmax, vmin=vmin, vmax=vmax
            )
    if len(epochs_subsets['GS_unsuccessful']) > 0 and len(epochs_subsets['GS_successful']) > 0:
        plot_power_comparison_between_2_conditions(
            session_ID, filtered_data_lfp, epochs_subsets['GS_successful'], 
            'successful STOP', mean_RT_dict['GS_unsuccessful'], 
            epochs_subsets['GS_unsuccessful'], 'unsuccessful STOP', mean_RT_dict['GS_unsuccessful'], (0, 1), 
            'GO signal', saving_path, fmax=fmax, vmin=vmin, vmax=vmax
            )
        
def plot_frequency_maps (frequency_maps_dict, session_ID, condition, saving_path):
    for key, frequency_map in frequency_maps_dict.items():
        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
        #colors = ('red', 'blue', '#20a39e', 'green')
        #colors = plt.colormaps["winter_r"](np.linspace(0, 1, 4))
        if condition == 'DBS OFF':
            colors = ('#20a39e', '#20a39e', '#20a39e', '#20a39e') # DBS OFF
        if condition == 'DBS ON':
            colors = ('#ef5b5b', '#ef5b5b', '#ef5b5b', '#ef5b5b')
        for ((freq_name, fmin, fmax), average), color, ax in zip(
            frequency_map, colors, axes.ravel()[::-1]
        ):
            times = average.times * 1e3
            gfp = np.sum(average.data**2, axis=0)
            gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
            ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
            ax.axhline(0, linestyle="--", color="grey", linewidth=2)
            ci_low, ci_up = bootstrap_confidence_interval(
                average.data, random_state=0, stat_fun=stat_fun
            )
            ci_low = rescale(ci_low, average.times, baseline=(None, 0))
            ci_up = rescale(ci_up, average.times, baseline=(None, 0))
            ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
            ax.grid(True)
            ax.set_ylabel("GFP")
            ax.annotate(
                f"{freq_name} ({fmin:d}-{fmax:d}Hz)",
                xy=(0.95, 0.8),
                horizontalalignment="right",
                xycoords="axes fraction",
            )
            #ax.set_xlim(-1000, 3000)
            ax.axvline(0, color="black", linestyle="--", label="GO signal")

        axes.ravel()[-1].set_xlabel("Time [ms]")
        #saving_path = "C:\\Users\\Juliette\\Research\\Projects\\mSST_analysis\\results\\sub009 DBS ON mSST"
        saving_path = os.path.abspath(saving_path)
        fig.suptitle(f"Global Field Power (GFP) for {key} \n {session_ID}")
        figtitle = f"GFP {key} - {session_ID}.png"
        save_file_path = join(saving_path, figtitle)
        plt.savefig(save_file_path, format='png')




def plot_av_freq_power_by_trial(
        session_ID,
        epochs_subsets,
        raw,
        available_keys,
        tfr_kwargs,
        saving_path,
        mean_RT_dict,
        vmin = None,
        vmax = None,
        apply_baseline: bool = False,
        baseline: tuple = (-0.5, -0.2)
):
    freqs = tfr_kwargs['freqs']
    cmap = 'coolwarm'

    for key in available_keys:
        data = epochs_subsets[key]

        # Compute TFR for each subject and each channel individually
        left_epochs = data.copy().pick([0])
        right_epochs = data.copy().pick([1])

        power_left = left_epochs.compute_tfr(**tfr_kwargs)
        power_right = right_epochs.compute_tfr(**tfr_kwargs)

        percent_change_left = power_left.copy().apply_baseline(baseline=baseline, mode="percent") * 100
        percent_change_right = power_right.copy().apply_baseline(baseline=baseline, mode="percent") * 100

        # Compute min/max ignoring NaNs
        print(f"Range of percent_change_left for {key}: Min = {np.nanmin(percent_change_left.data)}, Max = {np.nanmax(percent_change_left.data)}")
        print(f"Range of percent_change_right for {key}: Min = {np.nanmin(percent_change_right.data)}, Max = {np.nanmax(percent_change_right.data)}")

        # Compute baseline power 
        baseline_power_left = power_left.copy().crop(*baseline).data.mean(axis=-1)  # Average across time in baseline window
        baseline_power_right = power_right.copy().crop(*baseline).data.mean(axis=-1)

        print(f"Baseline power (left): Min = {np.nanmin(baseline_power_left)}, Max = {np.nanmax(baseline_power_left)}")
        print(f"Baseline power (right): Min = {np.nanmin(baseline_power_right)}, Max = {np.nanmax(baseline_power_right)}")

        """
        # Access the underlying data and compute the range
        print(f"Range of percent_change_left for {key}: Min = {percent_change_left.data.min()}, Max = {percent_change_left.data.max()}")
        print(f"Range of percent_change_right for {key}: Min = {percent_change_right.data.min()}, Max = {percent_change_right.data.max()}")

        baseline_power_left = power_left.copy().crop(*baseline).data.mean(axis=-1)  # Average across time in baseline window
        baseline_power_right = power_right.copy().crop(*baseline).data.mean(axis=-1)

        print(f"Baseline power (left): Min = {baseline_power_left.min()}, Max = {baseline_power_left.max()}")
        print(f"Baseline power (right): Min = {baseline_power_right.min()}, Max = {baseline_power_right.max()}")
        """

        # Time-frequency power plots
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        
        # compute average power over trials:
        avg_power_left = np.nanmean(percent_change_left, axis=0)[0]
        avg_power_right = np.nanmean(percent_change_right, axis=0)[0]

        times = power_left.times
        time_indices = (power_left.times >= -0.5) & (power_left.times <= 1.5)
        avg_power_left = avg_power_left[:, time_indices]
        avg_power_right = avg_power_right[:, time_indices]
        times = times[time_indices]
        if vmin == None:
            im0 = ax[0].imshow(avg_power_left, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                            aspect="auto", origin="lower", cmap=cmap)
            im1 = ax[1].imshow(avg_power_right, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                            aspect="auto", origin="lower", cmap=cmap)
        else:
            im0 = ax[0].imshow(avg_power_left, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                                vmin = vmin, vmax = vmax,
                                aspect="auto", origin="lower", cmap=cmap)
            im1 = ax[1].imshow(avg_power_right, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                                vmin = vmin, vmax = vmax,
                                aspect="auto", origin="lower", cmap=cmap)
                                
        ax[0].set_title(f"{raw.ch_names[0]}")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Frequency (Hz)")
        ax[0].axvline(0, color="white", linestyle="--")
        ax[1].set_title(f"{raw.ch_names[1]}")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Frequency (Hz)")
        ax[1].axvline(0, color="white", linestyle="--")

        # Add mean RT lines
        if 'GO' in key:
            mean_RT = mean_RT_dict['GO_successful']
        elif 'GS' in key:
            mean_RT = mean_RT_dict['GS_unsuccessful']
        elif 'GF' in key:
            mean_RT = mean_RT_dict['GF_successful']
        elif 'GC' in key:
            mean_RT = mean_RT_dict['GC_successful']

        ax[0].axvline(mean_RT / 1000, color="red", linestyle="--")
        ax[1].axvline(mean_RT / 1000, color="red", linestyle="--")

        # Add colorbar
        cbar = fig.colorbar(im0, ax=ax[0], pad=0.1, fraction=0.04, label = 'Mean % Change (from baseline)')
        cbar = fig.colorbar(im1, ax=ax[1], pad=0.1, fraction=0.04, label = 'Mean % Change (from baseline)')

        ax[0].legend(
            loc="upper right",
            labels=["GO signal", "Mean RT"],
            fontsize="small",
        )
        ax[1].legend(
            loc="upper right",
            labels=["GO signal", "Mean RT"],
            fontsize="small",
        )

        fig.suptitle(
                f'Average Time-Frequency Power for {key} trials'
                f'\n ({len(epochs_subsets[key])} trials)'
                f'\n {session_ID}')

        # Save the time-frequency plot
        if apply_baseline:
            figtitle = f"Baseline corrected - Average_Time_Frequency_Power_{key}_trials.png"
        else:
            figtitle = f"No baseline correction applied - Average_Time_Frequency_Power_{key}_trials.png"
        plt.savefig(join(saving_path, figtitle), transparent=False)
        #plt.show()
        #plt.close()




def plot_amplitude_and_difference_from_json(
        json_saving_path,
        session_ID,
        freq_band,
        saving_path
):
    # Load JSON data
    filepath = os.path.join(json_saving_path, f"{session_ID}.json")
    with open(filepath, "r") as f:
        session_data = json.load(f)

    # Extract available events
    all_keys = session_data.keys()
    event_keys = [key for key in all_keys if f"{freq_band}_amp" in key]
    times_key = next((key for key in all_keys if "amp_times" in key), None)

    if times_key is None:
        raise KeyError("No key containing 'amp_times' found in the JSON file.")

    times = session_data[times_key]

    event_pairs = [
        ('GO_successful', 'GO_unsuccessful'),
        ('GC_successful', 'GC_unsuccessful'),
        ('GF_successful', 'GF_unsuccessful'),
        ('GS_successful', 'GS_unsuccessful'),
        ('GO_successful', 'GF_successful'),
        ('GC_successful', 'GS_successful'),
        ('GS_successful', 'lm_GO')
    ]

    # Loop through event pairs
    for event1, event2 in event_pairs:
        for channel in ["Left_STN", "Right_STN"]:  # Adjust based on available channels
            key1 = f"{channel}_{freq_band}_amp_{event1}"
            key2 = f"{channel}_{freq_band}_amp_{event2}"
            
            if key1 not in session_data or key2 not in session_data:
                print(f"Skipping {event1} vs {event2} for {channel} (data missing).")
                continue

            # Load data
            beta_amp_1 = np.array(session_data[key1])
            beta_amp_2 = np.array(session_data[key2])

            # Compute difference
            difference = beta_amp_1 - beta_amp_2

            # Create the figure with two subplots
            fig, axes = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)

            # First subplot: Evoked responses
            axes[0].plot(times, beta_amp_1, color='tab:green', label=f"{channel} {event1}")
            axes[0].plot(times, beta_amp_2, color='tab:blue', label=f"{channel} {event2}")
            axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
            axes[0].axvline(0, color='black', linestyle='-', linewidth=0.8, label="GO signal")
            axes[0].set_title(f"{freq_band} Amplitude Response - {channel} - {session_ID}")
            axes[0].set_ylabel(f"{freq_band} amplitude (%)")
            #axes[0].legend()

            # Second subplot: Difference
            axes[1].plot(times, difference, color='tab:red', label=f"_Difference ({event1} - {event2})")
            axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
            axes[1].axvline(0, color='black', linestyle='-', linewidth=0.8, label="_GO signal")
            axes[1].set_title(f"Difference between {event1} and {event2}")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel(f"{freq_band} amplitude (%)")
            #axes[1].legend()

            axes[0].set_xlim(-0.5, 1.5)
            axes[1].set_xlim(-0.5, 1.5)

            fig.legend(loc='upper right', bbox_to_anchor=(1.3, 0.6))
            fig.tight_layout()

            # Save figure
            figtitle = f"{freq_band} Amplitude Response and Difference - {session_ID} - {channel} - {event1} vs {event2}.png"
            plt.savefig(join(saving_path, figtitle), transparent=False, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {figtitle}")
            #plt.show()





# def perc_pow_diff_on_off_contrast(
#         sub_dict_ON_OFF,
#         sub_dict_ON_OFF_cond2,
#         RT_dict_ON_OFF,
#         tfr_args,
#         cond1,
#         cond2,
#         t_min_max: list,
#         vmin_vmax: list,
#         cluster_results_dict: dict,
#         condition: str,
#         saving_path: str = None,
#         show_fig: bool = None,
#         ADD_RT: bool = True,
#         save_as: str = 'png'
#         ):
    
#     all_sub_RT_ON_cond1 = []
#     all_sub_RT_ON_cond2 = []
#     all_sub_RT_OFF_cond1 = []
#     all_sub_RT_OFF_cond2 = []
#     all_diff_left = []
#     all_diff_right = []
#     all_diff_both = []
#     subs_included = []

#     epoch_type1 = cond1.split('_')[0]
#     epoch_type2 = cond2.split('_')[0]
#     outcome_str1 = cond1.split('_')[1] 
#     outcome_str2 = cond2.split('_')[1]

#     outcome1 = 1.0 if outcome_str1 == 'successful' else 0.0
#     outcome2 = 1.0 if outcome_str2 == 'successful' else 0.0

#     # Collect epoch data for each condition
#     for sub in sub_dict_ON_OFF.keys():
#         print("Now processing: ", sub, condition)
#         for subject, epochs in sub_dict_ON_OFF[sub].items():
#             print(subject)
#             if 'DBS ON' in subject:
#                 type_mask1 = epochs.metadata["event"] == epoch_type1
#                 outcome_mask1 = epochs.metadata["key_resp_experiment.corr"] == outcome1
#                 data_ON_cond1 = epochs[type_mask1 & outcome_mask1]   
            
#                 epochs_cond2_ON = sub_dict_ON_OFF_cond2[sub][subject]
#                 type_mask2 = epochs_cond2_ON.metadata["event"] == epoch_type2
#                 outcome_mask2 = epochs_cond2_ON.metadata["key_resp_experiment.corr"] == outcome2
#                 data_ON_cond2 = epochs_cond2_ON[type_mask2 & outcome_mask2]

#                 if not (cond1 == 'GS_successful' or cond1 == 'stop_successful'):
#                     RT_ON_cond1 = epochs.metadata['key_resp_experiment.rt'].mean() * 1000
#                     all_sub_RT_ON_cond1.append(RT_ON_cond1)
#                     print(RT_ON_cond1)
#                 if not (cond2 == 'GS_successful' or cond2 == 'stop_successful'):
#                     RT_ON_cond2 = epochs_cond2_ON.metadata['key_resp_experiment.rt'].mean() * 1000
#                     all_sub_RT_ON_cond2.append(RT_ON_cond2)


#                 left_epochs_ON_cond1, right_epochs_ON_cond1 = data_ON_cond1.copy().pick(['Left_STN']), data_ON_cond1.copy().pick(['Right_STN'])
#                 power_left_ON_cond1 = left_epochs_ON_cond1.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
#                 power_right_ON_cond1 = right_epochs_ON_cond1.compute_tfr(**tfr_args)

#                 mean_power_left_ON_cond1 = np.nanmean(power_left_ON_cond1.data, axis=0).squeeze() # shape: (n freqs, n times)
#                 mean_power_right_ON_cond1 = np.nanmean(power_right_ON_cond1.data, axis=0).squeeze()
                
#                 left_epochs_ON_cond2, right_epochs_ON_cond2 = data_ON_cond2.copy().pick(['Left_STN']), data_ON_cond2.copy().pick(['Right_STN'])
#                 power_left_ON_cond2 = left_epochs_ON_cond2.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
#                 power_right_ON_cond2 = right_epochs_ON_cond2.compute_tfr(**tfr_args)

#                 mean_power_left_ON_cond2 = np.nanmean(power_left_ON_cond2.data, axis=0).squeeze() # shape: (n freqs, n times)
#                 mean_power_right_ON_cond2 = np.nanmean(power_right_ON_cond2.data, axis=0).squeeze()

#                 times = power_left_ON_cond1.times * 1000
#                 freqs = power_left_ON_cond1.freqs

#                 # Define baseline period for percentage change calculation
#                 baseline_indices = (times >= -500) & (times <= -200)

#                 # Percentage change for condition 1
#                 baseline_power_left_ON_cond1 = np.nanmean(mean_power_left_ON_cond1[:, baseline_indices], axis=1, keepdims=True)
#                 percentage_change_left_ON_cond1 = (mean_power_left_ON_cond1 - baseline_power_left_ON_cond1) / baseline_power_left_ON_cond1 * 100
#                 baseline_power_right_ON_cond1 = np.nanmean(mean_power_right_ON_cond1[:, baseline_indices], axis=1, keepdims=True)
#                 percentage_change_right_ON_cond1 = (mean_power_right_ON_cond1 - baseline_power_right_ON_cond1) / baseline_power_right_ON_cond1 * 100
        
#                 baseline_power_left_ON_cond2 = np.nanmean(mean_power_left_ON_cond2[:, baseline_indices], axis=1, keepdims=True)
#                 percentage_change_left_ON_cond2 = (mean_power_left_ON_cond2 - baseline_power_left_ON_cond2) / baseline_power_left_ON_cond2 * 100
#                 baseline_power_right_ON_cond2 = np.nanmean(mean_power_right_ON_cond2[:, baseline_indices], axis=1, keepdims=True)
#                 percentage_change_right_ON_cond2 = (mean_power_right_ON_cond2 - baseline_power_right_ON_cond2) / baseline_power_right_ON_cond2 * 100
        
#                 percentage_change_left_ON_contrast = percentage_change_left_ON_cond1 - percentage_change_left_ON_cond2
#                 percentage_change_right_ON_contrast = percentage_change_right_ON_cond1 - percentage_change_right_ON_cond2

#             if 'DBS OFF' in subject:
#                 data_OFF_cond1 = epochs[cond1]
#                 if not (cond1 == 'GS_successful' or cond1 == 'stop_successful'):
#                     RT_OFF_cond1 = RT_dict_ON_OFF[sub][subject][cond1]
#                     all_sub_RT_OFF_cond1.append(RT_OFF_cond1)
#                     print(RT_OFF_cond1)
#                 epochs_cond2_OFF = sub_dict_ON_OFF_cond2[sub][subject]
#                 data_OFF_cond2 = epochs_cond2_OFF[cond2]
#                 if not cond2 == 'stop_successful':
#                     RT_OFF_cond2 = RT_dict_ON_OFF[sub][subject][cond2]
#                     all_sub_RT_OFF_cond2.append(RT_OFF_cond2)

#                 left_epochs_OFF_cond1, right_epochs_OFF_cond1 = data_OFF_cond1.copy().pick(['Left_STN']), data_OFF_cond1.copy().pick(['Right_STN'])
#                 power_left_OFF_cond1 = left_epochs_OFF_cond1.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
#                 power_right_OFF_cond1 = right_epochs_OFF_cond1.compute_tfr(**tfr_args)

#                 mean_power_left_OFF_cond1 = np.nanmean(power_left_OFF_cond1.data, axis=0).squeeze() # shape: (n freqs, n times)
#                 mean_power_right_OFF_cond1 = np.nanmean(power_right_OFF_cond1.data, axis=0).squeeze()
                
#                 left_epochs_OFF_cond2, right_epochs_OFF_cond2 = data_OFF_cond2.copy().pick(['Left_STN']), data_OFF_cond2.copy().pick(['Right_STN'])
#                 power_left_OFF_cond2 = left_epochs_OFF_cond2.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
#                 power_right_OFF_cond2 = right_epochs_OFF_cond2.compute_tfr(**tfr_args)

#                 mean_power_left_OFF_cond2 = np.nanmean(power_left_OFF_cond2.data, axis=0).squeeze() # shape: (n freqs, n times)
#                 mean_power_right_OFF_cond2 = np.nanmean(power_right_OFF_cond2.data, axis=0).squeeze()

#                 times = power_left_OFF_cond1.times * 1000
#                 freqs = power_left_OFF_cond1.freqs

#                 # Define baseline period for percentage change calculation
#                 baseline_indices = (times >= -500) & (times <= -200)

#                 # Percentage change for condition 1
#                 baseline_power_left_OFF_cond1 = np.nanmean(mean_power_left_OFF_cond1[:, baseline_indices], axis=1, keepdims=True)
#                 percentage_change_left_OFF_cond1 = (mean_power_left_OFF_cond1 - baseline_power_left_OFF_cond1) / baseline_power_left_OFF_cond1 * 100

#                 baseline_power_right_OFF_cond1 = np.nanmean(mean_power_right_OFF_cond1[:, baseline_indices], axis=1, keepdims=True)
#                 percentage_change_right_OFF_cond1 = (mean_power_right_OFF_cond1 - baseline_power_right_OFF_cond1) / baseline_power_right_OFF_cond1 * 100
        
#                 # Percentage change for condition 2
#                 baseline_power_left_OFF_cond2 = np.nanmean(mean_power_left_OFF_cond2[:, baseline_indices], axis=1, keepdims=True)
#                 percentage_change_left_OFF_cond2 = (mean_power_left_OFF_cond2 - baseline_power_left_OFF_cond2) / baseline_power_left_OFF_cond2 * 100

#                 baseline_power_right_OFF_cond2 = np.nanmean(mean_power_right_OFF_cond2[:, baseline_indices], axis=1, keepdims=True)
#                 percentage_change_right_OFF_cond2 = (mean_power_right_OFF_cond2 - baseline_power_right_OFF_cond2) / baseline_power_right_OFF_cond2 * 100
    
#                 percentage_change_left_OFF_contrast = percentage_change_left_OFF_cond1 - percentage_change_left_OFF_cond2
#                 percentage_change_right_OFF_contrast = percentage_change_right_OFF_cond1 - percentage_change_right_OFF_cond2


#         diff_left = percentage_change_left_ON_contrast - percentage_change_left_OFF_contrast
#         #diff_left = percentage_change_left_OFF_contrast - percentage_change_left_ON_contrast
#         diff_right = percentage_change_right_ON_contrast - percentage_change_right_OFF_contrast
#         #diff_right = percentage_change_right_OFF_contrast - percentage_change_right_ON_contrast
#         all_diff_left.append(diff_left)
#         all_diff_right.append(diff_right)
#         all_diff_both.append(diff_left)
#         all_diff_both.append(diff_right)

#         subs_included.append(sub)

#     print(f'Subs included in analyses: \n {subs_included}')

#     all_diff_left_array = np.array(all_diff_left)  # shape: (n sub, n freqs, n times)
#     all_diff_right_array = np.array(all_diff_right)
#     all_diff_both_array = np.array(all_diff_both) # shape: (n sub x 2, n freqs, n times)

#     time_indices = (times >= t_min_max[0]) & (times <= t_min_max[1])
#     sliced_times = times[time_indices]
#     all_diff_left_array_sliced = all_diff_left_array[:,:,time_indices]
#     all_diff_right_array_sliced = all_diff_right_array[:,:,time_indices]
#     all_diff_both_array_sliced = all_diff_both_array[:,:,time_indices]

#     n_obs = all_diff_left_array.shape[0]
#     print(n_obs)
#     pval = 0.05
#     df = n_obs - 1
#     #threshold = scipy.stats.t.ppf(1-pval / 2, df) # two-tailed distribution
#     threshold = None
#     n_permutations = 1000


#     # Compute permutation cluster test for the left stn
#     T_obs_left, clusters_left, cluster_p_values_left, H0_left = mne.stats.permutation_cluster_1samp_test(
#     all_diff_left_array_sliced, n_permutations=n_permutations,
#     threshold=threshold, tail=0,
#     out_type= "mask", seed=11111, verbose=True)
#     print(f"p_values: {cluster_p_values_left}")
#     print(f"P_values shape: {cluster_p_values_left.shape}")

#     print("Clusters for Left STN")
#     identify_significant_clusters(
#         cluster_p_values_left, 
#         clusters_left,
#         sliced_times,
#         T_obs_left,
#         pval,
#         tfr_args,
#         cluster_results_dict,
#         condition,
#         'Left'
#         )
    
#     # Compute permutation cluster test for the right stn
#     T_obs_right, clusters_right, cluster_p_values_right, H0_right = mne.stats.permutation_cluster_1samp_test(
#     all_diff_right_array_sliced, n_permutations=n_permutations,
#     threshold=threshold, tail=0,
#     out_type= "mask", seed=11111, verbose=True)
#     print(f"p_values: {cluster_p_values_right}")
#     print(f"P_values shape: {cluster_p_values_right.shape}")

#     print("Clusters for Right STN")
#     identify_significant_clusters(
#         cluster_p_values_right, 
#         clusters_right,
#         sliced_times,
#         T_obs_right,
#         pval,
#         tfr_args,
#         cluster_results_dict,
#         condition,
#         'Right'
#         )

#     # Compute permutation cluster test for the left + right stn
#     T_obs_both, clusters_both, cluster_p_values_both, H0_both = mne.stats.permutation_cluster_1samp_test(
#     all_diff_both_array_sliced, n_permutations=n_permutations,
#     threshold=threshold, tail=0,
#     out_type= "mask", seed=11111, verbose=True)
#     print(f"p_values: {cluster_p_values_both}")
#     print(f"P_values shape: {cluster_p_values_both.shape}")

#     print("Clusters for Both STN")
#     identify_significant_clusters(
#         cluster_p_values_both, 
#         clusters_both,
#         sliced_times,
#         T_obs_both,
#         pval,
#         tfr_args,
#         cluster_results_dict,
#         condition,
#         'Both'
#         )


#     # Average the percentage signal changes across subjects for left STN and for right STN
#     avg_diff_left = np.nanmean(all_diff_left_array_sliced, axis=0)  # shape: (n freqs, n times)
#     print(f"avg diff left shape: {avg_diff_left.shape}")
#     avg_diff_right = np.nanmean(all_diff_right_array_sliced, axis=0)
#     avg_diff_both = np.nanmean(all_diff_both_array_sliced, axis=0)

#     # Slicing TFR data to include only the t_min, t_max time range
#     #time_indices = np.logical_and(times >= t_min_max[0], times <= t_min_max[1])

#     #sliced_data_left = avg_diff_left[:, time_indices]
#     #sliced_data_right = avg_diff_right[:, time_indices]
#     #sliced_data_both = avg_diff_both[:, time_indices]


#     ################
#     ### PLOTTING ###
#     ################    

#     # Create a figure with two subplots for Left and Right STN
#     fig, (ax_left, ax_right, ax_both) = plt.subplots(1, 3, figsize=(20, 8))

#     # Figure title for n_subjects
#     sub_num = len(all_diff_left)

#     if sub_num > 1:
#         #fig.suptitle(f"Power difference DBS OFF - DBS ON {condition}, nSub = {sub_num}")
#         fig.suptitle(f"Power difference DBS ON - DBS OFF {condition}, nSub = {sub_num}")
#     else:
#         #fig.suptitle(f"Power difference DBS OFF - DBS ON {condition}, {subject[:6]}")
#         fig.suptitle(f"Power difference DBS ON - DBS OFF {condition}, {subject[:6]}")


#     # Plot the percentage change difference for Left STN
#     im_left = ax_left.imshow(avg_diff_left, aspect='auto', origin='lower', 
#                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                             cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    
#     print(T_obs_left.shape)
#     for c, p_val in zip(clusters_left, cluster_p_values_left):
#         if p_val <= pval:
#             mask_L = np.zeros_like(T_obs_left, dtype=bool) 
#             mask_L[c] = True
#             ax_left.contour(mask_L, levels=[0.5], colors='black', linewidths=1.5,
#                             extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
    

#     ax_left.set_xlabel('Time from GO cue (ms)')
#     ax_left.set_ylabel('Frequency (Hz)')
#     ax_left.set_title(f'Left STN - {condition}')
#     fig.colorbar(im_left, ax=ax_left, label='Mean % Change (from baseline)')

#     # Plot the percentage change difference for Right STN
#     im_right = ax_right.imshow(avg_diff_right, aspect='auto', origin='lower', 
#                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                             cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    
#     for c, p_val in zip(clusters_right, cluster_p_values_right):
#         if p_val <= pval:
#             mask_R = np.zeros_like(T_obs_right, dtype=bool)
#             mask_R[c] = True
#             ax_right.contour(mask_R, levels=[0.5], colors='black', linewidths=1.5,
#                             extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
    

#     ax_right.set_xlabel('Time from GO cue (ms)')
#     ax_right.set_ylabel('Frequency (Hz)')
#     ax_right.set_title(f'Right STN - {condition}')
#     fig.colorbar(im_right, ax=ax_right, label='Mean % Change (from baseline)')

#     # Plot the percentage change difference for Left + Right STN
#     im_both = ax_both.imshow(avg_diff_both, aspect='auto', origin='lower', 
#                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                             cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    
#     for c, p_val in zip(clusters_both, cluster_p_values_both):
#         if p_val <= pval:
#             mask_R = np.zeros_like(T_obs_both, dtype=bool)
#             mask_R[c] = True
#             ax_both.contour(mask_R, levels=[0.5], colors='black', linewidths=1.5,
#                             extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
    

#     ax_both.set_xlabel('Time from GO cue (ms)')
#     ax_both.set_ylabel('Frequency (Hz)')
#     ax_both.set_title(f'Left + Right STN - {condition}')
#     fig.colorbar(im_both, ax=ax_both, label='Mean % Change (from baseline)')

#     print(f" {cond1} RT ON: {all_sub_RT_ON_cond1}")
#     print(f" {cond1} RT OFF: {all_sub_RT_OFF_cond1}")
#     print(f" {cond2} RT ON: {all_sub_RT_ON_cond2}")
#     print(f" {cond2} RT OFF: {all_sub_RT_OFF_cond2}")

#     if len(subs_included) > 1:
#         mean_RT_ON_cond1 = np.mean(all_sub_RT_ON_cond1)
#         mean_RT_ON_cond2 = np.mean(all_sub_RT_ON_cond2)
#         mean_RT_OFF_cond1 = np.mean(all_sub_RT_OFF_cond1)
#         mean_RT_OFF_cond2 = np.mean(all_sub_RT_OFF_cond2)
#         #mean_SSD = np.mean(all_sub_SSD)
#         #mean_SSRT = np.mean(all_sub_SSRT)
#     #else:
#         #mean_RT_ON = all_sub_RT_ON[0]
#         #mean_RT_OFF = all_sub_RT_OFF[0]
#         #mean_SSD = all_sub_SSD[0]
#         #mean_SSRT = all_sub_SSRT[0]
    
#     ax_left.axvline(mean_RT_ON_cond1, color='black', linestyle='--')
#     ax_right.axvline(mean_RT_ON_cond1, color='black', linestyle='--')
#     ax_both.axvline(mean_RT_ON_cond1, color='black', linestyle='--', label=f'Mean RT ON {cond1}')
#     ax_left.axvline(mean_RT_OFF_cond1, color='grey', linestyle='--')
#     ax_right.axvline(mean_RT_OFF_cond1, color='grey', linestyle='--')
#     ax_both.axvline(mean_RT_OFF_cond1, color='grey', linestyle='--', label=f'Mean RT OFF {cond1}')

#     ax_left.axvline(mean_RT_ON_cond2, color='blue', linestyle='--')
#     ax_right.axvline(mean_RT_ON_cond2, color='blue', linestyle='--')
#     ax_both.axvline(mean_RT_ON_cond2, color='blue', linestyle='--', label=f'Mean RT ON {cond2}')
#     ax_left.axvline(mean_RT_OFF_cond2, color='green', linestyle='--')
#     ax_right.axvline(mean_RT_OFF_cond2, color='green', linestyle='--')
#     ax_both.axvline(mean_RT_OFF_cond2, color='green', linestyle='--', label=f'Mean RT OFF {cond2}')

#     #ax_left.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')
#     #ax_right.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')    
#     #ax_both.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')    
#     #ax_left.axvline(mean_SSRT, color='blue', linestyle='--', label='Mean SSRT')
#     #ax_right.axvline(mean_SSRT, color='blue', linestyle='--', label='Mean SSRT')
#     #ax_both.axvline(mean_SSRT, color='blue', linestyle='--', label='Mean SSRT')    

#     fig.legend() 

#     # Allows for text in figure to be modified as text, when saved as PDF!
#     matplotlib.rcParams['pdf.fonttype'] = 42
#     matplotlib.rcParams['ps.fonttype'] = 42

#     if sub_num > 1:
#         # Save the figure if a saving path is provided
#         #figtitle = f"Power_diff_DBS_OFF-DBS_ON_{condition}.png"
#         #figtitle_pdf = f"Power_diff_DBS_OFF-DBS_ON_{condition}.pdf"
#         figtitle = f"Power_diff_DBS_ON-DBS_OFF_{condition}.png"
#         figtitle_pdf = f"Power_diff_DBS_ON-DBS_OFF_{condition}.pdf"
#     else:
#         # Save the figure if a saving path is provided
#         #figtitle = f"{subject[:6]}_Power_diff_DBS_OFF-DBS_ON_{condition}.png"
#         #figtitle_pdf = f"{subject[:6]}_Power_diff_DBS_OFF-DBS_ON_{condition}.pdf"
#         figtitle = f"{subject[:6]}_Power_diff_DBS_ON-DBS_OFF_{condition}.png"
#         figtitle_pdf = f"{subject[:6]}_Power_diff_DBS_ON-DBS_OFF_{condition}.pdf"

#     if saving_path is not None:
#         if save_as == 'png':
#             plt.savefig(join(saving_path, figtitle), transparent=False)
#         else:
#             plt.savefig(join(saving_path, figtitle_pdf), transparent=False)

#     if show_fig == True:
#         plt.show()
#     else:
#         plt.close('all')

#     return cluster_results_dict




# def perc_pow_diff_cond2(
#         sub_dict, 
#         sub_dict_lm_GO, 
#         RT_dict, 
#         stats_dict,
#         dbs_status:str, 
#         tfr_args, 
#         t_min_max:list, 
#         vmin_vmax:list,
#         epoch_cond1:str, 
#         epoch_cond2:str, 
#         cluster_results_dict: dict,
#         condition: str,
#         saving_path: str=None, 
#         show_fig: bool = None
#         ):
        
#     """
#     Calculates % power change for the two specified conditions and subtracts epoch_cond2 from epoch_cond1, giving percentage change unique for epoch_cond1. 
#     Loops through all subs in sub_dict. 

#     Input:
#     - sub_dict: dict. containing all epochs (cue or feedback)
#     - dbs_status: "DBS ON" or "DBS OFF" 
#     - tfr_args: TFR parameters
#     - tmin, tmax: epoch slicing
#     - epoch_cond1: Epoch of interest "Win_cue", "Loss_cue"
#     - epoch_cond2: Baseline epoch, e.g., "Neutral_cue" to subtract from main epoch. 
#     - saving_path: Path where plots will be saved. If None figures are not saved. 
#     - show_fig: Defaults to None and figure isn't shown, if True figure is shown.


#     THIS FUNCTION IS A VARIANT OF perc_pow_diff_cond BUT TAILORED TO THE ANALYSIS
#     AND COMPARISON OF LATENCY-MATCHED GO TRIALS WITH SUCCESSFUL STOP TRIALS ONLY

#     """
    
#     all_diff_left = []
#     all_diff_right = []
#     all_diff_both = []
#     all_sub_RT = []
#     all_sub_SSD = []
#     all_sub_SSRT = []

#     subs_included = []

#     # Collect epoch data for each condition
#     for subject, epochs in sub_dict.items():
#         if dbs_status in subject:            
#             # Epoch condition 1
#             if epoch_cond1 in epochs.event_id:
#                 data1 = epochs[epoch_cond1]
#                 sub_SSD = stats_dict[subject]['mean SSD (ms)']
#                 sub_SSRT = stats_dict[subject]['mean SSD (ms)'] + stats_dict[subject]['SSRT (ms)']
#                 all_sub_SSD.append(sub_SSD)
#                 all_sub_SSRT.append(sub_SSRT)
#             else:
#                 print(f"Condition {epoch_cond1} not found in subject {subject}")
#                 continue

#             # Epoch condition 2
#             epochs2 = sub_dict_lm_GO[subject]

#             if epoch_cond2 in epochs2.event_id:
#                 data2 = epochs2[epoch_cond2]
#                 sub_RT = RT_dict[subject][epoch_cond2]
#                 all_sub_RT.append(sub_RT)
#             else:
#                 print(f"Condition {epoch_cond2} not found in subject {subject}")

#             # Pick the left and right channels
#             left_epochs1, right_epochs1 = data1.copy().pick(['Left_STN']), data1.copy().pick(['Right_STN'])
#             left_epochs2, right_epochs2 = data2.copy().pick(['Left_STN']), data2.copy().pick(['Right_STN'])

#             # Compute TFR using Morlet wavelets
#             power_left_ep1 = left_epochs1.compute_tfr(**tfr_args)  # shape: (n epochs, n channels=1, n freqs, n times)
#             power_right_ep1 = right_epochs1.compute_tfr(**tfr_args)

#             mean_power_left_ep1 = np.nanmean(power_left_ep1.data, axis=0).squeeze() # shape: (n freqs, n times)
#             mean_power_right_ep1 = np.nanmean(power_right_ep1.data, axis=0).squeeze()

#             power_left_ep2 = left_epochs2.compute_tfr(**tfr_args)
#             power_right_ep2 = right_epochs2.compute_tfr(**tfr_args)

#             mean_power_left_ep2 = np.nanmean(power_left_ep2.data, axis=0).squeeze()
#             mean_power_right_ep2 = np.nanmean(power_right_ep2.data, axis=0).squeeze()            

#             times = power_left_ep1.times * 1000
#             freqs = power_left_ep1.freqs

#             # Define baseline period for percentage change calculation
#             baseline_indices = (times >= -500) & (times <= -200)

#             # Percentage change for condition 1
#             baseline_power_left1 = np.nanmean(mean_power_left_ep1[:, baseline_indices], axis=1, keepdims=True)
#             percentage_change_left1 = (mean_power_left_ep1 - baseline_power_left1) / baseline_power_left1 * 100

#             baseline_power_right1 = np.nanmean(mean_power_right_ep1[:, baseline_indices], axis=1, keepdims=True)
#             percentage_change_right1 = (mean_power_right_ep1 - baseline_power_right1) / baseline_power_right1 * 100

#             # Percentage change for condition 2
#             baseline_power_left2 = np.nanmean(mean_power_left_ep2[:,baseline_indices], axis=1, keepdims=True)
#             percentage_change_left2 = (mean_power_left_ep2 - baseline_power_left2) / baseline_power_left2 * 100
            
#             baseline_power_right2 = np.nanmean(mean_power_right_ep2[:, baseline_indices], axis=1, keepdims=True)
#             percentage_change_right2 = (mean_power_right_ep2 - baseline_power_right2) / baseline_power_right2 * 100

#             # Differences between Cond1 and Cond2 left and right STN
#             diff_left = percentage_change_left1 - percentage_change_left2
#             diff_right = percentage_change_right1 - percentage_change_right2

#             all_diff_left.append(diff_left)
#             all_diff_right.append(diff_right)  
#             all_diff_both.append(diff_left)
#             all_diff_both.append(diff_right)

#             subs_included.append(subject)

#     print(f'Subs included in analyses: \n {subs_included}')

#     all_diff_left_array = np.array(all_diff_left)  # shape: (n sub, n freqs, n times)
#     all_diff_right_array = np.array(all_diff_right)
#     all_diff_both_array = np.array(all_diff_both) # shape: (n sub x 2, n freqs, n times)

#     time_indices = (times >= t_min_max[0]) & (times <= t_min_max[1])
#     sliced_times = times[time_indices]
#     all_diff_left_array_sliced = all_diff_left_array[:,:,time_indices]
#     all_diff_right_array_sliced = all_diff_right_array[:,:,time_indices]
#     all_diff_both_array_sliced = all_diff_both_array[:,:,time_indices]

#     n_obs = all_diff_left_array.shape[0]
#     print(n_obs)
#     pval = 0.05
#     df = n_obs - 1
#     #threshold = scipy.stats.t.ppf(1-pval / 2, df) # two-tailed distribution
#     threshold = None
#     n_permutations = 1000


#     # Compute permutation cluster test for the left stn
#     T_obs_left, clusters_left, cluster_p_values_left, H0_left = mne.stats.permutation_cluster_1samp_test(
#     all_diff_left_array_sliced, n_permutations=n_permutations,
#     threshold=threshold, tail=0,
#     out_type= "mask", seed=11111, verbose=True)
#     print(f"p_values: {cluster_p_values_left}")
#     print(f"P_values shape: {cluster_p_values_left.shape}")


#     print("Clusters for Left STN")
#     cluster_results_dict = identify_significant_clusters(
#         cluster_p_values_left, 
#         clusters_left,
#         sliced_times,
#         T_obs_left,
#         pval,
#         tfr_args,
#         cluster_results_dict,
#         condition,
#         "Left"
#         )

#     # Compute permutation cluster test for the right stn
#     T_obs_right, clusters_right, cluster_p_values_right, H0_right = mne.stats.permutation_cluster_1samp_test(
#     all_diff_right_array_sliced, n_permutations=n_permutations,
#     threshold=threshold, tail=0,
#     out_type= "mask", seed=11111, verbose=True)
#     print(f"p_values: {cluster_p_values_right}")
#     print(f"P_values shape: {cluster_p_values_right.shape}")

#     print("Clusters for Right STN")
#     identify_significant_clusters(
#         cluster_p_values_right, 
#         clusters_right,
#         sliced_times,
#         T_obs_right,
#         pval,
#         tfr_args,
#         cluster_results_dict,
#         condition,
#         "Right"
#         )

#     # Compute permutation cluster test for the left + right stn
#     T_obs_both, clusters_both, cluster_p_values_both, H0_both = mne.stats.permutation_cluster_1samp_test(
#     all_diff_both_array_sliced, n_permutations=n_permutations,
#     threshold=threshold, tail=0,
#     out_type= "mask", seed=11111, verbose=True)
#     print(f"p_values: {cluster_p_values_both}")
#     print(f"P_values shape: {cluster_p_values_both.shape}")

#     print("Clusters for Both STN")
#     identify_significant_clusters(
#         cluster_p_values_both, 
#         clusters_both,
#         sliced_times,
#         T_obs_both,
#         pval,
#         tfr_args,
#         cluster_results_dict,
#         condition,
#         "Both"
#         )

#     # Average the percentage signal changes across subjects for left STN and for right STN
#     avg_diff_left = np.nanmean(all_diff_left_array_sliced, axis=0)  # shape: (n freqs, n times)
#     print(f"avg diff left shape: {avg_diff_left.shape}")
#     avg_diff_right = np.nanmean(all_diff_right_array_sliced, axis=0)
#     avg_diff_both = np.nanmean(all_diff_both_array_sliced, axis=0)

#     # Slicing TFR data to include only the t_min, t_max time range
#     #time_indices = np.logical_and(times >= t_min_max[0], times <= t_min_max[1])

#     #sliced_data_left = avg_diff_left[:, time_indices]
#     #sliced_data_right = avg_diff_right[:, time_indices]
#     #sliced_data_both = avg_diff_both[:, time_indices]


#     ################
#     ### PLOTTING ###
#     ################    

#     # Create a figure with two subplots for Left and Right STN
#     fig, (ax_left, ax_right, ax_both) = plt.subplots(1, 3, figsize=(20, 8))

#     # Figure title for n_subjects
#     sub_num = len(all_diff_left)

#     if sub_num > 1:
#         fig.suptitle(f"Power difference {epoch_cond1} - {epoch_cond2}, nSub = {sub_num}")
#     else:
#         fig.suptitle(f"Power difference {epoch_cond1} - {epoch_cond2}, {subject[:6]}")


#     # Plot the percentage change difference for Left STN
#     im_left = ax_left.imshow(avg_diff_left, aspect='auto', origin='lower', 
#                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                             cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    
#     print(T_obs_left.shape)
#     for c, p_val in zip(clusters_left, cluster_p_values_left):
#         if p_val <= pval:
#             mask_L = np.zeros_like(T_obs_left, dtype=bool) 
#             mask_L[c] = True
#             ax_left.contour(mask_L, levels=[0.5], colors='black', linewidths=1.5,
#                             extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
    

#     ax_left.set_xlabel('Time from GO cue (ms)')
#     ax_left.set_ylabel('Frequency (Hz)')
#     ax_left.set_title(f'Left STN - {dbs_status}')
#     fig.colorbar(im_left, ax=ax_left, label='Mean % Change (from baseline)')

#     # Plot the percentage change difference for Right STN
#     im_right = ax_right.imshow(avg_diff_right, aspect='auto', origin='lower', 
#                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                             cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    
#     for c, p_val in zip(clusters_right, cluster_p_values_right):
#         if p_val <= pval:
#             mask_R = np.zeros_like(T_obs_right, dtype=bool)
#             mask_R[c] = True
#             ax_right.contour(mask_R, levels=[0.5], colors='black', linewidths=1.5,
#                             extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
    

#     ax_right.set_xlabel('Time from GO cue (ms)')
#     ax_right.set_ylabel('Frequency (Hz)')
#     ax_right.set_title(f'Right STN - {dbs_status}')
#     fig.colorbar(im_right, ax=ax_right, label='Mean % Change (from baseline)')

#     # Plot the percentage change difference for Left + Right STN
#     im_both = ax_both.imshow(avg_diff_both, aspect='auto', origin='lower', 
#                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                             cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    
#     for c, p_val in zip(clusters_both, cluster_p_values_both):
#         if p_val <= pval:
#             mask_R = np.zeros_like(T_obs_both, dtype=bool)
#             mask_R[c] = True
#             ax_both.contour(mask_R, levels=[0.5], colors='black', linewidths=1.5,
#                             extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
    

#     ax_both.set_xlabel('Time from GO cue (ms)')
#     ax_both.set_ylabel('Frequency (Hz)')
#     ax_both.set_title(f'Left + Right STN - {dbs_status}')
#     fig.colorbar(im_both, ax=ax_both, label='Mean % Change (from baseline)')

#     if len(subs_included) > 1:
#         mean_RT = np.mean(all_sub_RT)
#         mean_SSD = np.mean(all_sub_SSD)
#         mean_SSRT = np.mean(all_sub_SSRT)
#     else:
#         mean_RT = all_sub_RT[0]
#         mean_SSD = all_sub_SSD[0]
#         mean_SSRT = all_sub_SSRT[0]
    
#     #ax_left.axvline(mean_RT, color='black', linestyle='--', label='Mean lm_GO RT')
#     #ax_right.axvline(mean_RT, color='black', linestyle='--', label='Mean lm_GO RT')
#     #ax_both.axvline(mean_RT, color='black', linestyle='--', label='Mean lm_GO RT')
#     #ax_left.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')
#     #ax_right.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')    
#     #ax_both.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')    
#     ax_left.axvline(mean_SSRT, color='black', linestyle='--')
#     ax_right.axvline(mean_SSRT, color='black', linestyle='--')
#     ax_both.axvline(mean_SSRT, color='black', linestyle='--', label='Mean SSRT')    

#     fig.legend() 

#     # Allows for text in figure to be modified as text, when saved as PDF!
#     matplotlib.rcParams['pdf.fonttype'] = 42
#     matplotlib.rcParams['ps.fonttype'] = 42

#     if sub_num > 1:
#         # Save the figure if a saving path is provided
#         figtitle = f"Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.png"
#         figtitle_pdf = f"Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.pdf"
#     else:
#         # Save the figure if a saving path is provided
#         figtitle = f"{subject[:6]}_Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.png"
#         figtitle_pdf = f"{subject[:6]}_Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.pdf"

#     if saving_path is not None:
#         plt.savefig(join(saving_path, figtitle), transparent=False)
#         # plt.savefig(join(saving_path, figtitle_pdf), transparent=False)

#     if show_fig == True:
#         plt.show()
#     else:
#         plt.close('all')

#     return cluster_results_dict



#updates from function below: 
            # # Pick the left and right channels
            # left_epochs1, right_epochs1 = data1.copy().pick(['Left_STN']), data1.copy().pick(['Right_STN'])
            # left_epochs2, right_epochs2 = data2.copy().pick(['Left_STN']), data2.copy().pick(['Right_STN'])

            # # Compute TFR using Morlet wavelets
            # power_left_ep1 = left_epochs1.compute_tfr(**tfr_args)
            # power_right_ep1 = right_epochs1.compute_tfr(**tfr_args)

            # power_left_ep1.data *= 1e12 # V² -> (µV)²
            # power_right_ep1.data *= 1e12 # V² -> (µV)²

            # power_left_ep1_squeeze = power_left_ep1.data.squeeze() # shape: (n_trials, n_freqs, n_times)
            # power_right_ep1_squeeze = power_right_ep1.data.squeeze()

            # mean_power_left_ep1 = np.nanmean(power_left_ep1.data, axis=0).squeeze() # shape: (n freqs, n times)
            # mean_power_right_ep1 = np.nanmean(power_right_ep1.data, axis=0).squeeze()

            # times = power_left_ep1.times * 1000
            # freqs = power_left_ep1.freqs

            # if epoch_type1.startswith('G'):
            #     # Define baseline period for percentage change calculation
            #     baseline_indices = (times >= -500) & (times <= -200)

            #     # Percentage change for condition 1
            #     baseline_power_left1 = np.nanmean(power_left_ep1_squeeze[:, :, baseline_indices], axis=2, keepdims=True)
            #     #percentage_change_left1 = (mean_power_left_ep1 - baseline_power_left1) / baseline_power_left1 * 100
            #     percentage_change_left_1 = 10.0 * np.log10(power_left_ep1_squeeze / baseline_power_left1)

            #     baseline_power_right1 = np.nanmean(power_right_ep1_squeeze[:, :, baseline_indices], axis=2, keepdims=True)
            #     #percentage_change_right1 = (mean_power_right_ep1 - baseline_power_right1) / baseline_power_right1 * 100
            #     percentage_change_right_1 = 10.0 * np.log10(power_right_ep1_squeeze / baseline_power_right1)

            #     percentage_change_left_ep1 = np.nanmean(percentage_change_left_1, axis=0)  # shape: (n_freqs, n_times)
            #     percentage_change_right_ep1 = np.nanmean(percentage_change_right_1, axis=0)  # shape: (n_freqs, n_times)


            # else: 
            #     if epoch_type1 == 'stop': 
            #         ssd_column = 'stop_signal_time'
            #     elif epoch_type1 == 'continue':
            #         ssd_column = 'continue_signal_time'

            #     baseline_start_per_trial_ep1 = - 500 - (np.array(data1.metadata[ssd_column]) * 1000)
            #     baseline_end_per_trial_ep1 = - 200 - (np.array(data1.metadata[ssd_column]) * 1000)

            #     percentage_change_left_single_trial_ep1 = np.empty_like(power_left_ep1_squeeze)  # same shape
            #     baseline_power_left_ep1 = np.empty((power_left_ep1_squeeze.shape[0], power_left_ep1_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

            #     for i in range(power_left_ep1_squeeze.shape[0]):  # loop over trials
            #         # Get trial-specific baseline window
            #         bl_start = baseline_start_per_trial_ep1[i]
            #         bl_end   = baseline_end_per_trial_ep1[i]

            #         # Find baseline indices in the common time axis
            #         bl_idx = (times >= bl_start) & (times <= bl_end)

            #         # Compute mean power in this window for all frequencies
            #         bl_mean = np.nanmean(power_left_ep1_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

            #         # Store baseline and percent change
            #         baseline_power_left_ep1[i] = bl_mean
            #         #percentage_change_left_single_trial[i] = ((power_left_squeeze[i] - bl_mean) / bl_mean) * 100
            #         #percentage_change_left_single_trial[i] = power_left_squeeze[i] - bl_mean
            #         percentage_change_left_single_trial_ep1[i] = 10.0 * np.log10(power_left_ep1_squeeze[i] / bl_mean)

            #     percentage_change_right_single_trial_ep1 = np.empty_like(power_right_ep1_squeeze)  # same shape
            #     baseline_power_right_ep1 = np.empty((power_right_ep1_squeeze.shape[0], power_right_ep1_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

            #     for i in range(power_right_ep1_squeeze.shape[0]):  # loop over trials
            #         # Get trial-specific baseline window
            #         bl_start = baseline_start_per_trial_ep1[i]
            #         bl_end   = baseline_end_per_trial_ep1[i]

            #         # Find baseline indices in the common time axis
            #         bl_idx = (times >= bl_start) & (times <= bl_end)

            #         # Compute mean power in this window for all frequencies
            #         bl_mean = np.nanmean(power_right_ep1_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

            #         # Store baseline and percent change
            #         baseline_power_right_ep1[i] = bl_mean
            #         #percentage_change_right_single_trial[i] = ((power_right_squeeze[i] - bl_mean) / bl_mean) * 100
            #         #percentage_change_right_single_trial[i] = power_right_squeeze[i] - bl_mean
            #         percentage_change_right_single_trial_ep1[i] = 10.0 * np.log10(power_right_ep1_squeeze[i] / bl_mean)

            #         percentage_change_left_ep1 = np.nanmean(percentage_change_left_single_trial_ep1, axis=0)  # shape: (n_freqs, n_times)
            #         percentage_change_right_ep1 = np.nanmean(percentage_change_right_single_trial_ep1, axis=0)  # shape: (n_freqs, n_times)


            # power_left_ep2 = left_epochs2.compute_tfr(**tfr_args)
            # power_right_ep2 = right_epochs2.compute_tfr(**tfr_args)

            # power_left_ep2.data *= 1e12 # V² -> (µV)²
            # power_right_ep2.data *= 1e12 # V² -> (µV)²

            # power_left_ep2_squeeze = power_left_ep2.data.squeeze() # shape: (n_trials, n_freqs, n_times)
            # power_right_ep2_squeeze = power_right_ep2.data.squeeze()

            # mean_power_left_ep2 = np.nanmean(power_left_ep2.data, axis=0).squeeze()
            # mean_power_right_ep2 = np.nanmean(power_right_ep2.data, axis=0).squeeze()    

            # if epoch_type2.startswith('G'):
            #     # Define baseline period for percentage change calculation
            #     baseline_indices = (times >= -500) & (times <= -200)

            #     # Percentage change for condition 2
            #     baseline_power_left2 = np.nanmean(power_left_ep2_squeeze[:, :, baseline_indices], axis=2, keepdims=True)
            #     #percentage_change_left1 = (mean_power_left_ep1 - baseline_power_left1) / baseline_power_left1 * 100
            #     percentage_change_left_2 = 10.0 * np.log10(power_left_ep2_squeeze / baseline_power_left2)

            #     baseline_power_right2 = np.nanmean(power_right_ep2_squeeze[:, :, baseline_indices], axis=2, keepdims=True)
            #     #percentage_change_right1 = (mean_power_right_ep1 - baseline_power_right1) / baseline_power_right1 * 100
            #     percentage_change_right_2 = 10.0 * np.log10(power_right_ep2_squeeze / baseline_power_right2)

            #     percentage_change_left_ep2 = np.nanmean(percentage_change_left_2, axis=0)  # shape: (n_freqs, n_times)
            #     percentage_change_right_ep2 = np.nanmean(percentage_change_right_2, axis=0)  # shape: (n_freqs, n_times)


            # else: 
            #     if epoch_type2 == 'stop': 
            #         ssd_column = 'stop_signal_time'
            #     elif epoch_type2 == 'continue':
            #         ssd_column = 'continue_signal_time'

            #     baseline_start_per_trial_ep2 = - 500 - (np.array(data2.metadata[ssd_column]) * 1000)
            #     baseline_end_per_trial_ep2 = - 200 - (np.array(data2.metadata[ssd_column]) * 1000)

            #     percentage_change_left_single_trial_ep2 = np.empty_like(power_left_ep2_squeeze)  # same shape
            #     baseline_power_left_ep2 = np.empty((power_left_ep2_squeeze.shape[0], power_left_ep2_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

            #     for i in range(power_left_ep2_squeeze.shape[0]):  # loop over trials
            #         # Get trial-specific baseline window
            #         bl_start = baseline_start_per_trial_ep2[i]
            #         bl_end   = baseline_end_per_trial_ep2[i]

            #         # Find baseline indices in the common time axis
            #         bl_idx = (times >= bl_start) & (times <= bl_end)

            #         # Compute mean power in this window for all frequencies
            #         bl_mean = np.nanmean(power_left_ep2_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

            #         # Store baseline and percent change
            #         baseline_power_left_ep2[i] = bl_mean
            #         #percentage_change_left_single_trial[i] = ((power_left_squeeze[i] - bl_mean) / bl_mean) * 100
            #         #percentage_change_left_single_trial[i] = power_left_squeeze[i] - bl_mean
            #         percentage_change_left_single_trial_ep2[i] = 10.0 * np.log10(power_left_ep2_squeeze[i] / bl_mean)

            #     percentage_change_right_single_trial_ep2 = np.empty_like(power_right_ep2_squeeze)  # same shape
            #     baseline_power_right_ep2 = np.empty((power_right_ep2_squeeze.shape[0], power_right_ep2_squeeze.shape[1], 1))  # (n_trials, n_freqs, 1)

            #     for i in range(power_right_ep2_squeeze.shape[0]):  # loop over trials
            #         # Get trial-specific baseline window
            #         bl_start = baseline_start_per_trial_ep2[i]
            #         bl_end   = baseline_end_per_trial_ep2[i]

            #         # Find baseline indices in the common time axis
            #         bl_idx = (times >= bl_start) & (times <= bl_end)

            #         # Compute mean power in this window for all frequencies
            #         bl_mean = np.nanmean(power_right_ep2_squeeze[i][ :, bl_idx], axis=1, keepdims=True)

            #         # Store baseline and percent change
            #         baseline_power_right_ep2[i] = bl_mean
            #         #percentage_change_right_single_trial[i] = ((power_right_squeeze[i] - bl_mean) / bl_mean) * 100
            #         #percentage_change_right_single_trial[i] = power_right_squeeze[i] - bl_mean
            #         percentage_change_right_single_trial_ep2[i] = 10.0 * np.log10(power_right_ep2_squeeze[i] / bl_mean)

            #         percentage_change_left_ep2 = np.nanmean(percentage_change_left_single_trial_ep2, axis=0)  # shape: (n_freqs, n_times)
            #         percentage_change_right_ep2 = np.nanmean(percentage_change_right_single_trial_ep2, axis=0)  # shape: (n_freqs, n_times)


            #     # # Percentage change for condition 2
            #     # baseline_power_left2 = np.nanmean(mean_power_left_ep2[:,baseline_indices], axis=1, keepdims=True)
            #     # percentage_change_left2 = (mean_power_left_ep2 - baseline_power_left2) / baseline_power_left2 * 100
                
            #     # baseline_power_right2 = np.nanmean(mean_power_right_ep2[:, baseline_indices], axis=1, keepdims=True)
            #     # percentage_change_right2 = (mean_power_right_ep2 - baseline_power_right2) / baseline_power_right2 * 100

# def perc_pow_diff_cond(
#         sub_dict, 
        # RT_dict,
        # stats_dict,
#         dbs_status:str, 
#         tfr_args, 
#         t_min_max:list, 
#         vmin_vmax:list,
#         epoch_cond1:str, 
#         epoch_cond2:str, 
#         cluster_results_dict: dict,
#         condition: str,
#         saving_path: str=None, 
#         show_fig: bool = None,
#         ADD_RT: bool = True
#         ):
        
#     """
#     Calculates % power change for the two specified conditions and subtracts epoch_cond2 from epoch_cond1, giving percentage change unique for epoch_cond1. 
#     Loops through all subs in sub_dict. 

#     Input:
#     - sub_dict: dict. containing all epochs (cue or feedback)
#     - dbs_status: "DBS ON" or "DBS OFF" 
#     - tfr_args: TFR parameters
#     - tmin, tmax: epoch slicing
#     - epoch_cond1: Epoch of interest "Win_cue", "Loss_cue"
#     - epoch_cond2: Baseline epoch, e.g., "Neutral_cue" to subtract from main epoch. 
#     - saving_path: Path where plots will be saved. If None figures are not saved. 
#     - show_fig: Defaults to None and figure isn't shown, if True figure is shown.
#     """
    
#     all_diff_left = []
#     all_diff_right = []
#     all_diff_both = []
#     all_sub_RT1 = []
#     all_sub_RT2 = []
#     all_sub_SSD = []

#     RT_plot1 = True
#     SSD_plot1 = False
#     RT_plot2 = True
#     SSD_plot2 = False

#     if epoch_cond1 == 'GS_successful':
#         RT_plot1 = False
#         SSD_plot1 = True

#     if epoch_cond1 == 'GS_unsuccessful':
#         SSD_plot1 = True

#     if epoch_cond2 == 'GS_successful':
#         RT_plot2 = False
#         SSD_plot2 = True

#     if epoch_cond2 == 'GS_unsuccessful':
#         SSD_plot2 = True    

#     if epoch_cond1 == 'STOP_successful':
#         RT_plot1 = False
    
#     if epoch_cond2 == 'STOP_successful':
#         RT_plot2 = False

#     subs_included = []

#     # Collect epoch data for each condition
#     for subject, epochs in sub_dict.items():
#         if dbs_status in subject:            
#             # Epoch condition 1
#             if epoch_cond1 in epochs.event_id:
#                 data1 = epochs[epoch_cond1]
#                 if RT_plot1 and ADD_RT: 
#                     sub_RT1 = RT_dict[subject][epoch_cond1] 
#                     all_sub_RT1.append(sub_RT1)

#                 if SSD_plot1:
#                     sub_SSD = stats_dict[subject]['mean SSD (ms)']
#                     all_sub_SSD.append(sub_SSD)
#             else:
#                 print(f"Condition {epoch_cond1} not found in subject {subject}")
#                 continue

#             # Epoch condition 2
#             if epoch_cond2 in epochs.event_id:
#                 data2 = epochs[epoch_cond2]
#                 if RT_plot2 and ADD_RT: 
#                     sub_RT2 = RT_dict[subject][epoch_cond2] 
#                     all_sub_RT2.append(sub_RT2)

#                 if SSD_plot2:
#                     sub_SSD = stats_dict[subject]['mean SSD (ms)']
#                     all_sub_SSD.append(sub_SSD)                
#             else:
#                 print(f"Condition {epoch_cond2} not found in subject {subject}")

#             # Pick the left and right channels
#             left_epochs1, right_epochs1 = data1.copy().pick(['Left_STN']), data1.copy().pick(['Right_STN'])
#             left_epochs2, right_epochs2 = data2.copy().pick(['Left_STN']), data2.copy().pick(['Right_STN'])

#             # Compute TFR using Morlet wavelets
#             power_left_ep1 = left_epochs1.compute_tfr(**tfr_args)
#             power_right_ep1 = right_epochs1.compute_tfr(**tfr_args)

#             mean_power_left_ep1 = np.nanmean(power_left_ep1.data, axis=0).squeeze() # shape: (n freqs, n times)
#             mean_power_right_ep1 = np.nanmean(power_right_ep1.data, axis=0).squeeze()

#             power_left_ep2 = left_epochs2.compute_tfr(**tfr_args)
#             power_right_ep2 = right_epochs2.compute_tfr(**tfr_args)

#             mean_power_left_ep2 = np.nanmean(power_left_ep2.data, axis=0).squeeze()
#             mean_power_right_ep2 = np.nanmean(power_right_ep2.data, axis=0).squeeze()    

#             times = power_left_ep1.times * 1000
#             freqs = power_left_ep1.freqs

#             # Define baseline period for percentage change calculation
#             baseline_indices = (times >= -500) & (times <= -200)

#             # Percentage change for condition 1
#             baseline_power_left1 = np.nanmean(mean_power_left_ep1[:, baseline_indices], axis=1, keepdims=True)
#             percentage_change_left1 = (mean_power_left_ep1 - baseline_power_left1) / baseline_power_left1 * 100

#             baseline_power_right1 = np.nanmean(mean_power_right_ep1[:, baseline_indices], axis=1, keepdims=True)
#             percentage_change_right1 = (mean_power_right_ep1 - baseline_power_right1) / baseline_power_right1 * 100

#             # Percentage change for condition 2
#             baseline_power_left2 = np.nanmean(mean_power_left_ep2[:,baseline_indices], axis=1, keepdims=True)
#             percentage_change_left2 = (mean_power_left_ep2 - baseline_power_left2) / baseline_power_left2 * 100
            
#             baseline_power_right2 = np.nanmean(mean_power_right_ep2[:, baseline_indices], axis=1, keepdims=True)
#             percentage_change_right2 = (mean_power_right_ep2 - baseline_power_right2) / baseline_power_right2 * 100

#             # Differences between Cond1 and Cond2 left and right STN
#             diff_left = percentage_change_left1 - percentage_change_left2
#             diff_right = percentage_change_right1 - percentage_change_right2

#             all_diff_left.append(diff_left)
#             all_diff_right.append(diff_right)  
#             all_diff_both.append(diff_left)
#             all_diff_both.append(diff_right)

#             subs_included.append(subject)


#     print(f'Subs included in analyses: \n {subs_included}')

#     all_diff_left_array = np.array(all_diff_left)  # shape: (n sub, n freqs, n times)
#     all_diff_right_array = np.array(all_diff_right)
#     all_diff_both_array = np.array(all_diff_both)

#     time_indices = (times >= t_min_max[0]) & (times <= t_min_max[1])
#     sliced_times = times[time_indices]
#     all_diff_left_array_sliced = all_diff_left_array[:,:,time_indices]
#     all_diff_right_array_sliced = all_diff_right_array[:,:,time_indices]
#     all_diff_both_array_sliced = all_diff_both_array[:,:,time_indices]

#     n_obs = all_diff_left_array_sliced.shape[0]
#     print(n_obs)
#     pval = 0.05
#     df = n_obs - 1
#     #threshold = scipy.stats.t.ppf(1-pval / 2, df) # two-tailed distribution
#     threshold = None
#     n_permutations = 1000


#     # Compute permutation cluster test for the left stn
#     T_obs_left, clusters_left, cluster_p_values_left, H0_left = mne.stats.permutation_cluster_1samp_test(
#     all_diff_left_array_sliced, n_permutations=n_permutations,
#     threshold=threshold, tail=0,
#     out_type= "mask", seed=11111, verbose=True)
#     print(f"p_values: {cluster_p_values_left}")
#     print(f"P_values shape: {cluster_p_values_left.shape}")

#     print("Clusters for Left STN")
#     cluster_results_dict = identify_significant_clusters(
#         cluster_p_values_left, 
#         clusters_left,
#         sliced_times,
#         T_obs_left,
#         pval,
#         tfr_args,
#         cluster_results_dict,
#         condition,
#         "Left"
#         )

#     # Compute permutation cluster test for the right stn
#     T_obs_right, clusters_right, cluster_p_values_right, H0_right = mne.stats.permutation_cluster_1samp_test(
#     all_diff_right_array_sliced, n_permutations=n_permutations,
#     threshold=threshold, tail=0,
#     out_type= "mask", seed=11111, verbose=True)
#     print(f"p_values: {cluster_p_values_right}")
#     print(f"P_values shape: {cluster_p_values_right.shape}")

#     print("Clusters for Right STN")
#     identify_significant_clusters(
#         cluster_p_values_right, 
#         clusters_right,
#         sliced_times,
#         T_obs_right,
#         pval,
#         tfr_args,
#         cluster_results_dict,
#         condition,
#         "Right"
#         )

#     # Compute permutation cluster test for the left + right stn
#     T_obs_both, clusters_both, cluster_p_values_both, H0_both = mne.stats.permutation_cluster_1samp_test(
#     all_diff_both_array_sliced, n_permutations=n_permutations,
#     threshold=threshold, tail=0,
#     out_type= "mask", seed=11111, verbose=True)
#     print(f"p_values: {cluster_p_values_both}")
#     print(f"P_values shape: {cluster_p_values_both.shape}")

#     print("Clusters for Both STN")
#     identify_significant_clusters(
#         cluster_p_values_both, 
#         clusters_both,
#         sliced_times,
#         T_obs_both,
#         pval,
#         tfr_args,
#         cluster_results_dict,
#         condition,
#         "Both"
#         )


#     # Average the percentage signal changes across subjects for left STN and for right STN
#     avg_diff_left = np.nanmean(all_diff_left_array_sliced, axis=0)
#     avg_diff_right = np.nanmean(all_diff_right_array_sliced, axis=0)
#     avg_diff_both = np.nanmean(all_diff_both_array_sliced, axis=0)

#     # Slicing TFR data to include only the t_min, t_max time range
#     #time_indices = np.logical_and(times >= t_min_max[0], times <= t_min_max[1])

#     #sliced_data_left = avg_diff_left[:, time_indices]
#     #sliced_data_right = avg_diff_right[:, time_indices]
#     #sliced_data_both = avg_diff_both[:, time_indices]


#     ################
#     ### PLOTTING ###
#     ################    

#     # Create a figure with two subplots for Left and Right STN
#     fig, (ax_left, ax_right, ax_both) = plt.subplots(1, 3, figsize=(20, 8))

#     # Figure title for n_subjects
#     sub_num = len(all_diff_left)

#     if sub_num > 1:
#         fig.suptitle(f"Power difference {epoch_cond1} - {epoch_cond2}, nSub = {sub_num}")
#     else:
#         fig.suptitle(f"Power difference {epoch_cond1} - {epoch_cond2}, {subject[:6]}")


#     # Plot the percentage change difference for Left STN
#     im_left = ax_left.imshow(avg_diff_left, aspect='auto', origin='lower', 
#                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                             cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    
#     for c, p_val in zip(clusters_left, cluster_p_values_left):
#         if p_val <= pval:
#             mask = np.zeros_like(T_obs_left, dtype=bool)  # Explicitly match dimensions
#             mask[c] = True
#             ax_left.contour(mask, levels=[0.5], colors='black', linewidths=1.5,
#                             extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
    

#     ax_left.set_xlabel('Time from GO cue (ms)')
#     ax_left.set_ylabel('Frequency (Hz)')
#     ax_left.set_title(f'Left STN - {dbs_status}')
#     fig.colorbar(im_left, ax=ax_left, label='Mean % Change (from baseline)')

#     # Plot the percentage change difference for Right STN
#     im_right = ax_right.imshow(avg_diff_right, aspect='auto', origin='lower', 
#                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                             cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    
#     for c, p_val in zip(clusters_right, cluster_p_values_right):
#         if p_val <= pval:
#             mask = np.zeros_like(T_obs_right, dtype=bool)  # Explicitly match dimensions
#             mask[c] = True
#             ax_right.contour(mask, levels=[0.5], colors='black', linewidths=1.5,
#                             extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
    

#     ax_right.set_xlabel('Time from GO cue (ms)')
#     ax_right.set_ylabel('Frequency (Hz)')
#     ax_right.set_title(f'Right STN - {dbs_status}')
#     fig.colorbar(im_right, ax=ax_right, label='Mean % Change (from baseline)')

#     # Plot the percentage change difference for Left + Right STN
#     im_both = ax_both.imshow(avg_diff_both, aspect='auto', origin='lower', 
#                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                             cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
    
#     for c, p_val in zip(clusters_both, cluster_p_values_both):
#         if p_val <= pval:
#             mask = np.zeros_like(T_obs_both, dtype=bool)  # Explicitly match dimensions
#             mask[c] = True
#             ax_both.contour(mask, levels=[0.5], colors='black', linewidths=1.5,
#                             extent=[t_min_max[0], t_min_max[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]])
    

#     ax_both.set_xlabel('Time from GO cue (ms)')
#     ax_both.set_ylabel('Frequency (Hz)')
#     ax_both.set_title(f'Left + Right STN - {dbs_status}')
#     fig.colorbar(im_both, ax=ax_both, label='Mean % Change (from baseline)')

#     if RT_plot1 and ADD_RT:
#         # Average mean RT across subjects
#         if len(subs_included) > 1:
#             mean_RT1 = np.mean(all_sub_RT1)
#         else:
#             mean_RT1 = all_sub_RT1[0]
#         ax_left.axvline(mean_RT1, color='black', linestyle='--')
#         ax_right.axvline(mean_RT1, color='black', linestyle='--')
#         ax_both.axvline(mean_RT1, color='black', linestyle='--', label=f'Mean RT {epoch_cond1}')

#     if SSD_plot1:
#         # Average mean RT across subjects
#         if len(subs_included) > 1:
#             mean_SSD = np.mean(all_sub_SSD)
#         else:
#             mean_SSD = all_sub_SSD[0]
#         ax_left.axvline(mean_SSD, color='grey', linestyle='--')
#         ax_right.axvline(mean_SSD, color='grey', linestyle='--')
#         ax_both.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')

#     if RT_plot2 and ADD_RT:
#         # Average mean RT across subjects
#         if len(subs_included) > 1:
#             mean_RT2 = np.mean(all_sub_RT2)
#         else:
#             mean_RT2 = all_sub_RT2[0]
#         ax_left.axvline(mean_RT2, color='blue', linestyle='--')
#         ax_right.axvline(mean_RT2, color='blue', linestyle='--')
#         ax_both.axvline(mean_RT2, color='blue', linestyle='--', label=f'Mean RT {epoch_cond2}')

#     if SSD_plot2:
#         # Average mean RT across subjects
#         if len(subs_included) > 1:
#             mean_SSD = np.mean(all_sub_SSD)
#         else:
#             mean_SSD = all_sub_SSD[0]
#         ax_left.axvline(mean_SSD, color='grey', linestyle='--')
#         ax_right.axvline(mean_SSD, color='grey', linestyle='--')    
#         ax_both.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')    

#     fig.legend()

#     # Allows for text in figure to be modified as text, when saved as PDF!
#     matplotlib.rcParams['pdf.fonttype'] = 42
#     matplotlib.rcParams['ps.fonttype'] = 42

#     if sub_num > 1:
#         # Save the figure if a saving path is provided
#         figtitle = f"Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.png"
#         figtitle_pdf = f"Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.pdf"
#     else:
#         # Save the figure if a saving path is provided
#         figtitle = f"{subject[:6]}_Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.png"
#         figtitle_pdf = f"{subject[:6]}_Power_diff_{epoch_cond1}_{epoch_cond2}_{dbs_status}.pdf"

#     if saving_path is not None:
#         plt.savefig(join(saving_path, figtitle_pdf), transparent=False)
#         # plt.savefig(join(saving_path, figtitle_pdf), transparent=False)

#     if show_fig == True:
#         plt.show()
#     else:
#         plt.close('all')

#     return cluster_results_dict


# def tfr_pow_change_cond(
#         sub_dict, 
#         RT_dict, 
#         stats_dict,
#         dbs_status:str, 
#         epoch_cond:str, 
#         tfr_args, 
#         t_min_max:list, 
#         vmin_vmax:list,
#         baseline_correction:bool=True,
#         saving_path: str=None, 
#         show_fig: bool = None,
#         ADD_RT: bool = True
#         ):

    
#     """
#     Function runs TFR on chosen epoch condition, averaging raw signal across epochs before performing TFR, and 
#     plots TFR plots with the percentage power change after cue presentation (from baseline), in either
#     DBS ON or DBS OFF. Loops through all subs in sub_dict.

#     Input:
#     - sub_dict: dict. containing all epochs
#     - dbs_status: "DBS ON" or "DBS OFF"
#     - epoch_cond: "Win_cue", "Loss_cue", etc.
#     - tfr_args: TFR parameters
#     - tmin, tmax: times for slicing epoch
#     - saving_path: Path where plots will be saved. If None figures are not saved. 
#     - show_fig: Defaults to None and figure isn't shown, if True figure is shown. 
#     """ 

#     all_percentage_change_left = []
#     all_percentage_change_right = []
#     all_percentage_change_both = []
#     all_sub_RT = []
#     all_sub_SSD = []

#     subs_included = []

#     RT_plot = True
#     SSD_plot = False
#     if epoch_cond == 'GS_successful':
#         RT_plot = False
#         SSD_plot = True

#     if epoch_cond == 'GS_unsuccessful':
#         SSD_plot = True

#     if epoch_cond == 'STOP_successful':
#         RT_plot = False

#     for subject, epochs in sub_dict.items():
#         if dbs_status in subject:
#             if epoch_cond in epochs.event_id:
#                 data = epochs[epoch_cond]
#                 if RT_plot and ADD_RT: 
#                     sub_RT = RT_dict[subject][epoch_cond] 
#                     all_sub_RT.append(sub_RT)

#                 if SSD_plot:
#                     sub_SSD = stats_dict[subject]['mean SSD (ms)']
#                     all_sub_SSD.append(sub_SSD)
#                 print(f"data found: {len(data)} epochs loaded for {epoch_cond}")

#                 # Compute TFR for each subject and each channel individually
#                 left_epochs = data.copy().pick(["Left_STN"])
#                 right_epochs = data.copy().pick(["Right_STN"])

#                 power_left = left_epochs.compute_tfr(**tfr_args)
#                 power_right = right_epochs.compute_tfr(**tfr_args)

#                 mean_power_left = np.nanmean(power_left.data, axis=0).squeeze()
#                 mean_power_right = np.nanmean(power_right.data, axis=0).squeeze()

#                 times = power_left.times * 1000
#                 freqs = power_left.freqs

#                 if baseline_correction:
#                     # Define baseline period for percentage change calculation
#                     baseline_indices = (times >= -500) & (times <= -200)

#                     # Calculate baseline power and percentage change for left STN and for right STN
#                     baseline_power_left = np.nanmean(mean_power_left[:, baseline_indices], axis=1, keepdims=True)
#                     percentage_change_left = (mean_power_left - baseline_power_left) / baseline_power_left * 100

#                     baseline_power_right = np.nanmean(mean_power_right[:, baseline_indices], axis=1, keepdims=True)
#                     percentage_change_right = (mean_power_right - baseline_power_right) / baseline_power_right * 100

#                     # Append each subject's percentage change to the lists
#                     all_percentage_change_left.append(percentage_change_left)
#                     all_percentage_change_right.append(percentage_change_right)
#                     all_percentage_change_both.append(percentage_change_left)
#                     all_percentage_change_both.append(percentage_change_right)
                
#                 else:
#                     print('No baseline correction applied')
#                     all_percentage_change_left.append(mean_power_left)
#                     all_percentage_change_right.append(mean_power_right)
#                     all_percentage_change_both.append(mean_power_left)
#                     all_percentage_change_both.append(mean_power_right)

#                 subs_included.append(subject)

#     print(f'Subs included in analyses: \n {subs_included}')

#     all_percentage_change_left = np.array(all_percentage_change_left)  # shape: (n sub, n freqs, n times)
#     all_percentage_change_right = np.array(all_percentage_change_right)
#     all_percentage_change_both = np.array(all_percentage_change_both)


#     # Average the percentage signal changes across subjects for left STN and for right STN
#     avg_percentage_change_left = np.nanmean(all_percentage_change_left, axis=0)
#     avg_percentage_change_right = np.nanmean(all_percentage_change_right, axis=0)
#     avg_percentage_change_both = np.nanmean(all_percentage_change_both, axis=0)

#     # ###################################################################################################################################
#     # # create a frequency mask for the frequencies of interest:
#     # delta_mask = (freqs >= 0) & (freqs <= 4)
#     # theta_mask = (freqs >= 4) & (freqs <= 8)
#     # alpha_mask = (freqs >= 8) & (freqs <= 12)
#     # beta_mask = (freqs >= 12) & (freqs <= 30)

#     # # subselect the frequencies of interest in all percentage change arrays and within a specific time range
#     # # (e.g., -500 to 1500ms)
#     # # Slicing the times to include only the t_min, t_max time range
#     # times = power_left.times * 1000  # Convert to milliseconds
#     # time_mask = (times >= -500) & (times <= 1500)
#     # # Select the time points of interest
#     # time_indices = np.where(time_mask)[0]
#     # # Select the frequencies of interest in the percentage change arrays
#     # avg_percentage_change_left_delta = avg_percentage_change_left[delta_mask, :][:, time_indices]
#     # avg_percentage_change_right_delta = avg_percentage_change_right[delta_mask, :][:, time_indices]
#     # avg_percentage_change_both_delta = avg_percentage_change_both[delta_mask, :][:, time_indices]
#     # avg_percentage_change_left_theta = avg_percentage_change_left[theta_mask, :][:, time_indices]
#     # avg_percentage_change_right_theta = avg_percentage_change_right[theta_mask, :][:, time_indices]
#     # avg_percentage_change_both_theta = avg_percentage_change_both[theta_mask, :][:, time_indices]
#     # avg_percentage_change_left_alpha = avg_percentage_change_left[alpha_mask, :][:, time_indices]
#     # avg_percentage_change_right_alpha = avg_percentage_change_right[alpha_mask, :][:, time_indices]
#     # avg_percentage_change_both_alpha = avg_percentage_change_both[alpha_mask, :][:, time_indices]
#     # avg_percentage_change_left_beta = avg_percentage_change_left[beta_mask, :][:, time_indices]
#     # avg_percentage_change_right_beta = avg_percentage_change_right[beta_mask, :][:, time_indices]
#     # avg_percentage_change_both_beta = avg_percentage_change_both[beta_mask, :][:, time_indices]

#     # # plot the delta, theta, alpha and beta frequency bands separately
#     # for band in ['delta', 'theta', 'alpha', 'beta']:
#     #     fig, (ax_left, ax_right, ax_both) = plt.subplots(1, 3, figsize=(20, 8))
#     #     ax_left.plot(times[time_indices], avg_percentage_change_left_delta if band == 'delta' else
#     #                 avg_percentage_change_left_theta if band == 'theta' else 
#     #                 avg_percentage_change_left_alpha if band == 'alpha' else
#     #                 avg_percentage_change_left_beta, label='Left STN')
#     #     ax_right.plot(times[time_indices], avg_percentage_change_right_delta if band == 'delta' else
#     #                 avg_percentage_change_right_theta if band == 'theta' else
#     #                 avg_percentage_change_right_alpha if band == 'alpha' else
#     #                 avg_percentage_change_right_beta, label='Right STN')
#     #     ax_both.plot(times[time_indices], avg_percentage_change_both_delta if band == 'delta' else
#     #                 avg_percentage_change_both_theta if band == 'theta' else
#     #                 avg_percentage_change_both_alpha if band == 'alpha' else
#     #                 avg_percentage_change_both_beta, label='Both STN')
#     #     ax_left.set_title(f"Left STN - {band.capitalize()}")
#     #     ax_right.set_title(f"Right STN - {band.capitalize()}")
#     #     ax_both.set_title(f"Both STN - {band.capitalize()}")
#     #     ax_left.set_xlabel("Time (ms)")
#     #     ax_right.set_xlabel("Time (ms)")
#     #     ax_both.set_xlabel("Time (ms)")
#     #     ax_left.set_ylabel("Percentage Change")
#     #     ax_right.set_ylabel("Percentage Change")
#     #     ax_both.set_ylabel("Percentage Change")
#     #     ax_left.legend()
#     #     ax_right.legend()
#     #     ax_both.legend()
#     #     plt.tight_layout()
#     #     plt.show()

#     # ################################################################################################################################

#     # Slicing TFR data to include only the t_min, t_max time range
#     time_indices = np.logical_and(times >= t_min_max[0], times <= t_min_max[1])
    
#     # selects only timepoints where time_indices = True, then removes dimensions with 1 (i.e., first dimension which is 1 channel)
#     # first dimension is n_channels (which is already as we do left and right separately)
#     sliced_data_left = avg_percentage_change_left[:, time_indices].squeeze()
#     sliced_data_right = avg_percentage_change_right[:, time_indices].squeeze()
#     sliced_data_both = avg_percentage_change_both[:, time_indices].squeeze()

#     # Stack the scores for Left and Right STN across all subjects, so the output variable can be used for permutation cluster test
#     group_all_change_left = np.stack(all_percentage_change_left)  # Shape: (n_subjects, n_frequencies, n_times)
#     group_all_change_right = np.stack(all_percentage_change_right)  # Shape: (n_subjects, n_frequencies, n_times)
#     group_all_change_both = np.stack(all_percentage_change_both)

#     # Get the indices where the condition is True
#     freq_indices = np.where((freqs >= 5) & (freqs <= 20))[0]
#     time_indices = np.where((times >= -500) & (times <= 0))[0]

#     # Use integer-based indexing
#     filtered_data_left = group_all_change_left[:, freq_indices, :][:, :, time_indices]
#     filtered_data_right = group_all_change_right[:, freq_indices, :][:, :, time_indices]
#     filtered_data_both = group_all_change_both[:, freq_indices, :][:, :, time_indices]

#     # Compute min and max along the frequency axis
#     min_values_left = np.min(filtered_data_left)  # Shape: (n_subjects, n_times)
#     print(min_values_left.shape)
#     print(min_values_left)
#     max_values_left = np.max(filtered_data_left)   # Shape: (n_subjects, n_times)
#     min_values_right = np.min(filtered_data_right)  # Shape: (n_subjects, n_times)
#     max_values_right = np.max(filtered_data_right) # Shape: (n_subjects, n_times)
#     min_values_both = np.min(filtered_data_both) # Shape: (n_subjects, n_times)
#     max_values_both = np.max(filtered_data_both) # Shape: (n_subjects, n_times)


#     ################
#     ### PLOTTING ###
#     ################

#     # Create a figure with two subplots for Left and Right STN
#     fig, (ax_left, ax_right, ax_both) = plt.subplots(1, 3, figsize=(20, 8))

#     # Figure title for n_subjects
#     sub_num = len(all_percentage_change_left)

#     if baseline_correction:
#         if sub_num > 1:
#             fig.suptitle(f"Power change - {epoch_cond}, nSub = {sub_num}")
#         else:
#             fig.suptitle(f"Power change - {epoch_cond}, {subject[:6]}")
#     else:
#         if sub_num > 1:
#             fig.suptitle(f"Power - {epoch_cond}, nSub = {sub_num}, no baseline correction")
#         else:
#             fig.suptitle(f"Power - {epoch_cond}, {subject[:6]}, no baseline correction")

#     # Set the x label based on what the epochs are centered on:
#     if epoch_cond.startswith('S'):
#         xlabel = 'Time from STOP cue (ms)'
#     elif epoch_cond.startswith('CONTINUE'):
#         xlabel = 'Time from CONTINUE cue (ms)'
#     else:
#         xlabel = 'Time from GO cue (ms)'
        
#     # Plot the Left STN
#     if baseline_correction:
#         im_left = ax_left.imshow(sliced_data_left, aspect='auto', origin='lower', 
#                                 extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                                 cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
#     else:
#         im_left = ax_left.imshow(sliced_data_left, aspect='auto', origin='lower', 
#                         extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                         cmap='jet', vmin=min_values_left, vmax=max_values_left)
#     ax_left.set_xlabel(xlabel)
#     ax_left.set_ylabel('Frequency (Hz)')
#     ax_left.set_title(f'Left STN - {dbs_status}')
#     if baseline_correction:
#         fig.colorbar(im_left, ax=ax_left, label='Mean % Change (from baseline)')
#     else:
#         fig.colorbar(im_left, ax=ax_left, label='Mean Power')

#     # Plot the Right STN
#     if baseline_correction:
#         im_right = ax_right.imshow(sliced_data_right, aspect='auto', origin='lower', 
#                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                             cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
#     else:    
#         im_right = ax_right.imshow(sliced_data_right, aspect='auto', origin='lower', 
#                             extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                             cmap='jet', vmin=min_values_right, vmax=max_values_right)        
#     ax_right.set_xlabel(xlabel)
#     ax_right.set_ylabel('Frequency (Hz)')
#     ax_right.set_title(f'Right STN - {dbs_status}')
#     if baseline_correction:
#         fig.colorbar(im_right, ax=ax_right, label='Mean % Change (from baseline)')
#     else:
#         fig.colorbar(im_right, ax=ax_right, label='Mean Power')

#     # Plot both STN combined
#     if baseline_correction:
#         im_both = ax_both.imshow(sliced_data_both, aspect='auto', origin='lower', 
#                                 extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                                 cmap='jet', vmin=vmin_vmax[0], vmax=vmin_vmax[-1])
#     else:
#         im_both = ax_both.imshow(sliced_data_both, aspect='auto', origin='lower', 
#                                 extent=[t_min_max[0], t_min_max[1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
#                                 cmap='jet', vmin=min_values_both, vmax=max_values_both)
#     ax_both.set_xlabel(xlabel)
#     ax_both.set_ylabel('Frequency (Hz)')
#     ax_both.set_title(f'Left + Right STN - {dbs_status}')
#     if baseline_correction:
#         fig.colorbar(im_both, ax=ax_both, label='Mean % Change (from baseline)')
#     else:
#         fig.colorbar(im_both, ax=ax_both, label='Mean Power')


#     if RT_plot and ADD_RT:
#         # Average mean RT across subjects
#         if len(subs_included) > 1:
#             mean_RT = np.mean(all_sub_RT)
#         else:
#             mean_RT = all_sub_RT[0]
#         ax_left.axvline(mean_RT, color='black', linestyle='--')
#         ax_right.axvline(mean_RT, color='black', linestyle='--')
#         ax_both.axvline(mean_RT, color='black', linestyle='--', label='Mean RT')

#     if SSD_plot:
#         # Average mean RT across subjects
#         if len(subs_included) > 1:
#             mean_SSD = np.mean(all_sub_SSD)
#         else:
#             mean_SSD = all_sub_SSD[0]
#         ax_left.axvline(mean_SSD, color='grey', linestyle='--')
#         ax_right.axvline(mean_SSD, color='grey', linestyle='--')
#         ax_both.axvline(mean_SSD, color='grey', linestyle='--', label='Mean SSD')
        
#     fig.legend()

#     # Allows for text in figure to be modified as text, when saved as PDF!
#     matplotlib.rcParams['pdf.fonttype'] = 42
#     matplotlib.rcParams['ps.fonttype'] = 42

#     if baseline_correction:
#         if sub_num > 1:
#             # Save the figure if a saving path is provided
#             figtitle = f"Power_change_{epoch_cond}_{dbs_status}.png"
#             figtitle_pdf = f"Power_change_{epoch_cond}_{dbs_status}.pdf"
#         else:
#         # Save the figure if a saving path is provided
#             figtitle = f"{subject[:6]}_Power_change_{epoch_cond}_{dbs_status}.png"
#             figtitle_pdf = f"{subject[:6]}_Power_change_{epoch_cond}_{dbs_status}.pdf"

#     else:
#         if sub_num > 1:
#             # Save the figure if a saving path is provided
#             figtitle = f"Power_{epoch_cond}_{dbs_status}_no_baseline_correction.png"
#             figtitle_pdf = f"Power_{epoch_cond}_{dbs_status}_no_baseline_correction.pdf"
#         else:
#             # Save the figure if a saving path is provided
#             figtitle = f"{subject[:6]}_Power_{epoch_cond}_{dbs_status}_no_baseline_correction.png"
#             figtitle_pdf = f"{subject[:6]}_Power_{epoch_cond}_{dbs_status}_no_baseline_correction.pdf"


#     if saving_path is not None:
#         plt.savefig(join(saving_path, figtitle_pdf), transparent=False)
#         print(f"Figure saved as {join(saving_path, figtitle_pdf)}")
#         # plt.savefig(join(saving_path, figtitle_pdf), transparent=False)

#     if show_fig == True:
#         plt.show()
#     else:
#         plt.close('all')


#     return group_all_change_left, group_all_change_right, group_all_change_both 


# def plot_beta_amplitude_and_difference(
#         epochs_beta_subsets,
#         epochs_beta_lm,
#         baseline,
#         session_ID,
#         saving_path,
#         session_dict
# ):
#     # Check for required event IDs
#     if 'GS_successful' in epochs_beta_subsets.event_id and 'lm_GO' in epochs_beta_lm.event_id:
#         gs_successful_epochs = epochs_beta_subsets['GS_successful']
#         lm_go_epochs = epochs_beta_lm['lm_GO']

#         # Loop through first two channels
#         channels = gs_successful_epochs.info['ch_names'][:2]
#         for channel in channels:
#             gs_successful_epochs_ch = gs_successful_epochs.copy().pick_channels([channel])
#             lm_go_epochs_ch = lm_go_epochs.copy().pick_channels([channel])

#             # Compute averages
#             evoked_gs_successful = gs_successful_epochs_ch.average()
#             evoked_lm_go = lm_go_epochs_ch.average()

#             baseline = (-0.5, -0.1)
#             baseline_indices = np.where((evoked_gs_successful.times >= baseline[0]) & (evoked_gs_successful.times <= baseline[1]))[0]

#             # Compute baseline mean and std for normalization
#             baseline_mean_gs_successful = np.mean(evoked_gs_successful.data[:, baseline_indices], axis=1)
#             baseline_std_gs_successful = np.std(evoked_gs_successful.data[:, baseline_indices], axis=1)

#             baseline_mean_lm_go = np.mean(evoked_lm_go.data[:, baseline_indices], axis=1)
#             baseline_std_lm_go = np.std(evoked_lm_go.data[:, baseline_indices], axis=1)

#             # Normalize responses
#             normalized_gs_successful = (evoked_gs_successful.data - baseline_mean_gs_successful[:, np.newaxis]) / baseline_std_gs_successful[:, np.newaxis]
#             normalized_lm_go = (evoked_lm_go.data - baseline_mean_lm_go[:, np.newaxis]) / baseline_std_lm_go[:, np.newaxis]

#             # Compute difference
#             difference = normalized_gs_successful - normalized_lm_go

#             session_dict[f"{channel}_beta_amp_GS"] = normalized_gs_successful[0, :]
#             session_dict[f"{channel}_beta_amp_lm_GO"] = normalized_lm_go[0, :]
#             session_dict[f"{channel}_beta_amp_diff"] = difference[0, :]
#             session_dict[f"{channel}_beta_amp_times"] = evoked_gs_successful.times

#             # Create the figure with two subplots
#             fig, axes = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)

#             # First subplot: 'GS_successful' and 'lm_GO' responses
#             axes[0].plot(evoked_gs_successful.times, normalized_gs_successful[0, :], color='tab:green', label=f"{channel} 'GS_successful'")
#             axes[0].plot(evoked_lm_go.times, normalized_lm_go[0, :], color='tab:blue', label=f"{channel} 'lm_GO'")
#             axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
#             axes[0].set_title(f"βA Response - {channel}")
#             axes[0].set_ylabel("Normalized βA")
#             axes[0].legend()

#             # Second subplot: Difference
#             axes[1].plot(evoked_gs_successful.times, difference[0, :], color='tab:red', label="Difference ('GS_successful' - 'lm_GO')")
#             axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
#             axes[1].axvspan(baseline[0], baseline[1], color='gray', alpha=0.2, label="Baseline")
#             axes[1].set_title("Difference between 'GS_successful' and 'lm_GO'")
#             axes[1].set_xlabel("Time (s)")
#             axes[1].set_ylabel("Normalized βA Difference")
#             axes[1].legend()

#             axes[0].set_xlim(-0.5,1.5)
#             axes[1].set_xlim(-0.5,1.5)

#             # Save figure
#             figtitle = f"βA Response and Difference - {session_ID} - {channel}.png"
#             plt.savefig(join(saving_path, figtitle), transparent=False, bbox_inches='tight')
#             plt.close(fig)

#     else:
#         print("Both 'GS_successful' in epochs_beta_subsets and 'lm_GO' in epochs_beta_lm must be present in the epochs.")

#     return session_dict


# def plot_beta_amplitude_sGS_versus_lm_GO(
#         epochs_beta_subsets,
#         epochs_beta_lm,
#         baseline,
#         session_ID,
#         saving_path
# ):
#     # Check for 'GS_successful' in epochs_beta_subsets and 'lm_GO' in epochs_beta_lm
#     if 'GS_successful' in epochs_beta_subsets.event_id and 'lm_GO' in epochs_beta_lm.event_id:
#         # Select 'GS_successful' from epochs_beta_subsets
#         gs_successful_epochs = epochs_beta_subsets['GS_successful']

#         # Select 'lm_GO' from epochs_beta_lm
#         lm_go_epochs = epochs_beta_lm['lm_GO']

#         # Loop through 2 first channels
#         channels = gs_successful_epochs.info['ch_names'][:2]
#         for channel in channels:
#             gs_successful_epochs_ch = gs_successful_epochs.copy().pick_channels([channel])
#             lm_go_epochs_ch = lm_go_epochs.copy().pick_channels([channel])

#             # Compute the averages for 'GS_successful' and 'lm_GO'
#             evoked_gs_successful = gs_successful_epochs_ch.average()
#             evoked_lm_go = lm_go_epochs_ch.average()

#             baseline = (-0.5, -0.1)
#             baseline_indices = np.where((evoked_gs_successful.times >= baseline[0]) & (evoked_gs_successful.times <= baseline[1]))[0]

#             # Compute baseline mean and std for normalization
#             baseline_data_gs_successful = evoked_gs_successful.data[:, baseline_indices]
#             baseline_mean_gs_successful = np.mean(baseline_data_gs_successful, axis=1)
#             baseline_std_gs_successful = np.std(baseline_data_gs_successful, axis=1)

#             baseline_data_lm_go = evoked_lm_go.data[:, baseline_indices]
#             baseline_mean_lm_go = np.mean(baseline_data_lm_go, axis=1)
#             baseline_std_lm_go = np.std(baseline_data_lm_go, axis=1)

#             # Normalize responses
#             normalized_gs_successful = (evoked_gs_successful.data - baseline_mean_gs_successful[:, np.newaxis]) / baseline_std_gs_successful[:, np.newaxis]
#             normalized_lm_go = (evoked_lm_go.data - baseline_mean_lm_go[:, np.newaxis]) / baseline_std_lm_go[:, np.newaxis]

#             # Create the plot
#             fig, ax = plt.subplots()

#             # Plot 'GS_successful' response
#             main_color_gs_successful = 'tab:green'
#             ax.plot(
#                 evoked_gs_successful.times, normalized_gs_successful[0, :],
#                 color=main_color_gs_successful, label=f"{channel} 'GS_successful'", zorder=2
#             )

#             # Plot 'lm_GO' response
#             main_color_lm_go = 'tab:blue'
#             ax.plot(
#                 evoked_lm_go.times, normalized_lm_go[0, :],
#                 color=main_color_lm_go, label=f"{channel} 'lm_GO'", zorder=2
#             )

#             # Customize the plot
#             ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
#             ax.set_xlim(-0.5, 1.5)  # Set x-axis limits
#             ax.set_title(f"βA Response for 'GS_successful' and 'lm_GO' (Normalized) - {channel}")
#             ax.set_xlabel("Time (s)")
#             ax.set_ylabel("Normalized βA")

#             # Add legend outside the plot
#             ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

#             # Save figure
#             figtitle = f"βA Response for 'GS_successful' and 'lm_GO' (Normalized) - {session_ID} - {channel}.png"
#             plt.savefig(join(saving_path, figtitle), transparent=False, bbox_inches='tight')

#             plt.close(fig)

#     else:
#         print("Both 'GS_successful' in epochs_beta_subsets and 'lm_GO' in epochs_beta_lm must be present in the epochs.")




# def plot_diff_beta_amplitude_sGS_minus_lm_GO(
#         epochs_beta_subsets,
#         epochs_beta_lm,
#         baseline,
#         session_ID,
#         saving_path          
# ):
#     # Check for 'GS_successful' in epochs_beta_subsets and 'lm_GO' in epochs_beta_lm
#     if 'GS_successful' in epochs_beta_subsets.event_id and 'lm_GO' in epochs_beta_lm.event_id:
#         # Select 'GS_successful' from epochs_beta_subsets
#         gs_successful_epochs = epochs_beta_subsets['GS_successful']

#         # Select 'lm_GO' from epochs_beta_lm
#         lm_go_epochs = epochs_beta_lm['lm_GO']

#         # Loop through 2 first channels
#         channels = gs_successful_epochs.info['ch_names'][:2]
#         for channel in channels:
#             gs_successful_epochs = gs_successful_epochs.pick_channels([channel])
#             lm_go_epochs = lm_go_epochs.pick_channels([channel])

#             # Compute the averages for 'GS_successful' and 'lm_GO'
#             evoked_gs_successful = gs_successful_epochs.average()
#             evoked_lm_go = lm_go_epochs.average()

#             # Define baseline (e.g., -0.5 to 0 seconds)
#             baseline = (-0.5, -0.1)
#             baseline_indices = np.where((evoked_gs_successful.times >= baseline[0]) & (evoked_gs_successful.times <= baseline[1]))[0]

#             # Compute baseline mean and std for normalization
#             baseline_data_gs_successful = evoked_gs_successful.data[:, baseline_indices]
#             baseline_mean_gs_successful = np.mean(baseline_data_gs_successful, axis=1)
#             baseline_std_gs_successful = np.std(baseline_data_gs_successful, axis=1)

#             baseline_data_lm_go = evoked_lm_go.data[:, baseline_indices]
#             baseline_mean_lm_go = np.mean(baseline_data_lm_go, axis=1)
#             baseline_std_lm_go = np.std(baseline_data_lm_go, axis=1)

#             # Normalize responses
#             normalized_gs_successful = (evoked_gs_successful.data - baseline_mean_gs_successful[:, np.newaxis]) / baseline_std_gs_successful[:, np.newaxis]
#             normalized_lm_go = (evoked_lm_go.data - baseline_mean_lm_go[:, np.newaxis]) / baseline_std_lm_go[:, np.newaxis]

#             # Compute the difference between 'GS_successful' and 'lm_GO'
#             difference = normalized_gs_successful - normalized_lm_go

#             # Create the plot for the difference
#             fig, ax = plt.subplots()

#             # Plot the difference
#             ax.plot(
#                 evoked_gs_successful.times, difference[0, :],
#                 color='tab:red', label="Difference ('GS_successful' - 'lm_GO')", zorder=1
#             )

#             # Add baseline and customize the plot
#             ax.axvspan(baseline[0], baseline[1], color='gray', alpha=0.2, label="Baseline")
#             ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
#             ax.set_xlim(-0.5, 1.5)  # Set x-axis limits
#             ax.set_title("Difference between 'GS_successful' and 'lm_GO' (Normalized)")
#             ax.set_xlabel("Time (s)")
#             ax.set_ylabel("Normalized βA Difference")
            
#             # Add legend outside the plot
#             ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

#             plt.show()

#     else:
#         print("Both 'GS_successful' in epochs_beta_subsets and 'lm_GO' in epochs_beta_lm must be present in the epochs.")




# def plot_power_comparison_outcome(
#         session_ID: str, 
#         raw: mne.io.Raw,
#         condition1_epochs: mne.Epochs, 
#         condition1_name: str, 
#         mean_RT_condition: float, 
#         condition2_epochs: mne.Epochs, 
#         condition2_name: str,
#         channel_n: list, 
#         centered_around: str, 
#         saving_path: str,
#         fmax: int = 50,
#         vmin = None,
#         vmax = None
#         ):
#     '''
#     This function plots the power comparison between two conditions for specific channels of interest.

#     session_ID: session ID
#     raw: raw data (loaded through mne.io.read_raw)
#     condition1_epochs: mne.Epochs object for condition 1
#     condition1_name: name of condition 1
#     mean_RT_condition1: mean reaction time for condition 1 (in milliseconds)
#     condition2_epochs: mne.Epochs object for condition 2
#     condition2_name: name of condition 2
#     mean_RT_condition2: mean reaction time for condition 2 (in milliseconds)
#     channel_n: list of channel indices to plot
#     centered_around: event to center the plot around (e.g. 'GO signal')
#     saving_path: path to save the figure
#     '''

#     # parameters for tfr computation
#     freqs = np.arange(1, fmax, 1)  # define frequencies of interest
#     tfr_kwargs = dict(
#         method="morlet",
#         freqs=freqs,
#         n_cycles=freqs/2.0,  # different number of cycle per frequency
#         decim=2,
#         return_itc=False,
#         average=False,
#     )
#     baseline = (-0.5, 0)

#     # loop between left and right STN
#     for ch in channel_n:
#         ch_name = raw.ch_names[ch]

#         percent_change_1, avg_power_1, times = compute_percent_change(condition1_epochs, ch, baseline, **tfr_kwargs)
#         percent_change_2, avg_power_2, times = compute_percent_change(condition2_epochs, ch, baseline, **tfr_kwargs)

#         # Create figure with 3 subplots
#         fig, (ax1, ax2, ax_diff) = plt.subplots(3, 1, figsize=(15, 15))

#         # Plot average power for condition 1
#         im1 = ax1.imshow(avg_power_1, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                         aspect="auto", origin="lower", cmap="coolwarm", 
#                         vmin=vmin, vmax=vmax
#                         )
#         ax1.set_title(f"Average Time-Frequency Power for {condition1_name} trials ({ch_name} - {len(condition1_epochs)} trials)")
#         ax1.set_xlabel("Time (ms)")
#         ax1.set_ylabel("Frequency (Hz)")
#         ax1.axvline(0, color="white", linestyle="--")
#         ax1.axvline(mean_RT_condition/1000, color="red", linestyle="--")
#         fig.colorbar(im1, ax=ax1)

#         # Plot average power for condition 2
#         im2 = ax2.imshow(avg_power_2, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                         aspect="auto", origin="lower", cmap="coolwarm", 
#                         vmin=vmin, vmax=vmax
#                         )
#         ax2.set_title(f"Average Time-Frequency Power for {condition2_name} trials ({ch_name} - {len(condition2_epochs)} trials)")
#         ax2.set_xlabel("Time (ms)")
#         ax2.set_ylabel("Frequency (Hz)")
#         ax2.axvline(0, color="white", linestyle="--")
#         ax2.axvline(mean_RT_condition/1000, color="red", linestyle="--")
#         fig.colorbar(im2, ax=ax2)

#         F_obs_plot, F_obs = perform_permutation_cluster_test(percent_change_1, percent_change_2)
#         max_F = np.nanmax(abs(F_obs_plot))

#         # Plot the cluster-corrected power contrast
#         ax_diff.imshow(F_obs, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                     aspect="auto", origin="lower", cmap="gray")
#         im_diff = ax_diff.imshow(F_obs_plot, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                                 aspect="auto", origin="lower", cmap="coolwarm", 
#                                 vmin=-max_F, vmax=max_F
#                                 )

#         ax_diff.set_title(f"{condition1_name} - {condition2_name} trials ({ch_name})")
#         ax_diff.set_xlabel("Time (ms)")
#         ax_diff.set_ylabel("Frequency (Hz)")
#         label1 = f"Mean RT"
#         ax_diff.axvline(mean_RT_condition/1000, color="green", linestyle="--", label=label1)
#         fig.colorbar(im_diff, ax=ax_diff)

#         ax_diff.axvline(0, color="white", linestyle="--", label=centered_around)
#         fig.legend(loc='center right')

#         plt.tight_layout()
#         figtitle = (f"Power Comparison between {condition1_name} and {condition2_name} trials ({ch_name} - {session_ID}).png")
#         plt.savefig(join(saving_path, figtitle), transparent=False)
    

# def plot_power_comparison_outcome(
#         session_ID: str, 
#         raw: mne.io.Raw,
#         condition1_epochs: mne.Epochs, 
#         condition1_name: str, 
#         mean_RT_condition: float, 
#         condition2_epochs: mne.Epochs, 
#         condition2_name: str,
#         channel_n: list, 
#         centered_around: str, 
#         saving_path: str,
#         fmax: int = 50,
#         vmin = None,
#         vmax = None
#         ):
#     '''
#     This function plots the power comparison between two conditions for specific channels of interest.

#     session_ID: session ID
#     raw: raw data (loaded through mne.io.read_raw)
#     condition1_epochs: mne.Epochs object for condition 1
#     condition1_name: name of condition 1
#     mean_RT_condition1: mean reaction time for condition 1 (in milliseconds)
#     condition2_epochs: mne.Epochs object for condition 2
#     condition2_name: name of condition 2
#     mean_RT_condition2: mean reaction time for condition 2 (in milliseconds)
#     channel_n: list of channel indices to plot
#     centered_around: event to center the plot around (e.g. 'GO signal')
#     saving_path: path to save the figure
#     '''

#     decim = 2
#     freqs = np.arange(1, fmax, 1)  # define frequencies of interest
#     n_cycles = freqs/2.0  # different number of cycle per frequency
#     tfr_kwargs = dict(
#         method="morlet",
#         freqs=freqs,
#         n_cycles=n_cycles,
#         decim=decim,
#         return_itc=False,
#         average=False,
#     )

#     # Compute TFR for both conditions
#     tfr_epochs_1 = condition1_epochs.compute_tfr(**tfr_kwargs)
#     tfr_epochs_2 = condition2_epochs.compute_tfr(**tfr_kwargs)

#     # Apply baseline correction
#     tfr_epochs_1.apply_baseline(mode="percent", baseline=(None, 0))
#     tfr_epochs_2.apply_baseline(mode="percent", baseline=(None, 0))

#     for ch in channel_n:
#         ch_name = raw.ch_names[ch]

#         # Extract power for single channel as 3D matrix (epochs x frequencies x times)
#         epochs_power_1 = tfr_epochs_1.data[:, ch, :, :]
#         epochs_power_2 = tfr_epochs_2.data[:, ch, :, :]

#         # Perform cluster permutation test
#         # Compute threshold based on a p-value:
#         pval = 0.05
#         dfn = 2 - 1  # degrees of freedom numerator
#         n_observations = len(epochs_power_1) + len(epochs_power_2)
#         dfd = n_observations - 2  # degrees of freedom denominator
#         threshold = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
#         print(f"Threshold = {threshold}")

#         #threshold = 6.0
#         F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
#             [epochs_power_1, epochs_power_2],
#             out_type="mask",
#             n_permutations=1000,
#             threshold=threshold,
#             tail=0,
#             seed=np.random.default_rng(seed=8675309),
#         )

#         times = 1e3 * condition1_epochs.times  # Change unit to ms

#         # Compute average power for each condition
#         avg_power_1 = epochs_power_1.mean(axis=0)
#         avg_power_2 = epochs_power_2.mean(axis=0)

#         # Create figure with 3 subplots
#         fig, (ax1, ax2, ax_diff) = plt.subplots(3, 1, figsize=(15, 15))

#         # Plot average power for condition 1
#         im1 = ax1.imshow(avg_power_1, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                         aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
#         ax1.set_title(f"Average Time-Frequency Power for {condition1_name} trials ({ch_name} - {len(condition1_epochs)} trials)")
#         ax1.set_xlabel("Time (ms)")
#         ax1.set_ylabel("Frequency (Hz)")
#         ax1.axvline(0, color="white", linestyle="--")
#         ax1.axvline(mean_RT_condition, color="red", linestyle="--")
#         fig.colorbar(im1, ax=ax1)

#         # Plot average power for condition 2
#         im2 = ax2.imshow(avg_power_2, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                         aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
#         ax2.set_title(f"Average Time-Frequency Power for {condition2_name} trials ({ch_name} - {len(condition2_epochs)} trials)")
#         ax2.set_xlabel("Time (ms)")
#         ax2.set_ylabel("Frequency (Hz)")
#         ax2.axvline(0, color="white", linestyle="--")
#         ax2.axvline(mean_RT_condition, color="red", linestyle="--")
#         fig.colorbar(im2, ax=ax2)

#         # Compute the difference between conditions
#         evoked_power_1 = epochs_power_1.mean(axis=0)
#         evoked_power_2 = epochs_power_2.mean(axis=0)
#         evoked_power_contrast = evoked_power_1 - evoked_power_2
#         signs = np.sign(evoked_power_contrast)

#         # Create new stats image with only significant clusters
#         F_obs_plot = np.nan * np.ones_like(F_obs)
#         for c, p_val in zip(clusters, cluster_p_values):
#             if p_val <= 0.05:
#                 F_obs_plot[c] = F_obs[c] * signs[c]

#         max_F = np.nanmax(abs(F_obs_plot))

#         # Plot the cluster-corrected power contrast
#         ax_diff.imshow(F_obs, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                     aspect="auto", origin="lower", cmap="gray")
#         im_diff = ax_diff.imshow(F_obs_plot, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                                 aspect="auto", origin="lower", cmap="coolwarm", vmin=-max_F, vmax=max_F)

#         ax_diff.set_title(f"{condition1_name} - {condition2_name} trials ({ch_name})")
#         ax_diff.set_xlabel("Time (ms)")
#         ax_diff.set_ylabel("Frequency (Hz)")
#         label1 = f"Mean RT"
#         ax_diff.axvline(mean_RT_condition, color="green", linestyle="--", label=label1)
#         fig.colorbar(im_diff, ax=ax_diff)

#         ax_diff.axvline(0, color="white", linestyle="--", label=centered_around)
#         fig.legend(loc='center right')

#         plt.tight_layout()
#         figtitle = (f"Power Comparison between {condition1_name} and {condition2_name} trials ({ch_name} - {session_ID}).png")
#         plt.savefig(join(saving_path,figtitle), transparent=False)


# def plot_power_comparison_between_2_conditions(
#         session_ID: str, 
#         raw: mne.io.Raw,
#         condition1_epochs: mne.Epochs, 
#         condition1_name: str, 
#         mean_RT_condition1: float, 
#         condition2_epochs: mne.Epochs, 
#         condition2_name: str,
#         mean_RT_condition2: float, 
#         channel_n: list, 
#         centered_around: str, 
#         saving_path: str,
#         fmax: int = 50,
#         vmin = None,
#         vmax = None
#         ):
#     '''
#     This function plots the power comparison between two conditions for specific channels of interest.

#     session_ID: session ID
#     raw: raw data (loaded through mne.io.read_raw)
#     condition1_epochs: mne.Epochs object for condition 1
#     condition1_name: name of condition 1
#     mean_RT_condition1: mean reaction time for condition 1 (in milliseconds)
#     condition2_epochs: mne.Epochs object for condition 2
#     condition2_name: name of condition 2
#     mean_RT_condition2: mean reaction time for condition 2 (in milliseconds)
#     channel_n: list of channel indices to plot
#     centered_around: event to center the plot around (e.g. 'GO signal')
#     saving_path: path to save the figure
#     '''

#     decim = 2
#     freqs = np.arange(1, fmax, 1)  # define frequencies of interest
#     n_cycles = freqs/2.0  # different number of cycle per frequency
#     tfr_kwargs = dict(
#         method="morlet",
#         freqs=freqs,
#         n_cycles=n_cycles,
#         decim=decim,
#         return_itc=False,
#         average=False,
#     )

#     # Compute TFR for both conditions
#     tfr_epochs_1 = condition1_epochs.compute_tfr(**tfr_kwargs)
#     tfr_epochs_2 = condition2_epochs.compute_tfr(**tfr_kwargs)

#     # Apply baseline correction
#     tfr_epochs_1.apply_baseline(mode="percent", baseline=(None, 0))
#     tfr_epochs_2.apply_baseline(mode="percent", baseline=(None, 0))

#     for ch in channel_n:
#         ch_name = raw.ch_names[ch]

#         # Extract power for single channel as 3D matrix (epochs x frequencies x times)
#         epochs_power_1 = tfr_epochs_1.data[:, ch, :, :]
#         epochs_power_2 = tfr_epochs_2.data[:, ch, :, :]

#         # Perform cluster permutation test
#         # Compute threshold based on a p-value:
#         pval = 0.05
#         dfn = 2 - 1  # degrees of freedom numerator
#         n_observations = len(epochs_power_1) + len(epochs_power_2)
#         dfd = n_observations - 2  # degrees of freedom denominator
#         threshold = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
#         print(f"Threshold = {threshold}")

#         #threshold = 6.0
#         F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
#             [epochs_power_1, epochs_power_2],
#             out_type="mask",
#             n_permutations=1000,
#             threshold=threshold,
#             tail=0,
#             seed=np.random.default_rng(seed=8675309),
#         )

#         times = 1e3 * condition1_epochs.times  # Change unit to ms

#         # Compute average power for each condition
#         avg_power_1 = epochs_power_1.mean(axis=0)
#         avg_power_2 = epochs_power_2.mean(axis=0)

#         # Find global vmin and vmax across both conditions
#         #vmin = min(avg_power_1.min()*2, avg_power_2.min())
#         #vmax = max(avg_power_1.max(), avg_power_2.max())

#         # Create figure with 3 subplots
#         fig, (ax1, ax2, ax_diff) = plt.subplots(3, 1, figsize=(15, 15))

#         # Plot average power for condition 1
#         if vmin:
#             im1 = ax1.imshow(avg_power_1, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                             aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
#         else:
#             im1 = ax1.imshow(avg_power_1, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                             aspect="auto", origin="lower", cmap="coolwarm")             
#         ax1.set_title(f"Average Time-Frequency Power for {condition1_name} trials ({ch_name} - {len(condition1_epochs)} trials)")
#         ax1.set_xlabel("Time (ms)")
#         ax1.set_ylabel("Frequency (Hz)")
#         ax1.axvline(0, color="white", linestyle="--")
#         ax1.axvline(mean_RT_condition1, color="orange", linestyle="--")
#         fig.colorbar(im1, ax=ax1)

#         # Plot average power for condition 2
#         if vmin:
#             im2 = ax2.imshow(avg_power_2, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                             aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
#         else:
#             im2 = ax2.imshow(avg_power_2, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                             aspect="auto", origin="lower", cmap="coolwarm")
#         ax2.set_title(f"Average Time-Frequency Power for {condition2_name} trials ({ch_name} - {len(condition2_epochs)} trials)")
#         ax2.set_xlabel("Time (ms)")
#         ax2.set_ylabel("Frequency (Hz)")
#         ax2.axvline(0, color="white", linestyle="--")
#         ax2.axvline(mean_RT_condition2, color="green", linestyle="--")
#         fig.colorbar(im2, ax=ax2)

#         # Compute the difference between conditions
#         evoked_power_1 = epochs_power_1.mean(axis=0)
#         evoked_power_2 = epochs_power_2.mean(axis=0)
#         evoked_power_contrast = evoked_power_1 - evoked_power_2
#         signs = np.sign(evoked_power_contrast)

#         # Create new stats image with only significant clusters
#         F_obs_plot = np.nan * np.ones_like(F_obs)
#         for c, p_val in zip(clusters, cluster_p_values):
#             if p_val <= 0.05:
#                 F_obs_plot[c] = F_obs[c] * signs[c]

#         max_F = np.nanmax(abs(F_obs_plot))

#         # Plot the cluster-corrected power contrast
#         ax_diff.imshow(F_obs, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                     aspect="auto", origin="lower", cmap="gray")
#         im_diff = ax_diff.imshow(F_obs_plot, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                                 aspect="auto", origin="lower", cmap="coolwarm", vmin=-max_F, vmax=max_F)

#         ax_diff.set_title(f"{condition1_name} - {condition2_name} trials ({ch_name})")
#         ax_diff.set_xlabel("Time (ms)")
#         ax_diff.set_ylabel("Frequency (Hz)")
#         label1 = f"Mean RT {condition1_name}"
#         label2 = f"Mean RT {condition2_name}"
#         ax_diff.axvline(mean_RT_condition1, color="orange", linestyle="--", label=label1)
#         ax_diff.axvline(mean_RT_condition2, color="green", linestyle="--", label=label2)
#         fig.colorbar(im_diff, ax=ax_diff)

#         ax_diff.axvline(0, color="white", linestyle="--", label=centered_around)
#         fig.legend(loc='center right')

#         plt.tight_layout()
#         figtitle = (f"Power Comparison between {condition1_name} and {condition2_name} trials ({ch_name} - {session_ID}).png")
#         plt.savefig(join(saving_path,figtitle), transparent=False)



# def plot_av_freq_power_by_trial(
#         epochs_subsets,
#         raw,
#         available_keys,
#         vmin,
#         vmax,
#         tfr_kwargs,
#         saving_path,
#         mean_GO_RT_successful,
#         mean_RT_unsuccessful_STOP,
#         mean_GF_RT_successful,
#         mean_GC_RT_successful
# ):
    
#     freqs = tfr_kwargs['freqs']
#     for key in available_keys:
#         fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        
#         tfr_epochs = epochs_subsets[key].compute_tfr(**tfr_kwargs)
#         tfr_epochs.apply_baseline(mode="percent", baseline=(None, 0))
        
#         for ch in [0, 1]:
#             ch_name = raw.ch_names[ch]

#             # Extract power for single channel as 3D matrix (epochs x frequencies x times)
#             epochs_power_1 = tfr_epochs.data[:, ch, :, :]
#             baseline_power = tfr_epochs.data[:, :, :, tfr_epochs.times < 0].mean(axis=-1)
#             plt.hist(baseline_power.flatten())
#             plt.title("Baseline Power Distribution")

#             times = epochs_subsets[key].times  # Change unit to ms

#             # Compute average power for each condition
#             avg_power_1 = epochs_power_1.mean(axis=0)

#             im = ax[ch].imshow(avg_power_1, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                             #vmin=vmin, vmax=vmax,
#                             aspect="auto", origin="lower", cmap="viridis")
#             ax[ch].set_title(f"Average Time-Frequency Power for {key} trials \n {ch_name} \n ({len(epochs_subsets[key])} trials)")
#             ax[ch].set_xlabel("Time (s)")
#             ax[ch].set_ylabel("Frequency (Hz)")
#             ax[ch].axvline(0, color="white", linestyle="--")
            
#             if 'GO' in key:
#                 mean_RT = mean_GO_RT_successful
#             elif 'GS' in key:
#                 mean_RT = mean_RT_unsuccessful_STOP
#             elif 'GF' in key:
#                 mean_RT = mean_GF_RT_successful
#             elif 'GC' in key:
#                 mean_RT = mean_GC_RT_successful

#             ax[ch].axvline(mean_RT / 1000, color="red", linestyle="--")
            
#             # Add colorbar
#             cbar = fig.colorbar(im, ax=ax[ch], label="Power (dB)", pad=0.1, fraction=0.04)
        
        
#         ax[0].legend(
#             loc="upper right",
#             labels=["GO signal", "Mean RT"],
#             fontsize="small",
#         )
        
#         # Save the figure
#         figtitle = f"Average Time-Frequency Power for {key} trials.png"
#         plt.savefig(join(saving_path, figtitle), transparent=True)


# def plot_av_freq_power_by_trial(
#         epochs_subsets,
#         raw,
#         available_keys,
#         vmin,
#         vmax,
#         tfr_kwargs,
#         saving_path,
#         mean_GO_RT_successful,
#         mean_RT_unsuccessful_STOP,
#         mean_GF_RT_successful,
#         mean_GC_RT_successful
# ):
#     freqs = tfr_kwargs['freqs']
#     for key in available_keys:
#         # Compute time-frequency representation for the current subset
#         tfr_epochs = epochs_subsets[key].compute_tfr(**tfr_kwargs)
#         tfr_epochs.apply_baseline(mode="percent", baseline=(None, 0))
        
#         # Plot baseline power distribution in a separate figure
#         baseline_power = tfr_epochs.data[:, :, :, tfr_epochs.times < 0].mean(axis=-1)
#         plt.figure(figsize=(6, 4))
#         plt.hist(baseline_power.flatten(), bins=50, color="skyblue", edgecolor="black")
#         plt.title(f"Baseline Power Distribution for {key} Trials")
#         plt.xlabel("Power")
#         plt.ylabel("Frequency")
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#         plt.tight_layout()
        
#         # Save the histogram plot
#         hist_title = f"Baseline_Power_Distribution_{key}.png"
#         plt.savefig(join(saving_path, hist_title), transparent=False)
#         plt.close()

#         # Time-frequency power plots
#         fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
#         for ch in [0, 1]:
#             ch_name = raw.ch_names[ch]

#             # Extract power for single channel as 3D matrix (epochs x frequencies x times)
#             epochs_power_1 = tfr_epochs.data[:, ch, :, :]
#             times = epochs_subsets[key].times  # Change unit to ms

#             # Compute average power for each condition
#             avg_power_1 = epochs_power_1.mean(axis=0)

#             # Plot average power for the channel
#             im = ax[ch].imshow(avg_power_1, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                                #vmin=vmin, vmax=vmax,
#                                aspect="auto", origin="lower", cmap="viridis")
#             ax[ch].set_title(f"Average Time-Frequency Power for {key} trials \n {ch_name} \n ({len(epochs_subsets[key])} trials)")
#             ax[ch].set_xlabel("Time (s)")
#             ax[ch].set_ylabel("Frequency (Hz)")
#             ax[ch].axvline(0, color="white", linestyle="--")

#             # Add mean RT lines
#             if 'GO' in key:
#                 mean_RT = mean_GO_RT_successful
#             elif 'GS' in key:
#                 mean_RT = mean_RT_unsuccessful_STOP
#             elif 'GF' in key:
#                 mean_RT = mean_GF_RT_successful
#             elif 'GC' in key:
#                 mean_RT = mean_GC_RT_successful

#             ax[ch].axvline(mean_RT / 1000, color="red", linestyle="--")

#             # Add colorbar
#             cbar = fig.colorbar(im, ax=ax[ch], label="Power (dB)", pad=0.1, fraction=0.04)

#         ax[0].legend(
#             loc="upper right",
#             labels=["GO signal", "Mean RT"],
#             fontsize="small",
#         )

#         # Save the time-frequency plot
#         figtitle = f"Average_Time_Frequency_Power_{key}_trials.png"
#         plt.savefig(join(saving_path, figtitle), transparent=False)
#         plt.close()


# def plot_av_freq_power_by_trial(
#         epochs_subsets,
#         raw,
#         available_keys,
#         tfr_kwargs,
#         saving_path,
#         mean_RT_dict,
#         vmin = None,
#         vmax = None,
#         apply_baseline: bool = False
# ):
#     freqs = tfr_kwargs['freqs']

#     for key in available_keys:
#         cmap = 'viridis'
#         # Compute time-frequency representation for the current subset
#         tfr_epochs = epochs_subsets[key].compute_tfr(**tfr_kwargs)

#         '''
#         # Save the baseline power distribution histogram
#         baseline_power = tfr_epochs.data[:, :, :, tfr_epochs.times < 0].mean(axis=-1)
#         plt.figure(figsize=(6, 4))
#         plt.hist(baseline_power.flatten(), bins=50, color="skyblue", edgecolor="black")
#         plt.title(f"Baseline Power Distribution for {key} Trials")
#         plt.xlabel("Power")
#         plt.ylabel("Frequency")
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         hist_title = f"Baseline_Power_Distribution_{key}.png"
#         plt.savefig(join(saving_path, hist_title), transparent=False)
#         plt.close()

#         # Plot baseline time-frequency decomposition
#         baseline_power_tfr = tfr_epochs.data[:, :, :, tfr_epochs.times < 0].mean(axis=0)  # Average over trials
#         baseline_avg_power = baseline_power_tfr.mean(axis=0)  # Average across channels

#         times_baseline = tfr_epochs.times[tfr_epochs.times < 0]
#         plt.figure(figsize=(8, 6))
#         plt.imshow(baseline_avg_power, aspect='auto', origin='lower',
#                    extent=[times_baseline[0], times_baseline[-1], freqs[0], freqs[-1]],
#                    cmap=cmap)
#         plt.colorbar(label="Power")
#         plt.title(f"Baseline Time-Frequency Decomposition for {key} Trials")
#         plt.xlabel("Time (s)")
#         plt.ylabel("Frequency (Hz)")
#         plt.tight_layout()
#         baseline_plot_title = f"Baseline_Time_Frequency_{key}.png"
#         plt.savefig(join(saving_path, baseline_plot_title), transparent=False)
#         plt.close()
#         '''

#         # Apply baseline normalization if needed
#         if apply_baseline:
#             tfr_epochs.apply_baseline(mode="percent", baseline=(None, 0))

#         # Time-frequency power plots
#         fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
#         for ch in [0, 1]:
#             ch_name = raw.ch_names[ch]

#             # Extract power for single channel as 3D matrix (epochs x frequencies x times)
#             epochs_power_1 = tfr_epochs.data[:, ch, :, :]
#             times = epochs_subsets[key].times  # Change unit to ms

#             # Compute average power for each condition
#             avg_power_1 = epochs_power_1.mean(axis=0)


#             # Plot average power for the channel
#             if apply_baseline:
#                 cmap = 'coolwarm'  # use a divergent colormap only when baseline correction has been applied
#             if vmin == None:
#                 im = ax[ch].imshow(avg_power_1, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                                 aspect="auto", origin="lower", cmap=cmap)
#             else:
#                 im = ax[ch].imshow(avg_power_1, extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                                    vmin = vmin, vmax = vmax,
#                                    aspect="auto", origin="lower", cmap=cmap)                 
#             ax[ch].set_title(f"Average Time-Frequency Power for {key} trials \n {ch_name} \n ({len(epochs_subsets[key])} trials)")
#             ax[ch].set_xlabel("Time (s)")
#             ax[ch].set_ylabel("Frequency (Hz)")
#             ax[ch].axvline(0, color="white", linestyle="--")

#             # Add mean RT lines
#             if 'GO' in key:
#                 mean_RT = mean_RT_dict['GO_successful']
#             elif 'GS' in key:
#                 mean_RT = mean_RT_dict['GS_unsuccessful']
#             elif 'GF' in key:
#                 mean_RT = mean_RT_dict['GF_successful']
#             elif 'GC' in key:
#                 mean_RT = mean_RT_dict['GC_successful']

#             ax[ch].axvline(mean_RT / 1000, color="red", linestyle="--")

#             # Add colorbar
#             cbar = fig.colorbar(im, ax=ax[ch], pad=0.1, fraction=0.04)

#         ax[0].legend(
#             loc="upper right",
#             labels=["GO signal", "Mean RT"],
#             fontsize="small",
#         )

#         # Save the time-frequency plot
#         if apply_baseline:
#             figtitle = f"Baseline corrected - Average_Time_Frequency_Power_{key}_trials.png"
#         else:
#              figtitle = f"No baseline correction applied - Average_Time_Frequency_Power_{key}_trials.png"
#         plt.savefig(join(saving_path, figtitle), transparent=False)
#         plt.close()
