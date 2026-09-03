import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import mne
import os

from functions.analysis import identify_significant_clusters
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test


def meta_function_tfr_intra_group(
        tfr_dict, # store tfr results
        sub_group, # specify which group to analyze (e.g. 'DBS OFF', 'DBS ON', 'controls')
        sub_dict_epochs, # dictionary containing all subjects' data
        single_sub, # True or False, whether to plot single subject TFRs or group averages
        epochs_cond, # what condition(s) to analyse (e.g. ['GO_successful_square'], ['GO_successful_square', 'GS_successful_square'])
        roi, # region of interest (e.g. ['Left_STN'], ['Right_STN'], ['Left_STN', 'Right_STN']). A list creates an average of all channels in the list.
        tfr_args, # dictionary containing tfr parameters (e.g. frequencies of interest, number of cycles, etc.)
        tmin_tmax = [0, 1.5], # time window of interest for plotting (e.g. [-0.5, 1.5])
        plot_contrast = False, # True or False, whether to plot contrasts between two conditions
        plot_single_condition = True, # True or False, whether to plot single conditions
        vmin_vmax = [-70, 70], # min and max values for color scale (e.g. [-70, 70] if in percentage change)
        baseline_correction = True, # True or False, whether to apply baseline correction
        baseline_correction_method = 'group_average', # method for baseline correction (e.g. 'group_average' or 'single_trial')
        saving_path = "C:\\Users\\Juliette\\Research\\Projects\\analysis_mSST\\results", # path to save the figures
        save_as = 'png', # format to save the figures (e.g. 'png', 'pdf')
        threshold_gc_dict = None # dictionary containing the GC threshold for each subject (only needed if using slowGC or fastGC conditions)
):
    """
    epochs_cond can be a list of one condition (e.g. ['GO_successful_square']) or a 
    list of two conditions to compare (e.g. ['GO_successful_square', 'GS_successful_square']). 
    The structure should always be 'condition_outcome_(cue to align on)'.
    For example:
    GS_successful_square means Stop Trials, successful trials, aligned to Go cue.
    GS_successful_triangle means Stop Trials, successful trials, aligned to Stop cue.
    GS_unsuccessful_feedback means Stop Trials, unsuccessful trials, aligned to feedback.
    GO_successful_response means Go Trials, successful trials, aligned to response.
    """

    # Validate epochs_cond format
    assert all(epochs_cond[i].split('_')[0] in ['lmGO', 'GO', 'GS', 'GF', 'GC', 'slowGC', 'fastGC'] for i in range(len(epochs_cond))), "Condition should start with 'lmGO', 'GO', 'GF', 'GC','GS', 'slowGC' or 'fastGC' " 

    if plot_contrast:
        if len(epochs_cond) == 2:
            suffix0 = epochs_cond[0].split('_')[-1]
            suffix1 = epochs_cond[1].split('_')[-1]
            if not (suffix0 == suffix1 or {suffix0, suffix1} == {"stop", "continue"}):
                raise ValueError("Both conditions should be aligned to the same cue for comparison (or be stop/continue)")
    # Validate plot_contrast parameter:
    if plot_contrast and len(epochs_cond) != 2:
        raise ValueError("To plot contrasts, exactly two conditions must be provided in epochs_cond.")


    # for group in sub_groups:
    for coi in epochs_cond:
        all_percentage_change = []
        subs_included = []

        # Then, extract the relevant epochs decompositions from sub_dict based on the specified conditions and ROIs.
        # sub-select epochs/subjects based on sub_groups
        group_key = sub_group if 'DBS' in sub_group else 'C'
        selected_sub_dict_epochs = {key: value for key, value in sub_dict_epochs.items() if group_key in key}
        mean_rt_all = []
        mean_ssd_all = []

        for sub, epochs in selected_sub_dict_epochs.items():        
            if single_sub: 
                all_percentage_change = []
                sub_num = 1

            threshold_GC = threshold_gc_dict.get(sub) if any(x in coi for x in ['slowGC', 'fastGC']) else None
            print(f"Processing sub {sub}")
            mean_power, times, freqs, mean_rt, mean_ssd = get_tfr_decomposition(
                epochs = epochs, 
                cond_of_interest = coi, 
                ch_names = roi, 
                tfr_args = tfr_args, 
                baseline_correction= baseline_correction,
                baseline_correction_method= baseline_correction_method,
                tmin_tmax = tmin_tmax,
                threshold_GC = threshold_GC
            )
            all_percentage_change.append(mean_power)
            subs_included.append(sub)
            mean_rt_all.append(mean_rt)
            mean_ssd_all.append(mean_ssd)

            # Then plot the average TFR if requested
            if single_sub : 
                sub_num = 1
                if plot_single_condition:
                    plot_tfr_result(
                        group = sub_group,
                        all_percentage_change = all_percentage_change,
                        times = times,
                        epochs_cond = coi,
                        roi = roi,
                        tfr_args = tfr_args,
                        tmin_tmax = tmin_tmax,
                        vmin_vmax = vmin_vmax,
                        baseline_correction = baseline_correction,
                        baseline_correction_method = baseline_correction_method,
                        sub_num = sub_num,
                        sub = sub,
                        mean_rt_all = mean_rt,
                        mean_ssd_all = mean_ssd,
                        saving_path = saving_path,
                        save_as = save_as
                        )

        if not single_sub:
            sub_num = len(subs_included)
            # Convert to array (shape: (n_subs, n_freqs, n_times))
            min_len = min(e.shape[1] for e in all_percentage_change)
            all_percentage_change = [e[:, :min_len] for e in all_percentage_change]
            all_percentage_change = np.stack(all_percentage_change, axis=0)

            if plot_single_condition:
                plot_tfr_result(
                    group = sub_group,
                    all_percentage_change = all_percentage_change,
                    times = times,
                    epochs_cond = coi,
                    roi = roi,
                    tfr_args = tfr_args,
                    tmin_tmax = tmin_tmax,
                    vmin_vmax = vmin_vmax,
                    baseline_correction = baseline_correction,
                    baseline_correction_method = baseline_correction_method,
                    sub_num = sub_num,
                    sub = None,
                    mean_rt_all = mean_rt_all,
                    mean_ssd_all = mean_ssd_all,
                    saving_path = saving_path,
                    save_as = save_as
                    )     

            if len(roi) > 1:
                tfr_dict[sub_group][coi] = {
                    'all_percentage_change': all_percentage_change,
                    'mean_rt_all': mean_rt_all,
                    'mean_ssd_all': mean_ssd_all,
                    'times': times,
                    'freqs': freqs} 
            else:
                tfr_dict[sub_group][coi] = {
                    'all_percentage_change': all_percentage_change,
                    roi[0]: all_percentage_change,
                    'mean_rt_all': mean_rt_all,
                    'mean_ssd_all': mean_ssd_all,
                    'times': times,
                    'freqs': freqs}                 


# If needed, add statistical tests between conditions and add to the plot.
    if plot_contrast:
        # subtract the percentage changes of the two conditions
        # check that the two conditions have the same shape
        # correct shape if needed:
        shape0 = tfr_dict[sub_group][epochs_cond[0]]['all_percentage_change'].shape
        shape1 = tfr_dict[sub_group][epochs_cond[1]]['all_percentage_change'].shape
        if shape0[2] != shape1[2]:
            print(f"The two conditions have different time dimensions: {shape0[2]} vs {shape1[2]}. Cropping to the minimum length for contrast.")
        min_len = min(shape0[2], shape1[2])
        tfr_dict[sub_group][epochs_cond[0]]['all_percentage_change'] = tfr_dict[sub_group][epochs_cond[0]]['all_percentage_change'][:, :, :min_len]
        tfr_dict[sub_group][epochs_cond[1]]['all_percentage_change'] = tfr_dict[sub_group][epochs_cond[1]]['all_percentage_change'][:, :, :min_len]
        times = times[:min_len]
        tfr_dict[sub_group][coi]['times'] = times
        
        contrast_percentage_change = tfr_dict[sub_group][epochs_cond[0]]['all_percentage_change'] - tfr_dict[sub_group][epochs_cond[1]]['all_percentage_change']
        tfr_dict[sub_group]['contrast'] = {
            'all_percentage_change': contrast_percentage_change,
            'times': times,
            'freqs': freqs}
        
        plot_tfr_result_contrast(
            group = sub_group,
            tfr_dict = tfr_dict,
            epochs_cond = epochs_cond,
            roi = roi,
            tfr_args = tfr_args,
            tmin_tmax = tmin_tmax,
            vmin_vmax = vmin_vmax,
            baseline_correction = baseline_correction,
            baseline_correction_method = baseline_correction_method,
            sub_num = sub_num,
            saving_path = saving_path,
            save_as = save_as            
            )    
    
    return tfr_dict


def meta_function_tfr_inter_groups(
        tfr_dict, # store tfr results
        groups, # specify which groups to compare (e.g. ['DBS OFF', 'DBS ON'], ['DBS OFF', 'controls'])
        sub_dict_epochs, # dictionary containing all subjects' data
        single_sub, # True or False, whether to plot single subject TFRs or group averages
        epochs_cond, # what condition(s) to analyse (e.g. ['GO_successful_square'], ['GO_successful_square', 'GS_successful_square'])
        roi, # region of interest (e.g. ['Left_STN'], ['Right_STN'], ['Left_STN', 'Right_STN']). A list of multiple channels will return an average of all channels in the list.
        tfr_args, # dictionary containing tfr parameters (e.g. frequencies of interest, number of cycles, etc.)
        tmin_tmax = [0, 1.5], # time window of interest for plotting (e.g. [-0.5, 1.5])
        plot_contrast_groups = True, # True or False, whether to plot contrasts between two groups
        plot_contrast_cond_group = False, # True or False, whether to plot contrasts between conditions and between two groups
        plot_single_condition = True, # True or False, whether to plot single conditions
        vmin_vmax = [-70, 70], # min and max values for color scale (e.g. [-70, 70] if in percentage change)
        baseline_correction = True, # True or False, whether to apply baseline correction
        baseline_correction_method = 'group_average', # method for baseline correction (e.g. 'group_average' or 'single_trial')
        saving_path = "C:\\Users\\Juliette\\Research\\Projects\\analysis_mSST\\results", # path to save the figures
        save_as = 'png', # format to save the figures (e.g. 'png', 'pdf')
        threshold_gc_dict = None # dictionary containing the GC threshold for each subject (only needed if using slowGC or fastGC conditions)
):
    """
    epochs_cond can be a list of one condition (e.g. ['GO_successful_square']) or a 
    list of two conditions to compare (e.g. ['GO_successful_square', 'GS_successful_square']). 
    The structure should always be 'condition_outcome_(cue to align on)'.
    For example:
    GS_successful_square means Stop Trials, successful trials, aligned to Go cue.
    GS_successful_triangle means Stop Trials, successful trials, aligned to Stop cue.
    GS_unsuccessful_feedback means Stop Trials, unsuccessful trials, aligned to feedback.
    GO_successful_response means Go Trials, successful trials, aligned to response.
    """
    
    # Validate epochs_cond format
    assert all(epochs_cond[i].split('_')[0] in ['lmGO', 'GO', 'GS', 'GF', 'GC', 'slowGC', 'fastGC'] for i in range(len(epochs_cond))), "Condition should start with 'lmGO', 'GO', 'GF', 'GC', 'slowGC' or 'fastGC'" 

    if plot_contrast_cond_group:
        if len(epochs_cond) == 2:
            suffix0 = epochs_cond[0].split('_')[-1]
            suffix1 = epochs_cond[1].split('_')[-1]
            if not (suffix0 == suffix1 or {suffix0, suffix1} == {"stop", "continue"}):
                raise ValueError("Both conditions should be aligned to the same cue for comparison (or be stop/continue)")
    # Validate plot_contrast parameter:
    if plot_contrast_cond_group and len(epochs_cond) != 2:
        raise ValueError("To plot contrasts, exactly two conditions must be provided in epochs_cond.")

    if len(groups) != 2:
        raise ValueError("To compare groups, exactly two groups must be provided in groups.")

    # for group in sub_groups:
    for coi in epochs_cond:
        for sub_group in groups:
            all_percentage_change = []
            subs_included = []

            # Then, extract the relevant epochs decompositions from sub_dict based on the specified conditions and ROIs.
            # sub-select epochs/subjects based on sub_groups
            group_key = sub_group if 'DBS' in sub_group else 'C'
            selected_sub_dict_epochs = {key: value for key, value in sub_dict_epochs.items() if group_key in key}
            # selected_sub_dict_epochs = {key: value for key, value in sub_dict_epochs.items() if sub_group in key}
            mean_rt_all = []
            mean_ssd_all = []

            for sub, epochs in selected_sub_dict_epochs.items():        
                if single_sub: 
                    all_percentage_change = []
                    sub_num = 1

                threshold_GC = threshold_gc_dict.get(sub) if any(x in coi for x in ['slowGC', 'fastGC']) else None
                mean_power, times, freqs, mean_rt, mean_ssd = get_tfr_decomposition(
                    epochs = epochs, 
                    cond_of_interest = coi, 
                    ch_names = roi, 
                    tfr_args = tfr_args, 
                    baseline_correction= baseline_correction,
                    baseline_correction_method= baseline_correction_method,
                    tmin_tmax = tmin_tmax,
                    threshold_GC = threshold_GC
                )
                all_percentage_change.append(mean_power)
                subs_included.append(sub)
                mean_rt_all.append(mean_rt)
                mean_ssd_all.append(mean_ssd)

                # Then plot the average TFR if requested
                if single_sub : 
                    sub_num = 1
                    if plot_single_condition:
                        plot_tfr_result(
                            group = sub_group,
                            all_percentage_change = all_percentage_change,
                            times = times,
                            epochs_cond = coi,
                            roi = roi,
                            tfr_args = tfr_args,
                            tmin_tmax = tmin_tmax,
                            vmin_vmax = vmin_vmax,
                            baseline_correction = baseline_correction,
                            baseline_correction_method = baseline_correction_method,
                            sub_num = sub_num,
                            sub = sub,
                            mean_rt_all = mean_rt,
                            mean_ssd_all = mean_ssd,
                            saving_path = saving_path,
                            save_as = save_as
                            )

            if not single_sub:
                sub_num = len(subs_included)
                # Convert to array (shape: (n_subs, n_freqs, n_times))
                min_len = min(e.shape[1] for e in all_percentage_change)
                all_percentage_change = [e[:, :min_len] for e in all_percentage_change]
                all_percentage_change = np.stack(all_percentage_change, axis=0)

                if plot_single_condition:
                    plot_tfr_result(
                        group = sub_group,
                        all_percentage_change = all_percentage_change,
                        times = times,
                        epochs_cond = coi,
                        roi = roi,
                        tfr_args = tfr_args,
                        tmin_tmax = tmin_tmax,
                        vmin_vmax = vmin_vmax,
                        baseline_correction = baseline_correction,
                        baseline_correction_method = baseline_correction_method,
                        sub_num = sub_num,
                        sub = None,
                        mean_rt_all = mean_rt_all,
                        mean_ssd_all = mean_ssd_all,
                        saving_path = saving_path,
                        save_as = save_as
                        )     

                tfr_dict[sub_group][coi] = {
                    'all_percentage_change': all_percentage_change,
                    'mean_rt_all': mean_rt_all,
                    'mean_ssd_all': mean_ssd_all,
                    'times': times,
                    'freqs': freqs} 

        if plot_contrast_groups:
            # shape0 = tfr_dict[groups[0]][coi]['all_percentage_change'].shape
            # shape1 = tfr_dict[groups[1]][coi]['all_percentage_change'].shape
            # if shape0[2] != shape1[2]:
            #     print(f"The two groups have different time dimensions: {shape0[2]} vs {shape1[2]}. Cropping to the minimum length for contrast.")
            # min_len = min(shape0[2], shape1[2])
            # tfr_dict[groups[0]][coi]['all_percentage_change'] = tfr_dict[groups[0]][coi]['all_percentage_change'][:, :, :min_len]
            # tfr_dict[groups[1]][coi]['all_percentage_change'] = tfr_dict[groups[1]][coi]['all_percentage_change'][:, :, :min_len]
            # times = times[:min_len]
            # # tfr_dict[coi]['times'] = times
            # contrast_percentage_change = tfr_dict[groups[0]][coi]['all_percentage_change'] - tfr_dict[groups[1]][coi]['all_percentage_change']
            # tfr_dict[coi] = {}
            # tfr_dict[coi]['contrast_groups'] = {
            #     'all_percentage_change': contrast_percentage_change,
            #     'times': times,
            #     'freqs': freqs}
            
            plot_tfr_result_contrast_groups(
                groups = groups,
                tfr_dict = tfr_dict,
                epochs_cond = coi,
                roi = roi,
                tfr_args = tfr_args,
                tmin_tmax = tmin_tmax,
                vmin_vmax = vmin_vmax,
                baseline_correction = baseline_correction,
                baseline_correction_method = baseline_correction_method,
                sub_num = sub_num,
                saving_path = saving_path,
                save_as = save_as
                )

    # If needed, add statistical tests between conditions and add to the plot.
    if plot_contrast_cond_group:
        # subtract the percentage changes of the two conditions
        # check that the two conditions have the same shape
        # correct shape if needed:
        # shape0 = tfr_dict[sub_group][epochs_cond[0]]['all_percentage_change'].shape
        # shape1 = tfr_dict[sub_group][epochs_cond[1]]['all_percentage_change'].shape
        # if shape0[2] != shape1[2]:
        #     print(f"The two conditions have different time dimensions: {shape0[2]} vs {shape1[2]}. Cropping to the minimum length for contrast.")
        # min_len = min(shape0[2], shape1[2])
        # tfr_dict[sub_group][epochs_cond[0]]['all_percentage_change'] = tfr_dict[sub_group][epochs_cond[0]]['all_percentage_change'][:, :, :min_len]
        # tfr_dict[sub_group][epochs_cond[1]]['all_percentage_change'] = tfr_dict[sub_group][epochs_cond[1]]['all_percentage_change'][:, :, :min_len]
        # times = times[:min_len]
        # tfr_dict[sub_group][coi]['times'] = times
        
        # contrast_percentage_change = tfr_dict[sub_group][epochs_cond[0]]['all_percentage_change'] - tfr_dict[sub_group][epochs_cond[1]]['all_percentage_change']
        # tfr_dict[sub_group]['contrast'] = {
        #     'all_percentage_change': contrast_percentage_change,
        #     'times': times,
        #     'freqs': freqs}
        
        plot_tfr_result_contrast(
            group = sub_group,
            tfr_dict = tfr_dict,
            epochs_cond = epochs_cond,
            roi = roi,
            tfr_args = tfr_args,
            tmin_tmax = tmin_tmax,
            vmin_vmax = vmin_vmax,
            baseline_correction = baseline_correction,
            baseline_correction_method = baseline_correction_method,
            sub_num = sub_num,
            saving_path = saving_path,
            save_as = save_as            
            )    
        
    return tfr_dict



def get_tfr_decomposition(
        epochs, 
        cond_of_interest, 
        ch_names, 
        tfr_args, 
        baseline_correction, 
        baseline_correction_method, 
        tmin_tmax,
        threshold_GC=None,
        handle_bads: str = 'ignore' # options: 'ignore', 'drop'
        ):
    latency_matched = False
    slowGC = False
    fastGC = False
    mean_rt = None
    mean_ssd = None
    
    if cond_of_interest.split('_')[0] == 'lmGO':
        latency_matched = True
        epoch_type = 'GO'
    elif cond_of_interest.split('_')[0] == 'slowGC':
        slowGC = True
        epoch_type = 'GC'
    elif cond_of_interest.split('_')[0] == 'fastGC':
        fastGC = True
        epoch_type = 'GC'
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

    if slowGC:
        rt = np.asarray(data.metadata['key_resp_experiment.rt'])
        ssd = np.asarray(data.metadata['continue_signal_time'])
        threshold = threshold_GC/1000 # this value should be added for each trial to the SSD 
        rt_from_continue = rt - ssd
        slow_mask = rt_from_continue >= threshold
        # slow_mask = rt >= (ssd + threshold)
        data = data[slow_mask]

    if fastGC:
        rt = np.asarray(data.metadata['key_resp_experiment.rt'])
        ssd = np.asarray(data.metadata['continue_signal_time'])
        threshold = threshold_GC/1000 # this value should be added for each trial to the SSD 
        rt_from_continue = rt - ssd
        fast_mask = rt_from_continue < threshold
        data = data[fast_mask]

    # get RTs and SSDs for later plotting:
    rt = np.asarray(data.metadata['key_resp_experiment.rt'])
    if np.any(~np.isnan(rt)):        # at least one value that is not nan
        mean_rt = np.nanmean(rt)
    else:
        mean_rt = None

    ssd = None
    if epoch_type == 'GS':
        ssd = np.asarray(data.metadata['stop_signal_time'])
    elif epoch_type == 'GC':
        ssd = np.asarray(data.metadata['continue_signal_time'])
    mean_ssd = np.nanmean(ssd) if ssd is not None else None

    # Select only desired channels
    # check if channels are bad and handle them according to the specified strategy
    if handle_bads == 'drop':  # THIS STRATEGY IS NOT READY YET, DON'T USE IT FOR NOW
        print('Warning: Dropping bad channels. This strategy is not fully implemented yet.')
        ch_names = [ch for ch in ch_names if ch not in epochs.info['bads']]
    elif handle_bads == 'ignore':
        print(f'Ignoring bad channels. They will be included in the analysis even if the following channels were initially labeled as bads: {data.info["bads"]}')
        # remove "bads" annotation to ignore that they are bad channels
        data.info['bads'] = []

    epochs = data.copy().pick(ch_names)

    # Compute TFR
    power = epochs.compute_tfr(**tfr_args)  # shape: (n_epochs, n_channels, n_freqs, n_times)      
    power.data *= 1e12  # V² → (µV)²
    
    # Average across channels if multiple channels are specified
    if len(ch_names) > 1:
        power_mean = np.nanmean(power.data, axis=1)  # (n_epochs, n_freqs, n_times)
    else:
        power_mean = power.data[:, 0, :, :]  # (n_epochs, n_freqs, n_times)

    times = power.times * 1000
    dt_ms = 1000/round(epochs.info['sfreq'])
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
            # baseline_power = np.empty((power_mean.shape[0], power_mean.shape[1], 1))  # (n_trials, n_freqs, 1)
            mean_power = np.nanmean(power_mean, axis=0)  # (n_freqs, n_times)
            # Define baseline period for change calculation
            baseline_indices = (times >= -500) & (times <= -200)
            baseline_power = np.nanmean(mean_power[:, baseline_indices], axis=1, keepdims=True)  # shape: (n_freqs, 1 time)            
        
        for i in range(power_mean.shape[0]):
            t0_idx = t0_per_trial[i] + tmin_tmax[0]*1000
            t1_idx = t0_per_trial[i] + tmin_tmax[1]*1000
            time_idx = (times >= t0_idx) & (times <= t1_idx)
            new_epochs.append(power_mean[i][:, time_idx])  # shape: (n_freqs, n_times_new)

        min_len = min(e.shape[1] for e in new_epochs)
        new_epochs = [e[:, :min_len] for e in new_epochs]
        new_epochs = np.stack(new_epochs, axis=0)  # shape: (n_epochs, n_freqs, n_times_new)
        new_times = np.arange(tmin_tmax[0]*1000, tmin_tmax[1]*1000, dt_ms)
        mean_power = np.nanmean(new_epochs, axis=0)
        times = new_times
        change = (mean_power - baseline_power) / baseline_power * 100  # percent change  # shape: (n_freqs, n_times)
        # mean_power = change

    else:
        mean_power = np.nanmean(power_mean, axis=0)  # (n_freqs, n_times)

        if baseline_correction and baseline_correction_method == 'group_average':
            # Define baseline period for change calculation
            baseline_indices = (times >= -500) & (times <= -200)
            baseline_power = np.nanmean(mean_power[:, baseline_indices], axis=1, keepdims=True)  # shape: (n_freqs, 1 time)
            change = (mean_power - baseline_power) / baseline_power * 100  # percent change  # shape: (n_freqs, n_times)
            # mean_power = change
        
        time_idx = (times >= tmin_tmax[0]*1000) & (times <= tmin_tmax[1]*1000)
        change = change[:, time_idx]
        times = times[time_idx]

    return change, times, freqs, mean_rt, mean_ssd        


def plot_tfr_result(
        group,
        all_percentage_change,
        times,
        epochs_cond,
        roi,
        tfr_args,
        tmin_tmax,
        vmin_vmax,
        baseline_correction,
        baseline_correction_method,
        sub_num,
        sub = None,
        mean_rt_all = None,
        mean_ssd_all = None,
        saving_path = None,
        save_as = 'png'
        ):
    add_rt = False
    add_ssd = False

    title_prefix = "Power change" if baseline_correction else "Power"
    bc_note = "" if baseline_correction else ", no baseline correction"
    subject_info = f"nSub = {sub_num}" if sub_num > 1 else sub
    cond_type, outcome = epochs_cond.split('_')[:2]

    # Set the x label based on what the epochs are centered on:
    aligned_on = epochs_cond.split('_')[-1]
    if aligned_on == 'response':
        xlabel = 'Time from RESPONSE (s)'
    elif aligned_on == 'feedback':
        xlabel = 'Time from FEEDBACK (s)'
    elif aligned_on == 'square':
        xlabel = 'Time from GO cue (s)'
        add_rt = True if (
        (cond_type in ('lmGO', 'GO', 'GF', 'GC', 'slowGC', 'fastGC') and outcome == 'successful') or
        (cond_type == 'GS' and outcome == 'unsuccessful')
        ) else False
        add_ssd = True if any(x in epochs_cond for x in ('GS', 'GC', 'slowGC', 'fastGC')) else False
    elif aligned_on == 'triangle':
        xlabel = 'Time from STOP/CONTINUE cue (s)'
        add_rt = True if (cond_type == 'GS' and outcome == 'unsuccessful') or (cond_type in ('GC', 'slowGC', 'fastGC') and outcome == 'successful') else False

    # Compute grand averages
    avg_percentage_change = np.nanmean(all_percentage_change, axis=0)       

    # Compute min and max along the frequency axis
    if not baseline_correction:
        min_values = np.min(avg_percentage_change)  # Shape: (n_subjects, n_times)
        max_values = np.max(avg_percentage_change)
        
    # Plot Left STN
    # plt.figure(figsize=(4, 6))
    fig, ax = plt.subplots(figsize=(4, 6), constrained_layout=True)
    vmin, vmax = (vmin_vmax if baseline_correction else (min_values, max_values))
    ax.imshow(avg_percentage_change, aspect='auto', origin='lower', 
                            extent=[times[0], times[-1], tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency (Hz)')
    title_text = f"{title_prefix} \n {group} \n {epochs_cond} \n {roi} \n {subject_info}{bc_note}"
    fig.suptitle(title_text, fontsize=10)
    # ax.set_title(f"{title_prefix} \n {group} \n {epochs_cond} \n {roi} \n {subject_info}{bc_note}")
    if add_rt:
        avg_rt = np.nanmean(mean_rt_all)*1000
        if aligned_on == 'triangle':
            avg_rt -= np.nanmean(mean_ssd_all)*1000
        ax.axvline(x=avg_rt, color='black', linestyle='-', label='Mean RT')
    if add_ssd and mean_ssd_all is not None:
        avg_ssd = np.nanmean(mean_ssd_all)*1000
        ax.axvline(x=avg_ssd, color='k', linestyle='--', label='Mean SSD')
        
    if baseline_correction:
        if baseline_correction_method == 'single_trial':
            colorbar_label = 'Change from baseline (dB)'
        else:
            colorbar_label = 'Percent change from baseline (%)'
    else:
        colorbar_label = 'Mean Power (µV²)'
    fig.colorbar(ax.images[0], label=colorbar_label)    
    fig.savefig(os.path.join(saving_path, f"{group}_{epochs_cond}_{roi}_{subject_info}.{save_as}"))



def plot_tfr_result_contrast(
        group,
        tfr_dict,
        epochs_cond,
        roi,
        tfr_args,
        tmin_tmax,
        vmin_vmax,
        baseline_correction,
        baseline_correction_method,
        sub_num,
        saving_path,
        save_as = 'png'
        ):
    add_rt1 = False
    add_rt2 = False
    add_ssd1 = False
    add_ssd2 = False

    contrast = epochs_cond[0] + ' - ' + epochs_cond[1]

    title_prefix = "Power change" if baseline_correction else "Power"
    bc_note = "" if baseline_correction else ", no baseline correction"
    subject_info = f"nSub = {sub_num}" #if sub_num > 1 else sub
    cond_type1, outcome1 = epochs_cond[0].split('_')[:2]
    cond_type2, outcome2 = epochs_cond[1].split('_')[:2]

    # Set the x label based on what the epochs are centered on:
    aligned_on = epochs_cond[0].split('_')[-1] # both conditions should always be centered on the same cue

    if aligned_on == 'response':
        xlabel = 'Time from RESPONSE (s)'
    elif aligned_on == 'feedback':
        xlabel = 'Time from FEEDBACK (s)'
    elif aligned_on == 'square':
        xlabel = 'Time from GO cue (s)'
        add_rt1 = True if (
        (cond_type1 in ('lmGO', 'GO', 'GF', 'GC', 'slowGC', 'fastGC') and outcome1 == 'successful') or
        (cond_type1 == 'GS' and outcome1 == 'unsuccessful')
        ) else False
        add_rt2 = True if (
        (cond_type2 in ('lmGO', 'GO', 'GF', 'GC', 'slowGC', 'fastGC') and outcome2 == 'successful') or
        (cond_type2 == 'GS' and outcome2 == 'unsuccessful')
        ) else False
        add_ssd1 = True if any(x in epochs_cond[0] for x in ('GS', 'GC', 'slowGC', 'fastGC')) else False
        add_ssd2 = True if any(x in epochs_cond[1] for x in ('GS', 'GC', 'slowGC', 'fastGC')) else False
    elif aligned_on == 'triangle':
        xlabel = 'Time from STOP/CONTINUE cue (s)'
        add_rt1 = True if (cond_type1 == 'GS' and outcome1 == 'unsuccessful') or (cond_type1 in ('GC', 'slowGC', 'fastGC') and outcome1 == 'successful') else False
        add_rt2 = True if (cond_type2 == 'GS' and outcome2 == 'unsuccessful') or (cond_type2 in ('GC', 'slowGC', 'fastGC') and outcome2 == 'successful') else False

    all_percentage_change = tfr_dict[group]['contrast']['all_percentage_change']
    mean_rt_all1 = tfr_dict[group][epochs_cond[0]]['mean_rt_all']
    mean_rt_all2 = tfr_dict[group][epochs_cond[1]]['mean_rt_all']
    mean_ssd_all1 = tfr_dict[group][epochs_cond[0]]['mean_ssd_all']
    mean_ssd_all2 = tfr_dict[group][epochs_cond[1]]['mean_ssd_all']
    n_obs = all_percentage_change.shape[0]
    pval = 0.05
    # df = n_obs - 1
    # threshold = scipy.stats.t.ppf(1-pval / 2, df) # two-tailed distribution
    threshold = None
    n_permutations = 1000

    # Compute permutation cluster test for the left stn
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
    all_percentage_change, n_permutations=n_permutations,
    threshold=threshold, tail=0,
    out_type= "mask", seed=11111, verbose=True)
    print(f"p_values: {cluster_p_values}")
    
    condition = group + '_' + contrast
    identify_significant_clusters(
        cluster_p_values, 
        clusters,
        tfr_dict[group]['contrast']['times'],
        T_obs,
        pval,
        tfr_args,
        condition = condition,
        roi = roi,
        saving_path = saving_path
        )

    # Compute grand averages
    avg_percentage_change = np.nanmean(all_percentage_change, axis=0)     
    assert T_obs.shape == avg_percentage_change.shape  

    # Compute min and max along the frequency axis
    if not baseline_correction:
        min_values = np.min(avg_percentage_change)  # Shape: (n_subjects, n_times)
        max_values = np.max(avg_percentage_change)
        
    # Plot
    # plt.figure(figsize=(4, 6))
    fig, ax = plt.subplots(figsize=(4, 6), constrained_layout=True)
    vmin, vmax = (vmin_vmax if baseline_correction else (min_values, max_values))
    times = tfr_dict[group]['contrast']['times'] 
    freqs = tfr_dict[group]['contrast']['freqs']
    ax.imshow(avg_percentage_change, aspect='auto', origin='lower', 
                            # extent=[tmin_tmax[0]*1000, tmin_tmax[1]*1000, tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            extent = [times[0], times[-1], freqs[0], freqs[-1]],
                            cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency (Hz)')
    title_text = f"{title_prefix} \n {group} \n {contrast} \n {roi} \n {subject_info}{bc_note}"
    fig.suptitle(title_text, fontsize=10)
    # ax.set_title(f"{title_prefix} \n {contrast} \n {roi} \n {subject_info}{bc_note}")
        
    if baseline_correction:
        if baseline_correction_method == 'single_trial':
            colorbar_label = 'Change from baseline (dB)'
        else:
            colorbar_label = 'Percent change from baseline (%)'
    else:
        colorbar_label = 'Mean Power (µV²)'
    fig.colorbar(ax.images[0], label=colorbar_label)   
    
    # add significant clusters on the plot if group-level analysis:
    if sub_num > 1:
        for c, p_val in zip(clusters, cluster_p_values):
            if p_val <= 0.05:
                mask = np.zeros_like(T_obs, dtype=bool)  # Explicitly match dimensions
                mask[c] = True
                ax.contour(mask, levels=[0.5], colors='black', linewidths=1.5,
                            extent=[tmin_tmax[0]*1000, tmin_tmax[1]*1000, tfr_args["freqs"][0], tfr_args["freqs"][-1]])

    if add_rt1:
        avg_rt1 = np.nanmean(mean_rt_all1)*1000
        if aligned_on == 'triangle':
            avg_rt1 -= np.nanmean(mean_ssd_all1)*1000
        ax.axvline(x=avg_rt1, color='black', linestyle='-', label=f'Mean RT {epochs_cond[0]}')
    if add_rt2:
        avg_rt2 = np.nanmean(mean_rt_all2)*1000
        if aligned_on == 'triangle':
            avg_rt2 -= np.nanmean(mean_ssd_all2)*1000
        ax.axvline(x=avg_rt2, color='gray', linestyle='-', label=f'Mean RT {epochs_cond[1]}')
    if add_ssd1 and mean_ssd_all1 is not None:
        avg_ssd1 = np.nanmean(mean_ssd_all1)*1000
        ax.axvline(x=avg_ssd1, color='k', linestyle='--', label='Mean SSD')
    if add_ssd2 and mean_ssd_all2 is not None:
        avg_ssd2 = np.nanmean(mean_ssd_all2)*1000
        ax.axvline(x=avg_ssd2, color='k', linestyle='--', label='Mean SSD')
    fig.savefig(os.path.join(saving_path, f"{group}_{contrast}_{roi}_nSub{str(sub_num)}.{save_as}"))
     






def plot_tfr_result_contrast_groups(
        groups,
        tfr_dict,
        epochs_cond,
        roi,
        tfr_args,
        tmin_tmax,
        vmin_vmax,
        baseline_correction,
        baseline_correction_method,
        sub_num,
        saving_path,
        save_as = 'png'
        ):
    
    add_rt = False
    add_ssd = False
    dependent_measures = False  # by default, measures are treatet as independent

    contrast = groups[0] + ' - ' + groups[1]

    title_prefix = "Power change" if baseline_correction else "Power"
    bc_note = "" if baseline_correction else ", no baseline correction"
    subject_info = f"nSub = {sub_num}" #if sub_num > 1 else sub
    cond_type, outcome = epochs_cond.split('_')[:2]
    # cond_type2, outcome2 = epochs_cond[1].split('_')[:2]

    # Set the x label based on what the epochs are centered on:
    aligned_on = epochs_cond.split('_')[-1] # both conditions should always be centered on the same cue

    if aligned_on == 'response':
        xlabel = 'Time from RESPONSE (s)'
    elif aligned_on == 'feedback':
        xlabel = 'Time from FEEDBACK (s)'
    elif aligned_on == 'square':
        xlabel = 'Time from GO cue (s)'
        add_rt = True if (
        (cond_type in ('GO', 'GF', 'GC') and outcome == 'successful') or
        (cond_type == 'GS' and outcome == 'unsuccessful')
        ) else False
        add_ssd = True if any(x in epochs_cond for x in ('GS', 'GC')) else False
    elif aligned_on == 'triangle':
        xlabel = 'Time from STOP/CONTINUE cue (s)'
        add_rt = True if (cond_type == 'GS' and outcome == 'unsuccessful') or (cond_type == 'GC' and outcome == 'successful') else False

    # check that time dimensions are the same for both groups, if not crop to the minimum length
    shape0 = tfr_dict[groups[0]][epochs_cond]['all_percentage_change'].shape
    shape1 = tfr_dict[groups[1]][epochs_cond]['all_percentage_change'].shape
    if shape0[2] != shape1[2]:
        print(f"The two groups have different time dimensions: {shape0[2]} vs {shape1[2]}. Cropping to the minimum length for contrast.")
    min_len = min(shape0[2], shape1[2])
    times = tfr_dict[groups[0]][epochs_cond]['times']
    freqs = tfr_dict[groups[0]][epochs_cond]['freqs']
    tfr_dict[groups[0]][epochs_cond]['all_percentage_change'] = tfr_dict[groups[0]][epochs_cond]['all_percentage_change'][:, :, :min_len]
    tfr_dict[groups[1]][epochs_cond]['all_percentage_change'] = tfr_dict[groups[1]][epochs_cond]['all_percentage_change'][:, :, :min_len]
    times = times[:min_len]

    if 'DBS' in groups[0] and 'DBS' in groups[1]:
        # if both groups are DBS groups then measures are dependent because they come from the same subjects
        dependent_measures = True

        # tfr_dict[coi]['times'] = times
        contrast_percentage_change = tfr_dict[groups[0]][epochs_cond]['all_percentage_change'] - tfr_dict[groups[1]][epochs_cond]['all_percentage_change']
        tfr_dict[epochs_cond] = {}
        tfr_dict[epochs_cond]['contrast_groups'] = {
            'all_percentage_change': contrast_percentage_change,
            'times': times,
            'freqs': freqs}
        all_percentage_change = tfr_dict[epochs_cond]['contrast_groups']['all_percentage_change']

    else:
        # independent measures, at least one group is controls
        all_percentage_change = [tfr_dict[groups[0]][epochs_cond]['all_percentage_change'], 
                                      tfr_dict[groups[1]][epochs_cond]['all_percentage_change']]      

    mean_rt_all1 = tfr_dict[groups[0]][epochs_cond]['mean_rt_all']
    mean_rt_all2 = tfr_dict[groups[1]][epochs_cond]['mean_rt_all']
    mean_ssd_all1 = tfr_dict[groups[0]][epochs_cond]['mean_ssd_all']
    mean_ssd_all2 = tfr_dict[groups[1]][epochs_cond]['mean_ssd_all']
    # n_obs = all_percentage_change.shape[0]
    # # print(n_obs)
    pval = 0.05
    # df = n_obs - 1
    # threshold = scipy.stats.t.ppf(1-pval / 2, df) # two-tailed distribution
    threshold = None
    n_permutations = 1000

    # Compute permutation cluster test 
    if dependent_measures:
        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        all_percentage_change, n_permutations=n_permutations,
        threshold=threshold, tail=0,
        out_type= "mask", seed=11111, verbose=True)

    else:
        # F_observed, clusters, cluster_p_values, H0 for independent samples
        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
        all_percentage_change, n_permutations=n_permutations,
        threshold=threshold, tail=0,
        out_type= "mask", seed=11111, verbose=True)
    print(f"p_values: {cluster_p_values}")
    print(f"P_values shape: {cluster_p_values.shape}")
    
    condition = epochs_cond + '_' + contrast
    identify_significant_clusters(
        cluster_p_values, 
        clusters,
        tfr_dict[epochs_cond]['contrast_groups']['times'],
        T_obs,
        pval,
        tfr_args,
        condition = condition,
        roi = roi,
        saving_path = saving_path
        )

    # Compute grand averages
    avg_percentage_change = np.nanmean(all_percentage_change, axis=0)     
    assert T_obs.shape == avg_percentage_change.shape  

    # Compute min and max along the frequency axis
    if not baseline_correction:
        min_values = np.min(avg_percentage_change)  # Shape: (n_subjects, n_times)
        max_values = np.max(avg_percentage_change)
        
    # Plot
    # plt.figure(figsize=(4, 6))
    fig, ax = plt.subplots(figsize=(4, 6), constrained_layout=True)
    vmin, vmax = (vmin_vmax if baseline_correction else (min_values, max_values))
    times = tfr_dict[epochs_cond]['contrast_groups']['times'] 
    print(f'vmin: {vmin}, vmax: {vmax}')
    ax.imshow(avg_percentage_change, aspect='auto', origin='lower', 
                            # extent=[tmin_tmax[0]*1000, tmin_tmax[1]*1000, tfr_args["freqs"][0], tfr_args["freqs"][-1]], 
                            extent = [times[0], times[-1], freqs[0], freqs[-1]],
                            cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency (Hz)')
    title_text = f"{title_prefix} \n {epochs_cond} \n {contrast} \n {roi} \n {subject_info}{bc_note}"
    fig.suptitle(title_text, fontsize=10)
    # ax.set_title(f"{title_prefix} \n {contrast} \n {roi} \n {subject_info}{bc_note}")
        
    if baseline_correction:
        if baseline_correction_method == 'single_trial':
            colorbar_label = 'Change from baseline (dB)'
        else:
            colorbar_label = 'Percent change from baseline (%)'
    else:
        colorbar_label = 'Mean Power (µV²)'
    fig.colorbar(ax.images[0], label=colorbar_label)   
    
    # add significant clusters on the plot if group-level analysis:
    if sub_num > 1:
        for c, p_val in zip(clusters, cluster_p_values):
            if p_val <= pval:
                mask = np.zeros_like(T_obs, dtype=bool)  # Explicitly match dimensions
                mask[c] = True
                ax.contour(mask, levels=[0.5], colors='black', linewidths=1.5,
                            extent=[tmin_tmax[0]*1000, tmin_tmax[1]*1000, tfr_args["freqs"][0], tfr_args["freqs"][-1]])

    if add_rt:
        avg_rt1 = np.nanmean(mean_rt_all1)*1000
        avg_rt2 = np.nanmean(mean_rt_all2)*1000
        if aligned_on == 'triangle':
            avg_rt1 -= np.nanmean(mean_ssd_all1)*1000
            avg_rt2 -= np.nanmean(mean_ssd_all2)*1000
        ax.axvline(x=avg_rt1, color='black', linestyle='-', label=f'Mean RT {epochs_cond} {groups[0]}')
        ax.axvline(x=avg_rt2, color='gray', linestyle='-', label=f'Mean RT {epochs_cond} {groups[1]}')

    if add_ssd :
        if mean_ssd_all1 is not None:
            avg_ssd1 = np.nanmean(mean_ssd_all1)*1000
            ax.axvline(x=avg_ssd1, color='k', linestyle='--', label=f'Mean SSD {groups[0]}')
        if mean_ssd_all2 is not None:
            avg_ssd2 = np.nanmean(mean_ssd_all2)*1000
            ax.axvline(x=avg_ssd2, color='k', linestyle='--', label=f'Mean SSD {groups[1]}')
    
    fig.savefig(os.path.join(saving_path, f"{epochs_cond}_{contrast}_{roi}_nSub{str(sub_num)}.{save_as}"))



def stn_erp_change_diff_cond(
        sub_dict_epochs,
        dbs_status,
        epoch_cond1,
        epoch_cond2,
        condition,
        condition_color_dict,
        saving_path,
        alpha=0.05,
        n_permutations=1000,
        save_as='png'
):
    """
    Compare two conditions in STN intracranial EEG using a
    paired/repeated-measures cluster permutation test.

    Statistics
    ----------
    For each channel:
        difference = condition 1 - condition 2

    A one-sample cluster permutation test is then performed
    on these within-subject differences against zero.

    Reaction times
    --------------
    Mean RT is calculated separately for each condition within
    each subject, using the metadata field:

        'key_resp_experiment.rt'

    The reported group RT is the mean of the subject-level
    mean RTs, so subjects are weighted equally regardless of
    their number of trials.

    Parameters
    ----------
    sub_dict_epochs : dict
        Dictionary with subject IDs as keys and MNE Epochs as values.

    dbs_status : str
        String used to select the relevant subjects.

    epoch_cond1 : str
        First condition, e.g. 'GO_successful'.

    epoch_cond2 : str
        Second condition, e.g. 'GF_successful'.

    condition : str
        Label used for the figure title, e.g. 'DBS ON'.

    condition_color_dict : dict
        Dictionary mapping condition names to plotting colors.

    alpha : float
        Significance threshold.

    n_permutations : int
        Number of permutations.

    Returns
    -------
    avg_cond1 : mne.Evoked
        Grand-average ERP for condition 1.

    avg_cond2 : mne.Evoked
        Grand-average ERP for condition 2.

    cluster_results : list
        List containing (clusters, p_values) for each channel.

    subjects_included : list
        Subjects included in the analysis.

    mean_rt_cond1 : float
        Mean subject-level RT for condition 1, in seconds.

    mean_rt_cond2 : float
        Mean subject-level RT for condition 2, in seconds.

    rt_by_subject : dict
        Subject-level mean RTs for both conditions.
    """

    # =========================================================
    # PARAMETERS
    # =========================================================

    tmin, tmax = -0.5, 1.5
    sfreq = 250

    common_times = np.arange(
        tmin,
        tmax + 1 / sfreq,
        1 / sfreq
    )

    all_cond1 = []
    all_cond2 = []
    subs_included = []

    # Store subject-level RTs
    rt_cond1_subjects = []
    rt_cond2_subjects = []

    rt_by_subject = {}

    latency_matched1 = False
    latency_matched2 = False

    # =========================================================
    # COLLECT DATA
    # =========================================================

    for subject, epochs in sub_dict_epochs.items():

        if dbs_status not in subject:
            continue

        # -----------------------------------------------------
        # Parse condition 1
        # -----------------------------------------------------

        epoch_type1, outcome_str1 = epoch_cond1.split('_')
        if epoch_type1 == 'lmGO':
            latency_matched1 = True
            epoch_type1 = 'GO'

        outcome1 = (
            1.0 if outcome_str1 == "successful"
            else 0.0
        )

        # -----------------------------------------------------
        # Parse condition 2
        # -----------------------------------------------------

        epoch_type2, outcome_str2 = epoch_cond2.split('_')
        if epoch_type2 == 'lmGO':
            latency_matched2 = True
            epoch_type2 = 'GO'

        outcome2 = (
            1.0 if outcome_str2 == "successful"
            else 0.0
        )

        # -----------------------------------------------------
        # Select condition 1 trials
        # -----------------------------------------------------

        type_mask1 = (
            epochs.metadata["event"] == epoch_type1
        )

        outcome_mask1 = (
            epochs.metadata["key_resp_experiment.corr"]
            == outcome1
        )

        data1 = epochs[
            type_mask1 & outcome_mask1
        ]

        if latency_matched1:
            rt = np.asarray(data1.metadata['key_resp_experiment.rt'])
            threshold = np.percentile(rt, 50)
            slow_mask = rt >= threshold
            data1 = data1[slow_mask] 

        # -----------------------------------------------------
        # Select condition 2 trials
        # -----------------------------------------------------

        type_mask2 = (
            epochs.metadata["event"] == epoch_type2
        )

        outcome_mask2 = (
            epochs.metadata["key_resp_experiment.corr"]
            == outcome2
        )

        data2 = epochs[
            type_mask2 & outcome_mask2
        ]

        if latency_matched2:
            rt = np.asarray(data2.metadata['key_resp_experiment.rt'])
            threshold = np.percentile(rt, 50)
            slow_mask = rt >= threshold
            data2 = data2[slow_mask] 

        # -----------------------------------------------------
        # Make sure both conditions exist
        # -----------------------------------------------------

        if len(data1) == 0 or len(data2) == 0:

            print(
                f"Skipping {subject}: "
                f"{epoch_cond1} = {len(data1)} trials, "
                f"{epoch_cond2} = {len(data2)} trials"
            )

            continue

        # -----------------------------------------------------
        # Make sure channel structure is identical
        # -----------------------------------------------------

        if data1.ch_names != data2.ch_names:

            raise RuntimeError(
                f"Channel mismatch for {subject}.\n"
                f"{epoch_cond1}: {data1.ch_names}\n"
                f"{epoch_cond2}: {data2.ch_names}"
            )

        # =====================================================
        # REACTION TIMES
        # =====================================================

        # Extract RTs from metadata
        rt1 = data1.metadata[
            "key_resp_experiment.rt"
        ].to_numpy()

        rt2 = data2.metadata[
            "key_resp_experiment.rt"
        ].to_numpy()

        # Remove missing RTs (NaN)
        rt1 = rt1[~np.isnan(rt1)]
        rt2 = rt2[~np.isnan(rt2)]

        # Subject-level mean RT
        mean_rt1_subject = (
            np.mean(rt1)
            if len(rt1) > 0
            else np.nan
        )

        mean_rt2_subject = (
            np.mean(rt2)
            if len(rt2) > 0
            else np.nan
        )

        rt_cond1_subjects.append(
            mean_rt1_subject
        )

        rt_cond2_subjects.append(
            mean_rt2_subject
        )

        # Store subject-level RTs
        rt_by_subject[subject] = {
            epoch_cond1: mean_rt1_subject,
            epoch_cond2: mean_rt2_subject
        }

        # =====================================================
        # CROP
        # =====================================================

        cropped_data1 = data1.copy().crop(
            tmin=tmin,
            tmax=tmax
        )

        cropped_data2 = data2.copy().crop(
            tmin=tmin,
            tmax=tmax
        )

        # =====================================================
        # BASELINE CORRECTION
        # =====================================================

        crunched_data1 = (
            cropped_data1
            .copy()
            .apply_baseline((-0.5, 0))
        )

        crunched_data2 = (
            cropped_data2
            .copy()
            .apply_baseline((-0.5, 0))
        )


        # =====================================================
        # AVERAGE WITHIN SUBJECT
        # =====================================================

        averaged_data1 = crunched_data1.average()
        averaged_data2 = crunched_data2.average()

        # print("Single-trial range:")
        # print(
        #     np.min(crunched_data1.get_data()),
        #     np.max(crunched_data1.get_data())
        # )

        # print("\nERP range:")
        # print(
        #     np.min(averaged_data1.data),
        #     np.max(averaged_data1.data)
        # )
        # =====================================================
        # INTERPOLATE TO COMMON TIME GRID
        # =====================================================

        new_data1 = np.vstack([
            np.interp(
                common_times,
                averaged_data1.times,
                ch
            )
            for ch in averaged_data1.data
        ])

        new_data2 = np.vstack([
            np.interp(
                common_times,
                averaged_data2.times,
                ch
            )
            for ch in averaged_data2.data
        ])

        # =====================================================
        # CREATE EVOKED OBJECTS
        # =====================================================

        evoked_interp1 = mne.EvokedArray(
            new_data1,
            averaged_data1.info.copy(),
            tmin=common_times[0]
        )

        evoked_interp2 = mne.EvokedArray(
            new_data2,
            averaged_data2.info.copy(),
            tmin=common_times[0]
        )

        # =====================================================
        # STORE
        # =====================================================

        all_cond1.append(evoked_interp1)
        all_cond2.append(evoked_interp2)

        subs_included.append(subject)


    # =========================================================
    # CHECK SUBJECTS
    # =========================================================

    if len(all_cond1) == 0:

        raise RuntimeError(
            "No subjects were included. "
            "Check dbs_status and condition names."
        )

    print(
        f"\nNumber of subjects included: "
        f"{len(subs_included)}"
    )

    print(
        "Subjects:",
        subs_included
    )


    # =========================================================
    # GROUP-LEVEL REACTION TIMES
    # =========================================================

    mean_rt_cond1 = np.nanmean(
        rt_cond1_subjects
    )

    mean_rt_cond2 = np.nanmean(
        rt_cond2_subjects
    )

    # Convert to milliseconds for display
    mean_rt_cond1_ms = (
        mean_rt_cond1 * 1000
    )

    mean_rt_cond2_ms = (
        mean_rt_cond2 * 1000
    )

    print("\nMean reaction times:")
    print(
        f"{epoch_cond1}: "
        f"{mean_rt_cond1_ms:.1f} ms"
    )

    print(
        f"{epoch_cond2}: "
        f"{mean_rt_cond2_ms:.1f} ms"
    )


    # =========================================================
    # CONVERT ERP DATA TO ARRAYS
    # =========================================================

    X1 = np.array([
        evk.data
        for evk in all_cond1
    ])

    X2 = np.array([
        evk.data
        for evk in all_cond2
    ])

    # Shape:
    #
    # subjects × channels × time

    print("\nData shape:")
    print("Condition 1:", X1.shape)
    print("Condition 2:", X2.shape)

    n_subjects = X1.shape[0]
    n_channels = X1.shape[1]

    times = common_times


    # =========================================================
    # GRAND AVERAGES
    # =========================================================

    avg_cond1 = mne.grand_average(
        all_cond1
    )

    avg_cond2 = mne.grand_average(
        all_cond2
    )


    # =========================================================
    # PAIRED / REPEATED-MEASURES CLUSTER TEST
    # =========================================================

    cluster_results = []

    for ch in range(n_channels):

        # -----------------------------------------------------
        # Within-subject difference
        #
        # condition 1 - condition 2
        # -----------------------------------------------------

        difference = (
            X1[:, ch, :]
            -
            X2[:, ch, :]
        )

        # -----------------------------------------------------
        # One-sample cluster permutation test
        # -----------------------------------------------------

        T_obs, clusters, p_values, H0 = (
            permutation_cluster_1samp_test(
                difference,
                n_permutations=n_permutations,
                threshold=None,
                tail=0,
                out_type='indices',
                seed=42
            )
        )

        cluster_results.append(
            (
                clusters,
                p_values
            )
        )


    # =========================================================
    # PLOT INTRACRANIAL ERPs
    # =========================================================

    ch_names = avg_cond1.ch_names

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(
            10,
            max(3, 2.5 * n_channels)
        ),
        sharex=True,
        squeeze=False
    )

    axes = axes[:, 0]

    color1 = condition_color_dict[
        epoch_cond1
    ]

    color2 = condition_color_dict[
        epoch_cond2
    ]


    # =========================================================
    # PLOT EACH CHANNEL
    # =========================================================

    for ch_idx, ax in enumerate(axes):

        # -----------------------------------------------------
        # ERP condition 1
        # -----------------------------------------------------

        ax.plot(
            times,
            avg_cond1.data[ch_idx],
            color=color1,
            linewidth=1.5,
            label=(
                f"{epoch_cond1} "
                f"(RT = {mean_rt_cond1_ms:.0f} ms)"
            )
        )
        ax.axvline(mean_rt_cond1, color=color1, linestyle=':', linewidth=1.0)

        # -----------------------------------------------------
        # ERP condition 2
        # -----------------------------------------------------

        ax.plot(
            times,
            avg_cond2.data[ch_idx],
            color=color2,
            linewidth=1.5,
            label=(
                f"{epoch_cond2} "
                f"(RT = {mean_rt_cond2_ms:.0f} ms)"
            )
        )
        ax.axvline(mean_rt_cond2, color=color2, linestyle=':', linewidth=1.0)

        # =====================================================
        # SIGNIFICANT CLUSTERS
        # =====================================================

        clusters, p_values = cluster_results[ch_idx]

        for cluster, p_val in zip(
            clusters,
            p_values
        ):

            if p_val < alpha:

                # cluster is a tuple containing
                # the indices along the tested dimension

                cluster_times = (
                    times[cluster[0]]
                )

                print(
                    f"Significant cluster | "
                    f"{ch_names[ch_idx]} | "
                    f"p = {p_val:.4f} | "
                    f"{cluster_times[0]:.3f}–"
                    f"{cluster_times[-1]:.3f} s"
                )

                # -------------------------------------------------
                # Shade significant time period
                # -------------------------------------------------

                ax.axvspan(
                    cluster_times[0],
                    cluster_times[-1],
                    color='red',
                    alpha=0.20
                )


        # =====================================================
        # FORMATTING
        # =====================================================

        ax.axvline(
            0,
            color='black',
            linestyle='--',
            linewidth=0.8
        )

        ax.axhline(
            0,
            color='black',
            linewidth=0.5
        )

        ax.set_ylabel(
            ch_names[ch_idx]
        )

        ax.grid(
            alpha=0.2
        )


    # =========================================================
    # FIGURE LABELS
    # =========================================================

    axes[-1].set_xlabel(
        "Time (s)"
    )

    axes[0].legend(
        loc='upper right'
    )

    fig.suptitle(
        f"{condition} — STN intracranial ERP",
        fontsize=14
    )

    fig.tight_layout()

    plt.savefig(
        os.path.join(
            saving_path,
            f"{condition}_STN_ERP_{epoch_cond1}_vs_{epoch_cond2}.{save_as}"
        ),
        dpi=300
    )
    plt.show()


    # =========================================================
    # RETURN
    # =========================================================

    return (
        avg_cond1,
        avg_cond2,
        cluster_results,
        subs_included,
        # mean_rt_cond1,
        # mean_rt_cond2,
        # rt_by_subject
    )



def stn_erp_change_diff_on_off(
        sub_dict_epochs,
        epoch_cond,
        condition_color_dict,
        saving_path,
        alpha=0.05,
        n_permutations=1000,
        save_as='png'
):
    """
    Compare the same trial type between DBS ON and DBS OFF
    using a paired/repeated-measures cluster permutation test.

    Example
    -------
    epoch_cond = 'GO_successful'

    The analysis then compares:

        GO_successful DBS ON
        vs.
        GO_successful DBS OFF

    Statistics
    ----------
    For each channel:

        difference = DBS ON - DBS OFF

    A one-sample cluster permutation test is then performed
    on these within-subject differences against zero.

    Reaction times
    --------------
    Mean RT is calculated separately for DBS ON and DBS OFF
    within each subject, using:

        'key_resp_experiment.rt'

    The reported group RT is the mean of the subject-level
    mean RTs, so subjects are weighted equally.

    Parameters
    ----------
    sub_dict_epochs : dict
        Dictionary with subject/session IDs as keys and MNE
        Epochs as values.

        Keys must contain either 'DBS ON' or 'DBS OFF'.

        Example:
            'sub01 DBS ON'
            'sub01 DBS OFF'

    epoch_cond : str
        Trial type to analyse, e.g. 'GO_successful'.

    condition_color_dict : dict
        Dictionary mapping 'DBS ON' and 'DBS OFF' to plotting colors.

        Example:
            {
                'DBS ON': 'red',
                'DBS OFF': 'blue'
            }

    saving_path : str
        Directory where the figure is saved.

    alpha : float
        Significance threshold.

    n_permutations : int
        Number of permutations.

    save_as : str
        Figure format, e.g. 'png' or 'pdf'.

    Returns
    -------
    avg_on : mne.Evoked
        Grand-average ERP for DBS ON.

    avg_off : mne.Evoked
        Grand-average ERP for DBS OFF.

    cluster_results : list
        List containing (clusters, p_values) for each channel.

    subjects_included : list
        Subjects included in the paired analysis.

    mean_rt_on : float
        Mean subject-level RT for DBS ON, in seconds.

    mean_rt_off : float
        Mean subject-level RT for DBS OFF, in seconds.

    rt_by_subject : dict
        Subject-level mean RTs for DBS ON and DBS OFF.
    """

    # =========================================================
    # PARAMETERS
    # =========================================================

    tmin, tmax = -0.5, 1.5
    sfreq = 250

    common_times = np.arange(
        tmin,
        tmax + 1 / sfreq,
        1 / sfreq
    )

    # Store one ERP per subject/session
    on_by_subject = {}
    off_by_subject = {}

    # Store subject-level RTs
    rt_on_by_subject = {}
    rt_off_by_subject = {}

    latency_matched = False

    # =========================================================
    # PARSE TRIAL CONDITION
    # =========================================================

    epoch_type, outcome_str = epoch_cond.split('_')

    if epoch_type == 'lmGO':
        latency_matched = True
        epoch_type = 'GO'

    outcome = (
        1.0 if outcome_str == 'successful'
        else 0.0
    )

    # =========================================================
    # COLLECT DATA
    # =========================================================

    for subject_session, epochs in sub_dict_epochs.items():

        # -----------------------------------------------------
        # Determine DBS status
        # -----------------------------------------------------

        if 'DBS ON' in subject_session:
            dbs_status = 'DBS ON'

        elif 'DBS OFF' in subject_session:
            dbs_status = 'DBS OFF'

        else:
            continue

        # -----------------------------------------------------
        # Determine subject ID
        #
        # Removes the DBS status from the dictionary key.
        #
        # Example:
        #   'sub01 DBS ON'  -> 'sub01'
        #   'sub01 DBS OFF' -> 'sub01'
        # -----------------------------------------------------

        subject = (
            subject_session
            .replace('DBS ON', '')
            .replace('DBS OFF', '')
            .strip()
        )

        # -----------------------------------------------------
        # Select desired trial type
        # -----------------------------------------------------

        type_mask = (
            epochs.metadata["event"] == epoch_type
        )

        outcome_mask = (
            epochs.metadata["key_resp_experiment.corr"]
            == outcome
        )

        data = epochs[
            type_mask & outcome_mask
        ]

        # -----------------------------------------------------
        # Optional latency matching
        # -----------------------------------------------------

        if latency_matched:

            if len(data) == 0:
                continue

            rt = np.asarray(
                data.metadata["key_resp_experiment.rt"]
            )

            threshold = np.percentile(rt, 50)

            slow_mask = rt >= threshold

            data = data[slow_mask]

        # -----------------------------------------------------
        # Check that trials exist
        # -----------------------------------------------------

        if len(data) == 0:

            print(
                f"Skipping {subject_session}: "
                f"no {epoch_cond} trials"
            )

            continue

        # =====================================================
        # REACTION TIME
        # =====================================================

        rt = data.metadata[
            "key_resp_experiment.rt"
        ].to_numpy()

        # Remove NaNs
        rt = rt[~np.isnan(rt)]

        mean_rt_subject = (
            np.mean(rt)
            if len(rt) > 0
            else np.nan
        )

        if dbs_status == 'DBS ON':
            rt_on_by_subject[subject] = mean_rt_subject

        else:
            rt_off_by_subject[subject] = mean_rt_subject

        # =====================================================
        # CROP
        # =====================================================

        cropped_data = data.copy().crop(
            tmin=tmin,
            tmax=tmax
        )

        # =====================================================
        # BASELINE CORRECTION
        # =====================================================

        crunched_data = (
            cropped_data
            .copy()
            .apply_baseline((-0.5, 0))
        )

        # =====================================================
        # AVERAGE WITHIN SUBJECT
        # =====================================================

        averaged_data = crunched_data.average()

        # =====================================================
        # INTERPOLATE TO COMMON TIME GRID
        # =====================================================

        new_data = np.vstack([
            np.interp(
                common_times,
                averaged_data.times,
                ch
            )
            for ch in averaged_data.data
        ])

        # =====================================================
        # CREATE EVOKED OBJECT
        # =====================================================

        evoked_interp = mne.EvokedArray(
            new_data,
            averaged_data.info.copy(),
            tmin=common_times[0]
        )

        # =====================================================
        # STORE BY SUBJECT AND DBS STATUS
        # =====================================================

        if dbs_status == 'DBS ON':
            on_by_subject[subject] = evoked_interp

        else:
            off_by_subject[subject] = evoked_interp

    # =========================================================
    # FIND SUBJECTS WITH BOTH DBS ON AND DBS OFF
    # =========================================================

    subjects_included = sorted(
        set(on_by_subject.keys())
        &
        set(off_by_subject.keys())
    )

    if len(subjects_included) == 0:

        raise RuntimeError(
            "No subjects have both DBS ON and DBS OFF data "
            f"for {epoch_cond}."
        )

    print(
        f"\nNumber of paired subjects included: "
        f"{len(subjects_included)}"
    )

    print(
        "Subjects:",
        subjects_included
    )

    # =========================================================
    # CHECK CHANNEL STRUCTURE
    # =========================================================

    for subject in subjects_included:

        if (
            on_by_subject[subject].ch_names
            != off_by_subject[subject].ch_names
        ):

            raise RuntimeError(
                f"Channel mismatch for {subject}.\n"
                f"DBS ON: "
                f"{on_by_subject[subject].ch_names}\n"
                f"DBS OFF: "
                f"{off_by_subject[subject].ch_names}"
            )

    # =========================================================
    # CREATE PAIRED ARRAYS
    # =========================================================

    all_cond_on = [
        on_by_subject[sub]
        for sub in subjects_included
    ]

    all_cond_off = [
        off_by_subject[sub]
        for sub in subjects_included
    ]

    # =========================================================
    # GROUP-LEVEL REACTION TIMES
    # =========================================================

    mean_rt_on = np.nanmean([
        rt_on_by_subject[sub]
        for sub in subjects_included
    ])

    mean_rt_off = np.nanmean([
        rt_off_by_subject[sub]
        for sub in subjects_included
    ])

    mean_rt_on_ms = mean_rt_on * 1000
    mean_rt_off_ms = mean_rt_off * 1000

    print("\nMean reaction times:")

    print(
        f"{epoch_cond} — DBS ON: "
        f"{mean_rt_on_ms:.1f} ms"
    )

    print(
        f"{epoch_cond} — DBS OFF: "
        f"{mean_rt_off_ms:.1f} ms"
    )

    # =========================================================
    # STORE RTs BY SUBJECT
    # =========================================================

    rt_by_subject = {}

    for subject in subjects_included:

        rt_by_subject[subject] = {
            'DBS ON': rt_on_by_subject[subject],
            'DBS OFF': rt_off_by_subject[subject]
        }

    # =========================================================
    # CONVERT ERP DATA TO ARRAYS
    # =========================================================

    X_on = np.array([
        evk.data
        for evk in all_cond_on
    ])

    X_off = np.array([
        evk.data
        for evk in all_cond_off
    ])

    # Shape:
    # subjects × channels × time

    print("\nData shape:")
    print("DBS ON:", X_on.shape)
    print("DBS OFF:", X_off.shape)

    n_subjects = X_on.shape[0]
    n_channels = X_on.shape[1]

    times = common_times

    # =========================================================
    # GRAND AVERAGES
    # =========================================================

    avg_on = mne.grand_average(
        all_cond_on
    )

    avg_off = mne.grand_average(
        all_cond_off
    )

    # =========================================================
    # PAIRED / REPEATED-MEASURES CLUSTER TEST
    # =========================================================

    cluster_results = []

    for ch in range(n_channels):

        # -----------------------------------------------------
        # Within-subject difference
        #
        # DBS ON - DBS OFF
        # -----------------------------------------------------

        difference = (
            X_on[:, ch, :]
            -
            X_off[:, ch, :]
        )

        # -----------------------------------------------------
        # One-sample cluster permutation test
        # -----------------------------------------------------

        T_obs, clusters, p_values, H0 = (
            permutation_cluster_1samp_test(
                difference,
                n_permutations=n_permutations,
                threshold=None,
                tail=0,
                out_type='indices',
                seed=42
            )
        )

        cluster_results.append(
            (
                clusters,
                p_values
            )
        )

    # =========================================================
    # PLOT INTRACRANIAL ERPs
    # =========================================================

    ch_names = avg_on.ch_names

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(
            10,
            max(3, 2.5 * n_channels)
        ),
        sharex=True,
        squeeze=False
    )

    axes = axes[:, 0]

    color_on = condition_color_dict['DBS ON']
    color_off = condition_color_dict['DBS OFF']

    # =========================================================
    # PLOT EACH CHANNEL
    # =========================================================

    for ch_idx, ax in enumerate(axes):

        # -----------------------------------------------------
        # DBS ON
        # -----------------------------------------------------

        ax.plot(
            times,
            avg_on.data[ch_idx],
            color=color_on,
            linewidth=1.5,
            label=(
                f"{epoch_cond} — DBS ON "
                f"(RT = {mean_rt_on_ms:.0f} ms)"
            )
        )

        ax.axvline(
            mean_rt_on,
            color=color_on,
            linestyle=':',
            linewidth=1.0
        )

        # -----------------------------------------------------
        # DBS OFF
        # -----------------------------------------------------

        ax.plot(
            times,
            avg_off.data[ch_idx],
            color=color_off,
            linewidth=1.5,
            label=(
                f"{epoch_cond} — DBS OFF "
                f"(RT = {mean_rt_off_ms:.0f} ms)"
            )
        )

        ax.axvline(
            mean_rt_off,
            color=color_off,
            linestyle=':',
            linewidth=1.0
        )

        # =====================================================
        # SIGNIFICANT CLUSTERS
        # =====================================================

        clusters, p_values = cluster_results[ch_idx]

        for cluster, p_val in zip(
            clusters,
            p_values
        ):

            if p_val < alpha:

                # Cluster is a tuple containing
                # the indices along the tested dimension

                cluster_times = (
                    times[cluster[0]]
                )

                print(
                    f"Significant cluster | "
                    f"{ch_names[ch_idx]} | "
                    f"p = {p_val:.4f} | "
                    f"{cluster_times[0]:.3f}–"
                    f"{cluster_times[-1]:.3f} s"
                )

                # -------------------------------------------------
                # Shade significant time period
                # -------------------------------------------------

                ax.axvspan(
                    cluster_times[0],
                    cluster_times[-1],
                    color='red',
                    alpha=0.20
                )

        # =====================================================
        # FORMATTING
        # =====================================================

        ax.axvline(
            0,
            color='black',
            linestyle='--',
            linewidth=0.8
        )

        ax.axhline(
            0,
            color='black',
            linewidth=0.5
        )

        ax.set_ylabel(
            ch_names[ch_idx]
        )

        ax.grid(
            alpha=0.2
        )

    # =========================================================
    # FIGURE LABELS
    # =========================================================

    axes[-1].set_xlabel(
        "Time (s)"
    )

    axes[0].legend(
        loc='upper right'
    )

    fig.suptitle(
        f"{epoch_cond} — DBS ON vs DBS OFF — "
        f"STN intracranial ERP",
        fontsize=14
    )

    fig.tight_layout()

    # =========================================================
    # SAVE
    # =========================================================

    plt.savefig(
        os.path.join(
            saving_path,
            f"DBS_ON_vs_OFF_STN_ERP_"
            f"{epoch_cond}.{save_as}"
        ),
        dpi=300
    )

    plt.show()

    # =========================================================
    # RETURN
    # =========================================================

    return (
        avg_on,
        avg_off,
        cluster_results,
        subjects_included,
        mean_rt_on,
        mean_rt_off,
        rt_by_subject
    )



def eeg_erp_change_diff_cond(
        sub_dict_epochs,
        epoch_cond1,
        epoch_cond2,
        comparison,
        condition,
        ROI,
        condition_color_dict,
        saving_path,
        alpha=0.05,
        n_permutations=1000,
        save_as='png'
):
    """
    Compare two conditions in STN intracranial EEG using a
    paired/repeated-measures cluster permutation test.

    ROI handling
    ------------
    ROI should be a list of channel names, e.g.

        ROI = ['O1', 'O2', 'Oz']

    If only one requested channel is present, that channel is used.

    If multiple requested channels are present, their ERP waveforms
    are averaged to create a single ROI waveform.

    Channels listed in ROI but absent from a subject are ignored.

    Statistics
    ----------
    A one-sample cluster permutation test is performed on the
    within-subject condition difference of the ROI waveform.

    Returns
    -------
    avg_cond1 : mne.Evoked
        Grand-average ERP for condition 1 in the ROI.

    avg_cond2 : mne.Evoked
        Grand-average ERP for condition 2 in the ROI.

    cluster_results : list
        List containing (clusters, p_values) for the ROI.

    subjects_included : list
        Subjects included in the analysis.
    """

    # =========================================================
    # PARAMETERS
    # =========================================================

    tmin, tmax = -0.5, 1.5
    sfreq = 250

    common_times = np.arange(
        tmin,
        tmax + 1 / sfreq,
        1 / sfreq
    )

    all_cond1 = []
    all_cond2 = []
    subs_included = []

    # Store subject-level RTs
    rt_cond1_subjects = []
    rt_cond2_subjects = []

    rt_by_subject = {}

    latency_matched1 = False
    latency_matched2 = False

    # =========================================================
    # CHECK ROI
    # =========================================================

    if isinstance(ROI, str):
        ROI = [ROI]

    if ROI is None or len(ROI) == 0:
        raise ValueError("ROI must contain at least one channel name.")

    print(f"\nRequested ROI channels: {ROI}")

    # =========================================================
    # COLLECT DATA
    # =========================================================

    for subject, epochs in sub_dict_epochs.items():

        # if condition not in subject:
        #     continue
        
        # only keep subjects of interest:
        if condition == 'control':
            if 'DBS' in subject:
                continue
        else:
            if condition not in subject:
                continue

        # -----------------------------------------------------
        # Parse condition 1
        # -----------------------------------------------------

        epoch_type1, outcome_str1 = epoch_cond1.split('_')

        if epoch_type1 == 'lmGO':
            latency_matched1 = True
            epoch_type1 = 'GO'

        outcome1 = (
            1.0 if outcome_str1 == "successful"
            else 0.0
        )

        # -----------------------------------------------------
        # Parse condition 2
        # -----------------------------------------------------

        epoch_type2, outcome_str2 = epoch_cond2.split('_')

        if epoch_type2 == 'lmGO':
            latency_matched2 = True
            epoch_type2 = 'GO'

        outcome2 = (
            1.0 if outcome_str2 == "successful"
            else 0.0
        )

        # -----------------------------------------------------
        # Select condition 1 trials
        # -----------------------------------------------------

        type_mask1 = (
            epochs.metadata["event"] == epoch_type1
        )

        outcome_mask1 = (
            epochs.metadata["key_resp_experiment.corr"]
            == outcome1
        )

        data1 = epochs[
            type_mask1 & outcome_mask1
        ]

        if latency_matched1:
            rt = np.asarray(
                data1.metadata['key_resp_experiment.rt']
            )
            threshold = np.percentile(rt, 50)
            slow_mask = rt >= threshold
            data1 = data1[slow_mask]

        # -----------------------------------------------------
        # Select condition 2 trials
        # -----------------------------------------------------

        type_mask2 = (
            epochs.metadata["event"] == epoch_type2
        )

        outcome_mask2 = (
            epochs.metadata["key_resp_experiment.corr"]
            == outcome2
        )

        data2 = epochs[
            type_mask2 & outcome_mask2
        ]

        if latency_matched2:
            rt = np.asarray(
                data2.metadata['key_resp_experiment.rt']
            )
            threshold = np.percentile(rt, 50)
            slow_mask = rt >= threshold
            data2 = data2[slow_mask]

        # -----------------------------------------------------
        # Make sure both conditions exist
        # -----------------------------------------------------

        if len(data1) == 0 or len(data2) == 0:

            print(
                f"Skipping {subject}: "
                f"{epoch_cond1} = {len(data1)} trials, "
                f"{epoch_cond2} = {len(data2)} trials"
            )

            continue

        # -----------------------------------------------------
        # Make sure channel structure is identical
        # -----------------------------------------------------

        if data1.ch_names != data2.ch_names:

            raise RuntimeError(
                f"Channel mismatch for {subject}.\n"
                f"{epoch_cond1}: {data1.ch_names}\n"
                f"{epoch_cond2}: {data2.ch_names}"
            )

        # -----------------------------------------------------
        # Check which ROI channels are actually present
        # -----------------------------------------------------

        available_roi = [
            ch for ch in ROI
            if ch in data1.ch_names
        ]

        if len(available_roi) == 0:

            print(
                f"Skipping {subject}: none of the requested "
                f"ROI channels are present.\n"
                f"Requested ROI: {ROI}\n"
                f"Available channels: {data1.ch_names}"
            )

            continue

        print(
            f"{subject}: using ROI channels = {available_roi}"
        )

        # =====================================================
        # REACTION TIMES
        # =====================================================

        rt1 = data1.metadata[
            "key_resp_experiment.rt"
        ].to_numpy()

        rt2 = data2.metadata[
            "key_resp_experiment.rt"
        ].to_numpy()

        # Remove missing RTs
        rt1 = rt1[~np.isnan(rt1)]
        rt2 = rt2[~np.isnan(rt2)]

        # Subject-level mean RT
        mean_rt1_subject = (
            np.mean(rt1)
            if len(rt1) > 0
            else np.nan
        )

        mean_rt2_subject = (
            np.mean(rt2)
            if len(rt2) > 0
            else np.nan
        )

        rt_cond1_subjects.append(
            mean_rt1_subject
        )

        rt_cond2_subjects.append(
            mean_rt2_subject
        )

        rt_by_subject[subject] = {
            epoch_cond1: mean_rt1_subject,
            epoch_cond2: mean_rt2_subject
        }

        # =====================================================
        # CROP
        # =====================================================

        cropped_data1 = data1.copy().crop(
            tmin=tmin,
            tmax=tmax
        )

        cropped_data2 = data2.copy().crop(
            tmin=tmin,
            tmax=tmax
        )

        # =====================================================
        # BASELINE CORRECTION
        # =====================================================

        crunched_data1 = (
            cropped_data1
            .copy()
            .apply_baseline((-0.5, 0))
        )

        crunched_data2 = (
            cropped_data2
            .copy()
            .apply_baseline((-0.5, 0))
        )

        # =====================================================
        # AVERAGE WITHIN SUBJECT
        # =====================================================

        averaged_data1 = crunched_data1.average()
        averaged_data2 = crunched_data2.average()

        # =====================================================
        # SELECT ROI CHANNELS
        # =====================================================

        roi_indices = [
            averaged_data1.ch_names.index(ch)
            for ch in available_roi
        ]

        roi_data1 = averaged_data1.data[roi_indices, :]
        roi_data2 = averaged_data2.data[roi_indices, :]

        # -----------------------------------------------------
        # Average all available ROI channels
        #
        # Shape before:
        #     n_ROI_channels x time
        #
        # Shape after:
        #     1 x time
        # -----------------------------------------------------

        roi_average1 = np.mean(
            roi_data1,
            axis=0,
            keepdims=True
        )

        roi_average2 = np.mean(
            roi_data2,
            axis=0,
            keepdims=True
        )

        # =====================================================
        # INTERPOLATE TO COMMON TIME GRID
        # =====================================================

        new_data1 = np.vstack([
            np.interp(
                common_times,
                averaged_data1.times,
                roi_average1[0]
            )
        ])

        new_data2 = np.vstack([
            np.interp(
                common_times,
                averaged_data2.times,
                roi_average2[0]
            )
        ])

        # =====================================================
        # CREATE ROI EVOKED OBJECTS
        # =====================================================

        roi_info = mne.create_info(
            ch_names=['ROI'],
            sfreq=sfreq,
            ch_types='eeg'
        )

        evoked_interp1 = mne.EvokedArray(
            new_data1,
            roi_info,
            tmin=common_times[0]
        )

        evoked_interp2 = mne.EvokedArray(
            new_data2,
            roi_info,
            tmin=common_times[0]
        )

        # =====================================================
        # STORE
        # =====================================================

        all_cond1.append(evoked_interp1)
        all_cond2.append(evoked_interp2)

        subs_included.append(subject)

    # =========================================================
    # CHECK SUBJECTS
    # =========================================================

    if len(all_cond1) == 0:

        raise RuntimeError(
            "No subjects were included. "
            "Check condition, condition names, "
            "and whether ROI channels are present."
        )

    print(
        f"\nNumber of subjects included: "
        f"{len(subs_included)}"
    )

    print(
        "Subjects:",
        subs_included
    )

    # =========================================================
    # GROUP-LEVEL REACTION TIMES
    # =========================================================

    mean_rt_cond1 = np.nanmean(
        rt_cond1_subjects
    )

    mean_rt_cond2 = np.nanmean(
        rt_cond2_subjects
    )

    # Convert to milliseconds for display
    mean_rt_cond1_ms = (
        mean_rt_cond1 * 1000
    )

    mean_rt_cond2_ms = (
        mean_rt_cond2 * 1000
    )

    print("\nMean reaction times:")

    print(
        f"{epoch_cond1}: "
        f"{mean_rt_cond1_ms:.1f} ms"
    )

    print(
        f"{epoch_cond2}: "
        f"{mean_rt_cond2_ms:.1f} ms"
    )

    # =========================================================
    # CONVERT ERP DATA TO ARRAYS
    # =========================================================

    X1 = np.array([
        evk.data
        for evk in all_cond1
    ])

    X2 = np.array([
        evk.data
        for evk in all_cond2
    ])

    # Shape:
    # subjects x 1 ROI channel x time

    print("\nData shape:")
    print("Condition 1:", X1.shape)
    print("Condition 2:", X2.shape)

    n_subjects = X1.shape[0]
    n_channels = X1.shape[1]

    times = common_times

    # =========================================================
    # GRAND AVERAGES
    # =========================================================

    avg_cond1 = mne.grand_average(
        all_cond1
    )

    avg_cond2 = mne.grand_average(
        all_cond2
    )

    # Give the channel a useful name
    avg_cond1.rename_channels({
        'ROI': 'ROI'
    })

    avg_cond2.rename_channels({
        'ROI': 'ROI'
    })

    # =========================================================
    # PAIRED / REPEATED-MEASURES CLUSTER TEST
    # =========================================================

    cluster_results = []

    for ch in range(n_channels):

        # -----------------------------------------------------
        # Within-subject difference
        #
        # condition 1 - condition 2
        # -----------------------------------------------------

        difference = (
            X1[:, ch, :]
            -
            X2[:, ch, :]
        )

        # -----------------------------------------------------
        # One-sample cluster permutation test
        # -----------------------------------------------------

        T_obs, clusters, p_values, H0 = (
            permutation_cluster_1samp_test(
                difference,
                n_permutations=n_permutations,
                threshold=None,
                tail=0,
                out_type='indices',
                seed=42
            )
        )

        cluster_results.append(
            (
                clusters,
                p_values
            )
        )

    # =========================================================
    # PLOT ROI ERP
    # =========================================================

    ch_names = avg_cond1.ch_names

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 4)
    )

    color1 = condition_color_dict[
        epoch_cond1
    ]

    color2 = condition_color_dict[
        epoch_cond2
    ]

    # =========================================================
    # PLOT CONDITION 1
    # =========================================================

    ax.plot(
        times,
        avg_cond1.data[0],
        color=color1,
        linewidth=1.5,
        label=(
            f"{epoch_cond1} "
            f"(RT = {mean_rt_cond1_ms:.0f} ms)"
        )
    )

    ax.axvline(
        mean_rt_cond1,
        color=color1,
        linestyle=':',
        linewidth=1.0
    )

    # =========================================================
    # PLOT CONDITION 2
    # =========================================================

    ax.plot(
        times,
        avg_cond2.data[0],
        color=color2,
        linewidth=1.5,
        label=(
            f"{epoch_cond2} "
            f"(RT = {mean_rt_cond2_ms:.0f} ms)"
        )
    )

    ax.axvline(
        mean_rt_cond2,
        color=color2,
        linestyle=':',
        linewidth=1.0
    )

    # =========================================================
    # SIGNIFICANT CLUSTERS
    # =========================================================

    clusters, p_values = cluster_results[0]

    for cluster, p_val in zip(
        clusters,
        p_values
    ):

        if p_val < alpha:

            cluster_times = (
                times[cluster[0]]
            )

            print(
                f"Significant cluster | "
                f"ROI = {ROI} | "
                f"p = {p_val:.4f} | "
                f"{cluster_times[0]:.3f}–"
                f"{cluster_times[-1]:.3f} s"
            )

            ax.axvspan(
                cluster_times[0],
                cluster_times[-1],
                color='red',
                alpha=0.20
            )

    # =========================================================
    # FORMATTING
    # =========================================================

    ax.axvline(
        0,
        color='black',
        linestyle='--',
        linewidth=0.8
    )

    ax.axhline(
        0,
        color='black',
        linewidth=0.5
    )

    ax.set_ylabel(
        'Amplitude'
    )

    ax.set_xlabel(
        'Time (s)'
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        loc='upper right'
    )

    # =========================================================
    # FIGURE LABELS
    # =========================================================

    fig.suptitle(
        f"{comparison} — EEG ERP\n"
        f"ROI: {', '.join(ROI)}",
        fontsize=14
    )

    fig.tight_layout()

    plt.savefig(
        os.path.join(
            saving_path,
            f"{comparison}_EEG_ERP_"
            f"{epoch_cond1}_vs_{epoch_cond2}."
            f"{save_as}"
        ),
        dpi=300
    )

    plt.show()

    # =========================================================
    # RETURN
    # =========================================================

    return (
        avg_cond1,
        avg_cond2,
        cluster_results,
        subs_included
    )




def eeg_erp_change_diff_on_off(
        sub_dict_epochs,
        epoch_cond,
        condition_color_dict,
        saving_path,
        ROI,
        alpha=0.05,
        n_permutations=1000,
        save_as='png'
):
    """
    Compare the same trial type between DBS ON and DBS OFF.

    Data are restricted to the specified ROI channels.

    If the ROI contains one channel:
        that channel is used directly.

    If the ROI contains multiple channels:
        channels are averaged within each participant first.

    Participant-level ROI ERPs are then averaged across participants
    for the grand-average ERP.

    The statistical analysis is performed on the participant-level
    DBS ON - DBS OFF ROI differences.

    Parameters
    ----------
    sub_dict_epochs : dict
        Dictionary with subject/session IDs as keys and MNE Epochs
        as values.

        Keys must contain either 'DBS ON' or 'DBS OFF'.

    epoch_cond : str
        Trial type to analyse, e.g. 'GO_successful'.

    condition_color_dict : dict
        Dictionary mapping 'DBS ON' and 'DBS OFF' to plotting colors.

    saving_path : str
        Directory where the figure is saved.

    ROI : list of str
        Channel names defining the ROI.

        Example:
            ['STN_L']

        or:

            ['STN_L_1', 'STN_L_2', 'STN_L_3']

        If multiple channels are supplied, they are averaged within
        each participant before the group-level analysis.

    alpha : float
        Significance threshold.

    n_permutations : int
        Number of permutations.

    save_as : str
        Figure format, e.g. 'png' or 'pdf'.

    Returns
    -------
    avg_on : mne.Evoked
        Grand-average ROI ERP for DBS ON.

    avg_off : mne.Evoked
        Grand-average ROI ERP for DBS OFF.

    cluster_results : tuple
        (clusters, p_values) from the one-sample permutation test
        performed on the participant-level ROI differences.

    subjects_included : list
        Subjects included in the paired analysis.

    mean_rt_on : float
        Mean subject-level RT for DBS ON, in seconds.

    mean_rt_off : float
        Mean subject-level RT for DBS OFF, in seconds.

    rt_by_subject : dict
        Subject-level mean RTs for DBS ON and DBS OFF.
    """

    # =========================================================
    # PARAMETERS
    # =========================================================

    tmin, tmax = -0.5, 1.5
    sfreq = 250

    common_times = np.arange(
        tmin,
        tmax + 1 / sfreq,
        1 / sfreq
    )

    # ---------------------------------------------------------
    # Check ROI definition
    # ---------------------------------------------------------

    if isinstance(ROI, str):
        ROI = [ROI]

    if len(ROI) == 0:
        raise ValueError(
            "ROI must contain at least one channel."
        )

    # Store one ROI ERP per subject/session
    on_by_subject = {}
    off_by_subject = {}

    # Store subject-level RTs
    rt_on_by_subject = {}
    rt_off_by_subject = {}

    latency_matched = False

    # =========================================================
    # PARSE TRIAL CONDITION
    # =========================================================

    epoch_type, outcome_str = epoch_cond.split('_')

    if epoch_type == 'lmGO':
        latency_matched = True
        epoch_type = 'GO'

    outcome = (
        1.0 if outcome_str == 'successful'
        else 0.0
    )

    # =========================================================
    # COLLECT DATA
    # =========================================================

    for subject_session, epochs in sub_dict_epochs.items():

        # -----------------------------------------------------
        # Determine DBS status
        # -----------------------------------------------------

        if 'DBS ON' in subject_session:
            dbs_status = 'DBS ON'

        elif 'DBS OFF' in subject_session:
            dbs_status = 'DBS OFF'

        else:
            continue

        # -----------------------------------------------------
        # Determine subject ID
        # -----------------------------------------------------

        subject = (
            subject_session
            .replace('DBS ON', '')
            .replace('DBS OFF', '')
            .strip()
        )

        # =====================================================
        # CHECK ROI CHANNELS
        # =====================================================

        missing_channels = [
            ch for ch in ROI
            if ch not in epochs.ch_names
        ]

        if missing_channels:

            print(
                f"Skipping {subject_session}: "
                f"ROI channel(s) not found: "
                f"{missing_channels}"
            )

            continue

        # -----------------------------------------------------
        # Select only ROI channels
        # -----------------------------------------------------

        roi_data = epochs.copy().pick(
            ROI
        )

        # =====================================================
        # SELECT DESIRED TRIAL TYPE
        # =====================================================

        type_mask = (
            roi_data.metadata["event"] == epoch_type
        )

        outcome_mask = (
            roi_data.metadata["key_resp_experiment.corr"]
            == outcome
        )

        data = roi_data[
            type_mask & outcome_mask
        ]

        # -----------------------------------------------------
        # Optional latency matching
        # -----------------------------------------------------

        if latency_matched:

            if len(data) == 0:
                continue

            rt = np.asarray(
                data.metadata["key_resp_experiment.rt"]
            )

            threshold = np.percentile(
                rt,
                50
            )

            slow_mask = rt >= threshold

            data = data[slow_mask]

        # -----------------------------------------------------
        # Check that trials exist
        # -----------------------------------------------------

        if len(data) == 0:

            print(
                f"Skipping {subject_session}: "
                f"no {epoch_cond} trials"
            )

            continue

        # =====================================================
        # REACTION TIME
        # =====================================================

        rt = data.metadata[
            "key_resp_experiment.rt"
        ].to_numpy()

        rt = rt[~np.isnan(rt)]

        mean_rt_subject = (
            np.mean(rt)
            if len(rt) > 0
            else np.nan
        )

        if dbs_status == 'DBS ON':
            rt_on_by_subject[subject] = mean_rt_subject

        else:
            rt_off_by_subject[subject] = mean_rt_subject

        # =====================================================
        # CROP
        # =====================================================

        cropped_data = data.copy().crop(
            tmin=tmin,
            tmax=tmax
        )

        # =====================================================
        # BASELINE CORRECTION
        # =====================================================

        crunched_data = (
            cropped_data
            .copy()
            .apply_baseline((-0.5, 0))
        )

        # =====================================================
        # AVERAGE TRIALS WITHIN SUBJECT
        # =====================================================

        averaged_data = crunched_data.average()

        # =====================================================
        # INTERPOLATE TO COMMON TIME GRID
        # =====================================================

        new_data = np.vstack([
            np.interp(
                common_times,
                averaged_data.times,
                ch
            )
            for ch in averaged_data.data
        ])

        # =====================================================
        # AVERAGE ACROSS ROI CHANNELS
        # =====================================================

        # Shape before averaging:
        #
        #     ROI channels × time
        #
        # Shape after averaging:
        #
        #     time
        #
        # If there is only one ROI channel, this simply
        # returns that channel's time series.

        roi_erp = np.mean(
            new_data,
            axis=0
        )

        # =====================================================
        # STORE PARTICIPANT-LEVEL ROI ERP
        # =====================================================

        if dbs_status == 'DBS ON':
            on_by_subject[subject] = roi_erp

        else:
            off_by_subject[subject] = roi_erp

    # =========================================================
    # FIND SUBJECTS WITH BOTH DBS ON AND DBS OFF
    # =========================================================

    subjects_included = sorted(
        set(on_by_subject.keys())
        &
        set(off_by_subject.keys())
    )

    if len(subjects_included) == 0:

        raise RuntimeError(
            "No subjects have both DBS ON and DBS OFF data "
            f"for {epoch_cond}."
        )

    print(
        f"\nNumber of paired subjects included: "
        f"{len(subjects_included)}"
    )

    print(
        "Subjects:",
        subjects_included
    )

    print(
        "ROI channels:",
        ROI
    )

    # =========================================================
    # CREATE PAIRED ARRAYS
    # =========================================================

    X_on = np.array([
        on_by_subject[sub]
        for sub in subjects_included
    ])

    X_off = np.array([
        off_by_subject[sub]
        for sub in subjects_included
    ])

    # Shape:
    #
    #     subjects × time

    print("\nData shape:")
    print("DBS ON:", X_on.shape)
    print("DBS OFF:", X_off.shape)

    n_subjects = X_on.shape[0]

    times = common_times

    # =========================================================
    # GRAND AVERAGE ACROSS PARTICIPANTS
    # =========================================================

    grand_on = np.mean(
        X_on,
        axis=0
    )

    grand_off = np.mean(
        X_off,
        axis=0
    )

    # ---------------------------------------------------------
    # Create Evoked objects for the ROI
    # ---------------------------------------------------------

    # Use the first ROI channel's info as the template.
    #
    # Since the ROI has already been averaged across channels,
    # the resulting Evoked represents the ROI rather than an
    # individual electrode.

    first_subject = subjects_included[0]

    # Find an original Evoked/info object by rebuilding a
    # minimal info object from the first available dataset.

    template_epochs = None

    for subject_session, epochs in sub_dict_epochs.items():

        subject = (
            subject_session
            .replace('DBS ON', '')
            .replace('DBS OFF', '')
            .strip()
        )

        if subject == first_subject:

            template_epochs = epochs
            break

    if template_epochs is None:
        raise RuntimeError(
            "Could not find template epochs for ROI Evoked."
        )

    roi_info = (
        template_epochs
        .copy()
        .pick([ROI[0]])
        .info
        .copy()
    )

    avg_on = mne.EvokedArray(
        grand_on[np.newaxis, :],
        roi_info,
        tmin=common_times[0]
    )

    avg_off = mne.EvokedArray(
        grand_off[np.newaxis, :],
        roi_info,
        tmin=common_times[0]
    )

    # Rename the channel to make clear that it represents
    # the averaged ROI.

    if len(ROI) == 1:

        avg_on.rename_channels({
            ROI[0]: ROI[0]
        })

        avg_off.rename_channels({
            ROI[0]: ROI[0]
        })

    else:

        avg_on.rename_channels({
            ROI[0]: 'ROI'
        })

        avg_off.rename_channels({
            ROI[0]: 'ROI'
        })

    # =========================================================
    # GROUP-LEVEL REACTION TIMES
    # =========================================================

    mean_rt_on = np.nanmean([
        rt_on_by_subject[sub]
        for sub in subjects_included
    ])

    mean_rt_off = np.nanmean([
        rt_off_by_subject[sub]
        for sub in subjects_included
    ])

    mean_rt_on_ms = mean_rt_on * 1000
    mean_rt_off_ms = mean_rt_off * 1000

    print("\nMean reaction times:")

    print(
        f"{epoch_cond} — DBS ON: "
        f"{mean_rt_on_ms:.1f} ms"
    )

    print(
        f"{epoch_cond} — DBS OFF: "
        f"{mean_rt_off_ms:.1f} ms"
    )

    # =========================================================
    # STORE RTs BY SUBJECT
    # =========================================================

    rt_by_subject = {}

    for subject in subjects_included:

        rt_by_subject[subject] = {
            'DBS ON': rt_on_by_subject[subject],
            'DBS OFF': rt_off_by_subject[subject]
        }

    # =========================================================
    # PAIRED / REPEATED-MEASURES CLUSTER TEST
    # =========================================================

    # One ROI time series per participant.
    #
    # Therefore the statistical input has shape:
    #
    #     subjects × time

    difference = (
        X_on
        -
        X_off
    )

    T_obs, clusters, p_values, H0 = (
        permutation_cluster_1samp_test(
            difference,
            n_permutations=n_permutations,
            threshold=None,
            tail=0,
            out_type='indices',
            seed=42
        )
    )

    cluster_results = (
        clusters,
        p_values
    )

    # =========================================================
    # PLOT ROI ERP
    # =========================================================

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 5)
    )

    color_on = condition_color_dict['DBS ON']
    color_off = condition_color_dict['DBS OFF']

    # =========================================================
    # DBS ON
    # =========================================================

    ax.plot(
        times,
        avg_on.data[0],
        color=color_on,
        linewidth=1.5,
        label=(
            f"{epoch_cond} — DBS ON "
            f"(RT = {mean_rt_on_ms:.0f} ms)"
        )
    )

    ax.axvline(
        mean_rt_on,
        color=color_on,
        linestyle=':',
        linewidth=1.0
    )

    # =========================================================
    # DBS OFF
    # =========================================================

    ax.plot(
        times,
        avg_off.data[0],
        color=color_off,
        linewidth=1.5,
        label=(
            f"{epoch_cond} — DBS OFF "
            f"(RT = {mean_rt_off_ms:.0f} ms)"
        )
    )

    ax.axvline(
        mean_rt_off,
        color=color_off,
        linestyle=':',
        linewidth=1.0
    )

    # =========================================================
    # SIGNIFICANT CLUSTERS
    # =========================================================

    for cluster, p_val in zip(
        clusters,
        p_values
    ):

        if p_val < alpha:

            cluster_times = (
                times[cluster[0]]
            )

            print(
                f"Significant cluster | "
                f"ROI = {ROI} | "
                f"p = {p_val:.4f} | "
                f"{cluster_times[0]:.3f}–"
                f"{cluster_times[-1]:.3f} s"
            )

            ax.axvspan(
                cluster_times[0],
                cluster_times[-1],
                color='red',
                alpha=0.20
            )

    # =========================================================
    # FORMATTING
    # =========================================================

    ax.axvline(
        0,
        color='black',
        linestyle='--',
        linewidth=0.8
    )

    ax.axhline(
        0,
        color='black',
        linewidth=0.5
    )

    if len(ROI) == 1:

        roi_label = ROI[0]

    else:

        roi_label = (
            "ROI: "
            + ", ".join(ROI)
        )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_title(
        f"{epoch_cond} — DBS ON vs DBS OFF\n"
        f"{roi_label}"
    )

    ax.legend(
        loc='upper right'
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    # =========================================================
    # SAVE
    # =========================================================

    plt.savefig(
        os.path.join(
            saving_path,
            f"DBS_ON_vs_OFF_STN_ROI_ERP_"
            f"{epoch_cond}.{save_as}"
        ),
        dpi=300
    )

    plt.show()

    # =========================================================
    # RETURN
    # =========================================================

    return (
        avg_on,
        avg_off,
        cluster_results,
        subjects_included,
        mean_rt_on,
        mean_rt_off,
        rt_by_subject
    )




def eeg_erp_change_control_vs_pd(
        sub_dict_epochs,
        epoch_cond,
        condition_color_dict,
        saving_path,
        ROI,
        pd_condition='DBS ON',
        alpha=0.05,
        n_permutations=1000,
        save_as='png'
):
    """
    Compare the same trial type between Healthy controls and PD.

    The PD group can be either DBS ON or DBS OFF. The two groups
    consist of different subjects, so the statistical comparison is
    performed as an independent-samples cluster permutation test.

    Data are restricted to the specified ROI channels.

    If the ROI contains one channel:
        that channel is used directly.

    If the ROI contains multiple channels:
        channels are averaged within each participant first.

    Participant-level ROI ERPs are then averaged across participants
    for the group-level ERP.

    Statistical analysis
    --------------------
    Healthy controls and PD subjects are treated as independent groups.

    For each time point:

        difference = Healthy controls - PD

    An independent-samples cluster permutation test is then performed
    across time.

    Parameters
    ----------
    sub_dict_epochs : dict
        Dictionary with subject/session IDs as keys and MNE Epochs
        as values.

        Keys must identify subjects as either:

            'control'

        or:

            'DBS ON'
            'DBS OFF'

        Examples:
            'sub01 control'
            'sub02 DBS ON'
            'sub03 DBS OFF'

    epoch_cond : str
        Trial type to analyse, e.g. 'GO_successful'.

    condition_color_dict : dict
        Dictionary mapping the plotting conditions to colors.

        Example:
            {
                'control': 'blue',
                'DBS ON': 'red',
                'DBS OFF': 'green'
            }

        Only 'control' and the selected pd_condition are required.

    saving_path : str
        Directory where the figure is saved.

    ROI : list of str
        Channel names defining the ROI.

        Example:
            ['STN_L']

        or:

            ['STN_L_1', 'STN_L_2', 'STN_L_3']

        If multiple channels are supplied, they are averaged within
        each participant before the group-level analysis.

    pd_condition : str
        Which PD group to compare against Healthy controls.

        Must be either:
            'DBS ON'
        or:
            'DBS OFF'

    alpha : float
        Significance threshold.

    n_permutations : int
        Number of permutations.

    save_as : str
        Figure format, e.g. 'png' or 'pdf'.

    Returns
    -------
    avg_control : mne.Evoked
        Grand-average ROI ERP for Healthy controls.

    avg_pd : mne.Evoked
        Grand-average ROI ERP for PD.

    cluster_results : tuple
        Tuple containing:

            (clusters, p_values)

        from the independent-samples cluster permutation test.

    controls_included : list
        Healthy control subjects included in the analysis.

    pd_included : list
        PD subjects included in the analysis.

    mean_rt_control : float
        Mean subject-level RT for Healthy controls, in seconds.

    mean_rt_pd : float
        Mean subject-level RT for PD, in seconds.

    rt_by_subject : dict
        Subject-level mean RTs and group labels.
    """

    # =========================================================
    # PARAMETERS
    # =========================================================

    tmin, tmax = -0.5, 1.5
    sfreq = 250

    common_times = np.arange(
        tmin,
        tmax + 1 / sfreq,
        1 / sfreq
    )

    # =========================================================
    # CHECK PD CONDITION
    # =========================================================

    if pd_condition not in ['DBS ON', 'DBS OFF']:

        raise ValueError(
            "pd_condition must be either "
            "'DBS ON' or 'DBS OFF'."
        )

    # =========================================================
    # CHECK ROI DEFINITION
    # =========================================================

    if isinstance(ROI, str):
        ROI = [ROI]

    if len(ROI) == 0:

        raise ValueError(
            "ROI must contain at least one channel."
        )

    # =========================================================
    # STORAGE
    # =========================================================

    # One ROI ERP per subject
    control_by_subject = {}
    pd_by_subject = {}

    # Subject-level RTs
    rt_control_by_subject = {}
    rt_pd_by_subject = {}

    latency_matched = False

    # =========================================================
    # PARSE TRIAL CONDITION
    # =========================================================

    epoch_type, outcome_str = epoch_cond.split('_')

    if epoch_type == 'lmGO':

        latency_matched = True
        epoch_type = 'GO'

    outcome = (
        1.0
        if outcome_str == 'successful'
        else 0.0
    )

    # =========================================================
    # COLLECT DATA
    # =========================================================

    for subject_session, epochs in sub_dict_epochs.items():

        # =====================================================
        # DETERMINE GROUP
        # =====================================================

        if 'C' in subject_session:

            group = 'control'

        elif pd_condition in subject_session:

            group = 'PD'

        else:

            # Ignore the other PD condition.
            #
            # For example, if pd_condition == 'DBS ON',
            # subjects labelled 'DBS OFF' are ignored.

            continue

        # =====================================================
        # DETERMINE SUBJECT ID
        # =====================================================

        subject = (
            subject_session
            .replace('DBS ON', '')
            .replace('DBS OFF', '')
            .replace('control', '')
            .strip()
        )

        # =====================================================
        # CHECK ROI CHANNELS
        # =====================================================

        missing_channels = [
            ch
            for ch in ROI
            if ch not in epochs.ch_names
        ]

        if missing_channels:

            print(
                f"Skipping {subject_session}: "
                f"ROI channel(s) not found: "
                f"{missing_channels}"
            )

            continue

        # =====================================================
        # SELECT ONLY ROI CHANNELS
        # =====================================================

        roi_data = (
            epochs
            .copy()
            .pick(ROI)
        )

        # =====================================================
        # SELECT DESIRED TRIAL TYPE
        # =====================================================

        type_mask = (
            roi_data.metadata["event"]
            == epoch_type
        )

        outcome_mask = (
            roi_data.metadata[
                "key_resp_experiment.corr"
            ]
            == outcome
        )

        data = roi_data[
            type_mask & outcome_mask
        ]

        # =====================================================
        # OPTIONAL LATENCY MATCHING
        # =====================================================

        if latency_matched:

            if len(data) == 0:
                continue

            rt = np.asarray(
                data.metadata[
                    "key_resp_experiment.rt"
                ]
            )

            threshold = np.percentile(
                rt,
                50
            )

            slow_mask = rt >= threshold

            data = data[slow_mask]

        # =====================================================
        # CHECK THAT TRIALS EXIST
        # =====================================================

        if len(data) == 0:

            print(
                f"Skipping {subject_session}: "
                f"no {epoch_cond} trials"
            )

            continue

        # =====================================================
        # REACTION TIME
        # =====================================================

        rt = data.metadata[
            "key_resp_experiment.rt"
        ].to_numpy()

        # Remove NaNs
        rt = rt[~np.isnan(rt)]

        mean_rt_subject = (
            np.mean(rt)
            if len(rt) > 0
            else np.nan
        )

        if group == 'control':

            rt_control_by_subject[
                subject
            ] = mean_rt_subject

        else:

            rt_pd_by_subject[
                subject
            ] = mean_rt_subject

        # =====================================================
        # CROP
        # =====================================================

        cropped_data = (
            data
            .copy()
            .crop(
                tmin=tmin,
                tmax=tmax
            )
        )

        # =====================================================
        # BASELINE CORRECTION
        # =====================================================

        crunched_data = (
            cropped_data
            .copy()
            .apply_baseline(
                (-0.5, 0)
            )
        )

        # =====================================================
        # AVERAGE TRIALS WITHIN SUBJECT
        # =====================================================

        averaged_data = (
            crunched_data.average()
        )

        # =====================================================
        # INTERPOLATE TO COMMON TIME GRID
        # =====================================================

        new_data = np.vstack([
            np.interp(
                common_times,
                averaged_data.times,
                ch
            )
            for ch in averaged_data.data
        ])

        # =====================================================
        # AVERAGE ACROSS ROI CHANNELS
        # =====================================================

        # Before:
        #
        #     ROI channels × time
        #
        # After:
        #
        #     time
        #
        # For one channel, this simply returns that channel.
        #
        # For multiple channels, every channel contributes
        # equally to the participant's ROI ERP.

        roi_erp = np.mean(
            new_data,
            axis=0
        )

        # =====================================================
        # STORE PARTICIPANT-LEVEL ROI ERP
        # =====================================================

        if group == 'control':

            control_by_subject[
                subject
            ] = roi_erp

        else:

            pd_by_subject[
                subject
            ] = roi_erp

    # =========================================================
    # CHECK GROUPS
    # =========================================================

    controls_included = sorted(
        control_by_subject.keys()
    )

    pd_included = sorted(
        pd_by_subject.keys()
    )

    if len(controls_included) == 0:

        raise RuntimeError(
            "No Healthy control subjects have usable data "
            f"for {epoch_cond}."
        )

    if len(pd_included) == 0:

        raise RuntimeError(
            f"No PD subjects with {pd_condition} have usable "
            f"data for {epoch_cond}."
        )

    # =========================================================
    # PRINT GROUP INFORMATION
    # =========================================================

    print(
        f"\nHealthy controls included: "
        f"{len(controls_included)}"
    )

    print(
        "Controls:",
        controls_included
    )

    print(
        f"\nPD subjects included "
        f"({pd_condition}): "
        f"{len(pd_included)}"
    )

    print(
        "PD:",
        pd_included
    )

    print(
        "\nROI channels:",
        ROI
    )

    # =========================================================
    # CONVERT ERP DATA TO ARRAYS
    # =========================================================

    X_control = np.array([
        control_by_subject[sub]
        for sub in controls_included
    ])

    X_pd = np.array([
        pd_by_subject[sub]
        for sub in pd_included
    ])

    # Shape:
    #
    #     controls × time
    #
    #     PD × time

    print("\nData shape:")
    print(
        "Healthy controls:",
        X_control.shape
    )

    print(
        f"PD ({pd_condition}):",
        X_pd.shape
    )

    n_controls = X_control.shape[0]
    n_pd = X_pd.shape[0]

    times = common_times

    # =========================================================
    # GRAND AVERAGES
    # =========================================================

    grand_control = np.mean(
        X_control,
        axis=0
    )

    grand_pd = np.mean(
        X_pd,
        axis=0
    )

    # =========================================================
    # CREATE EVOKED OBJECTS FOR ROI
    # =========================================================

    # Find a template Epochs object containing the first ROI
    # channel. This is only used to create the MNE Info object.

    first_subject = (
        controls_included[0]
    )

    template_epochs = None

    for subject_session, epochs in sub_dict_epochs.items():

        subject = (
            subject_session
            .replace('DBS ON', '')
            .replace('DBS OFF', '')
            .replace('control', '')
            .strip()
        )

        if subject == first_subject:

            if all(
                ch in epochs.ch_names
                for ch in ROI
            ):

                template_epochs = epochs
                break

    # If the first control cannot provide the template,
    # try any dataset containing the ROI.

    if template_epochs is None:

        for subject_session, epochs in sub_dict_epochs.items():

            if all(
                ch in epochs.ch_names
                for ch in ROI
            ):

                template_epochs = epochs
                break

    if template_epochs is None:

        raise RuntimeError(
            "Could not find a dataset containing the ROI "
            "channels needed to create the Evoked object."
        )

    # Use the first ROI channel as an Info template.
    #
    # The actual ERP has already been averaged across all
    # ROI channels, so the resulting Evoked represents the ROI.

    roi_info = (
        template_epochs
        .copy()
        .pick([ROI[0]])
        .info
        .copy()
    )

    avg_control = mne.EvokedArray(
        grand_control[np.newaxis, :],
        roi_info,
        tmin=common_times[0]
    )

    avg_pd = mne.EvokedArray(
        grand_pd[np.newaxis, :],
        roi_info,
        tmin=common_times[0]
    )

    # Rename the channel to indicate that it represents
    # the ROI when multiple channels were averaged.

    if len(ROI) > 1:

        avg_control.rename_channels({
            ROI[0]: 'ROI'
        })

        avg_pd.rename_channels({
            ROI[0]: 'ROI'
        })

    # =========================================================
    # GROUP-LEVEL REACTION TIMES
    # =========================================================

    mean_rt_control = np.nanmean([
        rt_control_by_subject[sub]
        for sub in controls_included
    ])

    mean_rt_pd = np.nanmean([
        rt_pd_by_subject[sub]
        for sub in pd_included
    ])

    mean_rt_control_ms = (
        mean_rt_control * 1000
    )

    mean_rt_pd_ms = (
        mean_rt_pd * 1000
    )

    print(
        "\nMean reaction times:"
    )

    print(
        f"{epoch_cond} — Healthy controls: "
        f"{mean_rt_control_ms:.1f} ms"
    )

    print(
        f"{epoch_cond} — PD {pd_condition}: "
        f"{mean_rt_pd_ms:.1f} ms"
    )

    # =========================================================
    # STORE RTs BY SUBJECT
    # =========================================================

    rt_by_subject = {}

    for subject in controls_included:

        rt_by_subject[subject] = {
            'group': 'control',
            'RT': rt_control_by_subject[subject]
        }

    for subject in pd_included:

        rt_by_subject[subject] = {
            'group': 'PD',
            'condition': pd_condition,
            'RT': rt_pd_by_subject[subject]
        }

    # =========================================================
    # INDEPENDENT-SAMPLES CLUSTER PERMUTATION TEST
    # =========================================================

    # IMPORTANT:
    #
    # Controls and PD are different subjects.
    #
    # Therefore, we DO NOT calculate:
    #
    #     X_control - X_pd
    #
    # because the observations are not paired.
    #
    # Instead, permutation_cluster_test receives the two
    # independent groups separately:
    #
    #     [controls × time, PD × time]

    T_obs, clusters, p_values, H0 = (
        permutation_cluster_test(
            [
                X_control,
                X_pd
            ],
            n_permutations=n_permutations,
            threshold=None,
            tail=0,
            out_type='indices',
            seed=42
        )
    )

    cluster_results = (
        clusters,
        p_values
    )

    # =========================================================
    # PLOT ROI ERPs
    # =========================================================

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 5)
    )

    # =========================================================
    # COLORS
    # =========================================================

    color_control = (
        condition_color_dict['control']
    )

    color_pd = (
        condition_color_dict[pd_condition]
    )

    # =========================================================
    # HEALTHY CONTROLS
    # =========================================================

    ax.plot(
        times,
        avg_control.data[0],
        color=color_control,
        linewidth=1.5,
        label=(
            f"{epoch_cond} — Healthy controls "
            f"(n = {n_controls}, "
            f"RT = {mean_rt_control_ms:.0f} ms)"
        )
    )

    ax.axvline(
        mean_rt_control,
        color=color_control,
        linestyle=':',
        linewidth=1.0
    )

    # =========================================================
    # PD
    # =========================================================

    ax.plot(
        times,
        avg_pd.data[0],
        color=color_pd,
        linewidth=1.5,
        label=(
            f"{epoch_cond} — PD {pd_condition} "
            f"(n = {n_pd}, "
            f"RT = {mean_rt_pd_ms:.0f} ms)"
        )
    )

    ax.axvline(
        mean_rt_pd,
        color=color_pd,
        linestyle=':',
        linewidth=1.0
    )

    # =========================================================
    # SIGNIFICANT CLUSTERS
    # =========================================================

    for cluster, p_val in zip(
        clusters,
        p_values
    ):

        if p_val < alpha:

            # Cluster is a tuple containing the indices
            # along the tested time dimension.

            cluster_times = (
                times[cluster[0]]
            )

            print(
                f"Significant cluster | "
                f"Healthy controls vs PD {pd_condition} | "
                f"ROI = {ROI} | "
                f"p = {p_val:.4f} | "
                f"{cluster_times[0]:.3f}–"
                f"{cluster_times[-1]:.3f} s"
            )

            ax.axvspan(
                cluster_times[0],
                cluster_times[-1],
                color='red',
                alpha=0.20
            )

    # =========================================================
    # FORMATTING
    # =========================================================

    ax.axvline(
        0,
        color='black',
        linestyle='--',
        linewidth=0.8
    )

    ax.axhline(
        0,
        color='black',
        linewidth=0.5
    )

    if len(ROI) == 1:

        roi_label = ROI[0]

    else:

        roi_label = (
            "ROI: "
            +
            ", ".join(ROI)
        )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_title(
        f"{epoch_cond} — Healthy controls vs "
        f"PD {pd_condition}\n"
        f"{roi_label}"
    )

    ax.legend(
        loc='upper right'
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    # =========================================================
    # SAVE
    # =========================================================

    plt.savefig(
        os.path.join(
            saving_path,
            f"Control_vs_PD_{pd_condition.replace(' ', '_')}_"
            f"ROI_ERP_{epoch_cond}.{save_as}"
        ),
        dpi=300
    )

    plt.show()

    # =========================================================
    # RETURN
    # =========================================================

    return (
        avg_control,
        avg_pd,
        cluster_results,
        controls_included,
        pd_included,
        mean_rt_control,
        mean_rt_pd,
        rt_by_subject
    )
