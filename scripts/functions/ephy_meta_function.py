import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import mne
import os

from functions.analysis import identify_significant_clusters


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
        threshold_GC=None
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
