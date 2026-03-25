import numpy as np
import matplotlib.pyplot as plt

from functions.analysis import get_tfr_decomposition
from functions.ephy_plotting import plot_tfr_result

def meta_function_tfr(
        sub_groups, # contains the different sub-groups to compare (e.g. ['DBS ON'], ['DBS ON', 'DBS OFF'], ['controls', 'DBS OFF'])
        sub_dict_epochs, # dictionary containing all subjects' data
        single_sub, # True or False, whether to plot single subject TFRs or group averages
        epochs_cond, # what condition(s) to analyse (e.g. ['GO_successful_GO'], ['GO_successful_GO - GS_successful_GO'])
        roi, # region of interest (e.g. ['LEFT_STN'], ['RIGHT_STN'], ['LEFT_STN', 'RIGHT_STN']). A list creates an average of all channels in the list.
        tfr_args, # dictionary containing tfr parameters (e.g. frequencies of interest, number of cycles, etc.)
        tmin_tmax = [0, 1.5], # time window of interest for plotting (e.g. [-0.5, 1.5])
        vmin_vmax = [-70, 70], # min and max values for color scale (e.g. [-70, 70] if in percentage change)
        baseline_correction = True, # True or False, whether to apply baseline correction
        baseline_correction_method = 'group_average', # method for baseline correction (e.g. 'group_average' or 'single_trial')
        saving_path = "C:\\Users\\Juliette\\Research\\Projects\\analysis_mSST\\results", # path to save the figures
        save_as = 'png', # format to save the figures (e.g. 'png', 'pdf')
):
    """
    epochs_cond can be a list of one condition (e.g. ['GO_successful_GO']) or a 
    list of two conditions to compare (e.g. ['GO_successful_GO - GS_successful_GO']). 
    The structure should always be 'condition_outcome_(cue to align on)'.
    For example:
    GS_successful_square means Stop Trials, successful trials, aligned to Go cue.
    GS_successful_triangle means Stop Trials, successful trials, aligned to Stop cue.
    GS_unsuccessful_feedback means Stop Trials, unsuccessful trials, aligned to feedback.
    GO_successful_response means Go Trials, successful trials, aligned to response.
    """
    all_percentage_change = []
    subs_included = []

    # First, extract information about what the user wants to plot/analyze:
    single_condition = True if len(epochs_cond) == 1 else False
    coi = epochs_cond[0] if single_condition else epochs_cond[0] + ' - ' + epochs_cond[1]
    if len(sub_groups) == 1:
        dbs_status = sub_groups[0]
    else:
        dbs_status = sub_groups[0] + ' - ' + sub_groups[1]

    # Validate epochs_cond format
    assert all(epochs_cond[i].split('_')[0] in ['GO', 'GS', 'GF', 'GC'] for i in range(len(epochs_cond))), "Condition should start with 'GO', 'GF', 'GC' or 'GS'" 
    if len(epochs_cond) > 1:
        suffix0 = epochs_cond[0].split('_')[-1]
        suffix1 = epochs_cond[1].split('_')[-1]
        if not (suffix0 == suffix1 or {suffix0, suffix1} == {"stop", "continue"}):
            raise ValueError("Both conditions should be aligned to the same cue for comparison (or be stop/continue)")

    # Then, extract the relevant epochs decompositions from sub_dict based on the specified conditions and ROIs.
    # tfr_dict = {}
    
    # first implement at single condition level:
    # sub-select epochs/subjects based on sub_groups
    selected_sub_dict_epochs = {key: value for key, value in sub_dict_epochs.items() if dbs_status in key}
    for sub, epochs in selected_sub_dict_epochs.items():
        if single_sub: 
            all_percentage_change = []
            sub_num = 1

        mean_power, times, freqs = get_tfr_decomposition(
            epochs = epochs, 
            cond_of_interest = epochs_cond[0], 
            ch_names = roi[0], 
            tfr_args = tfr_args, 
            baseline_correction= baseline_correction,
            baseline_correction_method= baseline_correction_method,
            tmin_tmax = tmin_tmax
        )
        # tfr_dict[sub] = {
        #     'mean_power': mean_power,
        #     'times': times,
        #     'freqs': freqs}        

        all_percentage_change.append(mean_power)
        subs_included.append(sub)

        # Then plot the average TFR requested
        # Figure title
        if single_sub : 
            sub_num = 1
            plot_tfr_result(
                all_percentage_change = all_percentage_change,
                epochs_cond = epochs_cond,
                roi = roi,
                tfr_args = tfr_args,
                tmin_tmax = tmin_tmax,
                vmin_vmax = vmin_vmax,
                baseline_correction = baseline_correction,
                baseline_correction_method = baseline_correction_method,
                sub_num = sub_num,
                sub = sub
                )

    if not single_sub:
        sub_num = len(subs_included)
        # Convert to array (shape: (n_subs, n_freqs, n_times))
        min_len = min(e.shape[1] for e in all_percentage_change)
        all_percentage_change = [e[:, :min_len] for e in all_percentage_change]
        all_percentage_change = np.stack(all_percentage_change, axis=0)
        # Compute grand averages
        avg_percentage_change = np.nanmean(all_percentage_change, axis=0)
        plot_tfr_result(
            all_percentage_change = avg_percentage_change,
            epochs_cond = epochs_cond,
            roi = roi,
            tfr_args = tfr_args,
            tmin_tmax = tmin_tmax,
            vmin_vmax = vmin_vmax,
            baseline_correction = baseline_correction,
            baseline_correction_method = baseline_correction_method,
            sub_num = sub_num,
            sub = None
            )        


# If needed, add statistical tests between conditions and add to the plot.





# Return plots and/or statistical results.