import mne
import numpy as np
import os
import json

from functions import ephy_plotting, utils, analysis

working_path = os.path.join(os.path.dirname(os.getcwd()), 'analysis_mSST')
results_path = os.path.join(working_path, "results")
behav_results_saving_path = os.path.join(results_path, "behav_results")
# read the json file containing the included and excluded subjects, based on the behavioral results
included_excluded_file = os.path.join(behav_results_saving_path, 'final_included_subjects.json')
with open(included_excluded_file, 'r') as file:
    included_subjects = json.load(file)

# keep only subjects starting with "sub":
included_subjects = [subj for subj in included_subjects if subj.startswith('sub')]
print(f'Included_subjects: {included_subjects}')
onedrive_path = utils._get_onedrive_path()

#  Set saving path
freq_wide = 'all frequencies'
fmax = 80
saving_path_group0 = os.path.join(results_path, 'group_level', 'eeg_perc_sig_change', freq_wide)
os.makedirs(saving_path_group0, exist_ok=True)  # Create the directory if it doesn't exist
saving_path_on_off = os.path.join(results_path, 'ON_vs_OFF', 'EEG', freq_wide)
os.makedirs(saving_path_on_off, exist_ok=True)

# Set source path for epochs
epochs_path = os.path.join(results_path, 'eeg_epochs')

sub_dict_epochs_high_sf = {}  #  Stores the epochs for each subject/session

included_subjects.remove('sub033 DBS OFF mSST') # head tremor artifacts, EEG recording bad
included_subjects.remove('sub033 DBS ON mSST')  # head tremor artifacts, EEG recording bad

ROIs = {
    #'sensorimotor': ['C3', 'CP5', 'CP1']
    #'frontal': ['Fz']
    # 'preSMA': ['Fz', 'FC1', 'FC2', 'Cz'#, 'FC5', 'FC6'
    #              ],
    #'r_IFG': ['F8', 'FC6', 'F4'],
    # 'l_M1': ['C3', 'FC5', 'CP1'],
    # 'r_DLPFC': ['F4', 'FC2'],
    # 'l_DLPFC': ['F3', 'FC1'],
    # 'central': ['Cz'],
    #'C3': ['C3'],
    # 'l_IFG': ['F7', 'FC5', 'F3'],
    # 'r_M1': ['C4', 'FC6', 'CP2'],
    # 'occipital': ['O1', 'O2', 'Oz'],
    'FC2': ['FC2'],
}

for session_ID in included_subjects:
    epochs = mne.read_epochs(os.path.join(epochs_path, f"{session_ID}_EEG_cleaned-long-epo.fif"), preload=True)
    #epochs.resample(2048, npad='auto')
    sub_dict_epochs_high_sf[session_ID] = epochs

new_sf = 200

sub_dict_epochs = {}

print('Now resampling epochs to 200 Hz for time-frequency analysis...')
for subject, data in sub_dict_epochs_high_sf.items():
    sub_dict_epochs[subject] = data.copy().resample(sfreq=new_sf)

######################
### TFR PARAMETERS ###
######################

decim = 1 
freqs = np.arange(1, fmax, 1) 
# For 500ms time resolution at 1 Hz: n_cycles = 1 * 0.5 = 0.5
# For 50ms time resolution at 40 Hz: n_cycles = 40 * 0.05 = 2
# Linear interpolation between these points
#n_cycles = 0.5 + (freqs - 1) * (2 - 0.5) / (40 - 1)
#n_cycles = freqs / 2.0
n_cycles = np.minimum(np.maximum(freqs / 2.0, 2), 20)

tfr_args = dict(
    method="morlet",
    freqs=freqs,
    n_cycles=n_cycles,
    decim=decim,
    return_itc=False,
    average=False
)        
baseline_correction_method = 'group_average'  # 'single_trial' or 'group_average'
tmin_tmax = [0, 1500]
vmin_vmax = [-70, 70] if baseline_correction_method == 'group_average' else [-5, 5]

sub_nums = []  #  List to store unique subject numbers

for sub in included_subjects:
    sub = sub[:6]
    if sub not in sub_nums:  # Check if sub is already in sub_nums
        sub_nums.append(sub)

for roi in ROIs.keys():
    print(f"Now processing ROI: {roi}")
    saving_path_group = os.path.join(saving_path_group0, roi)
    os.makedirs(saving_path_group, exist_ok=True)  # Create the directory if it doesn't exist
    # Create single subject level trial specific plots:
    for sub in sub_nums:
        print(f"Now processing sub: {sub}")
        single_sub_dict_subsets = {key: value for key, value in sub_dict_epochs.items() if sub in key}
        print(single_sub_dict_subsets.keys())
        saving_path_single = os.path.join(results_path, 'single_sub', f'{sub} mSST','eeg_perc_sig_change', freq_wide, roi)
        os.makedirs(saving_path_single, exist_ok=True)  # Create the directory if it doesn't exist

        ## single condition            
        for dbs_status in ['DBS OFF', 'DBS ON']:
            if any(dbs_status in key for key in single_sub_dict_subsets.keys()):
                for cond in [
                    'GO_successful', 
                    # #'GO_unsuccessful', 
                    'GF_successful', 
                    # #'GF_unsuccessful',
                    'GC_successful', 
                    # #'GC_unsuccessful',
                    'GS_successful', 
                    'GS_unsuccessful',
                    'stop_successful',
                    'stop_unsuccessful',
                    'continue_successful'
                    ]:
                    ephy_plotting.eeg_tfr_pow_change_cond(
                        sub_dict = single_sub_dict_subsets, 
                        dbs_status = dbs_status, 
                        epoch_cond = cond,
                        ch_of_interest = ROIs[roi],
                        tfr_args = tfr_args, 
                        t_min_max = tmin_tmax, 
                        vmin_vmax = vmin_vmax,
                        baseline_correction=True,
                        saving_path=saving_path_single,
                        show_fig=False,
                        save_as = 'png',
                        baseline_correction_method = baseline_correction_method
                        )
                    
    # Create single subject level condition comparison plots:
    for sub in sub_nums:
        print(f"Now processing sub: {sub}")
        single_sub_dict_subsets = {key: value for key, value in sub_dict_epochs.items() if sub in key}
        print(single_sub_dict_subsets.keys())
        saving_path_single = os.path.join(results_path, 'single_sub', f'{sub} mSST','eeg_perc_sig_change', freq_wide)
        os.makedirs(saving_path_single, exist_ok=True)  # Create the directory if it doesn't exist

        ## difference conditions          
        for dbs_status in ['DBS OFF', 'DBS ON']:
            if any(dbs_status in key for key in single_sub_dict_subsets.keys()):
                condition = f"{dbs_status} GS_successful - GO_successful {sub}"
                ephy_plotting.eeg_perc_pow_diff_cond(
                                sub_dict = single_sub_dict_subsets,  
                                dbs_status = dbs_status,  
                                tfr_args = tfr_args, 
                                t_min_max = tmin_tmax, 
                                vmin_vmax = vmin_vmax,
                                epoch_cond1 = "GS_successful",
                                epoch_cond2 = "GO_successful",
                                condition = condition,
                                ch_of_interest = ROIs[roi],
                                saving_path = saving_path_single,
                                show_fig = False,
                                add_rt = True
                                )
                
                condition = f"{dbs_status} GO_successful - GF_successful {sub}"
                ephy_plotting.eeg_perc_pow_diff_cond(
                                sub_dict = single_sub_dict_subsets,  
                                dbs_status = dbs_status,  
                                tfr_args = tfr_args, 
                                t_min_max = tmin_tmax, 
                                vmin_vmax = vmin_vmax,
                                epoch_cond1 = "GO_successful",
                                epoch_cond2 = "GF_successful",
                                condition = condition,
                                ch_of_interest = ROIs[roi],
                                saving_path = saving_path_single,
                                show_fig = False,
                                add_rt = True
                                )
                condition = f"{dbs_status} GS_successful - GS_unsuccessful {sub}"
                ephy_plotting.eeg_perc_pow_diff_cond(
                                sub_dict = single_sub_dict_subsets,  
                                dbs_status = dbs_status,  
                                tfr_args = tfr_args, 
                                t_min_max = tmin_tmax,
                                vmin_vmax = vmin_vmax,
                                epoch_cond1 = "GS_successful",
                                epoch_cond2 = "GS_unsuccessful",
                                condition = condition,
                                ch_of_interest = ROIs[roi],
                                saving_path = saving_path_single,
                                show_fig = False,
                                add_rt = True
                                )
                condition = f"{dbs_status} stop_successful - stop_unsuccessful {sub}"
                ephy_plotting.eeg_perc_pow_diff_cond(
                                sub_dict = single_sub_dict_subsets,  
                                dbs_status = dbs_status,  
                                tfr_args = tfr_args, 
                                t_min_max = tmin_tmax, 
                                vmin_vmax = vmin_vmax,
                                epoch_cond1 = "stop_successful",
                                epoch_cond2 = "stop_unsuccessful",
                                condition = condition,
                                ch_of_interest = ROIs[roi],
                                saving_path = saving_path_single,
                                show_fig = False,
                                add_rt = True
                                )
                condition = f"{dbs_status} stop_successful - continue_successful {sub}"
                ephy_plotting.eeg_perc_pow_diff_cond(
                                sub_dict = single_sub_dict_subsets,  
                                dbs_status = dbs_status,  
                                tfr_args = tfr_args, 
                                t_min_max = tmin_tmax, 
                                vmin_vmax = vmin_vmax,
                                epoch_cond1 = "stop_successful",
                                epoch_cond2 = "continue_successful",
                                condition = condition,
                                ch_of_interest = ROIs[roi],
                                saving_path = saving_path_single,
                                show_fig = False,
                                add_rt = True
                                )
                
    # Create group level plots of each condition:
    for dbs_status in [
        'DBS OFF', 
        'DBS ON'
        ]:                
        for cond in [
            'lm_GO_successful',
            'GO_successful', 
            # #'GO_unsuccessful', 
            'GF_successful', 
            #'GF_unsuccessful',
            'GC_successful', 
            #'GC_unsuccessful',
            'GS_successful', 
            'GS_unsuccessful',
            'stop_successful',
            'stop_unsuccessful',
            'continue_successful'
            ]:
            print(f"Now processing: {dbs_status} - {cond} ")
            ephy_plotting.eeg_tfr_pow_change_cond(
                        sub_dict = sub_dict_epochs, 
                        dbs_status = dbs_status, 
                        epoch_cond = cond, 
                        ch_of_interest = ROIs[roi],
                        tfr_args = tfr_args, 
                        t_min_max = tmin_tmax, 
                        vmin_vmax = vmin_vmax,
                        baseline_correction = True,
                        saving_path = saving_path_group,
                        show_fig = False,
                        add_rt = True,
                        baseline_correction_method = baseline_correction_method
                        )

    # Create group level plots of condition comparisons:
    for dbs_status in [
        'DBS OFF', 
        'DBS ON'
        ]:
        condition = f"{dbs_status} GS successful - lm_GO successful"
        print(f"Now processing: {condition}")
        ephy_plotting.eeg_perc_pow_diff_cond(
                    sub_dict = sub_dict_epochs, 
                    dbs_status = dbs_status,  
                    tfr_args = tfr_args, 
                    t_min_max = tmin_tmax, 
                    vmin_vmax = vmin_vmax,
                    epoch_cond1 = "GS_successful",
                    epoch_cond2 = "lm_GO_successful",
                    condition = condition,
                    ch_of_interest = ROIs[roi],
                    saving_path = saving_path_group,
                    show_fig = False,
                    add_rt = True,
                    baseline_correction_method=baseline_correction_method
                    )
        
        # condition = f"{dbs_status} GS successful - GO successful"
        # print(f"Now processing: {condition}")
        # ephy_plotting.eeg_perc_pow_diff_cond(
        #             sub_dict = sub_dict_epochs, 
        #             dbs_status = dbs_status,  
        #             tfr_args = tfr_args, 
        #             t_min_max = tmin_tmax, 
        #             vmin_vmax = vmin_vmax,
        #             epoch_cond1 = "GS_successful",
        #             epoch_cond2 = "GO_successful",
        #             condition = condition,
        #             ch_of_interest = ch_of_interest,
        #             saving_path = saving_path_group,
        #             show_fig = True,
        #             add_rt = True,
        #             baseline_correction_method=baseline_correction_method
        #             )
        condition = f"{dbs_status} GO successful - GS successful"
        print(f"Now processing: {condition}")
        ephy_plotting.eeg_perc_pow_diff_cond(
                    sub_dict = sub_dict_epochs, 
                    dbs_status = dbs_status,  
                    tfr_args = tfr_args, 
                    t_min_max = tmin_tmax, 
                    vmin_vmax = vmin_vmax,
                    epoch_cond1 = "GO_successful",
                    epoch_cond2 = "GS_successful",
                    condition = condition,
                    ch_of_interest = ROIs[roi],
                    saving_path = saving_path_group,
                    show_fig = False,
                    add_rt = True,
                    baseline_correction_method=baseline_correction_method
                    )
        
        condition = f"{dbs_status} GO successful - GF successful"
        print(f"Now processing: {condition}")
        ephy_plotting.eeg_perc_pow_diff_cond(
                    sub_dict = sub_dict_epochs, 
                    dbs_status = dbs_status,  
                    tfr_args = tfr_args, 
                    t_min_max = tmin_tmax, 
                    vmin_vmax = vmin_vmax,
                    epoch_cond1 = "GO_successful",
                    epoch_cond2 = "GF_successful",
                    condition = condition,
                    ch_of_interest = ROIs[roi],
                    saving_path = saving_path_group,
                    show_fig = False,
                    add_rt = True,
                    baseline_correction_method=baseline_correction_method
                    )

        condition = f"{dbs_status} GS successful - GS unsuccessful"
        print(f"Now processing: {condition}")
        ephy_plotting.eeg_perc_pow_diff_cond(
                    sub_dict = sub_dict_epochs,
                    dbs_status = dbs_status,  
                    tfr_args = tfr_args, 
                    t_min_max = tmin_tmax, 
                    vmin_vmax = vmin_vmax,
                    epoch_cond1 = "GS_successful",
                    epoch_cond2 = "GS_unsuccessful",
                    condition = condition,
                    ch_of_interest = ROIs[roi],
                    saving_path = saving_path_group,
                    show_fig = False,
                    add_rt = True,
                    save_as = 'png',
                    baseline_correction_method=baseline_correction_method
                    )

        condition = f"{dbs_status} stop successful - stop unsuccessful"
        print(f"Now processing: {condition}")
        ephy_plotting.eeg_perc_pow_diff_cond(
                    sub_dict = sub_dict_epochs,
                    dbs_status = dbs_status,  
                    tfr_args = tfr_args, 
                    t_min_max = tmin_tmax, 
                    vmin_vmax = vmin_vmax,
                    epoch_cond1 = "stop_successful",
                    epoch_cond2 = "stop_unsuccessful",
                    condition = condition,
                    ch_of_interest = ROIs[roi],
                    saving_path = saving_path_group,
                    show_fig = False,
                    add_rt = True,
                    baseline_correction_method=baseline_correction_method
                    )
        
        # condition = f"{dbs_status} stop unsuccessful - stop successful"
        # print(f"Now processing: {condition}")
        # ephy_plotting.perc_pow_diff_cond(
        #             sub_dict = sub_dict_epochs,
        #             dbs_status = dbs_status,  
        #             tfr_args = tfr_args, 
        #             t_min_max = tmin_tmax, 
        #             vmin_vmax = vmin_vmax,
        #             epoch_cond1 = "stop_unsuccessful",
        #             epoch_cond2 = "stop_successful",
        #             condition = condition,
        #             saving_path = saving_path_group,
        #             show_fig = True,
        #             add_rt = True
        #             )
        
        condition = f"{dbs_status} stop successful - continue successful"
        print(f"Now processing: {condition}")
        ephy_plotting.eeg_perc_pow_diff_cond(
                    sub_dict = sub_dict_epochs,
                    dbs_status = dbs_status,  
                    tfr_args = tfr_args, 
                    t_min_max = tmin_tmax, 
                    vmin_vmax = vmin_vmax,
                    epoch_cond1 = "stop_successful",
                    epoch_cond2 = "continue_successful",
                    condition = condition,
                    ch_of_interest = ROIs[roi],
                    saving_path = saving_path_group,
                    show_fig = False,
                    add_rt = True,
                    baseline_correction_method=baseline_correction_method
                    )
        
        condition = f"{dbs_status} GS successful - GC successful"
        print(f"Now processing: {condition}")
        ephy_plotting.eeg_perc_pow_diff_cond(
                    sub_dict = sub_dict_epochs,
                    dbs_status = dbs_status,  
                    tfr_args = tfr_args, 
                    t_min_max = tmin_tmax, 
                    vmin_vmax = vmin_vmax,
                    epoch_cond1 = "GS_successful",
                    epoch_cond2 = "GC_successful",
                    condition = condition,
                    ch_of_interest = ROIs[roi],
                    saving_path = saving_path_group,
                    show_fig = False,
                    add_rt = True, 
                    baseline_correction_method=baseline_correction_method
                    )
        
        # condition = f"{dbs_status} stop unsuccessful - continue successful"
        # print(f"Now processing: {condition}")
        # ephy_plotting.eeg_perc_pow_diff_cond(
        #             sub_dict = sub_dict_epochs,
        #             dbs_status = dbs_status,  
        #             tfr_args = tfr_args, 
        #             t_min_max = tmin_tmax, 
        #             vmin_vmax = vmin_vmax,
        #             epoch_cond1 = "stop_unsuccessful",
        #             epoch_cond2 = "continue_successful",
        #             condition = condition,
        #             ch_of_interest = ROIs[roi],
        #             saving_path = saving_path_group,
        #             show_fig = False,
        #             add_rt = True,
        #             baseline_correction_method=baseline_correction_method
        #             )
        
        condition = f"{dbs_status} continue successful - stop unsuccessful"
        print(f"Now processing: {condition}")
        ephy_plotting.eeg_perc_pow_diff_cond(
                    sub_dict = sub_dict_epochs,
                    dbs_status = dbs_status,  
                    tfr_args = tfr_args, 
                    t_min_max = tmin_tmax, 
                    vmin_vmax = vmin_vmax,
                    epoch_cond1 = "continue_successful",
                    epoch_cond2 = "stop_unsuccessful",
                    condition = condition,
                    ch_of_interest = ROIs[roi],
                    saving_path = saving_path_group,
                    show_fig = False,
                    add_rt = True,
                    baseline_correction_method=baseline_correction_method
                    )

        # condition = f"{dbs_status} GS unsuccessful - GC successful"
        # print(f"Now processing: {condition}")
        # ephy_plotting.eeg_perc_pow_diff_cond(
        #             sub_dict = sub_dict_epochs,
        #             dbs_status = dbs_status,  
        #             tfr_args = tfr_args, 
        #             t_min_max = tmin_tmax, 
        #             vmin_vmax = vmin_vmax,
        #             epoch_cond1 = "GS_unsuccessful",
        #             epoch_cond2 = "GC_successful",
        #             condition = condition,
        #             ch_of_interest = ROIs[roi],
        #             saving_path = saving_path_group,
        #             show_fig = False,
        #             add_rt = True,
        #             baseline_correction_method=baseline_correction_method
        #             )            
            