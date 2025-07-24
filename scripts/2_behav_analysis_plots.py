# load librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
import seaborn as sns
import statsmodels.api as sm
import scipy
import json

from functions import utils
from functions import plotting
from functions import io


#################### 
# Prepare the data #
####################


# Use the Included or excluded.xlsx document:
onedrive_path = utils._get_onedrive_path()
working_path = os.path.dirname(os.getcwd())
results_path = join(working_path, "mSST_analysis", "results")
#subject_df = pd.read_excel(join(results_path, "Included or excluded subjects.xlsx"))
# extract the included subjects from the subject_df
#included_subjects = (subject_df[subject_df["included"] == "Yes"]["subject"].values).tolist()


# Set the saving path
behav_results_saving_path = join(results_path, "behav_results")
print(f"Results will be saved in : {behav_results_saving_path}")
# json_path = join(behav_results_saving_path, "JSON_test_23_june")
# if not os.path.isdir(behav_results_saving_path):
#     os.makedirs(behav_results_saving_path)

# read the json file containing the included and excluded subjects
# Open and read the JSON file
included_excluded_file = join(behav_results_saving_path, 'final_included_subjects.json')
with open(included_excluded_file, 'r') as file:
    included_subjects = json.load(file)


# Create a color palette for the subjects, and for the conditions
subject_colors = utils.create_color_palette(included_subjects)
plotting.plot_color_palette(subject_colors, behav_results_saving_path)
color_dict = {'DBS OFF': '#20a39e', 'DBS ON': '#ef5b5b', 'control': '#ffba49', 'preop': '#8E7DBE'}

# Load all data for all included subjects
data = io.load_behav_data(included_subjects, onedrive_path)

# Compute statistics for each loaded subject
stats = {}
stats = utils.extract_stats(data)

# Group the stats data by condition
# Initialize empty dictionaries
stats_OFF = {}
stats_ON = {}
stats_CONTROL = {}
stats_PREOP = {}

# Loop through the original dictionary and filter into sub-dictionaries
for key, value in stats.items():
    if "OFF" in key:
        stats_OFF[key] = value
    elif "ON" in key:
        stats_ON[key] = value
    elif "C" in key:
        stats_CONTROL[key] = value
    elif "preop" in key:
        stats_PREOP[key] = value


#############################################
#         Plot inhibition functions         #
#############################################
grouped_df = utils.create_grouped_df_for_inhibitory_functions(
    included_subjects,
    stats
)

plotting.plot_inhibitory_function_per_subject(grouped_df, color_dict, behav_results_saving_path)
plotting.plot_inhibitory_function_per_subject_zscored(
    grouped_df,
    stats,
    color_dict,
    behav_results_saving_path
)
plotting.plot_inhibitory_functions_per_groups(
        grouped_df,
        stats,
        color_dict,
        behav_results_saving_path
)


#############################################
#    Correlation reaction times and SSD     #
#############################################
rt_inhibition_df = utils.create_inhibition_df(
    included_subjects,
    stats
)
plotting.plot_reaction_time_relative_to_SSD(
        rt_inhibition_df,
        color_dict,
        behav_results_saving_path
)


#############################################
# Plot the data at the single subject level #
#############################################

plotting.plot_go_gf_rt_single_sub(stats_OFF,
        stats_ON,
        stats_CONTROL,
        stats_PREOP,
        color_dict,
        behav_results_saving_path
        )

plotting.plot_dbs_effect_success_rate_single_sub(
        stats_OFF,
        stats_ON,
        behav_results_saving_path
)

plotting.plot_dbs_effect_reaction_time_single_sub(
        stats_OFF,
        stats_ON,
        behav_results_saving_path
)




###################################
# Plot the data DBS ON vs DBS OFF #
###################################

df_reshaped = plotting.plot_prep_cost_on_vs_off_only_sub_with_2_sessions(
        stats_OFF,
        stats_ON,
        subject_colors,
        behav_results_saving_path
)

plotting.plot_prep_cost_on_vs_off_all_sub(
        stats_OFF,
        stats_ON,
        subject_colors,
        behav_results_saving_path)

plotting.plot_SSRT_on_vs_off_all_sub(
        stats_OFF,
        stats_ON,
        subject_colors,
        behav_results_saving_path
)

trial_types = ['GO', 'GC', 'GF', 'Go-STOP']
for trial in trial_types:
    plotting.plot_percent_success_on_vs_off(
            stats_OFF=stats_OFF,
            stats_ON=stats_ON,
            trial_type=trial,
            subject_colors=subject_colors,
            behav_results_saving_path=behav_results_saving_path
    )

for trial in trial_types:
    plotting.plot_reaction_time_on_vs_off(
            stats_OFF=stats_OFF,
            stats_ON=stats_ON,
            trial_type=trial,
            subject_colors=subject_colors,
            behav_results_saving_path=behav_results_saving_path
    )

plotting.plot_dbs_effect_on_rt_all_sub_with_2_sessions_all_trial_types(
        stats_OFF,
        stats_ON,
        subject_colors,
        behav_results_saving_path
)

plotting.plot_early_press_on_vs_off(
        stats_OFF,
        stats_ON,
        subject_colors,
        behav_results_saving_path
)


#################################################################
# Plot the data with all subjects, including controls and preop #
#################################################################

plotting.plot_go_gf_rt_group(
    stats_OFF,
    stats_ON,
    stats_CONTROL,
    stats_PREOP,
    color_dict,
    behav_results_saving_path
)

df_proactive_all = plotting.plot_prep_cost_all_groups(
    stats_OFF,
    stats_ON,
    stats_CONTROL,
    stats_PREOP,
    color_dict,
    behav_results_saving_path
)

df_reactive_all = plotting.plot_SSRT_all_groups(
    stats_OFF,
    stats_ON,
    stats_CONTROL,
    stats_PREOP,
    color_dict,
    behav_results_saving_path        
)


df_merged = utils.prepare_merged_dataframe(
    df_proactive_all,
    df_reactive_all,
    stats_OFF,
    stats_ON,
    stats_CONTROL,
    stats_PREOP,
    behav_results_saving_path
)


#############################
# Plot correlation analysis #
#############################


plotting.plot_corr_prep_cost_SSRT(df_merged, behav_results_saving_path)

plotting.plot_corr_SSD_SSRT(df_merged, behav_results_saving_path)

plotting.plot_corr_gort_ssrt(
    stats,    
    behav_results_saving_path
)






















