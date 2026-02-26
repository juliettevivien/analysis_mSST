# load librairies
import os
from os.path import join
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy

from functions import utils, io, behav_analysis

# pick format to save figures : png for quick visualization, pdf for illustrator
save_as = 'png'
color_dict = {
    'DBS ON': '#20a39e', 
    'DBS OFF': '#ef5b5b', 
    'control': '#ffba49', 
    'preop': '#8E7DBE',
    'Session 1': "#206ea1", 
    'Session 2': "#5FA363", 
    }

# pick tests to run:
scales_visualization = False
scales_correlation = False
visualize_updrs = False
visualize_rt_distribution = False
visualize_rt_distribution_overlapped = False
correlate_ssrt_prep_cost = False
correlate_ssrt_bis = False
plot_prep_cost = True
plot_ssrt = True

# set paths
onedrive_path = utils._get_onedrive_path()
working_path = os.path.dirname(os.getcwd())
results_path = join(working_path, "analysis_mSST", "results")
behav_results_saving_path = join(results_path, "behav_results")
if not os.path.isdir(behav_results_saving_path):
    os.makedirs(behav_results_saving_path)


# load subjects from the JSON file created in the previous script
with open(join(results_path, 'final_included_subjects.json'), 'r') as f:
    included_subjects = json.load(f)
print(f'Included subjects: {included_subjects}')
# create a color palette for the included subjects
subject_colors = utils.create_color_palette(included_subjects)
utils.plot_color_palette(
    subject_colors = subject_colors,
    save_as='png', 
    saving_path=behav_results_saving_path
    )

# load excel files for all included subjects and extract stats
data = io.load_behav_data(included_subjects, onedrive_path)
stats = utils.extract_stats(data)

# get hand for each subject
sub_hand_dict = {}
for sub, datas in stats.items():
    hand = datas['hand']
    sub_hand_dict[sub] = hand

# also load recording information and scales information for all included subjects
# fetch excel file containing recording and subject information
excel_file = join(onedrive_path, 'WP3_rec_info.xlsx')
# select the relevant sheet called 'Subjects List'
subject_info_df = pd.read_excel(excel_file, sheet_name='Subjects List')
control_info_df = pd.read_excel(excel_file, sheet_name='Controls List')

# get session order
behav_analysis.get_session_order(
    included_subjects = included_subjects,
    subject_info_df = subject_info_df,
    behav_results_saving_path = behav_results_saving_path
)

# get the behavioral scores for all scales of interest
scale_names = ['UPDRS_ON', 'UPDRS_OFF', 'MOCA', 'BDI', 'SAS', 'BIS_TOTAL', 'BIS_nonplanning', 'BIS_motor', 'BIS_attentional', 'OCI_TOTAL', 'OCI_washing', 'OCI_obsessing', 'OCI_hoarding', 'OCI_ordering', 'OCI_checking', 'OCI_neutralizing']
subs = [subj.split(' ')[0] for subj in included_subjects]
# remove duplicates in subs and order the list:
subs = sorted(list(set(subs)))


sub_scale_dict = {}
for sub in subs:
    if sub.startswith('C'):
        sub_line = control_info_df[control_info_df['StudyCode'] == sub]
    else:
        sub_line = subject_info_df[subject_info_df['StudyCode'] == sub]
    scale_dict = {}
    for scale in scale_names:
        if scale in sub_line.columns:
            scale_value = sub_line[scale].values[0]
        else:
            scale_value = None
        scale_dict[scale] = scale_value
    sub_scale_dict[sub] = scale_dict    

# save sub_scale_dict to an excel file:
sub_scale_df = pd.DataFrame.from_dict(sub_scale_dict, orient='index')
sub_scale_df.to_excel(join(behav_results_saving_path, 'sub_scale_dict.xlsx'))

if scales_visualization:
    temp_save = join(behav_results_saving_path, "scales")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)
    behav_analysis.visualize_scales_scores(
        scale_names = scale_names,
        sub_scale_dict = sub_scale_dict,
        subs = subs,
        subject_colors = subject_colors,
        color_dict = color_dict,
        visualize_by = 'subject',
        saving_path = temp_save,
        save_as = save_as
    )

if scales_correlation:
    temp_save = join(behav_results_saving_path, "scales")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)    
    behav_analysis.correlate_two_scales(
        scale_1 = 'SAS',
        scale_2 = 'BDI',
        sub_scale_dict = sub_scale_dict,
        subject_colors = subject_colors,
        saving_path = temp_save,
        save_as = save_as
    )

    behav_analysis.correlate_two_scales(
        scale_1 = 'OCI_TOTAL',
        scale_2 = 'BIS_TOTAL',
        sub_scale_dict = sub_scale_dict,
        subject_colors = subject_colors,
        saving_path = temp_save,
        save_as = save_as
    )

if visualize_updrs:
    temp_save = join(behav_results_saving_path, "scales")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)
    behav_analysis.visualize_updrs_scores(
    sub_scale_dict = sub_scale_dict, 
    subject_colors = subject_colors,
    color_dict = color_dict, 
    colored_by = 'condition',
    saving_path = temp_save, 
    save_as = save_as
    )

if visualize_rt_distribution:
    temp_save = join(behav_results_saving_path, "RT distribution")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)    
    behav_analysis.visualize_rt_distribution(
        stats = stats,
        color_dict = color_dict,
        saving_path = temp_save,
        save_as = save_as
    )

if visualize_rt_distribution_overlapped:
    temp_save = join(behav_results_saving_path, "RT distribution")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)    
    behav_analysis.visualize_rt_distribution_overlapped(
        stats = stats,
        color_dict = color_dict,
        saving_path = temp_save,
        save_as = save_as
    )

if correlate_ssrt_prep_cost:
    temp_save = join(behav_results_saving_path, "correlations")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)    
    behav_analysis.correlate_ssrt_prep_cost(
        stats = stats,
        subject_colors = subject_colors,
        color_dict = color_dict,
        saving_path = temp_save,
        save_as = save_as
    )

if correlate_ssrt_bis:
    temp_save = join(behav_results_saving_path, "correlations")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)    
    behav_analysis.correlate_ssrt_bis(
        stats = stats,
        sub_scale_dict = sub_scale_dict,
        subject_colors = subject_colors,
        bis_scale_name = 'BIS_TOTAL',
        color_dict = color_dict,
        saving_path = temp_save,
        save_as = save_as
    )
    behav_analysis.correlate_ssrt_bis(
        stats = stats,
        sub_scale_dict = sub_scale_dict,
        subject_colors = subject_colors,
        bis_scale_name = 'BIS_nonplanning',
        color_dict = color_dict,
        saving_path = temp_save,
        save_as = save_as
    )    
    behav_analysis.correlate_ssrt_bis(
        stats = stats,
        sub_scale_dict = sub_scale_dict,
        subject_colors = subject_colors,
        bis_scale_name = 'BIS_motor',
        color_dict = color_dict,
        saving_path = temp_save,
        save_as = save_as
    )    
    behav_analysis.correlate_ssrt_bis(
        stats = stats,
        sub_scale_dict = sub_scale_dict,
        subject_colors = subject_colors,
        bis_scale_name = 'BIS_attentional',
        color_dict = color_dict,
        saving_path = temp_save,
        save_as = save_as
    )        

## TO-DO : make function more flexible to accept one scale 
# and one task score e.g. to correlate proactive inhibition
# with BIS non planning

if plot_prep_cost:
    variable_of_interest = 'Preparation cost (ms)'
    temp_save = join(behav_results_saving_path, "Response inhibition")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)    
    behav_analysis.plot_variable_of_interest(
        stats = stats,
        color_dict = color_dict,
        subject_colors = subject_colors,
        variable_of_interest = variable_of_interest,
        colored_by = 'subject',
        saving_path = temp_save,
        save_as = save_as
    )

if plot_ssrt:
    variable_of_interest = 'SSRT (ms)'
    temp_save = join(behav_results_saving_path, "Response inhibition")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)    
    behav_analysis.plot_variable_of_interest(
        stats = stats,
        color_dict = color_dict,
        subject_colors = subject_colors,
        variable_of_interest = variable_of_interest,
        colored_by = 'subject',
        saving_path = temp_save,
        save_as = save_as
    )    