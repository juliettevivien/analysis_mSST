# load librairies
import os
from os.path import join
import json


from functions import utils, io, hrm

# pick format to save figures : png for quick visualization, pdf for illustrator
save_as = 'png'
color_dict = {
    'DBS OFF': '#20a39e', 
    'DBS ON': '#ef5b5b', 
    'control': '#ffba49', 
    'preop': '#8E7DBE'
    }

# pick tests to run:
single_sess_indep = True
group_level_indep = True
single_sub_rt_ssd_relation = True
group_rt_ssd_relation = True
check_success_rate = True

# load subjects
included_subjects = [
    #'preop001 mSST',  # preop group hasn't really be recorded, let's discard for now

    # 'C001 mSST', 'C002 mSST', # pilot participants
    'C003 mSST', 'C004 mSST', 'C006 mSST', 
    'C007 mSST','C008 mSST','C009 mSST', 
    'C010 mSST', 'C011 mSST', 'C012 mSST',
    'sub006 DBS ON mSST', 'sub006 DBS OFF mSST', 
    #'sub007 DBS OFF mSST', # no ON session
    'sub008 DBS ON mSST', 'sub008 DBS OFF mSST', 
    'sub009 DBS ON mSST', 'sub009 DBS OFF mSST', 
    'sub011 DBS OFF mSST', 'sub011 DBS ON mSST', 
    #'sub012 DBS ON mSST', # no OFF session, file crashed
    #'sub013 DBS OFF mSST', # no ON session
    #'sub014 DBS ON mSST', # no OFF session
    'sub015 DBS OFF mSST', 'sub015 DBS ON mSST', 
    'sub017 DBS ON mSST',  'sub017 DBS OFF mSST', 
    'sub019 DBS ON mSST', 'sub019 DBS OFF mSST', 
    #'sub020 DBS ON mSST', # no OFF session
    'sub021 DBS OFF mSST', 'sub021 DBS ON mSST', 
    #'sub022 DBS ON mSST', # no OFF session
    'sub023 DBS OFF mSST', 'sub023 DBS ON mSST',
    'sub024 DBS ON mSST', 'sub024 DBS OFF mSST',
    'sub025 DBS ON mSST', 'sub025 DBS OFF mSST',
    'sub027 DBS ON mSST', 'sub027 DBS OFF mSST',
    'sub028 DBS OFF mSST', 'sub028 DBS ON mSST',
    'sub029 DBS ON mSST', 'sub029 DBS OFF mSST',
    #'sub030 DBS ON mSST', 'sub030 DBS OFF mSST', # didn't finish OFF session, symptoms too strong
    #'sub031 DBS OFF mSST', # didn't finish ON session, too tired, didn't want to continue
    'sub032 DBS ON mSST', 'sub032 DBS OFF mSST',
    'sub033 DBS ON mSST', 'sub033 DBS OFF mSST'
    ]


# set paths
onedrive_path = utils._get_onedrive_path()
working_path = os.path.dirname(os.getcwd())
results_path = join(working_path, "analysis_mSST", "results")
behav_results_saving_path = join(results_path, "behav_results", "hrm_assumptions_checking")
if not os.path.isdir(behav_results_saving_path):
    os.makedirs(behav_results_saving_path)

# prepare a list to store subjects that should be excluded
excluded_subjects = []

# load behavioral data (csv files)
data = io.load_behav_data(included_subjects, onedrive_path)

# extract main statistics
stats = utils.extract_stats(data)

# store stats in a json file
# If no file was found, create a new JSON file
filename = "stats.json"
file_path = os.path.join(results_path, filename)
if not os.path.isfile(file_path):
   with open(file_path, "w", encoding="utf-8") as file:
           json.dump({}, file, indent=4)

# Save the updated or new JSON file
with open(file_path, "w", encoding="utf-8") as file:
    json.dump(stats, file, indent=4)

# # create sub-dictionnaries of stats for each category:
# # Initialize empty dictionaries
# stats_OFF = {}
# stats_ON = {}
# stats_CONTROL = {}
# stats_PREOP = {}

# # Loop through the original dictionary and filter into sub-dictionaries
# for key, value in stats.items():
#     if "OFF" in key:
#         stats_OFF[key] = value
#     elif "ON" in key:
#         stats_ON[key] = value
#     elif "C" in key:
#         stats_CONTROL[key] = value
#     elif "preop" in key:
#         stats_PREOP[key] = value

######## TEST 1: independence assumption ###########
# 1a: single session level #
if single_sess_indep:
    print("Testing independence assumption at the single session level...")
    temp_save = join(behav_results_saving_path, "independence single sub")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)
    excluded_subjects = hrm.check_independence_assumption_single_sub(
        stats = stats,
        excluded_subjects = excluded_subjects,
        color_dict = color_dict,
        save_as = save_as,
        saving_path = temp_save
    )

# 1b: group level (test replicated from Hervault et al., 2021) #
if group_level_indep:
    print("Testing independence assumption at the group level...")
    temp_save = join(behav_results_saving_path, "independence group level")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)
    excluded_subjects = hrm.check_independence_assumption_group_level(
        stats = stats,    
        excluded_subjects = excluded_subjects,
        save_as = save_as,
        saving_path = temp_save
    )

# 1c: If (u)GS RT with small SSD > (u)GS RT with long SSD, 
# the independence assumption is violated #
if single_sub_rt_ssd_relation:
    print("Testing RT-SSD relation at the single subject level...")
    temp_save = join(behav_results_saving_path, "single sub RT-SSD relation")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)
    excluded_subjects = hrm.check_independence_assumption_rt_ssd_relation(
        stats = stats, 
        excluded_subjects = excluded_subjects,
        color_dict = color_dict,   
        save_as = save_as,
        saving_path = temp_save
    )

# 1d : a prediction of the model is that RT of unsuccessful stop 
# trials should lengthen with increasing stop signal delay (ssd),
# because gradually more movements fail to be inhibited as the stop
# qignal is delayed.
# replication from Hervault et al., 2021
if group_rt_ssd_relation:
    print("Testing RT-SSD relation at the group level...")
    temp_save = join(behav_results_saving_path, "group RT-SSD relation")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)
    hrm.check_independence_assumption_rt_ssd_relation_group_level(
        stats = stats, 
        excluded_subjects = excluded_subjects,
        save_as = save_as,
        saving_path = temp_save
    ) 

######## TEST 2: success rate ###########    
# 2a: Cut-offs:
#    - Success rate on GO trials should be >70%
#    - Success rate on Stop trials should be comprised between 35-65% (see Ray et al 2012 - Charlotte sent on Teams)

# Here, percent correct trial for each trial type is calculated after removing 
# early presses (therefore, it only takes into account trials during which the 
# cue was actally presented).


if check_success_rate:
    print("Testing success rate...")
    temp_save = join(behav_results_saving_path, "success rate")
    if not os.path.isdir(temp_save):
        os.makedirs(temp_save)
    excluded_subjects = hrm.check_success_rate(
        stats = stats,
        excluded_subjects = excluded_subjects,
        color_dict = color_dict,
        save_as = save_as,
        saving_path = temp_save
    )    

# save excluded subjects in a txt file and JSON file
with open(join(behav_results_saving_path, "excluded_subjects.txt"), "w") as f:
    for sub in excluded_subjects:
        f.write(sub + "\n")
with open(join(results_path, 'final_included_subjects.json'), 'w') as f:
    json.dump(included_subjects, f, indent=4)   

# save remaining included_subjects in a txt file
included_subjects = [sub for sub in included_subjects if sub not in excluded_subjects]
with open(join(behav_results_saving_path, "included_subjects.txt"), "w") as f:
    for sub in included_subjects:
        f.write(sub + "\n")      

