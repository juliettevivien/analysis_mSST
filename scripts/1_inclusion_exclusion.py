# load librairies
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from os.path import join
import seaborn as sns
import scipy
import json
from collections import defaultdict

from functions import utils


included_subjects = [
    'preop001 mSST', 

    'C002 mSST', 'C003 mSST', 'C004 mSST', 'C006 mSST', 
    
    'sub006 DBS ON mSST', 'sub006 DBS OFF mSST', 
    'sub007 DBS OFF mSST', 
    'sub008 DBS ON mSST', 'sub008 DBS OFF mSST', 
    'sub009 DBS ON mSST', 'sub009 DBS OFF mSST', 
    'sub011 DBS OFF mSST', 'sub011 DBS ON mSST', 
    'sub012 DBS ON mSST', 
    'sub013 DBS OFF mSST', 
    'sub014 DBS ON mSST', 
    'sub015 DBS OFF mSST', 'sub015 DBS ON mSST', 
    'sub017 DBS ON mSST',  'sub017 DBS OFF mSST', 
    'sub019 DBS ON mSST', 'sub019 DBS OFF mSST', 
    'sub020 DBS ON mSST', 
    'sub021 DBS OFF mSST', 'sub021 DBS ON mSST', 
    'sub022 DBS ON mSST', 
    'sub023 DBS OFF mSST', 'sub023 DBS ON mSST',
    'sub024 DBS ON mSST'
    ]
excluded_subjects = []

onedrive_path = utils._get_onedrive_path()
working_path = os.path.dirname(os.getcwd())
results_path = join(working_path, "mSST_analysis", "results")
behav_results_saving_path = join(results_path, "behav_results")
if not os.path.isdir(behav_results_saving_path):
    os.makedirs(behav_results_saving_path)

# load data by looping through the subjects
data = {}
for subject in included_subjects:
    if subject.startswith('sub'):
        subject_ID, na, condition, task = subject.split(' ')
        sub_onedrive_path = join(onedrive_path, subject_ID)
        sub_onedrive_path_raw = join(sub_onedrive_path, 'raw_data')
        sub_onedrive_path_behav = join(sub_onedrive_path_raw, 'BEHAVIOR')
        sub_onedrive_path_condition = join(sub_onedrive_path_behav, (na + ' ' + condition))
        sub_onedrive_path_task = join(sub_onedrive_path_condition, task)
        behav_filename = [f for f in os.listdir(sub_onedrive_path_task) if f.endswith('.csv')]
        filepath = join(sub_onedrive_path_task, behav_filename[0])
        df = pd.read_csv(filepath)
        
    elif subject.startswith('C'):
        subject_ID, task = subject.split(' ')
        sub_onedrive_path = join(onedrive_path, subject_ID)
        sub_onedrive_path_raw = join(sub_onedrive_path, 'raw_data')
        sub_onedrive_path_behav = join(sub_onedrive_path_raw, 'BEHAVIOR')
        sub_onedrive_path_task = join(sub_onedrive_path_behav, task)
        behav_filename = [f for f in os.listdir(sub_onedrive_path_task) if f.endswith('.csv')]
        filepath = join(sub_onedrive_path_task, behav_filename[0])
        df = pd.read_csv(filepath)
    
    elif subject.startswith('preop'):
        subject_ID, task = subject.split(' ')
        sub_onedrive_path = join(onedrive_path, subject_ID)
        sub_onedrive_path_raw = join(sub_onedrive_path, 'raw_data')
        sub_onedrive_path_behav = join(sub_onedrive_path_raw, 'BEHAVIOR')
        sub_onedrive_path_behav_preop = join(sub_onedrive_path_behav, 'PRE OP')
        sub_onedrive_path_task = join(sub_onedrive_path_behav_preop, task)
        behav_filename = [f for f in os.listdir(sub_onedrive_path_task) if f.endswith('.csv')]
        filepath = join(sub_onedrive_path_task, behav_filename[0])
        df = pd.read_csv(filepath)        

    data[subject] = df


color_dict = {
'dbs_off': '#20a39e', 
'dbs_on': '#ef5b5b', 
'control': '#ffba49', 
'preop': '#8E7DBE'
}

stats = utils.extract_stats(data)

results_dict = {}
# create subdictionnaries for each subject:
for subject in included_subjects:
    results_dict[subject] = {}

# If no file was found, create a new JSON file
filename = "stats.json"
file_path = os.path.join(results_path, filename)
#if not os.path.isfile(file_path):
#    with open(file_path, "w", encoding="utf-8") as file:
#            json.dump({}, file, indent=4)

# Save the updated or new JSON file
with open(file_path, "w", encoding="utf-8") as file:
    json.dump(stats, file, indent=4)

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

# # Define trial types to include only Go trials and Stop trials
# trial_types = ['go_trial', 'stop_trial']
# p_value_dict = {}

# # Loop through the filtered dictionaries (e.g., stats_OFF, stats_ON, etc.)
# for condition, condition_stats in [('dbs_off', stats_OFF), 
#                                    ('dbs_on', stats_ON), 
#                                    ('control', stats_CONTROL), 
#                                    ('preop', stats_PREOP)]:
#     for subject_id, subject_data in condition_stats.items():
#         # Gather data for the selected trial types into a DataFrame
#         data = []
#         for trial_type in trial_types:
#             if f"{trial_type} RTs (ms)" in subject_data:  # Ensure the trial type key exists
#                 data.extend([(trial_type, val) for val in subject_data[f"{trial_type} RTs (ms)"]])

#         df = pd.DataFrame(data, columns=['Trial Type', 'Reaction Time'])

#         # Extract reaction times for statistical comparison
#         go_data = df[df['Trial Type'] == 'go_trial']['Reaction Time']
#         stop_data = df[df['Trial Type'] == 'stop_trial']['Reaction Time']

#         # Test for normality:
#         normal_go = scipy.stats.shapiro(go_data)[1] > 0.05 if len(go_data) >= 3 else False
#         normal_stop = scipy.stats.shapiro(stop_data)[1] > 0.05 if len(stop_data) >= 3 else False

#         # Perform Independent t-test if normality is met; otherwise, fallback to Mann-Whitney
#         if len(go_data) >= 3 and len(stop_data) >= 3:
#             if normal_go and normal_stop:
#                 stat, p_value = scipy.stats.ttest_ind(stop_data, go_data, equal_var=False, alternative = 'greater')  # Welch's t-test
#                 test = 't-test'
#             else:
#                 stat, p_value = scipy.stats.mannwhitneyu(stop_data, go_data, alternative='greater')
#                 test = 'Mann-Whitney U test'
#         else:
#             p_value = None  # Not enough data for test

#         p_value_dict[f"{subject_id}_{test}"] = p_value
#         results_dict[subject_id]['independance test used'] = test
#         results_dict[subject_id]['p-value independance'] = p_value

#         # Initialize the plot
#         plt.figure(figsize=(8, 6))

#         # Plot violins for the single condition using the custom palette
#         sns.violinplot(data=df, x='Trial Type', y='Reaction Time', 
#                        inner=None, color=color_dict[condition], alpha=0.2)  # Use condition color

#         # Overlay data points with a strip plot
#         sns.stripplot(data=df, x='Trial Type', y='Reaction Time', 
#                       jitter=True, color=color_dict[condition], marker='o', size=4)

#         # Calculate and plot means manually for each trial type
#         for i, trial_type in enumerate(trial_types):
#             condition_data = df[df['Trial Type'] == trial_type]
#             mean_value = condition_data['Reaction Time'].mean()
#             plt.scatter(x=i, y=mean_value, color='black', marker="_", s=200, label='Mean' if i == 0 else "", zorder=5)

#         # Annotate statistical results
#         if p_value is not None:
#             annotation = "*" if p_value <= 0.1 else "ns"
#             max_val = df['Reaction Time'].max()
#             plt.text(0.5, max_val + 10, f"p = {p_value:.3f}\n{annotation}\n{test}", 
#                      ha='center', va='bottom', color='black', fontsize=12)

#         # Customize plot
#         plt.xlabel('Trial Type')
#         plt.ylabel('Reaction Time (ms)')
#         plt.title(f'Reaction Times for Subject {subject_id} ({condition.upper()})')
#         plt.tight_layout()
#         plt.savefig(join(behav_results_saving_path, f"test_independance_{subject_id}_{condition}.png"))
#         plt.close()

# # Convert dictionary to DataFrame
# df_p_value_dict = pd.DataFrame(list(p_value_dict.items()), columns=['Subject_test', 'P-Value'])

# # Save to Excel
# df_p_value_dict.to_excel(join(behav_results_saving_path,'p_values_independance_test.xlsx'), index=False)

# print("Excel file saved as 'p_values_independance_test.xlsx'.")

# paired t-test for Go and Stop trials accross all subjects
mean_RT_go = []
mean_RT_stop = []
subs = []
for sub in stats.keys():
    subs.append(sub)
    mean_RT_go.append(stats[sub]['go_trial mean RT (ms)'])
    mean_RT_stop.append(stats[sub]['stop_trial mean RT (ms)'])

# Perform the paired t-test
t_stat, p_value = scipy.stats.ttest_rel(mean_RT_stop, mean_RT_go, alternative='less')
print(f"Paired t-test results: t-statistic({len(subs)-1}) = {t_stat}, p-value = {p_value}")

# Convert to NumPy arrays if needed
go = np.array(mean_RT_go)
stop = np.array(mean_RT_stop)

# Calculate difference
diff = go - stop
print(diff)
# Look for negative values in diff and print the indexes
negative_indices = np.where(diff < 0)[0]
if len(negative_indices) > 0:
    print(f"Negative differences found at indices: {negative_indices}, corresponding subjects: {[subs[i] for i in negative_indices]}")
    excluded_subjects = [subs[i] for i in negative_indices]
print(f"Excluded subjects: {excluded_subjects}")
mean_diff = np.mean(diff)
sd_diff = np.std(diff, ddof=1)

# Cohen's d
cohens_d = mean_diff / sd_diff
print(f"Cohen's d = {cohens_d:.2f}")

# Determine significance level
if p_value < 0.001:
    stars = '***'
elif p_value < 0.01:
    stars = '**'
elif p_value < 0.05:
    stars = '*'
else:
    stars = 'ns'  # not significant

# Prepare long-format DataFrame
df_long = pd.DataFrame({
    'Subject': subs * 2,
    'Trial Type': ['GO'] * len(subs) + ['Failed-STOP'] * len(subs),
    'Mean RT (ms)': mean_RT_go + mean_RT_stop
})

plt.figure(figsize=(6, 10))

# First: draw the boxplot for GO and STOP trials
sns.boxplot(
    data=df_long,
    x='Trial Type',
    y='Mean RT (ms)',
    color='lightgray',
    fliersize=0  # hide individual outlier dots from the boxplot
)

# Then: overlay individual subject lines
palette = sns.color_palette('tab20', n_colors=len(subs))
subject_colors = dict(zip(subs, palette))

for subject in subs:
    go_rt = df_long[(df_long['Subject'] == subject) & (df_long['Trial Type'] == 'GO')]['Mean RT (ms)'].values[0]
    stop_rt = df_long[(df_long['Subject'] == subject) & (df_long['Trial Type'] == 'Failed-STOP')]['Mean RT (ms)'].values[0]
    plt.plot(['GO', 'Failed-STOP'], [go_rt, stop_rt], marker='o', color=subject_colors[subject], label=subject)

# Add the significance line and stars
y_max = df_long['Mean RT (ms)'].max()
line_height = y_max + 20
text_height = y_max + 30
plt.plot([0, 1], [line_height, line_height], color='black', linewidth=1.5)
plt.text(0.5, text_height, stars, ha='center', va='bottom', fontsize=14)
plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Mean RTs for GO vs Failed-STOP Trials per Subject \n p = {:.3f}, Cohen\'s d = {:.2f}'.format(p_value, cohens_d))
plt.ylabel('Mean Reaction Time (ms)')
plt.xlabel('')
plt.tight_layout()
plt.savefig(join(behav_results_saving_path, 'GO_vs_Failed_STOP_Trials_all_Subjects.png'))
plt.close()


from scipy.stats import ttest_1samp

ssds = []
rt_failed_stop = []
subs = []
for sub in stats.keys():
    ssds.append(stats[sub]['unsuccessful stop SSD (ms)'])
    rt_failed_stop.append(stats[sub]['stop_trial RTs (ms)'])
    subs.append(sub)

# Create a DataFrame for the SSD and RT data
df_ssd_rt = pd.DataFrame({
    'Subject': subs,
    'SSD (ms)': ssds,
    'RT Failed Stop (ms)': rt_failed_stop
})

long_data = []

for idx, row in df_ssd_rt.iterrows():
    subject = row['Subject']
    ssds = row['SSD (ms)']
    rts = row['RT Failed Stop (ms)']
    
    for ssd, rt in zip(ssds, rts):
        long_data.append({'Subject': subject, 'SSD': ssd, 'RT': rt})

df_long = pd.DataFrame(long_data)

# Step 2: Bin SSDs into 50ms bins
bin_size = 50
df_long['SSD_bin'] = (df_long['SSD'] // bin_size) * bin_size  # e.g., 0–49 => 0, 50–99 => 50, etc.

# Step 3: Compute average RT per subject per bin
mean_rt_per_sub_bin = df_long.groupby(['Subject', 'SSD_bin'])['RT'].mean().reset_index()

# Step 4: Compute average RT across subjects per SSD bin
mean_rt_per_bin = mean_rt_per_sub_bin.groupby('SSD_bin')['RT'].mean().sort_index()

# Step 5: Compute differences between consecutive SSD bins
ssd_bins = mean_rt_per_bin.index.to_list()
rt_means = mean_rt_per_bin.values
rt_diffs = np.diff(rt_means)

# Step 6: One-sided t-test: Are the differences significantly > 0?
t_stat, p_value = ttest_1samp(rt_diffs, popmean=0, alternative='greater')
df = len(rt_diffs) - 1
print(f"t({df}) = {t_stat:.3f}, p = {p_value:.4f}")
# Output the results
print("SSD bins (ms):", ssd_bins)
print("Mean RT per bin (ms):", rt_means)
print("RT differences between bins:", rt_diffs)
print(f"One-sided t-test result: t = {t_stat:.3f}, p = {p_value:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(ssd_bins, rt_means, marker='o', linestyle='-')
plt.xlabel('SSD (ms)')
plt.ylabel('Mean RT (Failed STOP Trials)')
plt.title('RTs on failed STOP trials increase with SSD \n t = {:.3f}, p = {:.4f}'.format(t_stat, p_value))
plt.savefig(join(behav_results_saving_path, 'RTs_on_failed_STOP_trials_increase_with_SSD.png'))
plt.close()



# separate the stats dictionary into 4 dictionaries: one for the DBS OFF condition, one for the DBS ON condition, one for the control group and one for the preop group:
stats_dbs_off = {}
stats_dbs_on = {}
stats_control = {}
stats_preop = {}

for subject in stats:
    if subject.startswith('sub'):
        if 'OFF' in subject:
            stats_dbs_off[subject] = stats[subject]
        elif 'ON' in subject:
            stats_dbs_on[subject] = stats[subject]
    elif subject.startswith('C'):
        stats_control[subject] = stats[subject]
    elif subject.startswith('preop'):
        stats_preop[subject] = stats[subject]


# Create plots to compare groups:
# Plot the percentage of correct trials for each trial type across the 3 groups:
trial_types = ['go_trial', 'stop_trial', 'go_fast_trial', 'go_continue_trial']
bar_width = 0.2
index = np.arange(len(trial_types))
opacity = 0.8

# Define conditions and corresponding dictionaries
conditions = {
    'control': stats_control,
    'dbs_off': stats_dbs_off,
    'dbs_on': stats_dbs_on, 
    'preop': stats_preop
}

# Initialize dictionaries to hold results for each condition
results = {condition: [] for condition in conditions.keys()}

# Loop through each condition and subject
for condition, data_dict in conditions.items():
    for subject_id, metrics in data_dict.items():
        # Retrieve the required metrics and store them in the result dictionary
        results[condition].append([
            metrics['percent correct go_trial'],
            metrics['percent correct stop_trial'],
            metrics['percent correct go_fast_trial'],
            metrics['percent correct go_continue_trial']
        ])

# Access data for each condition
control = results['control']
dbs_off = results['dbs_off']
dbs_on = results['dbs_on']
preop = results['preop']

# Number of subjects in each condition
n_control = len(control)
n_dbs_off = len(dbs_off)
n_dbs_on = len(dbs_on)
n_preop = len(preop)

# Calculate means and standard deviations for each condition
control_means = np.mean(control, axis=0)
control_std = np.std(control, axis=0)

dbs_off_means = np.mean(dbs_off, axis=0)
dbs_off_std = np.std(dbs_off, axis=0)

dbs_on_means = np.mean(dbs_on, axis=0)
dbs_on_std = np.std(dbs_on, axis=0)

preop_means = np.mean(preop, axis = 0)
preop_std = np.std(preop, axis = 0)

# Update error bars with standard deviations
error_control = control_std
error_dbs_off = dbs_off_std
error_dbs_on = dbs_on_std
error_preop = preop_std

# Define plotting parameters
index = np.arange(len(control_means))  # Assuming each metric is a separate bar
bar_width = 0.2
opacity = 0.8

# Plot bars with error bars
plt.figure(figsize=(10, 6))

bar1 = plt.bar(index, control_means, bar_width, alpha=opacity, color='#ffba49', label='Control',
               yerr=error_control, capsize=5)
bar2 = plt.bar(index + bar_width, dbs_off_means, bar_width, alpha=opacity, color='#20a39e', label='DBS OFF',
               yerr=error_dbs_off, capsize=5)
bar3 = plt.bar(index + 2 * bar_width, dbs_on_means, bar_width, alpha=opacity, color='#ef5b5b', label='DBS ON',
               yerr=error_dbs_on, capsize=5)
bar4 = plt.bar(index + 3 * bar_width, preop_means, bar_width, alpha=opacity, color='#8E7DBE', label='Pre-OP',
               yerr=error_preop, capsize=5)

# Add subject count annotations on top of each bar
for i, (mean, std) in enumerate(zip(control_means, control_std)):
    plt.text(i, mean + std + 1, f'n={n_control}', ha='center', va='bottom', color='black')

for i, (mean, std) in enumerate(zip(dbs_off_means, dbs_off_std)):
    plt.text(i + bar_width, mean + std + 1, f'n={n_dbs_off}', ha='center', va='bottom', color='black')

for i, (mean, std) in enumerate(zip(dbs_on_means, dbs_on_std)):
    plt.text(i + 2 * bar_width, mean + std + 1, f'n={n_dbs_on}', ha='center', va='bottom', color='black')

for i, (mean, std) in enumerate(zip(preop_means, preop_std)):
    plt.text(i + 3 * bar_width, mean + std + 1, f'n={n_preop}', ha='center', va='bottom', color='black')


# Add labels, title, and legend
plt.xlabel('Trial type')
plt.ylabel('Percent Correct')
plt.title('Performance Across Conditions')
plt.xticks(index + bar_width, ['Go Trial', 'Stop Trial', 'Go Fast Trial', 'Go Continue Trial'])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(join(behav_results_saving_path, f"Performance Across Conditions - Group level (all subjects).png"))
plt.close()



single_subject = {}
for subject in stats.keys():
    # Retrieve values for each trial type in both conditions
    values = [
        stats[subject]['percent correct go_trial'],
        stats[subject]['percent correct stop_trial'],
        stats[subject]['percent correct go_fast_trial'],
        stats[subject]['percent correct go_continue_trial']
    ]
    single_subject[subject] = values

# Define trial types
trial_types = ['go_trial', 'stop_trial', 'go_fast_trial', 'go_continue_trial']

# List to store subject data
data_list = []

# Loop through subjects in the dictionary
for subject_id, values in single_subject.items():
    
    # Identify condition
    if subject_id.startswith('C'):
        condition = 'control'
    elif subject_id.startswith('preop'):
        condition = 'preop'
    elif 'OFF' in subject_id:
        condition = 'dbs_off'
    elif 'ON' in subject_id:
        condition = 'dbs_on'
    
    # Append a dictionary for each subject
    data_list.append({
        'Subject': subject_id,
        'Condition': condition,
        'go_trial': values[0],
        'stop_trial': values[1],
        'go_fast_trial': values[2],
        'go_continue_trial': values[3]
    })

    if values[0] < 75:
        print(f"Warning: {subject_id} has a go_trial success rate below 75%: {values[0]}%")
        excluded_subjects.append(subject_id)
    if values[1] > 65:
        print(f"Warning: {subject_id} has a stop_trial success rate above 65%: {values[1]}%")
        excluded_subjects.append(subject_id)

# Convert to DataFrame
df_summary = pd.DataFrame(data_list)

# Save to Excel
df_summary.to_excel(join(behav_results_saving_path,'summary_success_rates.xlsx'), index=False)

# Define trial types and bar width
trial_types = ['go_trial', 'stop_trial', 'go_fast_trial', 'go_continue_trial']
bar_width = 0.3
index = np.arange(len(trial_types))
opacity = 0.8

# Access data for a single subject 
for subject_id in single_subject.keys():
    # Retrieve values for each trial type

    if subject_id.startswith('C'):
        condition = 'control'
    elif subject_id.startswith('preop'):
        condition = 'preop'
    elif 'OFF' in subject_id:
        condition = 'dbs_off'
    elif 'ON' in subject_id:
        condition = 'dbs_on'

    values = single_subject[subject_id]
    results_dict[subject_id]['go success rate'] = values[0]
    results_dict[subject_id]['stop success rate'] = values[1]
    results_dict[subject_id]['go fast success rate'] = values[2]
    results_dict[subject_id]['go continue success rate'] = values[3]

    # Plot bars for each condition
    plt.figure(figsize=(10, 6))

    plt.bar(index, values, bar_width, alpha=opacity, color=color_dict[condition], label='control', capsize=5)

    for i, (val) in enumerate(values):
        plt.text(x=i, y=val + 1, s=f'{val:.1f}%', ha='center', va='bottom', color='black')

    # Add labels, title, and legend
    plt.xlabel('Trial Type')
    plt.ylabel('Percent Correct')
    plt.title(f'Performance for Subject {subject_id}')
    plt.xticks(index, ['Go Trial', 'Stop Trial', 'Go Fast Trial', 'Go Continue Trial'])
    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, f"Performance for {subject_id}.png"))
    plt.close()

excluded_subjects = list(set(excluded_subjects))  # Remove duplicates

# Step 1: Organize sessions by subject ID
subject_sessions = defaultdict(list)
for subj in included_subjects:
    if subj.startswith("sub"):
        subj_id = subj.split()[0]
        subject_sessions[subj_id].append(subj)

# Step 2: Normalize excluded list by subject ID
excluded_ids = set(s.split()[0] for s in excluded_subjects)

# Step 3: Update excluded list to include both sessions (ON and OFF) or all if only one
final_excluded_subjects = set()
for subj_id in excluded_ids:
    sessions = subject_sessions.get(subj_id, [])
    final_excluded_subjects.update(sessions)

# Step 4: Also exclude subjects with only one session
# for subj_id, sessions in subject_sessions.items():
#     if len(sessions) == 1:
#         final_excluded_subjects.update(sessions)

# Optional: Convert to sorted list for readability
final_excluded_subjects = sorted(final_excluded_subjects)
final_excluded_subjects = final_excluded_subjects + excluded_subjects
final_included_subjects = [s for s in included_subjects if s not in final_excluded_subjects]
final_included_subjects
# Save the final included subjects to a JSON file
with open(join(behav_results_saving_path, 'final_included_subjects.json'), 'w') as f:
    json.dump(final_included_subjects, f, indent=4)


# # Convert to DataFrame
# df = pd.DataFrame.from_dict(results_dict, orient='index')

# # Reset index and rename the first column to 'subject'
# df.reset_index(inplace=True)
# df.rename(columns={'index': 'subject'}, inplace=True)

# # Compute mean and standard deviation of "go success rate"
# mean_go_success = df["go success rate"].mean()
# std_go_success = df["go success rate"].std()
# #threshold = mean_go_success - std_go_success
# threshold = 75


# # Apply the inclusion criteria
# df["included"] = df.apply(lambda row: "Yes" if (row["p-value independance"] > 0.1) and 
#                                                (row["go success rate"] > threshold) else "No", axis=1)

# # Save to an Excel file
# df.to_excel(join(results_path,"Included or excluded subjects.xlsx"), index=False)

# print(f"Results saved to {results_path}")

