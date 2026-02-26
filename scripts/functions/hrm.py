from unicodedata import name

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from os.path import join


def check_independence_assumption_single_sub(
    stats,
    excluded_subjects,
    color_dict,
    save_as,
    saving_path
):
    # Define trial types to include only Go trials and Stop trials
    trial_types = ['go_trial', 'stop_trial']
    p_value_dict = {}
    normal_go_count = 0
    normal_stop_count = 0

    results_dict = {}
    # create subdictionnaries for each subject:
    for subject in stats.keys():
        results_dict[subject] = {}

    # Loop through the filtered dictionaries (e.g., stats_OFF, stats_ON, etc.)
    # for condition, condition_stats in [('dbs_off', stats_OFF), 
    #                                 ('dbs_on', stats_ON), 
    #                                 ('control', stats_CONTROL), 
    #                                 ('preop', stats_PREOP)]:
        # for subject_id, subject_data in condition_stats.items():
    for subject_id, subject_data in stats.items():    
        # Gather data for the selected trial types into a DataFrame
        data = []
        if subject_id.startswith('C'):
            condition = 'control'
        elif subject_id.startswith('preop'):
            condition = 'preop'
        else:
            if 'ON' in subject_id:
                condition = 'dbs_on'
            else:
                condition = 'dbs_off'
        for trial_type in trial_types:
            if f"{trial_type} RTs (ms)" in subject_data:  # Ensure the trial type key exists
                data.extend([(trial_type, val) for val in subject_data[f"{trial_type} RTs (ms)"]])

        df = pd.DataFrame(data, columns=['Trial Type', 'Reaction Time'])

        # Extract reaction times for statistical comparison
        go_data = df[df['Trial Type'] == 'go_trial']['Reaction Time']
        stop_data = df[df['Trial Type'] == 'stop_trial']['Reaction Time']

        # Test for normality:
        normal_go = scipy.stats.shapiro(go_data)[1] > 0.05 if len(go_data) >= 3 else False
        normal_go_count += 1 if normal_go else 0
        normal_stop = scipy.stats.shapiro(stop_data)[1] > 0.05 if len(stop_data) >= 3 else False
        normal_stop_count += 1 if normal_stop else 0

        # Perform Independent t-test if normality is met; otherwise, fallback to Mann-Whitney
        if len(go_data) >= 3 and len(stop_data) >= 3:
            if normal_go and normal_stop:
                stat, p_value = scipy.stats.ttest_ind(stop_data, go_data, equal_var=False, alternative = 'greater')  # Welch's t-test
                test = 't-test'
            else:
                stat, p_value = scipy.stats.mannwhitneyu(stop_data, go_data, alternative='greater')
                test = 'Mann-Whitney U test'
        else:
            p_value = None  # Not enough data for test

        p_value_dict[f"{subject_id}_{test}"] = p_value
        results_dict[subject_id]['independance test used'] = test
        results_dict[subject_id]['p-value independance'] = p_value

        if p_value < 0.05:
            excluded_subjects.append(subject_id)
        
        # if another session is available for the excluded subjects, exclude these as well:
        for excluded_sub in excluded_subjects:
            base_sub_id = excluded_sub.split(' ')[0]  # get the base subject ID without condition
            for sub in stats.keys():
                if sub.startswith(base_sub_id) and sub not in excluded_subjects:
                    excluded_subjects.append(sub)                

        # Initialize the plot
        plt.figure(figsize=(8, 6))

        # Plot violins for the single condition using the custom palette
        sns.violinplot(data=df, x='Trial Type', y='Reaction Time', 
                    inner=None, color=color_dict[condition], alpha=0.2)  # Use condition color

        # Overlay data points with a strip plot
        sns.stripplot(data=df, x='Trial Type', y='Reaction Time', 
                    jitter=True, color=color_dict[condition], marker='o', size=4)

        # Calculate and plot means manually for each trial type
        for i, trial_type in enumerate(trial_types):
            condition_data = df[df['Trial Type'] == trial_type]
            mean_value = condition_data['Reaction Time'].mean()
            plt.scatter(x=i, y=mean_value, color='black', marker="_", s=200, label='Mean' if i == 0 else "", zorder=5)

        # Annotate statistical results
        if p_value is not None:
            annotation = "*" if p_value <= 0.05 else "ns"
            max_val = df['Reaction Time'].max()
            plt.text(0.5, max_val + 10, f"p = {p_value:.3f}\n{annotation}\n{test}", 
                    ha='center', va='bottom', color='black', fontsize=12)

        # Customize plot
        plt.xlabel('Trial Type')
        plt.ylabel('Reaction Time (ms)')
        plt.title(f'Reaction Times for Subject {subject_id} ({condition.upper()})')
        plt.tight_layout()
        plt.savefig(join(saving_path, f"test_independance_{subject_id}_{condition}.{save_as}"))
        plt.close()

    # Convert dictionary to DataFrame
    df_p_value_dict = pd.DataFrame(list(p_value_dict.items()), columns=['Subject_test', 'P-Value'])

    # Save to Excel
    df_p_value_dict.to_excel(join(saving_path,'p_values_independance_test.xlsx'), index=False)

    print("Excel file saved as 'p_values_independance_test.xlsx'.")

    return excluded_subjects



def check_independence_assumption_group_level(
        stats, 
        excluded_subjects,
        save_as, 
        saving_path
        ):    
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
    with open(join(saving_path, 'results.txt'), 'w') as a:
        a.write(f"Paired t-test results: t-statistic({len(subs)-1}) = {t_stat}, p-value = {p_value}\n")
    # Convert to NumPy arrays if needed
    go = np.array(mean_RT_go)
    stop = np.array(mean_RT_stop)

    # Calculate difference
    diff = go - stop
    # Look for negative values in diff and print the indexes
    negative_indices = np.where(diff < 0)[0]
    if len(negative_indices) > 0:
        print(f"Negative differences found at indices: {negative_indices}, corresponding subjects: {[subs[i] for i in negative_indices]}")
        with open(join(saving_path, 'results.txt'), 'a') as a:
            # a.write('%s: 05' % name)
            a.write(f"Negative differences found at indices: {negative_indices}, corresponding subjects: {[subs[i] for i in negative_indices]}")
        new_excluded_subjects = [subs[i] for i in negative_indices]

    # if another session is available for the excluded subjects, exclude these as well:
    for excluded_sub in new_excluded_subjects:
        base_sub_id = excluded_sub.split(' ')[0]  # get the base subject ID without condition
        for sub in stats.keys():
            if sub.startswith(base_sub_id) and sub not in new_excluded_subjects:
                new_excluded_subjects.append(sub)
    excluded_subjects.extend(new_excluded_subjects)
    excluded_subjects = list(set(excluded_subjects)) # remove duplicates
    print(f"Excluded subjects: {new_excluded_subjects}")

    with open(join(saving_path, 'results.txt'), 'a') as a:
        a.write(f"Excluded subjects: {new_excluded_subjects}")
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)

    # Cohen's d
    cohens_d = mean_diff / sd_diff
    print(f"Cohen's d = {cohens_d:.2f}")
    with open(join(saving_path, 'results.txt'), 'a') as a:
        a.write(f"\nCohen's d = {cohens_d:.2f}")

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
    plt.title('Mean RTs for GO vs Failed-STOP Trials per Subject')
    plt.ylabel('Mean Reaction Time (ms)')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(join(saving_path, f"independence_assumption_group_level.{save_as}"), dpi=300)
    plt.close()

    with open(join(saving_path, 'results.txt'), 'a') as a:
        a.write(f"\n\nOverall, participants had a shorter mean RT in unsuccessful STOP trials than in GO trials (t({len(subs)-1}) = {t_stat}, p-value = {p_value}, d = {cohens_d:.2f}). At the individual level, this didn't hold true for 1 subject ({excluded_subjects}) who was therefore removed from subsequent analysis.")

    return excluded_subjects


def check_independence_assumption_rt_ssd_relation(
        stats,  
        excluded_subjects, 
        color_dict, 
        save_as,
        saving_path
    ):
    p_value_dict_SSD = {}
    new_excluded_subjects = []  

    with open(join(saving_path, 'results.txt'), 'w') as a:
        a.write("Results of the SSD test:\n")

    for subject in stats:
        if subject.startswith('C'):
            condition = 'control'
        elif subject.startswith('preop'):
            condition = 'preop'
        else:
            if 'ON' in subject:
                condition = 'dbs_on'
            else:
                condition = 'dbs_off'
        u_GS_RT = stats[subject]['stop_trial RTs (ms)']
        SSD = stats[subject]['unsuccessful stop SSD (ms)']
        
        # Convert to DataFrame for easier grouping
        data = pd.DataFrame({'SSD': SSD, 'u_GS_RT': u_GS_RT})
        
        # Calculate mean RT for each unique SSD
        mean_values = data.groupby('SSD')['u_GS_RT'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        plt.scatter(SSD, u_GS_RT, label='Individual Data', color='cornflowerblue')
        plt.plot(mean_values['SSD'], mean_values['u_GS_RT'], 'o-', color='orange', label='Mean RT')
        plt.xlabel('SSD (ms)')
        plt.ylabel('Unsuccessful Stop RT (ms)')
        plt.title(f'{subject}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(join(saving_path, f"independence_assumption_RT_SSD_correlation_{subject}.{save_as}"), dpi=300)
        plt.close()

        threshold = np.median(SSD) 
        small_ssd = data[data['SSD'] <= threshold]
        long_ssd = data[data['SSD'] > threshold]

        # Calculate mean RT
        mean_small = small_ssd['u_GS_RT'].mean()
        mean_long = long_ssd['u_GS_RT'].mean()

        # stat, p = scipy.stats.shapiro(small_ssd['u_GS_RT'])
        # if p > 0.05:
        #     print("Data appears normally distributed (p =", p, ")")
        # else:
        #     print("Data does NOT appear normal (p =", p, ")")

        # stat, p = scipy.stats.shapiro(long_ssd['u_GS_RT'])
        # if p > 0.05:
        #     print("Data appears normally distributed (p =", p, ")")
        # else:
        #     print("Data does NOT appear normal (p =", p, ")")    

        # Perform a non-parametric test (mann whitney):
        #u_stat, p_value = ttest_ind(small_ssd['u_GS_RT'], long_ssd['u_GS_RT'], alternative='greater')
        stat, p_value = scipy.stats.mannwhitneyu(small_ssd['u_GS_RT'], long_ssd['u_GS_RT'], alternative='greater')
        p_value_dict_SSD[subject] = p_value

        with open(join(saving_path, 'results.txt'), 'a') as a:
            a.write(f"\n\n{subject} - Mean RT (Small SSD): {mean_small:.2f}, Mean RT (Long SSD): {mean_long:.2f}, Mann-Whitney U test p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            new_excluded_subjects.append(subject)

        plt.figure(figsize=(8, 6))
        sns.boxplot(x=['Small SSD']*len(small_ssd) + ['Long SSD']*len(long_ssd),
                    y=pd.concat([small_ssd['u_GS_RT'], long_ssd['u_GS_RT']]),
                    color=color_dict[condition]
                    )
        # Annotate statistical results
        if p_value is not None:
            annotation = "ns" if p_value >= 0.05 else "*"    
            plt.text(0.5, max(mean_small, mean_long) + 100, f"p = {p_value}\n{annotation}", 
                        ha='center', va='bottom', color='black', fontsize=12)
        plt.ylabel('Unsuccessful Stop RT (ms)')
        plt.title(f'Comparison of RT by SSD - {subject}')
        plt.savefig(join(saving_path, f"independence_assumption_RT_SSD_boxplot_{subject}.{save_as}"), dpi=300)
        plt.close()

    # Convert dictionary to DataFrame
    df_p_value_dict_SSD = pd.DataFrame(list(p_value_dict_SSD.items()), columns=['Subject', 'P-Value'])

    # Save to Excel
    df_p_value_dict_SSD.to_excel(join(saving_path,'p_values_independance_test_SSD.xlsx'), index=False)

    # if another session is available for the excluded subjects, exclude these as well:
    for excluded_sub in new_excluded_subjects:
        base_sub_id = excluded_sub.split(' ')[0]  # get the base subject ID without condition
        for sub in stats.keys():
            if sub.startswith(base_sub_id) and sub not in new_excluded_subjects:
                new_excluded_subjects.append(sub)
    excluded_subjects.extend(new_excluded_subjects)
    excluded_subjects = list(set(excluded_subjects)) # remove duplicates

    return excluded_subjects



def check_independence_assumption_rt_ssd_relation_group_level(
        stats, 
        color_dict,   
        excluded_subjects,
        save_as,
        saving_path
    ):

    ssds = []
    rt_failed_stop = []
    subs = []
    for sub in stats.keys():
        if sub in excluded_subjects:
            continue
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

    # Bin SSDs into 50ms bins
    bin_size = 50
    df_long['SSD_bin'] = (df_long['SSD'] // bin_size) * bin_size  # e.g., 0–49 => 0, 50–99 => 50, etc.

    # Compute average RT per subject per bin
    mean_rt_per_sub_bin = df_long.groupby(['Subject', 'SSD_bin'])['RT'].mean().reset_index()

    # Compute average RT across subjects per SSD bin
    mean_rt_per_bin = mean_rt_per_sub_bin.groupby('SSD_bin')['RT'].mean().sort_index()

    # Get the standard deviation across subjects per SSD bin
    std_rt_per_bin = mean_rt_per_sub_bin.groupby('SSD_bin')['RT'].std(ddof=1).sort_index()

    # Compute differences between consecutive SSD bins
    ssd_bins = mean_rt_per_bin.index.to_list()
    rt_means = mean_rt_per_bin.values
    rt_diffs = np.diff(rt_means)

    # One-sided t-test: Are the differences significantly > 0?
    t_stat, p_value = scipy.stats.ttest_1samp(rt_diffs, popmean=0, alternative='greater')
    df = len(rt_diffs) - 1 # degree of freedom

    plt.figure(figsize=(8, 5))
    plt.errorbar(ssd_bins, rt_means, yerr=std_rt_per_bin.values, fmt='o-', capsize=5)
    plt.xlabel('SSD (ms)')
    plt.ylabel('Mean RT (Failed STOP Trials) ± std')
    plt.title('RTs on failed STOP trials increase with SSD')
    plt.savefig(join(saving_path, f"group_level_independence_assumption_RT_SSD.{save_as}"), dpi=300)
    plt.close()

    with open(join(saving_path, 'results.txt'), 'w') as a:
        a.write(f"\n\nGroup-level RT-SSD relation:\n")
    with open(join(saving_path, 'results.txt'), 'a') as a:
        a.write(f"t({df}) = {t_stat:.3f}, p = {p_value:.4f}\n")
        a.write(f"SSD bins (ms): {ssd_bins}\n")
        a.write(f"Mean RT per bin (ms): {rt_means}\n")
        a.write(f"RT differences between bins: {rt_diffs}\n")
        a.write(f"The differential values were significantly positive (t({df}) =  {t_stat:.3f}, p = {p_value:.4f}) which confirms the prediction of the model, that the RT in unsuccessful STOP trials should lengthen with increasing Stop signal delay (SSD).")


def check_success_rate(
    stats, 
    excluded_subjects, 
    color_dict,
    save_as,
    saving_path
):
    new_excluded_subjects = []
    single_subject = {}
    
    for subject in stats.keys():
        if subject in excluded_subjects:
            continue
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

        if values[0] < 70:
            print(f"Warning: {subject_id} has a go_trial success rate below 70%: {values[0]}%")
            if subject_id not in new_excluded_subjects:
                new_excluded_subjects.append(subject_id)
        if values[1] > 65 or values[1] < 35:
            print(f"Warning: {subject_id} has a stop_trial success rate outside 35-65%: {values[1]}%")
            if subject_id not in new_excluded_subjects:
                new_excluded_subjects.append(subject_id)

        # Plot bars for each condition
        bar_width = 0.3
        index = np.arange(len(trial_types))
        opacity = 0.8
        plt.figure(figsize=(10, 6))

        plt.bar(index, values, bar_width, alpha=opacity, color=color_dict[condition], label='control', capsize=5)

        for i, (val) in enumerate(values):
            plt.text(x=i, y=val + 1, s=f'{val:.1f}%', ha='center', va='bottom', color='black')

        # Add labels, title, and legend
        plt.xlabel('Trial Type')
        plt.ylabel('Percent Correct')
        plt.title(f'Performance for Subject {subject_id}')
        plt.xticks(index, ['Go Trial', 'Stop Trial', 'Go Fast Trial', 'Go Continue Trial'])
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig(join(saving_path, f"Performance for {subject_id}.{save_as}"), dpi=300)
        plt.close()    

    # if another session is available for the excluded subjects, exclude these as well:
    for excluded_sub in new_excluded_subjects:
        base_sub_id = excluded_sub.split(' ')[0]  # get the base subject ID without condition
        for sub in stats.keys():
            if sub.startswith(base_sub_id) and sub not in new_excluded_subjects:
                new_excluded_subjects.append(sub)

    # remove duplicates
    excluded_subjects = list(set(excluded_subjects + new_excluded_subjects))            
                
    # Convert to DataFrame
    df_summary = pd.DataFrame(data_list)

    # Save to Excel
    df_summary.to_excel(join(saving_path,'summary success rates.xlsx'), index=False)

    return excluded_subjects
