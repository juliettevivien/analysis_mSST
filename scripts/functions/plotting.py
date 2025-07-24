import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from os.path import join

from functions import utils


def plot_reaction_time_relative_to_SSD(
        rt_inhibition_df,
        color_dict,
        behav_results_saving_path
):
    for sub in rt_inhibition_df['subject'].unique():
        plt.figure(figsize=(10, 6))
        subset_df = rt_inhibition_df[rt_inhibition_df['subject'] == sub]
        sns.scatterplot(data=subset_df, x='SSD', y='RT', hue='condition', palette=color_dict, marker='o')
        # pearson correlation for each condition if multiple conditions
        corr_str = ''
        for condition in subset_df['condition'].unique():
            condition_df = subset_df[subset_df['condition'] == condition]
            corr, p_value = scipy.stats.pearsonr(condition_df['SSD'], condition_df['RT'])
            corr_str = corr_str + f'\n Pearson r {condition} = {corr:.2f}, p = {p_value:.3f}'
        plt.title(f'Reaction Time relative to SSD for {sub} \n {corr_str}')
        plt.xlabel('Stop Signal Delay (SSD) [ms]')
        plt.xlim(0, 1000)
        plt.ylabel('Reaction Time (ms)')
        plt.legend(title='Condition')
        plt.savefig(join(behav_results_saving_path, f'{sub}_RT_relative_to_SSD.png'), dpi=300)


def plot_inhibitory_functions_per_groups(
        grouped_df,
        stats,
        color_dict,
        behav_results_saving_path
):
    all_data = []  # To collect ZRFT and p across all subjects

    for sub in grouped_df['subject'].unique():
        subset_df = grouped_df[grouped_df['subject'] == sub]
        
        for condition in subset_df['condition'].unique():
            condition_df = subset_df[subset_df['condition'] == condition]
            
            if condition in ['control', 'preop']:
                session = f'{sub} mSST'
            else:
                session = f'{sub} {condition} mSST'
            
            # Check if session is in stats
            if session not in stats:
                continue
            
            go_rt = stats[session]['go_trial RTs (ms)']
            ssrt = stats[session]['SSRT (ms)']
            MRT = np.mean(go_rt)
            SDRT = np.std(go_rt)
            
            condition_df = condition_df.copy()
            condition_df['ZRFT'] = (MRT - condition_df['SSD'] - ssrt) / SDRT
            condition_df['subject'] = sub
            condition_df['session'] = session
            condition_df['condition'] = condition
            
            all_data.append(condition_df[['ZRFT', 'p', 'condition']])

    # Combine all ZRFTs into one DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # Bin ZRFTs to align across subjects
    combined_df['ZRFT_bin'] = pd.cut(combined_df['ZRFT'], bins=np.arange(-4, 4.1, 0.5))

    # Now group by condition and ZRFT bin
    mean_df = (
        combined_df.groupby(['condition', 'ZRFT_bin'])
        .agg(mean_p=('p', 'mean'), n=('p', 'count'))
        .reset_index()
    )

    # For plotting, convert bin intervals to center points
    mean_df['ZRFT_center'] = mean_df['ZRFT_bin'].apply(lambda x: x.mid)

    # Plotting
    plt.figure(figsize=(8, 5))
    for condition in mean_df['condition'].unique():
        cond_df = mean_df[mean_df['condition'] == condition]
        plt.plot(cond_df['ZRFT_center'], cond_df['mean_p'], marker='o', label=condition, color=color_dict.get(condition, 'black'))

    plt.xlabel('ZRFT (Z-score Relative Finishing Time)')
    plt.ylabel('P(respond | signal)')
    plt.title('Group Averaged Inhibition Functions by Condition')
    plt.xlim(-4, 4)
    plt.ylim(0, 1)
    plt.legend()
    plt.axhline(0.5, linestyle='--', color='gray', linewidth=1)
    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, 'Group Averaged Inhibition Functions.png'), dpi=300)



def plot_inhibitory_function_per_subject(
        grouped_df,
        color_dict,
        behav_results_saving_path
):
    for sub in grouped_df['subject'].unique():
        plt.figure(figsize=(10, 6))
        subset_df = grouped_df[grouped_df['subject'] == sub]
        sns.lineplot(data=subset_df, x='SSD', y='p', hue='condition', palette=color_dict, marker='o')
        sns.scatterplot(data=subset_df, x='SSD', y='p', size='count', legend=False, hue='condition', palette=color_dict,
                    sizes=(10, 300), zorder=7)
        plt.title(f'Inhibition Function for {sub}')
        plt.xlabel('Stop Signal Delay (SSD) [ms]')
        plt.axhline(y=0.5, color='gray', linestyle='--', label='50% Response Rate')
        plt.xlim(0, 1000)
        plt.ylabel('p(Respond|Signal)')
        plt.legend(title='Condition')
        plt.savefig(join(behav_results_saving_path, f'{sub}_inhibition_function.png'), dpi=300)


def plot_inhibitory_function_per_subject_zscored(
    grouped_df,
    stats,
    color_dict,
    behav_results_saving_path
):
    for sub in grouped_df['subject'].unique():
        subset_df = grouped_df[grouped_df['subject'] == sub]
        
        # Create one figure per subject
        plt.figure(figsize=(8, 5))
        
        for condition in subset_df['condition'].unique():
            condition_df = subset_df[subset_df['condition'] == condition]
            
            # Define the stats session name
            if condition in ['control', 'preop']:
                session = f'{sub} mSST'
            else:
                session = f'{sub} {condition} mSST'
            
            print(f"Now processing {session}")
            
            # Pull go RTs and SSRT
            go_rt = stats[session]['go_trial RTs (ms)']
            ssrt = stats[session]['SSRT (ms)']
            MRT = np.mean(go_rt)
            SDRT = np.std(go_rt)
            
            # Calculate ZRFT
            condition_df = condition_df.copy()  # avoid SettingWithCopyWarning
            condition_df['ZRFT'] = (MRT - condition_df['SSD'] - ssrt) / SDRT

            # Plot, using the color from your color_dict
            color = color_dict.get(condition, 'gray')  # default to gray if not found
            plt.plot(
                condition_df['ZRFT'], 
                condition_df['p'], 
                marker='o', 
                linestyle='-',
                label=condition,
                color=color
            )

        # Plot customization (one per subject)
        plt.xlabel('ZRFT (Z-score Relative Finishing Time)')
        plt.ylabel('P(respond | signal)')
        plt.title(f'ZRFT Inhibition Function – {sub}')
        plt.xlim(-4, 4)
        plt.ylim(0, 1.05)
        plt.axhline(0.5, linestyle='--', color='gray', linewidth=1)
        plt.axvline(0, linestyle='--', color='gray', linewidth=1)
        plt.legend(title='Condition')
        plt.tight_layout()
        plt.savefig(join(behav_results_saving_path, f'{sub}_ZRFT_inhibition_function.png'), dpi=300)


def plot_color_palette(subject_colors, behav_results_saving_path):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 2))

    # Create color dots
    for i, (label, color) in enumerate(subject_colors.items()):
        ax.add_patch(plt.Rectangle((i * 0.07, 0), 0.06, 1, color=color))

    # Set axis limits and remove ticks
    ax.set_xlim(0, len(subject_colors) * 0.07)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add labels
    for i, label in enumerate(subject_colors.keys()):
        ax.text(i * 0.07 + 0.03, 1.02, label, ha='center', va='bottom', rotation=45, fontsize=8)

    # Save and Display the figure
    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, "Subject color palette.png"), dpi=300)
    plt.close()


def plot_go_gf_rt_single_sub(
        stats_OFF,
        stats_ON,
        stats_CONTROL,
        stats_PREOP,
        color_dict,
        behav_results_saving_path
):
    # Define trial types to include only go trials and go_fast_trials
    trial_types = ['go_trial', 'go_fast_trial']

    p_value_dict = {}

    # Loop through the filtered dictionaries (e.g., stats_OFF, stats_ON, etc.)
    for condition, condition_stats in [('DBS OFF', stats_OFF), 
                                    ('DBS ON', stats_ON), 
                                    ('control', stats_CONTROL), 
                                    ('preop', stats_PREOP)]:
        for subject_id, subject_data in condition_stats.items():
            # Gather data for the selected trial types into a DataFrame
            data = []
            for trial_type in trial_types:
                if f"{trial_type} RTs (ms)" in subject_data:  # Ensure the trial type key exists
                    data.extend([(trial_type, val) for val in subject_data[f"{trial_type} RTs (ms)"]])

            df = pd.DataFrame(data, columns=['Trial Type', 'Reaction Time'])

            # Extract reaction times for statistical comparison
            go_data = df[df['Trial Type'] == 'go_trial']['Reaction Time']
            go_fast_data = df[df['Trial Type'] == 'go_fast_trial']['Reaction Time']

            # Perform Mann-Whitney U test if both trial types have sufficient data
            if len(go_data) >= 2 and len(go_fast_data) >= 2:  # Minimum data size for test
                stat, p_value = scipy.stats.mannwhitneyu(go_fast_data, go_data, alternative='less')
                print(f"P-value for {subject_id} ({condition}): {p_value:.5f}")
            else:
                p_value = None  # Not enough data to perform test

            p_value_dict[subject_id] = p_value

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
                annotation = "ns" if p_value >= 0.1 else "*"
                max_val = df['Reaction Time'].max()
                plt.text(0.5, max_val + 10, f"p = {p_value}\n{annotation}", 
                        ha='center', va='bottom', color='black', fontsize=12)

            # Customize plot
            plt.xlabel('Trial Type')
            plt.ylabel('Reaction Time (ms)')
            plt.title(f'Reaction Times for Subject {subject_id}')
            plt.tight_layout()
            plt.savefig(join(behav_results_saving_path, f'{subject_id} RT - GO vs GO FAST.png'))
            plt.close()

    # Convert dictionary to DataFrame
    df_p_value_dict = pd.DataFrame(list(p_value_dict.items()), columns=['Subject', 'P-Value'])

    # Save to Excel
    df_p_value_dict.to_excel(join(behav_results_saving_path,'p_values_proactive_inhibition.xlsx'), index=False)

    print("Excel file saved as 'p_values_proactive_inhibition.xlsx'.")



def plot_go_gf_rt_group(
    stats_OFF,
    stats_ON,
    stats_CONTROL,
    stats_PREOP,
    color_dict,
    behav_results_saving_path,
    show_fig = False
):
        
    patients_by_condition = {
        'control': len(stats_CONTROL.keys()),
        'DBS OFF': len(stats_OFF.keys()),
        'DBS ON': len(stats_ON.keys()),
        'preop': len(stats_PREOP.keys())
    }

    # Combine data across participants for each condition
    group_data = {}
    for condition, condition_stats in [('control', stats_CONTROL), 
                                    ('DBS OFF', stats_OFF), 
                                    ('DBS ON', stats_ON), 
                                    ('preop', stats_PREOP)]:
        group_data[condition] = {trial_type: [] for trial_type in ['go_trial', 'go_fast_trial']}

        for subject_id, subject_data in condition_stats.items():
            for trial_type in ['go_trial', 'go_fast_trial']:
                if f"{trial_type} RTs (ms)" in subject_data:
                    group_data[condition][trial_type].extend(subject_data[f"{trial_type} RTs (ms)"])

    data = []
    for condition, trials in group_data.items():
        for trial_type, reaction_times in trials.items():
            data.extend([(condition, trial_type, rt) for rt in reaction_times])

    df_group = pd.DataFrame(data, columns=['Condition', 'Trial Type', 'Reaction Time'])

    # Create the group-level plot
    plt.figure(figsize=(10, 8))

    sns.set_style('white')
    fig, axs = plt.subplots(ncols=4, figsize=(8, 4), sharey=True)

    # Loop through each condition and plot
    for condition, ax in zip(color_dict.keys(), axs):
        condition_data = df_group[df_group['Condition'] == condition]

        # Calculate the means for go and go_fast trials
        means = condition_data.groupby('Trial Type')['Reaction Time'].mean()
        
        # Perform a paired t-test
        go_rt = condition_data[condition_data['Trial Type'] == 'go_trial']['Reaction Time']
        go_fast_rt = condition_data[condition_data['Trial Type'] == 'go_fast_trial']['Reaction Time']
        

            # Perform Mann-Whitney U test if both trial types have sufficient data
        try:
            stat, p_value = scipy.stats.mannwhitneyu(go_fast_rt, go_rt, alternative='less')
        except ValueError:  # If t-test fails due to insufficient data
            p_value = 1.0

        # Plot the violin plot
        sns.violinplot(ax=ax, data=condition_data, x='Trial Type', y='Reaction Time', hue='Trial Type',
                    palette={'go_trial': 'white', 'go_fast_trial': 'lightgrey'}, bw_method=0.2, density_norm='width', split=True, inner=None)

        # Overlay individual points with stripplot, colored by condition
        sns.stripplot(ax=ax, data=condition_data, x='Trial Type', y='Reaction Time',
                    hue='Condition', palette=color_dict, dodge=False, jitter=True, size=2, alpha=0.7)

        # Add horizontal line for the means
        for i, trial_type in enumerate(['go_trial', 'go_fast_trial']):
            ax.plot([i - 0.2, i + 0.2], [means[trial_type]] * 2, color='black', linewidth=2, zorder=3)

        # Annotate with stars based on p-value
        if p_value > 0.05:
            stars = "ns"
        elif p_value <= 0.05 and p_value > 0.01:
            stars = "*"
        elif p_value <= 0.01 and p_value > 0.001:
            stars = "**"
        elif p_value <= 0.001 and p_value > 0.0001:
            stars = "***"
        else:  # p_value <= 0.0001
            stars = "****"

        # Print the p-value in the console
        print(f"P-value for {condition} (go_trial vs go_fast_trial): {p_value:.2e}")
        print(f"Stat for {condition} (go_trial vs go_fast_trial): {stat:.2e}")

        # Add stars above the plot
        ax.text(0.5, max(go_rt.max(), go_fast_rt.max()) + 60, stars, 
                ha='center', color='black', fontsize=15)


        ax.set_title(f"{condition.capitalize()} (n={patients_by_condition[condition]})")
        ax.set_xlabel('')
        ax.set_ylabel('Reaction Time (ms)')
        ax.legend_.remove()

    # Finalize the plot
    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, 'Group RT - GO vs GO FAST.png'))
    if show_fig:
        plt.show()
    else:
        plt.close()




def plot_prep_cost_on_vs_off_all_sub(
        stats_OFF,
        stats_ON,
        subject_colors,
        behav_results_saving_path
):
    
    # Define conditions and corresponding dictionaries
    conditions = {
        'DBS OFF': stats_OFF,
        'DBS ON': stats_ON,
    }

    # Initialize dictionaries to hold results for each condition
    results = {condition: {} for condition in conditions.keys()}

    # Loop through each condition and subject
    for condition, data_dict in conditions.items():
        for subject_id, metrics in data_dict.items():
            sub_id = subject_id.split()[0]  # Extract subject ID
            prep_cost = metrics['go_trial mean RT (ms)'] - metrics['go_fast_trial mean RT (ms)']
            results[condition][sub_id] = prep_cost
            utils.update_or_create_json(behav_results_saving_path, subject_id, new_key="Prep Cost (ms)", new_value=prep_cost)

    # Prepare DataFrame
    data = []
    for condition, subject_dict in results.items():
        for subject_id, prep_cost in subject_dict.items():
            data.append({'Subject': subject_id, 'Condition': condition, 'preparation cost (ms)': prep_cost})

    df_proactive = pd.DataFrame(data)

    # Reshape data to include all subjects, even those with one condition only
    df_reshaped = df_proactive.pivot(index="Subject", columns="Condition", values="preparation cost (ms)").reset_index()

    # Ensure both 'dbs_off' and 'dbs_on' columns exist
    dbs_off = df_reshaped.get('DBS OFF', pd.Series(index=df_reshaped.index, dtype=float)).values
    dbs_on = df_reshaped.get('DBS ON', pd.Series(index=df_reshaped.index, dtype=float)).values

    # Handle missing values: Only run Wilcoxon test on pairs
    valid_indices = ~np.isnan(dbs_off) & ~np.isnan(dbs_on)
    if valid_indices.sum() > 0:
        test_result, p_value = scipy.stats.wilcoxon(dbs_off[valid_indices], dbs_on[valid_indices])
    else:
        test_result, p_value = None, None

    # Calculate mean and std
    group_stats = df_proactive.groupby('Condition')['preparation cost (ms)'].agg(['mean', 'std'])
    print("Mean and standard deviation for each condition:")
    print(group_stats)

    # Initialize plot
    plt.figure(figsize=(10, 6))

    # Create violin plot
    sns.violinplot(
        data=df_proactive, x='Condition', y='preparation cost (ms)', hue='Condition', 
        split=False, inner=None, width=0.6,
        palette={'DBS OFF': '#20a39e', 'DBS ON': '#ef5b5b'}, alpha=0.2, legend=False
    )

    # Unique subject colors
    subjects = df_proactive['Subject'].unique()
    #subject_palette = sns.color_palette("husl", len(subjects))  # Ensure distinct colors
    #subject_colors = dict(zip(subjects, subject_palette))

    # Retrieve x-coordinates for each condition
    ax = plt.gca()
    condition_x_positions = {label.get_text(): pos for label, pos in zip(ax.get_xticklabels(), ax.get_xticks())}

    jitter_strength = 0.15
    subject_jitter = {}

    # Overlay subject points
    for subject_id in df_proactive['Subject'].unique():
        subject_data = df_proactive[df_proactive['Subject'] == subject_id]

        if subject_id not in subject_jitter:
            subject_jitter[subject_id] = np.random.uniform(-jitter_strength, jitter_strength)
        
        jitter_value = subject_jitter[subject_id]
        
        x_coords = []
        y_coords = []
        
        for _, row in subject_data.iterrows():
            cond = row['Condition']
            x_coords.append(condition_x_positions[cond] + jitter_value)
            y_coords.append(row['preparation cost (ms)'])

        # Plot single values
        if len(y_coords) == 1:
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder=100)
        else:
            plt.plot(x_coords, y_coords, color='gray', alpha=0.5)
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder=100)

    # Annotate statistical results
    if p_value is not None:
        annotation = "ns (Wilcoxon signed-rank test)" if p_value >= 0.05 else "* (Wilcoxon signed-rank test)"
        max_val = df_proactive['preparation cost (ms)'].max()
        plt.text(0.5, max_val + 10, f"statistic = {test_result}\n p = {p_value}\n{annotation}", 
                ha='center', va='bottom', color='black', fontsize=14)
        plt.axhline(0, color = 'grey', linestyle='--')

    # Add labels, title, and legend
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('Mean Preparation Cost (ms)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Mean Preparation Cost Across Conditions', fontsize=16)

    # Custom subject legend
    legend_patches = [matplotlib.patches.Patch(color=subject_colors[subj], label=subj) for subj in subjects]
    plt.legend(handles=legend_patches, title="Subjects", fontsize=12, title_fontsize=14, loc="upper right", bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, "Preparation Cost - DBS ON vs DBS OFF.png"), dpi=300)
    plt.close()




def plot_prep_cost_on_vs_off_only_sub_with_2_sessions(
        stats_OFF,
        stats_ON,
        subject_colors,
        behav_results_saving_path,
        show_fig = False
):
    # Define conditions and corresponding dictionaries
    conditions = {
        'DBS OFF': stats_OFF,
        'DBS ON': stats_ON,
    }

    # Initialize dictionaries to hold results for each condition
    results = {condition: {} for condition in conditions.keys()}

    # Loop through each condition and subject
    for condition, data_dict in conditions.items():
        for subject_id, metrics in data_dict.items():
            #print(subject_id)
            # Extract the subject ID (first part of subject_id before the first space)
            sub_id = subject_id.split()[0]
            # Retrieve the required metrics and store them in the result dictionary
            prep_cost = (metrics['go_trial mean RT (ms)'] - metrics['go_fast_trial mean RT (ms)'])
            #print(prep_cost)
            if prep_cost < 0:
                #continue
                print(f"Negative preparation cost for {subject_id}: {prep_cost} ms.")
                results[condition][sub_id] = prep_cost
                utils.update_or_create_json(behav_results_saving_path, subject_id, new_key = "Prep Cost (ms)", new_value = prep_cost)

            else:
                results[condition][sub_id] = prep_cost
                utils.update_or_create_json(behav_results_saving_path, subject_id, new_key = "Prep Cost (ms)", new_value = prep_cost)

    # Prepare data for DataFrame
    data = []
    for condition, subject_dict in results.items():
        for subject_id, prep_cost in subject_dict.items():
            data.append({'Subject': subject_id, 'Condition': condition, 'preparation cost (ms)': prep_cost})

    # Create DataFrame
    df_proactive = pd.DataFrame(data)

    # Reshape using pivot
    df_reshaped = df_proactive.pivot(index="Subject", columns="Condition", values="preparation cost (ms)")

    # Rename columns for clarity
    df_reshaped = df_reshaped.rename(columns={"dbs_off": "Pre_treatment", "dbs_on": "Post_treatment"})

    # Reset index (optional, keeps Subject as a column)
    df_reshaped.reset_index(inplace=True)

    # Pivot the data to get dbs_off and dbs_on side by side
    pivot_df = df_proactive.pivot(index='Subject', columns='Condition', values='preparation cost (ms)')

    # Extract the arrays
    dbs_off = pivot_df['DBS OFF'].values
    dbs_on = pivot_df['DBS ON'].values

    # Handle missing values: Only run Wilcoxon test on pairs
    valid_indices = ~np.isnan(dbs_off) & ~np.isnan(dbs_on)
    if valid_indices.sum() > 0:
        test_result, p_value = scipy.stats.wilcoxon(dbs_off[valid_indices], dbs_on[valid_indices])
    else:
        test_result, p_value = None, None

    # Calculate mean and standard deviation for each condition
    #group_stats = df_proactive.groupby('Condition')['preparation cost (ms)'].agg(['mean', 'std'])

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Create the violin plot
    sns.violinplot(data=df_proactive, x='Condition', y='preparation cost (ms)', hue='Condition', 
                split=False, inner=None, width=0.6,
                palette={'DBS OFF': '#20a39e', 'DBS ON': '#ef5b5b'}, 
                alpha=0.2, legend=False)

    # Retrieve x-coordinates for each condition
    ax = plt.gca()  # Get current axis
    condition_x_positions = {label.get_text(): pos for label, pos in zip(ax.get_xticklabels(), ax.get_xticks())}

    # Jitter strength
    jitter_strength = 0.05

    # Store jittered x-values for each subject
    subject_jitter = {}

    # Overlay individual subject points (manually, replacing sns.stripplot)
    for subject_id in df_proactive['Subject'].unique():
        # Get data for this subject in 'dbs_off' and 'dbs_on' conditions
        subject_data = df_proactive[(df_proactive['Subject'] == subject_id) & (df_proactive['Condition'].isin(['DBS OFF', 'DBS ON']))]
        
        if len(subject_data) == 2:
            # Assign a consistent jitter for this subject
            if subject_id not in subject_jitter:
                subject_jitter[subject_id] = np.random.uniform(-jitter_strength, jitter_strength)
            
            jitter_value = subject_jitter[subject_id]

            # Jittered x-coordinates
            x_coords = [condition_x_positions[subject_data.iloc[i]['Condition']] + jitter_value for i in range(2)]
            y_coords = subject_data['preparation cost (ms)'].values
            
            # Draw connecting lines
            plt.plot(x_coords, y_coords, color='gray', alpha=0.5)
            plt.axhline(0, color = 'grey', linestyle='--')

            # Scatter the subject's data points with color
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder= 100, label=subject_id if subject_id not in subject_jitter else "")

    # Annotate statistical results
    if p_value is not None:
        annotation = "ns (Wilcoxon signed-rank test)" if p_value >= 0.05 else "* (Wilcoxon signed-rank test)"
        max_val = df_proactive['preparation cost (ms)'].max()
        plt.text(0.5, max_val + 10, f"statistic = {test_result}\n p = {p_value}\n{annotation}", 
                    ha='center', va='bottom', color='black', fontsize=14)

    # Add labels, title, and legend
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('Mean Preparation Cost (ms)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Mean Preparation Cost Across Conditions (mean RT go trial - mean RT go fast trial)', fontsize=16)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Subject ID', title_fontsize='14', fontsize='12')

    # Create custom legend for subjects
    valid_subjects = df_reshaped.dropna()['Subject'].unique()
    legend_patches = [matplotlib.patches.Patch(color=subject_colors[subj], label=subj) for subj in valid_subjects]
    plt.legend(handles=legend_patches, title="Subjects", fontsize=12, title_fontsize=14, loc="upper right", bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, "Preparation Cost - DBS ON vs DBS OFF only subjects with 2 sessions.png"), dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()

    return df_reshaped



def plot_prep_cost_all_groups(
    stats_OFF,
    stats_ON,
    stats_CONTROL,
    stats_PREOP,
    color_dict,
    behav_results_saving_path,
    show_fig = False        
):
    # Define conditions and corresponding dictionaries
    conditions = {
        'control': stats_CONTROL,
        'DBS OFF': stats_OFF,
        'DBS ON': stats_ON,
        'preop': stats_PREOP
    }

    # Initialize dictionaries to hold results for each condition
    results = {condition: {} for condition in conditions.keys()}

    # Loop through each condition and subject
    for condition, data_dict in conditions.items():
        for subject_id, metrics in data_dict.items():
            # Extract the subject ID (first part of subject_id before the first space)
            sub_id = subject_id.split()[0]
            # Retrieve the required metrics and store them in the result dictionary
            prep_cost = (metrics['go_trial mean RT (ms)'] - metrics['go_fast_trial mean RT (ms)'])
            results[condition][sub_id] = prep_cost

    # Prepare data for DataFrame
    data = []
    for condition, subject_dict in results.items():
        for subject_id, prep_cost in subject_dict.items():
            data.append({'Subject': subject_id, 'Condition': condition, 'preparation cost (ms)': prep_cost})

    # Create DataFrame
    df_proactive_all = pd.DataFrame(data)

    # Calculate mean and standard deviation for each condition
    group_stats = df_proactive_all.groupby('Condition')['preparation cost (ms)'].agg(['mean', 'std'])
    print("Mean and standard deviation for each condition:")
    print(group_stats)

    # Perform pairwise Mann-Whitney U tests to compare each condition
    conditions = ['control', 'DBS OFF', 'DBS ON', 'preop']

    print("\nPairwise Mann-Whitney U tests (alternative='two-sided'):")

    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            condition1 = conditions[i]
            condition2 = conditions[j]

            data1 = df_proactive_all[df_proactive_all['Condition'] == condition1]['preparation cost (ms)'].dropna()
            data2 = df_proactive_all[df_proactive_all['Condition'] == condition2]['preparation cost (ms)'].dropna()

            # Mann-Whitney U test
            stat, p_value = scipy.stats.mannwhitneyu(data1, data2, alternative='two-sided')

            print(f"{condition1} vs {condition2}: U-statistic={stat}, p-value={p_value:.5f}")

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Create a violin plot
    sns.violinplot(data=df_proactive_all, x='Condition', y='preparation cost (ms)', hue='Condition', 
                split=False, inner=None, 
                palette=color_dict, 
                alpha=0.2, legend=False)

    # Overlay individual data points
    stripplot = sns.stripplot(data=df_proactive_all, x='Condition', y='preparation cost (ms)', hue='Condition', 
                            dodge=False, jitter=False, color='black', marker='o', size=10,
                            palette=color_dict)

    # Retrieve the x-coordinates for each condition label
    condition_x_positions = {label.get_text(): pos for label, pos in zip(stripplot.get_xticklabels(), stripplot.get_xticks())}

    # Draw lines between corresponding subject points in 'dbs_off' and 'dbs_on'
    for subject_id in df_proactive_all['Subject'].unique():
        # Get data for this subject in 'dbs_off' and 'dbs_on' conditions
        subject_data = df_proactive_all[(df_proactive_all['Subject'] == subject_id) & (df_proactive_all['Condition'].isin(['dbs_off', 'dbs_on']))]
        if len(subject_data) == 2:
            # Use x-coordinates based on the dictionary created above
            x_coords = [condition_x_positions[subject_data.iloc[i]['Condition']] for i in range(2)]
            y_coords = subject_data['preparation cost (ms)'].values
            plt.plot(x_coords, y_coords, marker='o', color='gray', alpha=0.5)

    # Calculate number of subjects in each group
    subject_counts = df_proactive_all.groupby('Condition')['Subject'].nunique()

    # Add "n=number of subjects" above each violin
    for condition, count in subject_counts.items():
        x_position = condition_x_positions[condition]  # Get the x-position for the condition
        plt.text(x_position, df_proactive_all['preparation cost (ms)'].max() + 100, f'n={count}', 
                horizontalalignment='center', fontsize=12, color='black')

    # Add labels, title, and legend
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('Mean Preparation Cost (ms)', fontsize=14)
    plt.title('Mean Preparation Cost Across Conditions (mean RT go trial - mean RT go fast trial)', fontsize=16)

    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, "Preparation Cost - All groups.png"), dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()

    return df_proactive_all



def plot_SSRT_on_vs_off_all_sub(
        stats_OFF,
        stats_ON,
        subject_colors,
        behav_results_saving_path,
        show_fig = False
):

    # Define conditions and corresponding dictionaries
    conditions = {
        'DBS OFF': stats_OFF,
        'DBS ON': stats_ON,
    }

    # Initialize dictionaries to hold results for each condition
    results = {condition: {} for condition in conditions.keys()}

    # Loop through each condition and subject
    for condition, data_dict in conditions.items():
        for subject_id, metrics in data_dict.items():
            # Extract the subject ID (first part of subject_id before the first space)
            sub_id = subject_id.split()[0]
            ssrt_value = metrics['SSRT (ms)']
            ssd_value = metrics['mean SSD (ms)']
            results[condition][sub_id] = ssrt_value
            utils.update_or_create_json(behav_results_saving_path, subject_id, new_key = "mean SSD (ms)", new_value = ssd_value)
            utils.update_or_create_json(behav_results_saving_path, subject_id, new_key = "SSRT (ms)", new_value = ssrt_value)

    # Prepare data for DataFrame
    data = []
    for condition, subject_dict in results.items():
        for subject_id, ssrt_value in subject_dict.items():
            data.append({'Subject': subject_id, 'Condition': condition, 'SSRT (ms)': ssrt_value})

    # Create DataFrame
    df_reactive = pd.DataFrame(data)

    # Pivot the data to get dbs_off and dbs_on side by side
    pivot_df = df_reactive.pivot(index='Subject', columns='Condition', values='SSRT (ms)')

    # Extract the arrays
    dbs_off = pivot_df.get('DBS OFF', pd.Series(index=pivot_df.index, dtype=float)).values
    dbs_on = pivot_df.get('DBS ON', pd.Series(index=pivot_df.index, dtype=float)).values

    # Handle missing values: Only run Wilcoxon test on pairs
    valid_indices = ~np.isnan(dbs_off) & ~np.isnan(dbs_on)
    if valid_indices.sum() > 0:
        test_result, p_value = scipy.stats.wilcoxon(dbs_off[valid_indices], dbs_on[valid_indices])
        print("test done")
    else:
        test_result, p_value = None, None

    # Calculate mean and std
    group_stats = df_reactive.groupby('Condition')['SSRT (ms)'].agg(['mean', 'std'])
    print("Mean and standard deviation for each condition:")
    print(group_stats)

    #ttest_result = scipy.stats.ttest_rel(dbs_on, dbs_off)

    # Calculate mean and standard deviation for each condition
    #group_stats = df_reactive.groupby('Condition')['SSRT (ms)'].agg(['mean', 'std'])
    #print("Mean and standard deviation for each condition:")
    #print(group_stats)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Create a violin plot
    stripplot = sns.violinplot(data=df_reactive, x='Condition', y='SSRT (ms)', hue='Condition', 
                            split=False, inner=None, width=0.6,
                            palette={'DBS OFF': '#20a39e', 'DBS ON': '#ef5b5b'}, 
                            alpha=0.2, legend=False)

    # Overlay individual data points
    subjects = df_reactive['Subject'].unique()

    # Retrieve x-coordinates for each condition
    ax = plt.gca()  # Get current axis
    condition_x_positions = {label.get_text(): pos for label, pos in zip(ax.get_xticklabels(), ax.get_xticks())}

    # Jitter strength
    jitter_strength = 0.15

    # Store jittered x-values for each subject
    subject_jitter = {}

    """
    # Overlay individual subject points (manually, replacing sns.stripplot)
    for subject_id in df_reactive['Subject'].unique():
        # Get data for this subject in 'dbs_off' and 'dbs_on' conditions
        subject_data = df_reactive[(df_reactive['Subject'] == subject_id) & (df_reactive['Condition'].isin(['dbs_off', 'dbs_on']))]
        
        if len(subject_data) == 2:
            # Assign a consistent jitter for this subject
            if subject_id not in subject_jitter:
                subject_jitter[subject_id] = np.random.uniform(-jitter_strength, jitter_strength)
            
            jitter_value = subject_jitter[subject_id]

            # Jittered x-coordinates
            x_coords = [condition_x_positions[subject_data.iloc[i]['Condition']] + jitter_value for i in range(2)]
            y_coords = subject_data['SSRT (ms)'].values

            # Draw connecting lines
            plt.plot(x_coords, y_coords, color='gray', alpha=0.5)

            # Scatter the subject's data points with color
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder = 100, label=subject_id if subject_id not in subject_jitter else "")
    """

    # Overlay subject points
    for subject_id in df_reactive['Subject'].unique():
        # Get data for this subject in 'dbs_off' and 'dbs_on' conditions
        subject_data = df_reactive[(df_reactive['Subject'] == subject_id)]

        if subject_id not in subject_jitter:
            subject_jitter[subject_id] = np.random.uniform(-jitter_strength, jitter_strength)
        
        jitter_value = subject_jitter[subject_id]
        
        x_coords = []
        y_coords = []
        
        for _, row in subject_data.iterrows():
            cond = row['Condition']
            x_coords.append(condition_x_positions[cond] + jitter_value)
            y_coords.append(row['SSRT (ms)'])

        # Plot single values
        if len(y_coords) == 1:
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder=100)
        else:
            plt.plot(x_coords, y_coords, color='gray', alpha=0.5)
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder=100)


    # Statistical test
    #test_result, p_value = scipy.stats.wilcoxon(dbs_on, dbs_off)
    #print(f"Wilcoxon signed-rank test: p = {p_value:.5f}, test statistic = {test_result}")

    # Annotate statistical results
    if p_value is not None:
        annotation = "ns (Wilcoxon signed-rank test)" if p_value >= 0.05 else "* (Wilcoxon signed-rank test)"
        max_val = df_reactive['SSRT (ms)'].max()
        plt.text(0.5, max_val + 10, f"statistic = {test_result}\n p = {p_value}\n{annotation}", 
                ha='center', va='bottom', color='black', fontsize=14)

    # Add labels, title, and legend
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('SSRT (ms)', fontsize=14)
    plt.title('SSRT Across Conditions (SSRT = nth GO RT - mean SSD)', fontsize=16)

    # Increase tick label font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Move the legend outside the plot area to avoid overlap
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Create custom legend for subjects
    legend_patches = [matplotlib.patches.Patch(color=subject_colors[subj], label=subj) for subj in subjects]
    plt.legend(handles=legend_patches, title="Subjects", fontsize=12, title_fontsize=14, loc="upper right", bbox_to_anchor=(1.2, 1))

    # Improve layout to fit everything
    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, "SSRT - DBS ON vs DBS OFF.png"), dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()






def plot_SSRT_all_groups(
    stats_OFF,
    stats_ON,
    stats_CONTROL,
    stats_PREOP,
    color_dict,
    behav_results_saving_path,
    show_fig = False        
):
    # Define conditions and corresponding dictionaries
    conditions = {
        'control': stats_CONTROL,
        'DBS OFF': stats_OFF,
        'DBS ON': stats_ON,
        'preop': stats_PREOP
    }

    # Initialize dictionaries to hold results for each condition, keyed by subject ID
    results = {condition: {} for condition in conditions.keys()}

    # Loop through each condition and subject
    for condition, data_dict in conditions.items():
        for subject_id, metrics in data_dict.items():
            # Extract the subject ID (first part of subject_id before the first space)
            sub_id = subject_id.split()[0]
            ssrt_value = metrics['SSRT (ms)']
            results[condition][sub_id] = ssrt_value

    # Prepare data for DataFrame
    data = []
    for condition, subject_dict in results.items():
        for subject_id, ssrt_value in subject_dict.items():
            data.append({'Subject': subject_id, 'Condition': condition, 'SSRT (ms)': ssrt_value})

    # Create DataFrame
    df_reactive_all = pd.DataFrame(data)
    print(df_reactive_all)

    # Calculate mean and standard deviation for each condition
    group_stats = df_reactive_all.groupby('Condition')['SSRT (ms)'].agg(['mean', 'std'])
    print("Mean and standard deviation for each condition:")
    print(group_stats)

    # Perform pairwise Mann-Whitney U tests to compare each condition
    conditions = ['control', 'DBS OFF', 'DBS ON', 'preop']

    print("\nPairwise Mann-Whitney U tests (alternative='two-sided'):")

    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            condition1 = conditions[i]
            condition2 = conditions[j]

            data1 = df_reactive_all[df_reactive_all['Condition'] == condition1]['SSRT (ms)'].dropna()
            data2 = df_reactive_all[df_reactive_all['Condition'] == condition2]['SSRT (ms)'].dropna()

            # Mann-Whitney U test
            stat, p_value = scipy.stats.mannwhitneyu(data1, data2, alternative='two-sided')

            print(f"{condition1} vs {condition2}: U-statistic={stat}, p-value={p_value:.5f}")


    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Create a violin plot
    sns.violinplot(data=df_reactive_all, x='Condition', y='SSRT (ms)', hue='Condition', 
                split=False, inner=None, 
                palette=color_dict, 
                alpha=0.2, legend=False)

    # Overlay individual data points
    stripplot = sns.stripplot(data=df_reactive_all, x='Condition', y='SSRT (ms)', hue='Condition', 
                            dodge=False, jitter=False, color='black', marker='o', size=10,
                            palette=color_dict)

    # Retrieve the x-coordinates for each condition label
    condition_x_positions = {label.get_text(): pos for label, pos in zip(stripplot.get_xticklabels(), stripplot.get_xticks())}

    # Draw lines between corresponding subject points in 'dbs_off' and 'dbs_on'
    for subject_id in df_reactive_all['Subject'].unique():
        # Get data for this subject in 'dbs_off' and 'dbs_on' conditions
        subject_data = df_reactive_all[(df_reactive_all['Subject'] == subject_id) & (df_reactive_all['Condition'].isin(['DBS OFF', 'DBS ON']))]
        if len(subject_data) == 2:
            # Use x-coordinates based on the dictionary created above
            x_coords = [condition_x_positions[subject_data.iloc[i]['Condition']] for i in range(2)]
            y_coords = subject_data['SSRT (ms)'].values
            plt.plot(x_coords, y_coords, marker='o', color='gray', alpha=0.5)

    # Calculate number of subjects in each group
    subject_counts = df_reactive_all.groupby('Condition')['Subject'].nunique()

    # Add "n=number of subjects" above each violin
    for condition, count in subject_counts.items():
        x_position = condition_x_positions[condition]  # Get the x-position for the condition
        plt.text(x_position, df_reactive_all['SSRT (ms)'].max() + 100, f'n={count}', 
                horizontalalignment='center', fontsize=12, color='black')


    # Add labels and title
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('SSRT (ms)', fontsize=14)
    plt.title('SSRT Across Conditions', fontsize=16)

    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, "SSRT - All groups.png"), dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()


    return df_reactive_all



def plot_corr_prep_cost_SSRT(
        df_merged,
        behav_results_saving_path,
        show_fig = False
):
    # plot SSRT vs preparation cost:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_merged, x='preparation cost (ms)', y='SSRT (ms)', hue='Subject', palette='tab20')

    # test if there is a significant correlation or not between the variables SSRT and Preparation cost:
    r, p = scipy.stats.pearsonr(df_merged['preparation cost (ms)'], df_merged['SSRT (ms)'])
    print(f"Pearson's r = {r:.3f}, p = {p:.3f}")

    plt.xlabel('Preparation cost (ms)')
    plt.ylabel('SSRT (ms)')
    plt.title(f"Preparation cost vs SSRT (Pearson's r = {r:.3f}, p = {p:.3f})")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #plt.text(50, 0.5, f"Pearson's r = {r:.3f}, p = {p:.3f}", ha='left', va='bottom', color='black', fontsize=12)

    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, "Correlation between preparation cost and SSRT.png"), dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()



def plot_corr_gort_ssrt(
    stats,    
    behav_results_saving_path,
    show_fig = False
):

    GO_RT_dict = {}
    SSRT_dict = {}

    for subject in stats.keys():
        GO_RT = stats[subject]['go_trial mean RT (ms)']
        SSRT = stats[subject]['SSRT (ms)']
        GO_RT_dict[subject] = GO_RT
        SSRT_dict[subject] = SSRT

    # Assuming GO_RT_dict and SSRT_dict are dictionaries
    subjects = list(GO_RT_dict.keys())
    go_rt_values = list(GO_RT_dict.values())
    ssrt_values = list(SSRT_dict.values())

    # Generate unique colors for each subject
    colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))

    # Plot each point with a specific color
    for i, subject in enumerate(subjects):
        plt.plot(go_rt_values[i], ssrt_values[i], 'o', color=colors[i], label=subject)

    plt.ylabel('SSRT (ms)')
    plt.xlabel('GO RT (ms)')
    plt.legend(title='Subjects', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Calculate correlation
    corr, p_value = scipy.stats.pearsonr(go_rt_values, ssrt_values)
    plt.title(f'Correlation: r = {corr:.2f}, p = {p_value:.3f}')
    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, "Correlation Go RT and SSRT.png"), dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()



def plot_corr_SSD_SSRT(
        df_merged,
        behav_results_saving_path,
        show_fig = False
):
    
    # plot SSRT vs SSD:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_merged, x='mean SSD (ms)', y='SSRT (ms)', hue='Subject', palette='tab20')

    # test if there is a significant correlation or not between the variables SSRT and SSD:
    r, p = scipy.stats.pearsonr(df_merged['mean SSD (ms)'], df_merged['SSRT (ms)'])
    print(f"Pearson's r = {r:.3f}, p = {p:.3f}")

    plt.xlabel('SSD (ms)')
    plt.ylabel('SSRT (ms)')
    plt.title(f"SSD vs SSRT (Pearson's r = {r:.3f}, p = {p:.6f})")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #plt.text(50, 0.5, f"Pearson's r = {r:.3f}, p = {p:.3f}", ha='left', va='bottom', color='black', fontsize=12)

    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, "Correlation between SSD and SSRT.png"), dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()
    

def plot_dbs_effect_success_rate_single_sub(
        stats_OFF,
        stats_ON,
        behav_results_saving_path,
        show_fig = False
):
    # extract subjects if they are present in both dbs conditions:
    subjects_dbs = []
    subject_IDs_OFF = []
    subject_IDs_ON = []

    for subject in stats_OFF:
        subject_ID = subject.split(' ')[0]
        subject_IDs_OFF.append(subject_ID)


    for subject in stats_ON:
        subject_ID = subject.split(' ')[0]
        subject_IDs_ON.append(subject_ID)


    for i in subject_IDs_OFF:
        if i in subject_IDs_ON:
            subjects_dbs.append(i)

    # in the case of the subjects that are present in both dbs conditions, create for each subject a dictionary holding the values from both dbs conditions:
    stats_dbs = {}

    for subject in subjects_dbs:
        stats_dbs[subject] = {}
        for condition in stats_OFF:
            if subject in condition:
                stats_dbs[subject]['OFF'] = stats_OFF[condition]
        for condition in stats_ON:
            if subject in condition:
                stats_dbs[subject]['ON'] = stats_ON[condition]


    # Define trial types and bar width
    trial_types = ['go_trial', 'stop_trial', 'go_fast_trial', 'go_continue_trial']
    bar_width = 0.3
    index = np.arange(len(trial_types))
    opacity = 0.8

    # Access data for a single subject in OFF and ON conditions
    for subject_id in stats_dbs.keys():
        stats_dbs_new = {
            'OFF': stats_dbs[subject_id]['OFF'],
            'ON': stats_dbs[subject_id]['ON']
        }

        # Retrieve values for each trial type in both conditions
        off_values = [
            stats_dbs_new['OFF']['percent correct go_trial'],
            stats_dbs_new['OFF']['percent correct stop_trial'],
            stats_dbs_new['OFF']['percent correct go_fast_trial'],
            stats_dbs_new['OFF']['percent correct go_continue_trial']
        ]

        on_values = [
            stats_dbs_new['ON']['percent correct go_trial'],
            stats_dbs_new['ON']['percent correct stop_trial'],
            stats_dbs_new['ON']['percent correct go_fast_trial'],
            stats_dbs_new['ON']['percent correct go_continue_trial']
        ]

        # Calculate means and standard deviations (although for one subject, they are simply the values)
        off_means = np.array(off_values)
        on_means = np.array(on_values)

        # Plot bars for each condition
        plt.figure(figsize=(10, 6))

        bar1 = plt.bar(index, off_means, bar_width, alpha=opacity, color='#20a39e', label='DBS OFF', capsize=5)
        bar2 = plt.bar(index + bar_width, on_means, bar_width, alpha=opacity, color='#ef5b5b', label='DBS ON', capsize=5)

        # Add condition labels on top of each bar
        for i, (off_val, on_val) in enumerate(zip(off_means, on_means)):
            plt.text(i, off_val + 1, f'{off_val:.1f}%', ha='center', va='bottom', color='black')
            plt.text(i + bar_width, on_val + 1, f'{on_val:.1f}%', ha='center', va='bottom', color='black')

        # Add labels, title, and legend
        plt.xlabel('Trial Type')
        plt.ylabel('Percent Correct')
        plt.title(f'Performance for Subject {subject_id}')
        plt.xticks(index + bar_width / 2, ['Go Trial', 'Stop Trial', 'Go Fast Trial', 'Go Continue Trial'])
        
        """
        # For each tick, add a short horizontal line at the level of 70%:
        thresholds = [70, 50, 70, 70]
        # Add short horizontal threshold lines
        for idx, threshold in enumerate(thresholds):
            plt.plot([idx - bar_width * (1 / 2) , idx + bar_width * (1/ 2) + bar_width], [threshold, threshold], color='black', linestyle='--', label='Expected performance' if idx == 0 else "")
        """
            
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig(join(behav_results_saving_path, f"Effect of DBS on success rate - {subject_id}.png"), dpi=300)
        if show_fig:
            plt.show()
        else:
            plt.close()



# create a function to plot off_vs_on percent success:
def plot_percent_success_on_vs_off(
        stats_OFF,
        stats_ON,
        trial_type,
        subject_colors,
        behav_results_saving_path,
        show_fig = False
):
    
    # Define conditions and corresponding dictionaries
    conditions = {
        'DBS OFF': stats_OFF,
        'DBS ON': stats_ON,
    }

    trial_type_keys = {'GO': 'percent correct go_trial',
            'GC': 'percent correct go_continue_trial',
            'GF': 'percent correct go_fast_trial',
            'Go-STOP': 'percent correct stop_trial'}

    # Initialize dictionaries to hold results for each condition
    results = {condition: {} for condition in conditions.keys()}

    # Loop through each condition and subject
    for condition, data_dict in conditions.items():
        for subject_id, metrics in data_dict.items():
            # Extract the subject ID (first part of subject_id before the first space)
            sub_id = subject_id.split()[0]
            # Calculate the desired  correct gf %
            correct = metrics[trial_type_keys[trial_type]]
            results[condition][sub_id] = correct


    # Prepare data for DataFrame
    data = []
    for condition, subject_dict in results.items():
        for subject_id, correct in subject_dict.items():
            data.append({'Subject': subject_id, 'Condition': condition, 'Correct': correct})

    # Create DataFrame
    df_correct = pd.DataFrame(data)

    # Pivot the data to get dbs_off and dbs_on side by side
    pivot_df = df_correct.pivot(index='Subject', columns='Condition', values='Correct')

    # Extract the arrays
    #dbs_off = pivot_df['dbs_off'].values
    #dbs_on = pivot_df['dbs_on'].values
    dbs_off = pivot_df.get('DBS OFF', pd.Series(index=pivot_df.index, dtype=float)).values
    dbs_on = pivot_df.get('DBS ON', pd.Series(index=pivot_df.index, dtype=float)).values

    # Handle missing values: Only run Wilcoxon test on pairs
    valid_indices = ~np.isnan(dbs_off) & ~np.isnan(dbs_on)
    if valid_indices.sum() > 0:
        test_result, p_value = scipy.stats.wilcoxon(dbs_off[valid_indices], dbs_on[valid_indices])
    else:
        test_result, p_value = None, None

    # Calculate mean and std
    group_stats = df_correct.groupby('Condition')['Correct'].agg(['mean', 'std'])

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Create a violin plot
    stripplot = sns.violinplot(data=df_correct, x='Condition', y='Correct', hue='Condition', 
                            split=False, inner=None, width=0.6,
                            palette={'DBS OFF': '#20a39e', 'DBS ON': '#ef5b5b'}, 
                            alpha=0.2, legend=False)

    # Retrieve x-coordinates for each condition
    ax = plt.gca()  # Get current axis
    condition_x_positions = {label.get_text(): pos for label, pos in zip(ax.get_xticklabels(), ax.get_xticks())}

    # Jitter strength
    jitter_strength = 0.15

    # Store jittered x-values for each subject
    subject_jitter = {}

    # Overlay subject points
    for subject_id in df_correct['Subject'].unique():
        # Get data for this subject in 'dbs_off' and 'dbs_on' conditions
        subject_data = df_correct[(df_correct['Subject'] == subject_id)]

        if subject_id not in subject_jitter:
            subject_jitter[subject_id] = np.random.uniform(-jitter_strength, jitter_strength)
        
        jitter_value = subject_jitter[subject_id]
        
        x_coords = []
        y_coords = []
        
        for _, row in subject_data.iterrows():
            cond = row['Condition']
            x_coords.append(condition_x_positions[cond] + jitter_value)
            y_coords.append(row['Correct'])

        # Plot single values
        if len(y_coords) == 1:
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder=100)
        else:
            plt.plot(x_coords, y_coords, color='gray', alpha=0.5)
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder=100)


    # Annotate statistical results
    if p_value is not None:
        annotation = "ns (Wilcoxon signed-rank test)" if p_value >= 0.05 else "* (Wilcoxon signed-rank test)"
        max_val = df_correct['Correct'].max()
        plt.text(0.5, 10, f"statistic = {test_result}\n p = {p_value}\n{annotation}", 
                ha='center', va='bottom', color='black', fontsize=14)

    # Add labels, title, and legend
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel(f'Correct {trial_type} (%)', fontsize=14)
    plt.title(f'Percentage correct {trial_type} trials', fontsize=16)
    plt.ylim(0,120)

    # Increase tick label font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Move the legend outside the plot area to avoid overlap
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Create custom legend for subjects
    legend_patches = [matplotlib.patches.Patch(color=subject_colors[subj], label=subj) for subj in df_correct['Subject'].unique()]
    plt.legend(handles=legend_patches, title="Subjects", fontsize=12, title_fontsize=14, loc="upper right", bbox_to_anchor=(1.2, 1))

    # Improve layout to fit everything
    plt.tight_layout()

    plt.savefig(join(behav_results_saving_path, f'percent_correct_{trial_type}.png'), dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()





# create a function to plot off_vs_on percent success:
def plot_reaction_time_on_vs_off(
        stats_OFF,
        stats_ON,
        trial_type,
        subject_colors,
        behav_results_saving_path,
        show_fig = False
):
    
    # Define conditions and corresponding dictionaries
    conditions = {
        'dbs_off': stats_OFF,
        'dbs_on': stats_ON,
    }

    trial_type_keys = {'GO': 'go_trial mean RT (ms)',
            'GC':  'go_continue_trial mean RT (ms)',
            'GF': 'go_fast_trial mean RT (ms)',
            'Go-STOP': 'stop_trial mean RT (ms)'}

    # Initialize dictionaries to hold results for each condition
    results = {condition: {} for condition in conditions.keys()}

    # Loop through each condition and subject
    for condition, data_dict in conditions.items():
        for subject_id, metrics in data_dict.items():
            # Extract the subject ID (first part of subject_id before the first space)
            sub_id = subject_id.split()[0]
            # Calculate the desired  correct gf %
            rt = metrics[trial_type_keys[trial_type]]
            results[condition][sub_id] = rt


    # Prepare data for DataFrame
    data = []
    for condition, subject_dict in results.items():
        for subject_id, rt in subject_dict.items():
            data.append({'Subject': subject_id, 'Condition': condition, 'mean RT (ms)': rt})

    # Create DataFrame
    df_rt = pd.DataFrame(data)

    # Pivot the data to get dbs_off and dbs_on side by side
    pivot_df = df_rt.pivot(index='Subject', columns='Condition', values='mean RT (ms)')

    # Extract the arrays
    #dbs_off = pivot_df['dbs_off'].values
    #dbs_on = pivot_df['dbs_on'].values
    dbs_off = pivot_df.get('dbs_off', pd.Series(index=pivot_df.index, dtype=float)).values
    dbs_on = pivot_df.get('dbs_on', pd.Series(index=pivot_df.index, dtype=float)).values

    # Handle missing values: Only run Wilcoxon test on pairs
    valid_indices = ~np.isnan(dbs_off) & ~np.isnan(dbs_on)
    if valid_indices.sum() > 0:
        test_result, p_value = scipy.stats.wilcoxon(dbs_off[valid_indices], dbs_on[valid_indices])
    else:
        test_result, p_value = None, None

    # Calculate mean and std
    group_stats = df_rt.groupby('Condition')['mean RT (ms)'].agg(['mean', 'std'])


    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Create a violin plot
    stripplot = sns.violinplot(data=df_rt, x='Condition', y='mean RT (ms)', hue='Condition', 
                            split=False, inner=None, width=0.6,
                            palette={'dbs_off': '#20a39e', 'dbs_on': '#ef5b5b'}, 
                            alpha=0.2, legend=False)

    # Retrieve x-coordinates for each condition
    ax = plt.gca()  # Get current axis
    condition_x_positions = {label.get_text(): pos for label, pos in zip(ax.get_xticklabels(), ax.get_xticks())}

    # Jitter strength
    jitter_strength = 0.15

    # Store jittered x-values for each subject
    subject_jitter = {}

    # Overlay subject points
    for subject_id in df_rt['Subject'].unique():
        # Get data for this subject in 'dbs_off' and 'dbs_on' conditions
        subject_data = df_rt[(df_rt['Subject'] == subject_id)]

        if subject_id not in subject_jitter:
            subject_jitter[subject_id] = np.random.uniform(-jitter_strength, jitter_strength)
        
        jitter_value = subject_jitter[subject_id]
        
        x_coords = []
        y_coords = []
        
        for _, row in subject_data.iterrows():
            cond = row['Condition']
            x_coords.append(condition_x_positions[cond] + jitter_value)
            y_coords.append(row['mean RT (ms)'])

        # Plot single values
        if len(y_coords) == 1:
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder=100)
        else:
            plt.plot(x_coords, y_coords, color='gray', alpha=0.5)
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder=100)

    # Annotate statistical results
    if p_value is not None:
        annotation = "ns (Wilcoxon signed-rank test)" if p_value >= 0.05 else "* (Wilcoxon signed-rank test)"
        max_val = df_rt['mean RT (ms)'].max()
        plt.text(0.5, 1000, f"statistic = {test_result}\n p = {p_value}\n{annotation}", 
                ha='center', va='bottom', color='black', fontsize=14)

    # Add labels, title, and legend
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel(f'Mean RT {trial_type} (ms)', fontsize=14)
    plt.title(f'Mean reaction time {trial_type} trials', fontsize=16)

    # Increase tick label font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(300,1200)

    # Move the legend outside the plot area to avoid overlap
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Create custom legend for subjects
    legend_patches = [matplotlib.patches.Patch(color=subject_colors[subj], label=subj) for subj in df_rt['Subject'].unique()]
    plt.legend(handles=legend_patches, title="Subjects", fontsize=12, title_fontsize=14, loc="upper right", bbox_to_anchor=(1.2, 1))

    # Improve layout to fit everything
    plt.tight_layout()

    plt.savefig(join(behav_results_saving_path, f'reaction_time_{trial_type}.png'), dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()



def plot_dbs_effect_reaction_time_single_sub(
        stats_OFF,
        stats_ON,
        behav_results_saving_path,
        show_fig = False
):

    # extract subjects if they are present in both dbs conditions:
    subjects_dbs = []
    subject_IDs_OFF = []
    subject_IDs_ON = []

    for subject in stats_OFF:
        subject_ID = subject.split(' ')[0]
        subject_IDs_OFF.append(subject_ID)


    for subject in stats_ON:
        subject_ID = subject.split(' ')[0]
        subject_IDs_ON.append(subject_ID)


    for i in subject_IDs_OFF:
        if i in subject_IDs_ON:
            subjects_dbs.append(i)

    # in the case of the subjects that are present in both dbs conditions, create for each subject a dictionary holding the values from both dbs conditions:
    stats_dbs = {}

    for subject in subjects_dbs:
        stats_dbs[subject] = {}
        for condition in stats_OFF:
            if subject in condition:
                stats_dbs[subject]['OFF'] = stats_OFF[condition]
        for condition in stats_ON:
            if subject in condition:
                stats_dbs[subject]['ON'] = stats_ON[condition]

    # Define trial types and create index
    trial_types = ['go_trial', 'stop_trial', 'go_fast_trial', 'go_continue_trial']
    index = np.arange(len(trial_types))
    stats_dbs_new = {}

    for subject_id in stats_dbs.keys():
        stats_dbs_new = {
            'OFF': stats_dbs[subject_id]['OFF'],
            'ON': stats_dbs[subject_id]['ON']
        }

        # Gather data for each trial type in a DataFrame
        data = []
        for trial_type, off, on in zip(trial_types, 
                                        [stats_dbs_new['OFF'][f"{tt} RTs (ms)"] for tt in trial_types],
                                        [stats_dbs_new['ON'][f"{tt} RTs (ms)"] for tt in trial_types]):
            data.extend([(trial_type, 'DBS OFF', val) for val in off])
            data.extend([(trial_type, 'DBS ON', val) for val in on])

        df = pd.DataFrame(data, columns=['Trial Type', 'Condition', 'Reaction Time'])

        # Initialize the plot
        plt.figure(figsize=(10, 6))

        # Plot violins with seaborn, setting hue for the two conditions
        sns.violinplot(data=df, x='Trial Type', y='Reaction Time', hue='Condition', 
                        split=False, inner=None, palette={'DBS OFF': '#20a39e', 'DBS ON': '#ef5b5b'}, alpha=0.2, legend=False)

        # Overlay data points with a strip plot
        sns.stripplot(data=df, x='Trial Type', y='Reaction Time', hue='Condition', 
                        dodge=True, jitter=True, color='black', marker='o', size=3, 
                        palette={'DBS OFF': '#20a39e', 'DBS ON': '#ef5b5b'}, legend=True)

        # Calculate and plot means manually for each condition and trial type
        for i, trial_type in enumerate(trial_types):
            for condition in ['DBS OFF', 'DBS ON']:
                condition_data = df[(df['Trial Type'] == trial_type) & (df['Condition'] == condition)]
                mean_value = condition_data['Reaction Time'].mean()
                plt.scatter(x=(i - 0.2 if condition == 'DBS OFF' else i + 0.2), 
                            y=mean_value, color='black', marker="_", s=200, label=f'{condition} Mean' if i == 0 else "", zorder=5)

            # Perform Mann-Whitney U test between the two conditions (DBS OFF vs DBS ON) for each trial type
            off_data = df[(df['Trial Type'] == trial_type) & (df['Condition'] == 'DBS OFF')]['Reaction Time']
            on_data = df[(df['Trial Type'] == trial_type) & (df['Condition'] == 'DBS ON')]['Reaction Time']
            
            # Perform Mann-Whitney U test
            if len(off_data) >= 2 and len(on_data) >= 2:  # Minimum data size for the test
                stat, p_value = scipy.stats.mannwhitneyu(off_data, on_data, alternative='two-sided')
                if p_value < 0.05:  # If significant difference
                    max_val = max(off_data.max(), on_data.max())
                    plt.text(i, max_val + 10, '*', ha='center', va='bottom', color='black', fontsize=16)
                else:
                    max_val = max(off_data.max(), on_data.max())
                    plt.text(i, max_val + 10, 'ns', ha='center', va='bottom', color='black', fontsize=16)

        # Customize plot
        plt.xlabel('Trial Type')
        plt.ylabel('Reaction Time (ms)')
        plt.title(f'Reaction Times for Subject {subject_id}')
        plt.legend(title='Condition', loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(join(behav_results_saving_path, f"Effect of DBS on success rate - {subject_id}.png"), dpi=300)
        if show_fig:
            plt.show()
        else:
            plt.close()



def plot_dbs_effect_on_rt_all_sub_with_2_sessions_all_trial_types(
        stats_OFF,
        stats_ON,
        subject_colors,
        behav_results_saving_path,
        show_fig = False
):

    # extract subjects if they are present in both dbs conditions:
    subjects_dbs = []
    subject_IDs_OFF = []
    subject_IDs_ON = []

    for subject in stats_OFF:
        subject_ID = subject.split(' ')[0]
        subject_IDs_OFF.append(subject_ID)

    for subject in stats_ON:
        subject_ID = subject.split(' ')[0]
        subject_IDs_ON.append(subject_ID)

    for i in subject_IDs_OFF:
        if i in subject_IDs_ON:
            subjects_dbs.append(i)

    # in the case of the subjects that are present in both dbs conditions, create for each subject a dictionary holding the values from both dbs conditions:
    stats_dbs = {}

    for subject in subjects_dbs:
        stats_dbs[subject] = {}
        for condition in stats_OFF:
            if subject in condition:
                stats_dbs[subject]['OFF'] = stats_OFF[condition]
        for condition in stats_ON:
            if subject in condition:
                stats_dbs[subject]['ON'] = stats_ON[condition]

    # Define trial types
    trial_types = ['go_trial', 'stop_trial', 'go_fast_trial', 'go_continue_trial']

    # Initialize an empty list to store aggregated data
    reaction_time_data = []

    # Loop over subjects in stats_dbs
    for subject_id in stats_dbs.keys():
        for trial_type in trial_types:
            # Compute mean reaction times for OFF and ON conditions
            mean_off_rt = np.mean(stats_dbs[subject_id]['OFF'][f"{trial_type} RTs (ms)"])
            #mean_off_rt = np.median(stats_dbs[subject_id]['OFF'][f"{trial_type} RTs (ms)"]) 
            mean_on_rt = np.mean(stats_dbs[subject_id]['ON'][f"{trial_type} RTs (ms)"])
            #mean_on_rt = np.median(stats_dbs[subject_id]['ON'][f"{trial_type} RTs (ms)"])

            # Append mean values to the list
            reaction_time_data.append([subject_id, trial_type, mean_off_rt, mean_on_rt])

    # Convert list to DataFrame
    reaction_time_df = pd.DataFrame(reaction_time_data, columns=['subject_id', 'trial_type', 'off_rt', 'on_rt'])

    # Get unique trial types
    trial_types = reaction_time_df['trial_type'].unique()

    # Perform paired t-tests for each trial type
    for trial_type in trial_types:
        trial_data = reaction_time_df[reaction_time_df['trial_type'] == trial_type]    
        t_stat, p_value = scipy.stats.ttest_rel(trial_data['off_rt'], trial_data['on_rt'])
        # calculate the correlation coefficient between OFF and ON conditions
        corr_coef, p_corr = scipy.stats.pearsonr(trial_data['off_rt'], trial_data['on_rt'])
        print(f"Trial Type: {trial_type}, t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}, Correlation Coefficient: {corr_coef:.3f}")
        # plot the correlation for each trial type
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=trial_data['off_rt'], y=trial_data['on_rt'], hue = trial_data['subject_id'], edgecolor='black', s=100)
        plt.title(f'Correlation between OFF and ON RTs for {trial_type}\n Correlation Coefficient: {corr_coef:.3f}, p= {p_corr:.3f}')
        plt.xlabel('OFF RT (ms)')
        plt.ylabel('ON RT (ms)')
        plt.savefig(join(behav_results_saving_path, f'Correlation_OFF_ON_{trial_type}.png'), dpi=300)
        if show_fig:
            plt.show()
        else:
            plt.close()


    # Prepare the data in a long format suitable for seaborn's violin plot
    plot_data = []

    for trial_type in trial_types:
        trial_data = reaction_time_df[reaction_time_df['trial_type'] == trial_type]

        # Add data for OFF and ON conditions in long format
        trial_data_off = trial_data[['subject_id', 'off_rt']].rename(columns={'off_rt': 'reaction_time'})
        trial_data_off['condition'] = 'OFF'

        trial_data_on = trial_data[['subject_id', 'on_rt']].rename(columns={'on_rt': 'reaction_time'})
        trial_data_on['condition'] = 'ON'

        # Combine both conditions into one DataFrame
        trial_data_combined = pd.concat([trial_data_off, trial_data_on])

        # Add trial type information for plotting
        trial_data_combined['trial_type'] = trial_type
        
        plot_data.append(trial_data_combined)

    # Combine all trial data into one dataframe
    plot_data = pd.concat(plot_data)

    valid_subjects = plot_data['subject_id'].unique()

    # Set up the plot
    plt.figure(figsize=(12, 10))

    # Create a violin plot for each trial type
    sns.violinplot(x='trial_type', y='reaction_time', hue='condition', data=plot_data, split=True, 
                palette={'OFF': '#20a39e', 'ON': '#ef5b5b'}, alpha=0.2, inner='quart', linewidth=1.25)

    # Initialize lists for legend handles and labels
    subject_handles = []
    subject_labels = []

    # Add colored dots and connecting lines for each participant
    for i, trial_type in enumerate(trial_types):
        trial_data_for_dots = plot_data[plot_data['trial_type'] == trial_type]
        
        # Create a color map for each participant
        #subject_colors = {subject: sns.color_palette("deep", len(trial_data_for_dots['subject_id'].unique()))[i] 
                        #for i, subject in enumerate(trial_data_for_dots['subject_id'].unique())}

        for subject_id in valid_subjects:
            subject_data = trial_data_for_dots[trial_data_for_dots['subject_id'] == subject_id]
            
            # Ensure OFF and ON values are present
            if len(subject_data) == 2:
                off_value = subject_data[subject_data['condition'] == 'OFF']['reaction_time'].values[0]
                on_value = subject_data[subject_data['condition'] == 'ON']['reaction_time'].values[0]
                
                # Offset x-positions: slightly left for OFF (-0.15) and right for ON (+0.15)
                x_pos = [i - 0.15, i + 0.15]
                y_pos = [off_value, on_value]
                
                # Plot small line connecting the OFF and ON dots
                plt.plot(x_pos, y_pos, color=subject_colors[subject_id], linestyle='-', linewidth=1.5, alpha=0.7)
            
            # Scatter plot for each participant's result
            scatter = plt.scatter(
                [i - 0.15, i + 0.15], 
                subject_data['reaction_time'], 
                color=subject_colors[subject_id], edgecolors='black', s=100
            )
            
            # Add to the legend (only add each subject once)
            if subject_id not in subject_labels:
                subject_handles.append(scatter)
                subject_labels.append(subject_id)

    """
    # Add t-test results to each subplot
    for i, trial_type in enumerate(trial_types):
        trial_data_for_ttest = plot_data[plot_data['trial_type'] == trial_type]
        t_stat, p_value = scipy.stats.ttest_rel(
            trial_data_for_ttest[trial_data_for_ttest['condition'] == 'OFF']['reaction_time'],
            trial_data_for_ttest[trial_data_for_ttest['condition'] == 'ON']['reaction_time']
        )
        
        plt.text(i, plot_data['reaction_time'].max() + 10, f"t = {t_stat:.3f}\np = {p_value:.3f}", 
                horizontalalignment='center', fontsize=12, verticalalignment='bottom')"
    """

    # Add mannwhitneyu test results to each subplots
    for i, trial_type in enumerate(trial_types):
        trial_data_for_test = plot_data[plot_data['trial_type'] == trial_type]
        stat, p_value = scipy.stats.wilcoxon(
            trial_data_for_test[trial_data_for_test['condition'] == 'OFF']['reaction_time'],
            trial_data_for_test[trial_data_for_test['condition'] == 'ON']['reaction_time']
        )
        
        plt.text(i, plot_data['reaction_time'].max() + 10, f"stat = {stat:.3f}\np = {p_value:.3f}", 
                horizontalalignment='center', fontsize=12, verticalalignment='bottom')


    # Set labels and title
    plt.xlabel('Trial Type', fontsize=14)
    plt.ylabel('Reaction Time (ms)', fontsize=14)
    plt.title('Reaction Time Comparison Between OFF and ON Conditions', fontsize=16)

    # Custom legend for OFF and ON condition colors
    from matplotlib.patches import Patch
    condition_legend_handles = [
        Patch(color='#20a39e', label='OFF'),
        Patch(color='#ef5b5b', label='ON')
    ]

    # Create two legends: one for OFF/ON conditions, one for subjects
    legend1 = plt.legend(handles=condition_legend_handles, title="Condition", loc='upper right', fontsize=12)
    plt.gca().add_artist(legend1)  # Ensure the first legend stays

    # Second legend for subject IDs
    plt.legend(handles=subject_handles, labels=subject_labels, title="Subject ID", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot
    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, f"Effect of DBS on RT - all trials - all subjects.png"), dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()








# create a function to plot off_vs_on percent success:
def plot_early_press_on_vs_off(
        stats_OFF,
        stats_ON,
        subject_colors,
        behav_results_saving_path,
        show_fig = False
):
    
    # Define conditions and corresponding dictionaries
    conditions = {
        'dbs_off': stats_OFF,
        'dbs_on': stats_ON,
    }

    # Initialize dictionaries to hold results for each condition
    results = {condition: {} for condition in conditions.keys()}

    # Loop through each condition and subject
    for condition, data_dict in conditions.items():
        for subject_id, metrics in data_dict.items():
            # Extract the subject ID (first part of subject_id before the first space)
            sub_id = subject_id.split()[0]
            # Calculate the desired  correct gf %
            early = metrics['early presses']
            results[condition][sub_id] = early


    # Prepare data for DataFrame
    data = []
    for condition, subject_dict in results.items():
        for subject_id, rt in subject_dict.items():
            data.append({'Subject': subject_id, 'Condition': condition, 'early presses': rt})

    # Create DataFrame
    df_early = pd.DataFrame(data)

    # Pivot the data to get dbs_off and dbs_on side by side
    pivot_df = df_early.pivot(index='Subject', columns='Condition', values='early presses')

    # Extract the arrays
    #dbs_off = pivot_df['dbs_off'].values
    #dbs_on = pivot_df['dbs_on'].values
    dbs_off = pivot_df.get('dbs_off', pd.Series(index=pivot_df.index, dtype=float)).values
    dbs_on = pivot_df.get('dbs_on', pd.Series(index=pivot_df.index, dtype=float)).values

    # Handle missing values: Only run Wilcoxon test on pairs
    valid_indices = ~np.isnan(dbs_off) & ~np.isnan(dbs_on)
    if valid_indices.sum() > 0:
        test_result, p_value = scipy.stats.wilcoxon(dbs_off[valid_indices], dbs_on[valid_indices])
    else:
        test_result, p_value = None, None

    # Calculate mean and std
    group_stats = df_early.groupby('Condition')['early presses'].agg(['mean', 'std'])
    print("Mean and standard deviation for each condition:")
    print(group_stats)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Create a violin plot
    stripplot = sns.violinplot(data=df_early, x='Condition', y='early presses', hue='Condition', 
                            split=False, inner=None, width=0.6,
                            palette={'dbs_off': '#20a39e', 'dbs_on': '#ef5b5b'}, 
                            alpha=0.2, legend=False)

    # Retrieve x-coordinates for each condition
    ax = plt.gca()  # Get current axis
    condition_x_positions = {label.get_text(): pos for label, pos in zip(ax.get_xticklabels(), ax.get_xticks())}

    # Jitter strength
    jitter_strength = 0.15

    # Store jittered x-values for each subject
    subject_jitter = {}

    # Overlay subject points
    for subject_id in df_early['Subject'].unique():
        # Get data for this subject in 'dbs_off' and 'dbs_on' conditions
        subject_data = df_early[(df_early['Subject'] == subject_id)]

        if subject_id not in subject_jitter:
            subject_jitter[subject_id] = np.random.uniform(-jitter_strength, jitter_strength)
        
        jitter_value = subject_jitter[subject_id]
        
        x_coords = []
        y_coords = []
        
        for _, row in subject_data.iterrows():
            cond = row['Condition']
            x_coords.append(condition_x_positions[cond] + jitter_value)
            y_coords.append(row['early presses'])

        # Plot single values
        if len(y_coords) == 1:
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder=100)
        else:
            plt.plot(x_coords, y_coords, color='gray', alpha=0.5)
            plt.scatter(x_coords, y_coords, color=subject_colors[subject_id], edgecolor='black', s=100, zorder=100)


    # Annotate statistical results
    if p_value is not None:
        annotation = "ns (Wilcoxon signed-rank test)" if p_value >= 0.05 else "* (Wilcoxon signed-rank test)"
        max_val = df_early['early presses'].max()
        plt.text(0.5, max_val + 1, f"statistic = {test_result}\n p = {p_value}\n{annotation}", 
                ha='center', va='bottom', color='black', fontsize=14)

    # Add labels, title, and legend
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel(f'Number of early presses', fontsize=14)
    plt.title(f'Number of early presses', fontsize=16)

    # Increase tick label font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Move the legend outside the plot area to avoid overlap
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Create custom legend for subjects
    legend_patches = [matplotlib.patches.Patch(color=subject_colors[subj], label=subj) for subj in df_early['Subject'].unique()]
    plt.legend(handles=legend_patches, title="Subjects", fontsize=12, title_fontsize=14, loc="upper right", bbox_to_anchor=(1.2, 1))

    # Improve layout to fit everything
    plt.tight_layout()
    plt.savefig(join(behav_results_saving_path, f'Early presses.png'), dpi=300)
    if show_fig:
        plt.show()
    else:
        plt.close()