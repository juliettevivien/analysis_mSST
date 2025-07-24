import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import find_peaks

from functions.io import load_data_from_json
from functions.stats_tests import compute_mean_std

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Function to process a single session
def process_session(
        session_data, 
        side, 
        freq_prefix, 
        condition1_key, 
        condition2_key, 
        time_key="amp_times"
        ):
    
    # Construct keys dynamically based on the side and condition
    cond1_key = f"{side}_STN_{freq_prefix}_amp_{condition1_key}"
    cond2_key = f"{side}_STN_{freq_prefix}_amp_{condition2_key}"

    if cond1_key in session_data and cond2_key in session_data and time_key in session_data:
        cond1_values = np.array(session_data[cond1_key])
        cond2_values = np.array(session_data[cond2_key])
        time_values = np.array(session_data[time_key])
        
        return (cond1_values, cond2_values, time_values)
    return None, None, None





def find_significant_onset(diff_series, time_points, threshold_high):
    # Find indices where diff_series exceeds threshold_high
    significant_indices_high = np.where(diff_series > threshold_high)[0]

    # Check if 20 consecutive points remain above the threshold
    for idx in significant_indices_high:
        #print(diff_series[idx:idx+20])
        if idx + 20 <= len(diff_series) and (diff_series[idx:idx+20] > threshold_high).all():
            return time_points[idx]  # Return first time point meeting criteria
    
    return None  # No significant onset found






# Function to plot data
def plot_group(
        title, 
        freq_prefix,
        cond1_label,
        cond1_mean, 
        cond1_var, 
        cond2_label,
        cond2_mean, 
        cond2_var, 
        diff_mean, 
        diff_var, 
        time_values, 
        saving_path, 
        behav_dict,
        mean_significant=0, 
        significant_var=0
        ):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # First plot (both trial types overlapped)
    ax1 = axes[0]

    # find min and max for the plot limits:
    ylim_min = round(np.min([np.min(cond1_mean), np.min(cond2_mean)]) - 5)
    ylim_max = round(np.max([np.max(cond1_mean), np.max(cond2_mean)]) + 5)

    ax1.plot(time_values, cond1_mean, label=cond1_label, color="grey")
    ax1.fill_between(time_values, cond1_mean - cond1_var, cond1_mean + cond1_var, color="grey", alpha=0.3)
    ax1.plot(time_values, cond2_mean, label=cond2_label, color="black")
    ax1.fill_between(time_values, cond2_mean - cond2_var, cond2_mean + cond2_var, color="black", alpha=0.5)
    ax1.set_ylabel(f"{freq_prefix} amplitude (%)", fontsize=14)
    ax1.set_title(title)
    ax1.legend()
    #ax1.set_ylim(-5,10)
    ax1.set_ylim(ylim_min, ylim_max)

    # Second plot (Difference between cond1 and cond2)
    ax2 = axes[1]

    # find min and max for the second plot limits:
    ylim_min_2 = round(np.min(diff_mean)) - 5
    ylim_max_2 = round(np.max(diff_mean)) + 5

    ax2.plot(time_values, diff_mean, label=f"{cond1_label} - {cond2_label}", color='grey')
    ax2.fill_between(time_values, diff_mean - diff_var, diff_mean + diff_var, color="grey", alpha=0.3)
    ax2.set_xlabel("Time from GO cue (ms)", fontsize=14)
    ax2.set_ylabel(f"{freq_prefix} amplitude (%)", fontsize=14)
    #ax2.set_ylim(-5,10)
    ax2.set_ylim(ylim_min_2, ylim_max_2)


    # Check if "SSD+SSRT" is present
    exclude_ssrt = "SSD+SSRT" in behav_dict
    # Filter the dictionary
    filtered_behav_dict = {k: v for k, v in behav_dict.items() if "_var" not in k and not (exclude_ssrt and k == "SSRT (ms)")}
    i=1

    for behav_key in filtered_behav_dict.keys():
        mean_value = behav_dict[behav_key]
        var_value = behav_dict[behav_key + "_var"]
        ax1.errorbar(mean_value, ylim_max - 4, xerr=var_value, fmt='o', color='black', ms=3)
        ax1.annotate(behav_key,
                     xy=(mean_value, ylim_max - 4),  # Point position
                     xytext=(mean_value, ylim_max - 3 + i),  # Text position
                     fontsize=10, color='black',
                     ha='center')
        
        ax2.errorbar(mean_value, ylim_max_2 - 4, xerr=var_value, fmt='o', color='black', ms=3)
        ax2.annotate(behav_key,
                     xy=(mean_value, ylim_max_2 - 4),  # Point position
                     xytext=(mean_value, ylim_max_2 - 3 + i),  # Text position
                     fontsize=10, color='black',
                     ha='center')
        
        i += 1
    ax2.legend()

    ax2.errorbar(mean_significant, ylim_min_2 + 4, xerr=significant_var, fmt='o', color='grey', ms=3)  # Add the point with variance
    ax2.annotate("p < 0.05",
                xy=(mean_significant, ylim_min_2 + 4),  # Point position
                xytext=(mean_significant, ylim_min_2 + 2),  # Text position
                fontsize=10, color='grey',
                ha='center')
    ax2.legend()
    ax1.set_xlim(-500, 1500)
    ax2.set_xlim(-500, 1500)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(saving_path, f"{title}.png"))
    plt.close()






def plot_on_off(        
        title, 
        freq_prefix,
        cond1_label,
        cond1_mean_off,
        cond1_var_off,
        cond1_mean_on,
        cond1_var_on,  
        cond2_label,
        cond2_mean_off,
        cond2_var_off,
        cond2_mean_on,
        cond2_var_on,  
        diff_mean_off,
        diff_var_off,
        diff_mean_on,
        diff_var_on,  
        time_values, 
        saving_path, 
        behav_dict,
        mean_significant=0, 
        significant_var=0
        ):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # First plot (both trial types overlapped)
    ax1 = axes[0]

    ax1.plot(time_values, cond1_mean_off, label=f"{cond1_label} - OFF", color="grey")
    ax1.fill_between(time_values, cond1_mean_off - cond1_var_off, cond1_mean_off + cond1_var_off, color="grey", alpha=0.3)
    ax1.plot(time_values, cond2_mean_off, label=f"{cond2_label} - OFF", color="black")
    ax1.fill_between(time_values, cond2_mean_off - cond2_var_off, cond2_mean_off + cond2_var_off, color="black", alpha=0.5)

    ax1.plot(time_values, cond1_mean_on, label=f"{cond1_label} - ON", color="lightblue")
    ax1.fill_between(time_values, cond1_mean_on - cond1_var_on, cond1_mean_on + cond1_var_on, color="lightblue", alpha=0.3)
    ax1.plot(time_values, cond2_mean_on, label=f"{cond2_label} - ON", color="darkblue")
    ax1.fill_between(time_values, cond2_mean_on - cond2_var_on, cond2_mean_on + cond2_var_on, color="darkblue", alpha=0.5)
    ax1.set_ylabel(f"{freq_prefix} amplitude (%)", fontsize=14)
    ax1.set_title(title)
    ax1.legend()

    # Second plot (Difference between cond1 and cond2)
    ax2 = axes[1]
    ax2.plot(time_values, diff_mean_off, label=f"{cond1_label} - {cond2_label} - DBS OFF", color='grey')
    ax2.fill_between(time_values, diff_mean_off - diff_var_off, diff_mean_off + diff_var_off, color="grey", alpha=0.3)
    ax2.plot(time_values, diff_mean_on, label=f"{cond1_label} - {cond2_label} - DBS ON", color='lightblue')
    ax2.fill_between(time_values, diff_mean_on - diff_var_on, diff_mean_on + diff_var_on, color="lightblue", alpha=0.3)    
    ax2.set_xlabel("Time from GO cue (ms)", fontsize=14)
    ax2.set_ylabel(f"{freq_prefix} amplitude (%)", fontsize=14)

    
    # Check if "SSD+SSRT" is present
    exclude_ssrt = "SSD+SSRT" in behav_dict
    # Filter the dictionary
    filtered_behav_dict = {k: v for k, v in behav_dict.items() if "_var" not in k and not (exclude_ssrt and k == "SSRT (ms)")}
    i=1

    for behav_key in filtered_behav_dict.keys():
        #print(behav_key)
        mean_value = behav_dict[behav_key]
        var_value = behav_dict[behav_key + "_var"]
        ax1.errorbar(x=mean_value, y=2, xerr=var_value, fmt='o', color='black', ms=3)
        ax1.annotate(behav_key,
                     xy=(mean_value, 2),  # Point position
                     xytext=(mean_value, 2 + i),  # Text position
                     fontsize=10, color='black',
                     ha='center')
        
        ax2.errorbar(x=mean_value, y=-2, xerr=var_value, fmt='o', color='black', ms=3)
        ax2.annotate(behav_key,
                     xy=(mean_value, -2),  # Point position
                     xytext=(mean_value, -2 - i),  # Text position
                     fontsize=10, color='black',
                     ha='center')
        i += 1
    
    ax2.legend()
    
    ax2.errorbar(mean_significant, 0, xerr=significant_var, fmt='o', color='grey', ms=3)  # Add the point with variance
    ax2.annotate("p < 0.05",
                xy=(mean_significant, 10),  # Point position
                xytext=(mean_significant, 10 - 2),  # Text position
                fontsize=10, color='grey',
                ha='center')
    ax2.legend()
    
    ax1.set_xlim(-500, 1500)
    ax2.set_xlim(-500, 1500)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(saving_path, f"{title}.png"))




def compare_freq_response_2_cond(
    cond1_label, # "GO_successful"
    cond2_label, # "GF_successful"
    freq_prefix, # "high-beta"
    dbs_status, # = "DBS ON" or "DBS OFF"
    behav_keys, # = [#"mean SSD (ms)", "SSRT (ms)" or "go_trial mean RT (ms)", "go_fast_trial mean RT (ms)"]
    sub_dict_freq_response: dict,
    sub_dict_stats: dict,
    saving_path: str   
):


    # Initialize dictionary to store data
    data = {
            "Left": {
                "cond1": [], "cond2": [], "cond1-cond2": [], "significant_point":[]
                }, 
            "Right": {
                "cond1": [], "cond2": [], "cond1-cond2": [], "significant_point":[]
                }, 
            "n_sub":0
            }

    behav_dict = {
            "mean SSD (ms)": [],
            "SSRT (ms)": [],
            "SSD+SSRT": [],
            "go_trial mean RT (ms)": [],
            "go_fast_trial mean RT (ms)": [],
            #"stop_trial mean RT (ms)": []
            }

    behav_dict_stats = {}

    for subject, session_data in sub_dict_freq_response.items():
        if dbs_status in subject:
            data["n_sub"] += 1
            #dbs_status = (subject.split()[1] + ' ' + subject.split()[2])
            for behav_key in behav_keys:
                    behav_dict[behav_key].append(sub_dict_stats[subject][behav_key])
            if "mean SSD (ms)" in behav_keys and "SSRT (ms)" in behav_keys:
                behav_dict["SSD+SSRT"].append(sub_dict_stats[subject]["mean SSD (ms)"] + sub_dict_stats[subject]["SSRT (ms)"])

        # Process left and right sides for the extracted condition
            for side in ["Left", "Right"
                         ]:
                (
                    cond1_values, cond2_values, time_values
                    ) = process_session(
                        session_data = session_data,
                        side = side,
                        freq_prefix=freq_prefix,
                        condition1_key=cond1_label,
                        condition2_key=cond2_label,
                        time_key="amp_times"
                        )

                time_values = time_values*1000  # Convert to milliseconds

                # Find the peak latency for each condition:
                period_of_interest = (time_values >= 0) & (time_values <= 1500)
                time_interest = time_values[period_of_interest]
                cond1_interest = cond1_values[period_of_interest]
                cond2_interest = cond2_values[period_of_interest]
                peak_latency_cond1 = find_peaks(-cond1_interest)
                peak_latency_cond2 = find_peaks(-cond2_interest)

                pre_stimulus_period = (time_values >= -500) & (time_values <= 0) # pre-stimulus baseline period (500 ms before stimulus onset)
                if cond1_values is not None:
                    data[side]["cond1"].append(cond1_values)
                    data[side]["cond2"].append(cond2_values)
                    data[side]["cond1-cond2"].append(cond1_values - cond2_values)
                    # Compute the mean and standard deviation of the freq response during the pre-stimulus period:
                    baseline_diff = (cond1_values - cond2_values)[pre_stimulus_period]
                    mean_baseline = np.nanmean(baseline_diff)
                    std_baseline = np.std(baseline_diff)
                    # Define the significance threshold (mean ± 1.96 SD from the pre-stimulus baseline)
                    threshold_high = mean_baseline + 1.96 * std_baseline
                    threshold_low = mean_baseline - 1.96 * std_baseline

                    # Filter out values where time <= 0
                    time_indices = (time_values >= 0) & (time_values <= 1500)
                    diff_mean_pos = (cond1_values - cond2_values)[time_indices]  # Subset of diff_mean for time > 0
                    time_values_pos = time_values[time_indices]  # Subset of time values for time > 0
                    onset_latency_diff = find_significant_onset(diff_mean_pos, time_values_pos, threshold_high)
                    if onset_latency_diff:
                        data[side]["significant_point"].append(onset_latency_diff)
                        print("found significant onset latency")
                        #print(onset_latency_diff)
                    else:
                        data[side]["significant_point"].append(np.NaN)
                        print("No significant onset latency found")
                        #print("No significant onset latency for Successful STOP - Latency Matched Go.")

    # Compute stats for each condition and side
    stats = {}
    for side in ["Left", "Right"]:
        stats[side] = {
            "cond1": compute_mean_std(data[side]["cond1"]),
            "cond2": compute_mean_std(data[side]["cond2"]),
            "cond1-cond2": compute_mean_std(data[side]["cond1-cond2"]),
            "significant_point": compute_mean_std(data[side]["significant_point"])
        }

    for behav_key in behav_keys:
        (behav_dict_stats[behav_key], 
        behav_dict_stats[behav_key + "_var"]
        ) = compute_mean_std(
            behav_dict[behav_key]
            )
    
    if "mean SSD (ms)" in behav_keys and "SSRT (ms)" in behav_keys:
        (behav_dict_stats["SSD+SSRT"], 
        behav_dict_stats["SSD+SSRT_var"]
        ) = compute_mean_std(
            behav_dict["SSD+SSRT"]
            )
        

    dbs_cond1 = []
    dbs_cond2 = []
    dbs_cond1_cond2 = []
    var_significant_dbs = []
    significant_dbs = []

    # Plotting Grand Average for each condition
    for side in ["Left", "Right"
                 ]:
        n_sub = data["n_sub"]
        title = f"{dbs_status} - Grand Average {freq_prefix}, {cond1_label} - {cond2_label} ({side.capitalize()}, n = {n_sub})"
        cond1_mean, cond1_var = stats[side]["cond1"]
        cond2_mean, cond2_var = stats[side]["cond2"]
        diff_mean, diff_var = stats[side]["cond1-cond2"]
        mean_significant, significant_var = stats[side]["significant_point"]
        
        plot_group(
            title = title, 
            freq_prefix = freq_prefix, 
            cond1_label=cond1_label,
            cond1_mean=cond1_mean,
            cond1_var=cond1_var,
            cond2_label=cond2_label,
            cond2_mean=cond2_mean,
            cond2_var=cond2_var,
            diff_mean=diff_mean,
            diff_var=diff_var,
            time_values=time_values,
            saving_path=saving_path,
            behav_dict=behav_dict_stats,
            mean_significant=mean_significant,
            significant_var=significant_var
            )
        """
        if on_or_off_cond == "DBS ON" and side == "Left":
            dbs_on_cond1.append(cond1_mean)
            dbs_on_cond2.append(cond2_mean)
            dbs_on_cond1_cond2.append(diff_mean)
            significant_dbs_on.append(mean_significant)
            var_significant_dbs_on.append(significant_var)
        elif on_or_off_cond == "DBS OFF" and side == "Left":
            dbs_off_cond1.append(cond1_mean)
            dbs_off_cond2.append(cond2_mean)
            dbs_off_cond1_cond2.append(diff_mean)
            significant_dbs_off.append(mean_significant)
            var_significant_dbs_off.append(significant_var)

    # Stack the arrays vertically and compute the mean across the arrays
    mean_array_dbs_on_cond1, var_dbs_on_cond1 = np.mean(dbs_on_cond1, axis=0), (np.std(dbs_on_cond1, axis=0)/np.sqrt(len(dbs_on_cond1)))
    mean_array_dbs_on_cond2, var_dbs_on_cond2 = np.mean(dbs_on_cond2, axis=0), (np.std(dbs_on_cond2, axis=0)/np.sqrt(len(dbs_on_cond2)))
    mean_array_dbs_on_cond1_cond2, var_dbs_on_cond1_cond2 = np.mean(dbs_on_cond1_cond2, axis=0), (np.std(dbs_on_cond1_cond2, axis=0)/np.sqrt(len(dbs_on_cond1_cond2)))
    mean_array_dbs_off_cond1, var_dbs_off_cond1 = np.mean(dbs_off_cond1, axis=0), (np.std(dbs_off_cond1, axis=0)/np.sqrt(len(dbs_off_cond1)))
    mean_array_dbs_off_cond2, var_dbs_off_cond2 = np.mean(dbs_off_cond2, axis=0), (np.std(dbs_off_cond2, axis=0)/np.sqrt(len(dbs_off_cond2)))
    mean_array_dbs_off_cond1_cond2, var_dbs_off_cond1_cond2 = np.mean(dbs_off_cond1_cond2, axis=0), (np.std(dbs_off_cond1_cond2, axis=0)/np.sqrt(len(dbs_off_cond1_cond2)))
    mean_significant_dbs_on, var_significant_dbs_on = np.mean(significant_dbs_on), (np.std(significant_dbs_on)/np.sqrt(len(significant_dbs_on)))
    mean_significant_dbs_off, var_significant_dbs_off = np.mean(significant_dbs_off), (np.std(significant_dbs_off)/np.sqrt(len(significant_dbs_off)))


    
    # Plot grand averages with mean SSD and SSRT values
    plot_group(
        title = f"DBS ON - Grand Average {freq_prefix} (Left STN)", 
        cond1_label=cond1_label,
        cond1_mean=mean_array_dbs_on_cond1,
        cond1_var=var_dbs_on_cond1,
        cond2_label=cond2_label,
        cond2_mean=mean_array_dbs_on_cond2,
        cond2_var=var_dbs_on_cond2,
        diff_mean=mean_array_dbs_on_cond1_cond2,
        diff_var=var_dbs_on_cond1_cond2,
        time_values=time_values,
        saving_path=saving_path,
        behav_dict=behav_dict_stats["DBS ON"],
        mean_significant=mean_significant_dbs_on,
        significant_var=var_significant_dbs_on
        )
    
    plot_group(
        title = f"DBS OFF - Grand Average {freq_prefix} (Left STN)", 
        cond1_label=cond1_label,
        cond1_mean=mean_array_dbs_off_cond1,
        cond1_var=var_dbs_off_cond1,
        cond2_label=cond2_label,
        cond2_mean=mean_array_dbs_off_cond2,
        cond2_var=var_dbs_off_cond2,
        diff_mean=mean_array_dbs_off_cond1_cond2,
        diff_var=var_dbs_off_cond1_cond2,
        time_values=time_values,
        saving_path=saving_path,
        behav_dict=behav_dict_stats["DBS OFF"],
        mean_significant=mean_significant_dbs_off,
        significant_var=var_significant_dbs_off
        )

    plot_on_off(        
            title = f"Grand Average {freq_prefix} response - Left STN - DBS ON vs DBS OFF, {cond1_label} - {cond2_label}", 
            freq_prefix = freq_prefix,
            cond1_label = cond1_label,
            cond1_mean_off = mean_array_dbs_off_cond1,
            cond1_var_off = var_dbs_off_cond1,
            cond1_mean_on = mean_array_dbs_on_cond1,
            cond1_var_on = var_dbs_on_cond1,
            cond2_label = cond2_label,
            cond2_mean_off = mean_array_dbs_off_cond2,
            cond2_var_off = var_dbs_off_cond2,
            cond2_mean_on = mean_array_dbs_on_cond2,
            cond2_var_on = var_dbs_on_cond2,
            diff_mean_off = mean_array_dbs_off_cond1_cond2,
            diff_var_off = var_dbs_off_cond1_cond2,
            diff_mean_on = mean_array_dbs_on_cond1_cond2,
            diff_var_on= var_dbs_on_cond1_cond2,
            time_values = time_values,
            saving_path = saving_path,
            behav_dict = behav_dict_stats,
            mean_significant =  0,
            significant_var = 0)
    """

