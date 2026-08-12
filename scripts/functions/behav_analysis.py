from os.path import join
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns


def get_session_order(
    included_subjects,
    subject_info_df,
    behav_results_saving_path    
):

    session_dict = {}
    session_1_ON = []
    session_1_OFF = []

    for subject in included_subjects:
        if subject.startswith('C'):
            continue
        sub = subject.split(' ')[0]
        condition = subject.split(' ')[2]
        # get line corresponding to this subject in excel file, keep only columns starting with 'Session'
        sub_line = subject_info_df[subject_info_df['StudyCode'] == sub].filter(regex='^Session')
        session_1 = sub_line['Session 1'].values[0]
        session_2 = sub_line['Session 2'].values[0]
        if condition in session_1:
            session_dict[subject] = 1
            if condition == 'OFF':
                session_1_OFF.append(subject.split(' ')[0])
            elif condition == 'ON':
                session_1_ON.append(subject.split(' ')[0])
        elif condition in session_2:
            session_dict[subject] = 2
        
    # Also return how many subjects started with first session in OFF condition
    print(f'{len(session_1_OFF)} subjects started with first session in OFF condition')
    print(f'{len(session_1_ON)} subjects started with first session in ON condition')
    # save in a txt file:
    with open(join(behav_results_saving_path, 'session_info.txt'), 'w') as f:
        f.write(f'{len(session_1_OFF)} subjects started with first session in OFF condition:\n')
        f.write(f'{session_1_OFF}\n')
        f.write(f'{len(session_1_ON)} subjects started with first session in ON condition:\n')
        f.write(f'{session_1_ON}\n')


def visualize_scales_scores(
    scale_names,
    sub_scale_dict,
    subs,
    subject_colors,
    color_dict,
    visualize_by,
    saving_path,
    save_as,
    show_plot = False
):
    for scale in scale_names:
        if visualize_by == 'subject':
            print(f'Visualizing {scale} scores for all subjects')
            plt.figure(figsize=(10, 5))
            # extract the value for each subject and plot it as a dot plot, with a different color for each subject
            for sub in subs:
                scale_value = sub_scale_dict[sub][scale]
                # print(f'{sub} : {scale_value}')
                if scale_value is not None and not pd.isna(scale_value):
                    plt.scatter(x=sub, y=abs(scale_value), color=subject_colors[sub], s=100)
            plt.xticks(rotation=90)
            plt.title(f'{scale} scores for all subjects')
            plt.ylabel(f'{scale} score')
            plt.tight_layout()
            plt.savefig(join(saving_path, f'{scale}_scores.{save_as}'), dpi = 300)
            if show_plot:
                plt.show()
            else:
                plt.close()
        elif visualize_by == 'condition':
            if scale in ['UPDRS_ON', 'UPDRS_OFF']:
                continue
            df = build_long_df(sub_scale_dict)
            dotplot_group_comparison(
                df, 
                scale, 
                color_dict,
                saving_path,
                save_as,
                show_plot)    

def build_long_df(sub_scale_dict):
    rows = []

    for subj, scores in sub_scale_dict.items():
        for scale, value in scores.items():
            if subj.startswith('sub'):
                rows.append({
                    'subject': subj,
                    'group': 'patient',
                    'scale': scale,
                    'value': value
                })
            else:
                rows.append({
                    'subject': subj,
                    'group': 'control',
                    'scale': scale,
                    'value': value
                })    

    return pd.DataFrame(rows)


def dotplot_group_comparison(
        df, 
        scale,
        color_dict,
        saving_path,
        save_as,
        show_plot = False
        ):

    scale_df = df[df['scale'] == scale]

    # Statistical test: patients vs controls
    patients = scale_df.loc[scale_df['group'] == 'patient', 'value'].dropna()
    controls = scale_df.loc[scale_df['group'] == 'control', 'value'].dropna()

    U, p = scipy.stats.mannwhitneyu(
        patients,
        controls,
        alternative='two-sided'
    )

    print(f"{scale}: Mann–Whitney U test")
    print(f"Patients n={len(patients)}, Controls n={len(controls)}")
    print(f"U = {U:.2f}, p = {p:.4f}")

    plt.figure(figsize=(6, 5))

    sns.stripplot(
        data=scale_df,
        x='group',
        y='value',
        jitter=0.25,
        size=7,
        palette={'patient': color_dict['DBS ON'],
                 'control': color_dict['control']},
        hue='group',
        legend=False         
    )

    sns.pointplot(
        data=scale_df,
        x='group',
        y='value',
        errorbar='sd',
        linestyle = 'none',
        markers='_',
        color='black'
    )
    plt.suptitle(f"{scale}\n Mean ± std \n Mann–Whitney U test: U = {U:.2f}, p = {p:.4f}", fontsize=10)
    plt.xlabel('')
    plt.ylabel('Score')
    # plt.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(join(saving_path, f'{scale}_scores_by_group.{save_as}'), dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def correlate_two_scales(
    scale_1,
    scale_2,
    sub_scale_dict,
    subject_colors,
    color_dict,
    saving_path,
    save_as,
    show_plot=False
):

    plt.figure(figsize=(10, 6))

    # Store values for each subgroup
    x_control, y_control = [], []
    x_pd, y_pd = [], []

    # Plot individual subjects
    for sub in sub_scale_dict:

        x = sub_scale_dict[sub][scale_1]
        y = sub_scale_dict[sub][scale_2]

        plt.scatter(
            x, y,
            color=subject_colors[sub],
            label=sub
        )

        if sub.startswith("C"):
            x_control.append(x)
            y_control.append(y)
        elif sub.startswith("sub"):
            x_pd.append(x)
            y_pd.append(y)

    # -------- Control regression --------
    if len(x_control) > 1:
        slope, intercept, r, p, _ = scipy.stats.linregress(
            x_control, y_control
        )

        x_fit = np.sort(np.array(x_control))
        y_fit = intercept + slope * x_fit

        plt.plot(
            x_fit,
            y_fit,
            color=color_dict["control"],
            linewidth=2.5,
            label=f"Controls: r={r:.2f}, p={p:.3f}"
        )

    # -------- PD regression --------
    if len(x_pd) > 1:
        slope, intercept, r, p, _ = scipy.stats.linregress(
            x_pd, y_pd
        )

        x_fit = np.sort(np.array(x_pd))
        y_fit = intercept + slope * x_fit

        plt.plot(
            x_fit,
            y_fit,
            color=color_dict["DBS ON"],   # choose another color if desired
            linewidth=2.5,
            label=f"PD: r={r:.2f}, p={p:.3f}"
        )

    plt.xlabel(f"{scale_1} Score")
    plt.ylabel(f"{scale_2} Score")
    plt.title(f"Correlation between {scale_1} and {scale_2} scores")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(
        join(saving_path, f"{scale_1}_{scale_2}_correlation.{save_as}"),
        dpi=300
    )

    if show_plot:
        plt.show()
    else:
        plt.close()


# def correlate_two_scales(
#     scale_1,
#     scale_2,
#     sub_scale_dict,
#     subject_colors,
#     saving_path,
#     save_as,
#     show_plot = False    
#     ):
#         # correlation between SAS and BDI scores across all subjects, with a linear regression line:
#         plt.figure(figsize=(10, 6))
#         for sub in sub_scale_dict:
#             plt.scatter(sub_scale_dict[sub][scale_1], sub_scale_dict[sub][scale_2], color = subject_colors[sub], label=sub)
#         # add linear regression line
#         x = [sub_scale_dict[sub][scale_1] for sub in sub_scale_dict]
#         y = [sub_scale_dict[sub][scale_2] for sub in sub_scale_dict]
#         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
#         plt.plot(x, intercept + slope * np.array(x), 'black', label='')
#         plt.xlabel(f'{scale_1} Score')
#         plt.ylabel(f'{scale_2} Score')
#         plt.title(f'Correlation between {scale_1} and {scale_2} scores \n Linear fit: r={r_value:.2f}, p={p_value:.3f}')
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.tight_layout()
#         plt.savefig(join(saving_path, f'{scale_1}_{scale_2}_correlation.{save_as}'), dpi=300)
#         if show_plot:
#             plt.show()
#         else:
#             plt.close()

# def scale_comparison_pd_hc(
#         df, 
#         scale, 
#         color_dict,
#         saving_path,
#         save_as,
#         show_plot = False
#         ):
#     plot_df = df.copy()

#     plot_df["group"] = plot_df.index.to_series().map(
#         lambda x: "control" if x.startswith("C") else "patient"
#     )

#     fig, ax = plt.subplots(figsize=(6, 4))

#     sns.stripplot(
#         data=plot_df,
#         x="group",
#         y=scale,
#         jitter=0.25,
#         size=7,
#         palette={
#             "patient": color_dict["DBS ON"],
#             "control": color_dict["control"],
#         },
#         ax=ax
#     )
#     sns.pointplot(
#         data=plot_df,
#         x="group",
#         y=scale,
#         errorbar="sd",      # or ("ci", 95), "se", etc.
#         join=False,         # deprecated, see below
#         markers="_",
#         color="black",
#         ax=ax
#     )

#     ax.set_title(scale)
#     ax.set_xlabel("")
#     ax.set_ylabel("Score")
#     ax.spines[["top", "right"]].set_visible(False)
#     plt.savefig(join(saving_path, f'Comparison PD and HC - {scale}.{save_as}'), dpi=300)
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()


def visualize_updrs_scores(
    sub_scale_dict, 
    subject_colors,
    color_dict, 
    colored_by,
    saving_path, 
    save_as,
    show_plot = False
    ):

    updrs_on = []
    updrs_off = []

    for _, scores in sub_scale_dict.items():
        if 'UPDRS_ON' in scores and 'UPDRS_OFF' in scores:
            if scores['UPDRS_ON'] is not None and not pd.isna(scores['UPDRS_ON']):
                updrs_on.append(scores['UPDRS_ON'])
            if scores['UPDRS_OFF'] is not None and not pd.isna(scores['UPDRS_OFF']):
                updrs_off.append(scores['UPDRS_OFF'])

    updrs_on = np.array(updrs_on)
    updrs_off = np.array(updrs_off)

    # Paired statistical test
    stat, p = scipy.stats.wilcoxon(
        updrs_off,
        updrs_on,
        alternative='two-sided'
    )

    print("UPDRS ON vs OFF: Wilcoxon signed-rank test")
    print(f"n = {len(updrs_on)}")
    print(f"Statistic = {stat:.2f}, p = {p:.4f}")
    print(f"OFF median: {np.median(updrs_off)}")
    print(f"ON median: {np.median(updrs_on)}")

    plt.figure(figsize=(4, 5))

    # Individual paired lines
    for on, off in zip(updrs_on, updrs_off):
        plt.plot(['OFF', 'ON'], [off, on],
                color='gray', alpha=0.4, linewidth=1)

    if colored_by == 'condition':
        # Scatter
        plt.scatter(['OFF'] * len(updrs_off), updrs_off,
                    color=color_dict['DBS OFF'], s=60, zorder=3)

        plt.scatter(['ON'] * len(updrs_on), updrs_on,
                    color=color_dict['DBS ON'], s=60, zorder=3)
    elif colored_by == 'subject':
        for sub in sub_scale_dict:
            if 'UPDRS_ON' in sub_scale_dict[sub] and 'UPDRS_OFF' in sub_scale_dict[sub]:
                on_score = sub_scale_dict[sub]['UPDRS_ON']
                off_score = sub_scale_dict[sub]['UPDRS_OFF']
                if on_score is not None and off_score is not None and not pd.isna(on_score) and not pd.isna(off_score):
                    plt.scatter(['OFF'], [off_score], color=subject_colors[sub], s=60, zorder=3)
                    plt.scatter(['ON'], [on_score], color=subject_colors[sub], s=60, zorder=3)

    # Mean ± SEM
    for label, data in zip(['OFF', 'ON'], [updrs_off, updrs_on]):
        mean = data.mean()
        sd = data.std(ddof=1)
        plt.errorbar(label, mean, yerr=sd,
                    color='black', capsize=5, linewidth=2)

    plt.ylabel('UPDRS score')
    plt.title('UPDRS ON vs OFF (Patients)\n Mean ± std \n Wilcoxon signed-rank test: \n statistic = {:.2f}, p = {:.4f}'.format(stat, p))
    sns.despine()
    plt.tight_layout()
    plt.savefig(join(saving_path, f"updrs_on_off_by_{colored_by}.{save_as}"), dpi = 300)
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_rt_distribution(
    stats, 
    color_dict, 
    saving_path,
    save_as,
    show_plot = False    
):
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

    conditions = ['control', 'DBS ON', 'DBS OFF']
    stats_list = [stats_CONTROL, stats_ON, stats_OFF]

    trial_types = ['GO', 'GF', 'GC', 'GS']
    ylims = {'GC': 0.003, 'GS': 0.005, 'GO': 0.003, 'GF': 0.005}

    fig, axes = plt.subplots(
        nrows=3, ncols=4,
        figsize=(15, 12),
        sharex=True
    )

    for row, (condition, stats_cond) in enumerate(zip(conditions, stats_list)):

        all_gc_RTs = []
        all_gs_RTs = []
        all_go_RTs = []
        all_gf_RTs = []

        for subject in stats_cond.keys():
            if 'GC RTs from continue cue (ms)' in stats_cond[subject]:
                all_gc_RTs.extend(stats_cond[subject]['GC RTs from continue cue (ms)'])
            if 'GS RTs from stop cue (ms)' in stats_cond[subject]:
                all_gs_RTs.extend(stats_cond[subject]['GS RTs from stop cue (ms)'])
            if 'go_trial RTs (ms)' in stats_cond[subject]:
                all_go_RTs.extend(stats_cond[subject]['go_trial RTs (ms)'])
            if 'go_fast_trial RTs (ms)' in stats_cond[subject]:
                all_gf_RTs.extend(stats_cond[subject]['go_fast_trial RTs (ms)'])

        data_dict = {
            'GC': all_gc_RTs,
            'GS': all_gs_RTs,
            'GO': all_go_RTs,
            'GF': all_gf_RTs
        }

        for col, trial in enumerate(trial_types):
            ax = axes[row, col]

            ax.hist(
                data_dict[trial],
                bins=20,
                color=color_dict[condition],
                alpha=0.7,
                density=True
            )

            sns.kdeplot(
                data_dict[trial],
                ax=ax,
                color='black',
                fill=False,
                alpha=0.3
            )

            ax.set_ylim(0, ylims[trial])

            # Column titles (top row only)
            if row == 0:
                ax.set_title(trial)

            # Row labels (first column only)
            if col == 0:
                ax.set_ylabel(f'{condition} n= {len(stats_cond)}\nDensity')

            # X label (bottom row only)
            if row == 2:
                ax.set_xlabel('Reaction Time (ms)')

    plt.tight_layout()
    plt.savefig(join(saving_path, f"RT_distributions.{save_as}"),
        dpi=300
    )
    if show_plot:
        plt.show()
    else:    
        plt.close()

def visualize_rt_distribution_overlapped(
    stats, 
    color_dict, 
    saving_path,
    save_as,
    show_plot = False    
):
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

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

    for col, trial in enumerate(['GO', 'GF', 'GC', 'GS']):
        ax = axes[col]
        n_per_cond = {}

        for condition, stats_cond in zip(
            ['control', 'DBS ON', 'DBS OFF'],
            [stats_CONTROL, stats_ON, stats_OFF]
        ):
            data = []
            n_per_cond[condition] = len(stats_cond)

            for subject in stats_cond.keys():
                key_map = {
                    'GC': 'GC RTs from continue cue (ms)',
                    'GS': 'GS RTs from stop cue (ms)',
                    'GO': 'go_trial RTs (ms)',
                    'GF': 'go_fast_trial RTs (ms)'
                }
                if key_map[trial] in stats_cond[subject]:
                    data.extend(stats_cond[subject][key_map[trial]])

            sns.kdeplot(
                data,
                ax=ax,
                label=condition,
                color=color_dict[condition],
                linewidth=2
            )

        ax.set_title(trial)
        if trial in ['GC', 'GS']:
            ax.set_xlabel('RT from Continue/Stop cue(ms)')
        else:
            ax.set_xlabel('RT from GO/GF cue(ms)')    

    axes[0].set_ylabel('Density')
    axes[0].legend()
    fig.suptitle(f'Reaction Time Distributions by Condition \n HC = {n_per_cond["control"]} \n PD = {n_per_cond["DBS ON"]}')
    plt.tight_layout()
    plt.savefig(join(saving_path, f"RTs_kde_overlay.{save_as}"), dpi = 300)
    if show_plot:
        plt.show()
    else:
        plt.close()
    

def correlate_ssrt_prep_cost(
    stats,
    subject_colors,
    color_dict,
    saving_path,
    save_as,
    show_plot = False
):    
    ssrt_dict = {'DBS ON': [], 'DBS OFF': [], 'control': []}
    prep_cost_dict = {'DBS ON': [], 'DBS OFF': [], 'control': []}

    plt.figure(figsize=(8, 6))
    for subject in stats.keys():
        cond = 'control' if subject.startswith('C') else ('DBS ON' if 'ON' in subject else 'DBS OFF')
        marker = 's' if cond == 'control' else ('o' if cond == 'DBS ON' else '^')
        ssrt = stats[subject]['SSRT (ms)']
        prep_cost = stats[subject]['Preparation cost (ms)']
        plt.scatter(prep_cost, ssrt, color = subject_colors[subject.split(' ')[0]], s=100, marker=marker)

        ssrt_dict[cond].append(ssrt)
        prep_cost_dict[cond].append(prep_cost)

    # fit linear regression line for each condition:
    p_vals = {}
    for cond in ['control', 'DBS ON', 'DBS OFF']:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(prep_cost_dict[cond], ssrt_dict[cond])
        plt.plot(prep_cost_dict[cond], intercept + slope * np.array(prep_cost_dict[cond]), color = color_dict[cond])    
        p_vals[cond] = p_value

    # fit linear regression line across all conditions:
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(prep_cost_dict['control'] + prep_cost_dict['DBS ON'] + prep_cost_dict['DBS OFF'], ssrt_dict['control'] + ssrt_dict['DBS ON'] + ssrt_dict['DBS OFF'])
    plt.plot(prep_cost_dict['control'] + prep_cost_dict['DBS ON'] + prep_cost_dict['DBS OFF'], intercept + slope * np.array(prep_cost_dict['control'] + prep_cost_dict['DBS ON'] + prep_cost_dict['DBS OFF']), 'black', label=f'Linear fit: r={r_value:.2f}, p={p_value:.3f}')
    
    # plt.scatter(prep_cost, ssrt, color=subject_colors[subject.split(' ')[0]], s=100)
    plt.xlabel('Preparation Cost (ms)')
    plt.ylabel('SSRT (ms)')
    plt.title(f'Correlation between Preparation Cost and SSRT\nLinear fit group: r={r_value:.2f}, p={p_value:.3f}')

    markers=["s","o","^"]
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
    # handles = [f(markers[i], "k") for i in range(3)]
    # labels = ["control", "dbs on", "dbs off"]
    # plt.legend(handles, labels, loc=3, framealpha=1)
    markers = ["s", "o", "^"]
    conds_ordered = ['control', 'DBS ON', 'DBS OFF']
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f(markers[i], color_dict[conds_ordered[i]]) for i in range(3)]
    labels = [f"control (p={p_vals['control']:.3f})", f"DBS ON (p={p_vals['DBS ON']:.3f})", f"DBS OFF (p={p_vals['DBS OFF']:.3f})"]
    legend = plt.legend(handles, labels, loc=3, framealpha=1)
    # Optional: also color the legend text to match
    for text, cond in zip(legend.get_texts(), conds_ordered):
        text.set_color(color_dict[cond])
    plt.tight_layout()
    plt.savefig(join(saving_path, f'ssrt_prep_cost_correlation.{save_as}'), dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()

def correlate_behav_measure_with_scale(
        stats,
        sub_scale_dict,
        behav_measure_name,
        scale_name,
        subject_colors,
        color_dict,
        saving_path,
        save_as,
        show_plot = False
    ):
    behav_measure_dict = {'DBS ON': [], 'DBS OFF': [], 'control': []}
    scale_dict = {'DBS ON': [], 'DBS OFF': [], 'control': []}

    plt.figure(figsize=(8, 6))
    for subject in stats.keys():
        cond = 'control' if subject.startswith('C') else ('DBS ON' if 'ON' in subject else 'DBS OFF')
        marker = 's' if cond == 'control' else ('o' if cond == 'DBS ON' else '^')
        behav_measure = stats[subject][behav_measure_name]
        scale_measure = sub_scale_dict[subject.split(' ')[0]][scale_name]
        plt.scatter(behav_measure, scale_measure, color = subject_colors[subject.split(' ')[0]], s=100, marker=marker)

        behav_measure_dict[cond].append(behav_measure)
        scale_dict[cond].append(scale_measure)

    # # fit linear regression line for each condition:
    # p_vals = {}
    # for cond in ['control', 'DBS ON', 'DBS OFF']:
    #     slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(behav_measure_dict[cond], scale_dict[cond])
    #     plt.plot(behav_measure_dict[cond], intercept + slope * np.array(behav_measure_dict[cond]), color = color_dict[cond])
    #     p_vals[cond] = p_value

    # # fit linear regression line across all conditions:        
    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(behav_measure_dict['control'] + behav_measure_dict['DBS ON'] + behav_measure_dict['DBS OFF'], scale_dict['control'] + scale_dict['DBS ON'] + scale_dict['DBS OFF'])
    # plt.plot(behav_measure_dict['control'] + behav_measure_dict['DBS ON'] + behav_measure_dict['DBS OFF'], intercept + slope * np.array(behav_measure_dict['control'] + behav_measure_dict['DBS ON'] + behav_measure_dict['DBS OFF']), 'black', label=f'Linear fit: r={r_value:.2f}, p={p_value:.3f}')

    # Fit linear regression line for each condition

    p_vals = {}

    for cond in ['control', 'DBS ON', 'DBS OFF']:
        x = np.array(behav_measure_dict[cond], dtype=float)
        y = np.array(scale_dict[cond], dtype=float)

        # Keep only subjects with valid values for BOTH measures
        valid = np.isfinite(x) & np.isfinite(y)
        x_clean = x[valid]
        y_clean = y[valid]

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            x_clean, y_clean
        )

        plt.plot(
            x_clean,
            intercept + slope * x_clean,
            color=color_dict[cond]
        )

        p_vals[cond] = p_value


    # Fit linear regression line across all conditions

    x_all = np.concatenate([
        np.array(behav_measure_dict['control'], dtype=float),
        np.array(behav_measure_dict['DBS ON'], dtype=float),
        np.array(behav_measure_dict['DBS OFF'], dtype=float)
    ])

    y_all = np.concatenate([
        np.array(scale_dict['control'], dtype=float),
        np.array(scale_dict['DBS ON'], dtype=float),
        np.array(scale_dict['DBS OFF'], dtype=float)
    ])

    # Remove subjects with NaN/inf in either variable
    valid = np.isfinite(x_all) & np.isfinite(y_all)
    x_all_clean = x_all[valid]
    y_all_clean = y_all[valid]

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        x_all_clean,
        y_all_clean
    )

    # Sort x values so the regression line is plotted cleanly
    x_plot = np.sort(x_all_clean)

    plt.plot(
        x_plot,
        intercept + slope * x_plot,
        'black',
        label=f'Linear fit: r={r_value:.2f}, p={p_value:.3f}'
    )

    plt.ylabel(f'{scale_name} Score')
    plt.xlabel(f'{behav_measure_name}')
    plt.title(f'Correlation between {scale_name} and {behav_measure_name}\nLinear fit group: r={r_value:.2f}, p={p_value:.3f}')
    markers=["s","o","^"]
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
    # handles = [f(markers[i], "k") for i in range(3)]
    # labels = [f"control (p={p_vals['control']:.3f})", f"DBS ON (p={p_vals['DBS ON']:.3f})", f"DBS OFF (p={p_vals['DBS OFF']:.3f})"]
    # plt.legend(handles, labels, loc=3, framealpha=1)
    markers = ["s", "o", "^"]
    conds_ordered = ['control', 'DBS ON', 'DBS OFF']
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f(markers[i], color_dict[conds_ordered[i]]) for i in range(3)]
    labels = [f"control (p={p_vals['control']:.3f})", f"DBS ON (p={p_vals['DBS ON']:.3f})", f"DBS OFF (p={p_vals['DBS OFF']:.3f})"]
    legend = plt.legend(handles, labels, loc=3, framealpha=1)
    # Optional: also color the legend text to match
    for text, cond in zip(legend.get_texts(), conds_ordered):
        text.set_color(color_dict[cond])
    plt.tight_layout()
    plt.savefig(join(saving_path, f'{behav_measure_name}_{scale_name}_correlation.{save_as}'), dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def add_stat_annotation(ax, x1, x2, y, h, p_val, fontsize=11):
    """Draw a significance bracket between x1 and x2 at height y, with bar height h."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.3, color='black')
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    elif p_val < 0.05:
        stars = '*'
    else:
        stars = 'ns'
    ax.text((x1 + x2) / 2, y + h, f"{stars} (p={p_val:.3f})",
             ha='center', va='bottom', fontsize=fontsize)


def plot_variable_of_interest(
        stats,
        color_dict,
        subject_colors,
        variable_of_interest,
        colored_by,
        saving_path,
        save_as,
        show_plot = False,
        run_stats = True,          # NEW: toggle statistical testing
        correct_multiple_comparisons = False  # NEW: optional Bonferroni correction
    ):
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

    # Define conditions and corresponding dictionaries
    conditions = {
        'control': stats_CONTROL,
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
            # Retrieve the required metrics and store them in the result dictionary
            var = (metrics[variable_of_interest])
            results[condition][sub_id] = var

    # Prepare data for DataFrame
    data = []
    for condition, subject_dict in results.items():
        for subject_id, var in subject_dict.items():
            data.append({'Subject': subject_id, 'Condition': condition, variable_of_interest: var})

    # Create DataFrame
    df_all = pd.DataFrame(data)

    # Calculate mean and standard deviation for each condition
    group_stats = df_all.groupby('Condition')[variable_of_interest].agg(['mean', 'std'])

    # Perform pairwise Mann-Whitney U tests to compare each condition
    conditions = ['control', 'DBS OFF', 'DBS ON']

    # Initialize the plot
    plt.figure(figsize=(5, 6))


    sns.pointplot(data=df_all, x='Condition', y=variable_of_interest,
                linestyle='none', errorbar='sd', marker='_', color='black', legend=False)

    # Overlay individual data points
    if colored_by == 'subject':
        stripplot = sns.stripplot(data=df_all, x='Condition', y=variable_of_interest, jitter = 0.25,
                                size=7, palette=subject_colors, hue='Subject', legend=False)
    elif colored_by == 'condition':
        stripplot = sns.stripplot(data=df_all, x='Condition', y=variable_of_interest, jitter = 0.25,
                                size=7, palette=color_dict, hue='Condition', legend=False)

    # Retrieve the x-coordinates for each condition label
    condition_x_positions = {label.get_text(): pos for label, pos in zip(stripplot.get_xticklabels(), stripplot.get_xticks())}

    # Draw lines between corresponding subject points in 'DBS OFF' and 'DBS ON'
    # for subject_id in df_proactive_all['Subject'].unique():
    #     # Get data for this subject in 'DBS OFF' and 'DBS ON' conditions
    #     subject_data = df_proactive_all[(df_proactive_all['Subject'] == subject_id) & (df_proactive_all['Condition'].isin(['DBS OFF', 'DBS ON']))]
    #     if len(subject_data) == 2:
    #         # Use x-coordinates based on the dictionary created above
    #         x_coords = [condition_x_positions[subject_data.iloc[i]['Condition']] for i in range(2)]
    #         y_coords = subject_data[variable_of_interest].values
    #         plt.plot(x_coords, y_coords, marker='o', color='gray', alpha=0.5)

    # Calculate number of subjects in each group
    subject_counts = df_all.groupby('Condition')['Subject'].nunique()

    # # Add "n=number of subjects" above each violin
    # for condition, count in subject_counts.items():
    #     x_position = condition_x_positions[condition]  # Get the x-position for the condition
    #     plt.text(x_position, df_all[variable_of_interest].max() + 100, f'n={count}', 
    #             horizontalalignment='center', fontsize=12, color='black')

    # ------------------------------------------------------------------
    # NEW: Statistical testing
    # ------------------------------------------------------------------
    if run_stats:
        ax = plt.gca()

        # --- Control vs. DBS OFF: independent samples -> Mann-Whitney U ---
        control_vals = df_all.loc[df_all['Condition'] == 'control', variable_of_interest].values
        off_vals = df_all.loc[df_all['Condition'] == 'DBS OFF', variable_of_interest].values
        on_vals = df_all.loc[df_all['Condition'] == 'DBS ON', variable_of_interest].values

        p_control_off = np.nan
        if len(control_vals) > 0 and len(off_vals) > 0:
            _, p_control_off = scipy.stats.mannwhitneyu(control_vals, off_vals, alternative='two-sided')

        # --- Control vs. DBS ON: independent samples -> Mann-Whitney U ---
        p_control_on = np.nan
        if len(control_vals) > 0 and len(on_vals) > 0:
            _, p_control_on = scipy.stats.mannwhitneyu(control_vals, on_vals, alternative='two-sided')

        # --- DBS OFF vs. DBS ON: paired samples -> Wilcoxon signed-rank ---
        # Match subjects present in BOTH conditions (order-safe)
        off_series = df_all[df_all['Condition'] == 'DBS OFF'].set_index('Subject')[variable_of_interest]
        on_series = df_all[df_all['Condition'] == 'DBS ON'].set_index('Subject')[variable_of_interest]
        common_subjects = off_series.index.intersection(on_series.index)

        p_off_on = np.nan
        if len(common_subjects) > 0:
            off_paired = off_series.loc[common_subjects].values
            on_paired = on_series.loc[common_subjects].values
            _, p_off_on = scipy.stats.wilcoxon(off_paired, on_paired)

        # --- Optional: correct for multiple comparisons (Bonferroni) ---
        p_values = {'control_off': p_control_off, 'control_on': p_control_on, 'off_on': p_off_on}
        if correct_multiple_comparisons:
            n_tests = sum(~np.isnan(list(p_values.values())))
            p_values = {k: (v * n_tests if not np.isnan(v) else v) for k, v in p_values.items()}
            p_values = {k: min(v, 1.0) if not np.isnan(v) else v for k, v in p_values.items()}

        # --- Annotate brackets on the plot ---
        y_max = df_all[variable_of_interest].max()
        y_range = df_all[variable_of_interest].max() - df_all[variable_of_interest].min()
        h = y_range * 0.04
        gap = y_range * 0.12

        if not np.isnan(p_values['control_off']):
            add_stat_annotation(ax, condition_x_positions['control'], condition_x_positions['DBS OFF'],
                                 y_max + gap, h, p_values['control_off'])
        if not np.isnan(p_values['off_on']):
            add_stat_annotation(ax, condition_x_positions['DBS OFF'], condition_x_positions['DBS ON'],
                                 y_max + gap * 2.2, h, p_values['off_on'])
        if not np.isnan(p_values['control_on']):
            add_stat_annotation(ax, condition_x_positions['control'], condition_x_positions['DBS ON'],
                                 y_max + gap * 3.4, h, p_values['control_on'])

        # Give the annotations room by extending the y-axis
        ax.set_ylim(top=y_max + gap * 4.2)
    # ------------------------------------------------------------------

    # Add labels, title, and legend
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel(f'Mean {variable_of_interest}', fontsize=14)
    plt.title(f'Mean {variable_of_interest} Across Conditions \n n HC = {subject_counts["control"]} \n n PD = {subject_counts["DBS OFF"]}', fontsize=10)
    plt.tight_layout()
    plt.savefig(join(saving_path, f"{variable_of_interest} - Colored by {colored_by}.{save_as}"), dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()

# def plot_variable_of_interest(
#         stats,
#         color_dict,
#         subject_colors,
#         variable_of_interest,
#         colored_by,
#         saving_path,
#         save_as,
#         show_plot = False
#     ):
#     # Initialize empty dictionaries
#     stats_OFF = {}
#     stats_ON = {}
#     stats_CONTROL = {}
#     stats_PREOP = {}

#     # Loop through the original dictionary and filter into sub-dictionaries
#     for key, value in stats.items():
#         if "OFF" in key:
#             stats_OFF[key] = value
#         elif "ON" in key:
#             stats_ON[key] = value
#         elif "C" in key:
#             stats_CONTROL[key] = value
#         elif "preop" in key:
#             stats_PREOP[key] = value

#     # Define conditions and corresponding dictionaries
#     conditions = {
#         'control': stats_CONTROL,
#         'DBS OFF': stats_OFF,
#         'DBS ON': stats_ON,
#     }

#     # Initialize dictionaries to hold results for each condition
#     results = {condition: {} for condition in conditions.keys()}

#     # Loop through each condition and subject
#     for condition, data_dict in conditions.items():
#         for subject_id, metrics in data_dict.items():
#             # Extract the subject ID (first part of subject_id before the first space)
#             sub_id = subject_id.split()[0]
#             # Retrieve the required metrics and store them in the result dictionary
#             var = (metrics[variable_of_interest])
#             results[condition][sub_id] = var

#     # Prepare data for DataFrame
#     data = []
#     for condition, subject_dict in results.items():
#         for subject_id, var in subject_dict.items():
#             data.append({'Subject': subject_id, 'Condition': condition, variable_of_interest: var})

#     # Create DataFrame
#     df_all = pd.DataFrame(data)

#     # Calculate mean and standard deviation for each condition
#     group_stats = df_all.groupby('Condition')[variable_of_interest].agg(['mean', 'std'])

#     # Perform pairwise Mann-Whitney U tests to compare each condition
#     conditions = ['control', 'DBS OFF', 'DBS ON']

#     # Initialize the plot
#     plt.figure(figsize=(5, 6))


#     sns.pointplot(data=df_all, x='Condition', y=variable_of_interest,
#                 linestyle='none', errorbar='sd', marker='_', color='black', legend=False)

#     # Overlay individual data points
#     if colored_by == 'subject':
#         stripplot = sns.stripplot(data=df_all, x='Condition', y=variable_of_interest, jitter = 0.25,
#                                 size=7, palette=subject_colors, hue='Subject', legend=False)
#     elif colored_by == 'condition':
#         stripplot = sns.stripplot(data=df_all, x='Condition', y=variable_of_interest, jitter = 0.25,
#                                 size=7, palette=color_dict, hue='Condition', legend=False)

#     # Retrieve the x-coordinates for each condition label
#     condition_x_positions = {label.get_text(): pos for label, pos in zip(stripplot.get_xticklabels(), stripplot.get_xticks())}

#     # Draw lines between corresponding subject points in 'DBS OFF' and 'DBS ON'
#     # for subject_id in df_proactive_all['Subject'].unique():
#     #     # Get data for this subject in 'DBS OFF' and 'DBS ON' conditions
#     #     subject_data = df_proactive_all[(df_proactive_all['Subject'] == subject_id) & (df_proactive_all['Condition'].isin(['DBS OFF', 'DBS ON']))]
#     #     if len(subject_data) == 2:
#     #         # Use x-coordinates based on the dictionary created above
#     #         x_coords = [condition_x_positions[subject_data.iloc[i]['Condition']] for i in range(2)]
#     #         y_coords = subject_data[variable_of_interest].values
#     #         plt.plot(x_coords, y_coords, marker='o', color='gray', alpha=0.5)

#     # Calculate number of subjects in each group
#     subject_counts = df_all.groupby('Condition')['Subject'].nunique()

#     # # Add "n=number of subjects" above each violin
#     # for condition, count in subject_counts.items():
#     #     x_position = condition_x_positions[condition]  # Get the x-position for the condition
#     #     plt.text(x_position, df_all[variable_of_interest].max() + 100, f'n={count}', 
#     #             horizontalalignment='center', fontsize=12, color='black')

#     # Add labels, title, and legend
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlabel('Condition', fontsize=14)
#     plt.ylabel(f'Mean {variable_of_interest}', fontsize=14)
#     plt.title(f'Mean {variable_of_interest} Across Conditions \n n HC = {subject_counts["control"]} \n n PD = {subject_counts["DBS OFF"]}', fontsize=10)
#     plt.tight_layout()
#     plt.savefig(join(saving_path, f"{variable_of_interest} - Colored by {colored_by}.{save_as}"), dpi=300)
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()

def plot_rt_all(
    stats,
    color_dict,
    subject_colors,
    colored_by,
    saving_path,
    save_as,
    show_plot = False
):
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

    # Define conditions and corresponding dictionaries
    conditions = {
        'control': stats_CONTROL,
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
            # Retrieve the required metrics and store them in the result dictionary
            go_rt = metrics['go_trial mean RT (ms)']
            gc_rt = metrics['go_continue_trial mean RT (ms)']
            gs_rt = metrics['stop_trial mean RT (ms)']
            gf_rt = metrics['go_fast_trial mean RT (ms)']
            results[condition][sub_id] = {
                'go_rt': go_rt,
                'gc_rt': gc_rt,
                'gs_rt': gs_rt,
                'gf_rt': gf_rt
            }

    plt.figure(figsize=(15, 5))
    for i, trial_type in enumerate(['go_rt', 'gf_rt', 'gc_rt', 'gs_rt']):
        plt.subplot(1, 4, i+1)
        for condition, subject_dict in results.items():
            for subject_id, metrics in subject_dict.items():
                rt = metrics[trial_type]
                if colored_by == 'subject':
                    # add a line to connect the points for the same subject across conditions:
                    subject_conditions = []
                    subject_rts = []
                    for cond, sub_dict in results.items():
                        if subject_id in sub_dict:
                            subject_conditions.append(cond)
                            subject_rts.append(sub_dict[subject_id][trial_type])
                    plt.plot(subject_conditions, subject_rts, color=subject_colors[subject_id], alpha=0.5)
                    plt.scatter(condition, rt, color=subject_colors[subject_id], s=100, zorder=50)

                elif colored_by == 'condition':
                    # add a line to connect the points for the same subject across conditions:
                    subject_conditions = []
                    subject_rts = []
                    for cond, sub_dict in results.items():
                        if subject_id in sub_dict:
                            subject_conditions.append(cond)
                            subject_rts.append(sub_dict[subject_id][trial_type])
                    plt.plot(subject_conditions, subject_rts, color='lightgray', alpha=0.5)                                    
                    plt.scatter(condition, rt, color=color_dict[condition], s=100, zorder=50)
                
                # also add mean and std for each condition:
                condition_rts = [subject_dict[sub_id][trial_type] for sub_id in subject_dict.keys()]
                mean_rt = np.mean(condition_rts)
                std_rt = np.std(condition_rts)
                plt.errorbar(condition, mean_rt, yerr=std_rt, color='black', capsize=5, linewidth=2, zorder=100)
                plt.scatter(condition, mean_rt, color='black', marker = '_', s=100, zorder=101)                    

        plt.title(trial_type[0].capitalize() + trial_type[1].capitalize() + ' mean RT')
        plt.ylabel('Reaction Time (ms)')
        plt.ylim(300, 1000)
    plt.suptitle('Reaction Times for all trial types\n Mean ± std', fontsize=16)
    plt.tight_layout()
    plt.savefig(join(saving_path, f"RTs_all_trials_colored_by_{colored_by}.{save_as}"), dpi=300)
    if show_plot:
        plt.show()
    else:    
        plt.close()

def plot_perf_all(
    stats,
    color_dict,
    subject_colors,
    colored_by,
    saving_path,
    save_as,
    show_plot = False        
):
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

    # Define conditions and corresponding dictionaries
    conditions = {
        'control': stats_CONTROL,
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
            # Retrieve the required metrics and store them in the result dictionary
            go_correct = metrics['percent correct go_trial']
            gc_correct = metrics['percent correct go_continue_trial']
            gs_correct = metrics['percent correct stop_trial']
            gf_correct = metrics[ 'percent correct go_fast_trial']
            results[condition][sub_id] = {
                'go_correct': go_correct,
                'gf_correct': gf_correct,
                'gc_correct': gc_correct,
                'gs_correct': gs_correct        
            }

    plt.figure(figsize=(15, 5))
    for i, trial_type in enumerate(['go_correct', 'gf_correct', 'gc_correct', 'gs_correct']):
        plt.subplot(1, 4, i+1)
        if trial_type == 'go_correct':
            plt.axhline(y=70, color='lightgray', linestyle='--', linewidth=1, label ='70% threshold')
            plt.legend(loc='upper right')  
        if trial_type == 'gs_correct':
            plt.axhline(y=35, color='lightgray', linestyle='--', linewidth=1, label ='35-65% threshold')    
            plt.axhline(y=65, color='lightgray', linestyle='--', linewidth=1)
            plt.legend(loc='upper right')    
        for condition, subject_dict in results.items():
            for subject_id, metrics in subject_dict.items():
                rt = metrics[trial_type]
                if colored_by == 'subject':
                    # add a line to connect the points for the same subject across conditions:
                    subject_conditions = []
                    subject_rts = []
                    for cond, sub_dict in results.items():
                        if subject_id in sub_dict:
                            subject_conditions.append(cond)
                            subject_rts.append(sub_dict[subject_id][trial_type])
                    plt.plot(subject_conditions, subject_rts, color=subject_colors[subject_id], alpha=0.5)
                    plt.scatter(condition, rt, color=subject_colors[subject_id], s=100, zorder=50)

                elif colored_by == 'condition':
                    # add a line to connect the points for the same subject across conditions:
                    subject_conditions = []
                    subject_rts = []
                    for cond, sub_dict in results.items():
                        if subject_id in sub_dict:
                            subject_conditions.append(cond)
                            subject_rts.append(sub_dict[subject_id][trial_type])
                    plt.plot(subject_conditions, subject_rts, color='lightgray', alpha=0.5)                                    
                    plt.scatter(condition, rt, color=color_dict[condition], s=100, zorder=50)
                
                # also add mean and std for each condition:
                condition_rts = [subject_dict[sub_id][trial_type] for sub_id in subject_dict.keys()]
                mean_rt = np.mean(condition_rts)
                std_rt = np.std(condition_rts)
                plt.errorbar(condition, mean_rt, yerr=std_rt, color='black', capsize=5, linewidth=2, zorder=100)
                plt.scatter(condition, mean_rt, color='black', marker = '_', s=100, zorder=101)                    

        plt.title(trial_type[0].capitalize() + trial_type[1].capitalize() + ' % correct')
        plt.ylabel('Percentage Correct')
        plt.ylim(30, 110)
    plt.suptitle('Percentage Correct for all trial types\n Mean ± std', fontsize=16)
    plt.tight_layout()
    plt.savefig(join(saving_path, f"Performance_all_trials_colored_by_{colored_by}.{save_as}"), dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()    



def plot_kde_on_off_per_trial_type(
        stats, 
        color_dict, 
        save_path=None, 
        save_as='png',
        show_plot = False):
    
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

    # x_kde = np.linspace(0, 1200, 1000)
    x_kde = np.linspace(-1, 3, 1000)
    all_subject_kdes = {}

    for condition, stats_cond in zip(
        ['DBS OFF', 'DBS ON', 'control'],
        [stats_OFF, stats_ON, stats_CONTROL]
    ):
        all_subject_kdes[condition] = {}

        for subject in stats_cond.keys():

            subject_data = stats_cond[subject]

            # initialize subject container
            all_subject_kdes[condition][subject] = {
                'CSD': None
            }
            # all_subject_kdes[condition][subject] = {}

            # store CSD directly
            CSD = subject_data.get('CSD (ms)', None)
            if CSD is not None:
                CSD = np.asarray(CSD, dtype=float)
                CSD = CSD[np.isfinite(CSD)]
                all_subject_kdes[condition][subject]['CSD'] = CSD

            trial_dict = {
                'GC': subject_data.get('GC RTs from continue cue (ms)', []), 
                'GS': subject_data.get('GS RTs from stop cue (ms)', []), 
                'GO': subject_data.get('go_trial RTs (ms)', []),
                'GF': subject_data.get('go_fast_trial RTs (ms)', [])
            }

            # CSD = subject_data.get('CSD (ms)', None)
            # if CSD is not None:
            #     trial_dict['CSD'] = [CSD]

            for trial_type, rts in trial_dict.items():

                # convert to numpy array and remove NaN/inf
                rts = np.asarray(rts, dtype=float)
                rts = rts[np.isfinite(rts)]

                # convert to seconds
                rts = rts / 1000.0

                # optional: remove impossible RTs
                # rts = rts[(rts > 0) & (rts < 2000)]

                # need enough unique values for KDE
                if len(rts) > 1 and len(np.unique(rts)) > 1:
                    try:
                        kde = scipy.stats.gaussian_kde(rts) #, bw_method=0.2
                        y_kde = kde(x_kde)

                        all_subject_kdes[condition][subject][trial_type] = {
                            'x': x_kde.copy(),
                            'y': y_kde,
                            'raw_rts': rts
                        }

                    except Exception as e:
                        print(f'KDE failed: {condition} | {subject} | {trial_type}: {e}')
                        all_subject_kdes[condition][subject][trial_type] = None
                else:
                    all_subject_kdes[condition][subject][trial_type] = None

    # Plot for each subject the kde in DBS OFF and DBS on for each trial type (GC, GS, GO, GO_FAST):
    for trial_type in ['GC', 'GS', 'GO', 'GF']:
        for subject in all_subject_kdes['DBS OFF'].keys():
            plt.figure(figsize=(10, 6)) # one new figure per subject
            for condition in ['DBS OFF', 'DBS ON']:
                sub = subject if condition == 'DBS OFF' else subject.replace('OFF', 'ON')
                plt.plot(all_subject_kdes[condition][sub][trial_type]['x'], all_subject_kdes[condition][sub][trial_type]['y'], color=color_dict[condition], label=condition)

            if trial_type in ['GC', 'GS']:
                plt.xlabel('Reaction Time from triangle (s)')
            else:
                plt.xlabel('Reaction Time from square cue (s)')
            # plt.xlim(0, 1200)
            plt.xlim(0,1)
            # plt.ylim(0, 0.004)
            plt.title(f'{trial_type} Reaction Time KDEs ({sub.split(" ")[0]})')
            plt.legend()
            plt.tight_layout()
            if save_path is not None:
                plt.savefig(join(save_path, f'{sub.split(" ")[0]}_{trial_type}_kde.{save_as}'), dpi=300, bbox_inches='tight')
            if show_plot:
                plt.show()
            else:
                plt.close()



def plot_kde_compare_trial_type_per_sub(
        stats, 
        color_dict, 
        save_path=None, 
        save_as='png',
        show_plot = False):
    

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


    # x_kde = np.linspace(0, 1200, 1000)
    x_kde = np.linspace(-1, 3, 1000)
    all_subject_kdes = {}

    for condition, stats_cond in zip(
        ['DBS OFF', 'DBS ON', 'control'],
        [stats_OFF, stats_ON, stats_CONTROL]
    ):
        all_subject_kdes[condition] = {}

        for subject in stats_cond.keys():

            subject_data = stats_cond[subject]

            # initialize subject container
            all_subject_kdes[condition][subject] = {
                'CSD': None
            }
            # all_subject_kdes[condition][subject] = {}

            # store CSD directly
            CSD = subject_data.get('CSD (ms)', None)
            if CSD is not None:
                CSD = np.asarray(CSD, dtype=float)
                CSD = CSD[np.isfinite(CSD)]
                all_subject_kdes[condition][subject]['CSD'] = CSD

            trial_dict = {
                'GC': subject_data.get('go_continue_trial RTs (ms)', []), 
                'GS': subject_data.get('stop_trial RTs (ms)', []), 
                'GO': subject_data.get('go_trial RTs (ms)', []),
                'GF': subject_data.get('go_fast_trial RTs (ms)', [])
            }

            # CSD = subject_data.get('CSD (ms)', None)
            # if CSD is not None:
            #     trial_dict['CSD'] = [CSD]

            for trial_type, rts in trial_dict.items():

                # convert to numpy array and remove NaN/inf
                rts = np.asarray(rts, dtype=float)
                rts = rts[np.isfinite(rts)]

                # convert to seconds
                rts = rts / 1000.0

                # optional: remove impossible RTs
                # rts = rts[(rts > 0) & (rts < 2000)]

                # need enough unique values for KDE
                if len(rts) > 1 and len(np.unique(rts)) > 1:
                    try:
                        kde = scipy.stats.gaussian_kde(rts) #, bw_method=0.2
                        y_kde = kde(x_kde)

                        all_subject_kdes[condition][subject][trial_type] = {
                            'x': x_kde.copy(),
                            'y': y_kde,
                            'raw_rts': rts
                        }

                    except Exception as e:
                        print(f'KDE failed: {condition} | {subject} | {trial_type}: {e}')
                        all_subject_kdes[condition][subject][trial_type] = None
                else:
                    all_subject_kdes[condition][subject][trial_type] = None

    for condition in ['DBS OFF', 'DBS ON', 'control']:
        for subject in all_subject_kdes[condition].keys():
            plt.figure(figsize=(10, 6))
            for trial_type in ['GC', 'GS', 'GO', 'GF']:
                kde_data = all_subject_kdes[condition][subject][trial_type]
                if kde_data is not None:
                    plt.plot(kde_data['x'], kde_data['y'])
            plt.title(f'Reaction Time KDE ({condition} - {subject})')
            plt.xlabel('Reaction Time from GO (s)')
            plt.ylabel('Density')
            plt.xlim(0, 2)
            plt.legend(['GC', 'GS', 'GO', 'GF'])
            plt.tight_layout()
            if save_path is not None:
                plt.savefig(join(save_path, f'{subject}_kde.{save_as}'), dpi=300, bbox_inches='tight')
            if show_plot:
                plt.show()
            else:
                plt.close()
