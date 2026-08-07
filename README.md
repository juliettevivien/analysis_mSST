# Analysis of behavior and electrophysiology during the mSST task

Notebooks contain most preprocessing and analysis code. 
They are sub-divided into three categories:
- "EEG_"
- "LFP_"
- "BEHAV_"

In each category, numerical labelling is used:
0 is usually associated with pre-processing steps (e.g., cleaning the data by filtering or removing artifacts, or checking inclusion/exclusion criteria)
1, 2, 3, ... are usually associated with different analysis (e.g. TFR analysis, ERP analysis, RT analysis...)

**Available notebooks per section:**

EEG:

EEG_0a_crop_into_blocks.ipynb  # Separate full session into blocks
EEG_0b_preprocess_blocks.ipynb  # Preprocess each block separately
EEG_0c_concat_blocks_epoch_preproc.ipynb  # Epoch per block, concatenate and remove bad epochs


BEHAV:

BEHAV_0_check_HRM_assumptions.ipynb  # Decide which subjects to include in the analysis based on 2 criteria : independence assumption of GO and STOP processes, success rate
BEHAV_1_scales.ipynb  # Analysis of behavioral scales (UPDRS, MOCA, BIS,...)
BEHAV_2_rt.ipynb
BEHAV_3_performance.ipynb
BEHAV_4_proactive_inhibition.ipynb
BEHAV_5_reactive_inhibition.ipynb
BEHAV_6_correlation_task_scales.ipynb


LFP:
LFP_0_epoch_preproc.ipynb
