import scipy    
import numpy as np
import mne

def perform_permutation_cluster_test(percent_change_1, percent_change_2):
    # parameters for cluster permutation test
    pval = 0.05
    dfn = 2 - 1  # degrees of freedom numerator
    n_observations = len(percent_change_1.data) + len(percent_change_2.data)
    dfd = n_observations - 2  # degrees of freedom denominator
    threshold = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
    print(f"Threshold = {threshold}")

    # Extract power for single channel as 3D matrix (epochs x frequencies x times)
    epochs_power_1 = percent_change_1.data[:, 0, :, :]
    epochs_power_2 = percent_change_2.data[:, 0, :, :]

    F_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
        [epochs_power_1, epochs_power_2],
        out_type="mask",
        n_permutations=1000,
        threshold=threshold,
        tail=0,
        seed=np.random.default_rng(seed=8675309),
    )

    # Compute the difference between conditions
    evoked_power_1 = np.nanmean(epochs_power_1, axis=0)
    evoked_power_2 = np.nanmean(epochs_power_2, axis=0)
    evoked_power_contrast = evoked_power_1 - evoked_power_2
    signs = np.sign(evoked_power_contrast)

    # Create new stats image with only significant clusters
    F_obs_plot = np.nan * np.ones_like(F_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            F_obs_plot[c] = F_obs[c] * signs[c]

    return F_obs_plot, F_obs


# Convert lists to numpy arrays for calculations
def compute_mean_std(data):
    mean = np.nanmean(data, axis=0)
    var = np.nanstd(data, axis=0)/np.sqrt(len(data))
    #var = np.std(data, axis=0)
    #print(var)
    return mean, var
