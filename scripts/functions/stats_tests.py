import os
os.environ["OMP_NUM_THREADS"] = "1"

import scipy    
import numpy as np
import mne
import pandas as pd
import scipy.stats
from scipy.stats import skew, kurtosis
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from os.path import join
from threadpoolctl import threadpool_limits


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


""" GC analysis bimodal distribution """

"""
Statistical analysis of RT bimodality across DBS OFF / DBS ON / control,
built to sit alongside plot_kde_on_off_per_trial_type().

Core idea
---------
The KDE plots show the shape of the distribution but not a number you can
run a test on. This module fits a 2-component Gaussian mixture (fast mode +
slow mode) to each subject's raw RTs per condition, extracts the fast-mode
weight and mode locations, checks whether bimodality is even statistically
supported (BIC, dip test, bimodality coefficient), and then runs paired
DBS-OFF-vs-ON comparisons across subjects.

Requires: numpy, pandas, scipy, scikit-learn, matplotlib, statsmodels
Optional: diptest  (pip install diptest --break-system-packages)
"""

# try:
#     import diptest
#     HAS_DIPTEST = True
# except ImportError:
#     HAS_DIPTEST = False
#     print("`diptest` not installed - dip statistic will be NaN. "
#           "Install with: pip install diptest --break-system-packages")



# ---------------------------------------------------------------------
# Build a tidy per-subject/condition table
# ---------------------------------------------------------------------

def build_gmm_summary_table(stats, trial_type='GC', rt_key_map=None):
    """
    For every subject with a matching DBS-OFF/DBS-ON pair, fit the GMM
    (+ dip test / bimodality coefficient) to the raw RTs for `trial_type`
    in each condition, and return one row per subject x condition.

    Assumes subject keys look like '<id> OFF' / '<id> ON', matching the
    convention already used in plot_kde_on_off_per_trial_type().
    """
    if rt_key_map is None:
        rt_key_map = {
            'GC': 'GC RTs from continue cue (ms)',
            'GS': 'GS RTs from stop cue (ms)',
            'GO': 'go_trial RTs (ms)',
            'GF': 'go_fast_trial RTs (ms)',
        }
    rt_key = rt_key_map[trial_type]

    off_subjects = {k: v for k, v in stats.items() if 'OFF' in k}
    rows = []

    for off_key, off_data in off_subjects.items():
        on_key = off_key.replace('OFF', 'ON')
        if on_key not in stats:
            print(f'No matching ON session for {off_key}, skipping.')
            continue

        subj_id = off_key.split(' ')[0]

        for cond_label, cond_data in [
            ('DBS OFF', off_data),
            ('DBS ON', stats[on_key]),
        ]:
            rts_ms = np.asarray(cond_data.get(rt_key, []), dtype=float)
            rts_ms = rts_ms[np.isfinite(rts_ms)]
            rts_s = rts_ms / 1000.0

            gmm_res = fit_gmm_2component(rts_s)
            # bc = bimodality_coefficient(rts_s)

            # dip_stat, dip_p = (np.nan, np.nan)
            # if HAS_DIPTEST and len(rts_s) >= 10:
            #     dip_stat, dip_p = diptest.diptest(rts_s)

            row = {
                'subject': subj_id,
                'condition': cond_label,
                'n_trials': len(rts_s),
                # 'bimodality_coeff': bc,
                # 'dip_stat': dip_stat,
                # 'dip_p': dip_p,
            }
            if gmm_res is not None:
                row.update(gmm_res)
            rows.append(row)

    return pd.DataFrame(rows)



def fit_gmm_2component(rts, n_init=10, random_state=0):
    """
    Fit 1- and 2-component 1D Gaussian mixtures to raw RTs (seconds).

    Returns a dict with the sorted (fast, slow) weights/means/stds from the
    2-component fit, plus BIC for both fits so you can check whether the
    2-component model is actually justified for this subject/condition
    (favors_bimodal = True if BIC prefers 2 components).
    """
    rts = np.asarray(rts, dtype=float)
    rts = rts[np.isfinite(rts)]
    if len(rts) < 10:
        return None

    X = rts.reshape(-1, 1)

    with threadpool_limits(limits=1):  # this threadpool_limits thing was just added to avoid a warning from KMeans
        gmm1 = GaussianMixture(
            n_components=1,
            random_state=random_state
        ).fit(X)

        gmm2 = GaussianMixture(
            n_components=2,
            n_init=n_init,
            random_state=random_state
        ).fit(X)

    # gmm1 = GaussianMixture(n_components=1, random_state=random_state).fit(X)
    # gmm2 = GaussianMixture(n_components=2, n_init=n_init,
    #                         random_state=random_state).fit(X)

    bic1, bic2 = gmm1.bic(X), gmm2.bic(X)

    order = np.argsort(gmm2.means_.flatten())
    means = gmm2.means_.flatten()[order]
    weights = gmm2.weights_[order]
    stds = np.sqrt(gmm2.covariances_.flatten())[order]

    return {
        'weight_fast': weights[0],
        'weight_slow': weights[1],
        'mean_fast': means[0],
        'mean_slow': means[1],
        'std_fast': stds[0],
        'std_slow': stds[1],
        'bic_1comp': bic1,
        'bic_2comp': bic2,
        'favors_bimodal': bic2 < bic1,
        'n_trials': len(rts),
    }



def compare_off_vs_on(df, metric='weight_fast', verbose=True):
    """
    Paired comparison of a per-subject metric between DBS OFF and ON.
    Primary test: Wilcoxon signed-rank (robust to non-normal, bounded
    metrics like mixture weights). Paired t-test reported for reference.
    """
    wide = df.pivot(index='subject', columns='condition', values=metric).dropna()
    off = wide['DBS OFF'].values
    on = wide['DBS ON'].values

    wstat, wp = scipy.stats.wilcoxon(off, on)
    tstat, tp = scipy.stats.ttest_rel(off, on)
    diff = on - off

    if verbose:
        print(f'{metric}: n = {len(off)} subjects')
        print(f'  OFF mean = {off.mean():.3f}  |  ON mean = {on.mean():.3f}  '
              f'|  mean diff (ON-OFF) = {diff.mean():+.3f}')
        print(f'  Wilcoxon signed-rank: W = {wstat:.2f}, p = {wp:.4f}')
        print(f'  Paired t-test:        t = {tstat:.2f}, p = {tp:.4f}')

    return {
        'metric': metric, 'n': len(off),
        'off_mean': off.mean(), 'on_mean': on.mean(),
        'wilcoxon_stat': wstat, 'wilcoxon_p': wp,
        'ttest_stat': tstat, 'ttest_p': tp,
        'subjects': wide.index.tolist(), 'off': off, 'on': on,
    }

def compare_gs_with_gc_fast_mode(stats, min_trials=10):
    """
    GS RTs are failed-stop RTs only (no RT exists for successfully
    inhibited trials), so they're a selection-biased sample of the
    fastest, most motorically-committed responses -- mechanistically
    this is thought to be the same population the GC fast mode reflects
    (responses effectively locked in before the second cue is fully
    processed), not an independent trial-type distribution. So instead
    of fitting a GMM to GS or comparing its mean RT to GC's overall mean,
    this compares GS mean RT directly to GC's fast-mode LOCATION
    (mean_fast from the GMM fit), per subject and condition.

    Two tests:
      1. Within each condition, is GS mean RT reliably different from
         GC's mean_fast (paired across subjects)? A gap close to zero
         supports "GS RTs = GC's fast pathway".
      2. Does that gap change between DBS OFF and ON? If DBS shifts the
         gap, GS and GC's fast mode are decoupling under stimulation --
         if the gap is stable, whatever DBS does to the fast pathway
         shows up consistently in both trial types.
    """
    df_gc = build_gmm_summary_table(stats, trial_type='GC')
    df_gs = build_unimodal_summary_table(stats, trial_type='GS')

    gc_wide = df_gc.pivot(index='subject', columns='condition',
                           values='mean_fast')
    gs_wide = df_gs.pivot(index='subject', columns='condition',
                           values='mean_rt')

    merged = gc_wide.join(gs_wide, lsuffix='_gc_fast', rsuffix='_gs').dropna()

    results = {}
    for cond in ['DBS OFF', 'DBS ON']:
        gc_col = f'{cond}_gc_fast'
        gs_col = f'{cond}_gs'
        gap = merged[gs_col] - merged[gc_col]
        wstat, wp = scipy.stats.wilcoxon(merged[gs_col], merged[gc_col])
        print(f'\n{cond}: GS mean RT vs GC mean_fast (n={len(merged)})')
        print(f'  GS mean = {merged[gs_col].mean():.3f}  |  '
              f'GC fast mean = {merged[gc_col].mean():.3f}  |  '
              f'gap = {gap.mean():+.3f}')
        print(f'  Wilcoxon: W = {wstat:.2f}, p = {wp:.4f}')
        results[cond] = {'gap_mean': gap.mean(), 'wilcoxon_p': wp}

    gap_off = merged['DBS OFF_gs'] - merged['DBS OFF_gc_fast']
    gap_on = merged['DBS ON_gs'] - merged['DBS ON_gc_fast']
    wstat2, wp2 = scipy.stats.wilcoxon(gap_off, gap_on)
    print(f'\nDoes the GS-vs-GCfast gap change with DBS? (n={len(merged)})')
    print(f'  gap OFF = {gap_off.mean():+.3f}  |  gap ON = {gap_on.mean():+.3f}')
    print(f'  Wilcoxon: W = {wstat2:.2f}, p = {wp2:.4f}')

    results['gap_change_p'] = wp2
    results['merged'] = merged
    return results


def build_unimodal_summary_table(stats, trial_type='GO', rt_key_map=None):
    """
    Companion to build_gmm_summary_table(), for trial types where the
    2-component GMM mostly isn't justified (check favors_bimodal first).
    Computes plain descriptive stats (mean, median, SD, skew) plus an
    ex-Gaussian fit (mu, sigma, tau) per subject/condition -- these don't
    presuppose a discrete second mode, so they're the right metrics to
    test the "DBS causes a general RT slowing" alternative on trial types
    without a real bimodal structure.
    """
    if rt_key_map is None:
        rt_key_map = {
            'GC': 'GC RTs from continue cue (ms)',
            'GS': 'GS RTs from stop cue (ms)',
            'GO': 'go_trial RTs (ms)',
            'GF': 'go_fast_trial RTs (ms)',
        }
    rt_key = rt_key_map[trial_type]

    off_subjects = {k: v for k, v in stats.items() if 'OFF' in k}
    rows = []

    for off_key, off_data in off_subjects.items():
        on_key = off_key.replace('OFF', 'ON')
        if on_key not in stats:
            continue
        subj_id = off_key.split(' ')[0]

        for cond_label, cond_data in [
            ('DBS OFF', off_data),
            ('DBS ON', stats[on_key]),
        ]:
            rts_ms = np.asarray(cond_data.get(rt_key, []), dtype=float)
            rts_ms = rts_ms[np.isfinite(rts_ms)]
            rts_s = rts_ms / 1000.0
            if len(rts_s) < 5:
                continue

            row = {
                'subject': subj_id,
                'condition': cond_label,
                'n_trials': len(rts_s),
                'mean_rt': rts_s.mean(),
                'median_rt': np.median(rts_s),
                'sd_rt': rts_s.std(ddof=1),
                'skew_rt': skew(rts_s, bias=False),
            }
            # exg = fit_exgaussian(rts_s)
            # if exg is not None:
            #     row.update({f'exg_{k}': v for k, v in exg.items()})
            rows.append(row)

    return pd.DataFrame(rows)


# def fit_pooled_gmm_per_subject(stats, trial_type='GC', rt_key_map=None,
#                                 n_init=10, random_state=0, min_trials=20):
#     """
#     Fit one 2-component GMM per subject, pooling RTs across DBS OFF and
#     DBS ON, to obtain subject-specific but condition-INVARIANT component
#     locations/scales. These fixed components are the reference
#     distributions used later to classify individual trials.

#     Returns {subject: {'mean_fast', 'sd_fast', 'mean_slow', 'sd_slow',
#     'pooled_weight_fast', 'bic_1comp', 'bic_2comp', 'favors_bimodal',
#     'n_pooled_trials'}}.
#     """
#     if rt_key_map is None:
#         rt_key_map = {
#             'GC': 'GC RTs from continue cue (ms)',
#             'GS': 'GS RTs from stop cue (ms)',
#             'GO': 'go_trial RTs (ms)',
#             'GF': 'go_fast_trial RTs (ms)',
#         }
#     rt_key = rt_key_map[trial_type]
#     off_subjects = {k: v for k, v in stats.items() if 'OFF' in k}
#     out = {}

#     for off_key, off_data in off_subjects.items():
#         on_key = off_key.replace('OFF', 'ON')
#         if on_key not in stats:
#             continue
#         subj_id = off_key.split(' ')[0]

#         rts_off = np.asarray(off_data.get(rt_key, []), dtype=float)
#         rts_on = np.asarray(stats[on_key].get(rt_key, []), dtype=float)
#         rts_off = rts_off[np.isfinite(rts_off)] / 1000.0
#         rts_on = rts_on[np.isfinite(rts_on)] / 1000.0
#         pooled = np.concatenate([rts_off, rts_on])

#         if len(pooled) < min_trials:
#             print(f'{subj_id}: only {len(pooled)} pooled trials, skipping.')
#             continue

#         X = pooled.reshape(-1, 1)

#         with threadpool_limits(limits=1): 
#             gmm1 = GaussianMixture(n_components=1, random_state=random_state).fit(X)
#             gmm2 = GaussianMixture(n_components=2, n_init=n_init,
#                                     random_state=random_state).fit(X)
#             bic1, bic2 = gmm1.bic(X), gmm2.bic(X)

#         order = np.argsort(gmm2.means_.flatten())
#         means = gmm2.means_.flatten()[order]
#         weights = gmm2.weights_[order]
#         sds = np.sqrt(gmm2.covariances_.flatten())[order]

#         out[subj_id] = {
#             'mean_fast': means[0], 'sd_fast': sds[0],
#             'mean_slow': means[1], 'sd_slow': sds[1],
#             'pooled_weight_fast': weights[0],
#             'bic_1comp': bic1, 'bic_2comp': bic2,
#             'favors_bimodal': bic2 < bic1,
#             'n_pooled_trials': len(pooled),
#         }

#     n_ok = sum(v['favors_bimodal'] for v in out.values())
#     print(f'Pooled 2-component model favored for {n_ok}/{len(out)} subjects.')
#     return out




# def fit_exgaussian(rts):
#     """
#     Fit an ex-Gaussian (Normal + Exponential convolution) to raw RTs via
#     scipy's exponnorm, which is parameterized as K = tau/sigma.

#     This is the standard unimodal-but-skewed RT model: mu/sigma describe
#     the Gaussian "core" of the distribution, tau describes the length of
#     the slow right tail. Use this for trial types (e.g. GO/GF) where the
#     GMM mostly does NOT favor two components -- it gives you a slow-tail
#     metric (tau) without assuming a discrete second mode exists.
#     """
#     rts = np.asarray(rts, dtype=float)
#     rts = rts[np.isfinite(rts)]
#     if len(rts) < 10:
#         return None
#     try:
#         K, loc, scale = scipy.stats.exponnorm.fit(rts)
#         sigma = scale
#         tau = K * scale
#         mu = loc
#         return {'mu': mu, 'sigma': sigma, 'tau': tau}
#     except Exception as e:
#         print(f'Ex-Gaussian fit failed: {e}')
#         return None

















# # ---------------------------------------------------------------------
# # Per-distribution bimodality metrics
# # ---------------------------------------------------------------------

# def bimodality_coefficient(rts):
#     """
#     Sarle's bimodality coefficient (SAS formula, small-sample corrected).
#     BC > 5/9 (~0.555) is the classic (rough) threshold suggesting bimodality;
#     treat this as a heuristic, not a p-value.
#     """
#     rts = np.asarray(rts, dtype=float)
#     rts = rts[np.isfinite(rts)]
#     n = len(rts)
#     if n < 4:
#         return np.nan
#     g = skew(rts, bias=False)
#     k = kurtosis(rts, fisher=True, bias=False)  # excess kurtosis
#     bc = (g**2 + 1) / (k + 3 + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)))
#     return bc





# # ---------------------------------------------------------------------
# # Paired OFF vs ON comparison
# # ---------------------------------------------------------------------



# def restrict_to_bimodal_subjects(df):
#     """
#     Filter a build_gmm_summary_table() output down to subjects where
#     favors_bimodal is True in BOTH DBS OFF and DBS ON -- i.e. subjects
#     for whom the fast/slow split is genuinely supported in both
#     conditions, so a weight_fast comparison actually means what you
#     want it to mean for every included subject.

#     Prints how many subjects were dropped so you can report it.
#     """
#     ok = (df.groupby('subject')['favors_bimodal']
#             .apply(lambda s: s.all())
#             .pipe(lambda s: s[s].index))
#     n_total = df['subject'].nunique()
#     n_kept = len(ok)
#     print(f'Restricting to consistently-bimodal subjects: '
#           f'{n_kept}/{n_total} kept, {n_total - n_kept} dropped.')
#     return df[df['subject'].isin(ok)].copy()



# def check_trial_counts(stats, trial_type, rt_key_map=None, min_trials=10):
#     """
#     Print per-subject/condition trial counts for a trial type, flagging
#     any below min_trials. Run this for GS before trusting any per-subject
#     GS statistic -- failed-stop trials are usually a minority of stop
#     trials, so n can be much lower than for GC/GO/GF, and both the
#     ex-Gaussian fit and simple means get noisy at low n.
#     """
#     if rt_key_map is None:
#         rt_key_map = {
#             'GC': 'GC RTs from continue cue (ms)',
#             'GS': 'GS RTs from stop cue (ms)',
#             'GO': 'go_trial RTs (ms)',
#             'GF': 'go_fast_trial RTs (ms)',
#         }
#     rt_key = rt_key_map[trial_type]
#     rows = []
#     for key, data in stats.items():
#         if 'OFF' not in key and 'ON' not in key:
#             continue
#         n = len(np.asarray(data.get(rt_key, []), dtype=float))
#         rows.append({'session': key, 'n_trials': n, 'low_n': n < min_trials})
#     df = pd.DataFrame(rows).sort_values('n_trials')
#     n_low = df['low_n'].sum()
#     print(f'{trial_type}: {n_low}/{len(df)} sessions below {min_trials} trials')
#     if n_low > 0:
#         print(df[df['low_n']].to_string(index=False))
#     return df




# def compare_multiple_metrics(df, metrics=('weight_fast', 'mean_fast',
#                                             'mean_slow', 'bimodality_coeff')):
#     """Run compare_off_vs_on for several metrics and return a results table."""
#     results = [compare_off_vs_on(df, m, verbose=False) for m in metrics
#                if m in df.columns]
#     out = pd.DataFrame(results)[['metric', 'n', 'off_mean', 'on_mean',
#                                   'wilcoxon_p', 'ttest_p']]
#     return out


# # ---------------------------------------------------------------------
# # Bootstrap CI for the paired OFF -> ON difference
# # ---------------------------------------------------------------------

# def bootstrap_paired_diff(off, on, n_boot=10000, ci=95, random_state=0,
#                            statistic='mean'):
#     """
#     Subject-level bootstrap for a paired OFF/ON difference.

#     Resamples subjects (paired OFF/ON values kept together) with
#     replacement n_boot times, recomputes the group-level difference
#     each time, and returns the observed diff plus a percentile CI.
#     This is the right unit of resampling here: each subject already
#     contributes a single GMM-derived value per condition, so we're
#     bootstrapping across subjects, not across trials.

#     Parameters
#     ----------
#     off, on : 1D arrays, paired (same subject order, e.g. from
#         df.pivot(...)['DBS OFF'].values / ['DBS ON'].values)
#     statistic : 'mean' or 'median' — which summary of (on - off) to bootstrap
#     """
#     off = np.asarray(off, dtype=float)
#     on = np.asarray(on, dtype=float)
#     assert len(off) == len(on), 'off/on must be paired (same length/order)'
#     n = len(off)
#     rng = np.random.default_rng(random_state)

#     diffs = on - off
#     stat_fn = np.mean if statistic == 'mean' else np.median
#     observed = stat_fn(diffs)

#     boot_stats = np.empty(n_boot)
#     for b in range(n_boot):
#         idx = rng.integers(0, n, size=n)  # resample subjects with replacement
#         boot_stats[b] = stat_fn(diffs[idx])

#     alpha = (100 - ci) / 2
#     lo, hi = np.percentile(boot_stats, [alpha, 100 - alpha])

#     # bootstrap p-value: proportion of resamples on the other side of 0
#     # from the observed effect direction (two-sided)
#     p_boot = 2 * min(np.mean(boot_stats >= 0), np.mean(boot_stats <= 0))
#     p_boot = min(p_boot, 1.0)

#     return {
#         'n_subjects': n,
#         'observed_diff': observed,
#         'ci_low': lo,
#         'ci_high': hi,
#         'ci_level': ci,
#         'excludes_zero': (lo > 0) or (hi < 0),
#         'p_boot': p_boot,
#         'boot_distribution': boot_stats,
#     }


# def bootstrap_ci_for_metric(df, metric='weight_fast', n_boot=10000, ci=95,
#                              statistic='mean', verbose=True):
#     """Convenience wrapper: pivot df, then bootstrap the OFF->ON diff."""
#     wide = df.pivot(index='subject', columns='condition', values=metric).dropna()
#     off = wide['DBS OFF'].values
#     on = wide['DBS ON'].values

#     res = bootstrap_paired_diff(off, on, n_boot=n_boot, ci=ci,
#                                  statistic=statistic)
#     res['metric'] = metric

#     if verbose:
#         print(f'{metric} (n={res["n_subjects"]}, {statistic} diff, '
#               f'{n_boot} bootstraps)')
#         print(f'  ON - OFF = {res["observed_diff"]:+.3f}   '
#               f'{ci}% CI [{res["ci_low"]:+.3f}, {res["ci_high"]:+.3f}]'
#               f'{"  (excludes 0)" if res["excludes_zero"] else "  (includes 0)"}')
#         print(f'  bootstrap p (two-sided) ~= {res["p_boot"]:.4f}')

#     return res


# def plot_bootstrap_distribution(res, save_path=None, save_as='png',
#                                  xlabel=None, title=None, show_plot=True):
#     """Histogram of the bootstrap distribution with the observed diff and CI."""
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.hist(res['boot_distribution'], bins=60, color='grey', alpha=0.7)
#     ax.axvline(0, color='black', linestyle='--', linewidth=1, label='0 (no effect)')
#     ax.axvline(res['observed_diff'], color='crimson', linewidth=2,
#                label='Observed diff')
#     ax.axvline(res['ci_low'], color='crimson', linestyle=':', linewidth=1)
#     ax.axvline(res['ci_high'], color='crimson', linestyle=':', linewidth=1,
#                label=f'{res["ci_level"]}% CI')

#     ax.set_xlabel(xlabel or f'{res.get("metric", "metric")} (ON - OFF)')
#     ax.set_ylabel('Bootstrap resamples')
#     ax.set_title(title or 'Bootstrap distribution of paired DBS effect')
#     ax.legend()
#     plt.tight_layout()

#     if save_path is not None:
#         fname = f'{res.get("metric", "metric")}_bootstrap.{save_as}'
#         plt.savefig(join(save_path, fname), dpi=300, bbox_inches='tight')
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()


# # ---------------------------------------------------------------------
# # Plotting
# # ---------------------------------------------------------------------

# def plot_paired_metric(df, metric='weight_fast', save_path=None, save_as='png',
#                         ylabel=None, title=None, show_plot=True):
#     """Spaghetti plot: one line per subject, DBS OFF -> DBS ON."""
#     wide = df.pivot(index='subject', columns='condition', values=metric).dropna()

#     fig, ax = plt.subplots(figsize=(5, 6))
#     for subj, row in wide.iterrows():
#         ax.plot(['DBS OFF', 'DBS ON'], [row['DBS OFF'], row['DBS ON']],
#                 marker='o', color='grey', alpha=0.6)

#     means = wide.mean()
#     ax.plot(['DBS OFF', 'DBS ON'], [means['DBS OFF'], means['DBS ON']],
#             marker='o', color='black', linewidth=3, label='Group mean')

#     ax.set_ylabel(ylabel or metric)
#     ax.set_title(title or f'{metric}: DBS OFF vs ON (GC trials)')
#     ax.legend()
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(join(save_path, f'{metric}_paired_plot.{save_as}'),
#                      dpi=300, bbox_inches='tight')
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()


# # ---------------------------------------------------------------------
# # Control comparison across trial types (is the effect conflict-specific?)
# # ---------------------------------------------------------------------

# def compute_delta_table(stats, trial_types=('GC', 'GO', 'GF'),
#                          metric='weight_fast', rt_key_map=None):
#     """
#     Build a subject x trial_type table of the per-subject ON-OFF delta for
#     `metric`, by rerunning build_gmm_summary_table for each trial type.

#     GO and GF have no square+triangle conflict (just press-asap-to-square),
#     so they're the natural control: if DBS reallocates fast/slow weight on
#     GC (and possibly GS) but leaves GO/GF untouched, that's evidence the
#     effect is specific to the go/continue-cue conflict rather than a
#     general RT-slowing under stimulation.

#     Returns
#     -------
#     delta_df : subject x trial_type DataFrame of ON-OFF deltas
#     per_type_dfs : dict {trial_type: raw build_gmm_summary_table output}
#     """
#     per_type_dfs = {}
#     delta_series = {}

#     for tt in trial_types:
#         df_tt = build_gmm_summary_table(stats, trial_type=tt,
#                                          rt_key_map=rt_key_map)
#         per_type_dfs[tt] = df_tt

#         if metric not in df_tt.columns:
#             print(f'{metric} missing for trial_type={tt}, skipping.')
#             continue

#         wide = df_tt.pivot(index='subject', columns='condition',
#                             values=metric).dropna()
#         delta_series[tt] = wide['DBS ON'] - wide['DBS OFF']

#     delta_df = pd.DataFrame(delta_series)
#     return delta_df, per_type_dfs


# def compare_conflict_vs_nonconflict(delta_df, conflict_types=('GC',),
#                                      nonconflict_types=('GO', 'GF')):
#     """
#     Average each subject's delta across the conflict trial type(s) and
#     across the non-conflict trial type(s), then paired-test conflict-delta
#     vs nonconflict-delta across subjects (Wilcoxon + paired t-test).

#     A significant, more-negative delta for conflict trials (relative to
#     non-conflict) is what you'd expect if DBS is specifically reallocating
#     weight away from the fast mode under go/continue-cue conflict, rather
#     than producing a general RT slowing that would show up equally on
#     GO/GF trials too.
#     """
#     conflict_types = [t for t in conflict_types if t in delta_df.columns]
#     nonconflict_types = [t for t in nonconflict_types if t in delta_df.columns]

#     sub = delta_df[conflict_types + nonconflict_types].dropna()
#     conflict_delta = sub[conflict_types].mean(axis=1)
#     nonconflict_delta = sub[nonconflict_types].mean(axis=1)

#     wstat, wp = scipy.stats.wilcoxon(conflict_delta, nonconflict_delta)
#     tstat, tp = scipy.stats.ttest_rel(conflict_delta, nonconflict_delta)

#     print(f'Conflict {conflict_types} vs non-conflict {nonconflict_types} '
#           f'delta (n={len(sub)} subjects)')
#     print(f'  conflict mean delta    = {conflict_delta.mean():+.3f}')
#     print(f'  non-conflict mean delta = {nonconflict_delta.mean():+.3f}')
#     print(f'  Wilcoxon: W = {wstat:.2f}, p = {wp:.4f}')
#     print(f'  Paired t: t = {tstat:.2f}, p = {tp:.4f}')

#     return {
#         'conflict_delta': conflict_delta, 'nonconflict_delta': nonconflict_delta,
#         'wilcoxon_p': wp, 'ttest_p': tp,
#     }


# def plot_delta_by_trial_type(delta_df, ylabel=None, title=None,
#                               save_path=None, save_as='png', show_plot=True):
#     """
#     Strip/box plot of per-subject ON-OFF deltas, one column per trial type,
#     so you can visually compare conflict (GC/GS) vs non-conflict (GO/GF)
#     trial types side by side.
#     """
#     trial_types = list(delta_df.columns)
#     fig, ax = plt.subplots(figsize=(1.6 * len(trial_types) + 2, 5))

#     ax.boxplot([delta_df[tt].dropna().values for tt in trial_types],
#                labels=trial_types, showmeans=True)

#     rng = np.random.default_rng(0)
#     for i, tt in enumerate(trial_types, start=1):
#         vals = delta_df[tt].dropna().values
#         jitter = rng.normal(0, 0.04, size=len(vals))
#         ax.scatter(np.full(len(vals), i) + jitter, vals,
#                    color='grey', alpha=0.6, zorder=3)

#     ax.axhline(0, color='black', linestyle='--', linewidth=1)
#     ax.set_ylabel(ylabel or 'ON - OFF delta')
#     ax.set_title(title or 'DBS effect by trial type (conflict vs control)')
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(join(save_path, f'delta_by_trial_type.{save_as}'),
#                      dpi=300, bbox_inches='tight')
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()


# # ---------------------------------------------------------------------
# # Trial-level mixed-effects model (trial_type x condition interaction)
# # ---------------------------------------------------------------------
# #
# # IMPORTANT: this tests for shifts in mean RT location by trial_type and
# # condition. It is a COMPLEMENT to the GMM weight_fast analysis, not a
# # replacement. Your GC result was that mean_fast and mean_slow barely
# # moved (0.199->0.186, 0.577->0.583) while weight_fast dropped -- a pure
# # reallocation-of-mass effect. A mean-based model is NOT well-suited to
# # detect that kind of effect, so don't be surprised (or worried) if GC's
# # interaction term here comes out small/non-significant even though the
# # weight-based effect is real. What this model IS good for: a single,
# # properly-powered test of whether GS/GO/GF show general RT shifts under
# # DBS that the two-stage delta-averaging approach couldn't detect well.

# def build_trial_level_table(stats, trial_types=('GC', 'GS', 'GO', 'GF'),
#                              rt_key_map=None):
#     """
#     Long-format table: one row per trial, with subject/condition/trial_type
#     and RT in seconds (+ log RT). Feeds fit_mixedlm_interaction().
#     """
#     if rt_key_map is None:
#         rt_key_map = {
#             'GC': 'GC RTs from continue cue (ms)',
#             'GS': 'GS RTs from stop cue (ms)',
#             'GO': 'go_trial RTs (ms)',
#             'GF': 'go_fast_trial RTs (ms)',
#         }

#     off_subjects = {k: v for k, v in stats.items() if 'OFF' in k}
#     rows = []

#     for off_key, off_data in off_subjects.items():
#         on_key = off_key.replace('OFF', 'ON')
#         if on_key not in stats:
#             continue
#         subj_id = off_key.split(' ')[0]

#         for cond_label, cond_data in [
#             ('DBS OFF', off_data),
#             ('DBS ON', stats[on_key]),
#         ]:
#             for tt in trial_types:
#                 rt_key = rt_key_map[tt]
#                 rts_ms = np.asarray(cond_data.get(rt_key, []), dtype=float)
#                 rts_ms = rts_ms[np.isfinite(rts_ms)]
#                 rts_s = rts_ms / 1000.0
#                 for rt in rts_s:
#                     rows.append({
#                         'subject': subj_id,
#                         'condition': cond_label,
#                         'trial_type': tt,
#                         'rt': rt,
#                         'log_rt': np.log(rt),
#                     })

#     return pd.DataFrame(rows)


# def fit_mixedlm_interaction(trial_df, dv='log_rt',
#                              trial_type_ref='GO', condition_ref='DBS OFF'):
#     """
#     Mixed-effects model: dv ~ trial_type * condition, random intercept
#     per subject. Requires statsmodels.

#     trial_type_ref / condition_ref set the reference levels, so the
#     trial_type[GC]:condition[DBS ON] coefficient directly answers
#     "does the DBS effect on mean RT differ for GC relative to GO"
#     (and likewise for GS, GF).

#     Prints the interaction terms and returns the fitted model.
#     """
#     import statsmodels.formula.api as smf

#     df = trial_df.copy()

#     # Drop non-finite dv values (e.g. log(rt) for rt <= 0) BEFORE fitting.
#     # If these are left in, patsy's missing-value handling silently drops
#     # them from the design matrix while a separately-passed `groups` array
#     # keeps its original length, causing a row-count mismatch inside
#     # MixedLM (IndexError: index ... out of bounds). Filtering + resetting
#     # the index here keeps groups aligned with the design matrix.
#     n_before = len(df)
#     df = df[np.isfinite(df[dv])].reset_index(drop=True)
#     n_dropped = n_before - len(df)
#     if n_dropped > 0:
#         print(f'Dropped {n_dropped} rows with non-finite {dv} '
#               f'(e.g. rt <= 0) before fitting.')

#     df['trial_type'] = pd.Categorical(
#         df['trial_type'],
#         categories=[trial_type_ref] + [t for t in df['trial_type'].unique()
#                                         if t != trial_type_ref])
#     df['condition'] = pd.Categorical(
#         df['condition'],
#         categories=[condition_ref] + [c for c in df['condition'].unique()
#                                        if c != condition_ref])

#     formula = f'{dv} ~ C(trial_type, Treatment("{trial_type_ref}")) * ' \
#               f'C(condition, Treatment("{condition_ref}"))'
#     model = smf.mixedlm(formula, data=df, groups=df['subject'])
#     fit = model.fit()

#     print(fit.summary())
#     print('\nInteraction terms (trial_type x condition) are the ones that '
#           'test whether the DBS effect on mean RT differs by trial type, '
#           f'relative to {trial_type_ref} as baseline.')

#     return fit


# # ---------------------------------------------------------------------
# # Trial-level test of slow-mode weight, matched to the top-down RT model
# # ---------------------------------------------------------------------
# #
# # Motivation: the earlier per-subject/per-condition GMM approach fit an
# # independent 2-component mixture for each of the 24 subject x condition
# # cells, then paired-tested the resulting 12 weight_fast values. That is
# # fragile (n=12, sensitive to single subjects, vulnerable to independent
# # fits drifting or label-switching between conditions) and does not use
# # the full trial-level data the way the top-down RT ~ Condition *
# # Trial_type LMM does.
# #
# # This section instead: (1) fits ONE 2-component GMM per subject, pooling
# # GC trials across BOTH conditions, to get fixed, condition-invariant
# # component locations -- justified by the earlier finding that mean_fast
# # and mean_slow did not differ by condition; (2) classifies every
# # individual trial (whichever condition it came from) against those fixed
# # components via a flat-prior likelihood ratio, giving each trial a
# # posterior probability of slow-mode membership; (3) regresses that
# # trial-level probability on Condition with the SAME random-effects
# # structure (1 + Condition per subject) used for the RT model, so the
# # slow-weight result is directly comparable in power and structure to the
# # already-reported RT interaction.

# def fit_pooled_gmm_per_subject(stats, trial_type='GC', rt_key_map=None,
#                                 n_init=10, random_state=0, min_trials=20):
#     """
#     Fit one 2-component GMM per subject, pooling RTs across DBS OFF and
#     DBS ON, to obtain subject-specific but condition-INVARIANT component
#     locations/scales. These fixed components are the reference
#     distributions used later to classify individual trials.

#     Returns {subject: {'mean_fast', 'sd_fast', 'mean_slow', 'sd_slow',
#     'pooled_weight_fast', 'bic_1comp', 'bic_2comp', 'favors_bimodal',
#     'n_pooled_trials'}}.
#     """
#     if rt_key_map is None:
#         rt_key_map = {
#             'GC': 'GC RTs from continue cue (ms)',
#             'GS': 'GS RTs from stop cue (ms)',
#             'GO': 'go_trial RTs (ms)',
#             'GF': 'go_fast_trial RTs (ms)',
#         }
#     rt_key = rt_key_map[trial_type]
#     off_subjects = {k: v for k, v in stats.items() if 'OFF' in k}
#     out = {}

#     for off_key, off_data in off_subjects.items():
#         on_key = off_key.replace('OFF', 'ON')
#         if on_key not in stats:
#             continue
#         subj_id = off_key.split(' ')[0]

#         rts_off = np.asarray(off_data.get(rt_key, []), dtype=float)
#         rts_on = np.asarray(stats[on_key].get(rt_key, []), dtype=float)
#         rts_off = rts_off[np.isfinite(rts_off)] / 1000.0
#         rts_on = rts_on[np.isfinite(rts_on)] / 1000.0
#         pooled = np.concatenate([rts_off, rts_on])

#         if len(pooled) < min_trials:
#             print(f'{subj_id}: only {len(pooled)} pooled trials, skipping.')
#             continue

#         X = pooled.reshape(-1, 1)
#         gmm1 = GaussianMixture(n_components=1, random_state=random_state).fit(X)
#         gmm2 = GaussianMixture(n_components=2, n_init=n_init,
#                                 random_state=random_state).fit(X)
#         bic1, bic2 = gmm1.bic(X), gmm2.bic(X)

#         order = np.argsort(gmm2.means_.flatten())
#         means = gmm2.means_.flatten()[order]
#         weights = gmm2.weights_[order]
#         sds = np.sqrt(gmm2.covariances_.flatten())[order]

#         out[subj_id] = {
#             'mean_fast': means[0], 'sd_fast': sds[0],
#             'mean_slow': means[1], 'sd_slow': sds[1],
#             'pooled_weight_fast': weights[0],
#             'bic_1comp': bic1, 'bic_2comp': bic2,
#             'favors_bimodal': bic2 < bic1,
#             'n_pooled_trials': len(pooled),
#         }

#     n_ok = sum(v['favors_bimodal'] for v in out.values())
#     print(f'Pooled 2-component model favored for {n_ok}/{len(out)} subjects.')
#     return out


# def compute_trial_slow_probabilities(stats, pooled_params, trial_type='GC',
#                                       rt_key_map=None):
#     """
#     For every individual trial (both conditions), compute the posterior
#     probability the trial belongs to the SLOW component, using the
#     subject's fixed pooled Gaussian parameters and a FLAT (0.5/0.5) prior
#     -- i.e. a pure likelihood-ratio classification. Using a flat prior
#     (rather than the pooled fit's own weight) matters: the pooled weight
#     already averages over both conditions, so using it as the prior would
#     partially launder out the very condition effect we're testing for.

#     Returns a long-format trial-level DataFrame: subject, condition, rt,
#     p_slow, and logit_p_slow (clipped logit of p_slow) for use as a
#     mixed-model outcome.
#     """
#     if rt_key_map is None:
#         rt_key_map = {
#             'GC': 'GC RTs from continue cue (ms)',
#             'GS': 'GS RTs from stop cue (ms)',
#             'GO': 'go_trial RTs (ms)',
#             'GF': 'go_fast_trial RTs (ms)',
#         }
#     rt_key = rt_key_map[trial_type]
#     off_subjects = {k: v for k, v in stats.items() if 'OFF' in k}
#     rows = []

#     for off_key, off_data in off_subjects.items():
#         on_key = off_key.replace('OFF', 'ON')
#         if on_key not in stats:
#             continue
#         subj_id = off_key.split(' ')[0]
#         if subj_id not in pooled_params:
#             continue
#         p = pooled_params[subj_id]

#         for cond_label, cond_data in [('DBS OFF', off_data),
#                                        ('DBS ON', stats[on_key])]:
#             rts_ms = np.asarray(cond_data.get(rt_key, []), dtype=float)
#             rts_ms = rts_ms[np.isfinite(rts_ms)]
#             rts_s = rts_ms / 1000.0
#             if len(rts_s) == 0:
#                 continue

#             dens_fast = scipy.stats.norm.pdf(rts_s, loc=p['mean_fast'],
#                                               scale=p['sd_fast'])
#             dens_slow = scipy.stats.norm.pdf(rts_s, loc=p['mean_slow'],
#                                               scale=p['sd_slow'])
#             p_slow = dens_slow / (dens_fast + dens_slow + 1e-300)
#             p_slow_clip = np.clip(p_slow, 1e-4, 1 - 1e-4)
#             logit_p = np.log(p_slow_clip / (1 - p_slow_clip))

#             for rt, ps, lp in zip(rts_s, p_slow, logit_p):
#                 rows.append({
#                     'subject': subj_id, 'condition': cond_label,
#                     'rt': rt, 'p_slow': ps, 'logit_p_slow': lp,
#                 })

#     return pd.DataFrame(rows)


# def fit_slow_weight_mixedlm(trial_slow_df, dv='logit_p_slow',
#                              condition_ref='DBS OFF'):
#     """
#     Trial-level test of whether DBS increases the (soft) probability that
#     a trial is generated by the slow component -- same random-effects
#     structure (1 + Condition per subject) as the RT model, so this is
#     directly comparable in power/design to the already-reported RT result.
#     """
#     import statsmodels.formula.api as smf

#     df = trial_slow_df.copy()
#     df['condition'] = pd.Categorical(
#         df['condition'],
#         categories=[condition_ref] + [c for c in df['condition'].unique()
#                                        if c != condition_ref])

#     cond_term = f'C(condition, Treatment("{condition_ref}"))'
#     model = smf.mixedlm(f'{dv} ~ {cond_term}', data=df, groups=df['subject'],
#                          re_formula=f'1 + {cond_term}')
#     fit = model.fit(reml=True, method='lbfgs')
#     print(fit.summary())
#     return fit


# def fit_slow_weight_hard_assignment_gee(trial_slow_df, condition_ref='DBS OFF',
#                                          threshold=0.5):
#     """
#     Robustness check: hard-assign each trial to fast/slow (p_slow >
#     threshold) and fit a population-averaged logistic GEE model
#     (exchangeable within-subject correlation, cluster-robust SEs). This
#     doesn't rely on the continuous logit-transformed outcome or on
#     mixed-model variance-component estimation, so agreement between this
#     and fit_slow_weight_mixedlm() is a useful convergence check.
#     """
#     import statsmodels.api as sm
#     import statsmodels.formula.api as smf

#     df = trial_slow_df.copy()
#     df['slow_trial'] = (df['p_slow'] > threshold).astype(int)
#     df['condition'] = pd.Categorical(
#         df['condition'],
#         categories=[condition_ref] + [c for c in df['condition'].unique()
#                                        if c != condition_ref])

#     cond_term = f'C(condition, Treatment("{condition_ref}"))'
#     model = smf.gee(f'slow_trial ~ {cond_term}', groups='subject', data=df,
#                      family=sm.families.Binomial(),
#                      cov_struct=sm.cov_struct.Exchangeable())
#     fit = model.fit()
#     print(fit.summary())
#     return fit
