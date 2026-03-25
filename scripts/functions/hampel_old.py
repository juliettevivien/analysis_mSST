#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:13:54 2024

@author: okohl

Functions that allow to apply hampel filter to electophysiological recordings
acquired during DBS-On.

Two implementations of hampel filter can be selected based on cleaning method
parameter:
    'median': replace identified outliers with moving median (similar to Allen et al. 2010)
    'attenuation': attenuated outlier segments by scaling them according to power in surrounding samples
    
Hampel-Filter in short: 
        1) Time Courses are transformed into frequency domain with short fast 
            fourier transform (sftf).
        2) MAD outlier detection is applied to the spectra to identify outliers (=DBS-related peaks).
        3) Only detected outliers in frequency range of interest are kept.
        4) Outliers are attenuated or replaced with moving median.
        5) Cleaned spectra are projected back to time courses with inverse sftf

Additionally: 
    + two plotting functions are provided that should enable to quickly
      check effects of hampelfilter and therby, inform the tuning of the cval/n_sigmas value.
    + Function writing obtained cleaned TCs to mne.Raw obejct for further MNE use.

"""

import numpy as np 
from scipy.ndimage import median_filter
from scipy.signal import stft, istft, check_NOLA, welch
import mne
import matplotlib.pyplot as plt


def moving_vars(series, window_size, scale="normal"):
    """
    Calculate the moving median and moving MAD (Median Absolute Deviation) of a time series.

    Parameters
    ----------
    series : np.ndarray
        The input time series. Can be 1d (time) or 2d (channels x time).
    window_size : int
        The size of the moving window (should be an odd number).
    scale: If 'normal', scales the MAD for normal distribution consistency (multiplies by 1.4826).
        Use None or False for unscaled MAD.

    Returns
    -------
    moving_median : np.ndarray
        The moving median of the input series.
    moving_mad : np.ndarray
        The moving MAD of the input series.
    """

    # if series is 2d adjust time window to be 2d array
    if series.ndim == 2:
        window_size = (1,window_size)

    # Calculate the moving median using a median filter
    moving_median = median_filter(series, size=window_size, mode='reflect')
    
    # Calculate the deviations from the moving median
    deviations = np.abs(series - moving_median)
    
    # Calculate the moving MAD
    moving_mad = median_filter(deviations, size=window_size, mode='reflect')
    
    # Optionally scale the MAD to approximate estimate of standard deviation under assumption of normality
    if scale == 'normal':
        moving_mad *= 1.4826
    # plt.plot(moving_mad, label='Moving MAD')
    # plt.plot(moving_median, label='Moving Median')

    return moving_median, moving_mad


def detect_mad_outliers(series, window_size, n_sigmas=6):
    """
    Detect outliers in a time series using the moving median and moving MAD.

    Parameters
    ----------
    series : np.ndarray
        The input time series.Can be 1d (time) or 2d (channels x time).
    window_size : int
        The size of the moving window (should be an odd number).
    n_sigmas : int
        The threshold in terms of MADs for detecting outliers.

    Returns
    -------
    outliers : np.ndarray
       A binary array of the same length as `series` where 1 indicates an outlier and 0 indicates a normal point.
    """

    moving_median, moving_mad = moving_vars(series, window_size)

    # Identify outliers where the deviation exceeds n_sigmas * moving MAD
    outliers = np.abs(series - moving_median) > n_sigmas * moving_mad

    return outliers

def attenuate_outliers(data, outlier_mask, extension_samples=8, filter_thresh=98, scaling_thresh=50):
    """
    Processes data by handling outlier segments based on thresholds, using a binary mask for outliers.

    Parameters
    ----------
    data : np.ndarray
        The input data array, where rows are time points and columns are different variables (e.g., sensors, channels).
    outlier_mask : np.ndarray
        A binary array of the same length as `data` where 1 indicates an outlier and 0 indicates a normal point.
    extension_samples : int
        The number of samples to extend the range around the outliers.
    filter_thresh : float
        The lower percentile threshold for filtering outliers.
        Only moments with values lower then threhold are attenuated.
    scaling_thresh : float
        The upper percentile threshold for scaling outliers.
        Determines how string outlier samples are attenuated.

    Returns
    -------
    cleaned_data : np.ndarray
        The processed data with outliers handled according to the defined thresholds.
    """
 
    # Find the start and end of each outlier segment
    diff_outliers = np.diff(outlier_mask.astype(int))
    start_indices = np.where(diff_outliers == 1)[0] + 1
    end_indices = np.where(diff_outliers == -1)[0]

    # Handle cases where the outlier starts at the first sample or ends at the last sample
    if outlier_mask[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    if outlier_mask[-1] == 1:
        end_indices = np.append(end_indices, len(outlier_mask) - 1)

    # Initialize the output with the original time series
    cleaned_data = np.copy(data)

    # Process each outlier segment
    for start, end in zip(start_indices, end_indices):
        selected = np.arange(start, end + 1)

        # Extend the segment by windval on both sides
        extended_selection = np.arange(max(0, start - extension_samples), min(len(data), end + extension_samples + 1))

        # Exclude the selected (original) outlier indices from extended_selection
        extension = np.setdiff1d(extended_selection, selected)

        # Compute the square of the data in the extended window
        extended_data = data[extension] ** 2

        # Calculate the lower and upper thresholds  
        # l1: Moments in outlier segment that exceed l1 are not attenuated
        # l2: Scaling factor which determines strength of attenuation
        l1 = np.percentile(extended_data, filter_thresh, axis=0)
        l2 = np.percentile(extended_data, scaling_thresh, axis=0)

        # Process the selected outlier data
        outlier_data = data[selected]
        is_outlier = (outlier_data ** 2) > l1

        # Calculate the multiplier
        scaling_factor = np.sqrt(l2) / np.abs(outlier_data)

        # Apply the multiplier to the outliers that exceed the lower threshold
        outlier_data[is_outlier] = outlier_data[is_outlier] * scaling_factor[is_outlier]

        # Update the data with the processed outliers
        cleaned_data[selected] = outlier_data

    return cleaned_data


def median_interpolate_outliers(series, outlier_mask, window_size):
    """
    Processes data by handling outlier segments based on thresholds, using a binary mask for outliers.

    Parameters
    ----------
    data : np.ndarray
        The input data array, where rows are time points and columns are different variables (e.g., sensors, channels).
    outlier_mask : np.ndarray
        A binary array of the same length as `data` where 1 indicates an outlier and 0 indicates a normal point.
    window_size : int
        Size of moving window used for moving median and moving MAD calculation (in samples).

    Returns
    -------
    cleaned_data : np.ndarray
        The processed data with outliers handled according to the defined thresholds.
    """
 
    # Get moving median for time series
    moving_median, _ = moving_vars(series, window_size, scale="normal")
 
    # Insert median at samples where outliers were detected
    cleaned_data = series.copy()
    cleaned_data[outlier_mask] = moving_median[outlier_mask]
    
    return cleaned_data


def hampel_filter(tc, sfreq, cleaning_method='attenuation', sftf_window=60, moving_window=3, cval=8, frequency_range=[14,80]):
    """
    Clean raw MEG data from DBS-Stimulation Artefacts.
    
    Function detects DBS-artefacts with the help of MAD-outlier detection and
    'cleanes' them either replacing them with a moving median or attenuating them
    based on a procedure developed in Duesseldorf.
    
    Conducted steps are:
        1) Time Courses are transformed into frequency domain with short fast 
            fourier transform (sftf).
        2) MAD outlier detection is applied to the spectra to identify peaks.
        3) Detected outliers are adjusted based on frequency range of interest.
        4) Outliers are attenuated or replaced with moving median.
        5) Cleaned spectra are projected back to time courses with inverse sftf

    Importantly, outliers are detected in avaerg spectrum across all channels and
    if outlier frequency bin is detected, outlier 'cleaning' is performed on all 
    channels even if no artefact is present in a particular channel.
    
    Parameters:
    file: str
        path to the raw .fif file
    sftf_window: int
        window size for STFT in seconds (default: 60)
    moving_window: int
        frequency range in Hz spanned by the sliding window for MAD outlier detection (default: 3)
    cval: int
        threshold for Hampel filter outlier detection (default: 6)
    extend_by: int
        number of samples that are added around outlier before outlier is attenuated. Corresponds to 6 frequency bins (0.024Hz when sampling rate = 250Hz).
    frequency_range: list
        Frequency range to which outlier cleaning is applied.
        If only one frequency range give it as list, if more

    Returns:
    cleaned_raw: 
        MNE raw object with cleaned data
    """
    
    # Make sure that frequency range is nested list
    frequency_range = ensure_nested_list(frequency_range)
    
    # --- Bring Time courses into frequency domain with STFT ---
    
    # Set STFT parameters
    nperseg = int(sftf_window * sfreq)  # Window Size
    noverlap = nperseg // 2
    fs = sfreq  # Sampling frequency
    return_onesided = True  # Return one-sided spectrum
    scaling = 'psd'  # Scale to power spectral density
    
    # Check if parameters allow STFT to be inverted
    if check_NOLA(window='hann', nperseg=nperseg, noverlap=noverlap):
        print('Parameter selection allows STFT to be inverted.')
    else:
        raise ValueError('Parameter selection does not allow STFT to be inverted.')
    
    # Calculate STFT
    f, t, Zxx = stft(tc, 
                     fs=fs, 
                     nperseg=nperseg, 
                     noverlap=noverlap, 
                     return_onesided=return_onesided, 
                     scaling=scaling)
    
    # Calculate signal's energy and average PSDs across windows and channels
    Zxx2 = np.abs(Zxx) ** 2
    psd = Zxx2.mean(axis=(0,2)) 
    plt.plot(psd, label='Original')

    
    # --- Identify Outlier events based on MAD outlier detection and 'clean' them
    
    # Run MAD outlier detection -> cval crucial for sensitivity of outlier detection
    window_size = int(sfreq * moving_window)  # Convert to number of samples
    outliers = detect_mad_outliers(psd, window_size, n_sigmas=cval)
    
    # Unselect outlier frequency bins outside of frequency ranges of interest
    freq_mask = create_mask_from_frequency_ranges(frequency_range, f)
    outliers[~freq_mask] = 0
            
    # Attenuate detected outlier samples
    if cleaning_method == 'median':            
        # Median interpolation for outlier segments
        attenuated_real = np.apply_along_axis(median_interpolate_outliers, 1, Zxx.real, outlier_mask=outliers, window_size=window_size) # window size can be reduced improve filtering outcomes - but extends running time quite a bit
        attenuated_imag = np.apply_along_axis(median_interpolate_outliers, 1, Zxx.imag, outlier_mask=outliers, window_size=window_size) # window size can be reduced improve filtering outcomes - but extends running time quite a bit
    
    elif cleaning_method == 'attenuation':            
        # Attenuate identified outlier segments
        attenuated_real = np.apply_along_axis(attenuate_outliers, 1, Zxx.real, outlier_mask=outliers)
        attenuated_imag = np.apply_along_axis(attenuate_outliers, 1, Zxx.imag, outlier_mask=outliers)       

    # --- Project Frequency domain data back to time courses ---
    
    # Combine real and imaginary parts into cleaned signal
    cleaned_Zxx = attenuated_real + 1j * attenuated_imag
    
    # Test to see whether outliers outside of expected window were identified
    psd_clean = np.abs(cleaned_Zxx) ** 2
    psd_clean = psd_clean.mean(axis=(2))
    #plt.plot(psd_clean, label='Cleaned')
    plt.legend(bbox_to_anchor=(1, 0.5, 0.5, 0.5), loc='upper right')
    plt.suptitle(f'Power Spectra before and after Hampel Filter \nMethod: {cleaning_method}, cval: {cval}')
    
    # Calculate inverse STFT (ISTFT)
    _, cleaned_tc = istft(cleaned_Zxx, 
                          fs=fs, 
                          nperseg=nperseg, 
                          noverlap=noverlap, 
                          input_onesided=True, 
                          scaling=scaling)
    
    # Trim the cleaned time course to match the original length
    cleaned_tc = cleaned_tc[:, :tc.shape[1]]
    
    return cleaned_tc
    
    
def plot_hampel_filter1(tc, tc_cleaned, sfreq, freq_range=[2,45], nperseg=40):
    # Quick plot of power calculated from original time course and cleaned time course,
    # as well as, difference between power spectra. Importantly, freuqency
    # npersg is very large to get a good look at the artefacts.
         
    # Calculate psds
    f, psd = welch(tc, sfreq, nperseg=sfreq*nperseg)
    _, psd_clean = welch(tc_cleaned, sfreq, nperseg=sfreq*nperseg)
    psd_diff = psd - psd_clean
    
    # Set frequency range of interest
    wide = (f > freq_range[0]) & (f < freq_range[1])
    
    # Grab Data For Plotting
    f_in = f[wide]
    psd_in = psd[:,wide]
    psd_in_clean = psd_clean[:,wide]
    psd_diff = psd_in - psd_in_clean
    
    # Overview plot plotting
    fig, ax = plt.subplots(3, 1, dpi=300)
    
    # Plot original power spectrum
    ax[0].plot(f_in,psd_in.T,linewidth=.5) # Uncleaned Power Spectra
    ax[0].set_ylabel('Power')
    ax[0].set_xticklabels('')
    
    # Plot cleaned power spectrum
    ax[1].plot(f_in,psd_in_clean.T,linewidth=.5) # Cleaned Power Spectra
    ax[1].set_ylabel('Cleaned Power')
    
    # Plot Difference between cleaned and original power spectrum
    ax[2].plot(f_in,psd_diff.T,linewidth=.5) # Diff between Spectra
    ax[2].set_ylabel('Power\nDifference')
    ax[2].set_xlabel('Frequency (Hz)')
    
    plt.show()
    
def plot_hampel_filter2(raw, cleaned_raw, freq_ranges=[[1,50],[10,20],[25,35]],ephys_chans='misc', n_fft=1000, n_overlap=500):
    # Quick plot of power calculated from original time course and cleaned time course.
    # Zooms on 3 freq ranges of interest
    
    fig, axs = plt.subplots(ncols=3, nrows=2,
                            layout="constrained",dpi=300)
    
    l_freq, h_freq = freq_ranges[0]
    mne.viz.plot_raw_psd(raw, fmin=l_freq, fmax=h_freq , n_fft=n_fft, n_overlap=n_overlap, picks=ephys_chans, ax=axs[0,0])
    axs[0,0].set_title(f'{l_freq}-{h_freq}Hz')
    axs[0,0].tick_params(axis='both', which='major',labelsize=8)
    axs[0,0].locator_params(nbins=3)
    
    l_freq, h_freq = freq_ranges[0]
    mne.viz.plot_raw_psd(cleaned_raw, fmin=l_freq, fmax=h_freq ,n_fft=n_fft, n_overlap=n_overlap, picks=ephys_chans, ax=axs[1,0])
    axs[1,0].set_title('')
    axs[1,0].tick_params(axis='both', which='major',labelsize=8)
    axs[1,0].locator_params(nbins=3)
    
    l_freq, h_freq = freq_ranges[1]
    mne.viz.plot_raw_psd(raw, fmin=l_freq, fmax=h_freq , n_fft=n_fft, n_overlap=n_overlap, picks=ephys_chans, ax=axs[0,1])
    axs[0,1].set_title(f'{l_freq}-{h_freq}Hz')
    axs[0,1].set_ylabel('')
    axs[0,1].tick_params(axis='both', which='major',labelsize=8)
    axs[0,1].locator_params(nbins=3)
    
    l_freq, h_freq = freq_ranges[1]
    mne.viz.plot_raw_psd(cleaned_raw, fmin=l_freq, fmax=h_freq, n_fft=n_fft, n_overlap=n_overlap, picks=ephys_chans, ax=axs[1,1])
    axs[1,1].set_title('')
    axs[1,1].set_ylabel('')
    axs[1,1].tick_params(axis='both', which='major',labelsize=8)
    axs[1,1].locator_params(nbins=3)
    
    l_freq, h_freq = freq_ranges[2]
    mne.viz.plot_raw_psd(raw, fmin=l_freq, fmax=h_freq, n_fft=n_fft, n_overlap=n_overlap, picks=ephys_chans, ax=axs[0,2])
    axs[0,2].set_title(f'{l_freq}-{h_freq}Hz')
    axs[0,2].set_ylabel('')
    axs[0,2].tick_params(axis='both', which='major',labelsize=8)
    axs[0,2].locator_params(nbins=3)
    
    l_freq, h_freq = freq_ranges[2]
    mne.viz.plot_raw_psd(cleaned_raw, fmin=l_freq, fmax=h_freq, n_fft=n_fft, n_overlap=n_overlap, picks=ephys_chans, ax=axs[1,2])
    axs[1,2].set_title('')
    axs[1,2].set_ylabel('')
    axs[1,2].tick_params(axis='both', which='major',labelsize=8)
    axs[1,2].locator_params(nbins=3)
    
    plt.show()

def create_mne_raw(cleaned_tc, raw, ephys_chans="meg", extra_chans=["stim","emg"]):
    # Importantly, this fuction is a modified version of 
    # https://osl-ephys.readthedocs.io/en/latest/_modules/osl_ephys/source_recon/parcellation/parcellation.html#convert2mne_raw
    # Please reference osl-ephys if you use this function.

    # make sure that extra chans are in list
    if isinstance(extra_chans, str):
        extra_chans = [extra_chans]
    
    # Grab only MEEG channels and create info object
    raw1 = raw.copy().pick(ephys_chans)
    info = raw1.info
    
    # Create Raw object
    new_raw = mne.io.RawArray(cleaned_tc, info)

    # Copy timing info
    new_raw.set_meas_date(raw.info["meas_date"])
    new_raw.__dict__["_first_samps"] = raw.__dict__["_first_samps"]
    new_raw.__dict__["_last_samps"] = raw.__dict__["_last_samps"]
    new_raw.__dict__["_cropped_samp"] = raw.__dict__["_cropped_samp"]

    # Copy annotations from raw
    new_raw.set_annotations(raw._annotations)

    # Add extra channels
    for extra_chan in extra_chans:
        if extra_chan in raw:
            chan_raw = raw.copy().pick(extra_chan)
            chan_data = chan_raw.get_data()
            chan_info = mne.create_info(chan_raw.ch_names, raw.info["sfreq"], [extra_chan] * chan_data.shape[0])
            chan_raw = mne.io.RawArray(chan_data, chan_info)
            new_raw.add_channels([chan_raw], force_update_info=True)

    # Copy the description from the sensor-level Raw object
    new_raw.info["description"] = raw.info["description"]
    
    return new_raw
          
    
def ensure_nested_list(input_list):
    """
    Ensures that the input is a nested list.
    If the input is a list, it is wrapped in an additional list unless already nested.
    
    Args:
        input_list (list): The input list to check.
    
    Returns:
        list: A nested list version of the input.
    """
    # Check if the input is already a nested list
    if isinstance(input_list, list) and all(isinstance(item, list) for item in input_list):
        return input_list  # Already nested
    elif isinstance(input_list, list):
        return [input_list]  # Wrap the list in another list
    else:
        raise TypeError("Input must be a list.")
  
        
def create_mask_from_frequency_ranges(ranges, freq_bins):
    """
    Converts a list of frequency ranges (in Hz) into a mask for a power spectrum.
    
    Parameters:
        ranges (list of lists): Nested list with start and end frequencies in Hz, e.g., [[3, 5], [10, 12]].
        freq_bins (np.ndarray): Array of frequencies corresponding to the power spectrum bins.
        
    Returns:
        np.ndarray: Boolean mask of the same length as freq_bins.
    """
    # Initialize mask with all False values
    n_freq_bins = len(freq_bins)
    mask = np.zeros(n_freq_bins, dtype=bool)
    
    # Iterate over each range and set True for the indices within the range
    for start_freq, end_freq in ranges:
        # Find indices corresponding to the range
        indices = np.where((freq_bins >= start_freq) & (freq_bins <= end_freq))[0]
        mask[indices] = True
    
    return mask
    