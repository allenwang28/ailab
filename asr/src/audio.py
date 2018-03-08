# -*- coding: utf-8 *-* 
"""Audio Processing Module

This module provides anything that might be used for audio processing. 

Notes:
    A lot of the implementation is derived from
    http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

Todo:
    * better naming
    * sphinx.ext.todo
"""

import numpy as np
import pandas as pd
from scipy.fftpack import dct
import soundfile as sf
import os

def get_file_data(audio_file_path):
    """Get the signal and sample rate from a given audio file

    Gets the signal as an np.array and sample rate given an audio file path
    using soundfile.

    Args:
        audio_file_path (str): The path to the audio file.
            Supported file types:
                - flac

    Returns:
        np.array : the data of the audio file 
        int      : the sample rate of the audio file

    """
    with open(audio_file_path, 'rb') as f:
        signal, samplerate = sf.read(f)
    return (signal, samplerate)

def apply_pre_emphasis_to_signal(signal, pre_emphasis=0.97):
    """Applies pre-emphasis to a signal

    A pre-emphasis filter amplifies high frequencies in a signal.
    Its equation:

        y(t) = x(t) - a * x(t-1)

    where x is the signal and a is the pre_emphasis coefficient
    

    Args:
        signal (np.array): the input signal.
        pre_emphasis (int, optional): pre_emphasis coefficient.
            Typically used values: 0.95, 0.97. Defaults to 0.97

    Returns:
        np.array : The signal with applied pre-emphasis filter

    """
    return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])  

def extract_frames_from_signal(signal,
                               samplerate,
                               frame_size_in_ms=25,
                               frame_stride_in_ms=10,
                               window_function=None):
    """Extracts window frames from a provided signal.

    Args:
        signal (np.array): the input signal.
        samplerate (int): the sample rate.
        frame_size_in_ms (:obj:`int`, optional): frame size in milliseconds.
            Defaults to 25.
        frame_stride_in_ms (:obj:`int`, optional): frame stride in milliseconds. 
            Defaults to 10.
        window_function (:obj:`function`, optional): window function
            Defaults to None. For some options, refer to:
            https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.window.html

    Returns:
        np.array : The matrix of all extracted windows.

    """
    frame_size = frame_size_in_ms / 1000.
    frame_stride = frame_stride_in_ms / 1000.

    frame_length = frame_size * samplerate
    frame_step = frame_stride * samplerate

    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step))

    padded_signal_length = num_frames * frame_step + frame_length
    padding_length = padded_signal_length - signal_length
    padding = np.zeros(padding_length)

    # Pad the signal to accommodate window size.
    pad_signal = np.append(signal, padding)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)] 

    if window_function:
        frames = frames * window_function(frame_length)

    return frames

def get_fft_from_frames(frames, NFFT):
    """Get FFT from provided frames.

    Apply N-point fast fourier transform using numpy to extracted frames.

    Args:
        frames (np.array): the frames
        NFFT (int): N, typically 256 or 512

    Returns:
        np.array : the FFT

    """
    return np.fft.rfft(frames, NFFT)

def get_power_spectrum_from_frames(frames, NFFT):
    """Get the power spectrum from frames.

    The power spectrum is:
        |FFT(signal)|^2 / N

    Args:
        frames (np.array): the frames
        NFFT (int): N for the N-point FFT

    Returns:
        np.array : the power spectrum 
    """
    return np.abs(get_fft_from_frames(frames, NFFT))**2 / NFFT

def frequency_to_mel(f):
    """Convert frequency (Hz) to mel-scale"""
    return 2595 * np.log10(1 + (f/2)/700.)

def mel_to_frequency(m):
    """Convert from mel-scale to frequency (Hz)"""
    return 700 * (10**(m / 2595) - 1)

def get_filter_banks_from_power_spectrum(power_spectrum, NFFT, samplerate, num_filters=40):
    """Get filter banks from the power spectrum

    Args:
        power_spectrum (np.array): the power spectrum 
            extracted from the signal frames 
        NFFT (int): the NFFT used for extracting the power spectrum
        samplerate (int): the sample rate of the signal.
        num_filters (:obj:`int`, optional) : desired number of filters in the filter bank
            Defaults to 40.
    
    Returns:
        np.array : extracted filter banks

    """
    low_freq = 0
    high_freq = samplerate

    low_mel = frequency_to_mel(low_freq)
    high_mel = frequency_to_mel(high_freq)

    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    freq_points = mel_to_frequency(mel_points)
    fft_bins = np.floor((NFFT + 1) * freq_points / samplerate)

    filters = np.zeros((num_filters, int(np.floor(NFFT/2 + 1))))

    for m in range(1, num_filters + 1):
        left = int(fft_bins[m - 1])
        center = int(fft_bins[m])
        right = int(fft_bins[m + 1])

        for k in range(left, center):
            filters[m - 1, k] = (k - fft_bins[m - 1]) / (fft_bins[m] - fft_bins[m - 1])
        for k in range(center, right):
            filters[m - 1, k] = (fft_bins[m + 1] - k) / (fft_bins[m + 1] - fft_bins[m])
    filter_banks = np.dot(power_spectrum, filters.T)
    # Replace all instances of 0 with a very small number (epsilon) to avoid problems with log
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    return 20 * np.log10(filter_banks)

def get_mfcc_coefficients_from_filter_banks(filter_banks, cep_lifter=22):
    """Get MFCC coefficients from filter banks

    Applies DCT to decorrelate the filter bank coefficients.

    Note:
        For ASR, typically only cepstral coefficients 2-13 are retained.
        For completeness, this isn't automatically filtered in this function.

    Args:
        filter_banks (np.array): Filter banks extracted from the signal.
        cep_lifter (:obj:`int`, optional): Parameter in sinusoidal liftering.
            Defaults to 22. If None, then sinusoidal_liftering is not applied.

    Returns:
        np.array : MFCC coefficients

    """
    mfcc = dct(filter_banks, axis=1, norm='ortho')
    if cep_lifter:
        (n_frames, n_coeff) = mfcc.shape
        n = np.arange(n_coeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
    return mfcc


def mean_normalize(features):
    """Apply mean normalization.

    Mean normalization can be applied to MFCC coefficients or from filter banks.
    This is used to balance the spectrum and to improve the SNR.

    Args:
        features (np.array) : MFCCs or filter banks

    Returns:
        np.array : mean normalized
    """
    return features - (np.mean(features, axis=0) + 1e-8)

def get_filterbank_from_file(file_path):
    """Get the filter bank from an audio file.

    Extracts a filter bank using typical ASR parameters.

    Args:
        file_path (str): path to the file

    Returns:
        np.array : normalized filter bank

    """
    NFFT = 512

    signal, samplerate = get_file_data(file_path)
    signal = apply_pre_emphasis_to_signal(signal)

    frames = extract_frames_from_signal(signal, samplerate, window_function=np.hamming)
    power_spectrum = get_power_spectrum_from_frames(frames, NFFT)

    filter_banks = get_filter_banks_from_power_spectrum(power_spectrum, NFFT, samplerate)
    return mean_normalize(filter_banks)

def get_mfcc_from_file(file_path):
    """Get the MFCC from an audio file.

    Extracts MFCC features using typical ASR parameters.

    Args:
        file_path (str): path to the file

    Returns:
        np.array : normalized MFCC

    """
    NFFT = 512

    signal, samplerate = get_file_data(file_path)
    signal = apply_pre_emphasis_to_signal(signal)

    frames = extract_frames_from_signal(signal, samplerate, window_function=np.hamming)
    power_spectrum = get_power_spectrum_from_frames(frames, NFFT)

    filter_banks = get_filter_banks_from_power_spectrum(power_spectrum, NFFT, samplerate)
    mfcc = get_mfcc_coefficients_from_filter_banks(filter_banks)
    return mean_normalize(mfcc)

def get_banks_and_mfcc_from_file(file_path):
    """Get the filter banks and MFCC from an audio file.

    Extracts filter banks and MFCC features using typical ASR parameters.

    Args:
        file_path (str): path to the file

    Returns:
        np.array : normalized filter bank
        np.array : normalized MFCC

    """
    NFFT = 512

    signal, samplerate = get_file_data(file_path)
    signal = apply_pre_emphasis_to_signal(signal)

    frames = extract_frames_from_signal(signal, samplerate, window_function=np.hamming)
    power_spectrum = get_power_spectrum_from_frames(frames, NFFT)

    filter_banks = get_filter_banks_from_power_spectrum(power_spectrum, NFFT, samplerate)
    mfcc = get_mfcc_coefficients_from_filter_banks(filter_banks)
    return mean_normalize(filter_banks), mean_normalize(mfcc)

# Code to test this module
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #file_path = '../data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac'
    file_path = '../data/OSR_us_000_0010_8k.wav'
    
    filter_bank = get_filterbank_from_file(file_path)
    mfcc = get_mfcc_from_file(file_path)

    f = plt.figure()
    plt.imshow(np.flipud(filter_bank.T), cmap=plt.cm.jet, aspect=0.1, extent=[0,2,0,4])

    f = plt.figure()
    plt.imshow(np.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.1, extent=[0,2,0,4])
    plt.show()
