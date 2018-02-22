# -*- coding: utf-8 *-* 
"""Audio Processing Module

This module provides anything that might be used for audio processing. 

Todo:
    * Allow for more than just Hamming windows?
    * sphinx.ext.todo
"""

import numpy as np
import pandas as pd
import soundfile as sf
import os

class NoProvidedAudioException(Exception):
    """Exception used in AudioProcessor class if no audio data is provided.

    The AudioProcessor class has an optional attribute for a file path, and
    its functions have optional parameters for a signal. This is so that
    a user can provide either an audio file or the data directly.

    This exception will raise in the case that an audio file is not provided at all
    to a function requiring it.

    Args:
        msg (str): Human readable string describing the exception
        code (:obj:`int`, optional) : Error code.

    Attributes:
        msg (str): Human readable string describing the exception
        code (int) : Error code.

    """
    def __init__(self, msg, code=None):
        self.msg = msg
        self.code = code


class AudioProcessor(object):
    """Class for processing audio files.

    AudioProcessor is a class used for processing audio for tasks such as:
        - opening an audio file

    Attributes:
        - audio_file_path

    """

    def __init__(self, audio_file_path=None):
        """Initialize AudioProcessor

        Args:
            audio_file_path (:obj:`str`, optional): The path to the audio file.
                If a file is supplied, the class will use it by default any of its class functions.
                Defaults to None. 
        """
        if audio_file_path:
            self.set_audio_file(audio_file_path)
        else:
            self.__signal = None
            self.__samplerate= None
            self.__audio_file_path = None
        self.__frames = None

    def set_audio_file(self, audio_file_path):
        """Assigns an audio file to the class and opens it

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
        self.__audio_file = audio_file_path
        self.__signal = signal
        self.__samplerate = samplerate

        return (data, samplerate)

    def apply_pre_emphasis_to_signal(self, pre_emphasis=0.97, signal=None, samplerate=None):
        """Applies pre-emphasis to a signal

        A pre-emphasis filter amplifies high frequencies in a signal.
        Its equation:

            y(t) = x(t) - a x(t-1)

        where x is the signal and a is the pre_emphasis coefficient
        

        Args:
            pre_emphasis (int, optional): pre_emphasis coefficient.
                Typically used values: 0.95, 0.97. Defaults to 0.97
            signal (:obj:`np.array`, optional): the input signal .
                Defaults to None. If not provided, it will try and
                use the signal from the audio file.
            samplerate (:obj`int`, optional): the sample rate.
                Defaults to None. If not provided, it will try and
                use the sample rate from the audio file.

        Returns:
            np.array : The signal with applied pre-emphasis filter

        Raises:
            NoProvidedAudioException: If set_audio_file was never
                called or unsuccessful and a signal or samplerate 
                was not provided.

        """
        if not signal or not samplerate and not self.__audio_path:
            raise NoProvidedAudioException("apply_pre_emphasis_to_signal")

        if not signal or not samplerate:
            signal = self.__signal
            samplerate = self.__samplerate

        return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])  

    def extract_window_frames_from_signal(self,
                                          frame_size_in_ms=25,
                                          frame_stride_in_ms=10,
                                          apply_window_function=True,
                                          signal=None,
                                          samplerate=None):
        """Extracts window frames from a provided signal.

        Args:
            frame_size_in_ms (:obj:`int`, optional): frame size in milliseconds.
                Defaults to 25.
            frame_stride_in_ms (:obj:`int`, optional): frame stride in milliseconds. 
                Defaults to 10.
            apply_window_function (:obj:`bool`, optional): apply Hamming window function
                to frames. Defaults to True
            signal (:obj:`np.array`, optional): the input signal .
                Defaults to None. If not provided, it will try and
                use the signal from the audio file.
            samplerate (:obj`int`, optional): the sample rate.
                Defaults to None. If not provided, it will try and
                use the sample rate from the audio file.

        Returns:
            np.array : The matrix of all extracted windows.

        Raises:
            NoProvidedAudioException: If set_audio_file was never
                called or unsuccessful and a signal or samplerate 
                was not provided.

        """
        if not signal or not samplerate and not self.__audio_path:
            raise NoProvidedAudioException("apply_pre_emphasis_to_signal")

        if not signal or not samplerate:
            signal = self.__signal
            samplerate = self.__samplerate

        frame_size = frame_size_in_ms/1000.
        frame_stride = frame_stride_in_ms/1000.

        frame_length = frame_size * samplerate
        frame_step = frame_stride * samplerate

        signal_length = len(signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))

        # Make sure that we have at least 1 frame
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))

        # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        pad_signal = np.append(signal, z)

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)] 

        if apply_window_function:
            frames = frames * np.hamming(frame_length)

        self.__frames = frames
        return frames

    
