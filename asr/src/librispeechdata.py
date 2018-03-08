# -*- coding: utf-8 *-* 
"""LibriSpeech Folder Parsing Module

This module is used for processing the files provided by LibriSpeech
to be used in our models.

The LibriSpeech folder structure (from LibriSpeech/README.txt):
<corpus root>
    |
    .- README.TXT
    |
    .- READERS.TXT
    |
    .- CHAPTERS.TXT
    |
    .- BOOKS.TXT
    |
    .- train-clean-100/
                   |
                   .- 19/
                       |
                       .- 198/
                       |    |
                       |    .- 19-198.trans.txt
                       |    |    
                       |    .- 19-198-0001.flac
                       |    |
                       |    .- 14-208-0002.flac
                       |    |
                       |    ...
                       |
                       .- 227/
                            | ...
, where 19 is the ID of the reader, and 198 and 227 are the IDs of the chapters
read by this speaker. The *.trans.txt files contain the transcripts for each
of the utterances, derived from the respective chapter and the FLAC files contain
the audio itself.

Notes:
    So far, we have only been working from the dev-clean folder.
    It should work for all LibriSpeech folders, assuming their folder structures
    are consistent.

Todo:
    * Test for other folders
"""

import os
import numpy as np
import re
from audio import get_mfcc_from_file 

def voice_txt_dict_from_path(folder_path):
    """Creates a dictionary between voice ids and txt file paths

    Walks through the provided LibriSpeech folder directory and 
    creates a dictionary, with voice_id as a key and all associated text files

    Args:
        folder_path (str): The path to the LibriSpeech folder
    
    Returns:
        dict : Keys are the voice_ids, values are lists of the paths to trans.txt files.

    """
    voice_txt_dict = {}

    for dir_name, sub_directories, file_names in os.walk(folder_path):
        if len(sub_directories) == 0: # this is a root directory
            file_names = list(map(lambda x: os.path.join(dir_name, x), file_names))
            txt_file = list(filter(lambda x: x.endswith('txt'), file_names))[0]

            voice_id = os.path.basename(dir_name)

            if voice_id not in voice_txt_dict:
                voice_txt_dict[voice_id] = [txt_file]
            else:
                voice_txt_dict[voice_id].append(txt_file)

    return voice_txt_dict

def transcriptions_and_flac(txt_file_path):
    """Gets the transcriptions and .flac files from the path to a trans.txt file.

    Given the path to a trans.txt file, this function will return a list of 
    transcriptions and a list of the .flac files. Each flac file entry index corresponds 
    to the transcription entry index.

    Args:
        txt_file_path (str): The path to a trans.txt file

    Returns:
        list : A list of transcriptions
        list : A list of paths to .flac files
    """
    transcriptions = []
    flac_files = []

    parent_dir_path = os.path.dirname(txt_file_path)

    with open(txt_file_path, 'rb') as f:
        lines = [x.decode('utf8').strip() for x in f.readlines()]
    
    for line in lines:
        splitted = re.split(' ', line)
        file_name = splitted[0]
        transcriptions.append(" ".join(splitted[1:]))
        flac_files.append(os.path.join(parent_dir_path, "{0}.flac".format(file_name)))
 
    return transcriptions, flac_files 

def get_data_from_path(folder_path):
    """Gets the features, transcriptions, and ids from a folder path

    Given the path to a LibriSpeech directory, get all MFCC features from
    all .flac files, their associated speaker (voice_id), and the transcription

    Args:
        folder_path (str): The path to a LibriSpeech folder

    Returns:
        list : The MFCC features extracted from .flac files
        list : The transcriptions from .trans.txt files
        list : The voice ids

    """
    voice_txt_dict = voice_txt_dict_from_path(folder_path)

    features = []
    transcriptions = []
    ids = []

    for voice_id in voice_txt_dict:
        txt_files = voice_txt_dict[voice_id]
        for txt_file in txt_files:
            t, flac_files = transcriptions_and_flac(txt_file)

            transcriptions += t

            for flac_file in flac_files:
                features.append(get_mfcc_from_file(flac_file)[:, 1:12]) # Save only cepstral coefficients 2-13
                ids.append(voice_id)
    return features, transcriptions, ids

def save_data(data, folder_path, file_names=['features', 'transcriptions', 'ids']):
    """Given data, save to numpy arrays

    Given a list of features, transcriptions, and ids, save each to numpy files.

    Args:
        data (tuple): A tuple of 3 lists - features, transcriptions, ids.
        folder_path (str): The full path destination to save to
        file_names (:obj:`list`, optional): A list of names for each file,
            in order of features, transcriptions, ids.
            Defaults to ['features','transcriptions','ids']

    """
    features, transcriptions, ids = data

    # First 0-pad the features.
    max_feature_length = max(feature.shape[0] for feature in features)
    padded_features = []
    for feature in features:
        pad_length = max_feature_length - feature.shape[0]
        padded = np.pad(feature, ((0, pad_length), (0,0)), 'constant')
        padded_features.append(padded)

    padded_features = np.dstack(padded_features)
    transcriptions = np.array(transcriptions)
    ids = np.array(ids)

    np.save(os.path.join(folder_path, "{0}.npy".format(file_names[0])), padded_features)
    np.save(os.path.join(folder_path, "{0}.npy".format(file_names[1])), transcriptions)
    np.save(os.path.join(folder_path, "{0}.npy".format(file_names[2])), ids)

def mfcc_batch_generator(batch_size, folder_paths = [], label = 'transcription'):
    """Given folder paths, return a mfcc batch generator

    Saving all of the mfcc features from .flac files may be overkill.
    Use this to retrieve data in batches.

    Args:
        batch_size (int): The size of a batch
        folder_paths (:opt:`list`, optional): A list of LibriSpeech folder paths (strings)
            Defaults to an empty list.
        label (:opt:`str`, optional): The label. Possible values:
            - 'voice_id': specifies the speaker
            - 'transcription': the text sequence 
            Defaults to transcription.

    Returns:
        generator : batch generator of mfcc features and labels

    Raises:
        Exception : If invalid label provided.
    """
    for folder_path in folder_paths:
        voice_txt_dict = voice_txt_dict_from_path(folder_path)

        features = []
        transcriptions = []
        ids = []

        for voice_id in voice_txt_dict:
            txt_files = voice_txt_dict[voice_id]
            for txt_file in txt_files:
                t, flac_files = transcriptions_and_flac(txt_file)

                for flac_file, transcription in zip(flac_files, t):
                    features.append(get_mfcc_from_file(flac_file)[:, 1:12]) # Save only cepstral coefficients 2-13
                    ids.append(voice_id)
                    transcriptions.append(transcription)
                
                    if len(features) >= batch_size:
                        print ("Reset!")
                        if label.lower() == 'transcription':
                            yield features, transcriptions
                        elif label.lower() == 'voice_id':
                            yield features, ids
                        else:
                            raise Exception('Invalid label')
                        features = []
                        transcriptions = []
                        ids = []

if __name__ == "__main__":
    folder_path = "../data/LibriSpeech"
    #data = get_data_from_path(folder_path)
    #save_data(data, '../data/LibriSpeech')
    
    batch = mfcc_batch_generator(1, [folder_path])
    feature, transcription = next(batch)

    print (feature, transcription)
    print (len(feature))
    print (len(transcription))
    print (feature[0].shape)


