# -*- coding: utf-8 *-* 
import os
import numpy as np
import re
from audio import get_mfcc_from_file 

def voice_txt_dict_from_path(folder_path):
    voice_txt_dict = {}

    for dir_name, sub_directories, file_names in os.walk(folder_path):
        if len(sub_directories) == 0: # this is a root 
            file_names = list(map(lambda x: os.path.join(dir_name, x), file_names))
            txt_file = list(filter(lambda x: x.endswith('txt'), file_names))[0]

            voice_id = os.path.basename(dir_name)

            if voice_id not in voice_txt_dict:
                voice_txt_dict[voice_id] = [txt_file]
            else:
                voice_txt_dict[voice_id].append(txt_file)

    return voice_txt_dict

def transcriptions_and_flac(txt_file_path):
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
                features.append(get_mfcc_from_file(flac_file))
                ids.append(voice_id)
    return features, transcriptions, ids

if __name__ == "__main__":
    #file_path = "../data/LibriSpeech/dev-clean/2277/149896/2277-149896.trans.txt"
    folder_path = "../data/LibriSpeech"
    features, transcriptions, ids = get_data_from_path(folder_path)

    np.save('../data/features.npy', np.dstack(features))
    np.save('../data/transcriptions.npy', np.array(transcriptions))
    np.save('../data/ids.npy', np.array(ids))
