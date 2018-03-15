# -*- coding: utf-8 *-* 
"""Helper functions for processing data
"""
import numpy as np

alphabet_index_map = {'a' : 0, 'b': 1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 
                      'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 
                      'r':17, 's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25,
                      ' ':26, '0':27}
index_alphabet_map = {v: k for k, v in alphabet_index_map.items()}

def str_to_one_hot(strings, max_str_length):
    """One hot encode characters
       
    One hot encoder for characters. 
    Uses our assumed method of using only lower case
    ascii letters, spaces, and '0' to denote end of seq.
    '0' will translate to 0s.

    Args:
        chars (list of strings): 

    Returns:
        list of np.arrays: one hot encoded chars
    """
    idx_list = []
    for string in strings:
        idx_list.append([alphabet_index_map[ch] for ch in string])

    ohe = np.zeros((len(strings), max_str_length, len(alphabet_index_map)))

    for str_idx, ohe_idxs in enumerate(idx_list):
        for char_idx, ohe_idx in enumerate(ohe_idxs):
            ohe[str_idx][char_idx][ohe_idx] = 1
    return ohe 

def one_hot_to_str(ohe_lists):
    """Convert one hot encoded characters to string
    """
    strs = []
    for ohe_list in ohe_lists:
        char_list = []
        for ohe in ohe_list:
            ch = index_alphabet_map[int(np.nonzero(ohe)[0])]
            if ch != '0': # Don't include the 0 padding
                char_list.append(ch)
        strs.append("".join(char_list))
    return strs

if __name__ == "__main__":
    strs = ['abc', 'def']

    ohe = str_to_one_hot(strs, 3)
    print (ohe)
    print (one_hot_to_str(ohe))

