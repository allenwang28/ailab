
def one_hot_encode_strings(strings):
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
    ohe = np.zeros((27,len(strings)))


