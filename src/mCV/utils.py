import numpy as np


def duplicate_if_scalar(seq):
    """

    Parameters
    ----------
    seq : {number, array-like}

    Returns
    -------

    """
    # seq = np.atleast_1d(seq)
    if np.size(seq) == 1:
        seq = np.ravel([seq, seq])
    if np.size(seq) != 2:
        raise ValueError('Input should be of size 1 or 2')
    return seq
