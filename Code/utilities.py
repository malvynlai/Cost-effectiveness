import numpy as np
import pandas as pd
from numba import njit, prange
from datetime import datetime

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList


from scipy import stats
import compressedInputsRead as readInputs
import MCMC_improved_cleanv3_ver as MC
import sys
import time
import Visualizations.vizFinal as viz
import Visualizations.verFinal as viz2

np.random.seed(42)

# Set display options for NumPy
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Set display options for pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



import numpy as np

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

def average_age_death_after_state_numba(target_states_nb, ages, states):
    total = 0.0
    count = 0
    death  = first_hit_indices(states, [5])
   
    A = death + ages 
    condition = first_hit_indices(states, target_states_nb)
    B = (condition != -1)

    C = A[B].sum()/len(A[B]) +18
    return C
    


# 2) Helper: convert Python list → numba.typed.List[int32]
def to_numba_list(py_list):
    nb = NumbaList()
    for x in py_list:
        nb.append(np.int32(x))
    return nb

import numpy as np

def first_hit_mask(M, targets):
    """
    For each row in M, find the first column where M is any of `targets`.
    Return an array shape (n_rows, n_cols) with 1 at that position and 0 elsewhere.
    Rows with no occurrence of any target become all-zero.
    
    Parameters
    ----------
    M : array-like, shape (n_rows, n_cols)
    targets : scalar or sequence
        Value or list of values to look for.
    
    Returns
    -------
    out : ndarray of int8, same shape as M
    """
    M = np.asarray(M)
    n_rows, n_cols = M.shape

    # boolean mask of where M is in targets
    hit = np.isin(M, targets)          # shape (n_rows, n_cols), dtype=bool

    # first True index per row; rows with no hit get 0
    first_idx = hit.argmax(axis=1)     # shape (n_rows,), dtype=int

    # which rows actually had at least one hit
    has_hit = hit.any(axis=1)          # shape (n_rows,), dtype=bool

    # build output array of zeros
    out = np.zeros_like(M, dtype=np.int8)

    # scatter 1s only into rows that had a hit
    rows = np.nonzero(has_hit)[0]
    cols = first_idx[has_hit]
    out[rows, cols] = 1

    return out



def first_hit_indices(M, targets):
    """
    For each row i in M, return the column index of the first element
    that is in `targets`.  If no such element, return -1 for that row.
    
    Parameters
    ----------
    M : array-like, shape (n_rows, n_cols)
    targets : scalar or sequence
        The value(s) to look for.
    
    Returns
    -------
    idx : ndarray of shape (n_rows,), dtype=int
        idx[i] is the column of the first hit in row i, or -1 if none.
    """
    M = np.asarray(M)
    hit = np.isin(M, targets)      # boolean array, True where M[i,j] in targets
    
    # argmax gives first True per row, but returns 0 for rows with no True
    first_idx = hit.argmax(axis=1)
    
    # detect rows that actually had a hit
    has_hit = hit.any(axis=1)
    
    # set “no hit” rows to -1
    first_idx[~has_hit] = -1
    
    return first_idx


def average_age_reaching_state_numba(target_states_nb, ages, states):
    total = 0.0
    count = 0
    mask  = first_hit_indices(states, target_states_nb)
    A = mask + ages 
    mask2 = (mask != -1)
    C = A[mask2].sum()/len(A[mask2])
    return C + 18

# 2) Helper: convert a Python list of ints → numba.typed.List[int32]
def to_numba_list(py_list):
    nb = NumbaList()
    for x in py_list:
        nb.append(np.int32(x))
    return nb

def rewardsTotalAfter(statesIndices, s, rewards):
    n_rows, n_cols = s.shape
    row_sums = []  # will hold the sum of rewards *after* first event, per row

    for i in range(n_rows):
        row = s[i]
        # 1) find first occurrence of any interest‐state
        hits = np.where(np.isin(row, statesIndices))[0]

        if hits.size == 0:
            continue         # no event in this row → skip

        start = hits[0]      # the first time‐index of an event
        # 2) sum rewards from that point to end
        
        row_sum = rewards[i, start : ].sum()
        row_sums.append(row_sum)

    if not row_sums:
        return 0.0          # no rows had the event

    # 3) average across rows
    return float(np.mean(row_sums))


# Takes valid state indices for this state, rewards is discounted matrix
def rewardsTotal(statesIndices, s, rewards):
	mask  = np.isin(s, statesIndices)
	count   = np.sum(np.any(mask, axis=1)) 
	meanRewards =  np.sum(rewards[mask])
	return meanRewards

    # Takes valid state indices for this state, rewards is discounted matrix
def rewardsSum(statesIndices, s, rewards):
    mask  = np.isin(s, statesIndices)
    meanRewards =  np.sum(rewards[mask])
    return meanRewards

