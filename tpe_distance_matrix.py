# -*- coding: utf-8 -*-

import sys
import os
from numpy import *

def PairwiseSimilarity(sequences):
    """Compute hex sequence pair-wise similarity.

    Take sequences as inputs and generate sequence distance matirx
    based on pair-wise similarity.

    Args:
        sequences: list of hex sequences.

    Returns:
        distance matrix.
    """

    seq = sequences
    N = len(seq)

    # Define distance matrix
    d_matrix = zeros((N, N), float64)

    for i in range(len(seq)):
        for j in range(len(seq)):
            d_matrix[i][j] = -1

    # Similarity matrix
    s_matrix = zeros((N, N), float64)

    for i in range(N):
        for j in range(N):
            s_matrix[i][j] = -1

    # Find pairs
    for i in range(N):
        for j in range(N):

            if s_matrix[i][j] >= 0:
                continue

            seq1 = seq[i][1]
            seq2 = seq[j][1]
            minlen = min(len(seq1), len(seq2))
    
            len1 = len2 = sims = 0
            for x in range(minlen):
                if seq1[x] != 256:
                    len1 += 1.0

                    if seq1[x] == seq2[x]:
                        sims += 1.0

                if seq2[x] != 256:
                    len2 += 1.0

            maxlen = max(len1, len2)
            s_matrix[i][j] = sims / maxlen

    # Get distance matrix
    for i in range(N):
        for j in range(N):
            d_matrix[i][j] = s_matrix[i][i] - s_matrix[i][j]
    
    return d_matrix

def distance_matrix(sequences):
    """Compute hex sequence similarity.

    Take sequences as inputs and generate sequence distance matirx for 
    further clustering.

    Args:
        sequences: list of hex sequences.

    Returns:
        distance matrix.
    """

    if len(sequences) == 0:
        print("FATAL: No sequences found")
        sys.exit(-1)
    else:
        print("Found %d sequences" % len(sequences))
    
    print("Creating distance matrix start.")
    dmx = PairwiseSimilarity(sequences)
    print("Distance matrix complete.")
    return dmx

