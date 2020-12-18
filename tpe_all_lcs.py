# -*- coding: utf-8 -*-

import tpe_get_config

# Read minimum length configuration of `word'
MIN_WORD_LEN = int(tpe_get_config.getConfig("rule_parameter", "min_word_len"))

def lcs(s1_, s2_):
    """Extract LCS subsets from two strings.

    Take strings as inputs and generate LCS substrings recursively.

    Args:
        s1_, s2_: input strings.

    Returns:
        List of longgest substrings.
    """

    s1_len = len(s1_)
    s2_len = len(s2_)

    max_len = max(s1_len, s2_len)
    
    m = [[0 for x in range(1 + max_len)] for y in range(1 + max_len)] 

    longest = 0
    x_longest = -1
    y_longest = -1

    for x in range(1, 1 + s1_len):
        for y in range(1, 1 + s2_len):
            if s1_[x - 1] == s2_[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
                    y_longest = y
            else:
                m[x][y] = 0

    l_list = []

    # recursive termination condition
    if (x_longest == -1) or (y_longest == -1):
        return l_list

    # search for all common-substrings recursively: 
    # search for the longest first, then remove it and search for the longest in the front and back part separately
    # search recursively and document the longest of front part
    l2_backword = []
    if (x_longest - longest > 0) and (y_longest - longest > 0):
        l2_backword = lcs(s1_[0:x_longest - longest], s2_[0:y_longest - longest])

    if len(l2_backword) > 0:
        l_list.extend(l2_backword)

    # document the longest of the front part
    l1 = s1_[x_longest - longest: x_longest]
    if len(l1) >= MIN_WORD_LEN:
        l_list.append(l1)

    # search recursively and document the longest of the back part
    l2_forward = []
    if (x_longest < s1_len - 1) and (y_longest < s2_len - 1):
        l2_forward = lcs(s1_[x_longest:s1_len], s2_[y_longest:s2_len])

    if len(l2_forward) > 0:
        l_list.extend(l2_forward)

    return l_list