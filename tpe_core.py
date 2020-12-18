# -*- coding: utf-8 -*-

from lime.lime_text import LimeTextExplainer
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool
from sklearn.pipeline import make_pipeline
import sys
from time import time

from tpe_distance_matrix import distance_matrix
from tpe_all_lcs import lcs
import tpe_get_config

# Read configurations
MIN_WORD_LEN = int(tpe_get_config.getConfig("rule_parameter", "min_word_len"))
MIN_WORD_CONFIDENCE = float(tpe_get_config.getConfig("rule_parameter", "min_word_confidence"))
MAX_WORDS_NUM = int(tpe_get_config.getConfig("rule_parameter", "max_words_number"))

def get_instance_explained(df, instance_i, model, label_map, refer_label, xcol_name='text'):
    """Explain instance in a dataframe

    Test functions. Show keyword based explainations from lime.

    Args:
        df: dataframe to be explained.
        instance_i: index location of the specific instance.
        model: model that can classify instances.
        label_map: label text and value mappings.
        refer_label: the reference label for lime.
        xcol_name: column name for content to be explained in df.

    Returns:
        None. Just printable information.
    """

    # Locate instance content
    string = df[xcol_name].iloc[instance_i]
    row = df[xcol_name].iloc[[instance_i]]
    print('Model predict result %s' % model.predict(row))
    print(label_map)
    
    # Get explainations from LIME
    labels = list(label_map.values())
    explainer = LimeTextExplainer(class_names=labels)
    exp = explainer.explain_instance(string, model.predict_proba, num_features=10, num_samples=200, labels=labels)
    print(exp.as_list(label=label_map[refer_label]))

def get_lime_rules(df, model, label_map, refer_label, xcol_name='text'):
    """Generate lime based inference rules from the dataframe

    Texts or payloads are fed into this module. Use lime extract key 
    words in bulk.

    Args:
        df: dataframe to be explained.
        model: model that can classify instances.
        label_map: label text and value mappings.
        refer_label: the reference label for lime.
        xcol_name: column name for content to be explained in df.

    Returns:
        Inference rules based on list of keywords.
    """

    lime_rules = []
    df_lime = df
    
    # process each row
    labels = list(label_map.values())
    explainer = LimeTextExplainer(class_names=labels)
    for index, row in df_lime.iterrows():
        exp = explainer.explain_instance(row[xcol_name]
                                        , model.predict_proba
                                        , num_features=10
                                        , num_samples=200
                                        , labels=labels)
        
        # words that of length over 2 is confidential and the first n(n=6) words of higgest confidence are adopted
        exp_list = exp.as_list(label=label_map[refer_label])
        sorted_list = sorted(exp_list, key=lambda d:d[1], reverse=True)
        tmp_list = [t[0] for t in sorted_list if (t[1] > MIN_WORD_CONFIDENCE and len(t[0]) >= MIN_WORD_LEN)][:MAX_WORDS_NUM]
        # order strings for filtering and comparing
        if len(tmp_list) > 1:
            tmp_list.sort()
            lime_rules.append(tmp_list)

    # rules duplicate removal
    k = lime_rules
    k.sort()
    lime_rules = list(k for k,_ in itertools.groupby(k))

    return lime_rules

import itertools
from functools import partial
def get_lime_rules_parallel(df, func, model, label_map, refer_label, xcol_name='text', n_cores=20):
    """Generate lime based inference rules from the dataframe in parallel 

    Accelerate inference procedure with multiprocessing.

    Args:
        df: dataframe to be explained.
        func: main function for inference rule extraction.
        model: model that can classify instances.
        label_map: label text and value mappings.
        refer_label: the reference label for lime.
        xcol_name: column name for content to be explained in df.
        n_cores: number of cores to utilize.

    Returns:
        Inference rules based on list of keywords.
    """

    # split dataframe for pooling
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    
    parallel_result = pool.map(partial(func, model=model, label_map=label_map, refer_label=refer_label), df_split)
    lime_rules = [item for sublist in parallel_result for item in sublist]

    # rules duplicate removal
    k = lime_rules
    k.sort()
    lime_rules = list(k for k,_ in itertools.groupby(k))

    # resource release
    pool.close()
    pool.join()

    return lime_rules

def fuse_lime_and_scan_rules(lime_rules, scan_rules):
    """Concat inference and scan rules. 

    Make rules from lime and LCS into one list.

    Args:
        lime_rules: rule list from inference based method.
        scan_rules: rule list from LCS based method.

    Returns:
        dataframe with rule and corresponding rule class label.
    """

    # give priority to scan rules for hitting to reserve useful scan rules and improve effect of rule compression
    s1 = pd.Series(scan_rules + lime_rules)
    # 1 for scan rules and 2 for lime rules
    s2 = pd.Series([1]*len(scan_rules) + [2]*len(lime_rules))
    rule_df = pd.concat([s1, s2], axis=1)
    
    # reset index to start from 1
    rule_df.index = np.arange(1, len(rule_df) + 1)

    return rule_df


def get_scan_rules(df, xcol_name='text', content_direction='forward'):
    """Generate LCS based scan rules from the dataframe 

    Homogeneous texts or payloads are clustered into clusters and fed into
    LCS module to extract LCS subsets.

    Args:
        df: dataframe to be explained.
        xcol_name: column name for content to be explained in df.
        content_direction: cut out sequences from 'forward' or 'backward'

    Returns:
        Scan rules based on list of keywords.
    """

    # make a interception or completion of payload
    max_payload_len = 50
    payloads = df

    print('get_scan_rules')
    print('scan directoin: %s' % content_direction)
    if content_direction == 'forward':
        payloads[xcol_name] = payloads[xcol_name].apply(lambda x: x[:max_payload_len])
    elif content_direction == 'backward':
        max_payload_len = -1 * max_payload_len
        payloads[xcol_name] = payloads[xcol_name].apply(lambda x: x[max_payload_len:])

    sequences = list(payloads[xcol_name].items())

    # calculate distance matrix
    t0 = time()
    X = distance_matrix(sequences)
    print("matrix done in %0.3fs." % (time() - t0))
    
    # payload clustering
    print( 'Start clustering...')
    db = DBSCAN(eps=0.25, min_samples=10, metric='precomputed').fit(X)
    print( 'Finish clustering...')
    skl_labels = db.labels_
    
    # preprocess cluster labels
    for i in range(0, len(skl_labels)):
        if not skl_labels[i] == -1:
            skl_labels[i] += 1

    ss_df = pd.DataFrame(sequences)
    ss_df["label"] = skl_labels
    
    ss_df.columns = ['index', 'seq', 'label']
    se_df = ss_df[['seq', 'label']]
    
   # extract rules from each cluster
    scan_rules = []
    for label in se_df.label.unique():
        if label != -1:
            re = se_df.loc[ss_df['label'] == label][['seq']]
            target_array = re['seq'].tolist()

            # find seed sequence
            max_len = 0
            for l in target_array:
                if len(l) > max_len:
                    seed = l

            pattern_list = [seed]

            for l in target_array:
                pattern_new = []
                for pattern in pattern_list:
                    pattern_new.extend(lcs(pattern, l))
                pattern_list = pattern_new
            pattern_tup = (label, pattern_list)

            target_sequences_o = []
            for e in pattern_list:
                target_sequences_o.append("".join(e))
            if len(target_sequences_o) > 0:
                scan_rules.append(target_sequences_o)
    print(scan_rules)

    # rules duplicate removal
    k = scan_rules
    k.sort()
    scan_rules = list(k for k,_ in itertools.groupby(k))

    print('~~~~~~~~~~~~~scan rule~~~~~~~~~~~~~~~~')
    print('Dataframe len: %d' % payloads.shape[0])
    print('Before dupicated removal: %d' % len(k))
    print('After dupicated removal: %d' % len(scan_rules))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    return scan_rules


base = r'^{}'
expr = '(?=.*{})'
def string_2_rule(df_tobe_validate):
    """Convert keyword strings to regular expressions. 

    To validate the effectiveness of keyword rules, generate regular expressions.

    Args:
        df_tobe_validate: rule dataframe to be validated.

    Returns:
        rule dataframe to be validated with regular expressions strings.
    """

    for index, row in df_tobe_validate.iterrows():
        words = []
        for word in row['rule_strings']:
            tmp = []
            reg = ''
            for ch in word:
                # escape character conversion
                if ch in ['.', '^', '$', '*', '+', '?', '\\', '[', ']', '|', '{', '}', '(',')']:
                    tmp.append('\\')
                    tmp.append(ch)
                else:
                    tmp.append(ch)
            try:
                word_new = ''.join(tmp)
            except:
                sys.exit()
            words.append(word_new)
        regex = base.format(''.join(expr.format(w) for w in words))

        # append regex column content strings
        df_tobe_validate.loc[index, 'regex'] = regex
    return df_tobe_validate

def get_rules(df, model, label_map, refer_label, lime_flag=True, scan_flag=False, content_direction='forward', xcol_name='text', n_cores=20):
    """Generate rules for given dataframe with LCS and inference based methods. 

    Take dataframe and classification model as inputs, extract global rules for
    texts or payloads classification.

    Args:
        df: dataframe to be explained.
        model: model that can classify instances.
        label_map: label text and value mappings.
        refer_label: the reference label for lime.
        lime_flag: on-off flag for lime based inference rules.
        scan_flag: on-off flag for LCS based scan rules.
        content_direction: cut out sequences from 'forward' or 'backward'
        xcol_name: column name for content to be explained in df.
        n_cores: number of cores to utilize.

    Returns:
        Classification rules based on list of keywords.
    """

    if lime_flag == True:
        lime_rules = get_lime_rules_parallel(df, get_lime_rules, model, label_map, 'malicious')
    else:
        lime_rules = []
    
    if scan_flag == True:
        scan_rules = get_scan_rules(df, xcol_name='text', content_direction=content_direction)
    else:
        scan_rules = []

    result = fuse_lime_and_scan_rules(lime_rules, scan_rules)
    result.columns = ['rule_strings', 'rule_type']

    # get regular expressions.
    rules = string_2_rule(result.copy())
    return rules
