# -*- coding: utf-8 -*-

import numpy as np
from re import search
import pandas as pd
import tpe_get_config

MIN_SINGLE_RULE_ACCURACY = float(tpe_get_config.getConfig("rule_parameter", "min_single_rule_accuracy"))

def rule_deduplicate(rules):
    """Remove duplicate rules.

    Inference rules and scan rule are generated independently. Rules are keyword based and can
    be redundant from a sample matching point of view. To get rule set fit, conduct rule matching
    internally in rule set to get ride of redundant ones.

    Args:
        rules: rule dataframe generated before.

    Returns:
        Compressed rule dataframe.
    """

    rules['duplicate'] = 0
    print("before compress %d" % (rules.shape[0]))

    # traverse rules in rule set.
    for rule_index, rule in rules.iterrows():
        for rule_s_index, rule_s in rules.iterrows():
            if rule_index == rule_s_index:
                    continue 
            string = ''.join(rule_s['rule_strings'])
            if search(rule['regex'], string):
                if rules.loc[rule_s_index, 'duplicate'] > 0:
                    # only reserve match result that hit for the first time 
                    continue
                if rules.loc[rule_index, 'duplicate'] == rule_s_index:
                    # skip to rules that have been matchee
                    continue
                else:
                    rules.loc[rule_s_index, 'duplicate'] = rule_index

    # drop duplicate ones.
    result_rules = rules.loc[rules['duplicate'] == 0]
    print("after compress %d" % (result_rules.shape[0]))
    return result_rules



from multiprocessing import Pool
from re import search
from time import time
import sys

max_payload_len = 300
core_number = 20
sample_num = 50000
def match_func(payloads_i, rules):
    """Match text or payload samples to rules.

    Matching rules with payloads or texts, to further get rule validated.

    Args:
        payloads_i: dataframe with text column to be matched.
        rules: rule dataframe generated before.

    Returns:
        Payload dataframe with matched rule label.
    """

    payloads_i['text'] = payloads_i['text'].str.slice(0, max_payload_len)
    rule_df = rules.copy()

    payloads_i['match'] = 0
    payloads_i['rule_num'] = 0 # to approve rule index>0
    i = 0
    t0 = time()

    for p_index, payload in payloads_i.iterrows():
        for r_index, rule in rule_df.iterrows():
            if payloads_i.loc[p_index, 'match'] > 0:  
                # only reserve match result that hit for the first time 
                break

            if search(rule['regex'], payload['text']):
                payloads_i.loc[p_index, 'match'] = rule['rule_type']
                payloads_i.loc[p_index, 'rule_num'] = r_index         
    return payloads_i


def refine_rules_by_metrics(result_df):
    """Rule refinement according to matching metrics.

    Keep the rules that meet the accuracy needs.

    Args:
        result_df: dataframe with text column and rule matching labels.

    Returns:
        List of rules with low payload classification performance.
    """

    rule_remove_list = []
    for rule_index in list(result_df.rule_num.unique()):
        if rule_index == 0:
            # Only care about hiting part of rules, 0 means no rule hit
            continue
        
        metrics_for_rule = get_metrics(result_df[result_df.rule_num == rule_index])
        # define the lower bound for useful rules 
        if float(metrics_for_rule['acc']) < MIN_SINGLE_RULE_ACCURACY:
            rule_remove_list.append(rule_index)

    return rule_remove_list


from functools import partial
def parallelize_dataframe(df, func, rules, n_cores=core_number):
    """Parallel dataframe processing

    Accelerate dataframe processing with multiprocessing, for rule_validation

    Args:
        df: dataframe to be processed.
        func: main function for dataframe processing.
        rules: rules for sample matching.
        n_cores: number of cores to utilize.

    Returns:
        Result dataframe after processing (validation).
    """

    # accerlarate rule extraction
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)

    df_match = pd.concat(pool.map(partial(func, rules=rules), df_split))
    pool.close()
    pool.join()

    return df_match

def rule_validation(df, rules, final_flag=False, n_cores=core_number):
    """Validate the rule matching performance.

    With test dataset, test the classification performance globally in a loop manner.

    Args:
        df: dataframe to be validated.
        rules: rules for sample matching.
        final_flag: terminate the validation process loop.
        n_cores: number of cores to utilize.

    Returns:
        valid_df: dataframe with final validation results.
        refined_rules: final rule set after validation on test datset.
    """

    print("Rule matching start...")
    if final_flag == True:
        # If it is the final test, make duplicate removal first and guarantee the correspondance of hit results and rules index
        rules = rule_deduplicate(rules)
        refined_rules = rules
        valid_df = parallelize_dataframe(df, match_func, refined_rules, n_cores=n_cores)
    else:
        flag = 1
        ori_rules = rules

        # repeat the match process if there is rules of no use
        while flag == 1:
            flag = 0
            valid_df = parallelize_dataframe(df, match_func, ori_rules, n_cores=n_cores)

            # drop low acrruacy rate rules
            remove_list = refine_rules_by_metrics(valid_df)
            refined_rules = ori_rules.loc[~ori_rules.index.isin(remove_list)]
            if remove_list:
                flag = 1

                valid_df.loc[valid_df['rule_num'].isin(remove_list), ['match']] = 0
                valid_df.loc[valid_df['rule_num'].isin(remove_list), ['rule_num']] = 0
                ori_rules = refined_rules

        # validate duplicate removal of rules to improve later match efficiency 
        refined_rules = rule_deduplicate(refined_rules)

    print("Rule matching finished...")
    
    return valid_df, refined_rules 


def get_metrics(result_df):
    """Compute the classification performance.

    Accuracy, false positive rates, recall rates are computed.

    Args:
        result_df: dataframe with text column and rule matching labels.

    Returns:
        dict of metrics.
    """

    # Write own metrics for flexibility
    match_result_stat = result_df.groupby(['match', 'target'])['text'].count()
    matrix = match_result_stat
    
    mydict = {}
    for rule_type in range(0, 3, 1):
        for label in range(0, 2, 1):
            position = 'v' + str(rule_type) + str(label)
            mydict[position] = position
            try:
                mydict[position] = matrix[rule_type][label]
            except:
                mydict[position] = 0
    acc = 0
    fpr = 0
    rec_scan = 0
    rec_lime = 0
    rec = 0

    try:
        # 0 for malicious and 1 for normal
        acc = float("{0:.6f}".format((mydict['v01']+mydict['v10'] +mydict['v20'])/result_df.shape[0]))
        fpr = float("{0:.6f}".format((mydict['v11']+mydict['v21'])/(mydict['v11']+mydict['v21']+mydict['v10']+mydict['v20'])))
        rec = float("{0:.6f}".format((mydict['v10']+mydict['v20'])/(mydict['v10']+mydict['v20']+mydict['v00'])))
        rec_scan = float("{0:.6f}".format(mydict['v10']/(mydict['v10']+mydict['v20'])))
        rec_lime = float("{0:.6f}".format(mydict['v20']/(mydict['v10']+mydict['v20'])))
        metrics = dict([('acc', acc), ('fpr', fpr), ('rec', rec), ('rec_scan', rec_scan), ('rec_lime', rec_lime)])
    except:
        metrics = dict([('acc', acc), ('fpr', fpr), ('rec', rec), ('rec_scan', rec_scan), ('rec_lime', rec_lime)])
        
    return metrics


import matplotlib.pyplot as plt
import seaborn
def get_final_rules(match_result, rules_all):
    """Show mathcing results on test set and get hit rule dataframe.

    Args:
        match_result: rule matching result dataframe for .
        rules_all: all generated rules.

    Returns:
        dataframe of rule that actually hit by test samples.
    """

    rule_list = list(filter(lambda x: x > 0, list(match_result.rule_num.unique())))
    rule_list.sort()
    print('Matched rules total : %d' % (len(rule_list)))
    
    df_draw = match_result.loc[match_result.rule_num != 0].groupby(['rule_num'])['match'].count()
    df_draw.plot.bar(figsize=(15, 10), legend=True, fontsize=12)
    return rules_all.loc[rule_list]


from tpe_core import get_rules
def rule_matching_evaluation(df, model, seed_num, rein_num, eval_num, label_map, refer_label, lime_flag=True, scan_flag=False
                            , content_direction='forward', xcol_name='text', n_cores=20):
    """A integrated rule extraction, refinement and validation process.

    On the dataset, sample based methods are used. Seed rules are extracted and unmatched samples in 
    reinforcement samples are re-fed into extraction procedure. Validation are conducted in loops until
    certain condition is meet.

    Args:
        df: dataframe to be explained.
        model: model that can classify instances.
        seed_num: sample size for seed rule generation.
        rein_num: sample size for reinforcement procedure.
        eval_num: sample size for evaluation procedure.
        label_map: label text and value mappings.
        refer_label: the reference label for lime.
        lime_flag: on-off flag for lime based inference rules.
        scan_flag: on-off flag for LCS based scan rules.
        content_direction: cut out sequences from 'forward' or 'backward'
        xcol_name: column name for content to be explained in df.
        n_cores: number of cores to utilize.

    Returns:
        match_result: match result on evaluation test sets.
        rules_tobe_validate: final rules generated.
        matched_rules: rules hit by evaluation test samples.
    """

    # shuffle dataset
    df.sample(frac=1, random_state=1)

    # generate seed rules
    df_for_seed = df[df['target'] == label_map['malicious']].sample(seed_num, random_state=2)
    rules_seed = get_rules(df_for_seed, model, label_map, 'malicious', lime_flag=lime_flag, scan_flag=scan_flag, content_direction=content_direction, n_cores=n_cores)
    print(rules_seed)


    # reinforce rules
    max_iter_times = 2
    df_split = np.array_split(df, max_iter_times)

    rules_tobe_validate = rules_seed
    for i in range(0, max_iter_times):
        print('--------------------------------------------------------------------------------------------------------')
        print('--------------------------------------------------------------------------------------------------------')
        print('--------------------------------------------------------------------------------------------------------')
        print('Reinforce iteration loop %d'% (i+1))
        print('Seed rules number: %d' % rules_tobe_validate.shape[0])

        df_for_reinforce = df_split[i].sample(rein_num, random_state=3)
        match_result, rules_tobe_validate = rule_validation(df_for_reinforce, rules_tobe_validate, n_cores=n_cores)
    #     # make duplicate removal for each validation 
    #     rules_tobe_validate = rule_deduplicate(rules_tobe_validate)
        metrics = get_metrics(match_result)
        print(metrics)

        if float(metrics['acc']) > 0.98:
            print("Validation finished, metrics is fine.")
            break
        else:
            # Reinforcement the unrecognizable malicious flows according to validation results
            df_rein = match_result.loc[(match_result.match == 0) & (match_result.target == label_map['malicious'])][['text', 'target']]
            df_rein['text'] = df_rein['text'].astype(str)

            result_rein = get_rules(df_rein, model, label_map, 'malicious', lime_flag=lime_flag, scan_flag=scan_flag, content_direction=content_direction, n_cores=n_cores)
            result_final = pd.concat([rules_tobe_validate, result_rein])
            
            # index start from 1
            result_final.index = np.arange(1, len(result_final)+1)
            rules_tobe_validate = result_final
        
        print('New rein rules number: %d' % result_rein.shape[0])
        print('--------------------------------------------------------------------------------------------------------')
        print('--------------------------------------------------------------------------------------------------------')
        print('--------------------------------------------------------------------------------------------------------')


    df_for_final_eval = df.sample(seed_num, random_state=4)
    match_result, rules_tobe_validate = rule_validation(df_for_final_eval, rules_tobe_validate, final_flag=True, n_cores=n_cores)
    if rules_tobe_validate.shape[0] == 0:
        print("Rule extraction failed!!!!!")
        return 0, 0, 0
    else:
        print('The final results are:')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        matched_rules = get_final_rules(match_result, rules_tobe_validate)
        metrics = get_metrics(match_result)
        print(metrics)
        print(matched_rules)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("Final validation finished") 

    return match_result, rules_tobe_validate, matched_rules

