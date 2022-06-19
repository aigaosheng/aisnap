# -*- coding: utf-8 -*-
import sys
sys.path.append('...')
from frequent_patterns.apriori import apriori
from frequent_patterns.association_rules import association_rules
from numpy.testing import assert_array_equal
import pandas as pd
from frequent_patterns.preprocessing.transactionencoder import TransactionEncoder

MIN_EQUAL_SUPPORT_ITEM_ALLOWED = 2 #if item with support value equal more than the value, discard it
BOUND_MARK = '@#' 
def MineFact(noun_trans, min_support_v = 2, pattern_len = 1, topn = 5):    
    '''
    extract topn aspect with support frequent & ngram of aspect.
    pattern_len=1 better for aspect
    '''
    #extract all valid noun-phrases up to bigram
    valid_phrases = set()
    for s in noun_trans:
        if len(s) > 1:
            for w in zip(s[:-1], s[1:]):
                valid_phrases.add(BOUND_MARK.join(w))
        for s1 in s:
            valid_phrases.add(s1)

    te = TransactionEncoder()
    in_data = te.fit(noun_trans).transform(noun_trans, sparse=True)
    in_data_df = pd.SparseDataFrame(in_data, columns=te.columns_, default_fill_value=False)
    itemsets = apriori(in_data_df, min_support = float(min_support_v) / len(noun_trans), use_colnames=True, max_len = pattern_len)
    itemsets = sorted(zip(itemsets['support'], itemsets['itemsets']), key = lambda (s,v):(s,v), reverse = True)
    aspect = {}
    for i in range(pattern_len):
        aspect[i] = []
    for sp, wds in itemsets:
        aspect[len(wds)-1].append((sp,wds))
    pruned_aspect = {}
    for i in range(pattern_len):
        topn_used = topn
        if len(aspect[i]) > topn:
            threshold = aspect[i][topn-1][0]
            isprune = False
            eqct = 0
            for k in range(topn, len(aspect[i])):
                if abs(aspect[i][k][0] - threshold) < 1.0e-6:
                    eqct += 1
                    if eqct > MIN_EQUAL_SUPPORT_ITEM_ALLOWED:
                        isprune = True
                        break
            if isprune:
                #if euqal aspect more than threshold, backword pruning euqal aspect
                for k in range(topn - 1, -1, -1):
                    if abs(aspect[i][k][0] - threshold) > 1.0e-6:
                        topn_used = k + 1
                        break
            else:
                #if equal aspect is smaller, add them
                topn_used = topn + eqct
                
        pruned_aspect[i] = [wd for _, wd in aspect[i][:topn_used]]

    #pruning unigram aspect if it is in bigram
    '''for lv in range(pattern_len-2, -1, -1):
        new_aspect = []
        for wd in pruned_aspect[lv]:
            isprune = False
            for wd2 in pruned_aspect[lv+1]:
                if set(wd).issubset(set(wd2)):
                    isprune = True
                    break
            if not isprune:
                new_aspect.append(wd)
        pruned_aspect[lv] = new_aspect
    '''
    '''
    #pruning by string match
    ip = set()
    iprv = set()
    for k in range(len(pruned_aspect[0])):
        wd = pruned_aspect[0][k]
        for k2 in range(k+1, len(pruned_aspect[0])):
            wd2 = pruned_aspect[0][k2]
            ph = BOUND_MARK.join(wd+wd2)
            if ph in valid_phrases:
                ip.add((k,k2))
                iprv.add(k)
                iprv.add(k2)
            ph = BOUND_MARK.join(wd2+wd)
            if ph in valid_phrases:
                ip.add((k2, k))
                iprv.add(k)
                iprv.add(k2)
    
    ret_aspect = [pruned_aspect[0][i] for i in range(len(pruned_aspect[0])) if i not in iprv]
    for k, k2 in ip:
        ret_aspect.append(pruned_aspect[0][k] + pruned_aspect[0][k2])
    print(ret_aspect)
    '''
    ret_aspect = []
    for lv in range(pattern_len-1, -1, -1):
        ret_aspect += pruned_aspect[lv]
    #print(ret_aspect)
    #print(pruned_aspect)
    return ret_aspect
        
    '''rules = association_rules(res_df, metric="confidence", min_threshold=0.7)
    for a, b in zip(res_df, expect):
        assert_array_equal(a, b)
    for a in zip(rules['antecedants'],rules['consequents'],rules['support'],rules['confidence']):
        print(a)
    '''
    