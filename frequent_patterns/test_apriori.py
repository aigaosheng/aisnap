# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from apriori import apriori
from association_rules import association_rules
from numpy.testing import assert_array_equal
import pandas as pd
from preprocessing import TransactionEncoder

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

one_ary = np.array([[0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                    [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]])

cols = ['Apple', 'Corn', 'Dill', 'Eggs', 'Ice cream', 'Kidney Beans', 'Milk',
        'Nutmeg', 'Onion', 'Unicorn', 'Yogurt']

df = pd.DataFrame(one_ary, columns=cols)


def test_default():
    te = TransactionEncoder()
    oht_ary = te.fit(dataset).transform(dataset, sparse=True)
    sparse_df = pd.SparseDataFrame(oht_ary, columns=te.columns_, default_fill_value=False)
    res_df = apriori(sparse_df, use_colnames=True)
    expect = pd.DataFrame([[0.8, np.array([3]), 1],
                           [1.0, np.array([5]), 1],
                           [0.6, np.array([6]), 1],
                           [0.6, np.array([8]), 1],
                           [0.6, np.array([10]), 1],
                           [0.8, np.array([3, 5]), 2],
                           [0.6, np.array([3, 8]), 2],
                           [0.6, np.array([5, 6]), 2],
                           [0.6, np.array([5, 8]), 2],
                           [0.6, np.array([5, 10]), 2],
                           [0.6, np.array([3, 5, 8]), 3]],
                          columns=['support', 'itemsets', 'length'])
    rules = association_rules(res_df, metric="confidence", min_threshold=0.7)
    for a, b in zip(res_df, expect):
        assert_array_equal(a, b)
    for a in zip(rules['antecedants'],rules['consequents'],rules['support'],rules['confidence']):
        print(a)


def test_max_len():
    res_df1 = apriori(df)
    assert len(res_df1.iloc[-1, -1]) == 3

    res_df2 = apriori(df, max_len=2)
    assert len(res_df2.iloc[-1, -1]) == 2

test_default()