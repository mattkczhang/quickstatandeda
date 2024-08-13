import pytest
import numpy as np
import pandas as pd
import os
import shutil
import sys

sys.path.append('/Users/chonge/Documents/Document/quickstatandeda_pypi/quickstatandeda')
from quickstatandeda import edaFeatures

sample_data = pd.DataFrame({
    'id': [1,2,3,4,5,6],
    'num1': [10, 20, np.nan, 40, 50, 60],
    'num2': [0.5, 0.7, 0.9, 1.5, 2.0, np.nan],
    'num3': [87, 73, 65, 52, 44, 33],
    'num4': [3, 4, 3, 4, 5, 5],
    'date': [np.nan, '2011-02-15', '2012-11-03', '2013-12-31', '2014-05-30', '2016-01-21'],
    'datetime':['2010-01-03 06:09:30', '2011-02-15 11:29:30', '2012-11-03 15:50:21', '2013-12-31 03:19:11', '2014-05-30 01:55:49', '2016-01-21 07:28:18'],
    'cat1': ['a', 'a', 'a', 'b', 'b', 'b'],
    'cat2': ['up', np.nan, 'down', 'up', 'up', 'up'],
    'cat3': ['left', 'right', 'right', 'right', 'right', 'right'],
    'cat4': [np.nan, 'd', 'e', 'f', 'e', 'd'],
    'target': [3.22, 5.23, 7.44, 9.45, 11, np.nan]
})

num_features = ['num1','num2','num3','num4']
datetime_features = ['date','datetime']
cat_features = ['cat1','cat2','cat3','cat4']

def temp():
    edaFeatures(sample_data[['id', 'target', 'num1', 'cat1', 'cat2', 'cat3', 'cat4']], y='target',id='id')

def test_quickstatandeda():
    try:
        for n in [None, num_features[0], num_features]:
            for c in [None, cat_features[0], cat_features]:
                for d in [None, datetime_features[0], datetime_features]:
                    features = ['id','target']
                    if n is not None:
                        if type(n) == str:
                            features += [n]
                        else:
                            features += n
                    if c is not None:
                        if type(c) == str:
                            features += [c]
                        else:
                            features += c
                    if d is not None:
                        if type(d) == str:
                            features += [d]
                        else:
                            features += d
                    edaFeatures(sample_data[features], y='target', id='id')
                    if os.path.exists('EDA.html'): 
                        os.remove('EDA.html')
                    if os.path.exists('visuals'): 
                        shutil.rmtree('visuals')
                    print(str(features)+' -- PASS!')
    except ValueError as e:
        pytest.fail(f"my_function raised an exception: {e}")

