import pandas as pd
import numpy as np

def ifPivotable(df, index, column, value):
    if df.groupby([index,column])[value].count().max() == 1:
        return True
    else:
        return False


