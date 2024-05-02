'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''


import pandas as pd



def PdSeries(importanceArray, index):
    '''
    '''
    
    return pd.Series(importanceArray, index=index)


def PdDataFrame(data, columns):
    '''
    '''
    
    return pd.DataFrame(data=data, columns=columns)

def PdConcat(frames, axis):
    '''
    '''
    
    return pd.concat(frames, axis=axis)
    
    