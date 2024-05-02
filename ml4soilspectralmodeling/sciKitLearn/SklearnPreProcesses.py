'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, normalize
from sklearn.decomposition import PCA


def SklearnNormalizeDf(df, normFunction, return_norm=False):
    '''
    '''
    
    return normalize(df, norm=normFunction, return_norm=return_norm) 


def SklearnStandardScalerFit(data, with_mean=True, with_std=True):
    '''
    '''
    return StandardScaler(with_mean=with_mean, with_std=with_std).fit(data)
    
def SklearnStandardScalerTransform(data, with_mean=True, with_std=True):
    '''
    '''
    return StandardScaler(with_mean=with_mean, with_std=with_std).transform(data)
    
    
def SklearnStandardScalerFitTransform(data, with_mean=True, with_std=True):
    '''
    '''
    
    return StandardScaler(with_mean=with_mean, with_std=with_std).fit_transform(data)