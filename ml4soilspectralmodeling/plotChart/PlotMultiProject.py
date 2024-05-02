'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

import matplotlib.pyplot as plt

def MultiPlot(figRows, figCols, figSizeX, figSizeY):

    return plt.subplots(figRows, figCols, figsize=(figSizeX, figSizeY))

def ClosePlot(FPN):
    '''
    '''
    
    plt.close(fig=FPN)