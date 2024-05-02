'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''



   
def SetFigSize(xadd, yadd, sizeX, sizeY, subFigSizeX, subFigSizeY, cols, rows):
    '''  Set the figure size
    '''

    if sizeX == 0:

        figSizeX = subFigSizeX * rows + xadd

    else:

        figSizeX = subFigSizeX


    if sizeY == 0:

        figSizeY = subFigSizeY * cols + yadd

    else:

        figSizeY = subFigSizeY 
        
    return (figSizeX, figSizeY)


def GetPlotStyle(plotLayout):
    
    if plotLayout.linewidth: # linewidth == 0, no lines
        
        plotStyle = plotLayout.linestyle
    
    else:
        
        plotStyle = ''
        
    if plotLayout.pointsize:
        
        plotStyle += plotLayout.pointstyle
        
    return plotStyle
def GetAxisLabels(xlabels):
    '''
    '''
    
    xaxislabel = 'wavelength'
        
    yaxislabel = 'reflectance'
    
    if xlabels[0].startswith('pc-'):
        
        xaxislabel = 'principal components'
        
        yaxislabel = 'eigenvalues'
        
    elif xlabels[0].startswith('d'):
                
        yaxislabel = 'derivatives'
    
    return (xaxislabel, yaxislabel)