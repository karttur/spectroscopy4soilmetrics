'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

import matplotlib.pyplot as plt

import numpy as np

from sklearn.inspection import DecisionBoundaryDisplay

from support import DeepCopy, PdDataFrame

from plotChart import GetPlotStyle, GetAxisLabels

from sciKitLearn import SklearnNormalizeDf

def SetcolorRamp(n, colorRamp):
        ''' Slice predefined colormap to discrete colors for each band
        '''

        # Set colormap to use for plotting
        cmap = plt.get_cmap(colorRamp)

        # Segmenting colormap to the number of bands
        slicedCM = cmap(np.linspace(0, 1, n))
        
        return (slicedCM)
     
def PlotFilterExtract(plotLayout, filterTxt, originalDF, filterDF, plotFPN):
    """
    """
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.scatterCorrection.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(originalDF.index)-1)/maxSpectra )
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['original'] = {}
    
    subplotsD['filtered'] = {}
    
    subplotsD['original'] = {'label': 'Original data',
                                      'DF' : originalDF}
    
    subplotsD['filtered'] = { 'label': 'Filtered/extracted data',
                                      'DF' : filterDF}
        
    #fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
    filterfig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True, sharey=True )

    n = int(len(originalDF.index)/plotskipStep)+1
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    # Extract the columns bands) as floating wavelenhts   
    #columnsX = [float(item) for item in self.spectraDF.columns]
        
    origx = [float(i) for i in originalDF.columns]
    
    filterx =  [float(i) for i in filterDF.columns]
        
    # Get the plot style
    plotStyle =  GetPlotStyle(plotLayout)
                         
    for c, key in enumerate(subplotsD):
        
        if c == 1:
                
            ax[c].set(xlabel='wavelength')
            
            if plotLayout.filterExtraction.annotate.filtered:
                
                if plotLayout.filterExtraction.annotate.filtered == 'auto':
                 
                    txtStr = 'Filtered spectra\n  %s\n  %s total bands\n  showing every %s band' %(filterTxt, len(filterx),plotskipStep)
                    
                else:
                    
                    txtStr = plotLayout.filterExtraction.annotate.filtered
            
                ax[c].annotate(txtStr,
                           (plotLayout.filterExtraction.annotate.x,
                            plotLayout.filterExtraction.annotate.y),
                           xycoords = 'axes fraction' )
        else:
            
            if plotLayout.filterExtraction.annotate.original:
                
                if plotLayout.filterExtraction.annotate.original == 'auto':
                   
                    txtStr = 'Original spectra\n  %s total bands\n  showing every %s band' %(len(origx),plotskipStep) 
                
                else:
                    
                    txtStr = plotLayout.filterExtraction.annotate.original
  
                ax[c].annotate(txtStr,
                           (plotLayout.filterExtraction.annotate.x,
                            plotLayout.filterExtraction.annotate.y),
                           xycoords = 'axes fraction' )
                      
        ax[c].set(ylabel=plotLayout.filterExtraction.ylabels[c])
                                        
        # Loop over the spectra
        i = -1
        
        n = 0
            
        for index, row in subplotsD[key]['DF'].iterrows():
                
            i += 1
            
            if i % plotskipStep == 0:
                
                if c == 0:
                    
                    ax[c].plot(origx, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                else:
                    
                    ax[c].plot(filterx, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
                    
                n += 1
                
    # Set supTitle
    if plotLayout.filterExtraction.supTitle:
        
        if plotLayout.filterExtraction.supTitle == 'auto':
            
            filterfig.suptitle('Scatter Correction')
        
        else:
    
            filterfig.suptitle(plotLayout.filterExtraction.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        filterfig.tight_layout() 
        
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        filterfig.savefig(plotFPN)   # save the figure to file
           
    plt.close(filterfig)
                        
def PlotScatterCorr(plotLayout, plotFPN, corrTxtL, columns,
         trainOriginalDF, testOriginalDF,  
                    trainCorr1DF, testCorr1DF, trainCorr2DF=None, testCorr2DF=None):
    """
    """
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.scatterCorrection.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(trainOriginalDF.index)-1)/maxSpectra )
    
    ttratio = plotskipStep / ceil( (len(testOriginalDF.index)-1)/maxSpectra)
    
    annotateStrD = {0:'', 1:'', 2:'',}
    
    if plotLayout.scatterCorrection.annotate.input:          
        if plotLayout.scatterCorrection.annotate.input == 'auto':
            annotateStrD[0] = 'Input spectra\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[0] = plotLayout.scatterCorrection.annotate.input
            
    if plotLayout.scatterCorrection.annotate.firstcorrect:          
        if plotLayout.scatterCorrection.annotate.firstcorrect == 'auto':
            annotateStrD[1] = 'After %s correction\n  showing every %s band' %(corrTxtL[0], plotskipStep)
        else:
            annotateStrD[1] = plotLayout.scatterCorrection.annotate.firstcorrect
    
    if len(corrTxtL) > 1 and plotLayout.scatterCorrection.annotate.secondcorrect:          
        if plotLayout.scatterCorrection.annotate.secondcorrect == 'auto':
            annotateStrD[2] = 'After %s correction\n  showing every %s band' %(corrTxtL[1], plotskipStep)
        else:
            annotateStrD[2] = plotLayout.scatterCorrection.annotate.secondcorrect        
            
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['original'] = {}
        
    subplotsD['original']['train'] = {'label': 'Training data (original)',
                                      'DF' : trainOriginalDF}
    
    subplotsD['original']['test'] = { 'label': 'Test data (original)',
                                      'DF' : testOriginalDF}
        
        
    
    
    if len(corrTxtL) == 1:
        
        rmax = 1

        #fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
        scatplotfig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
        subplotsD['corrfinal'] = {}
        
        subplotsD['corrfinal']['train'] = { 'label': 'Scatter correction: %s' %(corrTxtL[0]),
                                       'DF' : trainCorr1DF}
    
        subplotsD['corrfinal']['test'] = { 'label': 'Scatter correction: %s' %(corrTxtL[0]),
                                    'DF' : testCorr1DF}

    else:

        rmax = 2
        
        #fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
        scatplotfig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
        # Intermed must be declared before final
        subplotsD['corrintermed'] = {}
        
        subplotsD['corrfinal'] = {}
                
        subplotsD['corrintermed']['train'] = { 'column': 0, 'row':1,
                                    'label': 'Scatter correction: %s' %(corrTxtL[0]),
                                    'DF' : trainCorr2DF}
    
        subplotsD['corrintermed']['test'] = { 'column': 1, 'row':1,
                                    'label': 'Scatter correction: %s' %(corrTxtL[0]),
                                    'DF' : testCorr2DF}
        
        subplotsD['corrfinal']['train'] = { 'column': 0, 'row':2,
                                    'label': 'Scatter correction: %s' %(corrTxtL[1]),
                                    'DF' : trainCorr1DF}
    
        subplotsD['corrfinal']['test'] = { 'column': 1, 'row':2,
                                    'label': 'scatter correct. %s' %(corrTxtL[1]),
                                    'DF' : testCorr1DF}
   
    n = int(len(trainOriginalDF.index)/plotskipStep)+1
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    x_spectra_integers = [int(i) for i in columns]
    
    # Get the plot style
    plotStyle =  GetPlotStyle(plotLayout)
    
    for r, key in enumerate(subplotsD):
        
        if r == rmax:
                
                for c in range(len(subplotsD[key])):
                
                    ax[r][c].set(xlabel='wavelength')
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            ax[r][c].annotate(annotateStrD[r],
                           (plotLayout.scatterCorrection.annotate.x,
                            plotLayout.scatterCorrection.annotate.y),
                           xycoords = 'axes fraction' )
            
            if c == 0:
                  
                ax[r][c].set(ylabel=plotLayout.scatterCorrection.ylabels[r])
                                        
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:
                        #ax[r][c].plot(x_spectra_integers, row, color=slicedCM[n])
                        
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                        
                    else:
                        
                        m = ceil(n*ttratio)
                        
                        #ax[r][c].plot(x_spectra_integers, row, color=slicedCM[m])
                        
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                                
                    n += 1
                
    # Set supTitle
    if plotLayout.scatterCorrection.supTitle:
        
        if plotLayout.scatterCorrection.supTitle == 'auto':
            
            scatplotfig.suptitle('Scatter Correction')
        
        else:
    
            scatplotfig.suptitle(plotLayout.scatterCorrection.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        scatplotfig.tight_layout()
            
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        scatplotfig.savefig(plotFPN)   # save the figure to file
    
        #infostr = 'Plots of scatter correction saved as:\n    %s' %(plotFPN)
        
        #print(infostr)
         
    plt.close(scatplotfig)

def PlotDerivatives(X_train, X_test, X_train_derivative, X_test_derivative, 
                    columns, dColumns, plotLayout, plotFPN):
    """
    """
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.derivative.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(X_train.index)-1)/maxSpectra )
    
    # ttration = trian-test ratio - only for adjusting colorramp
    ttratio = plotskipStep / ceil( (len(X_test.index)-1)/maxSpectra)
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['input'] = {}
    
    subplotsD['derivative'] = {}
    
    subplotsD['input']['train'] = {'label': 'Training data (input)',
                                      'DF' : X_train}
    
    subplotsD['input']['test'] = { 'label': 'Test data (input)',
                                      'DF' : X_test}
    
    subplotsD['derivative']['train'] = {'label': 'Derivative',
                                      'DF' : X_train_derivative}
    
    subplotsD['derivative']['test'] = { 'label': 'Derivatives',
                                      'DF' : X_test_derivative}
  
    #fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
    derivativeplotfig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
    n = int(len(X_train.index)/plotskipStep)+1
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    annotateStrD = {0:'', 1:'', 2:'',}
    
    if plotLayout.derivative.annotate.input:          
        if plotLayout.derivative.annotate.input == 'auto':
            annotateStrD[0] = 'Input spectra\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[0] = plotLayout.derivative.annotate.input
            
    if plotLayout.derivative.annotate.derivative:          
        if plotLayout.derivative.annotate.derivative == 'auto':
            annotateStrD[1] = 'Derivatives\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[1] = plotLayout.derivative.annotate.derivative
    
    # Get the plot style
    plotStyle =  GetPlotStyle(plotLayout)
    
    xD = {}
    
    xD[0] = list(columns.values())
    
    xD[1] = list(dColumns.values())
    
    for r, key in enumerate(subplotsD):
        
        if r == 1:
                
                for c in range(len(subplotsD[key])):
                
                    ax[r][c].set(xlabel='wavelength')
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            ax[r][c].annotate(annotateStrD[r],
                           (plotLayout.scatterCorrection.annotate.x,
                            plotLayout.scatterCorrection.annotate.y),
                           xycoords = 'axes fraction' )
            
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            if c == 0:
                  
                ax[r][c].set(ylabel=plotLayout.derivative.ylabels[r])
                                        
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:

                        ax[r][c].plot(xD[r], row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
  
                    else:
                        
                        m = ceil(n*ttratio)
                                               
                        ax[r][c].plot(xD[r], row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
            
                    n += 1
                
    # Set supTitle
    if plotLayout.derivative.supTitle:
        
        if plotLayout.derivative.supTitle == 'auto':
            
            derivativeplotfig.suptitle('Derivative')
        
        else:
    
            derivativeplotfig.suptitle(plotLayout.derivative.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        derivativeplotfig.tight_layout()
                        
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        derivativeplotfig.savefig(plotFPN)   # save the figure to file
    
        #infostr = 'Plots of standardisation saved as:\n    %s' %(plotFPN)
        
        #print(infostr)
         
    plt.close(derivativeplotfig)
    
def PlotPCA(plotLayout, plotFPN, pcaTxt, columnsD,
         trainInputDF, testInputDF, trainPCADF, testPCADF):
    """ Combine with standatdisation and derivation etc
    """
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.pca.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(trainInputDF.index)-1)/maxSpectra )
    
    # ttration = trian-test ratio - only for adjusting colorramp
    ttratio = plotskipStep / ceil( (len(testInputDF.index)-1)/maxSpectra)
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['original'] = {}
    
    subplotsD['standardised'] = {}
    
    subplotsD['original']['train'] = {'label': 'Training data (input)',
                                      'DF' : trainInputDF}
    
    subplotsD['original']['test'] = { 'label': 'Test data (input)',
                                      'DF' : testInputDF}
    
    subplotsD['standardised']['train'] = {'label': 'Principal components',
                                      'DF' : trainPCADF}
    
    subplotsD['standardised']['test'] = { 'label': 'Principal components',
                                      'DF' : testPCADF}
  
    #fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
    pcaplotfig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex='row', sharey='row' )
        
    n = int(len(trainInputDF.index)/plotskipStep)+1
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    
    
    annotateStrD = {0:'', 1:'', 2:'',}
    
    if plotLayout.pca.annotate.input:          
        if plotLayout.pca.annotate.input == 'auto':
            annotateStrD[0] = 'Input spectra\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[0] = plotLayout.pca.annotate.input
            
    if plotLayout.pca.annotate.output:          
        if plotLayout.pca.annotate.output == 'auto':
            annotateStrD[1] = 'Eigen vectors\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[1] = plotLayout.pca.annotate.output
    
    # Get the plot style
    plotStyle =  GetPlotStyle(plotLayout)

    for r, key in enumerate(subplotsD):
        
        if r == 0:
                
                for c in range(len(subplotsD[key])):
                
                    ax[r][c].set(xlabel='wavelength')
                    
                x_spectra_integers = list(columnsD.values())
                    
        if r == 1:
                
            for c in range(len(subplotsD[key])):
                
                ax[r][c].set(xlabel='component')
                
                x_spectra_integers = np.arange(0,len(trainPCADF.columns))
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            ax[r][c].annotate(annotateStrD[r],
                           (plotLayout.scatterCorrection.annotate.x,
                            plotLayout.scatterCorrection.annotate.y),
                           xycoords = 'axes fraction' )
            
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            if c == 0:
                  
                ax[r][c].set(ylabel=plotLayout.pca.ylabels[r])
                                        
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:
                        
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
   
                    else:
                        
                        m = ceil(n*ttratio)
                                                
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
           
                    n += 1
                
    # Set supTitle
    if plotLayout.pca.supTitle:
        
        if plotLayout.pca.supTitle == 'auto':
            
            pcaplotfig.suptitle('Principal Component Analysis (PCA)')
        
        else:
    
            pcaplotfig.suptitle(plotLayout.pca.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        pcaplotfig.tight_layout()
                
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        pcaplotfig.savefig(plotFPN)  
         
    plt.close(pcaplotfig)
    
def PlotStandardisation(plotLayout, plotFPN, standardTxt, columns,
         trainOriginalDF, testOriginalDF, trainStandarisedDF, testStandarisedDF):
    """
    """
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.standardisation.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(trainOriginalDF.index)-1)/maxSpectra )
    
    # ttration = trian-test ratio - only for adjusting colorramp
    ttratio = plotskipStep / ceil( (len(testOriginalDF.index)-1)/maxSpectra)
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['original'] = {}
    
    subplotsD['standardised'] = {}
    
    subplotsD['original']['train'] = {'label': 'Training data (input)',
                                      'DF' : trainOriginalDF}
    
    subplotsD['original']['test'] = { 'label': 'Test data (input)',
                                      'DF' : testOriginalDF}
    
    subplotsD['standardised']['train'] = {'label': 'Standardisation: %s' %(standardTxt),
                                      'DF' : trainStandarisedDF}
    
    subplotsD['standardised']['test'] = { 'label': 'Standardisation: %s' %(standardTxt),
                                      'DF' : testStandarisedDF}
  
    #fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
    standardplotfig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
    n = int(len(trainOriginalDF.index)/plotskipStep)+1
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    x_spectra_integers = [int(i) for i in columns]
    
    annotateStrD = {0:'', 1:'', 2:'',}
    
    if plotLayout.standardisation.annotate.input:          
        if plotLayout.standardisation.annotate.input == 'auto':
            annotateStrD[0] = 'Input spectra\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[0] = plotLayout.standardisation.annotate.input
            
    if plotLayout.standardisation.annotate.standard:          
        if plotLayout.standardisation.annotate.standard == 'auto':
            annotateStrD[1] = 'After %s\n  showing every %s band' %(standardTxt, plotskipStep)
        else:
            annotateStrD[1] = plotLayout.standardisation.annotate.standard
    
    # Get the plot style
    plotStyle =  GetPlotStyle(plotLayout)

    for r, key in enumerate(subplotsD):
        
        if r == 1:
                
                for c in range(len(subplotsD[key])):
                
                    ax[r][c].set(xlabel='wavelength')
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            ax[r][c].annotate(annotateStrD[r],
                           (plotLayout.scatterCorrection.annotate.x,
                            plotLayout.scatterCorrection.annotate.y),
                           xycoords = 'axes fraction' )
            
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            if c == 0:
                  
                ax[r][c].set(ylabel=plotLayout.scatterCorrection.ylabels[r])
                                        
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:
                        #ax[r][c].plot(x_spectra_integers, row, color=slicedCM[n])
                        
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                        
                    else:
                        
                        m = ceil(n*ttratio)
                        
                        #ax[r][c].plot(x_spectra_integers, row, color=slicedCM[m])
                        
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                                
                    n += 1
                
    # Set supTitle
    if plotLayout.standardisation.supTitle:
        
        if plotLayout.standardisation.supTitle == 'auto':
            
            standardplotfig.suptitle('Standardisation')
        
        else:
    
            standardplotfig.suptitle(plotLayout.standardisation.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        standardplotfig.tight_layout()
                
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        standardplotfig.savefig(plotFPN)   # save the figure to file
    
        #infostr = 'Plots of standardisation saved as:\n    %s' %(plotFPN)
        
        #print(infostr)
         
    plt.close(standardplotfig)

def PlotOutlierDetect(plotLayout, plotFPN,
         XtrainInliers, XtrainOutliers, XtestInliers, XtestOutliers,  
         postTrainSamples, nTrainOutliers, postTestSamples, nTestOutliers,
         detector, columnsX, targetFeature,  outlierFit, X):
    """
    """

    outliersfig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True, sharey=True )
                
    inScatSymb =  plotLayout.outlierDetection.scatter.inliers
    
    outScatSymb =  plotLayout.outlierDetection.scatter.outliers
    
    if plotLayout.outlierDetection.xlabel == 'auto':
                                
        xlabel = '%s (covariate)' %(columnsX[0])
        
    else:
            
        xlabel = plotLayout.outlierDetection.xlabel
    
    if plotLayout.outlierDetection.ylabel == 'auto':
        
        if columnsX[1] == 'target':
            
            ylabel = '%s (target feature)' %(targetFeature)
        
        else:
            
            ylabel = '%s (covariate)' %(columnsX[1])
            
    else:
        
        ylabel = plotLayout.outlierDetection.xlabel
            
    for i in range (2):
        
        if i == 0:
            inliersX =  XtrainInliers[columnsX[0]]
        
            inliersY =  XtrainInliers[columnsX[1]]
            
            outliersX =  XtrainOutliers[columnsX[0]]
            
            outliersY =  XtrainOutliers[columnsX[1]]
            
            if plotLayout.outlierDetection.annotate.apply:
    
                if plotLayout.outlierDetection.annotate.train == 'auto':
                 
                    txtStr = 'Inlier/outlier samples\n  in: %s, out: %s\n  method: %s' %(postTrainSamples, nTrainOutliers,
                                                detector)
                    
                else:
                    
                    txtStr = plotLayout.outlierDetection.annotate.train
                
            title = 'Outlier detection training dataset'
            
        else:
            
            inliersX =  XtestInliers[columnsX[0]]
        
            inliersY =  XtestInliers[columnsX[1]]
            
            outliersX =  XtestOutliers[columnsX[0]]
            
            outliersY =  XtestOutliers[columnsX[1]]
            
            if plotLayout.outlierDetection.annotate.apply:
    
                if plotLayout.outlierDetection.annotate.test == 'auto':
                 
                    txtStr = 'Inlier/outlier samples\n  in: %s, out: %s\n  method: %s' %(postTestSamples, nTestOutliers,
                                                detector)
                    
                else:
                    
                    txtStr = plotLayout.outlierDetection.annotate.train

            title = 'Outlier detection test dataset'
        
        DecisionBoundaryDisplay.from_estimator(
            outlierFit,
            X,
            response_method="decision_function",
            plot_method="contour",
            colors=plotLayout.outlierDetection.isolines.color,
            levels=[0],
            ax=ax[i],
        )
                         
        ax[i].scatter(inliersX, inliersY, color=inScatSymb.color, alpha=inScatSymb.alpha, s=inScatSymb.size)
        ax[i].scatter(outliersX, outliersY, color=outScatSymb.color, alpha=outScatSymb.alpha, s=outScatSymb.size)
        
        ax[i].set(
            xlabel= xlabel,
            ylabel= ylabel,
            title= title,
        )
        
        ax[i].annotate(txtStr,
               (plotLayout.outlierDetection.annotate.x,
                plotLayout.outlierDetection.annotate.y),
               xycoords = 'axes fraction' )
    
    # Set supTitle
    if plotLayout.outlierDetection.supTitle:
        '''
        if plotLayout.outlierDetection.supTitle == 'auto':
            
            outliersfig.suptitle('Outlier detection and removal')
        
        else:
    
            outliersfig.suptitle(plotLayout.outlierDetection.supTitle)
        '''
           
        if plotLayout.outlierDetection.supTitle:
            
            if '%s' in plotLayout.outlierDetection.supTitle:
                
                suptitle = plotLayout.outlierDetection.supTitle.replace('%s',targetFeature)
                
                outliersfig.suptitle(suptitle)
        
        else:
            
            outliersfig.suptitle(plotLayout.outlierDetection.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        outliersfig.tight_layout() 
                        
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
        
        outliersfig.savefig(plotFPN)    
       
def PlotVarianceThreshold(plotLayout, plotFPN,
         X_train, X_test, retainL, discardL, columns, scaler):
    """ 
    """
    
    # TODO: plot variance on right y-axis
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.varianceThreshold.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(X_train.index)-1)/maxSpectra )
    
    # ttration = trian-test ratio - only for adjusting colorramp
    ttratio = plotskipStep / ceil( (len(X_test.index)-1)/maxSpectra)
        
    trainSelectDF = X_train[ retainL ]
        
    testSelectDF = X_test[ retainL ]
    
    xlabels = list(columns.keys())
            
    xaxislabel = 'wavelength'
        
    yaxislabel = 'reflectance'
    
    if xlabels[0].startswith('pc-'):
        
        xaxislabel = 'principal components'
        
        yaxislabel = 'eigenvalues'
        
    elif xlabels[0].startswith('d'):
                
        yaxislabel = 'derivatives'
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['input'] = {}
    
    subplotsD['selected'] = {}
        
    subplotsD['input']['train'] = {'label': 'Training data (input)',
                                      'DF' : X_train}
    
    subplotsD['input']['test'] = { 'label': 'Test data (input)',
                                      'DF' : X_test}
    
    subplotsD['selected']['train'] = {'label': 'Selected covariates',
                                      'DF' : trainSelectDF}
    
    subplotsD['selected']['test'] = { 'label': 'Selected covariates',
                                      'DF' : testSelectDF}

    #fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
    varianceThresholdPlot, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
    n = int(len(X_train.index)/plotskipStep)+2
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    nCovars = len(retainL)+len(discardL)
    
    annotateStrD = {0:'', 1:'', 2:'',}
   
    if plotLayout.varianceThreshold.annotate.input:          
        if plotLayout.varianceThreshold.annotate.input == 'auto':
            annotateStrD[0] = 'Input bands\n  %s covars\n  showing every %s band' %(nCovars, plotskipStep)
        else:
            annotateStrD[0] = plotLayout.varianceThreshold.annotate.input
            
    if plotLayout.varianceThreshold.annotate.standard:          
        if plotLayout.varianceThreshold.annotate.standard == 'auto':
            annotateStrD[1] = 'Selected bands \n  %s selected; %s discarded\n  showing every %s band' %(len(retainL), len(discardL), plotskipStep)
        else:
            annotateStrD[1] = plotLayout.varianceThreshold.annotate.standard
    
    plotStyle =  GetPlotStyle(plotLayout)
    
    for r, key in enumerate(subplotsD):

        if r == 1:
                
                for c in range(len(subplotsD[key])):
                
                    ax[r][c].set(xlabel=xaxislabel)
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            plotcols = [item for item in subplotsD[key][subplotkey]['DF'].columns ]
            
            plotcolNr = [columns[key] for key in plotcols]
                     
            ax[r][c].annotate(annotateStrD[r],
                           (plotLayout.varianceThreshold.annotate.x,
                            plotLayout.varianceThreshold.annotate.y),
                           xycoords = 'axes fraction' )
           
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            if c == 0:
                
                if scaler == 'None':
                  
                    #ax[r][c].set(ylabel=plotLayout.varianceThreshold.ylabels[r])
                    ax[r][c].set(ylabel=yaxislabel)
                    
                else:
                    
                    ylabel = '%s %s' %(scaler, yaxislabel) 
                    
                    ax[r][c].set(ylabel=ylabel)
                                           
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:

                        ax[r][c].plot(plotcolNr, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                        
                        if r == 0:
                            
                            ax[1][c].plot(plotcolNr, row, plotStyle, color='grey', ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                    else:
                        
                        m = ceil(n*ttratio)
                        
                        ax[r][c].plot(plotcolNr, row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                        if r == 0:
                            
                            ax[1][c].plot(plotcolNr, row, plotStyle, color='grey', ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
           
                    n += 1
                      
            if r == 1 and plotLayout.varianceThreshold.axvline:
                
                for xvalue in discardL:
                    
                    ax[1][c].axvline(x=columns[xvalue], color='grey')           
                                   
    # Set supTitle
    if plotLayout.varianceThreshold.supTitle:
        
        if plotLayout.varianceThreshold.supTitle == 'auto':
            
            varianceThresholdPlot.suptitle('Variance threshold covariate selection')
        
        else:
    
            varianceThresholdPlot.suptitle(plotLayout.varianceThreshold.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        varianceThresholdPlot.tight_layout()
 
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
  
        varianceThresholdPlot.savefig(plotFPN)   # save the figure to file
             
    plt.close(varianceThresholdPlot)

def SetAxvspan(discardL, columns):
    '''
    '''
    columnKeys = list(columns.keys())

    axvspanD = {}
 
    firstDiscard = True
    
    for i,item in enumerate(columns):
        
        if item in discardL:
            
            if i == 0:
                
                spanBegin = columns[columnKeys[0]]
                #spanEnd = columns[columnKeys[i+1]]
                spanEnd = (columns[columnKeys[i+1]]+columns[columnKeys[i]])/2
                
                
            elif i == len(columns)-1:
                
                spanBegin = (columns[columnKeys[i-1]]+columns[columnKeys[i]])/2
                spanEnd = columns[columnKeys[i]]
                
            else:
                
                #spanBegin = columns[columnKeys[i-1]]
                #spanEnd = columns[columnKeys[i+1]]
                
                spanBegin = (columns[columnKeys[i-1]]+columns[columnKeys[i]])/2
                spanEnd = (columns[columnKeys[i+1]]+columns[columnKeys[i]])/2
                
            if firstDiscard:
            
                axvspanD[item] = {'begin': spanBegin, 'end':spanEnd}
                
                firstDiscard = False
                
                previousDiscard = item
                
            else:
                
                if axvspanD[previousDiscard]['end'] >= spanBegin:
                    
                    axvspanD[previousDiscard]['end'] = spanEnd
                
                else:
               
                    axvspanD[item] = {'begin': spanBegin, 'end':spanEnd}
                    
                    previousDiscard = item
        
    return axvspanD   

def PlotCoviariateSelection(selector, selectorSymbolisation,  plotLayout, plotFPN,
         X_train, X_test, retainL, discardL, columns, targetFeatureName, regressorName='None', scaler='None'):
    """ Plot for all covariate selections
    """
    
    # TODO: plot variance on right y-axis
    from math import ceil
    
    if plotLayout.axvspan.apply:
        
        axvspanD = SetAxvspan(discardL, columns)
        
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.varianceThreshold.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(X_train.index)-1)/maxSpectra )
    
    # ttration = trian-test ratio - only for adjusting colorramp
    ttratio = plotskipStep / ceil( (len(X_test.index)-1)/maxSpectra)
        
    trainSelectDF = X_train[ retainL ]
        
    testSelectDF = X_test[ retainL ]
    
    xlabels = list(columns.keys())
    
    xaxislabel, yaxislabel = GetAxisLabels(xlabels)
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['input'] = {}
    
    subplotsD['selected'] = {}
    
    subplotsD['discarded'] = {}
    
    subplotsD['input']['train'] = {'label': 'Training data (input)',
                                      'DF' : X_train}
    
    subplotsD['input']['test'] = { 'label': 'Test data (input)',
                                      'DF' : X_test}
    
    subplotsD['selected']['train'] = {'label': 'Selected covariates',
                                      'DF' : trainSelectDF}
    
    subplotsD['selected']['test'] = { 'label': 'Selected covariates',
                                      'DF' : testSelectDF}
    
    selectPlotFig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
    n = int(len(X_train.index)/plotskipStep)+2
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    nCovars = len(retainL)+len(discardL)
    
    annotateStrD = {0:'', 1:'', 2:'',}
   
    if selectorSymbolisation.annotate.input:          
        if selectorSymbolisation.annotate.input == 'auto':
 
            annotateStrD[0] = 'Input bands\n  %s covars\n  showing every %s band' %(nCovars, plotskipStep)
          
        else:
            annotateStrD[0] = selectorSymbolisation.annotate.input
    if selectorSymbolisation.annotate.output:
          
              
        if selectorSymbolisation.annotate.output == 'auto':
            if regressorName == 'None':
                annotateStrD[1] = 'Target feature: %s\n   %s selected; %s discarded' %(targetFeatureName, len(retainL), len(discardL))
            else:
                annotateStrD[1] = 'Target feature: %s\n  Regressor: %s\n  %s selected; %s discarded' %(targetFeatureName, regressorName, len(retainL), len(discardL))
        else:
            annotateStrD[1] = selectorSymbolisation.annotate.output
    
    plotStyle =  GetPlotStyle(plotLayout)
    
    for r, key in enumerate(subplotsD):

        if r == 1: # second (last) row - set xaxis label
                
            for c in range(len(subplotsD[key])):
                
                ax[r][c].set(xlabel=xaxislabel)
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            plotcols = [item for item in subplotsD[key][subplotkey]['DF'].columns ]
            
            plotcolNr = [columns[k] for k in plotcols]
                     
            ax[r][c].annotate(annotateStrD[r],
                           (selectorSymbolisation.annotate.x,
                            selectorSymbolisation.annotate.y),
                           xycoords = 'axes fraction',zorder=4 )
           
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            if c == 0:
                
                if scaler == 'None':
                  
                    ax[r][c].set(ylabel=yaxislabel)
                    
                else:
                    
                    ylabel = '%s %s' %(scaler, yaxislabel) 
                    
                    ax[r][c].set(ylabel=ylabel)
                                           
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:

                        ax[r][c].plot(plotcolNr, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth,zorder=2) 

                        
                        if r == 0:
                            
                            ax[1][c].plot(plotcolNr, row, plotStyle, color='lightgrey', ms=plotLayout.pointsize,lw=plotLayout.linewidth,zorder=1) 

                    else:
                        
                        m = ceil(n*ttratio)
                        
                        ax[r][c].plot(plotcolNr, row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth,zorder=2) 

                        if r == 0:
                            
                            ax[1][c].plot(plotcolNr, row, plotStyle, color='lightgrey', ms=plotLayout.pointsize,lw=plotLayout.linewidth,zorder=1) 
           
                    n += 1
                  
            if r == 1 and plotLayout.axvspan.apply:
                
                for span in axvspanD:
                    
                    ax[1][c].axvspan(axvspanD[span]['begin'], axvspanD[span]['end'], 
                                     ymin=plotLayout.axvspan.ymin, ymax=plotLayout.axvspan.ymax, 
                                     color=plotLayout.axvspan.color, alpha=plotLayout.axvspan.alpha,
                                     zorder=3)           
                                   
    # Set supTitle
    if selectorSymbolisation.supTitle:
        
        if plotLayout.varianceThreshold.supTitle == 'auto':
            
            if regressorName == 'None':
                
                suptitle = '%s covariate selection for %s' %(selector, targetFeatureName, regressorName)
            
            else:
                
                suptitle = '%s covariate selection for %s (regresson: %s)' %(selector, targetFeatureName, regressorName)
            
            selectPlotFig.suptitle(suptitle)
        
        else:
    
            if '%s' in selectorSymbolisation.supTitle:
                
                suptitle = selectorSymbolisation.supTitle.replace('%s',targetFeatureName)
                
                selectPlotFig.suptitle(suptitle)
            
            else:
                
                selectPlotFig.suptitle(selectorSymbolisation.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        selectPlotFig.tight_layout()
                 
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        selectPlotFig.savefig(plotFPN)   # save the figure to file
             
    plt.close(selectPlotFig)  