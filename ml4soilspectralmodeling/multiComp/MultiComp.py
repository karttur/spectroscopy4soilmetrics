'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

from support import PathJoin, PathExists, MakeDirs

from plotChart import MultiPlot

def SetMultiCompDstFPNs(rootPath, arrangeDataPath, multiProjectComparisonD):
    '''
    '''

    multiCompFP = PathJoin([rootPath,arrangeDataPath,'multicomp'])
    
    if not PathExists(multiCompFP):
        
        MakeDirs(multiCompFP)
        
    multiCompProjectFP = PathJoin([ multiCompFP, multiProjectComparisonD['prefix'] ])
    
    if not PathExists(multiCompProjectFP):
        
        MakeDirs(multiCompProjectFP)
        
    multiCompProjectImageFP = PathJoin([ multiCompProjectFP, 'images' ])
    
    if not PathExists(multiCompProjectImageFP):
        
        MakeDirs(multiCompProjectImageFP)
        
    multiCompProjectJsonFP = PathJoin([ multiCompProjectFP, 'json' ])
    
    if not PathExists(multiCompProjectJsonFP):
        
        MakeDirs(multiCompProjectJsonFP)
                   
    indexL = ['coefficientImportance','permutationImportance','treeBasedImportance','trainTest','Kfold']
    
    multCompImagesFPND = {}
    
    multCompJsonSummaryFPND = {}
    
    for targetFeature in multiProjectComparisonD['targetFeatures']:
            
        #print ('targetFeature', targetFeature)
        
        multCompSummaryFN = '%s_%s.json' %(multiProjectComparisonD['prefix'],targetFeature)
        
        multCompJsonSummaryFPND[targetFeature] = PathJoin([ multiCompProjectJsonFP, multCompSummaryFN ])
                
        multCompImagesFPND[targetFeature] = {}
        
           
        for i in indexL:
           
            #print ('i',i)
                      
            multCompImagesFN = '%s_%s_%s.png' %(multiProjectComparisonD['prefix'],targetFeature, i)
            
            multCompImagesFPND[targetFeature][i] = PathJoin([ multiCompProjectImageFP, multCompImagesFN ])
              
    return multCompImagesFPND, multCompJsonSummaryFPND

def SetMultCompPlots(multiProjectComparisonD, targetFeatureSymbolsD, figCols):
    '''
    '''

    if figCols == 0:
        
        exit('Multi comparisson requres at least one feature importance or one model test')

    multCompPlotIndexL = []
    
    multCompPlotsColumnD = {}
    
    multCompFig = {}
    
    multCompAxs = {}
    
    regressionModelL = []
    
    # Set the regression models to include:
    
    for r,row in enumerate(multiProjectComparisonD['modelling']['regressionModels']):

        if multiProjectComparisonD['modelling']['regressionModels'][row]['apply']:
            
            regressionModelL.append(row)
            
    figRows = len(regressionModelL)
        
    # Set the columns to include
    if multiProjectComparisonD['modelling']['featureImportance']['apply']:
        
        if multiProjectComparisonD['modelling']['featureImportance']['permutationImportance']['apply']:
        
            multCompPlotsColumnD['permutationImportance'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('permutationImportance')
        
        if multiProjectComparisonD['modelling']['featureImportance']['treeBasedImportance']['apply']:
        
            multCompPlotsColumnD['treeBasedImportance'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('treeBasedImportance')
                
        if multiProjectComparisonD['modelling']['featureImportance']['coefficientImportance']['apply']:
        
            multCompPlotsColumnD['coefficientImportance'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('coefficientImportance')
                
    if multiProjectComparisonD['modelling']['modelTests']['apply']:
        
        if multiProjectComparisonD['modelling']['modelTests']['trainTest']['apply']:
        
            multCompPlotsColumnD['trainTest'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('trainTest')
            
        if multiProjectComparisonD['modelling']['modelTests']['Kfold']['apply']:
        
            multCompPlotsColumnD['Kfold'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('Kfold')
                       
    # Set the figure size
    if multiProjectComparisonD['plot']['figSize']['x'] == 0:
        
        xadd = multiProjectComparisonD['plot']['figSize']['xadd']

        figSizeX = 3 * figCols + xadd

    else:

        figSizeX =multiProjectComparisonD['plot']['figSize']['x']

    if multiProjectComparisonD['plot']['figSize']['y'] == 0:
        
        yadd = multiProjectComparisonD['plot']['figSize']['yadd']

        figSizeY = 3 * figRows + yadd

    else:

        figSizeY =multiProjectComparisonD['plot']['figSize']['y']
                
    # Create column plots for each trial, with rows showing different regressors
    for targetFeature in multiProjectComparisonD['targetFeatures']:
        
        multCompFig[targetFeature] = {}; multCompAxs[targetFeature] = {}
        
        for index in multCompPlotIndexL:
            
            multCompFig[targetFeature][index], multCompAxs[targetFeature][index] = MultiPlot(figRows, figCols, figSizeX, figSizeY)

            if multiProjectComparisonD['plot']['tightLayout']:
    
                multCompFig[targetFeature][index].tight_layout()

            # Set subplot wspace and hspace
            if multiProjectComparisonD['plot']['hwspace']['wspace']:
    
                multCompFig[targetFeature][index].subplots_adjust(wspace=multiProjectComparisonD['plot']['hwspace']['wspace'])
    
            if multiProjectComparisonD['plot']['hwspace']['hspace']:
    
                multCompFig[targetFeature][index].subplots_adjust(hspace=multiProjectComparisonD['plot']['hwspace']['hspace'])
    
            if figCols < 3:
                
                label = targetFeatureSymbolsD['targetFeatureSymbols'][targetFeature]['short']
                
            else:
                
                label = targetFeatureSymbolsD['targetFeatureSymbols'][targetFeature]['label']
        
            if index in ['trainTest','Kfold']:
                    
                suptitle = "Model: %s; Target: %s (rows=regressors)\n" %(index, label)
            
            else:
                
                indextitle = '%s' %( index.replace('Importance', ' importance'))
                
                if figCols < 3:
                    
                    suptitle = "%s; Target: %s (rows=regressors)\n" %(indextitle.capitalize(), label)
                
                else:
                     
                    suptitle = "Covar evaluation: %s; Target: %s (rows=regressors)\n" %(indextitle, label)
                            
            # Set suptitle
            multCompFig[targetFeature][index].suptitle( suptitle )
    
            for c in range(figCols): 
                
                modelNrStr = '%s' %(c)
            
                if modelNrStr in multiProjectComparisonD['trialid']:
                    
                    trialId = multiProjectComparisonD['trialid'][modelNrStr]
                
                else:
                
                    trialId = 'trial_%s' %(c)
                                
                if figRows == 1:
                
                    multCompAxs[targetFeature][index][c].set_title( trialId )
                
                else:
                    
                    multCompAxs[targetFeature][index][0][c].set_title( trialId )
                                          
    return (multCompFig, multCompAxs, multCompPlotsColumnD)
