'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

from support import CheckMakeDocPaths, ReadProjectFile, PathJoin, PathExists, PathSplit, \
    ReadAnyJson, ReadModelJson, DumpAnyJson

from template import CreateArrangeParamJson

from multiComp import SetMultiCompDstFPNs, SetMultCompPlots

from mlModelPilot import MachineLearningModel

def ForceSetupFromMultiComp(paramD, modelNr, multCompPlotsColumns, multiProjectComparisonD):
    '''
    '''
    paramD['multcompplot'] = True
    
    paramD['modelNr'] = modelNr
    
    paramD['multCompPlotsColumns'] = multCompPlotsColumns
    
    # Replace the list of targetFeatures in paramD
    paramD['targetFeatures'] = multiProjectComparisonD['targetFeatures']
    
    # Replace the applied regressors, but not the hyper parameter definitions    
    for regressor in paramD["modelling"]['regressionModels']:
                       
        paramD["modelling"]['regressionModels'][regressor]['apply'] = multiProjectComparisonD["modelling"]['regressionModels'][regressor]['apply'] 
        
    # Replace all the processing steps boolean apply            
    processStepD = {}
    
    processStepD['spectraPreProcess'] = {}; processStepD['spectraInfoEnhancement'] = {}
    processStepD['manualFeatureSelection'] = {}; processStepD['generalFeatureSelection'] = {};
    processStepD['removeOutliers'] = {}; processStepD['specificFeatureAgglomeration'] = {};
    processStepD['specificFeatureSelection'] = {}; 
    
    processStepD['spectraPreProcess'] = ['filtering','multifiltering']
    processStepD['spectraInfoEnhancement'] = ['scatterCorrection','standardisation','derivatives','decompose']
    processStepD['manualFeatureSelection'] = []
    
    processStepD['generalFeatureSelection'] = ['varianceThreshold']
    processStepD['removeOutliers'] = []
    processStepD['specificFeatureAgglomeration'] = ['wardClustering'] 
    processStepD['specificFeatureSelection'] = ['univariateSelection','permutationSelector',
                                                'RFE']
    
    for pkey in processStepD:
                        
        paramD[pkey]['apply'] = multiProjectComparisonD[pkey]['apply']
        
        for subp in processStepD[pkey]:
            
            paramD[pkey][subp]['apply'] = multiProjectComparisonD[pkey][subp]['apply']
    
    # Replace the feature importance reporting
    paramD['modelling']['featureImportance'] = multiProjectComparisonD['modelling']['featureImportance']
     
    # Replace the model test
    paramD['modelling']['modelTests'] = multiProjectComparisonD['modelling']['modelTests']   

    return paramD

def SetupMultiComp(iniParams, jsonProcessObjectD, jsonProcessObjectL, targetFeatureSymbolsD):
    '''
    '''
    if not 'multiprojectcomparison' in jsonProcessObjectD or not jsonProcessObjectD['multiprojectcomparison']:
        
        multiProjectComparisonD = {'apply': False} 
    
    elif not PathExists(jsonProcessObjectD['multiprojectcomparison']):
        
        multiProjectComparisonD = {'apply': False} 
        
    else:    
    
        multiProjectComparisonD = ReadAnyJson(jsonProcessObjectD['multiprojectcomparison'])
           
    multCompFig = multCompAxs = False
    
    if multiProjectComparisonD['apply']:
        
        if (len(jsonProcessObjectL) < 2):
            
            exitStr = 'Exiting: multi comparison projects must have at least 2 project files\n    %s has only %s' %(iniParams['projFN'],(len(jsonProcessObjectL)))
            
            exitStr += '\n    Either add more projects or remove the "multiprojectcomparison" command'
            
            exit(exitStr)
                
        multCompImagesFPND, multCompJsonSummaryFPND = SetMultiCompDstFPNs(iniParams['rootpath'],iniParams['arrangeddatafolder'],
                                                                          multiProjectComparisonD)

        multCompFig, multCompAxs, multCompPlotsColumns = SetMultCompPlots( multiProjectComparisonD,targetFeatureSymbolsD, len(jsonProcessObjectL) )

        multCompSummaryD = {}
        
        for targetFeature in multiProjectComparisonD['targetFeatures']:
            
            multCompSummaryD[targetFeature] = {}
            
    return multiProjectComparisonD, multCompImagesFPND, multCompJsonSummaryFPND,\
        multCompFig, multCompAxs, multCompPlotsColumns, multCompSummaryD


def SetupScatterCorr(paramD):
    '''
    '''
                           
    if len(paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']) == 0:
    
        paramD['spectraInfoEnhancement']['scatterCorrection']['apply'] = False

    elif len(paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']) == 1:
        
        paramD['spectraInfoEnhancement']['scatterCorrection']['singles'] = \
            paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']
            
        paramD['spectraInfoEnhancement']['scatterCorrection']['duals'] = []
        
    else:
        
        paramD['spectraInfoEnhancement']['scatterCorrection']['singles'] = []

        paramD['spectraInfoEnhancement']['scatterCorrection']['duals'] = \
            paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']
    
    return paramD                    
                        
def SetupProcesses(iniParams):
    '''Setup and loop processes

    :param rootpath: path to project root folder
    :type: lstr

    :param sourcedatafolder: folder name of original OSSL data (source folder)
    :type: lstr

    :param arrangeddatafolder: folder name of arranged OSSL data (destination folder)
    :type: lstr

    :param projFN: project filename (in destination folder)
    :type: str

    :param jsonpath: folder name
    :type: str

    '''

    dstRootFP, jsonFP = CheckMakeDocPaths(iniParams['rootpath'],
                                          iniParams['arrangeddatafolder'],
                                          iniParams['jsonfolder'],
                                          iniParams['sourcedatafolder'])

    if iniParams['createjsonparams']:

        CreateArrangeParamJson(jsonFP,iniParams['projFN'],'mlmodel')

    jsonProcessObjectD = ReadProjectFile(dstRootFP, iniParams['projFN'])
       
    #jsonProcessObjectL = jsonProcessObjectD['projectFiles']
    jsonProcessObjectL = [PathJoin([jsonFP,x.strip()])  for x in jsonProcessObjectD['projectFiles'] if len(x) > 10 and x[0] != '#']
    
    # Get the target Feature Symbols
    targetFeatureSymbolsD = ReadAnyJson(iniParams['targetfeaturesymbols'])
    
    # Get the regression model symbols
    regressionModelSymbolsD = ReadAnyJson(iniParams['regressionmodelsymbols'])
    
    # Get the enhancement plot layout 
    enhancementPlotLayoutD = ReadAnyJson(iniParams['enhancementplotlayout'])
    
    # Get the plot model layout 
    modelPlotD = ReadAnyJson(iniParams['modelplot'])
    
    MultiArgs = SetupMultiComp(iniParams, jsonProcessObjectD, jsonProcessObjectL, targetFeatureSymbolsD)
    
    multiProjectComparisonD, multCompImagesFPND, multCompJsonSummaryFPND,\
        multCompFig, multCompAxs, multCompPlotsColumns, multCompSummaryD = MultiArgs
 
    '''
    if not 'multiprojectcomparison' in jsonProcessObjectD or not jsonProcessObjectD['multiprojectcomparison']:
        
        multiProjectComparisonD = {'apply': False} 
    
    elif not PathExists(jsonProcessObjectD['multiprojectcomparison']):
        
        multiProjectComparisonD = {'apply': False} 
        
    else:    
    
        multiProjectComparisonD = ReadAnyJson(jsonProcessObjectD['multiprojectcomparison'])
           
    multCompFig = multCompAxs = False
    
    if multiProjectComparisonD['apply']:
        
        if (len(jsonProcessObjectL) < 2):
            
            exitStr = 'Exiting: multi comparison projects must have at least 2 project files\n    %s has only %s' %(iniParams['projFN'],(len(jsonProcessObjectL)))
            
            exitStr += '\n    Either add more projects or remove the "multiprojectcomparison" command'
            
            exit(exitStr)
                
        multCompImagesFPND, multCompJsonSummaryFPND = SetMultiCompDstFPNs(iniParams['rootpath'],iniParams['arrangeddatafolder'],
                                                                          multiProjectComparisonD)

        multCompFig, multCompAxs, multCompPlotsColumns = SetMultCompPlots( multiProjectComparisonD,targetFeatureSymbolsD, len(jsonProcessObjectL) )

        multCompSummaryD = {}
        
        for targetFeature in multiProjectComparisonD['targetFeatures']:
            
            multCompSummaryD[targetFeature] = {}
    '''
    modelNr = 0    
    
    #Loop over all json files
    for jsonObj in jsonProcessObjectL:

        print ('    jsonObj:', jsonObj)

        paramD = ReadModelJson(jsonObj)
        
        # Setting of single and/or dual scatter correction
        if paramD['spectraInfoEnhancement']['apply']:
        
            if paramD['spectraInfoEnhancement']['scatterCorrection']['apply']:
                           
                if len(paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']) == 0:
                
                    paramD['spectraInfoEnhancement']['scatterCorrection']['apply'] = False
            
                elif len(paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']) == 1:
                    
                    paramD['spectraInfoEnhancement']['scatterCorrection']['singles'] = \
                        paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']
                        
                    paramD['spectraInfoEnhancement']['scatterCorrection']['duals'] = []
                    
                else:
                    
                    paramD['spectraInfoEnhancement']['scatterCorrection']['singles'] = []
          
                    paramD['spectraInfoEnhancement']['scatterCorrection']['duals'] = \
                        paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']
                                       
        # Add the targetFeatureSymbols
        paramD['targetFeatureSymbols'] = targetFeatureSymbolsD['targetFeatureSymbols']
        
        paramD['featureImportancePlot'] = targetFeatureSymbolsD['featureImportancePlot']
        
        # Add the regressionModelSymbols 
        paramD['regressionModelSymbols'] = regressionModelSymbolsD['regressionModelSymbols']
        
        # Add the targetFeatureSymbols
        paramD['featureImportancePlot'] = targetFeatureSymbolsD['featureImportancePlot']
        
        paramD['enhancementPlotLayout'] = enhancementPlotLayoutD['enhancementPlotLayout']
                
        # Add the plot model layout
        paramD['modelPlot'] = modelPlotD['modelPlot']

        paramD['multcompplot'] = False
        
        if multiProjectComparisonD['apply']:
            
            paramD = ForceSetupFromMultiComp(paramD, modelNr, multCompPlotsColumns, multiProjectComparisonD)
                                   
        # Setting of single and/or dual scatter correction
        if paramD['spectraInfoEnhancement']['apply'] and paramD['spectraInfoEnhancement']['scatterCorrection']['apply']:
                        
            paramD = SetupScatterCorr(paramD)
                        
        # Get the target feature transform
        targetFeatureTransformD = ReadAnyJson(paramD['input']['targetfeaturetransforms'])
    
        # Add the targetFeatureTransforms
        paramD['targetFeatureTransform'] = targetFeatureTransformD['targetFeatureTransform']
        
        # Invoke the modeling
        mlModel = MachineLearningModel(paramD)

        # Set the regressor models to apply
        mlModel._RegModelSelectSet()
        
        mlModel._CheckParams(PathSplit(jsonObj)[1]);

        # run the modeling
        mlModel._PilotModeling(iniParams['rootpath'],iniParams['sourcedatafolder'],  dstRootFP, multCompFig, multCompAxs)
        
        if multiProjectComparisonD['apply']: 
            
            modelNrStr = '%s' %(modelNr)
            
            if modelNrStr in multiProjectComparisonD['trialid']:
                
                trialid = multiProjectComparisonD['trialid'][modelNrStr]
            
            else:
            
                trialid = 'trial_%s' %(modelNr)
            
            for targetFeature in mlModel.targetFeatures:
            
                multCompSummaryD[targetFeature][trialid] = mlModel.multCompSummaryD[targetFeature]
               
        modelNr += 1
    
    if multiProjectComparisonD['apply']: 
        
        print ('All models in project Done') 
        
        #if multiProjectComparisonD['plot']['screenShow']:
        
        #    plt.show()
        
        #pp = pprint.PrettyPrinter(indent=2)
        #pp.pprint(multCompSummaryD)
        
        for targetFeature in multCompFig:
    
            for index in multCompFig[targetFeature]:
                
                jsonD = {targetFeature : multCompSummaryD[targetFeature]}
                
                DumpAnyJson(jsonD,multCompJsonSummaryFPND[targetFeature]) 
                
                if multiProjectComparisonD['plot']['savePng']: 
                             
                    multCompFig[targetFeature][index].savefig( multCompImagesFPND[targetFeature][index] )
        