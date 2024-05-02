'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

import numpy as np 

from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox

import pprint

from argumentObject import Obj

from mlRregressors import RegressionModels

from preProcesses import MLPreProcess, ScatterCorrection, Standardisation, Derivatives

from support import DeepCopy, PathJoin, PathExists, ReadAnyJson, PdDataFrame, \
    PathSplit, MakeDirs, ReadModelJson, DumpAnyJson
    
from plotChart import MultiPlot, ClosePlot

class MachineLearningModel(Obj, MLPreProcess, RegressionModels):
    ''' Machine Learning model of feature propertie from spectra
    '''

    def __init__(self,paramD):
        """ Convert input parameters from nested dict to nested class object

            :param dict paramD: parameters
        """

        # convert the input parameter dict to class objects
        Obj.__init__(self,paramD)
        
        # Initaite the pre processes
        MLPreProcess.__init__(self)

        # initiate the regression models
        RegressionModels.__init__(self)

        self.paramD = paramD

        # Set class object default data if required
        self._SetModelDefaults()

        # Deep copy parameters to a new object class called params
        self.params = DeepCopy(self)

        # Drop the plot and figure settings from paramD
        paramD.pop('modelPlot')

        # Deep copy the parameters to self.soillineD
        self.modelPlotD = DeepCopy(paramD)
        
    def _CheckParams(self, jsonProcessFN):
        ''' Check parameters
        '''
        
        if not hasattr(self,'targetFeatures'):
            exitStr = 'Exiting: the modelling process file %s\n    has not targetFeature' %(jsonProcessFN)
            exit(exitStr)
                
    def _SetSrcFPNs(self, rootFP, dstRootFP, sourcedatafolder):
        ''' Set source file paths and names
        '''

        # All OSSL data are download as a zipped subfolder with data given standard names as of below
               
        # if the path to rootFP starts with a dot '.' (= self) then use the default rootFP 
        if self.input.jsonSpectraDataFilePath[0] == '.':
            
            removeStart = 1
            
            if self.input.jsonSpectraDataFilePath[1] in ['/']:
                
                removeStart = 2
            
            dataSubFP = self.input.jsonSpectraDataFilePath[removeStart: len(self.input.jsonSpectraDataFilePath)]
               
            jsonSpectraDataFilePath = PathJoin([dstRootFP, dataSubFP])
            
        else:
            
            jsonSpectraDataFilePath = self.input.jsonSpectraDataFilePath
            
        if self.input.jsonSpectraParamsFilePath[0] == '.':
            
            removeStart = 1
            
            if self.input.jsonSpectraParamsFilePath[1] in ['/']:
                
                removeStart = 2
            
            paramSubFP = self.input.jsonSpectraParamsFilePath[removeStart: len(self.input.jsonSpectraParamsFilePath)]
               
            jsonSpectraParamsFilePath = PathJoin([dstRootFP, paramSubFP])
            

        else:
        
            jsonSpectraParamsFilePath = self.input.jsonSpectraParamsFilePath
            
        if not PathJoin([jsonSpectraDataFilePath]):
            
            exitStr = 'Data file not found: %s ' %(jsonSpectraDataFilePath)
            
            exit(exitStr)

        if not PathExists(jsonSpectraParamsFilePath):
            
            exitStr = 'Param file not found: %s ' %(jsonSpectraParamsFilePath)
            
            exit(exitStr)
            
        self.dataFPN = jsonSpectraDataFilePath
        
        self.jsonSpectraData = ReadAnyJson(jsonSpectraDataFilePath)
        
        self.jsonSpectraParams = ReadAnyJson(jsonSpectraParamsFilePath)
        
        '''
        # Open and load JSON data file
        with open(jsonSpectraDataFilePath) as jsonF:

            self.jsonSpectraData = json.load(jsonF)

        # Open and load JSON parameter file
        with open(jsonSpectraParamsFilePath) as jsonF:

            self.jsonSpectraParams = json.load(jsonF)
        '''    
        
    def _GetAbundanceData(self):
        '''
        '''

        # Get the list of substances included in this dataset
        substanceColumns = self.jsonSpectraParams['labData']

        #substanceColumns = self.jsonSpectraParams['targetFeatures']

        substanceOrderD = {}
        
        for substance in substanceColumns:
            
            substanceOrderD[substance] = substanceColumns.index(substance)

        n = 0
        
        # Loop over the samples
        for sample in self.jsonSpectraData['spectra']:
            
            # Dict error [TGTODO check]
            if not 'abundances' in sample:
                
                continue

            substanceL = [None] * len(substanceColumns)

            for abundance in sample['abundances']:

                substanceL[ substanceOrderD[abundance['substance']] ] = abundance['value']

            if n == 0:

                abundanceA = np.asarray(substanceL, dtype=float)

            else:

                abundanceA = np.vstack( (abundanceA, np.asarray(substanceL, dtype=float) ) )

            n += 1

        self.abundanceDf = PdDataFrame(abundanceA, substanceColumns)

        self.transformD = {}
        
        # Do any transformation requested
        for column in substanceColumns:
            
            self.transformD[column] = 'linear'
            
            if hasattr(self.params.targetFeatureTransform, column):
                
                targetTransform = getattr(self.params.targetFeatureTransform, column)
                
                if targetTransform.log:
                    
                    self.abundanceDf[column] = np.log(self.abundanceDf[column])
                    
                    self.transformD[column] = 'log'
                    
                elif targetTransform.sqrt:
                    
                    self.abundanceDf[column] = np.sqrt(self.abundanceDf[column])
                    
                    self.transformD[column] = 'sqrt'
                    
                elif targetTransform.reciprocal:
                    
                    self.abundanceDf[column] = np.reciprocal(self.abundanceDf[column])
                    
                    self.transformD[column] = 'reciprocal'
                    
                elif targetTransform.boxcox:
                    
                    pt = PowerTransformer(method='box-cox')
                    
                    try:
                    
                        pt.fit( np.atleast_2d(self.abundanceDf[column]) )
                        
                        print(pt.lambdas_)
                        
                        print(pt.transform( np.atleast_2d(self.abundanceDf[column]) ))
                        
                        X = pt.transform( np.atleast_2d(self.abundanceDf[column]) )
                        
                        print (X)
                        
                        self.abundanceDf[column], boxcoxLambda = boxcox(self.abundanceDf[column])
                                            
                        self.transformD[column] = 'boxcox'
                        
                        self.abundanceDf[column]
                        
                        print ( self.abundanceDf[column] )
                                              
                    except:
                        
                        print('cannot box-cox transform')
                elif targetTransform.yeojohnson:
                    
                    pt = PowerTransformer()
                    
                    pt.fit(np.atleast_2d(self.abundanceDf[column]))
                     
                    print(pt.lambdas_)
                    
                    print(pt.transform( np.atleast_2d(self.abundanceDf[column]) ) )
                    
                    X = pt.transform( np.atleast_2d(self.abundanceDf[column]) )
                    
                    print (X)
                    
    
                            
                elif targetTransform.quantile:
                    
                    pass
                    
                    #X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
                    #qt = QuantileTransformer(n_quantiles=10, random_state=0)
                                                
    def _StartStepSpectra(self, pdSpectra, startwl, stopwl, stepwl):
        '''
        '''

        wlmin = pdSpectra['wl'].min()
        
        wlmax = pdSpectra['wl'].max()
        
        step = (wlmax-wlmin)/(pdSpectra.shape[0])

        startindex = (startwl-wlmin)/step

        stopindex = (stopwl-wlmin)/step

        stepindex = stepwl/step

        indexL = []; iL = []

        i = 0

        while True:

            if i*stepindex+startindex > stopindex+1:

                break

            indexL.append(int(i*stepindex+startindex))
            iL.append(i)

            i+=1

        df = pdSpectra.iloc[indexL]

        return df, indexL[0]

    def _SpectraDerivativeFromDf(self,dataFrame,columns):
        ''' Create spectral derivates
        '''

        # Get the derivatives
        spectraDerivativeDF = dataFrame.diff(axis=1, periods=1)

        # Drop the first column as it will have only NaN
        spectraDerivativeDF = spectraDerivativeDF.drop(columns[0], axis=1)

        # Reset columns to integers
        columns = [int(i) for i in columns]

        # Create the derivative columns
        derivativeColumns = ['d%s' % int((columns[i-1]+columns[i])/2) for i in range(len(columns)) if i > 0]

        # Replace the columns
        spectraDerivativeDF.columns = derivativeColumns

        return spectraDerivativeDF, derivativeColumns

    def _GetBandData(self):
        ''' Read json data into numpy array and convert to pandas dataframe
        '''
        #self.varianceSelectTxt = None; 
        self.outlierTxt = None
        self.generalFeatureSelectTxt = None; 
        self.specificFeatureSelectionTxt = None; 
        self.agglomerateTxt = None

        # Use the wavelength as column headers
        columnsInt = self.jsonSpectraData['waveLength']

        # Convert the column headers to strings
        columnsStr = [str(c) for c in columnsInt]
        
        # create a headerDict to have the columns as both strings (key) and floats/integers (values)
        # Pandas DataFrame requires string, whereas pyplot requires numerical
        self.columns = dict(zip(columnsStr,columnsInt))
        
        self.originalColumns = dict(zip(columnsStr,columnsInt))
        
        n = 0

        # Loop over the spectra
        for sample in self.jsonSpectraData['spectra']:

            if n == 0:

                spectraA = np.asarray(sample['signalMean'])

            else:

                spectraA = np.vstack( (spectraA, np.asarray(sample['signalMean']) ) )

            n += 1

        self.spectraDF = PdDataFrame(spectraA, columnsStr)
               
        self.firstBand = columnsStr[0]
        
        self.lastBand = columnsStr[len(columnsStr)-1]
                
        self.resolutionBand = int(round(( self.columns[self.lastBand]-self.columns[self.firstBand] )/(len(self.columns)-1)))
        
    def _SetSubPlots(self):
        '''
        '''

        if self.modelPlot.rows.apply:

            self.nRegrModels = len(self.regressorModels)

            self.nTargetFeatures = len(self.targetFeatures)

            self.columnFig = {}

            self.columnAxs = {}
            
            

            if self.modelPlot.rows.targetFeatures.apply:

                self.targetFeaturePlotColumnD = {}

                for c, col in enumerate(self.modelPlot.rows.targetFeatures.columns):

                    self.targetFeaturePlotColumnD[col] = c

                self.targetFeaturesFigCols = len(self.modelPlot.rows.targetFeatures.columns)

                # Set the figure size
                xadd = self.modelPlot.rows.targetFeatures.figSize.xadd

                if  xadd == 0:

                    xadd = self.modelPlot.rows.subFigSize.xadd

                if self.modelPlot.rows.targetFeatures.figSize.x == 0:

                    figSizeX = self.modelPlot.rows.subFigSize.x * self.targetFeaturesFigCols + xadd

                else:

                    figSizeX = self.modelPlot.rows.targetFeatures.figSize.x

                yadd = self.modelPlot.rows.targetFeatures.figSize.yadd

                if  yadd == 0:

                    yadd = self.modelPlot.rows.subFigSize.yadd

                if self.modelPlot.rows.targetFeatures.figSize.y == 0:

                    figSizeY = self.modelPlot.rows.subFigSize.y * self.nTargetFeatures + yadd

                else:

                    figSizeY = self.modelPlot.rows.targetFeatures.figSize.y

                # Create column plots for individual targetFeatures, with rows showing different regressors
                for regrModel in self.regressorModels:

                    self.columnFig[regrModel[0]], self.columnAxs[regrModel[0]] = MultiPlot( self.nTargetFeatures, self.targetFeaturesFigCols, figSizeX, figSizeY )

                    if self.modelPlot.tightLayout:

                        self.columnFig[regrModel[0]].tight_layout()

                    # Set title
                    suptitle = "Regressor: %s; rows=target features; input features: %s;\n" %(regrModel[0], len(self.originalColumns))
                    suptitle += "%s; %s; %s; %s; \n" %(self.spectraInfoEnhancement.scatterCorrectiontxt, self.scalertxt, self.spectraInfoEnhancement.decompose.pcatxt, self.hyperParamtxt)
                    
                    # Set subplot wspace and hspace
                    if self.modelPlot.rows.regressionModels.hwspace.wspace:

                        self.columnFig[regrModel[0]].subplots_adjust(wspace=self.modelPlot.rows.regressionModels.hwspace.wspace)

                    if self.modelPlot.rows.regressionModels.hwspace.hspace:

                        self.columnFig[regrModel[0]].subplots_adjust(hspace=self.modelPlot.rows.regressionModels.hwspace.hspace)

                    #if self.varianceSelectTxt != None:
                    if self.generalFeatureSelectTxt != None:

                        suptitle += ', %s' %(self.generalFeatureSelectTxt)

                    if self.outlierTxt != None:

                        suptitle +=  ', %s' %(self.outlierTxt)

                    self.columnFig[regrModel[0]].suptitle(  suptitle )

                    for r,rows in enumerate(self.targetFeatures):

                        for c,cols in enumerate(self.modelPlot.rows.targetFeatures.columns):

                            # Set subplot titles:
                            if 'Importance' in cols:

                                if r == 0:

                                    title = '%s' %( cols.replace('Importance', ' Importance'))

                                    if (len(self.targetFeatures)) == 1:

                                        self.columnAxs[ regrModel[0] ][c].set_title(title)

                                    else:

                                        self.columnAxs[ regrModel[0] ][r,c].set_title(title)

                            else:

                                title = '%s %s' %( self.paramD['targetFeatureSymbols'][rows]['label'], cols)

                                if (len(self.targetFeatures)) == 1:

                                    self.columnAxs[ regrModel[0] ][c].set_title(title)

                                else:

                                    self.columnAxs[ regrModel[0] ][r,c].set_title(title)

            if self.modelPlot.rows.regressionModels.apply:

                self.regressionModelPlotColumnD = {}

                for c, col in enumerate(self.modelPlot.rows.regressionModels.columns):

                    self.regressionModelPlotColumnD[col] = c

                self.regressionModelFigCols = len(self.modelPlot.rows.regressionModels.columns)

                # Set the figure size

                xadd = self.modelPlot.rows.regressionModels.figSize.xadd

                if  xadd == 0:

                    xadd = self.modelPlot.rows.subFigSize.xadd

                if self.modelPlot.rows.regressionModels.figSize.x == 0:

                    figSizeX = self.modelPlot.rows.subFigSize.x * self.regressionModelFigCols + xadd

                else:

                    figSizeX = self.modelPlot.rows.regressionModels.figSize.x

                yadd = self.modelPlot.rows.regressionModels.figSize.yadd

                if  yadd == 0:

                    yadd = self.modelPlot.rows.subFigSize.yadd

                if self.modelPlot.rows.regressionModels.figSize.y == 0:

                    figSizeY = self.modelPlot.rows.subFigSize.y * self.nRegrModels + yadd

                else:

                    figSizeY = self.modelPlot.rows.regressionModels.figSize.x

                # Create column plots for individual regressionModels, with rows showing different regressors
                for targetFeature in self.targetFeatures:

                    self.columnFig[targetFeature], self.columnAxs[targetFeature] = MultiPlot( self.nRegrModels, self.regressionModelFigCols, figSizeX, figSizeY)

                    # ERROR If only one regressionModle then r == NONE

                    if self.modelPlot.tightLayout:

                        self.columnFig[targetFeature].tight_layout()

                    # Set subplot wspace and hspace
                    if self.modelPlot.rows.targetFeatures.hwspace.wspace:

                        self.columnFig[targetFeature].subplots_adjust(wspace=self.modelPlot.rows.targetFeatures.hwspace.wspace)

                    if self.modelPlot.rows.targetFeatures.hwspace.hspace:

                        self.columnFig[targetFeature].subplots_adjust(hspace=self.modelPlot.rows.targetFeatures.hwspace.hspace)

                    label = self.paramD['targetFeatureSymbols'][targetFeature]['label']

                    suptitle = "Target: %s, %s (rows=regressors)\n" %(label, self.hyperParamtxt )

                    suptitle += '%s input features' %(len(self.originalColumns))

                    if self.generalFeatureSelectTxt != None:

                        suptitle += ', %s' %(self.generalFeatureSelectTxt)

                    if self.outlierTxt != None:

                        suptitle +=  ', %s' %(self.outlierTxt)

                    # Set suotitle
                    self.columnFig[targetFeature].suptitle( suptitle )

                    # Set subplot titles:
                    for r,rows in enumerate(self.regressorModels):

                        for c,cols in enumerate(self.modelPlot.rows.regressionModels.columns):

                            #title = '%s %s' %(rows[0], cols)

                            # Set subplot titles:
                            if 'Importance' in cols:

                                if r == 0:

                                    title = '%s' %( cols.replace('Importance', ' Importance'))

                                    if (len(self.regressorModels)) == 1:

                                        self.columnAxs[targetFeature][c].set_title( title )

                                    else:

                                        self.columnAxs[targetFeature][r,c].set_title( title )

                            else:

                                title = '%s %s' %(rows[0], cols)
                                #title = '%s ' %( self.paramD['targetFeatureSymbols'][rows]['label'], cols)
                                if (len(self.regressorModels)) == 1:

                                    self.columnAxs[targetFeature][c].set_title( title )

                                else:

                                    self.columnAxs[targetFeature][r,c].set_title( title )

    def _GetPreProcessSummary(self):
        '''
        '''
        sumTxtL = []
        
        if self.spectraPreProcess.filtering.apply:
                
            sumTxtL.append('sf')
                
        elif self.spectraPreProcess.multifiltering.apply:
                
            sumTxtL.append('mf')
                             
        if self.spectraInfoEnhancement.apply:
             
            if self.spectraInfoEnhancement.scatterCorrection.apply:
                
                sumTxtL.append('sc')
                            
            if self.spectraInfoEnhancement.standardisation.apply:
                
                sumTxtL.append('st')
                                
            if self.spectraInfoEnhancement.derivatives.apply:
                
                sumTxtL.append('dv')

            if self.spectraInfoEnhancement.decompose.apply:
                        
                if self.spectraInfoEnhancement.decompose.pca.apply:
                    
                    sumTxtL.append('pc')
          
        # Any manual feature selection is applied to the original dataframe - i.e. affect all models the same
        if self.manualFeatureSelection.apply:

            sumTxtL.append('ma')
            
            return sumTxtL
                       
        if self.generalFeatureSelection.apply:

            if self.generalFeatureSelection.varianceThreshold.apply:
                
                sumTxtL.append('vt')
                
                
        if self.removeOutliers.apply:
    
            sumTxtL.append('ro')
                
        # Covariate (X) Agglomeration
        if self.specificFeatureAgglomeration.apply:

            if self.specificFeatureAgglomeration.wardClustering.apply:
                
                sumTxtL.append('wc')
        
        if  self.specificFeatureSelection.apply:
                            
            if self.specificFeatureSelection.univariateSelection.apply:
                        
                if self.specificFeatureSelection.univariateSelection.SelectKBest.apply:
                        
                    sumTxtL.append('us')
                                                  
                elif self.specificFeatureSelection.permutationSelector.apply:
    
                    sumTxtL.append('ps')
    
                elif self.specificFeatureSelection.RFE.apply:
                    
                    sumTxtL.append('rf')
                                
                elif self.specificFeatureSelection.treeBasedSelector.apply:
    
                    sumTxtL.append('tb')
                    
        return sumTxtL
        

    def _SetDstFPNs(self):
        ''' Set destination file paths and names
        '''
        
        self.ProducedImagesL = []
                
        self.ProducedJsonsL = []
 
        sumTxtL = self._GetPreProcessSummary()
        '''
        fixDict = {}
        fixDict['spectraPreProcess'] = {}
        fixDict['spectraPreProcess']['filtering'] = '@sf'; 
        fixDict['spectraPreProcess']['multifiltering'] = '@mf';
        
        fixDict['spectraInfoEnhancement'] = {}
        fixDict['spectraInfoEnhancement']['scatterCorrection'] = '@sc'
        
        fixDict['spectraInfoEnhancement']['standardisation'] = '@st'; 
        
        fixDict['spectraInfoEnhancement']['derivatives'] = '@dt';
        fixDict['spectraInfoEnhancement']['decompose'] = '@pc';
        
        #fixDict['spectraInfoEnhancement']['manualFeatureSelection'] = '@mn';
        
        fixDict['varianceThreshold'] = {}
        fixDict['varianceThreshold'] = '@vt';
        
        fixDict['removeOutliers'] = '@ro';
        
        fixDict['specificFeatureAgglomeration'] = {}
        fixDict['specificFeatureAgglomeration']['wardClustering'] = '@wc';
        
        fixDict['specificFeatureSelection'] = {}  
        fixDict['specificFeatureSelection']['univariateSelection'] = '@us';
        fixDict['specificFeatureSelection']['permutationSelector'] = '@ps';
        fixDict['specificFeatureSelection']['RFE'] = '@re';
        #fixDict['specificFeatureSelection']['treeBasedSelector'] = 'tb';
        
        subFixDict = {}
        
        subFixDict['filtering'] = {}
        subFixDict['filtering']['kernel'] = '&kl'
        subFixDict['filtering']['SavitzkyGolay'] = '&sg'
        subFixDict['filtering']['Gauss'] = '&gs'

        
        subFixDict['scatterCorrection'] = {}
        subFixDict['scatterCorrection']['l1'] = '&l1';  
        subFixDict['scatterCorrection']['l2'] = '&l2'; 
        subFixDict['scatterCorrection']['max'] = '&mx';
        subFixDict['scatterCorrection']['snv'] = '&sn'; 
        subFixDict['scatterCorrection']['msc'] = '&ms';
        
        subFixDict['standardisation'] = {}
        subFixDict['standardisation']['poissonscaling'] = '&po'; 
        subFixDict['standardisation']['paretoscaling'] = '&pa';
        subFixDict['standardisation']['meancentring'] = '&mc';
        subFixDict['standardisation']['unitscaling'] = '&us';
        
        txtStr = ''
        nStr = 0
        for p in fixDict:
            
            if hasattr(self, p):
                
                p1 = getattr(self, p)
                
                if p1.apply and  isinstance(fixDict[p], str) and fixDict[p][0] == '@':
                    
                    if nStr >= 1:
                                    
                        txtStr += '-' 
                                                    
                    txtStr += fixDict[p].replace('@','')
                    
                    nStr += 1
                
                if p1.apply:
                    
                    for px in fixDict[p]:
                        
                        if hasattr(p1, px):
                        
                            p2 = getattr(p1, px)
                            
                            if p2.apply and fixDict[p][px][0] == '@':
                                
                                if nStr >= 1:
                                    
                                    txtStr += '-' 
                                                                
                                txtStr += fixDict[p][px].replace('@','')
                                
                                nStr += 1
                                
                                if px in subFixDict:
                                    
                                    # Find the function of the preprocess
                                    
                                    pass
                                
                        
        '''
        
        sumPreProcessStr = '-'.join(sumTxtL)
                    
        FP = PathSplit(self.dataFPN)[0]
        
        FP = PathSplit(FP)[0]
        
        FP = PathSplit(FP)[0]
                
        subFP = '%s_%s-%s-%s' %(self.campaign.campaignShortId, self.firstBand, self.lastBand, self.resolutionBand)
        
        print (subFP)
        
        FP = PathJoin([FP, subFP])
 

        modelFP = PathJoin([FP,'mlmodel'])
        
        print (modelFP)
        
        if not PathExists(modelFP):

            MakeDirs(modelFP)

        modelresultFP = PathJoin([modelFP,'json'])

        if not PathExists(modelresultFP):

            MakeDirs(modelresultFP)

        pickleFP = PathJoin([modelFP,'pickle'])

        if not PathExists(pickleFP):

            MakeDirs(pickleFP)

        modelimageFP = PathJoin([modelFP,'images'])

        if not PathExists(modelimageFP):

            MakeDirs(modelimageFP)
         
        #reset bands if filtering is applied
        
        if self.spectraPreProcess.apply:
        
            if self.spectraPreProcess.filtering.apply:
            
                if self.spectraPreProcess.filtering.extraction.outputBandWidth and \
                    self.spectraPreProcess.filtering.extraction.mode != 'None' and \
                    self.spectraPreProcess.filtering.extraction.beginWaveLength < \
                    self.spectraPreProcess.filtering.extraction.endWaveLength:

                    self._BandExtraction(self.spectraPreProcess.filtering.extraction.mode,
                                    self.spectraPreProcess.filtering.extraction.beginWaveLength,
                                    self.spectraPreProcess.filtering.extraction.endWaveLength,
                                    self.spectraPreProcess.filtering.extraction.outputBandWidth)
                                   
        prefix = '%s_%s-%s-%s' %(self.output.prefix, self.firstBand, self.lastBand, self.resolutionBand)

        summaryJsonFN = '%s_%s_%s_summary.json' %(self.campaign.campaignId,prefix, sumPreProcessStr)

        self.summaryJsonFPN = PathJoin([modelresultFP,summaryJsonFN])
        
        print ('summaryJsonFN',summaryJsonFN)
        
        print ('self.summaryJsonFPN',self.summaryJsonFPN)

        regrJsonFN = '%s_%s_%s_results.json' %(self.campaign.campaignId,prefix, sumPreProcessStr)

        self.regrJsonFPN = PathJoin([modelresultFP,regrJsonFN])
        
        print ('self.regrJsonFPN',self.regrJsonFPN)

        paramJsonFN = '%s_%s_%s_params.json' %(self.campaign.campaignId,prefix, sumPreProcessStr)

        self.paramJsonFPN = PathJoin([modelresultFP,paramJsonFN])
        
        print ('self.paramJsonFPN',self.paramJsonFPN)

        self.imageFPND = {}; self.preProcessFPND = {}
        
        self.preProcessFPND['outliers'] = {}; 
        self.preProcessFPND['specificClustering'] = {}
        self.preProcessFPND['specificSelection'] = {}
        
        # Image files unrelated to targets and regressors 
        filterExtractFN = '%s_%s_%s_filterextract.png' %(self.campaign.campaignId,prefix, sumPreProcessStr)
        
        self.filterExtractPlotFPN = PathJoin([modelimageFP,filterExtractFN])
        
        self.ProducedImagesL.append(self.filterExtractPlotFPN)
        
        self.preProcessFPND['scatterCorrection'] = PathJoin([modelimageFP, '%s_%s_%s_scatter-correction.png' %(self.campaign.campaignId,prefix, sumPreProcessStr) ]) 
        self.preProcessFPND['standardisation'] = PathJoin([modelimageFP, '%s_%s_%s_standardisation.png' %(self.campaign.campaignId,prefix, sumPreProcessStr) ]) 
        self.preProcessFPND['derivatives'] = PathJoin([modelimageFP, '%s_%s_%s_derivatives.png' %(self.campaign.campaignId,prefix, sumPreProcessStr) ])
        self.preProcessFPND['decomposition'] = PathJoin([modelimageFP, '%s_%s_%s_decomposition.png' %(self.campaign.campaignId,prefix, sumPreProcessStr) ])
        self.preProcessFPND['varianceThreshold'] = PathJoin([modelimageFP, '%s_%s_%s_varaince-threshold-selector.png' %(self.campaign.campaignId,prefix, sumPreProcessStr)])
        
        self.ProducedImagesL.append(self.preProcessFPND['scatterCorrection'])
        self.ProducedImagesL.append(self.preProcessFPND['standardisation'])
        self.ProducedImagesL.append(self.preProcessFPND['derivatives'])
        self.ProducedImagesL.append(self.preProcessFPND['decomposition'])
        self.ProducedImagesL.append(self.preProcessFPND['varianceThreshold'])
        
        
        # the picke files save the regressor models for later use
        self.trainTestPickleFPND = {}

        self.KfoldPickleFPND = {}
        
        # loop over targetfeatures
        for targetFeature in self.paramD['targetFeatures']:
            
            self.preProcessFPND['outliers'][targetFeature] = PathJoin([modelimageFP, '%s_%s_%s_%s_specific-outliers.png' %(self.campaign.campaignId, prefix, sumPreProcessStr, targetFeature) ])
          
            self.ProducedImagesL.append(self.preProcessFPND['outliers'][targetFeature])
            
            self.preProcessFPND['specificClustering'][targetFeature] = PathJoin([modelimageFP, '%s_%s_%s_%s_specfic-cluster.png' %(self.campaign.campaignId, prefix, sumPreProcessStr, targetFeature) ])
            
            self.ProducedImagesL.append(self.preProcessFPND['specificClustering'][targetFeature])
            
            self.preProcessFPND['specificSelection'][targetFeature] = {}
            
            self.imageFPND[targetFeature] = {}

            self.trainTestPickleFPND[targetFeature] = {}; self.KfoldPickleFPND[targetFeature] = {}

            for regmodel in self.paramD['modelling']['regressionModels']:
                                
                self.preProcessFPND['specificSelection'][targetFeature][regmodel] = PathJoin([modelimageFP, '%s_%s_%s_%s_%s_specific-select.png' %(self.campaign.campaignId, prefix, sumPreProcessStr, targetFeature, regmodel)] )
                
                self.ProducedImagesL.append(self.preProcessFPND['specificSelection'][targetFeature][regmodel])
                
                trainTestPickleFN = '%s_%s_%s_%s_%s_trainTest.xsp'    %(self.campaign.campaignId, prefix, sumPreProcessStr, targetFeature, regmodel)
                
                KfoldPickleFN = '%s_%s_%s_%s_%s_Kfold.xsp'    %(self.campaign.campaignId, prefix, sumPreProcessStr, targetFeature, regmodel)

                self.trainTestPickleFPND[targetFeature][regmodel] = PathJoin([pickleFP, trainTestPickleFN])

                self.ProducedJsonsL.append(self.trainTestPickleFPND[targetFeature][regmodel])


                self.KfoldPickleFPND[targetFeature][regmodel] = PathJoin([pickleFP, KfoldPickleFN])

                self.ProducedJsonsL.append(self.KfoldPickleFPND[targetFeature][regmodel])
                
                self.imageFPND[targetFeature][regmodel] = {}

                if self.modelling.featureImportance.apply:

                    self.imageFPND[targetFeature][regmodel]['featureImportance'] = {}

                    imgFN = '%s_%s_%s_%s_%s_permut-imp.png'    %(self.campaign.campaignId, prefix, sumPreProcessStr, targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['featureImportance']['permutationImportance'] = PathJoin([modelimageFP, imgFN])

                    self.ProducedImagesL.append(self.imageFPND[targetFeature][regmodel]['featureImportance']['permutationImportance'])
       
       
                    imgFN = '%s_%s_%s_%s_%s_feat-imp.png'    %(self.campaign.campaignId, prefix, sumPreProcessStr,targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['featureImportance']['regressionImportance'] = PathJoin([modelimageFP, imgFN])
                    
                    self.ProducedImagesL.append(self.imageFPND[targetFeature][regmodel]['featureImportance']['regressionImportance'])
                    
                    imgFN = '%s_%s_%s_%s_%s_treebased-imp.png'    %(self.campaign.campaignId, prefix, sumPreProcessStr,targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['featureImportance']['treeBasedImportance'] = PathJoin([modelimageFP, imgFN])

                    self.ProducedImagesL.append(self.imageFPND[targetFeature][regmodel]['featureImportance']['treeBasedImportance'])
                    
                if self.modelling.modelTests.trainTest.apply:

                    imgFN = '%s_%s_%s_%s_%s_model_tt-result.png'    %(self.campaign.campaignId, prefix, sumPreProcessStr, targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['trainTest'] = PathJoin([modelimageFP, imgFN])

                    self.ProducedImagesL.append(self.imageFPND[targetFeature][regmodel]['trainTest'])


                if self.modelling.modelTests.Kfold.apply:

                    imgFN = '%s_%s_%s_%s_%s_model_kfold-result.png'    %(self.campaign.campaignId, prefix, sumPreProcessStr, targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['Kfold'] = PathJoin([modelimageFP, imgFN])

                    self.ProducedImagesL.append(self.imageFPND[targetFeature][regmodel]['Kfold'])
                    
            # Set multi row-image file names, per targetfeature
            imgFN = '%s_%s_%s_%s_multi-results.png' %(self.campaign.campaignId, prefix, sumPreProcessStr, targetFeature)

            self.imageFPND[targetFeature]['allmodels'] = PathJoin([modelimageFP, imgFN])

            self.ProducedImagesL.append(self.imageFPND[targetFeature]['allmodels'])

        for regmodel in self.paramD['modelling']['regressionModels']:

            self.imageFPND[regmodel] = {}

            # Set multi row-image file names, per regression model
            imgFN = '%s_%s_%s_%s_multi-results.png'    %(self.campaign.campaignId, prefix, sumPreProcessStr, regmodel)

            self.imageFPND[regmodel]['alltargets'] = PathJoin([modelimageFP, imgFN])
            
            self.ProducedImagesL.append(self.imageFPND[regmodel]['alltargets'])
            
        for img in self.ProducedImagesL:
            
            print (img)

    def _DumpJson(self):
        '''
        '''

        resultD = {}; summaryD = {}; self.multCompSummaryD = {}
        
        for targetFeature in self.targetFeatures:
        
            self.multCompSummaryD[targetFeature] = {}
        
        #resultD['targetFeatures'] = self.transformD
        
        #self.summaryD['targetFeatures'] = self.transformD

        resultD['originalInputColumns'] = len(self.originalColumns)
            
        if self.spectraInfoEnhancement.standardisation.apply:
            
            resultD['standardisation'] = True
            
        if self.spectraInfoEnhancement.decompose.pca.apply:
            
            resultD['pcaPreproc'] = True

        if self.removeOutliers.apply or self.generalFeatureSelection.apply:

            resultD['generalTweaks']= {}

            if self.removeOutliers.apply:

                resultD['generalTweaks']['removeOutliers'] = self.outliersRemovedD
                
            if self.generalFeatureSelection.apply:
                
                # either variance threshold or clustering - result saved in self.generalFeatureSelectedD
                resultD['generalTweaks']['generalFeatureSelection'] = self.generalFeatureSelectedD
                
                '''
                if self.generalFeatureSelection.varianceThreshold.apply:
    
                    resultD['generalTweaks']['varianceThreshold'] = self.varianceThresholdD
    
                if self.specificFeatureAgglomeration.apply:
    
                    resultD['generalTweaks']['featureAgglomeration'] = self.generalFeatureSelectedD
                '''
                
        if self.manualFeatureSelection.apply:

            resultD['manualFeatureSelection'] = True

        if self.specificFeatureSelection.apply:

            resultD['specificFeatureSelection'] = self.targetFeatureSelectedD

        if self.specificFeatureSelection.apply:

            resultD['modelFeatureSelection'] = self.specificFeatureSelectedD

        if self.modelling.featureImportance:

            resultD['featureImportance'] = self.modelFeatureImportanceD

        if self.modelling.hyperParameterTuning.apply:

            resultD['hyperParameterTuning'] = {}

            if self.modelling.hyperParameterTuning.randomTuning.apply:

                # Set the results from the hyperParameter Tuning
                resultD['hyperParameterTuning']['randomTuning'] = self.tunedHyperParamsD

            if self.modelling.hyperParameterTuning.exhaustiveTuning.apply:

                # Set the results from the hyperParameter Tuning
                resultD['hyperParameterTuning']['exhaustiveTuning'] = self.tunedHyperParamsD

        # Add the finally selected bands

        resultD['appliedModelingFeatures'] = self.finalFeatureLD

        # Add the final model results
        if self.modelling.modelTests.apply:

            resultD['modelResults'] = {}
            
            summaryD['modelResults'] = {}

            if self.modelling.modelTests.trainTest.apply:

                resultD['modelResults']['trainTest'] = self.trainTestResultD
                
                summaryD['modelResults']['trainTest'] = self.trainTestSummaryD
                
                for targetFeature in self.targetFeatures:
                
                    self.multCompSummaryD[targetFeature]['trainTest'] = self.trainTestSummaryD[targetFeature]
                
                    self.multCompSummaryD[targetFeature]['parameters'] = self.paramJsonFPN
                    
                    self.multCompSummaryD[targetFeature]['results'] = self.regrJsonFPN
                    
            if self.modelling.modelTests.Kfold.apply:

                resultD['modelResults']['Kfold'] = self.KfoldResultD
                
                summaryD['modelResults']['Kfold'] = self.KfoldSummaryD
                
                for targetFeature in self.targetFeatures:
                
                    self.multCompSummaryD[targetFeature]['Kfold'] = self.KfoldSummaryD[targetFeature]
                    
                    self.multCompSummaryD[targetFeature]['parameters'] = self.paramJsonFPN
                    
                    self.multCompSummaryD[targetFeature]['results'] = self.regrJsonFPN

        if self.verbose > 2:
            
            pp = pprint.PrettyPrinter(indent=2)
            pp.pprint(resultD)

 
        DumpAnyJson(resultD, self.regrJsonFPN  )

        DumpAnyJson(self.paramD, self.paramJsonFPN )
        
        DumpAnyJson(summaryD, self.summaryJsonFPN )
        
    def _PilotModeling(self,rootFP,sourcedatafolder,dstRootFP, multCompFig, multCompAxs):
        ''' Steer the sequence of processes for modeling spectra data in json format
        '''

        if len(self.targetFeatures) == 0:

            exit('Exiting - you have to set at least 1 target feature')

        if len(self.regressorModels) == 0:

            exit('Exiting - you have to set at least 1 regressor')
            
        # Set the source file names
        self._SetSrcFPNs(rootFP, dstRootFP, sourcedatafolder)
        
        # Get the band data as self.spectraDF
        self._GetBandData()

        # Get and add the abundance data
        self._GetAbundanceData()
        
        # set the destination file names
        self._SetDstFPNs()
        
        # Creata a list for all images
        self.figLibL = []

        self.hyperParamtxt = "hyper-param tuning: None"

        if self.modelling.hyperParameterTuning.apply:

            if self.modelling.hyperParameterTuning.exhaustiveTuning.apply:

                hyperParameterTuning = 'ExhaustiveTuning'

                self.tuningParamD = ReadModelJson(self.input.hyperParameterExhaustiveTuning)

                self.hyperParamtxt = "hyper-param tuning: grid search"

            elif self.modelling.hyperParameterTuning.randomTuning.apply:

                hyperParameterTuning = 'RandomTuning'

                self.tuningParamD = ReadModelJson(self.input.hyperParameterRandomTuning)

                self.hyperParamtxt = "hyper-param tuning: random"

            else:

                errorStr = 'Hyper parameter tuning requested, but no method assigned'

                exit(errorStr)

            self.hyperParams = Obj(self.tuningParamD )

        # Set the dictionaries to hold the model results
        self.trainTestResultD = {}; self.KfoldResultD  = {}; self.tunedHyperParamsD = {}
        self.generalFeatureSelectedD = {}; self.outliersRemovedD = {}; 
        self.targetFeatureSelectedD = {}
        self.specificFeatureSelectedD = {} 
        self.specificClusteringdD = {}
        self.modelFeatureImportanceD = {}
        self.finalFeatureLD = {}
        self.trainTestSummaryD = {}; self.KfoldSummaryD  = {};

        # Create the subDicts for all model + target related results
        for targetFeature in self.targetFeatures:

            self.tunedHyperParamsD[targetFeature] = {}; self.trainTestResultD[targetFeature] = {}
            self.KfoldResultD[targetFeature] = {}; self.specificFeatureSelectedD[targetFeature] = {}
            self.targetFeatureSelectedD[targetFeature] = {}; self.modelFeatureImportanceD[targetFeature] = {}
            self.finalFeatureLD[targetFeature] = {}
            self.trainTestSummaryD[targetFeature] = {}; self.KfoldSummaryD[targetFeature]  = {};
            self.specificClusteringdD[targetFeature] = {}

            for regModel in self.paramD['modelling']['regressionModels']:

                if self.paramD['modelling']['regressionModels'][regModel]['apply']:

                    self.trainTestResultD[targetFeature][regModel] = {}
                    self.KfoldResultD[targetFeature][regModel] = {}
                    self.specificFeatureSelectedD[targetFeature][regModel] = {}
                    self.modelFeatureImportanceD[targetFeature][regModel] = {}
                    self.finalFeatureLD[targetFeature][regModel] = {}
                    self.trainTestSummaryD[targetFeature][regModel] = {} 
                    self.KfoldSummaryD[targetFeature][regModel] = {}
                    
                    # Set the transformation to the output dict      
                    self.trainTestSummaryD[targetFeature]['transform'] = self.transformD[targetFeature]
                    
                    self.KfoldSummaryD[targetFeature]['transform'] = self.transformD[targetFeature]
                    
                    self.trainTestResultD[targetFeature]['transform'] = self.transformD[targetFeature]
 
                    self.KfoldResultD[targetFeature]['transform'] = self.transformD[targetFeature]
                        
                    if self.paramD['modelling']['hyperParameterTuning']['apply'] and self.tuningParamD[hyperParameterTuning][regModel]['apply']:

                        self.tunedHyperParamsD[targetFeature][regModel] = {}


        self.spectraInfoEnhancement.scatterCorrectiontxt, self.scalertxt, self.spectraInfoEnhancement.decompose.pcatxt, self.hyperParamtxt = 'NA','NA','NA','NA'
        
        self._SetSubPlots()
        
        # Filtering is applied separately for each spectrum and does not affect the distribution
        # between train and test datasets
        if self.spectraPreProcess.filtering.apply:
                
            self.filtertxt = self._FilterPrep()
                
        elif self.spectraPreProcess.multifiltering.apply:
                
            self.filtertxt = self._MultiFiltering()
            
        #if self.filtertxt != None:
            
        #    SNULLE
            
        # Extract a new pandas DF 
        # The DF used in the loop must have all rows with NaN for the target feature removed        
        self._ExtractDataFrameX()
        
        self._ResetDataFramesXY()
        
        # Scatter correction: except for Multiplicative Scatter Correction (MSC),
        # the correction is strictly per spectrum and could be done prior to the split 
        # MSC reguires a Meanspectra - that is returned from the function
        self.spectraInfoEnhancement.scatterCorrectiontxt = 'scatter correction: None'
                 
        if self.spectraInfoEnhancement.apply:
             
            if self.spectraInfoEnhancement.scatterCorrection.apply:
                
                scatterCorrectiontxt, self.X_train, self.X_test, self.scattCcorrMeanSpectra = \
                    ScatterCorrection(self.X_train, self.X_test, 
                    self.spectraInfoEnhancement.scatterCorrection, self.enhancementPlotLayout,
                    self.preProcessFPND['scatterCorrection'])
                
                self.spectraInfoEnhancement.scatterCorrectiontxt = 'scatter correction: %s' %(scatterCorrectiontxt)
                
            # standardisation can do meancentring, z-score normalisation, paretoscaling or poissionscaling
            # the standardisation is defined from the training data and applied to the testdata
            # 2 vectors are required for the standardisation; mean and variance (scaling) 
            scaler = 'None'
            
            if self.spectraInfoEnhancement.standardisation.apply:
                  
                #scaler, scalerMean, ScalerScale = self._Standardisation()
                self.X_train, self.X_test, scaler, scalerMean, ScalerScale = \
                    Standardisation(self.X_train, self.X_test,
                    self.spectraInfoEnhancement.standardisation, 
                    self.enhancementPlotLayout, 
                    self.preProcessFPND['standardisation'])
                
            if self.spectraInfoEnhancement.derivatives.apply:

                self.X_train, self.X_test, self.Xcolumns = Derivatives(self.X_train, 
                                self.X_test, self.spectraInfoEnhancement.derivatives.deriv,
                                self.spectraInfoEnhancement.derivatives.join, 
                                self.Xcolumns, self.enhancementPlotLayout, self.preProcessFPND['derivatives'])

            if self.spectraInfoEnhancement.decompose.apply:
                        
                if self.spectraInfoEnhancement.decompose.pca.apply:
                    
                    # PCA preprocess    
                    self.spectraInfoEnhancement.decompose.pcatxt = 'pca: None'
                    
                    self._PcaPreprocess()
                    
                    self.spectraInfoEnhancement.decompose.pcatxt = 'pca: %s comps' %(self.spectraInfoEnhancement.decompose.pca.n_components)
         
        # Any manual feature selection is applied to the original dataframe - i.e. affect all models the same
        if self.manualFeatureSelection.apply:

            self._ManualFeatureSelector()
                       
        if self.generalFeatureSelection.apply:

            if self.generalFeatureSelection.varianceThreshold.apply:
                
                self._VarianceSelector()
        
        # Loop over the target features to model
        for self.targetN, self.targetFeature in enumerate(self.targetFeatures):

            if self.verbose:

                infoStr = '\n    Target feature: %s' %(self.targetFeature)
                
                print (infoStr)

            self._ResetTargetFeature()
            # RemoveOutliers is applied per target feature
            
            if self.removeOutliers.apply:
    
                self._RemoveOutliers()
                
            # Covariate (X) Agglomeration
            if self.specificFeatureAgglomeration.apply:

                if self.specificFeatureAgglomeration.wardClustering.apply:
                                            
                    if self.specificFeatureAgglomeration.wardClustering.tuneWardClustering.apply:
                        
                        n_clusters = self._TuneWardClustering()

                    else:

                        n_clusters = self.specificFeatureAgglomeration.wardClustering.n_clusters
                    
                    self._WardClustering(n_clusters)
                
            self._SetTargetFeatureSymbol()
            
            #Loop over the defined models
            for self.regrN, self.regrModel in enumerate(self.regressorModels):
                
                print ('        regressor:', self.regrModel[0])
                
                #RESET COVARS
                self._ResetRegressorXyDF()
                
                # Specific feature selection - max one applied in each model
                if  self.specificFeatureSelection.apply:
                            
                    if self.specificFeatureSelection.univariateSelection.apply:
                        
                        if self.specificFeatureSelection.univariateSelection.SelectKBest.apply:
                        
                            self._UnivariateSelector()
                                                  
                    elif self.specificFeatureSelection.permutationSelector.apply:
    
                        self._PermutationSelector()
    
                    elif self.specificFeatureSelection.RFE.apply:
    
                        if self.regrModel[0] in ['KnnRegr','MLP', 'Cubist']:
                                
                            self._PermutationSelector()
    
                        else:

                            self._RFESelector()
                                
                    elif self.specificFeatureSelection.treeBasedSelector.apply:
    
                        self._TreeBasedFeatureSelection()

                if self.modelling.featureImportance.apply:

                    self._FeatureImportance(multCompAxs)

                if self.modelling.hyperParameterTuning.apply:

                    if self.modelling.hyperParameterTuning.exhaustiveTuning.apply:

                        self._ExhaustiveTuning()

                    elif self.modelling.hyperParameterTuning.randomTuning.apply:

                        self._RandomTuning()

                    # Reset the regressor with the optimized hyperparameter tuning
                    #NOTDONE
                    # Set the regressor models to apply
                    self._RegModelSelectSet()

                if self.verbose > 2:

                    # Report the regressor model settings (hyper parameters)
                    self._ReportRegModelParams()

                #if isinstance(self.y_test_r,pd.DataFrame):
                     
                #    exit ('obs is a dataframe, must be a dataseries')

                columns = [item for item in self.X_train_R.columns]
                
                # unchanged columns from the start as lists
                if type(columns) is list:
                    
                    self.finalFeatureLD[self.targetFeature][self.regrModel[0]] = columns
                
                else: # otherwise not

                    self.finalFeatureLD[self.targetFeature][self.regrModel[0]] = columns.tolist()
                                
                if self.modelling.modelTests.apply:

                    if self.modelling.modelTests.trainTest.apply:

                        self._RegrModTrainTest(multCompAxs)

                    if self.modelling.modelTests.Kfold.apply:

                        self._RegrModKFold(multCompAxs)
                        
        #if self.modelPlot.rows.screenShow:

        #    plt.show()

        if self.modelPlot.rows.savePng:

            if self.modelPlot.rows.targetFeatures.apply:

                for regModel in self.paramD['modelling']['regressionModels']:

                    if self.paramD['modelling']['regressionModels'][regModel]['apply']:

                        self.columnFig[regModel].savefig(self.imageFPND[regModel]['alltargets'])

            if self.modelPlot.rows.regressionModels.apply:

                for targetFeature in self.targetFeatures:

                    self.columnFig[targetFeature].savefig(self.imageFPND[targetFeature]['allmodels'])
                    
        for regModel in self.paramD['modelling']['regressionModels']:

            if self.paramD['modelling']['regressionModels'][regModel]['apply']:
        
                #plt.close(fig=self.columnFig[regModel])
                ClosePlot(self.columnFig[regModel])
                
        for targetFeature in self.targetFeatures:
            
            if self.modelPlot.rows.regressionModels.apply:

                #plt.close(fig=self.columnFig[targetFeature])
                ClosePlot(self.columnFig[targetFeature])
        
        self._DumpJson()