'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

import numpy as np

from numbers import Integral

from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal import savgol_filter

from sklearn import model_selection

from sklearn.decomposition import PCA

# Outlier detection
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from support import PdDataFrame, PdConcat, DeepCopy

from plotChart import PlotFilterExtract, PlotPCA, PlotOutlierDetect


class MLPreProcess:
    '''Machinelearning Pre processes object mode [TO BE MOVED TO STAND ALONE IF TIMEM PERMITS]
    '''
    def __init__(self):
        '''Initiate
        '''

        pass
    
    def _ExtractDataFrameX(self):
        ''' Extract the original dataframe to X (covariate) array and y (predict) column
        '''

        # define the list of covariates to use
        #columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]
        columnsX = self.spectraDF.columns

        columnsY = self.abundanceDf.columns
        
        frames = [self.spectraDF,self.abundanceDf]
    
        spectraDF = PdConcat(frames, 1)
                
        self.Xall = spectraDF[columnsX]
        
        columns = self.Xall.columns
        
        if '.' in columns[0]:
            
            XcolumnsNum = [float(item) for item in columns]
            
            XcolumnsStr = ["{0:.1f}".format(item) for item in XcolumnsNum]
            
        else:
            
            XcolumnsNum = [int(item) for item in columns]
            
            XcolumnsStr = list(map(str, XcolumnsNum))
               
        self.Xcolumns = dict(zip(XcolumnsStr,XcolumnsNum))
                
        self.Yall = spectraDF[columnsY]
        
        #Split the data into training and test subsets
        self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(self.Xall, self.Yall, test_size=self.datasetSplit.testSize)
                  
    def _ResetDataFramesXY(self):
        '''
        '''
        
        Xcolumns = list(self.Xall.keys())
        
        Ycolumns = list(self.Yall.keys())
        
        xtrain = np.array(self.X_train)
                    
        xtest = np.array(self.X_test)
                            
        self.X_train = PdDataFrame(xtrain, Xcolumns)
                    
        self.X_test = PdDataFrame(xtest, Xcolumns)
        
        ytrain = np.array(self.Y_train)
                    
        ytest = np.array(self.Y_test)
                            
        self.Y_train = PdDataFrame(ytrain, Ycolumns)
                    
        self.Y_test = PdDataFrame(ytest, Ycolumns)
        
    def _ExtractDataFrameTarget(self):
        ''' Extract the original dataframe to X (covariate) array and y (predict) column
        '''

        # Extract the target feature
        self.y = self.abundanceDf[self.targetFeature]

        # Append the target array to the self.spectraDF dataframe
        self.spectraDF['target'] = self.y

        # define the list of covariates to use
        #columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]
        columnsX = [item for item in self.spectraDF.columns]

        # extract all the covariate columns as Xll
        self.Xall = self.spectraDF[columnsX]
        
        # Drop the added target column from the dataframe
        self.spectraDF = self.spectraDF.drop('target', axis=1)

        # Remove all samples where the targetfeature is NaN
        self.Xall = self.Xall[~np.isnan(self.Xall).any(axis=1)]
        
        # Drop the added target column from self.X
        self.Xall = self.Xall.drop('target', axis=1)

        # Then also delete NaN from self.y
        self.y = self.y[~np.isnan(self.y)]
        
        # Remove all non-finite values   
        self.Xall = self.Xall[np.isfinite(self.y)] 
        
        self.y = self.y[np.isfinite(self.y)] 
        
    def _BandExtraction(self, extractionMode, beginWL, endWL, outputBandWidth):
        '''
        '''
        # Define the output wavelengths
        if extractionMode == 'noendpoints':
            
            outputWls = np.arange(beginWL+outputBandWidth, endWL, outputBandWidth)
            
        else:
                
            outputWls = np.arange(beginWL, endWL, outputBandWidth)
            
        print ('beginWL',beginWL)
        
        print ('endWL',endWL)
        
        print ('outputBandWidthL',outputBandWidth)
        
        print (self.firstBand)
        
        print (self.lastBand)
        
        print (self.resolutionBand)
        
        print ('outputWls',outputWls)
            
        self.firstBand = str(int(round((outputWls[0]))))
    
        self.lastBand = str(int(round((outputWls[len(outputWls)-1]))))
            
        self.resolutionBand = int(round(( outputWls[len(outputWls)-1]-outputWls[0] )/(len(outputWls)-1)))

        print (self.firstBand)
        
        print (self.lastBand)
        
        print (self.resolutionBand)
        
        #SNULLE
        
        return (outputWls)
        
     
    def _Filtering(self, extractionMode, beginWL, endWL, outputBandWidth, wlArr, halfwlstep):
        """ Filtering the spectral signal
        """
                
        filtertxt = 'None'
        
        # Get the input spectra
        columnsX = [item for item in self.spectraDF.columns]
                
        outputWls = [float(item) for item in columnsX]
       
        # extract the covariate columns as X
        X = self.spectraDF[columnsX]
        
        # If the filter is a kernel (including moving average)    
        if self.spectraPreProcess.filtering.movingaverage.kernel:
 
            sumkernel =  np.asarray(self.spectraPreProcess.filtering.movingaverage.kernel).sum()
            
            normkernel = self.spectraPreProcess.filtering.movingaverage.kernel/sumkernel
            
            X1 = convolve1d(X, normkernel, axis=-1, mode=self.spectraPreProcess.filtering.movingaverage.mode)
            
            filtertxt = 'kernel filter'
                 
        # If Gaussian filter
        elif self.spectraPreProcess.filtering.Gauss.sigma:
            
            sigma = self.spectraPreProcess.filtering.Gauss.sigma / ( (wlArr[len(wlArr)-1] - wlArr[0])/(len(wlArr)-1) )
                                    
            X1 = gaussian_filter1d(X, sigma, axis=-1, mode=self.spectraPreProcess.filtering.Gauss.mode)
                           
            filtertxt = 'Gaussian filter'
             
        # If Savitzky Golay filter
        elif self.spectraPreProcess.filtering.SavitzkyGolay.window_length:
            
            X1 = savgol_filter(X, window_length=self.spectraPreProcess.filtering.SavitzkyGolay.window_length, 
                               polyorder= self.spectraPreProcess.filtering.SavitzkyGolay.polyorder,
                               axis=-1,
                               mode=self.spectraPreProcess.filtering.SavitzkyGolay.mode)
                        
            filtertxt = 'Savitzky-Golay filter'
            
        else:
            
            X1 = X
                        
        if extractionMode.lower() in ['none', 'no'] or beginWL >= endWL:
            
            "No extraction"
            
            pass
            
        else:
            
            # Define the output wavelengths
            if extractionMode == 'noendpoints':
                
                outputWls = np.arange(beginWL+outputBandWidth, endWL, outputBandWidth)
                
            else:
                    
                outputWls = np.arange(beginWL, endWL, outputBandWidth)
                
            self.firstBand = str(int(round((outputWls[0]))))
        
            self.lastBand = str(int(round((outputWls[len(outputWls)-1]))))
                
            self.resolutionBand = int(round(( outputWls[len(outputWls)-1]-outputWls[0] )/(len(outputWls)-1)))
                                        
            xDF = PdDataFrame(X1, columnsX)
            
            arrL = []
            
            for row in xDF.values:
  
                spectraA = np.interp(outputWls, wlArr, row,
                            left=beginWL-halfwlstep,
                            right=endWL+halfwlstep)
                
                arrL.append(spectraA)
                
            X1 = np.asarray(arrL)
                                       
        return filtertxt, outputWls, X1
    
    def _FilterPrep(self):
        '''
        '''
        # Create an empty copy of the spectra DataFrame
        originalDF = self.spectraDF.copy()
                
        columnsNum = list(self.columns.values())
          
        # Convert bands to array)
        wlArr = np.asarray(columnsNum)
        
        halfwlstep = self.spectraPreProcess.filtering.extraction.outputBandWidth/2
        
        extractionMode = self.spectraPreProcess.filtering.extraction.mode
        
        beginWL = self.spectraPreProcess.filtering.extraction.beginWaveLength
        
        endWL = self.spectraPreProcess.filtering.extraction.endWaveLength-1
        
        outputBandWidth = self.spectraPreProcess.filtering.extraction.outputBandWidth
        
        # Run the filtering
        filtertxt, outputWls, X1 = self._Filtering(extractionMode, beginWL, endWL, outputBandWidth, wlArr, halfwlstep)
        
        if isinstance(outputWls[0], Integral):
            
            #outputWls = ["{d}".format(item) for item in outputWls]
            outputWlsStr = list(map(str, outputWls))

        else:
            
            outputWlsStr = ["{0:.1f}".format(item) for item in outputWls]
            
        # Reset self.columns
        self.columns = dict(zip(outputWlsStr,outputWls))
        
        self.spectraDF = PdDataFrame(X1, outputWlsStr)
        
        if self.enhancementPlotLayout.filterExtraction.apply:
            
            PlotFilterExtract( self.enhancementPlotLayout, filtertxt, originalDF, self.spectraDF, self.filterExtractPlotFPN)
            
        self.spectraDF = PdDataFrame(X1, outputWlsStr)
                    
        return filtertxt
                
    def _MultiFiltering(self):
        ''' Applies different filters over different parts of the spectra
        '''
        
        # Create an empty copy of the spectra DataFrame
        newSpectraDF = self.spectraDF[[]].copy()
             
        # Extract the columns bands) as floating wavelenhts   
        columnsX = [float(item) for item in self.spectraDF.columns]
        
        # Cnovert bands to array)
        wlArr = np.asarray(columnsX)
        
        outPutWlStr = []
        
        outPutWlNum = []
        
        # Loop over the wavelength regions defined for filtering
        for r, rang in enumerate(self.spectraPreProcess.multifiltering.beginWaveLength):
            
            # Deep copy the spectra DataFrame
            copySpectraDF = DeepCopy(self.spectraDF)

            # Set all the filteroptions to False before starting each loop
            self.spectraPreProcess.filtering.movingaverage.kernel = []
            self.spectraPreProcess.filtering.Gauss.sigma = 0
            self.spectraPreProcess.filtering.SavitzkyGolay.window_length = 0
                        
            if self.spectraPreProcess.multifiltering.movingaverage.kernel[r]:
                
                self.spectraPreProcess.filtering.movingaverage.kernel = self.spectraPreProcess.multifiltering.movingaverage.kernel[r]
                
            elif self.spectraPreProcess.multifiltering.SavitzkyGolay.window_length[r]:
                                
                self.spectraPreProcess.filtering.SavitzkyGolay.window_length = self.spectraPreProcess.multifiltering.SavitzkyGolay.window_length[r]
                
            elif self.spectraPreProcess.multifiltering.Gauss.sigma[r]:
                                
                self.spectraPreProcess.filtering.Gauss.sigma = self.spectraPreProcess.multifiltering.Gauss.sigma[r]
             
            halfwlstep = self.spectraPreProcess.multifiltering.outputBandWidth[r]/2
            extractionMode = self.spectraPreProcess.filtering.extraction.mode
            beginWL = self.spectraPreProcess.multifiltering.beginWaveLength[r]
            endWL = self.spectraPreProcess.multifiltering.endWaveLength[r]-1
            outputBandWidth = self.spectraPreProcess.multifiltering.outputBandWidth[r]
               
            # Run the filtering
            filtertxt, outputWls, X1 = self._Filtering(extractionMode, beginWL, endWL, outputBandWidth, wlArr, halfwlstep)
            
            if isinstance(outputWls[0], Integral):
            
                #outputWls = ["{d}".format(item) for item in outputWls]
                outputWlS = list(map(str, outputWls))
    
            else:
                    
                outputWlS = ["{0:.1f}".format(item) for item in outputWls]
                    
            outPutWlNum.extend(outputWls)
            
            outPutWlStr.extend(outputWlS)
        
            # Add filtered+extrad range with columns as strings
            newSpectraDF[ outputWlS ] = X1
           
        # reset columns 
        self.columns = dict(zip(outPutWlStr, outPutWlNum))
                            
        self.spectraDF = DeepCopy(copySpectraDF)
            
        if self.enhancementPlotLayout.filterExtraction.apply:
            
            PlotFilterExtract( self.enhancementPlotLayout, filtertxt, self.spectraDF, newSpectraDF, self.filterExtractPlotFPN)

        # Set the fitlered spectra to spectraDF
        self.spectraDF = newSpectraDF

        return 'multi'   
      
    def _ResetRegressorXyDF(self):
        '''
        '''

        self.X_train_R =  DeepCopy(self.X_train_T)
        self.X_test_R =  DeepCopy(self.X_test_T)
        self.y_train_r =  DeepCopy(self.y_train_t)
        self.y_test_r =  DeepCopy(self.y_test_t)
        
        self.X_columns_R = DeepCopy(self.X_columns_T)
        
        # Reomve all NoN and infinity
        self._ResetXY_R()
         
    def _PcaPreprocess(self):
        """ See https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/tutorial/astronomy/dimensionality_reduction.html
            for faster (random) algorithm
        """ 
        
        Xcolumns = [item for item in self.X_train]
                
        if (len(Xcolumns) < self.spectraInfoEnhancement.decompose.pca.n_components):
            
            exit('EXITING - the number of surviving bands are less than the PCA components requested')
                
        # set the new covariate columns:
        columnsStr = []
        
        columnsInt = []
        
        for i in range(self.spectraInfoEnhancement.decompose.pca.n_components):
            if i > 100:
                x = 'pc-%s' %(i)
            elif i > 10:
                x = 'pc-0%s' %(i)
            else:
                x = 'pc-00%s' %(i) 
                
            columnsInt.append(i)
            columnsStr.append(x)
        
        self.columns = dict(zip(columnsStr, columnsInt))
        
        pca = PCA(n_components=self.spectraInfoEnhancement.decompose.pca.n_components)
                
        X_pc = pca.fit_transform(self.X_train)
         
        self.pcaComponents = pca.components_
        
        
        X_train_transformed = pca.fit_transform(self.X_train)
        
        X_test_transformed = pca.transform(self.X_test)
                
        PlotPCA(self.enhancementPlotLayout, self.preProcessFPND['decomposition'], 'pca', self.Xcolumns,
                                self.X_train, self.X_test, 
                                PdDataFrame(X_train_transformed, columnsStr), 
                                PdDataFrame(X_test_transformed, columnsStr))

        self.X_train = PdDataFrame(X_train_transformed, columnsStr)
        
        self.X_test = PdDataFrame(X_test_transformed, columnsStr)
               
        self.Xcolumns = dict(zip(columnsStr, columnsInt))
        
    def _RemoveOutliers(self):
        """
        """
        
        if self.removeOutliers.contamination == 0:
            
            return
        
        def ExtractCovars(columnsX):
            '''
            '''
            
            extractTarget = False
            
            if 'target' in columnsX:
                
                targetIndex = columnsX.index('target')
                
                columnsX.pop(targetIndex)

                extractTarget = True
                
            xyTrainDF = PdDataFrame(self.X_train_T, columnsX)
            
            xyTrainDF.reset_index()
            
            xyTestDF = PdDataFrame(self.X_test_T, columnsX)
            
            xyTestDF.reset_index()
            
            if extractTarget:
            
                xyTrainDF['target'] = self.y_train_t

                xyTestDF['target'] = self.y_test_t
                
                columnsX.append('target')
            
            return (xyTrainDF, xyTestDF)
                  
        def RemoveOutliers(Xtrain, Xtest,  columnsX):
            '''
            '''
            
            # extract the covariate columns as X
            X = Xtrain[columnsX]
    
            iniTrainSamples = X.shape[0]
    
            if self.removeOutliers.detector.lower() in ['iforest','isolationforest']:
    
                outlierDetector = IsolationForest(contamination=self.removeOutliers.contamination)
                
            elif self.removeOutliers.detector.lower() in ['ee','eenvelope','ellipticenvelope']:
    
                outlierDetector = EllipticEnvelope(contamination=self.removeOutliers.contamination)
    
            elif self.removeOutliers.detector.lower() in ['lof','lofactor','localoutlierfactor']:
    
                outlierDetector = LocalOutlierFactor(contamination=self.removeOutliers.contamination)
    
            elif self.removeOutliers.detector.lower() in ['1csvm','1c-svm','oneclasssvm']:
    
                outlierDetector = OneClassSVM(nu=self.removeOutliers.contamination)
    
            else:
    
                exit('unknown outlier detector')
    
            # The warning "X does not have valid feature names" is issued, but it is a bug and will go in next version
            #yhat = outlierDetector.fit_predict(X)
            
            outlierFit = outlierDetector.fit(X)
            
            yhat = outlierFit.predict(X)
    
            # select all rows that are inliers           
            XtrainInliers = Xtrain[ yhat==1 ]
            
            # select all rows that are outliers
            XtrainOutliers = Xtrain[ yhat==-1 ]
            
            # Remove samples with outliers from the X and y dataset
            self.X_train_T = self.X_train_T[ yhat==1 ]
            
            self.y_train_t = self.y_train_t[ yhat==1 ]
                                    
            postTrainSamples = self.X_train_T.shape[0]
                        
            # Run the test data with the same detector
            # extract the covariate columns as X
            X = Xtest[columnsX]
            
            iniTestSamples = X.shape[0]
            
            yhat = outlierFit.predict(X)
            
            XtestInliers = X[ yhat==1 ]
            
            XtestOutliers = X[ yhat==-1 ]
            
            self.X_test_T = self.X_test_T[ yhat==1 ]
            
            self.y_test_t = self.y_test_t[ yhat==1 ]

            postTestSamples = self.X_test_T.shape[0]
            
            self.nTrainOutliers = iniTrainSamples - postTrainSamples
            self.nTestOutliers = iniTestSamples - postTestSamples
    
            self.outliersRemovedD['method'] = self.removeOutliers.detector
            self.outliersRemovedD['nOutliersRemoved'] = self.nTrainOutliers+self.nTestOutliers
    
            self.outlierTxt = '%s (%s) outliers removed ' %(self.nTrainOutliers,self.nTestOutliers )
    
            outlierTxt = '%s (%s) outliers removed' %(self.nTrainOutliers,self.nTestOutliers)
    
            if self.verbose:
    
                print ('            ',outlierTxt)
             
            if len(columnsX) == 2:
                
                # Use the orginal training data for defining the plot boundary between inliers and outliers
                X = Xtrain[columnsX]
                
                PlotOutlierDetect(self.enhancementPlotLayout, self.preProcessFPND['outliers'][self.targetFeature],
                                XtrainInliers, XtrainOutliers, XtestInliers, XtestOutliers,  
                                postTrainSamples, self.nTrainOutliers, postTestSamples, self.nTestOutliers,
                                self.removeOutliers.detector, columnsX, self.targetFeature, outlierFit, X)
                      
        ''' Main def'''
                
        columns = [item for item in self.X_train_T.columns]
            
        if len(self.removeOutliers.covarList) == 0:
            
            return 
          
        if self.removeOutliers.covarList[0] == '*':
        
            columnsX = [item for item in self.X_train_T.columns]
        
        else:
            
            columnsX = self.removeOutliers.covarList
            
        if self.removeOutliers.includeTarget:
            
            columnsX.append('target')
            
        for item in columnsX:
                
            if item != 'target' and item not in columns:
                    
                exitStr = 'EXITING - item %s missing in removeOutliers CovarL' %(item)
                
                exitStr += '\n    Available covars = %s' %(', '.join(columns) )
                
                exit(exitStr)
                            
        xyTrainDF, xyTestDF = ExtractCovars(columnsX)
                            
        RemoveOutliers(xyTrainDF, xyTestDF, columnsX)
        
        self._ResetXY_T()
                          
    def _CheckParams(self, jsonProcessFN):
        ''' Check parameters
        '''
        
        if not hasattr(self,'targetFeatures'):
            exitStr = 'Exiting: the modelling process file %s\n    has not targetFeature' %(jsonProcessFN)
            exit(exitStr)