'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

import numpy as np

from support import DeepCopy,  PdDataFrame, PdConcat

from sciKitLearn import SklearnStandardScalerFitTransform, SklearnStandardScalerFit,\
    SklearnStandardScalerTransform, SklearnNormalizeDf

from plotChart import PlotScatterCorr, PlotStandardisation, PlotDerivatives

def ScatterCorrectDF(trainDF, testDF, scattercorrection, columns):
    """ Scatter correction for spectral signals

        :returns: scatter corrected spectra
        :rtype: pandas dataframe
    """
    
    normD = {}
    
    normD['l1'] = { 'norm':'l1', 'label':'L1norm'}
    normD['l2'] = {'norm':'l2', 'label':'L2norm'}
    normD['max'] = {'norm':'max', 'label':'maxnorm'}
    normD['snv'] = {'norm':'max', 'label':'SNV'}
    normD['msc'] = {'norm':'max', 'label':'MSC'}
    
    M1 = [None, None]
    
    firstCorrTrainDf = None
                
    firstCorrTestDf = None
        
    scattCorrTxt = 'None'
    
    corrTxtL = []
    
    for singleScattCorr in scattercorrection.singles:
        
        if not singleScattCorr.lower() in normD:
            
            exitStr = 'EXITING - unrecognized scatter correction method: %s' %(singleScattCorr)
            
            exit(exitStr) 
            
        scattCorrTxt = normD[singleScattCorr.lower()]['label']
        
        corrTxtL.append(scattCorrTxt)
        
        #trainDFD[scatcorr] = {'label': normD[scatcorr]['label']}
        #testDFD[scatcorr] = {'label': normD[scatcorr]['label']}
        
        if singleScattCorr.lower() in ['l1','l2','max']:
            
            X1, N1 = SklearnNormalizeDf(trainDF, singleScattCorr.lower(), True) 
            
            X2, N2 = SklearnNormalizeDf(testDF, singleScattCorr.lower(), True) 

            trainDF = PdDataFrame(X1, columns)
            testDF = PdDataFrame(X2, columns)
            
        elif singleScattCorr.lower() == 'snv':
            
            X1 = np.array(trainDF[columns])
            
            X1 = snv(X1)
            
            X2 = np.array(testDF[columns])
            
            X2 = snv(X2)
            
            trainDF = PdDataFrame(X1, columns)
            testDF = PdDataFrame(X2, columns)

        elif singleScattCorr.lower() == 'msc':

            X1 = np.array(trainDF[columns])
            
            X1, M1[0] = msc(X1)
            
            X2 = np.array(testDF[columns])
            
            X2, M2 = msc(X2,M1[0])

            trainDF = PdDataFrame(X1, columns)
            testDF = PdDataFrame(X2, columns)
            
        else:
            
            exitStr = 'EXITING - unrecognized scatter correction method: %s' %(singleScattCorr)
            
            exit(exitStr)
            
    for s, dualScattCorr in enumerate(scattercorrection.duals):
        
        if not dualScattCorr in normD:
            
            exitStr = 'EXITING - unrecognized scatter correction method: %s' %(dualScattCorr)
            
            exit(exitStr)
        
        if s == 0:
            
            scattCorrTxt = normD[dualScattCorr]['label']
        
        else:
            
            scattCorrTxt += '+%s' %(normD[dualScattCorr]['label'])
   
        corrTxtL.append(scattCorrTxt)
        
        dualTrainDF = DeepCopy(trainDF)
        dualTestDF = DeepCopy(testDF)
                    
        print ('scatcorr',dualScattCorr)

        if dualScattCorr in ['l1','l2','max']:
            
            X1 = np.array(dualTrainDF[columns])
            
            X1 = SklearnNormalizeDf(X1, dualScattCorr.lower() ) 
                        
            X2 = np.array(dualTestDF[columns])
            
            X2 = SklearnNormalizeDf(X2, dualScattCorr.lower() ) 
          
        elif dualScattCorr  == 'snv':
            
            X1 = np.array(dualTrainDF[columns])
            
            X1 = snv(X1)
            
            X2 = np.array(dualTestDF[columns])
            
            X2 = snv(X2)
                  
        elif dualScattCorr == 'msc':
            
            X1 = np.array(dualTrainDF[columns])
            
            X1, M1[s] = msc(X1)
            
            X2 = np.array(dualTestDF[columns])
            
            X2, M2 = msc(X2,M1[s])
            
        else:
            
            exitStr = 'EXITING - unrecognized scatter correction method: %s' %(dualScattCorr)
            
            exit(exitStr)
            
        dualTrainDF = PdDataFrame(X1, columns)
            
        dualTestDF = PdDataFrame(X2, columns)
            
        if s == 0:
    
            firstCorrTrainDf = DeepCopy(dualTrainDF)
            
            firstCorrTestDf = DeepCopy(dualTestDF)
                
        trainDF= dualTrainDF
        
        testDF = dualTestDF
            
    return scattCorrTxt, corrTxtL, trainDF, testDF, firstCorrTrainDf, firstCorrTestDf, M1
                      
def ScatterCorrection(trainDF, testDF, scattercorrection, plotLayout, plotFPN):
    """ Scatter correction for spectral signals

        :returns: organised spectral derivates
        :rtype: pandas dataframe
    """
    
    columns = [item for item in trainDF]
        
    origTrainDF = DeepCopy(trainDF)
    
    origTestDF = DeepCopy(testDF)
    
    scattCorrTxt, corrTxtL, trainDF, testDF, firstCorrTrainDF, firstCorrTestDF, scatCorrMeanSpectraL = ScatterCorrectDF(trainDF, testDF, scattercorrection, columns)
    
    #NoN can develop and must be removed

    PlotScatterCorr(plotLayout, plotFPN, corrTxtL, columns, origTrainDF, origTestDF, trainDF, testDF, firstCorrTrainDF, firstCorrTestDF)

    return scattCorrTxt, trainDF, testDF,  scatCorrMeanSpectraL

def snv(input_data):
    ''' Perform Multiplicative scatter correction
    copied 20240311: https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/
    '''
  
    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(input_data)
    
    for i in range(input_data.shape[0]):
 
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
 
    return output_data

def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction
    copied 20240311: https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/
    '''
 
    # mean centre correction
    
    for i in range(input_data.shape[0]):
        
        input_data[i,:] -= input_data[i,:].mean()
 
    # Get the reference spectrum. If not given, estimate it from the mean    
    if reference is None:    
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference
 
    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 
 
    return (data_msc, ref)

def Standardisation(X_train, X_test, standardisation, plotLayout, plotFPN):
        """
        """
        
        scalertxt = 'None'
        
        origTrain = DeepCopy(X_train)
    
        origTest = DeepCopy(X_test)
        
        columns = [item for item in X_train.columns]
        
        # extract the covariate columns as X
        X1 = X_train[columns]
        
        X2 = X_test[columns]
        
        scaler = SklearnStandardScalerFit(X1)
                
        arrayLength = scaler.var_.shape[0]
        
        if standardisation.paretoscaling:
            #  No meancentring, scales each variable by the square root of the standard deviation

            # remove meancentring            
            scaler.mean_ = np.zeros(arrayLength)
                      
            # set var_ to its own square root
            scaler.var_ = np.sqrt(scaler.var_)
            
            # set scaling to the sqrt of the std
            scaler.scale_ = np.sqrt(scaler.var_)
                        
            X1A = scaler.transform(X1)  
            
            X2A = scaler.transform(X2) 

            scalertxt = 'Pareto'          
                
        elif standardisation.poissonscaling:
            # No meancentring, scales each variable by the square root of the mean of the variable
            
            scaler = SklearnStandardScalerFit(with_mean=False).fit(X1)
            
            # Set var_ to mean_
            scaler.var_ = scaler.mean_
            
            # set scaler to sqrt of mean_ (var_)
            scaler.scale_ = np.sqrt(scaler.var_)
            
            X1A = scaler.transform(X1) 
            
            X2A = scaler.transform(X2) 
            
            scalertxt = 'Poisson' 
            
        elif standardisation.meancentring:
            
            if standardisation.unitscaling:
                
                # This is a classical autoscaling or z-score normalisation
                X1A = SklearnStandardScalerFitTransform(X1)
                X2A = SklearnStandardScalerFitTransform(X2)
                
                scalertxt = 'Z-score' 
                   
            else:
                
                # This is meancentring
                X1A =SklearnStandardScalerFitTransform(X1,True,False)
                
                X2A = SklearnStandardScalerFitTransform(X2,True,False)
                
                scalertxt = 'meancentring' 
        
        elif standardisation.unitscaling:
            
            X1A = SklearnStandardScalerFitTransform(X1,False,True)
                
            X2A = SklearnStandardScalerFitTransform(X2,False,False)
            
            scalertxt = 'deviation'
        
        else:
            
            standardisation.apply = False
            
            exit('EXITING - standardisation.apply is set to true but no method defined\n either set standardisation.apply to false or pick a method')
            
        if standardisation.apply:
            # Reset the train and test dataframes                
            X_train = PdDataFrame(X1A, columns)
            X_test = PdDataFrame(X2A, columns)
                  
            PlotStandardisation(plotLayout, plotFPN, scalertxt, columns,
                                origTrain, origTest, X_train, X_test)
            
     
        return X_train, X_test, scalertxt, scaler.mean_, scaler.scale_
     
def Derivatives(X_train, X_test, deriv, joinDerivative, Xcolumns, plotLayout, plotFPN):
    '''
    '''

    if deriv < 1 or deriv > 2:
        
        return
        
    columnsStr = list(Xcolumns.keys())
    
    columnsNum = list(Xcolumns.values())
    
    for d in range(deriv):
        
        # Get the derivatives
        X_train_derivative = X_train.diff(axis=1, periods=1)

        # Drop the first column as it will have only NaN
        X_train_derivative = X_train_derivative.drop(columnsStr[0], axis=1)

        # Create the derivative columns
        derivativeColumnsNum = [ (columnsNum[i-1]+columnsNum[i])/2 for i in range(len(columnsNum)) if i > 0]
    
        X_train = X_train_derivative
    # if first derivate denote band dXXX, where XXX is the central wvalenght of the derivative
    # if 2nd derivative denote band ddXXX 
    # Check the numeric format of derivativeColumnsNum:
    
    if deriv == 1:
        
        dStr = 'd'
        
    else:
        
        dStr = 'dd'
        
    allIntegers = True
    
    for item in derivativeColumnsNum:

        if item % int(item) != 0:
            
            allIntegers = False
            
            break
       
    if allIntegers:
        
        derivativeColumnsNum = [int(item) for item in derivativeColumnsNum]
        
        derivativeColumnsStr = ['%s%s' % (dStr, i) for i in derivativeColumnsNum]
    
    else:
        
        derivativeColumnsStr = ['%s%.1f' % (dStr, i) for i in derivativeColumnsNum]
             
    dColumns = dict(zip(derivativeColumnsStr,derivativeColumnsNum) )

    # Replace the columns
    X_train_derivative.columns = derivativeColumnsStr
    
    # Repeat with test data
    # Get the derivatives
    X_test_derivative = X_test.diff(axis=1, periods=1)

    # Drop the first column as it will have only NaN
    X_test_derivative = X_test_derivative.drop(columnsStr[0], axis=1)
    
    # Replace the columns
    X_test_derivative.columns = derivativeColumnsStr
    
    PlotDerivatives(X_train, X_test, X_train_derivative, X_test_derivative, Xcolumns, dColumns, plotLayout, plotFPN )
    
    if joinDerivative:

        X_train_frames = [X_train, X_train_derivative]

        X_train = PdConcat(X_train_frames, axis=1)
        
        X_test_frames = [X_test, X_test_derivative]

        X_test = PdConcat(X_test_frames, axis=1)
        
        columns = {**Xcolumns, **dColumns }
        
    else:

        X_train = X_train_derivative
        
        X_test = X_test_derivative
        
        columns = dColumns
        
    print (columns)
    print (dColumns)
    BALLE
    return X_train, X_test, columns