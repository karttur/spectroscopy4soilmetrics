'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

import numpy as np

from scipy.stats import randint as sp_randint

from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline, make_pipeline
from joblib import Memory

from sklearn.cluster import FeatureAgglomeration

# Feature selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE, RFECV
#from sklearn.feature_selection import SelectFromModel

from sklearn.inspection import permutation_importance
from sklearn.metrics._regression import mean_absolute_error,\
    mean_absolute_percentage_error, median_absolute_error
    
from cubist import Cubist

from time import sleep

import pickle

from numbers import Integral

from math import sqrt

from support import PdSeries, PdDataFrame, PdConcat, DeepCopy

from plotChart import PlotFilterExtract, PlotCoviariateSelection

from plotChart import PlotRegrClass

# Third party imports
import tempfile

class RegressionModels(PlotRegrClass):

    '''Machinelearning using regression models
    '''
    def __init__(self):
        '''creates an empty instance of RegressionMode
        '''

        PlotRegrClass.__init__(self)
        
        self.modelSelectD = {}

        self.modelRetaindD = {}

        self.modD = {}

        #Create a list to hold retained columns
        self.retainD = {}

        self.retainPrintD = {}

        self.tunedModD = {}
        
        
    
        
    def _RegrModTrainTest(self, multCompAxs):
        '''
        '''

        #Retrieve the model name and the model itself
        name,model = self.regrModel

        #Fit the model
        model.fit(self.X_train_R, self.y_train_r)

        #Predict the independent variable in the test subset
        predict = model.predict(self.X_test_R)
        
        r2_total = r2_score(self.y_test_r, predict)
        
        rmse_total = sqrt(mean_squared_error(self.y_test_r, predict))
        
        medae_total = median_absolute_error(self.y_test_r, predict)
        
        mae_total = mean_absolute_error(self.y_test_r, predict)
        
        mape_total = mean_absolute_percentage_error(self.y_test_r, predict)
        
        

        self.trainTestResultD[self.targetFeature][name] = {'rmse':rmse_total,
                                                           'mae':mae_total,
                                                           'medae': medae_total,
                                                           'mape':mape_total,
                                                           'r2': r2_total,
                                                           'hyperParameterSetting': self.paramD['modelling']['regressionModels'][name]['hyperParams'],
                                                           'pickle': self.trainTestPickleFPND[self.targetFeature][name]
                                                           }
        
        self.trainTestSummaryD[self.targetFeature][name] = {'rmse':rmse_total,
                                                           'mae':mae_total,
                                                           'medae': medae_total,
                                                           'mape':mape_total,
                                                           'r2': r2_total,
                                                           }
        
        # Set regressor scores to 3 decimals
        self.trainTestResultD[self.targetFeature][name] = {k:(round(v,3) if isinstance(v,float) else v) for (k,v) in self.trainTestResultD[self.targetFeature][name].items()}

        self.trainTestSummaryD[self.targetFeature][name] = {k:(round(v,3) if isinstance(v,float) else v) for (k,v) in self.trainTestSummaryD[self.targetFeature][name].items()}

        # Save the complete model with cPickle
        pickle.dump(model, open(self.trainTestPickleFPND[self.targetFeature][name],  'wb'))

        if self.verbose:

            infoStr =  '\n                trainTest Model: %s\n' %(name)
            
            infoStr += '                    Root mean squared error (RMSE) total: %.2f\n' % rmse_total
            infoStr += '                    Variance (r2) score total: %.2f\n' % r2_total
            
            if self.verbose > 1:

                infoStr += '                    Mean absolute error (MAE) total: %.2f\n' %( mae_total)
    
                infoStr += '                    Mean absolute percent error (MAPE) total: %.2f\n' %( mape_total)
    
                infoStr += '                    Median absolute error (MedAE) total: %.2f\n' %( medae_total)
                            
                infoStr += '                    hyperParams: %s\n' %(self.paramD['modelling']['regressionModels'][name]['hyperParams'])
            
            print (infoStr)
                    
        if self.modelPlot.apply:
            txtstr = 'nspectra: %s\n' %(self.Xall.shape[0])
            txtstr += 'nbands: %s\n' %(self.Xall.shape[1])
            #txtstr += 'min wl: %s\n' %(self.bandL[0])
            #txtstr += 'max wl: %s\n' %(self.bandL[len(self.bandL)-1])
            #txtstr += 'bands: %s\n' %( ' ,'.join('({0})'.format(w) for w in self.aggBandL)  )
            #txtstr += 'width wl: %s' %(int(self.bandL[1]-self.bandL[0]))

            #txtstrHyperParams =  self.HPtuningtxt+'\nHyper Parameters:\n'+'\n'.join([key+': '+str(value) for key, value in self.tunedModD[name].items()])
            suptitle = '%s train/test model (testsize = %s)' %(self.targetFeature, self.datasetSplit.testSize)
            title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                      % {'mod':name,'rmse':mean_squared_error(self.y_test_r, predict),'r2': r2_score(self.y_test_r, predict)} )

            txtstr = ('RMSE: %(rmse)2f\nr2: %(r2)2f\nnTrain: %(i)d\nnTest: %(j)d' \
                      % {'rmse':self.trainTestResultD[self.targetFeature][name]['rmse'],
                         'r2': self.trainTestResultD[self.targetFeature][name]['r2'],
                         'i': self.X_train_R.shape[0], 'j': self.X_test_R.shape[0]})

            self._PlotRegr(self.y_test_r, predict, suptitle, title, txtstr, '',name, 'trainTest', multCompAxs)

    def _RegrModKFold(self, multCompAxs):
        """
        """

        #Retrieve the model name and the model itself
        name,model = self.regrModel

        predict = model_selection.cross_val_predict(model, self.X, self.y, cv=self.modelling.modelTests.Kfold.folds)

        rmse_total = sqrt(mean_squared_error(self.y, predict))

        r2_total = r2_score(self.y, predict)
        
        scoring = 'r2'

        r2_folded = model_selection.cross_val_score(model, self.X, self.y, cv=6, scoring=scoring)
        
        scoring = 'neg_mean_absolute_error'
        
        mae_folded = model_selection.cross_val_score(model, self.X, self.y, cv=6, scoring=scoring)

        scoring = 'neg_mean_absolute_percentage_error'
        
        mape_folded = model_selection.cross_val_score(model, self.X, self.y, cv=6, scoring=scoring)

        scoring = 'neg_median_absolute_error'
        
        medae_folded = model_selection.cross_val_score(model, self.X, self.y, cv=6, scoring=scoring)

        scoring = 'neg_root_mean_squared_error'
        
        rmse_folded = model_selection.cross_val_score(model, self.X, self.y, cv=6, scoring=scoring)

        self.KfoldResultD[self.targetFeature][name] = {'rmse_total': rmse_total,
                                                       'r2_total': r2_total,
                                                       
                                                       'rmse_folded_mean': -1*rmse_folded.mean(),
                                                       'rmse_folded_std': rmse_folded.std(),
                                                       
                                                       'mae_folded_mean': -1*mae_folded.mean(),
                                                       'mae_folded_std': mae_folded.std(),
                                                       
                                                       'mape_folded_mean': -1*mape_folded.mean(),
                                                       'mape_folded_std': mape_folded.std(),
                                                       
                                                       'medae_folded_mean': medae_folded.mean(),
                                                       'medae_folded_std': medae_folded.std(),
                                                       
                                                        'r2_folded_mean': r2_folded.mean(),
                                                        'r2_folded_std': r2_folded.std(),
                                                        'hyperParameterSetting': self.paramD['modelling']['regressionModels'][name]['hyperParams'],
                                                        'pickle': self.KfoldPickleFPND[self.targetFeature][name]
                                                        }
        
        self.KfoldSummaryD[self.targetFeature][name] = {'rmse_total': rmse_total,
                                                       'r2_total': r2_total,
                                                       
                                                       'rmse_folded_mean': -1*rmse_folded.mean(),
                                                       'rmse_fodled_std': rmse_folded.std(),
                                                       
                                                       'mae_folded_mean': -1*mae_folded.mean(),
                                                       'mae_folded_std': mae_folded.std(),
                                                       
                                                       'mape_folded_mean': -1*mape_folded.mean(),
                                                       'mape_folded_std': mape_folded.std(),
                                                       
                                                
                                                       'medae_folded_mean': medae_folded.mean(),
                                                       'medae_folded_std': medae_folded.std(),
                                                       
                                                 
                                                        'r2_folded_mean': r2_folded.mean(),
                                                        'r2_folded_std': r2_folded.std(),
                                                        }
        
        # Set regressor scores to 3 decimals
        self.KfoldResultD[self.targetFeature][name] = {k:(round(v,3) if isinstance(v,float) else v) for (k,v) in self.KfoldResultD[self.targetFeature][name].items()}

        self.KfoldSummaryD[self.targetFeature][name] = {k:(round(v,3) if isinstance(v,float) else v) for (k,v) in self.KfoldSummaryD[self.targetFeature][name].items()}

        # Save the complete model with cPickle
        pickle.dump(model, open(self.KfoldPickleFPND[self.targetFeature][name],  'wb'))

        if self.verbose:

            infoStr =  '\n                Kfold Model: %s\n' %(name)
            
            infoStr += '                    Root mean squared error (RMSE) total: %.2f\n' % rmse_total
            
            infoStr += '                    Variance (r2) score total: %.2f\n' % r2_total
            
            if self.verbose > 1:
            
                infoStr += '                    RMSE folded: %.2f (%.2f) \n' %( -1*rmse_folded.mean(),  rmse_folded.std())
                
                infoStr += '                    Mean absolute error (MAE) folded: %.2f (%.2f) \n' %( -1*mae_folded.mean(),  mae_folded.std())
    
                infoStr += '                    Mean absolute percent error (MAPE) folded: %.2f (%.2f) \n' %( -1*mape_folded.mean(),  mape_folded.std())
    
                infoStr += '                    Median absolute error (MedAE) folded: %.2f (%.2f) \n' %( -1*medae_folded.mean(),  medae_folded.std())
    
                infoStr += '                    Variance (r2) score folded: %.2f (%.2f) \n' %( r2_folded.mean(),  r2_folded.std())

                infoStr += '                    hyperParams: %s\n' %(self.paramD['modelling']['regressionModels'][name]['hyperParams'])

            print (infoStr)
            
        txtstr = 'nspectra: %s\n' %(self.X.shape[0])
        txtstr += 'nbands: %s\n' %(self.X.shape[1])
        #txtstr += 'min wl: %s\n' %(self.bandL[0])
        #txtstr += 'max wl: %s\n' %(self.bandL[len(self.bandL)-1])
        #txtstr += 'bands: %s\n' %( ' ,'.join('({0})'.format(w) for w in self.aggBandL)  )
        #txtstr += 'width wl: %s' %(int(self.bandL[1]-self.bandL[0]))

        #txtstrHyperParams =  self.HPtuningtxt+'\nHyper Parameters:\n'+'\n'.join([key+': '+str(value) for key, value in self.tunedModD[name].items()])
        suptitle = '%s Kfold model (nfolds = %s)' %(self.targetFeature, self.modelling.modelTests.Kfold.folds)
        title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                  % {'mod':name,'rmse':rmse_total,'r2': r2_total} )

        txtstr = ('RMSE: %(rmse)2f\nr2: %(r2)2f\nSamples: %(n)d' \
                      % {'rmse':self.KfoldResultD[self.targetFeature][name]['rmse_total'],
                         'r2': self.KfoldResultD[self.targetFeature][name]['r2_total'],
                         'n': self.X.shape[0]} )

        self._PlotRegr(self.y, predict, suptitle, title, txtstr, '',name, 'Kfold', multCompAxs)

    def _PyplotArgs(self, pyplotAttrs, argD):
                        
        args = [a for a in dir(self.featureImportancePlot.xticks) if not a.startswith('_')]
        
        argD = {"axis":"x"}
        
        for arg in args:
            
            val = '%s' %(getattr(pyplotAttrs, arg))
            
            if (val.replace('.','')).isnumeric():
                
                argD[arg] = getattr(pyplotAttrs, arg)
            
            else:

                argD[arg] = val
                
        return argD
    
    def _SetTargetFeatureSymbol(self):
        '''
        '''

        self.featureSymbolColor = 'black'
        
        self.featureSymbolAlpha = 0.1

        self.featureSymbolMarker = '.'

        self.featureSymbolSize = 100

        if hasattr(self, 'targetFeatureSymbols'):

            if hasattr(self.targetFeatureSymbols, self.targetFeature):

                symbol = getattr(self.targetFeatureSymbols, self.targetFeature)

                if hasattr(symbol, 'color'):

                    self.featureSymbolColor = getattr(symbol, 'color')
                    
                if hasattr(symbol, 'alpha'):

                    self.featureSymbolAlpha = getattr(symbol, 'alpha')

                if hasattr(symbol, 'size'):

                    self.featureSymbolSize = getattr(symbol, 'size')
                    
                    
    def _RegModelSelectSet(self):
        """ Set the regressors to evaluate
        """

        self.regressorModels = []

        if hasattr(self.modelling.regressionModels, 'OLS') and self.modelling.regressionModels.OLS.apply:

            self.regressorModels.append(('OLS', linear_model.LinearRegression(**self.paramD['modelling']['regressionModels']['OLS']['hyperParams'])))

            self.modelSelectD['OLS'] = []

        if hasattr(self.modelling.regressionModels, 'TheilSen') and self.modelling.regressionModels.TheilSen.apply:

            self.regressorModels.append(('TheilSen', linear_model.TheilSenRegressor(**self.paramD['modelling']['regressionModels']['OLS']['hyperParams'])))

            self.modelSelectD['TheilSen'] = []

        if hasattr(self.modelling.regressionModels, 'Huber') and self.modelling.regressionModels.Huber.apply:

            self.regressorModels.append(('Huber', linear_model.HuberRegressor(**self.paramD['modelling']['regressionModels']['OLS']['hyperParams'])))

            self.modelSelectD['Huber'] = []

        if hasattr(self.modelling.regressionModels, 'KnnRegr') and self.modelling.regressionModels.KnnRegr.apply:
            self.regressorModels.append(('KnnRegr', KNeighborsRegressor( **self.paramD['modelling']['regressionModels']['KnnRegr']['hyperParams'])))
            self.modelSelectD['KnnRegr'] = []

        if hasattr(self.modelling.regressionModels, 'DecTreeRegr') and self.modelling.regressionModels.DecTreeRegr.apply:
            self.regressorModels.append(('DecTreeRegr', DecisionTreeRegressor(**self.paramD['modelling']['regressionModels']['DecTreeRegr']['hyperParams'])))
            self.modelSelectD['DecTreeRegr'] = []

        if hasattr(self.modelling.regressionModels, 'SVR') and self.modelling.regressionModels.SVR.apply:
            self.regressorModels.append(('SVR', SVR(**self.paramD['modelling']['regressionModels']['SVR']['hyperParams'])))
            self.modelSelectD['SVR'] = []

        if hasattr(self.modelling.regressionModels, 'RandForRegr') and self.modelling.regressionModels.RandForRegr.apply:
            self.regressorModels.append(('RandForRegr', RandomForestRegressor( **self.paramD['modelling']['regressionModels']['RandForRegr']['hyperParams'])))
            self.modelSelectD['RandForRegr'] = []

        if hasattr(self.modelling.regressionModels, 'MLP') and self.modelling.regressionModels.MLP.apply:

            '''
            # First make a pipeline with standardscaler + MLP
            mlp = make_pipeline(
                StandardScaler(),
                MLPRegressor( **self.paramD['modelling']['regressionModels']['MLP']['hyperParams'])
            )
            '''
            mlp = Pipeline([('scl', StandardScaler()),
                    ('clf', MLPRegressor( **self.paramD['modelling']['regressionModels']['MLP']['hyperParams']) ) ])

            # Then add the pipeline as MLP
            self.regressorModels.append(('MLP', mlp))

            self.modelSelectD['MLP'] = []
        
        if hasattr(self.modelling.regressionModels, 'Cubist') and self.modelling.regressionModels.Cubist.apply:
            self.regressorModels.append(('Cubist', Cubist( **self.paramD['modelling']['regressionModels']['Cubist']['hyperParams'])))
            self.modelSelectD['Cubist'] = []
        '''    
        if hasattr(self.modelling.regressionModels, 'PLS') and self.modelling.regressionModels.PLS.apply:
            self.regressorModels.append(('PLS', PLSRegressor( **self.paramD['modelling']['regressionModels']['PLS']['hyperParams'])))
            self.modelSelectD['RandForRegr'] = []
        '''
    
    def _MultCompPlotFeatureImportance(self, featureArray, importanceArray, errorArray, index, yLabel, multCompAxs):
        '''
        '''

        nnFS = self.Xall.shape

        text = 'tot covars: %s' %(nnFS[1])

        #if self.specificFeatureSelectionTxt != None:

        #    text += '\n%s' %(self.specificFeatureSelectionTxt)

        #if self.agglomerateTxt != None:

        #    text += '\n%s' %(self.agglomerateTxt)
        # SNULLE

        if self.specificFeatureSelectionTxt != None:

            text += '\n%s' %(self.specificFeatureSelectionTxt)
            
        columnNr = self.modelNr
           
        if (len(self.regressorModels)) == 1: # only state the column
            
            multCompAxs[self.targetFeature][index][columnNr].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

            multCompAxs[self.targetFeature][index][columnNr].tick_params(labelleft=False)

            multCompAxs[self.targetFeature][index][columnNr].text(.3, .95, text, ha='left', va='top',
                                        transform=multCompAxs[self.targetFeature][index][columnNr].transAxes)

            #multCompAxs[self.targetFeature][index][columnNr].set_ylabel(yLabel)
            
            if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
        
                argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                            
                multCompAxs[self.targetFeature][index][columnNr].tick_params(**argD)
                
            multCompAxs[self.targetFeature][index][columnNr].set_ylim(ymin=0)


        else:
                            
            multCompAxs[self.targetFeature][index][self.regrN, columnNr].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

            multCompAxs[self.targetFeature][index][self.regrN, columnNr].tick_params(labelleft=False)

            multCompAxs[self.targetFeature][index][self.regrN, columnNr].text(.3, .95, text, ha='left', va='top',
                                            transform=multCompAxs[self.targetFeature][index][self.regrN, columnNr].transAxes)

            #multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylabel(yLabel)

            if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
        
                argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                            
                multCompAxs[self.targetFeature][index][self.regrN, columnNr].tick_params(**argD)
            
            multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylim(ymin=0)
            
        if index == 'coefficientImportance':

            if (len(self.regressorModels)) == 1:

                # Draw horisontal line ay y=y
                multCompAxs[self.targetFeature][index][columnNr].axhline(y=0, lw=1, c='black')

            else:

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].axhline(y=0, lw=1, c='black')

        # if at last row
        if self.regrN == self.nRegrModels-1:

            if (len(self.regressorModels)) == 1:

                multCompAxs[self.targetFeature][index][columnNr].set_xlabel('Features')

            else:

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_xlabel('Features')
                
        # if at first column
        if columnNr == 0:

            if (len(self.regressorModels)) == 1:
                        
                multCompAxs[self.targetFeature][index][columnNr].set_ylabel(self.regrModel[0])
                multCompAxs[self.targetFeature][index][columnNr].yaxis.set_label_position("left")

            else:

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylabel(self.regrModel[0])
                multCompAxs[self.targetFeature][index][self.regrN, columnNr].yaxis.set_label_position("left")

        # if at last column
        if columnNr+1 == multCompAxs[self.targetFeature][index].shape[0]:

            if (len(self.regressorModels)) == 1:

                multCompAxs[self.targetFeature][index][columnNr].set_ylabel(yLabel)
                multCompAxs[self.targetFeature][index][columnNr].yaxis.set_label_position("right")

            else:

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylabel(yLabel)
                multCompAxs[self.targetFeature][index][self.regrN, columnNr].yaxis.set_label_position("right")

    def _FeatureImportance(self, multCompAxs):
        '''
        '''
        
        def FeatureImp():
            '''
            '''
            if name in ['OLS','TheilSen','Huber', "Ridge", "ElasticNet", 'logistic', 'SVR']:

                if name in ['logistic','SVR']:
    
                    importances = model.coef_[0]
    
                else:
    
                    importances = model.coef_
    
                absImportances = abs(importances)
    
                sorted_idx = absImportances.argsort()
    
                importanceArray = importances[sorted_idx][::-1][0:maxFeatures]
    
                featureArray = np.asarray(columns)[sorted_idx][::-1][0:maxFeatures]
    
                featImpD = {}
    
                for i in range(len(featureArray)):
    
                    featImpD[featureArray[i]] = {'linearCoefficient': round(importanceArray[i],4)}
                  
                self.modelFeatureImportanceD[self.targetFeature][name]['featureImportance'] = featImpD
    
                if self.modelPlot.singles.apply:
    
                    title = "Linear feature coefficients\n Feature: %s; Model: %s" %(self.targetFeature, name)
    
                    xyLabels = ['Features','Coefficient']
    
                    pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['regressionImportance']
                                        
                    self._PlotFeatureImportanceSingles(featureArray, np.absolute(importanceArray), None, title, xyLabels, pngFPN)
    
                if self.modelPlot.rows.apply:
    
                    self._PlotFeatureImportanceRows(featureArray, np.absolute(importanceArray), None, 'coefficientImportance','rel. coef. weight')
    
                if self.multcompplot:
          
                    self._MultCompPlotFeatureImportance(featureArray, np.absolute(importanceArray), None, 'coefficientImportance', 'rel. coef. weight', multCompAxs)
    
            elif name in ['KnnRegr','MLP', 'Cubist']:
                ''' These models do not have any feature importance to report
                '''
                pass
    
            else:
    
                featImpD = {}
    
                importances = model.feature_importances_
    
                sorted_idx = importances.argsort()
    
                importanceArray = importances[sorted_idx][::-1][0:maxFeatures]
    
                featureArray = np.asarray(columns)[sorted_idx][::-1][0:maxFeatures]
    
                if name in ['RandForRegr']:
    
                    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
                    errorArray = std[sorted_idx][::-1][0:maxFeatures]
    
                    for i in range(len(featureArray)):
    
                        featImpD[featureArray[i]] = {'MDI': round(importanceArray[i],4),
                                                     'std': round(errorArray[i],4)}
                       
                else:
    
                    errorArray = None
    
                    for i in range(len(featureArray)):
    
                        featImpD[featureArray[i]] = {'MDI': importanceArray[i]}
                        
                        featImpD[featureArray[i]] = {k:(round(v,4) if isinstance(v,float) else v) for (k,v) in featImpD[featureArray[i]]}
    
                self.modelFeatureImportanceD[self.targetFeature][name]['featureImportance'] = featImpD
    
                if self.modelPlot.singles.apply:
    
                    title = "MDI feature importance\n Feature: %s; Model: %s" %(self.targetFeature, name)
    
                    xyLabel = ['Features', 'Mean impurity decrease']
    
                    pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['regressionImportance']
    
                    self._PlotFeatureImportanceSingles(featureArray, importanceArray, errorArray, title, xyLabel, pngFPN)
    
                if self.modelPlot.rows.apply:
    
                    self._PlotFeatureImportanceRows(featureArray, importanceArray, errorArray, 'featureImportance', 'rel. mean impur. decr.')

        
        def PermImp():
            '''
            '''
            n_repeats = self.modelling.featureImportance.permutationRepeats

            permImportance = permutation_importance(model, self.X_test_R, self.y_test_r, n_repeats=n_repeats)
    
            permImportanceMean = permImportance.importances_mean
    
            permImportanceStd = permImportance.importances_std
    
            sorted_idx = permImportanceMean.argsort()
    
            permImportanceArray = permImportanceMean[sorted_idx][::-1][0:maxFeatures]
            
            errorArray = permImportanceStd[sorted_idx][::-1][0:maxFeatures]
    
            featureArray = np.asarray(columns)[sorted_idx][::-1][0:maxFeatures]
    
            permImpD = {}
    
            for i in range(len(featureArray)):
    
                permImpD[featureArray[i]] = {'mean_accuracy_decrease': round(permImportanceArray[i],4),
                                             'std': round(errorArray[i],4)}
                
            self.modelFeatureImportanceD[self.targetFeature][name]['permutationsImportance'] = permImpD
    
            if self.modelPlot.singles.apply:
    
                title = "Permutation importance\n Feature: %s; Model: %s" %(self.targetFeature, name)
    
                xyLabel = ['Features', 'Mean accuracy decrease']
    
                pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['permutationImportance']
    
                self._PlotFeatureImportanceSingles(featureArray, permImportanceArray, errorArray, title, xyLabel, pngFPN)
                
            if self.modelPlot.rows.apply:
    
                self._PlotFeatureImportanceRows(featureArray, permImportanceArray, errorArray, 'permutationImportance', 'rel. Mean accur. decr.')
    
            if self.multcompplot:
          
                self._MultCompPlotFeatureImportance(featureArray, permImportanceArray, errorArray, 'permutationImportance', 'rel. Mean accur. decr.', multCompAxs)

        
        def TreeBasedImp():
            '''
            '''
 
            forest = RandomForestRegressor(random_state=0)

            forest.fit(self.X_test_R, self.y_test_r)
        
            treeBasedImportanceMean = forest.feature_importances_
     
            treeBasedImportanceStd = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    
            sorted_idx = treeBasedImportanceMean.argsort()
    
            treeBasedImportanceArray = treeBasedImportanceMean[sorted_idx][::-1][0:maxFeatures]
            
            errorArray = treeBasedImportanceStd[sorted_idx][::-1][0:maxFeatures]
    
            featureArray = np.asarray(columns)[sorted_idx][::-1][0:maxFeatures]
    
            treeBasedImpD = {}
    
            for i in range(len(featureArray)):
    
                treeBasedImpD[featureArray[i]] = {'mean_accuracy_decrease': round(treeBasedImportanceArray[i],4),
                                             'std': round(errorArray[i],4)}
                
            self.modelFeatureImportanceD[self.targetFeature][name]['treeBasedImportance'] = treeBasedImpD
    
            if self.modelPlot.singles.apply:
    
                title = "Tree Based importance\n Feature: %s; Model: %s" %(self.targetFeature, name)
    
                xyLabel = ['Features', 'Mean impure. decr.']
    
                pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['treeBasedImportance']
    
                self._PlotFeatureImportanceSingles(featureArray, treeBasedImportanceArray, errorArray, title, xyLabel, pngFPN)
                
            if self.modelPlot.rows.apply:
    
                self._PlotFeatureImportanceRows(featureArray, treeBasedImportanceArray, errorArray, 'treeBasedImportance', 'Mean impure. decr.')
    
            if self.multcompplot:
          
                self._MultCompPlotFeatureImportance(featureArray, treeBasedImportanceArray, errorArray, 'treeBasedImportance', 'Mean impure. decr.', multCompAxs)

        ''' Main function '''
        #Retrieve the model name and the model itself
        name,model = self.regrModel
        
        columns = [item for item in self.X_train_R.columns]
        
        #Fit the model
        model.fit(self.X_train_R, self.y_train_r)

        #Set the nr of feature (x-axis) items
        maxFeatures = min(self.modelling.featureImportance.reportMaxFeatures, len(columns))

        # Permutation importance
        PermImp()
        
        # Coefficient importance
        FeatureImp()
        
        # Treebased Importance
        TreeBasedImp()
        
    def _ManualFeatureSelector(self):
        '''
        '''

        exit('_ManualFeatureSelector NOT YET UPDATED')
        # Reset columns
        columns = self.manualFeatureSelection.spectra

        # Create the dataframe for the sepctra
        spectraDF = self.spectraDF[ columns  ]
        
        X_train = self.X_train[ columns  ]

        self.manualFeatureSelectdRawBands =  columns
        
        self.manualFeatureSelectdDerivates = []
        
        # Create any derivative covariates requested
        
        if hasattr(self.manualFeatureSelection, 'derivatives'):
        
            if hasattr(self.manualFeatureSelection.derivatives, 'firstWaveLength'):
                
                for b in range(len(self.manualFeatureSelection.derivatives.firstWaveLength)):
        
                    bandL = [self.manualFeatureSelection.derivatives.firstWaveLength[b],
                             self.manualFeatureSelection.derivatives.lastWaveLength[b]]
    
                self.manualFeatureSelectdDerivates = bandL
    
                derviationBandDF = X_train[ bandL  ]
    
                bandFrame, bandColumn = self._SpectraDerivativeFromDf(derviationBandDF,bandL)
    
                frames = [X_train,bandFrame]
    
                #spectraDF = pd.concat(frames, axis=1)
    
                columns.extend(bandColumn)

        # reset self.spectraDF
        self.X_train = X_train

    def _VarianceSelector(self):
        '''
        '''

        threshold = self.generalFeatureSelection.varianceThreshold.threshold

        columns = [item for item in self.X_train.columns]
      
        #Initiate the scaler
        if self.generalFeatureSelection.varianceThreshold.scaler == 'None':
            
            X_train_scaled = Xscaled = self.X_train
            X_test_scaled = self.X_test
                       
        elif self.generalFeatureSelection.varianceThreshold.scaler == 'MinMaxScaler':

            scaler = MinMaxScaler()
            
            scaler.fit(self.X_train)
            
            #Scale the data as defined by the scaler
            Xscaled = scaler.transform(self.X_train)
            
            X_train_scaled = PdDataFrame(scaler.transform(self.X_train), columns)
            X_test_scaled = PdDataFrame(scaler.transform(self.X_test), columns)
            
        else:
            
            exitStr = 'EXITING - the scaler %s is not implemented for varianceThreshold ' %(self.generalFeatureSelection.varianceThreshold.scaler)

            exit (exitStr)
            
        if isinstance(threshold, float):
            select = VarianceThreshold(threshold=threshold)
   
        elif isinstance(threshold, int):
            select = VarianceThreshold(threshold=0.0000001)
            
        elif isinstance(threshold, str):
            select = VarianceThreshold(threshold=0.0000001)
            thresholdPercent = int(threshold[0:len(threshold)-1])
            threshold = int(round(len(columns)*thresholdPercent/100 ))
            
        #Initiate  VarianceThreshold
        #Fit the independent variables
        select.fit(Xscaled)

        #Get the selected features from get_support as a boolean list with True or False
        selectedFeatures = select.get_support()
        
        completeL = []
        
        #Create a list to hold discarded columns
        discardL = []

        #Create a list to hold retained columns
        retainL = []

        for sf in range(len(selectedFeatures)):

            completeL.append([columns[sf],select.variances_[sf]])
            
            if selectedFeatures[sf]:
                retainL.append([columns[sf],select.variances_[sf]])

            else:
                discardL.append([columns[sf],select.variances_[sf]])
               
        completeL.sort(key = lambda x: x[1]) 

        if isinstance(threshold, int) or isinstance(threshold, str) :
            
            discardL = completeL[0:threshold]
            
            retainL = completeL[threshold:len(retainL)]
            
        else:
            
            retainL.sort(key = lambda x: x[1])
            
            discardL.sort(key = lambda x: x[1])
                       
        if self.generalFeatureSelection.varianceThreshold.onlyShowVarianceList:
            
            print ('                covariate variance')
            print ('                band (variance)')
            printL = ['%s (%.3f)'%(i[0],i[1]) for i in completeL]

            for row in printL:
                print ('                ',row)
                
            print('') 
            
            sleep(2)
                
            exit('Select the varianceThrehsold{"threshold"} set varianceThrehsold{"showVarianceList"}) to true and rerun')
                
        if self.verbose:

            print ('            Selecting features using VarianceThreshold, threhold =',threshold)

            print ('                Scaling function MinMaxScaler:')
                
        self.generalFeatureSelectedD['method'] = 'varianceThreshold'
        self.generalFeatureSelectedD['threshold'] = self.generalFeatureSelection.varianceThreshold.threshold
        #self.generalFeatureSelectedD['scaler'] = self.generalFeatureSelection.scaler
        self.generalFeatureSelectedD['nCovariatesRemoved'] = len(discardL)

        #varianceSelectTxt = '%s covariate(s) removed with %s' %(len(discardL),'VarianceThreshold')
        
        generalFeatureSelectTxt = '%s covariate(s) removed with %s' %(len(discardL),'VarianceThreshold')

        #self.varianceSelectTxt = '%s: %s' %('VarianceThreshold',len(discardL))
        
        self.generalFeatureSelectTxt = '%s: %s' %('VarianceThreshold',len(discardL))

        if self.verbose:

            print ('            ',generalFeatureSelectTxt)

            if self.verbose > 1:

                #print the selected features and their variance
                print ('            Discarded features [name, (variance):')

                printL = ['%s (%.3f)'%(i[0],i[1]) for i in discardL]

                for row in printL:
                    print ('                ',row)

                print ('            Retained features [name, (variance)]:')

                printretainL = ['%s (%.3f)'%(i[0], i[1]) for i in self.retainL]

                for row in printretainL:
                    print ('                ',row)

        retainL = [d[0] for d in retainL]
        
        discardL = [item[0] for item in discardL]
        
        PlotCoviariateSelection('Variance threshold', self.enhancementPlotLayout.varianceThreshold,  
                self.enhancementPlotLayout, self.preProcessFPND['varianceThreshold'],
                X_train_scaled, X_test_scaled, retainL, discardL, 
                self.Xcolumns, 'all', 'None',self.generalFeatureSelection.varianceThreshold.scaler)
        
        # Remake the X_train and X_test datasets
        self.X_train = self.X_train[ retainL ]
        
        self.X_test = self.X_test[ retainL ]
        
    def _ResetXY_T(self):
        
        self.X_train_T.reset_index(drop=True, inplace=True)
        
        self.X_test_T.reset_index(drop=True, inplace=True)
        
        self.y_train_t.reset_index(drop=True, inplace=True)
        
        self.y_test_t.reset_index(drop=True, inplace=True)
        
        # Remove all non-finite values       
        self.X_train_T = self.X_train_T[np.isfinite(self.y_train_t)] 
        
        self.y_train_t = self.y_train_t[np.isfinite(self.y_train_t)] 
        
        self.X_test_T = self.X_test_T[np.isfinite(self.y_test_t)] 

        self.y_test_t = self.y_test_t[np.isfinite(self.y_test_t)] 
        
    def _ResetXY_R(self):
        
        self.X_train_R.reset_index(drop=True, inplace=True)
        
        self.X_test_R.reset_index(drop=True, inplace=True)
        
        self.y_train_r.reset_index(drop=True, inplace=True)
        
        self.y_test_r.reset_index(drop=True, inplace=True)
        
        # Remove all non-finite values       
        self.X_train_R = self.X_train_R[np.isfinite(self.y_train_r)] 
        
        self.y_train_r = self.y_train_r[np.isfinite(self.y_train_r)] 
        
        self.X_test_R = self.X_test_R[np.isfinite(self.y_test_r)] 

        self.y_test_r = self.y_test_r[np.isfinite(self.y_test_r)]
        
    def _ResetTargetFeature(self):
        ''' Resets target feature for looping
        '''
        self.y_train_t = self.Y_train[self.targetFeature]
        
        self.y_test_t = self.Y_test[self.targetFeature]
                
        # Copy the X data
        self.X_train_T = DeepCopy(self.X_train)
        
        self.X_test_T = DeepCopy(self.X_test)

        # Remove all samples where the targetfeature is NaN
        self.X_train_T = self.X_train_T[~np.isnan(self.y_train_t)]
        
        self.y_train_t = self.y_train_t[~np.isnan(self.y_train_t)]
        
        self.X_test_T = self.X_test_T[~np.isnan(self.y_test_t)]
        
        self.y_test_t = self.y_test_t[~np.isnan(self.y_test_t)]
               
        self._ResetXY_T()
        
        self.X_columns_T = DeepCopy(self.Xcolumns)
        
    def _PrepCovarSelection(self, nSelect):
        '''
        '''
        
        nfeatures = self.X_train_R.shape[1]

        if nSelect >= nfeatures:

            if self.verbose:

                infostr = '            SKIPPING specific selection: Number of input features (%s) less than or equal to minimumm output covariates to select (%s).' %(nfeatures, nSelect)

                print (infostr)

            return (False, False)
        
        columns = [item for item in self.X_train_R.columns]

        return (nfeatures, columns)   
    
    def _CleanUpCovarSelection(self, selector, selectorSymbolisation, retainL, discardL):
        '''
        '''
        
        self.specificFeatureSelectedD[self.targetFeature][self.regrModel[0]]['method'] = selector

        self.specificFeatureSelectedD[self.targetFeature][self.regrModel[0]]['nFeaturesRemoved'] = len( discardL)

        self.specificFeatureSelectionTxt = '- %s %s' %(len(discardL), selector)

        if self.verbose:
             
            print ('\n            specificFeatureSelection:')

            print ('                Regressor: %(m)s' %{'m':self.regrModel[0]})

            print ('                ',self.specificFeatureSelectionTxt)

            if self.verbose > 1:
    
                print ('                Selected features: %s' %(', '.join(retainL)))
        
        pngFPN = self.preProcessFPND['specificSelection'][self.targetFeature][self.regrModel[0]]

        label = '%s Selection' %(selector)
        
        PlotCoviariateSelection(label, selectorSymbolisation,  
                self.enhancementPlotLayout, pngFPN,
                self.X_train_R, self.X_test_R, retainL, discardL, 
                self.X_columns_R, self.paramD['targetFeatureSymbols'][self.targetFeature]['label'], self.regrModel[0])
   
        # reset the covariates
        self.X_train_R = PdDataFrame(self.X_train_R, retainL)
        
        self.X_test_R = PdDataFrame(self.X_test_R, retainL)
        
        self._ResetXY_R()
        
        # Reset columns
        valueL = []
        
        for item in retainL:
            if item in self.columns:
                
                valueL.append(self.columns[item])
                
        self.X_columns_R = dict(zip(retainL,valueL))
        
    def _CleanUpSpecificeAgglomeration(self, selector, selectorSymbolisation, retainL, discardL):
        '''
        '''

        self.specificClusteringdD[self.targetFeature]['nFeaturesRemoved'] = len( discardL)

        self.specificClusteringTxt = '- %s %s' %(len(discardL), selector)
        
        #agglomeratetxt = '%s input features clustered to %s covariates using  %s' %(len(columns),len(self.aggColumnL),self.globalFeatureSelectedD['method'])

        #self.agglomerateTxt = '%s clustered from %s to %s Feat´s' %(self.globalFeatureSelectedD['method'], len(columns),len(self.aggColumnL))
        #FIXATEXT

        if self.verbose:
              
            print ('\n            specificFeatureAgglomeration:')
                
            if self.verbose > 1:
    
                print ('                Selected features: %s' %(', '.join(retainL)))
                        
        pngFPN = self.preProcessFPND['specificClustering'][self.targetFeature]
                
        label = '%s Agglomeration' %(selector)

        PlotCoviariateSelection(label, selectorSymbolisation,  
                self.enhancementPlotLayout, pngFPN,
                self.X_train_T, self.X_test_T, retainL, discardL, 
                self.X_columns_T, self.paramD['targetFeatureSymbols'][self.targetFeature]['label'])
   
        # reset the covariates
        self.X_train_T = PdDataFrame(self.X_train_T, retainL)
        
        self.X_test_T = PdDataFrame(self.X_test_T, retainL)
        
        # Reset columns
        valueL = []
        
        for item in retainL:
            if item in self.columns:
                
                valueL.append(self.columns[item])
                
        self.X_columns_T = dict(zip(retainL,valueL))
                                     
    def _UnivariateSelector(self):
        '''
        '''

        nfeatures, columns = self._PrepCovarSelection(self.specificFeatureSelection.univariateSelection.SelectKBest.n_features)
        
        if not nfeatures:
            
            return
        
        # Apply SelectKBest
        select = SelectKBest(score_func=f_regression, k=self.specificFeatureSelection.univariateSelection.SelectKBest.n_features)
        
        # Select and fit the independent variables, return the selected array

        select.fit(self.X_train_R, self.y_train_r)
        
        # Note that the returned select.get_feature_names_out() is not a list
        retainL = select.get_feature_names_out()
        
        discardL = list( set(columns).symmetric_difference(retainL) )
        
        retainL.sort();  discardL.sort()
                
        # Save the results in a dictionary
        scores = select.scores_
        
        pvalues = select.pvalues_
        
        covars = select.feature_names_in_
        
        self.selectKBestResultD = {}
        
        for c, covar in enumerate(covars):
            
            self.selectKBestResultD[covar] = {'score': scores[c], 'pvalue': pvalues[c]}
         
        self._CleanUpCovarSelection('Univar SelKBest', self.enhancementPlotLayout.univariateSelection, retainL, discardL )
                               
    def _PermutationSelector(self):
        '''
        '''

        nfeatures, columns = self._PrepCovarSelection(self.specificFeatureSelection.permutationSelector.n_features_to_select)
        
        if not nfeatures:
            
            return
        
        #Retrieve the model name and the model itself
        model = self.regrModel[1]
        
        #Fit the model
        model.fit(self.X_train_R, self.y_train_r)

        permImportance = permutation_importance(model, self.X_test_R, self.y_test_r)

        permImportanceMean = permImportance.importances_mean

        sorted_idx = permImportanceMean.argsort()

        retainL = np.asarray(columns)[sorted_idx][::-1][0:self.specificFeatureSelection.permutationSelector.n_features_to_select].tolist()
        
        r = set(retainL)
        
        discardL = [x for x in columns if x not in r]
        
        retainL.sort(); discardL.sort()
        
        self._CleanUpCovarSelection('Permut Select', self.enhancementPlotLayout.permutationSelector, retainL, discardL)
        
    def _TreeBasedFeatureSelector(self):
        ''' NOTIMPLEMENTED
        '''
        
        pass
        
        # See https://scikit-learn.org/stable/modules/feature_selection.html

    def _RFESelector(self):
        '''
        '''

        nfeatures, columns = self._PrepCovarSelection(self.specificFeatureSelection.RFE.n_features_to_select)
        
        if not nfeatures:
            
            return

        step = self.specificFeatureSelection.RFE.step

        if self.verbose:

            if self.specificFeatureSelection.RFE.CV:

                print ('\n            RFECV feature selection')

            else:

                print ('\n            RFE feature selection')

        #Retrieve the model name and the model itself
        model = self.regrModel[1]

        if self.specificFeatureSelection.RFE.CV:

            select = RFECV(estimator=model, min_features_to_select=self.specificFeatureSelection.RFE.n_features_to_select, step=step, cv= self.specificFeatureSelection.RFE.CV)
            
            selector = 'RFECV'
      
        else:
            
            selector = 'RFE'
            
            select = RFE(estimator=model, n_features_to_select=self.specificFeatureSelection.RFE.n_features_to_select, step=step)
                
        select.fit(self.X_train_R, self.y_train_r)

        selectedFeatures = select.get_support()

        #Create a list to hold discarded columns
        retainL = []; discardL = []

        for sf in range(len(selectedFeatures)):
            if selectedFeatures[sf]:
                retainL.append(columns[sf])

            else:
                discardL.append(columns[sf])
                
        label = '%s Selection' %(selector)
        
        self._CleanUpCovarSelection(label, self.enhancementPlotLayout.RFE, retainL, discardL)       
                                            
    def _WardClustering(self, n_clusters):
        '''
        '''

        nfeatures = self.X_train_T.shape[1]

        if nfeatures < n_clusters:

            n_clusters = nfeatures
            
            return
        
        ward = FeatureAgglomeration(n_clusters=n_clusters)

        #fit the clusters
        ward.fit(self.X_train_T, self.y_train_t)

        self.clustering =  ward.labels_

        # Get a list of bands
        bandsL =  list(self.X_train_T)

        self.aggColumnL = []

        self.aggBandL = []
        
        discardL = []

        for m in range(len(ward.labels_)):

            indices = [bandsL[i] for i, x in enumerate(ward.labels_) if x == m]

            if(len(indices) == 0):

                break

            self.aggColumnL.append(indices[0])

            self.aggBandL.append( ', '.join(indices) )
            
            discardL.extend( indices[1:len(indices)])
            
        self.aggColumnL.sort()
        
        discardL.sort()
            
        self._CleanUpSpecificeAgglomeration('WardClustering', self.enhancementPlotLayout.wardClustering, self.aggColumnL, discardL )
                                           
    def _TuneWardClustering(self):
        ''' Determines the optimal nr of cluster
        '''
        nfeatures = self.X_train_T.shape[1]

        nClustersL = self.specificFeatureAgglomeration.wardClustering.tuneWardClustering.clusters

        nClustersL = [c for c in nClustersL if c < nfeatures]

        kfolds = self.specificFeatureAgglomeration.wardClustering.tuneWardClustering.kfolds

        cv = KFold(kfolds)  # cross-validation generator for model selection

        ridge = BayesianRidge()

        cachedir = tempfile.mkdtemp()

        mem = Memory(location=cachedir)

        ward = FeatureAgglomeration(n_clusters=4, memory=mem)

        clf = Pipeline([('ward', ward), ('ridge', ridge)])

        # Select the optimal number of parcels with grid search
        clf = GridSearchCV(clf, {'ward__n_clusters': nClustersL}, n_jobs=1, cv=cv)

        clf.fit(self.X_train_T, self.y_train_t)  # set the best parameters

        if self.verbose:

            print ('            Report for tuning Ward Clustering')

        #report the top three results
        self._ReportSearch(clf.cv_results_,3)

        #rerun with the best cluster agglomeration result
        tunedClusters = clf.best_params_['ward__n_clusters']

        if self.verbose:

            print ('                Tuned Ward clusters:', tunedClusters)

        return (tunedClusters)
    
    
    def _RandomtuningParams(self,nFeatures):
        ''' Set hyper parameters for random tuning
        '''
        self.paramDist = {}

        self.HPtuningtxt = 'Random tuning'

        # specify parameters and distributions to sample from
        name, model = self.regrModel

        if name == 'KnnRegr':

            self.paramDist[name] = {"n_neighbors": sp_randint(self.hyperParams.RandomTuning.KnnRegr.n_neigbors.min,
                                                              self.hyperParams.RandomTuning.KnnRegr.n_neigbors.max),
                          'leaf_size': sp_randint(self.hyperParams.RandomTuning.KnnRegr.leaf_size.min,
                                                              self.hyperParams.RandomTuning.KnnRegr.leaf_size.max),
                          'weights': self.hyperParams.RandomTuning.KnnRegr.weights,
                          'p': self.hyperParams.RandomTuning.KnnRegr.weights,
                          'algorithm': self.hyperParams.RandomTuning.KnnRegr.algorithm}

        elif name =='DecTreeRegr':
            # Convert 0 to None for max_depth

            max_depth = [m if m > 0 else None for m in self.hyperParams.RandomTuning.DecTreeRegr.max_depth]

            self.paramDist[name] = {"max_depth": max_depth,
                        "min_samples_split": sp_randint(self.hyperParams.RandomTuning.DecTreeRegr.min_samples_split.min,
                                                        self.hyperParams.RandomTuning.DecTreeRegr.min_samples_split.max),
                        "min_samples_leaf": sp_randint(self.hyperParams.RandomTuning.DecTreeRegr.min_samples_leaf.min,
                                                        self.hyperParams.RandomTuning.DecTreeRegr.min_samples_leaf.max)}
        elif name =='SVR':

            self.paramDist[name] = {"kernel": self.hyperParams.RandomTuning.SVR.kernel,
                                    "epsilon": self.hyperParams.RandomTuning.SVR.epsilon,
                                    "C": self.hyperParams.RandomTuning.SVR.epsilon}

        elif name =='RandForRegr':

            max_depth = [m if m > 0 else None for m in self.hyperParams.RandomTuning.RandForRegr.tuningParams.max_depth]

            max_features_max = min(self.hyperParams.RandomTuning.RandForRegr.tuningParams.max_features.max,nFeatures)

            max_features_min = min(self.hyperParams.RandomTuning.RandForRegr.tuningParams.max_features.min,nFeatures)

            print (self.hyperParams.RandomTuning.RandForRegr.tuningParams.n_estimators.min)
            
            self.paramDist[name] = {"max_depth": max_depth,
                          "n_estimators": sp_randint(self.hyperParams.RandomTuning.RandForRegr.tuningParams.n_estimators.min,
                                                              self.hyperParams.RandomTuning.RandForRegr.tuningParams.n_estimators.max),
                          "max_features": sp_randint(max_features_min,
                                                              max_features_max),
                          "min_samples_split": sp_randint(self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_split.min,
                                                              self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_split.max),
                          "min_samples_leaf": sp_randint(self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_leaf.min,
                                                              self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_leaf.max),
                          "bootstrap": self.hyperParams.RandomTuning.RandForRegr.tuningParams.bootstrap}

        elif name =='MLP':

            self.paramDist[name] = {
                        "hidden_layer_sizes": self.hyperParams.RandomTuning.MLP.hidden_layer_sizes,
                        "solver": self.hyperParams.RandomTuning.MLP.solver,
                        "alpha": sp_randint(self.hyperParams.RandomTuning.MPL.tuningParams.alpha.min,
                                    self.hyperParams.RandomTuning.MPL.tuningParams.alpha.max),
                        "max_iter": sp_randint(self.hyperParams.RandomTuning.MPL.tuningParams.max_iter.min,
                                    self.hyperParams.RandomTuning.MPL.tuningParams.max_iter.max)}

    def _ExhaustivetuningParams(self,nFeatures):
        '''
        '''

        self.HPtuningtxt = 'Exhaustive tuning'

        # specify parameters and distributions to sample from
        self.paramGrid = {}

        name,model = self.regrModel

        if name == 'KnnRegr':

            self.paramGrid[name] = [{"n_neighbors": self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.n_neigbors,
                               'weights': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.weights,
                               'algorithm': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.algorithm,
                               'leaf_size': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.leaf_size,
                               'p': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.p}
                               ]
        elif name =='DecTreeRegr':
            max_depth = [m if m > 0 else None for m in self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.max_depth]

            self.paramGrid[name] = [{
                                "splitter": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.splitter,
                                "max_depth": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.max_depth,
                                "min_samples_split": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.min_samples_split,
                                "min_samples_leaf": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.min_samples_leaf}]

        elif name =='SVR':
            self.paramGrid[name] = [{"kernel": self.hyperParams.ExhaustiveTuning.SVR.tuningParams.kernel,
                                "epsilon": self.hyperParams.ExhaustiveTuning.SVR.tuningParams.epsilon,
                                "C": self.hyperParams.ExhaustiveTuning.SVR.tuningParams.C
                              }]

        elif name =='RandForRegr':
            max_depth = [m if m > 0 else None for m in self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.max_depth]

            self.paramGrid[name] = [{
                            "max_depth": max_depth,
                          "n_estimators": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.n_estimators,
                          "min_samples_split": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.min_samples_split,
                          "min_samples_leaf": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.min_samples_leaf,
                          "bootstrap": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.bootstrap}]

        elif name =='MLP':
            self.paramGrid[name] = [{
                        "hidden_layer_sizes": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.hidden_layer_sizes,
                        "solver": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.solver,
                        "alpha": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.alpha,
                        "max_iter": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.max_iter}]

    def _RandomTuning(self):
        '''
        '''

        #Retrieve the model name and the model itself
        name,mod = self.regrModel

        nFeatures = self.X.shape[1]

        # Get the tuning parameters
        self._RandomtuningParams(nFeatures)

        if self.verbose:

            print ('\n                HyperParameter random tuning:')

            print ('                    ',name, self.paramDist[name])

        search = RandomizedSearchCV(mod, param_distributions=self.paramDist[name],
                                           n_iter=self.params.hyperParameterTuning.nIterSearch)

        
        search.fit(self.X_train_R, self.y_train_r)
        
        resultD = self._ReportSearch(search.cv_results_,self.params.hyperParameterTuning.n_best_report)

        self.tunedHyperParamsD[self.targetFeature][name] = resultD

        # Set the hyperParameters to the best result
        for key in resultD[1]['hyperParameters']:

            self.paramD['modelling']['regressionModels'][name]['hyperParams'][key] = resultD[1]['hyperParameters'][key]

    def _ExhaustiveTuning(self):
        '''
        '''

        #Retrieve the model name and the model itself
        name,mod = self.regrModel

        nFeatures = self.X.shape[1]

        # Get the tuning parameters
        self._ExhaustivetuningParams(nFeatures)

        if self.verbose:

            print ('\n                HyperParameter exhaustive tuning:')

            print ('                    ',name, self.paramGrid[name])

        search = GridSearchCV(mod, param_grid=self.paramGrid[name])

       
        search.fit(self.X_train_R, self.y_train_r)
        
        resultD = self._ReportSearch(search.cv_results_,self.params.hyperParameterTuning.n_best_report)

        self.tunedHyperParamsD[self.targetFeature][name] = resultD

        # Set the hyperParameters to the best result
        for key in resultD[1]['hyperParameters']:

            self.paramD['modelling']['regressionModels'][name]['hyperParams'][key] = resultD[1]['hyperParameters'][key]

    def _ReportRegModelParams(self):
        '''
        '''

        print ('\n            %s hyper-parameters: %s' % (self.regrModel[0], self.regrModel[1]))
        '''
        for model in self.regressorModels:

            #Retrieve the model name and the model itself
            modelname,modelhyperparams = model

            print ('                name', modelname, modelhyperparams.get_params())
        '''

    def _ReportSearch(self, results, n_top=3):
        '''
        '''

        resultD = {}
        for i in range(1, n_top + 1):

            resultD[i] = {}

            candidates = np.flatnonzero(results['rank_test_score'] == i)

            for candidate in candidates:

                resultD[i]['mean_test_score'] = results['mean_test_score'][candidate]

                resultD[i]['std'] = round(results['std_test_score'][candidate],4)

                resultD[i]['std'] = round(results['std_test_score'][candidate],4)

                resultD[i]['hyperParameters'] = results['params'][candidate]

                if self.verbose:

                    print("                    Model with rank: {0}".format(i))

                    print("                    Mean validation score: {0:.3f} (std: {1:.3f})".format(
                          results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))

                    print("                    Parameters: {0}".format(results['params'][candidate]))

                    print("")

        return resultD