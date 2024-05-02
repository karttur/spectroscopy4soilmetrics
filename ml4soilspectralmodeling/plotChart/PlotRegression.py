'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

import matplotlib.pyplot as plt

import numpy as np 

from support import PdSeries

class PlotRegrClass:
    '''
    '''
    
    def __init__(self):
        '''Initiate
        '''

        pass
    
    def _PlotRegr(self, obs, pred, suptitle, title, txtstr,  txtstrHyperParams, regrModel, modeltest, multCompAxs):
        '''
        '''
        
        #if isinstance(obs, pd.DataFrame):
            
        #    exit('obs is a dataframe, must be a dataseries')

        if self.modelPlot.singles.apply:
            
            fig, ax = plt.subplots()
            ax.scatter(obs, pred, edgecolors=(0, 0, 0),  color=self.featureSymbolColor,
                       alpha=self.featureSymbolAlpha,
                       s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                       marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])
            ax.plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)
            ax.set_xlabel('Observations')
            ax.set_ylabel('Predictions')
            plt.suptitle(suptitle)
            plt.title(title)
            ax.text(obs.min(), (obs.max()-obs.min())*0.8, txtstr, fontdict=None,  wrap=True)

            if self.modelPlot.singles.screenShow:

                plt.show()

            if self.modelPlot.singles.savePng:
                
                self.figLibL.append(self.imageFPND[self.targetFeature][regrModel][modeltest])

                fig.savefig(self.imageFPND[self.targetFeature][regrModel][modeltest])
                
            plt.close(fig=fig)


        if self.modelPlot.rows.apply:

            if self.modelPlot.rows.targetFeatures.apply:

                # modeltest is either trainTest of Kfold
                if modeltest in self.modelPlot.rows.targetFeatures.columns:

                    if len(self.targetFeatures) == 1:

                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest] ].scatter(obs, pred, edgecolors=(0, 0, 0),  
                                color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                               s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                               marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest] ].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)

                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest]].text(.05, .95,
                                                        txtstr, ha='left', va='top',
                                                        transform=self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest]].transAxes)

                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest]].yaxis.set_label_position("right")


                    else:

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest] ].scatter(obs, pred, edgecolors=(0, 0, 0),  
                               color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                               s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                               marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest] ].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].text(.05, .95,
                                                        txtstr, ha='left', va='top',
                                                        transform=self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].transAxes)

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].yaxis.set_label_position("right")

                    # if at last column
                    if self.targetFeaturePlotColumnD[modeltest] == len(self.modelPlot.rows.regressionModels.columns)-1:

                        if len(self.targetFeatures) == 1:

                            self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest]].set_ylabel('Predictions')

                        else:

                            self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].set_ylabel('Predictions')

                    # if at last row
                    if self.targetN == self.nTargetFeatures-1:

                        if len(self.targetFeatures) == 1:
                            
                            self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest]].set_xlabel('Observations')

                        else:

                            self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].set_xlabel('Observations')

            if self.modelPlot.rows.regressionModels.apply:

                # modeltest is either trainTest of Kfold
                if modeltest in self.modelPlot.rows.regressionModels.columns:

                    if (len(self.regressorModels)) == 1:

                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest] ].scatter(obs, pred, edgecolors=(0, 0, 0),  
                               color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                               s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                               marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)


                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].text(.05, .95, txtstr, ha='left', va='top',
                                                        transform=self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].transAxes)

                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].yaxis.set_label_position("right")

                    else:

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest] ].scatter(obs, pred, edgecolors=(0, 0, 0),  
                                color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                               s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                               marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)


                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].text(.05, .95, txtstr, ha='left', va='top',
                                                        transform=self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].transAxes)

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].yaxis.set_label_position("right")

                    # if at last column
                    if self.regressionModelPlotColumnD[modeltest] == len(self.modelPlot.rows.targetFeatures.columns)-1:

                        if self.regrN == self.nRegrModels-1:

                            if (len(self.regressorModels)) == 1:

                                self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].set_ylabel('Predictions')

                            else:

                                self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].set_ylabel('Predictions')

                        else:

                            if (len(self.regressorModels)) == 1:

                                self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].set_ylabel('Predictions')

                            else:

                                self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].set_ylabel('Predictions')

                    # if at last row
                    if self.regrN == self.nRegrModels-1:

                        if (len(self.regressorModels)) == 1:

                            self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].set_xlabel('Observations')

                        else:

                            self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].set_xlabel('Observations')
                    '''
                    else:

                        if (len(self.regressorModels)) == 1:

                            self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].set_xlabel('Observations')

                        else:

                            self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].set_xlabel('Observations')
                    '''
                  
        if self.multcompplot:
 
            index = modeltest
            columnNr = self.modelNr
            

            if (len(self.regressorModels)) == 1: # only 1 row in subplot

                multCompAxs[self.targetFeature][index][columnNr].scatter(obs, pred, edgecolors=(0, 0, 0),  
                        color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                       s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                       marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                multCompAxs[self.targetFeature][index][columnNr].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)


                multCompAxs[self.targetFeature][index][columnNr].text(.05, .95, txtstr, ha='left', va='top',
                                                transform=multCompAxs[self.targetFeature][index][columnNr].transAxes)

                multCompAxs[self.targetFeature][index][columnNr].yaxis.set_label_position("right")

            else:

                multCompAxs[self.targetFeature][index][self.regrN, columnNr ].scatter(obs, pred, edgecolors=(0, 0, 0),  
                        color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                       s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                       marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].text(.05, .95, txtstr, ha='left', va='top',
                                                transform=multCompAxs[self.targetFeature][index][self.regrN, columnNr].transAxes)

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].yaxis.set_label_position("right")

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

                    multCompAxs[self.targetFeature][index][columnNr].set_ylabel('Predictions')

                else:

                    multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylabel('Predictions')

            # if at last row
            if self.regrN == self.nRegrModels-1:

                if (len(self.regressorModels)) == 1:

                    multCompAxs[self.targetFeature][index][columnNr].set_xlabel('Observations')

                else:

                    multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_xlabel('Observations')
                    
    def _PlotFeatureImportanceSingles(self, featureArray, importanceArray, errorArray, title, xyLabel, pngFPN):
        '''
        '''
        # Convert to a pandas series
        importanceDF = PdSeries(importanceArray, index=featureArray)

        singlefig, ax = plt.subplots()
        
        if isinstance(errorArray, np.ndarray):

            importanceDF.plot.bar(yerr=errorArray, color=self.featureSymbolColor, ax=ax)

        else:
            importanceDF.plot.bar(yerr=errorArray, color=self.featureSymbolColor, ax=ax)
            
        if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
            
            argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                                
            ax.tick_params(**argD)
                    
        ax.set_title(title)
        
        ax.set_ylim(ymin=0)

        if xyLabel[0]:

            ax.set_ylabel(xyLabel[0])

        if xyLabel[1]:

            ax.set_ylabel(xyLabel[1])

        if self.modelPlot.tightLayout:

            singlefig.tight_layout()

        if self.modelPlot.singles.screenShow:

            plt.show()

        if self.modelPlot.singles.savePng:

            singlefig.savefig(pngFPN)

        plt.close(fig=singlefig)

    def _PlotFeatureImportanceRows(self, featureArray, importanceArray, errorArray, importanceCategory, yLabel):
        '''
        '''

        nnFS = self.X_train.shape

        text = 'tot covars: %s' %(nnFS[1])

        if self.specificFeatureSelectionTxt != None:

            text += '\n%s' %(self.specificFeatureSelectionTxt)

        if self.generalFeatureSelectTxt != None:

            text += '\n%s' %(self.generalFeatureSelectTxt)

        if self.modelPlot.rows.targetFeatures.apply:

            if importanceCategory in self.modelPlot.rows.targetFeatures.columns:

                if (len(self.targetFeatures)) == 1:

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                    transform=self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].set_ylabel(yLabel)

                    if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
            
                        argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                                
                        #self.columnAxs[self.regrModel[0]].tick_params(**argD)
                        
                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].tick_params(**argD)

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].set_ylim(ymin=0) 
                    #self.columnAxs[self.regrModel[0]].set_ylim(ymin=0)

                else:
                    

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                    transform=self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].set_ylabel(yLabel)
                                           
                    if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
            
                        argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                                
                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].tick_params(**argD)

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].set_ylim(ymin=0)
                    
                if importanceCategory == 'featureImportance':

                    if (len(self.targetFeatures)) == 1:

                        # Draw horisontal line ay y=y
                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')

                    else:

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')
              
                # if at last row
                if self.targetN == self.nTargetFeatures-1:

                    if (len(self.targetFeatures)) == 1:

                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].set_xlabel('Features')

                    else:

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].set_xlabel('Features')

        if self.modelPlot.rows.regressionModels.apply:

            if importanceCategory in self.modelPlot.rows.regressionModels.columns:

                if (len(self.regressorModels)) == 1:

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                transform=self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].set_ylabel(yLabel)

                    if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
            
                        argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                                
                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].tick_params(**argD)

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].set_ylim(ymin=0)
                    
                else:

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                    transform=self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].set_ylabel(yLabel)

                    if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
            
                        argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                                
                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].tick_params(**argD)
                    
                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].set_ylim(ymin=0)
                            
                if importanceCategory == 'featureImportance':

                    if (len(self.regressorModels)) == 1:

                        # Draw horisontal line ay y=y
                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')

                    else:

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')
   
                # if at last row set the x-axis label
                if self.regrN == self.nRegrModels-1:

                    if (len(self.regressorModels)) == 1:

                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].set_xlabel('Features')

                    else:

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].set_xlabel('Features')

    