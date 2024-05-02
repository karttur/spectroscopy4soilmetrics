"""
plotChartt
==========================================

Package belonging to xSpectre´s Spectral Machine Learning Framework.

Author
------
Thomas Gumbricht (thomas.gumbricht@karttur.com)

"""

from .version import __version__, VERSION, metadataD

from .Layout import SetFigSize, GetPlotStyle, GetAxisLabels

from .PlotPreProcesses import PlotScatterCorr, PlotStandardisation, PlotDerivatives, PlotFilterExtract, \
    PlotCoviariateSelection, PlotPCA, PlotOutlierDetect
    
from .PlotMultiProject import MultiPlot, ClosePlot

from .PlotRegression import PlotRegrClass

#from .jsonSupport import DumpAnyJson, ReadAnyJson

#from .csvSupport import ReadCSV