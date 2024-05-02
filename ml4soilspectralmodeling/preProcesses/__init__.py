"""
PreProcess
==========================================

Package belonging to xSpectre´s Spectral Machine Learning Framework.

Author
------
Thomas Gumbricht (thomas.gumbricht@karttur.com)

"""

from .version import __version__, VERSION, metadataD

from preProcesses.StandAlonePreProcess import ScatterCorrection, Standardisation, Derivatives, snv, msc

from preProcesses.ClassPreProcess import MLPreProcess

#from .jsonSupport import DumpAnyJson, ReadAnyJson

#from .csvSupport import ReadCSV