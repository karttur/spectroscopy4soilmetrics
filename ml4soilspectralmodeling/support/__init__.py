"""
support
==========================================

Package belonging to xSpectre´s Spectral Machine Learning Framework.

Author
------
Thomas Gumbricht (thomas.gumbricht@karttur.com)

"""

from .version import __version__, VERSION, metadataD

from .osSupport import PathJoin, PathExists, PathSplit, MakeDirs, CheckMakeDocPaths

from .jsonSupport import DumpAnyJson, ReadAnyJson, ReadProjectFile, ReadModelJson

from .csvSupport import ReadCSV

from .copySupport import DeepCopy

from .pdSupport import PdSeries, PdDataFrame, PdConcat