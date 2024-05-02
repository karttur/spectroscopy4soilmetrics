'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

from os import path, makedirs

def PathJoin(partsL):
    '''
    '''
    joinedPath = partsL[0]
    
    for i in range(1, len(partsL)):
        
        joinedPath = path.join(joinedPath, partsL[i])
        
    return joinedPath

def PathExists(p):
    '''
    '''
    
    return path.exists(p)

def PathSplit(p):
    '''
    '''
    
    return path.split(p)

def MakeDirs(p):
    '''
    '''
    makedirs(p)
    

def CheckMakeDocPaths(rootpath,arrangeddatafolder, jsonpath, sourcedatafolder=False):
    """ Create the default json parameters file structure, only to create template if lacking

        :param str dstrootFP: directory path

        :param str jsonpath: subfolder under directory path
    """

    if not path.exists(rootpath):

        exitstr = "The rootpath does not exists: %s" %(rootpath)

        exit(exitstr)

    if sourcedatafolder:

        srcFP = path.join(path.dirname(__file__),rootpath,sourcedatafolder)

        if not path.exists(srcFP):

            exitstr = "The source data path to the original OSSL data does not exists:\n %s" %(srcFP)

            exit(exitstr)

    dstRootFP = path.join(path.dirname(__file__),rootpath,arrangeddatafolder)

    if not path.exists(dstRootFP):

        makedirs(dstRootFP)

    jsonFP = path.join(dstRootFP,jsonpath)

    if not path.exists(jsonFP):

        makedirs(jsonFP)
        
    return dstRootFP, jsonFP
        
        