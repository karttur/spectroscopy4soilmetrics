'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

import json

from support import PathJoin, PathExists

def DumpAnyJson(dumpD, jsonFPN, intent=2):
    ''' dump, any json object

    :param exportD: formatted dictionary
    :type exportD: dict
    '''

    jsonF = open(jsonFPN, "w")

    json.dump(dumpD, jsonF, indent = intent)

    jsonF.close()

def ReadAnyJson(jsonFPN):
    """ Read json parameter file

    :param jsonFPN: path to json file
    :type jsonFPN: str

    :return paramD: parameters
    :rtype: dict
   """

    with open(jsonFPN) as jsonF:

        jsonD = json.load(jsonF)

    return (jsonD)

def ReadProjectFile(dstRootFP,projFN):

    projFPN = PathJoin([dstRootFP,projFN])

    if not PathExists(projFPN):

        exitstr = 'EXITING, project file missing: %s.' %(projFPN)

        exit( exitstr )

    infostr = 'Processing %s' %(projFPN)

    print (infostr)

    '''
    # Open and read the text file linking to all json files defining the project
    with open(projFPN) as f:

        jsonL = f.readlines()
    
    # Clean the list of json objects from comments and whithespace etc
    jsonProcessObjectL = [os.path.join(jsonFP,x.strip())  for x in jsonL if len(x) > 10 and x[0] != '#']
    '''
    
    jsonProcessObjectD = ReadAnyJson(projFPN)

    return jsonProcessObjectD

def ReadModelJson(jsonFPN):
    """ Read the parameters for modeling

    :param jsonFPN: path to json file
    :type jsonFPN: str

    :return paramD: parameters
    :rtype: dict
    """
    
    if not PathExists(jsonFPN):
       
        print (jsonFPN)

    with open(jsonFPN) as jsonF:

        paramD = json.load(jsonF)

    return (paramD)
