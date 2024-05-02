'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

from support import ReadAnyJson, DeepCopy

from project import SetupProcesses


if __name__ == '__main__':
    ''' If script is run as stand alone
    '''
    #OutliersTest()

    '''
    if len(sys.argv) != 2:

        sys.exit('Give the link to the json file to run the process as the only argument')

    #Get the json file
    rootJsonFPN = sys.argv[1]

    if not os.path.exists(jsonFPN):

        exitstr = 'json file not found: %s' %(rootJsonFPN)
    '''
    

    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl_v4.json'
    
    iniParams = ReadAnyJson(rootJsonFPN)
            
if type( iniParams['projFN']) is list: 
          
        for proj in iniParams['projFN']:
            
            projParams = DeepCopy(iniParams)
            
            projParams['projFN'] = proj
            
            SetupProcesses(projParams)
           
else:
        
    SetupProcesses(iniParams)