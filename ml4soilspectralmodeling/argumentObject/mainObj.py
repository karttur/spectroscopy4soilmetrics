'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

class Obj(object):
    ''' Convert json parameters to class objects
    '''

    def __init__(self, paramD):
        ''' Convert input parameters from nested dict to nested class object

            :param dict paramD: parameters
        '''
        for k, v in paramD.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [Obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Obj(v) if isinstance(v, dict) else v)

    def _SetArrangeDefautls(self):
        ''' Set class object default data if missing
        '''

        if not hasattr(self, 'sitedata'):

            setattr(self, 'sitedata', [])

        sitedataMinL = ["id.layer_local_c",
                        "dataset.code_ascii_txt",
                        "longitude.point_wgs84_dd",
                        "latitude.point_wgs84_dd",
                        "location.point.error_any_m",
                        "layer.upper.depth_usda_cm",
                        "layer.lower.depth_usda_cm",
                        "id_vis","id_mir"]

        for item in sitedataMinL:

            if not item in self.sitedata:

                self.sitedata.append(item)
        '''
        self.visnirStep = int(self.input.visnirOutputBandWidth/ self.input.visnirInputBandWidth)

        self.mirStep = int(self.input.mirOutputBandWidth/ self.input.mirInputBandWidth)

        self.neonStep = int(self.input.neonOutputBandWidth/ self.input.neonInputBandWidth)
        '''

    def _SetPlotDefaults(self):
        ''' Set class object default data if required
        '''

        if self.modelPlot.singles.figSize.x == 0:

            self.modelPlot.singles.figSize.x = 8

        if self.modelPlot.singles.figSize.y == 0:

            self.modelPlot.singles.figSize.y = 6

    def _SetPlotTextPos(self, plot, xmin, xmax, ymin, ymax):
        ''' Set position of text objects for matplotlib

            :param float xmin: x-axis minimum

            :param float xmax: x-axis maximum

            :param float ymin: y-axis minimum

            :param float ymax: y-axis maximum

            :returns: text x position
            :rtype: float

            :returns: text y position
            :rtype: float
        '''

        x = plot.text.x*(xmax-xmin)+xmin

        y = plot.text.y*(ymax-ymin)+ymin

        return (x,y)

    def _SetSoilLineDefautls(self):
        ''' Set class object default data if required
        '''

        if self.modelPlot.singles.figSize.x == 0:

            self.modelPlot.singles.figSize.x = 8

        if self.modelPlot.singles.figSize.y == 0:

            self.modelPlot.singles.figSize.y = 6

    def _SetModelDefaults(self):
        ''' Set class object default data if required
        '''

        if self.modelPlot.singles.figSize.x == 0:

            self.modelPlot.singles.figSize.x = 4

        if self.modelPlot.singles.figSize.y == 0:

            self.modelPlot.singles.figSize.y = 4

        # Check if Manual feature selection is set
        if self.manualFeatureSelection.apply:

            # Turn off the derivates alteratnive (done as part of the manual selection if requested)
            self.spectraInfoEnhancement.derivatives.apply = False

            # Turn off all other feature selection/agglomeration options
            self.generalFeatureSelection.apply = False

            self.specificFeatureSelection.apply = False

            self.specificFeatureAgglomeration.apply = False

    def __iter__(self):
        '''
        '''
        
        for attr, value in self.__dict__.iteritems():
            yield attr, value