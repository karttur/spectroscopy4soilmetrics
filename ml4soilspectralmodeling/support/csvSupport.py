'''
Created on 29 Apr 2024

@author: thomasgumbricht
'''

import csv

def ReadCSV(FPN):
    ''' Standard reader for all OSSL csv data files

    :param FPN: path to csv file
    :type FPN: str

    :return headers: list of columns
    :rtype: list

    :return rowL: array of data
    :rtype: list of list
    '''

    rowL = []

    with open( FPN, 'r' ) as csvF:

        reader = csv.reader(csvF)

        headers = next(reader, None)

        for row in reader:

            rowL.append(row)

    return headers, rowL
