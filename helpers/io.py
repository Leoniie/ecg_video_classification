import pandas as pd
import numpy as np


def inputter_csv(file):
    array = pd.read_csv( file )
    array = array[ 'y' ].values
    array = np.reshape( array, (array.shape[ 0 ], 1) )

    print( "Input of {} done! \n Array has shape of {}.".format( file, array.shape ) )

    return array

def inputter_avi(file):
    # pass is a placeholder for the real function
    pass


def Outputter():
    # pass is a placeholder for the real function
    pass

    # TODO: define the arguments

    # TODO: name of the output file is 'daytime_solution.csv'

    # TODO: generate output file in folder './output'

    # no return required

